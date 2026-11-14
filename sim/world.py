"""
PyBullet tabletop for grasping with GraspPointSAM:

  - green table (as in GraspNet-1Billion), top at z = 0
  - Franka Emika Panda arm + Franka Hand (pybullet_data/franka_panda) at the origin
  - GraspNet-1Billion object models dropped at random poses in front of the arm
  - fixed RGB-D camera with the RealSense intrinsics of GraspNet-1Billion
    (the camera the model was trained on), looking at the workspace

Grasp execution: IK + collision check for the pre-grasp (10 cm back along the
approach) and grasp poses, joint-space move to the pre-grasp, straight-line
approach, close, straight-line lift, carry to the drop point, release.
"""

import time

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from .assets import prepare_object

DT = 1.0 / 240.0

# camera: GraspNet-1Billion RealSense D435 intrinsics 
CAM_W, CAM_H = 1280, 720
CAM_K = np.array([[927.17, 0.0, 651.32],
                  [0.0, 927.37, 349.62],
                  [0.0, 0.0, 1.0]])
CAM_EYE = (1.00, 0.0, 0.60)        # opposite the arm, looking down at the workspace
CAM_TARGET = (0.50, 0.0, 0.00)

#  workspace 
WORKSPACE = ((0.35, 0.65), (-0.20, 0.20))   # x, y range where objects are dropped (m)
DROP_POS  = (0.30, 0.55, 0.30)              # where grasped objects are released

#  Franka Panda (pybullet_data/franka_panda/panda.urdf) 
ARM_JOINTS = list(range(7))                  # panda_joint1..7
FINGERS = (9, 10)                         # panda_finger_joint1/2, q = half opening
EE_LINK    = 11                              # panda_grasptarget, 0.105 m from the hand
ARM_FORCES = [87.0] * 4 + [12.0] * 3         # URDF effort limits (Nm)
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # Franka "ready" pose
OBSERVE = [1.4, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # ready pose turned aside (out of view)
MAX_WIDTH  = 0.08                            # Franka Hand max opening (m)
# fingertips are 0.0584 (finger joint) + 0.0538 (finger) = 0.1122 m from the hand
FINGERTIP_FROM_EE = 0.1122 - 0.105
# graspnetAPI grasp frame (x approach, y closing, z height) -> Franka Hand frame
# (z approach, y finger motion): columns = hand x, y, z in the grasp frame
GRASP_TO_HAND = np.array([[0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0]])
TOP_DOWN = Rotation.from_euler('x', np.pi).as_quat()   # hand pointing down
IK_POS_TOL = 0.005                            # m
IK_ANG_TOL = np.deg2rad(3.0)
APPROACH = 0.10                             # pre-grasp distance (m)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 camera-to-world pose, OpenCV convention (x right, y down, z forward)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def gl_projection(K: np.ndarray, W: int, H: int, near: float, far: float) -> list:
    """OpenCV intrinsics -> OpenGL projection matrix (column-major list)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.array([
        [2 * fx / W, 0,          1 - 2 * cx / W,               0],
        [0,          2 * fy / H, 2 * cy / H - 1,               0],
        [0,          0,          -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0,          0,          -1,                           0],
    ])
    return P.T.flatten().tolist()


class GraspPlan:
    """Joint configurations and poses of one feasible grasp."""

    def __init__(self, obj_id, grasp, q_pre, q_grasp, pre, target, orn, half):
        self.obj_id, self.grasp = obj_id, grasp
        self.q_pre, self.q_grasp = q_pre, q_grasp
        self.pre, self.target, self.orn, self.half = pre, target, orn, half


class TabletopWorld:

    def __init__(self, gui: bool = True, friction: float = 0.5):
        self.gui = gui
        self.friction = friction
        p.connect(p.GUI if gui else p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.4, 60, -35, [0.4, 0.0, 0.1])

        half = [1.0, 1.0, 0.025]
        self.table = p.createMultiBody(
            0,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
            p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                rgbaColor=[0.05, 0.36, 0.32, 1]),   # GraspNet's green table
            basePosition=[0.4, 0.0, -0.025],
        )
        p.changeDynamics(self.table, -1, lateralFriction=friction)

        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True,
                                flags=p.URDF_USE_INERTIA_FROM_FILE)
        for link in (-1, 0):                      # base stands on the table
            p.setCollisionFilterPair(self.robot, self.table, link, -1, 0)
        for f in FINGERS:
            p.changeDynamics(self.robot, f, lateralFriction=1.0)
        movable = [j for j in range(p.getNumJoints(self.robot))
                   if p.getJointInfo(self.robot, j)[2] != p.JOINT_FIXED]   # 7 arm + 2 fingers
        self.lower  = [p.getJointInfo(self.robot, j)[8] for j in movable]
        self.upper  = [p.getJointInfo(self.robot, j)[9] for j in movable]
        self.ranges = [u - l for l, u in zip(self.lower, self.upper)]
        self.rest_q = HOME + [MAX_WIDTH / 2] * 2
        self._set_arm(OBSERVE, MAX_WIDTH / 2)

        self.T_world_cam = look_at(CAM_EYE, CAM_TARGET)
        if gui:
            p.addUserDebugLine(CAM_EYE, CAM_TARGET, [1, 0, 0], lineWidth=2)

        self.objects = {}   # obj_id -> body id

    def step(self, n: int) -> None:
        for _ in range(n):
            p.stepSimulation()
            if self.gui:
                time.sleep(DT)

    #  objects 

    def spawn_objects(self, obj_ids, graspnet_root, cache_dir, rng,
                      min_dist: float = 0.10) -> None:
        """Drop the objects one by one at random positions / yaw in the workspace."""
        placed = []
        for obj_id in obj_ids:
            m = prepare_object(obj_id, graspnet_root, cache_dir)
            for _ in range(100):
                xy = [rng.uniform(*WORKSPACE[0]), rng.uniform(*WORKSPACE[1])]
                if all(np.hypot(xy[0] - a, xy[1] - b) > min_dist for a, b in placed):
                    break
            placed.append(xy)
            body = p.createMultiBody(
                baseMass=m['mass'],
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_MESH, fileName=m['collision']),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName=m['visual']),
                basePosition=[*xy, 0.15],
                baseOrientation=Rotation.from_euler('z', rng.uniform(0, 2 * np.pi)).as_quat(),
            )
            p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(m['texture']),
                                rgbaColor=[1, 1, 1, 1])
            p.changeDynamics(body, -1, lateralFriction=self.friction,
                             rollingFriction=1e-3, spinningFriction=1e-3)
            self.objects[obj_id] = body
            self.step(120)
        self.step(240)
        self.remove_fallen()

    def remove_object(self, obj_id: int) -> None:
        p.removeBody(self.objects.pop(obj_id))

    def remove_fallen(self) -> list:
        """Remove objects that left the table area; returns their ids."""
        gone = []
        for obj_id, body in list(self.objects.items()):
            x, y, z = p.getBasePositionAndOrientation(body)[0]
            if z < -0.05 or not (-0.2 < x < 1.0 and -0.6 < y < 0.6):
                self.remove_object(obj_id)
                gone.append(obj_id)
        return gone

    def height(self, obj_id: int) -> float:
        return p.getBasePositionAndOrientation(self.objects[obj_id])[0][2]

    #  camera 

    def capture(self, near: float = 0.05, far: float = 3.0):
        """
        Returns rgb [H, W, 3] uint8, depth [H, W] float m (0 = no return),
        seg [H, W] int (obj_id + 1, 0 = background/table/robot), as in GraspNet's labels.
        """
        view = np.linalg.inv(self.T_world_cam @ np.diag([1, -1, -1, 1]))   # OpenCV -> OpenGL
        light = (self.T_world_cam[:3, 3] + [0.0, 0.0, 0.5]).tolist()
        _, _, rgba, dbuf, seg = p.getCameraImage(
            CAM_W, CAM_H, view.T.flatten().tolist(), gl_projection(CAM_K, CAM_W, CAM_H, near, far),
            renderer=p.ER_TINY_RENDERER, lightDirection=light, shadow=0,
            lightAmbientCoeff=0.6, lightDiffuseCoeff=0.5, lightSpecularCoeff=0.05,
        )
        rgb = np.reshape(rgba, (CAM_H, CAM_W, 4))[..., :3].astype(np.uint8)
        dbuf = np.reshape(dbuf, (CAM_H, CAM_W))
        z = far * near / (far - (far - near) * dbuf)
        z[dbuf >= 1.0] = 0.0

        seg = np.reshape(seg, (CAM_H, CAM_W))
        lab = np.zeros((CAM_H, CAM_W), dtype=np.int32)
        for obj_id, body in self.objects.items():
            lab[seg == body] = obj_id + 1
        return rgb, z, lab

    #  arm 

    def _joint_state(self) -> list:
        return [p.getJointState(self.robot, j)[0] for j in ARM_JOINTS + list(FINGERS)]

    def _restore_joints(self, q) -> None:
        for j, v in zip(ARM_JOINTS + list(FINGERS), q):
            p.resetJointState(self.robot, j, v)

    def _fingers(self, half: float, force: float = 20.0) -> None:
        for f in FINGERS:
            p.setJointMotorControl2(self.robot, f, p.POSITION_CONTROL,
                                    targetPosition=half, force=force)

    def _arm_targets(self, q) -> None:
        for j, v, f in zip(ARM_JOINTS, q, ARM_FORCES):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=v,
                                    force=f, maxVelocity=1.0)

    def _set_arm(self, q, half: float) -> None:
        """Teleport the arm to joint angles q and hold them; hand opened to 2 * half."""
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(self.robot, j, v)
        for f in FINGERS:
            p.resetJointState(self.robot, f, half)
        self._arm_targets(q)
        self._fingers(half)

    def _ik_solve(self, pos, orn) -> list:
        sol = p.calculateInverseKinematics(
            self.robot, EE_LINK, list(pos), list(orn),
            lowerLimits=self.lower, upperLimits=self.upper, jointRanges=self.ranges,
            restPoses=self.rest_q, maxNumIterations=200, residualThreshold=1e-5,
        )
        return list(np.clip(sol[:7], self.lower[:7], self.upper[:7]))

    def _ik(self, pos, orn):
        """IK by setting the joints directly (planning only, no physics step). -> (q, reachable)."""
        for _ in range(10):
            q = self._ik_solve(pos, orn)
            for j, v in zip(ARM_JOINTS, q):
                p.resetJointState(self.robot, j, v)
            ls = p.getLinkState(self.robot, EE_LINK, computeForwardKinematics=True)
            pos_err = np.linalg.norm(np.array(ls[4]) - pos)
            ang_err = (Rotation.from_quat(ls[5]).inv() * Rotation.from_quat(orn)).magnitude()
            if pos_err < 1e-3 and ang_err < 0.01:
                break
        return q, (pos_err < IK_POS_TOL and ang_err < IK_ANG_TOL)

    def _in_collision(self, tol: float) -> bool:
        p.performCollisionDetection()
        return any(c[8] < -tol for c in p.getContactPoints(bodyA=self.robot))

    def move_joints(self, q_goal, seconds: float = 1.5) -> None:
        """Joint-space interpolation from the current configuration."""
        q0 = np.array([p.getJointState(self.robot, j)[0] for j in ARM_JOINTS])
        for t in np.linspace(0, 1, int(seconds / DT)):
            self._arm_targets(q0 + t * (np.asarray(q_goal) - q0))
            self.step(1)
        self.step(30)

    def move_line(self, start, goal, orn, waypoints: int = 10, steps_each: int = 15) -> None:
        """Straight-line end-effector motion, IK from the current joint state."""
        for t in np.linspace(0, 1, waypoints + 1)[1:]:
            self._arm_targets(self._ik_solve(start + t * (goal - start), orn))
            self.step(steps_each)

    #  grasping 

    def plan_grasp(self, grasp: np.ndarray, collision_tol: float = 0.005):
        """
        One graspnetAPI grasp (camera frame, [17] array) -> GraspPlan, or the
        reason it is infeasible: 'too_wide' (> 8 cm), 'unreachable' (no IK),
        'collision' (arm/hand overlaps the scene at the grasp or pre-grasp pose).
        The arm is back where it was afterwards.
        """
        width, depth = float(grasp[1]), float(grasp[3])
        if width > MAX_WIDTH:
            return 'too_wide'
        half = min(width / 2 + 0.002, MAX_WIDTH / 2)

        T = self.T_world_cam
        R = T[:3, :3] @ grasp[4:13].reshape(3, 3)
        t = T[:3, :3] @ grasp[13:16] + T[:3, 3]
        approach = R[:, 0]
        target = t + (depth - FINGERTIP_FROM_EE) * approach    # fingertips at t + depth * approach
        pre    = target - APPROACH * approach
        orn    = Rotation.from_matrix(R @ GRASP_TO_HAND).as_quat()

        saved = self._joint_state()
        try:
            for f in FINGERS:
                p.resetJointState(self.robot, f, half)
            q_grasp, ok = self._ik(target, orn)
            if not ok:
                return 'unreachable'
            if self._in_collision(collision_tol):
                return 'collision'
            q_pre, ok = self._ik(pre, orn)
            if not ok:
                return 'unreachable'
            if self._in_collision(collision_tol):
                return 'collision'
        finally:
            self._restore_joints(saved)
        return GraspPlan(int(grasp[16]), grasp, q_pre, q_grasp, pre, target, orn, half)

    def draw_grasp(self, plan: GraspPlan) -> None:
        """GUI: approach line (blue) and finger line (orange) of the planned grasp."""
        if not self.gui:
            return
        p.removeAllUserDebugItems()
        p.addUserDebugLine(CAM_EYE, CAM_TARGET, [1, 0, 0], lineWidth=2)
        R = Rotation.from_quat(plan.orn).as_matrix()          # hand frame
        tip = plan.target + FINGERTIP_FROM_EE * R[:, 2]
        p.addUserDebugLine(plan.pre, tip, [0.1, 0.3, 1.0], lineWidth=3)
        p.addUserDebugLine(tip - plan.half * R[:, 1], tip + plan.half * R[:, 1],
                           [1.0, 0.5, 0.0], lineWidth=3)

    def execute(self, plan: GraspPlan, close_force: float = 20.0, lift: float = 0.15) -> bool:
        """
        Run a planned grasp. On success the object is carried to DROP_POS,
        released and removed from the scene. The arm ends in the OBSERVE pose.
        Returns True if the object was lifted.
        """
        z_before = self.height(plan.obj_id)
        self._fingers(plan.half)
        self.move_joints(plan.q_pre)
        self.move_line(plan.pre, plan.target, plan.orn)
        self.step(20)
        self._fingers(0.0, force=close_force)
        self.step(180)
        self.move_line(plan.target, plan.target + [0, 0, lift], plan.orn,
                       waypoints=15, steps_each=16)
        self.step(60)
        lifted = self.height(plan.obj_id) - z_before > 0.5 * lift

        if lifted:
            saved = self._joint_state()
            q_drop, _ = self._ik(np.array(DROP_POS), TOP_DOWN)
            self._restore_joints(saved)
            self.move_joints(q_drop)
            self._fingers(MAX_WIDTH / 2)
            self.step(120)
            if plan.obj_id in self.objects:
                self.remove_object(plan.obj_id)
        else:
            self._fingers(MAX_WIDTH / 2)
            self.step(60)
        self.move_joints(OBSERVE)
        self.remove_fallen()
        return lifted

    def close(self) -> None:
        p.disconnect()

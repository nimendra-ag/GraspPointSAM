"""
Step 2: Panda on a table, objects on the table, camera looking at the objects.

Run:  python step2_table_objects.py
"""
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

np.random.seed(0)  # same scene every run; change/remove for random scenes

# ---------------- World ----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# ---------------- Table (built from boxes, so we know its exact size) ----------------
TABLE_H = 0.60                    # height of the table top surface (m)
TABLE_SIZE = [1.2, 1.0]           # length (x), width (y) in metres
TABLE_CENTER = [0.35, 0.0]        # robot at x=0 sits near the back edge
TOP_THICK = 0.04


def make_box(half_extents, pos, color):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(baseMass=0,                 # mass 0 = static, never moves
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)


# table top
table = make_box([TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TOP_THICK / 2],
                 [TABLE_CENTER[0], TABLE_CENTER[1], TABLE_H - TOP_THICK / 2],
                 [0.6, 0.45, 0.3, 1])
# four legs
leg_h = TABLE_H - TOP_THICK
for dx in (-1, 1):
    for dy in (-1, 1):
        make_box([0.025, 0.025, leg_h / 2],
                 [TABLE_CENTER[0] + dx * (TABLE_SIZE[0] / 2 - 0.05),
                  TABLE_CENTER[1] + dy * (TABLE_SIZE[1] / 2 - 0.05),
                  leg_h / 2],
                 [0.4, 0.3, 0.2, 1])

# ---------------- Robot on the table ----------------
robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, TABLE_H], useFixedBase=True)
HOME = [0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.785]
for j, q in enumerate(HOME):
    p.resetJointState(robot, j, q)
for j in (9, 10):                                   # open the fingers
    p.resetJointState(robot, j, 0.04)

# ---------------- Objects on the table ----------------
OBJ_CENTER = [0.5, 0.0]           # where objects are scattered (in front of the robot)
OBJ_SPREAD = 0.12                 # +/- metres around OBJ_CENTER
N_OBJECTS = 6

candidates = ["objects/mug.urdf", "duck_vhacd.urdf", "cube_small.urdf",
              "sphere_small.urdf", "lego/lego.urdf"]
candidates += [f"random_urdfs/{i:03d}/{i:03d}.urdf" for i in range(0, 50)]  # varied shapes

objects = []
for i in range(N_OBJECTS):
    urdf = candidates[np.random.randint(len(candidates))]
    pos = [OBJ_CENTER[0] + np.random.uniform(-OBJ_SPREAD, OBJ_SPREAD),
           OBJ_CENTER[1] + np.random.uniform(-OBJ_SPREAD, OBJ_SPREAD),
           TABLE_H + 0.10 + 0.06 * i]               # stacked heights so they don't spawn inside each other
    orn = p.getQuaternionFromEuler(np.random.uniform(-np.pi, np.pi, 3))
    obj = p.loadURDF(urdf, pos, orn)
    p.changeDynamics(obj, -1, lateralFriction=1.0)
    objects.append(obj)
    print(f"object id {obj}: {urdf}")

# let the objects fall and settle on the table
for _ in range(1000):
    p.stepSimulation()

# ---------------- Camera looking at the objects ----------------
W, H = 640, 480
FOV, NEAR, FAR = 60, 0.01, 2.0
cam_target = [OBJ_CENTER[0], OBJ_CENTER[1], TABLE_H]          # centre of the objects
cam_eye = [OBJ_CENTER[0] + 0.15, OBJ_CENTER[1], TABLE_H + 0.70]  # above and slightly in front
# ~0.7 m from the objects, similar to GraspNet's camera distances.
# Tilted toward the robot just enough that the arm stays out of view.

view = p.computeViewMatrix(cam_eye, cam_target, cameraUpVector=[0, 0, 1])
proj = p.computeProjectionMatrixFOV(FOV, W / H, NEAR, FAR)
p.addUserDebugLine(cam_eye, cam_target, [1, 0, 0], lineWidth=2)


def capture():
    """RGB (H,W,3), depth in metres (H,W), segmentation (H,W) = body id per pixel."""
    _, _, rgba, zbuf, seg = p.getCameraImage(W, H, view, proj,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(rgba, (H, W, 4))[:, :, :3].astype(np.uint8)
    depth = FAR * NEAR / (FAR - (FAR - NEAR) * np.reshape(zbuf, (H, W)))
    seg = np.reshape(seg, (H, W))
    return rgb, depth, seg


rgb, depth, seg = capture()
print("depth range: %.3f - %.3f m" % (depth.min(), depth.max()))
print("bodies visible in image:", sorted(set(np.unique(seg)) - {-1}))
if robot in np.unique(seg):
    print("WARNING: the robot is in the camera view")

# ---------------- Show the images ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(rgb); ax[0].set_title("RGB")
im = ax[1].imshow(depth, cmap="viridis"); ax[1].set_title("Depth (m)")
fig.colorbar(im, ax=ax[1])
ax[2].imshow(seg, cmap="tab20"); ax[2].set_title("Segmentation (body id)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show(block=False)

# GUI view: look at the whole setup from an angle
p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=50, cameraPitch=-35,
                             cameraTargetPosition=[0.35, 0, TABLE_H])

while p.isConnected():
    p.stepSimulation()
    time.sleep(1 / 240)
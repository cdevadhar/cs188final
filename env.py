import cv2
import numpy as np
import threading
import queue
import robosuite as suite
from dex_retargeting.constants import HandType
from example.vector_retargeting.single_hand_detector import SingleHandDetector

class PDController:
    def __init__(self, kp, kd, target=None):
        """
        Initialize a proportional controller.

        Args:
            kp (float): Proportional gain.
            target (tuple or array): Target position.
        """
        self.kp = kp
        self.kd = kd
        self.lasterr = np.array([0,0,0])
        self.target = target

    def reset(self, target=None):
        """
        Reset the target position.

        Args:
            target (tuple or array, optional): New target position.
        """
        self.lasterr = np.array([0,0,0])
        self.target = target

    
    def update(self, current_pos):
        """
        Compute the control signal.

        Args:
            current_pos (array-like): Current position.

        Returns:
            np.ndarray: Control output vector.
        """
        cur_err = self.target - current_pos
        control = self.kp*cur_err + self.kd*(cur_err - self.lasterr)
        self.lasterr = cur_err
        return control

# ----------------------------
# Shared state
# ----------------------------
pd = PDController(kp=5.0, kd=0.5, target=np.zeros(3))
data_lock = threading.Lock()
stop_event = threading.Event()

hand_start = None
robot_start = None
latest_hand_pos = None
hand_open = 0
min_dist = 0.02
max_dist = 0.15

# ----------------------------
# Background thread: webcam + hand detection only (doesn't handle actions)
# ----------------------------
def webcam_thread_fn():
    global latest_hand_pos, hand_open

    hand_type = HandType.right
    detector = SingleHandDetector(
        hand_type="Right" if hand_type == HandType.right else "Left",
        selfie=True,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Try index 1 or 2.")
        stop_event.set()
        return

    scale = 0.5
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = frame[..., ::-1]

        _, joint_pos, keypoint_2d, wrist_rot, wrist_pos = detector.detect(rgb)
        if keypoint_2d is not None:
            frame = detector.draw_skeleton_on_image(frame, keypoint_2d, style="default")

        if joint_pos is not None and wrist_pos is not None:
            wrist = np.array([float(wrist_pos[0]), float(wrist_pos[1]), float(wrist_pos[2])])
            cv2.putText(frame, f"Wrist: {wrist_pos.round(2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            thumb_tip = joint_pos[4]
            pinky_tip = joint_pos[20]
            # print(joint_pos)
            # distance between thumb and pinky → 0 (closed) to 1 (open)
            hand_dist_unnorm = np.linalg.norm(thumb_tip - pinky_tip)
            hand_dist = np.clip(hand_dist_unnorm, min_dist, max_dist)

            hand_open = (hand_dist - min_dist) / (max_dist - min_dist)

            hand_open = 1-np.clip(hand_open * 1.0, 0, 1)
            print(hand_open)
            with data_lock:
                latest_hand_pos = wrist
        else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# Main thread: MuJoCo env + render (OpenGL MUST stay on main thread)
# ----------------------------
env = suite.make(
    env_name="NutAssembly",
    robots="Panda",
    gripper_types="InspireRightHand",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    camera_names="birdview"
)
obs = env.reset()

# Start webcam detection in background
cam_thread = threading.Thread(target=webcam_thread_fn, daemon=True)
cam_thread.start()

print("Live teleoperation started. Press Ctrl+C to quit.\n")
# print(env.robots[0].gripper['right'].joints)
scale = 4.0
try:
    while not stop_event.is_set():
        with data_lock:
            hand_pos = None if latest_hand_pos is None else latest_hand_pos.copy()
        robot_pos = obs["robot0_eef_pos"]
        if hand_pos is not None and hand_start is None:
            hand_start = hand_pos.copy()
            robot_start = robot_pos.copy()
        action = np.zeros(12)

        if hand_pos is not None and hand_start is not None:
            hand_delta = hand_pos - hand_start
            hand_delta_robot = np.array([
                hand_delta[2],
                -hand_delta[0],
                hand_delta[1]
            ])
            robot_target = robot_start + scale * hand_delta_robot

            pd.target = robot_target
            control = pd.update(robot_pos)

            action[:3] = control

            # map to gripper action (assuming 0=closed, 1=open)
            action[6:] = hand_open
        obs, reward, done, info = env.step(action)
        env.render()  # OpenGL render stays on main thread

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stop_event.set()
    cam_thread.join(timeout=3)
    env.close()
    print("Shutdown complete.")
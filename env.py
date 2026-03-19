import cv2
import numpy as np
import threading
import robosuite as suite
from dex_retargeting.constants import HandType
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
import argparse


import numpy as np

def estimate_frame_from_hand_points(wrist, index_mcp, middle_mcp):
    """
    Returns rotation matrix where:
      - x: points from middle_mcp to index_mcp (finger spread axis)
      - y: palm normal (facing away from palm)
      - z: points from wrist toward knuckles (finger extension axis)
    """
    # Finger spread axis (x)
    x_vec = index_mcp - middle_mcp
    x_vec /= np.linalg.norm(x_vec)

    # Wrist-to-knuckles axis (approximate z)
    z_raw = ((index_mcp + middle_mcp) / 2) - wrist
    z_raw /= np.linalg.norm(z_raw)

    # Palm normal via cross product (y), then re-orthogonalize z
    y_vec = np.cross(z_raw, x_vec)
    y_vec /= np.linalg.norm(y_vec)
    z_vec = np.cross(x_vec, y_vec)
    z_vec /= np.linalg.norm(z_vec)

    return np.stack([x_vec, y_vec, z_vec], axis=1)  # 3x3 rotation matrix

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

latest_hand_pos = None
latest_wrist_frame = None
initial_hand_rot = None
initial_robot_rot = None
USE_INDIVIDUAL_FINGERS = False
hand_open = 0
index_open =0
middle_open = 0
ring_open = 0
pinky_open = 0
thumb_open = 0
min_dist = 0.02
max_dist = 0.15

open_hand =  np.array([-1.4, -1.4, -1.4, -1.4, -2.9, 2.9])
close_hand =  np.array([1.4, 1.4, 1.4, 1.4, 2.9, 2.9])
MP_TO_ROBOT = np.array([
    [ 0,  0,  1],  
    [-1,  0,  0],  
    [ 0, -1,  0],  
])

# ----------------------------
# Background thread: webcam + hand detection only (doesn't handle actions)
# ----------------------------
def webcam_thread_fn():
    global latest_hand_pos, hand_open, index_open, middle_open, ring_open, pinky_open, thumb_open, pinch_dist, latest_wrist_frame

    mp_hands = mp.solutions.hands   

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                #hand landmarks from mediapipe
                wrist = hand_landmarks.landmark[0]
                index_mcp = hand_landmarks.landmark[5]
                pinky_mcp = hand_landmarks.landmark[17]
                middle_mcp = hand_landmarks.landmark[9]
                thumb_tip = hand_landmarks.landmark[4]
                pinky_tip = hand_landmarks.landmark[20]
                index_tip =  hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                wrist_f = np.array([wrist.x, wrist.y, wrist.z])
                index_f = np.array([index_mcp.x, index_mcp.y, index_mcp.z])
                middle_f = np.array([middle_mcp.x, middle_mcp.y, middle_mcp.z])
                
                wrist_frame = estimate_frame_from_hand_points(wrist_f,index_f,middle_f)

                x = wrist.x
                y = wrist.y
                
                depth = np.sqrt(
                    (index_mcp.x - pinky_mcp.x)**2 +
                    (index_mcp.y - pinky_mcp.y)**2
                )

                pinch_dist = np.sqrt(
                    (thumb_tip.x - pinky_tip.x)**2 +
                    (thumb_tip.y - pinky_tip.y)**2
                )

                hand_vec = np.array([x, y, depth])

                index_to_palm = np.sqrt(
                    (index_tip.x - x)**2 +
                    (index_tip.y - y)**2
                )
                middle_to_palm = np.sqrt(
                    (middle_tip.x - x)**2 +
                    (middle_tip.y - y)**2
                )
                ring_to_palm = np.sqrt(
                    (ring_tip.x - x)**2 +
                    (ring_tip.y - y)**2
                )
                pinky_to_palm = np.sqrt(
                (pinky_tip.x - x)**2 +
                (pinky_tip.y - y)**2
            )
                thum_to_palm = np.sqrt(
                (thumb_tip.x - pinky_mcp.x)**2 +
                (thumb_tip.y - pinky_mcp.y)**2
            )
                

                index_open =  1-np.clip(index_to_palm * 4, 0, 1)
                middle_open = 1-np.clip(middle_to_palm * 4, 0, 1)
                ring_open = 1-np.clip(ring_to_palm * 4, 0, 1)
                hand_open = 1-np.clip(pinch_dist * 4, 0, 1)
                pinky_open = 1 - np.clip(pinky_to_palm * 4, 0, 1)
                thumb_open = 1 - np.clip(thum_to_palm * 4, 0, 1)
                hand_open = 1-np.clip(pinch_dist * 4, 0, 1)

                with data_lock:
                    latest_hand_pos = hand_vec
                    if wrist_frame is not None:
                        latest_wrist_frame = wrist_frame

        if cv2.waitKey(1) & 0xFF == 27:
            stop_event.set()

    cap.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--finger_mode",
    action="store_true",
    help="Use individual finger retargeting instead of unified grip"
)
args = parser.parse_args()

USE_INDIVIDUAL_FINGERS = args.finger_mode
print("Finger mode:", "Individual" if USE_INDIVIDUAL_FINGERS else "Unified")

#robosuite environment
env = suite.make(
    env_name="PickPlace",
    robots="Panda",
    gripper_types="InspireRightHand",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
    camera_names="birdview"
)
obs = env.reset()
env.viewer.set_camera(0)
# Start webcam detection in background
cam_thread = threading.Thread(target=webcam_thread_fn, daemon=True)
cam_thread.start()

print("Live teleoperation started. Press Ctrl+C to quit.\n")
try:
    detected_hands = 0
    locked_state = None
    pickup_state = None
    pickup_hold = 0
    holding = False
    while not stop_event.is_set():
        with data_lock:
            hand_pos = None if latest_hand_pos is None else latest_hand_pos.copy()
            wrist_frame = None if latest_wrist_frame is None else latest_wrist_frame.copy()
            _thumb_open = thumb_open ** 0.7
            _index_open  = index_open** 0.7
            _middle_open = middle_open** 0.7
            _ring_open   = ring_open** 0.7
            _pinky_open  = pinky_open** 0.7
        robot_pos = obs["robot0_eef_pos"]
        action = np.zeros(12)
        if hand_pos is not None:
            detected_hands+=1
        if hand_pos is not None and detected_hands>50:

            # x range is about -0.5 to 0.5
            # y range is about 0.9 to 1.4
            # z stable range is about 0.05 (back) to 0.25 (front) on camera, -0.2 to 0.2 for robot


            milk_pos = obs['Milk_pos'].copy()
            # milk_pos[0]+=0.01
            if not pickup_state:
                milk_pos[2] += 0.3
            else:
                milk_pos[2] += 0.1

            cereal_pos = obs["Cereal_pos"].copy()
            # cereal_pos[0]+=0.01
            if not pickup_state:
                cereal_pos[2] += 0.3
            else:
                cereal_pos[2] += 0.1

            bread_pos = obs["Bread_pos"].copy()
            # bread_pos[0]+=0.01
            if not pickup_state:
                bread_pos[2] += 0.3
            else:
                bread_pos[2] += 0.1

            can_pos = obs["Can_pos"].copy()
            # can_pos[0]+=0.01
            if not pickup_state:
                can_pos[2] += 0.3
            else:
                can_pos[2] += 0.1

            milk_distance = np.linalg.norm(robot_pos-milk_pos)
            cereal_distance = np.linalg.norm(robot_pos-cereal_pos)
            bread_distance = np.linalg.norm(robot_pos-bread_pos)
            can_distance = np.linalg.norm(robot_pos-can_pos)

            if locked_state=="Milk":
                print(milk_distance)
                pd.target = milk_pos
                if pickup_state==1 and milk_distance<0.07:
                    pickup_state = 2
            
            elif locked_state=="Cereal":
                print(cereal_distance)
                pd.target = cereal_pos
                if pickup_state==1 and cereal_distance<0.07:
                    pickup_state = 2
            
            elif locked_state=="Bread":
                pd.target = bread_pos
                print(bread_distance)
                if pickup_state==1 and bread_distance<0.07:
                    pickup_state = 2

            elif locked_state=="Can":
                pd.target = can_pos
                print(can_distance)
                if pickup_state==1 and can_distance<0.07:
                    pickup_state = 2

            else:
                pd.target = [(hand_pos[2]-0.05)*3 -0.3, (hand_pos[0]-0.2)/0.6-0.5, (1-hand_pos[1])/2 + 0.9]
            

            # print([milk_distance, cereal_distance, bread_distance, can_distance])
            if not holding:
                if milk_distance<.15 and locked_state==None:
                    locked_state = "Milk"
                    print("LOCKED ONTO MILK")

                elif cereal_distance<.15 and locked_state==None:
                    locked_state = "Cereal"
                    print("LOCKED ONTO CEREAL")
                
                elif bread_distance<.15 and locked_state==None:
                    locked_state = "Bread"
                    print("LOCKED ONTO BREAD")
                
                elif can_distance<.15 and locked_state==None:
                    locked_state = "Can"
                    print("LOCKED ONTO CAN")
            
            # pd.target = [robot_pos[0]-0.1, robot_pos[1], robot_pos[2]]
            control = pd.update(robot_pos)

            action[:3] = control

            if pickup_state==2:
                pickup_hold+=1
            
            if pickup_hold>50:
                locked_state = None
                pickup_state = None
                pickup_hold = 0
                holding = True
            # map to gripper action (assuming 0=closed, 1=open)
            if USE_INDIVIDUAL_FINGERS: 
                action[6] = open_hand[0] + _pinky_open  * (close_hand[0] - open_hand[0])
                action[7] = open_hand[1] + _ring_open   * (close_hand[1] - open_hand[1])
                action[8] = open_hand[2] + _middle_open * (close_hand[2] - open_hand[2])
                action[9] = open_hand[3] + _index_open  * (close_hand[3] - open_hand[3])
                action[10] = open_hand[4] + _thumb_open   * (close_hand[4] - open_hand[4])
            else:   
                #map to all fingers at once
                if locked_state and not pickup_state and hand_open>0.5:
                    pickup_state = 1
                    action[6:] = open_hand
                elif locked_state and not pickup_state==2:
                    action[6:] = open_hand
                elif not locked_state or (locked_state and pickup_state==2):
                    action[6:] = open_hand + hand_open * (close_hand - open_hand) * 0.5
            if wrist_frame is not None:
                #wrist orientation
                robot_world_mat = MP_TO_ROBOT @ wrist_frame
                hand_rot_world = R.from_matrix(robot_world_mat)
                current_rot = R.from_quat(obs['robot0_eef_quat'])

                if initial_hand_rot is None:
                    initial_hand_rot = hand_rot_world
                    initial_robot_rot = current_rot

                relative_hand = hand_rot_world * initial_hand_rot.inv()

                target_robot_rot = relative_hand * initial_robot_rot

                rot_error = target_robot_rot * current_rot.inv()
                rotvec = rot_error.as_rotvec()
                
                #map to robot orientation 
                # action[3:6] = 0.8 * rotvec
      
            
        obs, reward, done, info = env.step(action)
        env.render()  # OpenGL render stays on main thread

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stop_event.set()
    cam_thread.join(timeout=3)
    env.close()
    print("Shutdown complete.")
import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from pytransform3d import rotations  # [WRIST PATCH] For rotation matrix/quaternion conversions

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


# [WRIST PATCH] Convert wrist_rot to a 3x3 rotation matrix
def _wrist_rot_to_matrix(wrist_rot: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert wrist_rot to a 3x3 rotation matrix.
    - If 3x3: use directly
    - If length 4: treat as quaternion (w, x, y, z)
    - If None or other shape: ignore
    """
    if wrist_rot is None:
        return None
    wrist_rot = np.asarray(wrist_rot)
    if wrist_rot.shape == (3, 3):
        return wrist_rot
    if wrist_rot.shape == (4,):
        return rotations.matrix_from_quaternion(wrist_rot)
    return None

# View correction for mapping camera(hand) frame → robot frame.
# This is the tilt that you experimentally found to look good:
# axis = [1, 0, 1], angle = -90 deg
R_tilt = rotations.matrix_from_axis_angle(
    np.array([1.0, 0.0, 1.0, -np.pi / 2.0])
)

# Final view rotation (you can later add more corrections if needed)
WRIST_VIEW_ROT = R_tilt


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Scene setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    # Scaling rules depending on robot type
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    # Load GLB-based URDF if exists
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    # Adjust initial robot pose to avoid clipping
    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # [WRIST PATCH] Save initial robot pose (position+rotation) for wrist rotation blending
    base_robot_pose = robot.get_pose()
    base_pos = base_robot_pose.p.copy()
    calib_wrist_R = [None]  # Store first detected wrist orientation for calibration

    # Mapping from retargeting joint order to SAPIEN joint order
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error("Failed to fetch camera frame for 5 seconds.")
            return

        # Run hand detector
        num_hands, joint_pos, keypoint_2d, wrist_rot_raw = detector.detect(rgb)
        wrist_R = _wrist_rot_to_matrix(wrist_rot_raw)

        # Auto calibration: first time we get a valid wrist_R
        if wrist_R is not None and calib_wrist_R[0] is None:
            calib_wrist_R[0] = wrist_R.copy()
            logger.info("Wrist orientation calibrated (auto).")

        # Draw skeleton for visualization
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            # Manual calibration when user presses 'c'
            if wrist_R is not None:
                calib_wrist_R[0] = wrist_R.copy()
                logger.info("Wrist orientation calibrated (manual 'c').")

        if joint_pos is None:
            logger.warning(f"{hand_type} hand not detected.")
        else:
            # Build retargeting reference (position or vector)
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices

            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # Finger retargeting
            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos[retargeting_to_sapien])

            # Wrist orientation retargeting
            if wrist_R is not None and calib_wrist_R[0] is not None:
                # Relative wrist rotation wrt calibrated pose
                R_rel = wrist_R @ calib_wrist_R[0].T

                base_T = base_robot_pose.to_transformation_matrix()
                R_robot0 = base_T[:3, :3]

                # Apply view correction and base rotation
                R_robot = WRIST_VIEW_ROT @ R_rel @ R_robot0
                q_robot = rotations.quaternion_from_matrix(R_robot)

                # Keep position fixed, only update rotation
                new_pose = sapien.Pose(base_pos, q_robot)
                robot.set_pose(new_pose)

        # Render a few times for smoother display
        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    """
    Detect the human hand pose from a live camera stream and retarget it to a robot hand.
    Now supports wrist orientation retargeting (robot rotates like the human wrist).
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1) # original size was 1000 but reduced to 1 (Erin's PC has delay issue)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)

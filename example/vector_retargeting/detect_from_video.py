import pickle
from pathlib import Path

import cv2
import numpy as np
import tqdm
import tyro

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from single_hand_detector import SingleHandDetector
from show_realtime_retargeting import (
    apply_joint_output_transform,
    canonicalize_x2_landmarks_for_control_mode,
    get_x2_control_mode,
    is_x2_robot,
    parse_joint_signs,
    transform_ref_value,
)


def get_robot_dir() -> Path:
    project_root = Path(__file__).absolute().parent.parent.parent
    robot_dir = project_root / "assets" / "robots" / "hands"
    return robot_dir if robot_dir.exists() else project_root


def get_signed_output_joint_limits(
    joint_limits: np.ndarray, joint_signs: np.ndarray
) -> np.ndarray:
    output_limits = joint_limits.copy()
    for index, sign in enumerate(joint_signs):
        if sign < 0:
            lower, upper = output_limits[index]
            output_limits[index] = [-upper, -lower]
    return output_limits


def retarget_video(
    retargeting: SeqRetargeting,
    video_path: str,
    output_path: str,
    config_path: str,
    hand_type: HandType,
):
    cap = cv2.VideoCapture(video_path)

    data = []
    frame_indices = []

    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        hand_label = "Right" if hand_type is HandType.right else "Left"
        detector = SingleHandDetector(hand_type=hand_label, selfie=False)
        x2_mode = is_x2_robot(Path(config_path).stem) or is_x2_robot(config_path)
        x2_control_mode = get_x2_control_mode(hand_type)
        ref_transform = "x2" if x2_mode else None
        joint_sign = None
        joint_signs = parse_joint_signs(joint_sign, retargeting.joint_names)
        joint_gains = np.ones(len(retargeting.joint_names), dtype=np.float32)
        joint_limits = retargeting.optimizer.robot.joint_limits
        output_joint_limits = get_signed_output_joint_limits(joint_limits, joint_signs)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm.tqdm(total=length) as pbar:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break
                current_frame_index = frame_index
                frame_index += 1
                pbar.update(1)

                rgb = frame[..., ::-1]
                (
                    num_box,
                    joint_pos,
                    keypoint_2d,
                    mediapipe_wrist_rot,
                    detection_meta,
                ) = detector.detect(rgb, return_meta=True)
                if num_box == 0:
                    continue
                joint_pos, _ = canonicalize_x2_landmarks_for_control_mode(
                    joint_pos,
                    x2_control_mode,
                    x2_mode,
                    detection_meta,
                )

                retargeting_type = retargeting.optimizer.retargeting_type
                indices = retargeting.optimizer.target_link_human_indices
                if retargeting_type == "POSITION":
                    indices = indices
                    ref_value = joint_pos[indices, :]
                else:
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = (
                        joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    )
                ref_value = transform_ref_value(ref_value, ref_transform)
                qpos = retargeting.retarget(ref_value)
                qpos = apply_joint_output_transform(
                    qpos, joint_signs, joint_gains, output_joint_limits
                )
                data.append(qpos)
                frame_indices.append(current_frame_index)

        meta_data = dict(
            config_path=config_path,
            hand_type=hand_type.name,
            x2_joint_sign=joint_sign or "default",
            frame_indices=frame_indices,
            dof=len(retargeting.optimizer.robot.dof_joint_names),
            joint_names=retargeting.optimizer.robot.dof_joint_names,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(dict(data=data, meta_data=meta_data), f)

        retargeting.verbose()
        cap.release()
        cv2.destroyAllWindows()


def main(
    robot_name: RobotName,
    video_path: str,
    output_path: str,
    retargeting_type: RetargetingType,
    hand_type: HandType,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        video_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand. X2 is an exception in this project:
            left/right are two control modes over the same rh_* URDF, so left landmarks are canonicalized first.
    """

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    if config_path is None:
        raise ValueError(
            f"No config for {robot_name.name} {retargeting_type.name} {hand_type.name}."
        )
    robot_dir = get_robot_dir()
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_video(retargeting, video_path, output_path, str(config_path), hand_type)


if __name__ == "__main__":
    tyro.cli(main)

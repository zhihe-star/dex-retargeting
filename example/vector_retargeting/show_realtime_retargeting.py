import multiprocessing
import time
from pathlib import Path
from queue import Empty, Full
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.kinematics_adaptor import KinematicAdaptor
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def get_robot_dir() -> Path:
    project_root = Path(__file__).absolute().parent.parent.parent
    robot_dir = project_root / "assets" / "robots" / "hands"
    return robot_dir if robot_dir.exists() else project_root


def is_realsense_camera(camera_path: Optional[str]) -> bool:
    if camera_path is None:
        return False
    return camera_path.lower() in {"realsense", "rs", "intel-realsense"}


def put_latest_frame(queue: multiprocessing.Queue, image: np.ndarray):
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass

    try:
        queue.put_nowait(image)
    except Full:
        pass


def is_x2_robot(robot_name: str) -> bool:
    return "x^2" in robot_name or "x2" in robot_name


X2_RIGHT_ROOT_QUAT = (0.5, -0.5, -0.5, -0.5)
X2_LEFT_ROOT_QUAT = (0.5, 0.5, -0.5, 0.5)


def get_x2_upright_pose(z: float = -0.05, hand_type: str = "Right") -> sapien.Pose:
    # Finger direction: local +X -> world +Z. Right palm local +Y -> world +X;
    # left palm local +Y -> world -X.
    quat = X2_LEFT_ROOT_QUAT if hand_type == "Left" else X2_RIGHT_ROOT_QUAT
    return sapien.Pose([0, 0, z], quat)


X2_LEFT_JOINT_SIGN = (
    "fingers=-1,"
    "rh_THJ4=-1,"
    "rh_THJ3=-1,"
    "rh_THJ2=1,"
    "rh_THJ1=1"
)
X2_FOUR_FINGER_JOINT_TOKENS = ("_FFJ", "_MFJ", "_RFJ", "_LFJ")
X2_LEFT_LANDMARK_MIRROR = np.array([-1.0, 1.0, 1.0], dtype=np.float32)
X2_LANDMARK_SEMANTIC_MAP = (
    "thumb->TH, index->FF, middle->MF, ring->RF, pinky->LF"
)


def get_x2_control_mode(hand_type) -> str:
    hand_name = getattr(hand_type, "name", str(hand_type)).lower()
    return "left_mode" if hand_name == "left" else "right_mode"


def canonicalize_x2_landmarks_for_control_mode(
    joint_pos: Optional[np.ndarray],
    control_mode: str,
    enabled: bool,
    detection_meta: Optional[dict] = None,
) -> tuple[Optional[np.ndarray], dict]:
    if joint_pos is None or not enabled:
        return joint_pos, {"enabled": False}

    palm_frame_points = None
    palm_frame_info = {}
    if detection_meta is not None:
        palm_frame_points = detection_meta.get("palm_frame_landmarks")
        palm_frame_info = detection_meta.get("palm_frame_info", {})

    if palm_frame_points is not None:
        canonical = palm_frame_points.copy()
        source = "palm_frame"
    else:
        canonical = joint_pos.copy()
        source = "operator_fallback"

    left_adapter = control_mode == "left_mode"
    if left_adapter:
        canonical *= X2_LEFT_LANDMARK_MIRROR
    return canonical, {
        "enabled": True,
        "source": source,
        "palm_frame": palm_frame_points is not None,
        "palm_frame_info": palm_frame_info,
        "left_adapter": left_adapter,
        "mirror": tuple(float(value) for value in X2_LEFT_LANDMARK_MIRROR),
        "semantic_map": X2_LANDMARK_SEMANTIC_MAP,
    }


def format_x2_landmark_canonicalization(info: dict) -> str:
    if not info.get("enabled"):
        return "off"
    mirror = info.get("mirror", (1.0, 1.0, 1.0))
    palm_status = "on" if info.get("palm_frame") else "fallback"
    left_status = "on" if info.get("left_adapter") else "off"
    return (
        f"source={info.get('source', 'unknown')}; "
        f"palm_frame={palm_status}; left_adapter={left_status}; "
        f"mirror xyz=({mirror[0]:+.0f},{mirror[1]:+.0f},{mirror[2]:+.0f}); "
        f"semantic {info.get('semantic_map', X2_LANDMARK_SEMANTIC_MAP)}"
    )


def is_x2_four_finger_bend_joint(joint_name: str) -> bool:
    return any(token in joint_name for token in X2_FOUR_FINGER_JOINT_TOKENS)


def apply_x2_palm_flex_guard(
    qpos: np.ndarray,
    joint_names: list[str],
    enabled: bool,
    control_mode: str = "right_mode",
) -> np.ndarray:
    if not enabled:
        return qpos

    guarded = qpos.copy()
    for index, joint_name in enumerate(joint_names):
        if is_x2_four_finger_bend_joint(joint_name):
            value = float(guarded[index])
            guarded[index] = max(value, 0.0)
    return guarded


def format_x2_palm_flex_guard_changes(
    before: np.ndarray,
    after: np.ndarray,
    joint_names: list[str],
    threshold: float = 1e-6,
) -> str:
    changes = []
    for joint_name, old_value, new_value in zip(joint_names, before, after):
        if not is_x2_four_finger_bend_joint(joint_name):
            continue
        if abs(float(old_value) - float(new_value)) > threshold:
            changes.append(f"{joint_name}:{float(old_value):+.3f}->{float(new_value):+.3f}")
    return ", ".join(changes) if changes else "none"


class JointSignKinematicAdaptor(KinematicAdaptor):
    def __init__(
        self,
        robot,
        target_joint_names: list[str],
        joint_names: list[str],
        joint_signs: np.ndarray,
        wrapped_adaptor: Optional[KinematicAdaptor] = None,
    ):
        super().__init__(robot, target_joint_names)
        self.wrapped_adaptor = wrapped_adaptor
        sign_by_name = dict(zip(joint_names, joint_signs))
        self.pin_signs = np.ones(robot.dof, dtype=np.float32)
        for index, name in enumerate(robot.dof_joint_names):
            self.pin_signs[index] = sign_by_name.get(name, 1.0)
        self.target_signs = self.pin_signs[self.idx_pin2target]

    def forward_qpos(self, qpos: np.ndarray) -> np.ndarray:
        adapted_qpos = qpos.copy()
        adapted_qpos *= self.pin_signs
        if self.wrapped_adaptor is not None:
            adapted_qpos = self.wrapped_adaptor.forward_qpos(adapted_qpos)
        return adapted_qpos

    def backward_jacobian(self, jacobian: np.ndarray) -> np.ndarray:
        if self.wrapped_adaptor is not None:
            target_jacobian = self.wrapped_adaptor.backward_jacobian(jacobian)
        else:
            target_jacobian = jacobian[..., self.idx_pin2target]
        return target_jacobian * self.target_signs


def enable_joint_sign_kinematics(retargeting, joint_signs: np.ndarray) -> None:
    optimizer = retargeting.optimizer
    wrapped_adaptor = optimizer.adaptor
    adaptor = JointSignKinematicAdaptor(
        optimizer.robot,
        optimizer.target_joint_names,
        retargeting.joint_names,
        joint_signs,
        wrapped_adaptor=wrapped_adaptor,
    )
    optimizer.set_kinematic_adaptor(adaptor)


def set_retargeting_optimizer_limits(retargeting, full_joint_limits: np.ndarray) -> None:
    target_limits = full_joint_limits[retargeting.optimizer.idx_pin2target]
    retargeting.joint_limits = target_limits.copy()
    retargeting.optimizer.set_joint_limit(target_limits)
    retargeting.last_qpos = np.clip(
        retargeting.last_qpos, target_limits[:, 0], target_limits[:, 1]
    )


def parse_joint_signs(joint_sign: Optional[str], joint_names: list[str]) -> np.ndarray:
    signs = np.ones(len(joint_names), dtype=np.float32)
    if joint_sign is None or joint_sign.lower() in {"", "none", "default"}:
        return signs

    groups = {
        "all": joint_names,
        "ff": [name for name in joint_names if "_FF" in name],
        "index": [name for name in joint_names if "_FF" in name],
        "mf": [name for name in joint_names if "_MF" in name],
        "middle": [name for name in joint_names if "_MF" in name],
        "rf": [name for name in joint_names if "_RF" in name],
        "ring": [name for name in joint_names if "_RF" in name],
        "lf": [name for name in joint_names if "_LF" in name],
        "pinky": [name for name in joint_names if "_LF" in name],
        "th": [name for name in joint_names if "_TH" in name],
        "thumb": [name for name in joint_names if "_TH" in name],
        "fingers": [name for name in joint_names if any(f"_{prefix}" in name for prefix in ("FF", "MF", "RF", "LF"))],
    }

    for raw_item in joint_sign.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item in {"1", "+1", "-1"}:
            key, value = "all", item
        elif "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid joint sign item '{item}'. Use e.g. all=-1 or rh_FFJ2=-1."
            )

        key = key.strip()
        sign = float(value.strip())
        if sign not in {-1.0, 1.0}:
            raise ValueError(f"Joint sign for {key} must be 1 or -1, got {sign}.")

        names = groups.get(key.lower(), [key])
        missing_names = [name for name in names if name not in joint_names]
        if missing_names:
            raise ValueError(f"Unknown joint sign target(s): {missing_names}")
        for name in names:
            signs[joint_names.index(name)] = sign
    return signs


def parse_joint_gains(joint_gain: Optional[str], joint_names: list[str]) -> np.ndarray:
    gains = np.ones(len(joint_names), dtype=np.float32)
    if joint_gain is None or joint_gain.lower() in {"", "none", "default"}:
        return gains

    groups = {
        "all": joint_names,
        "ff": [name for name in joint_names if "_FF" in name],
        "index": [name for name in joint_names if "_FF" in name],
        "mf": [name for name in joint_names if "_MF" in name],
        "middle": [name for name in joint_names if "_MF" in name],
        "rf": [name for name in joint_names if "_RF" in name],
        "ring": [name for name in joint_names if "_RF" in name],
        "lf": [name for name in joint_names if "_LF" in name],
        "pinky": [name for name in joint_names if "_LF" in name],
        "th": [name for name in joint_names if "_TH" in name],
        "thumb": [name for name in joint_names if "_TH" in name],
        "fingers": [name for name in joint_names if any(f"_{prefix}" in name for prefix in ("FF", "MF", "RF", "LF"))],
    }

    for raw_item in joint_gain.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            key, value = "all", item

        key = key.strip()
        gain = float(value.strip())
        if gain < 0:
            raise ValueError(f"Joint gain for {key} must be non-negative, got {gain}.")

        names = groups.get(key.lower(), [key])
        missing_names = [name for name in names if name not in joint_names]
        if missing_names:
            raise ValueError(f"Unknown joint gain target(s): {missing_names}")
        for name in names:
            gains[joint_names.index(name)] = gain
    return gains


def apply_joint_output_transform(
    qpos: np.ndarray,
    joint_signs: np.ndarray,
    joint_gains: np.ndarray,
    joint_limits: np.ndarray,
) -> np.ndarray:
    transformed_qpos = qpos * joint_signs * joint_gains
    return np.clip(transformed_qpos, joint_limits[:, 0], joint_limits[:, 1])


def get_signed_output_joint_limits(
    joint_limits: np.ndarray, joint_signs: np.ndarray
) -> np.ndarray:
    output_limits = joint_limits.copy()
    for index, sign in enumerate(joint_signs):
        if sign < 0:
            lower, upper = output_limits[index]
            output_limits[index] = [-upper, -lower]
    return output_limits


def format_qpos_by_finger(joint_names: list[str], qpos: np.ndarray) -> str:
    groups = [
        ("FF", "_FF"),
        ("MF", "_MF"),
        ("RF", "_RF"),
        ("LF", "_LF"),
        ("TH", "_TH"),
    ]
    parts = []
    for label, token in groups:
        values = [
            f"{name}:{qpos[i]:+.3f}"
            for i, name in enumerate(joint_names)
            if token in name
        ]
        if values:
            parts.append(f"{label}[{', '.join(values)}]")
    return " ".join(parts)


def get_limit_hits(
    joint_names: list[str],
    qpos: np.ndarray,
    joint_limits: np.ndarray,
    margin: float,
) -> list[str]:
    hits = []
    if len(joint_names) != len(qpos) or len(joint_limits) != len(qpos):
        return hits

    for name, value, limits in zip(joint_names, qpos, joint_limits):
        lower, upper = limits
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        lower_dist = value - lower
        upper_dist = upper - value
        if min(lower_dist, upper_dist) <= margin:
            side = "low" if lower_dist <= upper_dist else "high"
            hits.append(
                f"{name}:{value:+.3f}/{side}[{lower:+.2f},{upper:+.2f}]"
            )
    return hits


def format_limit_hits(
    joint_names: list[str],
    qpos: np.ndarray,
    joint_limits: np.ndarray,
    margin: float,
) -> str:
    hits = get_limit_hits(joint_names, qpos, joint_limits, margin)
    return ", ".join(hits) if hits else "none"


def reset_retargeting_pose(
    retargeting,
    robot,
    joint_signs: np.ndarray,
    joint_gains: np.ndarray,
    joint_limits: np.ndarray,
    retargeting_to_sapien: np.ndarray,
    output_joint_limits: Optional[np.ndarray] = None,
):
    if output_joint_limits is None:
        output_joint_limits = joint_limits

    retargeting.reset()
    neutral_qpos = retargeting.get_qpos()
    output_qpos = apply_joint_output_transform(
        neutral_qpos, joint_signs, joint_gains, output_joint_limits
    )
    robot.set_qpos(output_qpos[retargeting_to_sapien])


def remap_human_indices(
    target_link_human_indices: np.ndarray, finger_map: Optional[str]
) -> np.ndarray:
    if finger_map is None or finger_map.lower() in {"", "default", "none"}:
        return target_link_human_indices

    # MediaPipe landmark ids used by this project:
    # thumb: 1/2/3/4, index: 6/8, middle: 10/12, ring: 14/16, pinky: 18/20.
    maps = {
        "reverse": {
            6: 18,
            8: 20,
            10: 14,
            12: 16,
            14: 10,
            16: 12,
            18: 6,
            20: 8,
        },
        "mirror": {
            6: 18,
            8: 20,
            10: 14,
            12: 16,
            14: 10,
            16: 12,
            18: 6,
            20: 8,
        },
        "swap-index-pinky": {
            6: 18,
            8: 20,
            18: 6,
            20: 8,
        },
        "index-pinky": {
            6: 18,
            8: 20,
            18: 6,
            20: 8,
        },
    }

    key = finger_map.lower()
    if key not in maps:
        raise ValueError(
            f"Unknown finger_map '{finger_map}'. "
            "Use default, reverse, or swap-index-pinky."
        )

    remapped = target_link_human_indices.copy()
    for old_index, new_index in maps[key].items():
        remapped[target_link_human_indices == old_index] = new_index
    return remapped


def transform_ref_value(ref_value: np.ndarray, ref_transform: Optional[str]) -> np.ndarray:
    if ref_transform is None or ref_transform.lower() in {"", "none", "default"}:
        return ref_value

    def to_x2_frame(value: np.ndarray) -> np.ndarray:
        transformed = value[:, [2, 0, 1]].copy()
        return transformed

    def mirror_four_finger_spread(value: np.ndarray) -> np.ndarray:
        transformed = value.copy()
        finger_rows = [1, 2, 3, 4, 6, 7, 8, 9]
        valid_rows = [index for index in finger_rows if index < len(transformed)]
        transformed[valid_rows, 2] *= -1
        return transformed

    key = ref_transform.lower().replace("-", "_")
    if key == "x2_plain":
        return to_x2_frame(ref_value)
    if key == "x2":
        # MediaPipe vectors after SingleHandDetector are mostly +z along fingers,
        # while x2's URDF fingers extend along +x. Do not mirror the four fingers;
        # the x2 URDF link names are mapped to physical fingers in the config.
        return to_x2_frame(ref_value)
    if key == "x2_mirror":
        return mirror_four_finger_spread(to_x2_frame(ref_value))
    if key == "x2_flip_z":
        transformed = to_x2_frame(ref_value)
        transformed[:, 2] *= -1
        return transformed
    if key == "x2_flip_y":
        transformed = to_x2_frame(ref_value)
        transformed[:, 1] *= -1
        return transformed
    if key == "x2_flip_yz":
        transformed = to_x2_frame(ref_value)
        transformed[:, 1] *= -1
        transformed[:, 2] *= -1
        return transformed

    raise ValueError(
        f"Unknown ref_transform '{ref_transform}'. "
        "Use none, x2, x2_plain, x2_mirror, x2_flip_y, x2_flip_z, or x2_flip_yz."
    )


def parse_target_vector_gains(target_vector_gain: Optional[str], retargeting) -> np.ndarray:
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type == "POSITION":
        raise ValueError("--target-vector-gain only supports vector-style retargeting.")

    num_vectors = len(optimizer.task_link_names)
    gains = np.ones(num_vectors, dtype=np.float32)
    if target_vector_gain is None or target_vector_gain.lower() in {"", "none", "default"}:
        return gains

    human_task_indices = np.asarray(optimizer.target_link_human_indices)[1]
    groups = {
        "all": list(range(num_vectors)),
        "thumb": [i for i, index in enumerate(human_task_indices) if index in {1, 2, 3, 4}],
        "thumb_cmc": [i for i, index in enumerate(human_task_indices) if index == 1],
        "thumb_mcp": [i for i, index in enumerate(human_task_indices) if index == 2],
        "thumb_base": [i for i, index in enumerate(human_task_indices) if index == 2],
        "thumb_tip": [i for i, index in enumerate(human_task_indices) if index == 4],
        "thumb_mid": [i for i, index in enumerate(human_task_indices) if index in {2, 3}],
        "thumb_middle": [i for i, index in enumerate(human_task_indices) if index in {2, 3}],
        "thumb_ip": [i for i, index in enumerate(human_task_indices) if index == 3],
        "index": [i for i, index in enumerate(human_task_indices) if index in {6, 8}],
        "index_tip": [i for i, index in enumerate(human_task_indices) if index == 8],
        "index_mid": [i for i, index in enumerate(human_task_indices) if index == 6],
        "middle": [i for i, index in enumerate(human_task_indices) if index in {10, 12}],
        "middle_tip": [i for i, index in enumerate(human_task_indices) if index == 12],
        "middle_mid": [i for i, index in enumerate(human_task_indices) if index == 10],
        "ring": [i for i, index in enumerate(human_task_indices) if index in {14, 16}],
        "ring_tip": [i for i, index in enumerate(human_task_indices) if index == 16],
        "ring_mid": [i for i, index in enumerate(human_task_indices) if index == 14],
        "pinky": [i for i, index in enumerate(human_task_indices) if index in {18, 20}],
        "pinky_tip": [i for i, index in enumerate(human_task_indices) if index == 20],
        "pinky_mid": [i for i, index in enumerate(human_task_indices) if index == 18],
    }
    for i, link_name in enumerate(optimizer.task_link_names):
        groups.setdefault(link_name.lower(), []).append(i)
    for i, (origin_link, task_link) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names)
    ):
        key = f"{task_link}-{origin_link}".lower()
        groups.setdefault(key, []).append(i)
        groups.setdefault(key.replace("-", "_"), []).append(i)

    for raw_item in target_vector_gain.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            key, value = "all", item

        key = key.strip().lower().replace("-", "_")
        gain = float(value.strip())
        if gain < 0:
            raise ValueError(f"Target vector gain for {key} must be non-negative, got {gain}.")

        if key.isdigit():
            vector_indices = [int(key)]
        else:
            vector_indices = groups.get(key)
        if vector_indices is None or not vector_indices:
            raise ValueError(
                f"Unknown target vector gain target '{key}'. "
                "Use e.g. thumb=1.5, thumb_mcp=1.4, thumb_tip=1.6, "
                "thumb_ip=1.2, or rh_thtip=1.5."
            )
        for index in vector_indices:
            if index < 0 or index >= num_vectors:
                raise ValueError(f"Target vector index {index} out of range [0, {num_vectors}).")
            gains[index] = gain
    return gains


def apply_target_vector_gains(ref_value: np.ndarray, target_vector_gains: np.ndarray) -> np.ndarray:
    return ref_value * target_vector_gains[:, None]


def format_target_vector_gains(retargeting, target_vector_gains: np.ndarray) -> str:
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type == "POSITION":
        return "position retargeting has no target vectors"
    return ", ".join(
        f"{task_link}:{gain:.2f}"
        for task_link, gain in zip(optimizer.task_link_names, target_vector_gains)
    )


def human_landmark_label(index: int) -> str:
    labels = {
        0: "wrist",
        1: "thumb_cmc",
        2: "thumb_mcp",
        3: "thumb_ip",
        4: "thumb_tip",
        6: "index_mid",
        8: "index_tip",
        10: "middle_mid",
        12: "middle_tip",
        14: "ring_mid",
        16: "ring_tip",
        18: "pinky_mid",
        20: "pinky_tip",
    }
    return labels.get(int(index), f"landmark_{int(index)}")


def format_link_human_mapping(retargeting) -> str:
    indices = retargeting.optimizer.target_link_human_indices
    if retargeting.optimizer.retargeting_type == "POSITION":
        target_link_names = retargeting.optimizer.body_names
        return "\n".join(
            f"  {link_name} <- {human_landmark_label(index)}({int(index)})"
            for link_name, index in zip(target_link_names, indices)
        )

    origin_link_names = retargeting.optimizer.origin_link_names
    task_link_names = retargeting.optimizer.task_link_names
    return "\n".join(
        f"  {task_link} - {origin_link} <- "
        f"{human_landmark_label(task_index)}({int(task_index)}) - "
        f"{human_landmark_label(origin_index)}({int(origin_index)})"
        for origin_link, task_link, origin_index, task_index in zip(
            origin_link_names, task_link_names, indices[0], indices[1]
        )
    )


def debug_print_retargeting(
    frame_id: int,
    joint_pos: np.ndarray,
    raw_ref_value: np.ndarray,
    retarget_ref_value: np.ndarray,
    robot_qpos: np.ndarray,
    signed_robot_qpos: np.ndarray,
    joint_names: list[str],
    joint_limits: np.ndarray,
    limit_margin: float,
    signed_joint_limits: Optional[np.ndarray] = None,
):
    if signed_joint_limits is None:
        signed_joint_limits = joint_limits

    tip_names = ["thumb", "index", "middle", "ring", "pinky"]
    tip_indices = [4, 8, 12, 16, 20]
    pip_indices = [2, 6, 10, 14, 18]
    tip_norms = np.linalg.norm(joint_pos[tip_indices] - joint_pos[0], axis=1)
    pip_norms = np.linalg.norm(joint_pos[pip_indices] - joint_pos[0], axis=1)
    raw_ref_norms = np.linalg.norm(raw_ref_value, axis=1)
    retarget_ref_norms = np.linalg.norm(retarget_ref_value, axis=1)

    print("\n[retarget-debug]", f"frame={frame_id}", flush=True)
    print(
        "  human wrist->tip:",
        ", ".join(f"{name}={value:.3f}" for name, value in zip(tip_names, tip_norms)),
        flush=True,
    )
    print(
        "  human wrist->middle:",
        ", ".join(f"{name}={value:.3f}" for name, value in zip(tip_names, pip_norms)),
        flush=True,
    )
    print(
        "  raw ref norms:",
        np.array2string(raw_ref_norms, precision=3, suppress_small=True),
        flush=True,
    )
    print(
        "  raw ref xyz:",
        np.array2string(raw_ref_value, precision=3, suppress_small=True),
        flush=True,
    )
    print(
        "  retarget ref norms:",
        np.array2string(retarget_ref_norms, precision=3, suppress_small=True),
        flush=True,
    )
    print(
        "  retarget ref xyz:",
        np.array2string(retarget_ref_value, precision=3, suppress_small=True),
        flush=True,
    )
    print("  raw qpos:   ", format_qpos_by_finger(joint_names, robot_qpos), flush=True)
    print(
        "  signed qpos:",
        format_qpos_by_finger(joint_names, signed_robot_qpos),
        flush=True,
    )
    print(
        "  raw near limits:",
        format_limit_hits(joint_names, robot_qpos, joint_limits, limit_margin),
        flush=True,
    )
    if not np.allclose(robot_qpos, signed_robot_qpos):
        print(
            "  signed near limits:",
            format_limit_hits(
                joint_names, signed_robot_qpos, signed_joint_limits, limit_margin
            ),
            flush=True,
        )


def start_retargeting(
    queue: multiprocessing.Queue,
    robot_dir: str,
    config_path: str,
    scaling_factor: Optional[float] = None,
    low_pass_alpha: Optional[float] = None,
    normal_delta: Optional[float] = None,
    robot_visual_scale: Optional[float] = None,
    joint_sign: Optional[str] = None,
    joint_gain: Optional[str] = None,
    target_vector_gain: Optional[str] = None,
    finger_map: Optional[str] = None,
    ref_transform: Optional[str] = None,
    debug_retargeting: bool = False,
    debug_interval: int = 15,
    debug_limit_margin: float = 0.03,
    reset_on_hand_lost: bool = True,
    lost_reset_frames: int = 5,
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    override = {}
    if scaling_factor is not None:
        override["scaling_factor"] = scaling_factor
    if low_pass_alpha is not None:
        override["low_pass_alpha"] = low_pass_alpha
    if normal_delta is not None:
        override["normal_delta"] = normal_delta
    override = override or None
    config = RetargetingConfig.load_from_file(config_path, override)
    if config.target_link_human_indices is not None:
        config.target_link_human_indices = remap_human_indices(
            config.target_link_human_indices, finger_map
        )
    retargeting = config.build()
    target_vector_gains = parse_target_vector_gains(target_vector_gain, retargeting)

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    # Setup
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

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    x2_control_mode = get_x2_control_mode(hand_type)
    if ref_transform is None and is_x2_robot(robot_name):
        # X2 left/right use the same command-frame bending semantics; the
        # world-space hand direction is mirrored by the robot root pose.
        ref_transform = "x2"
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif is_x2_robot(robot_name):
        loader.scale = 0.75
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5
    if robot_visual_scale is not None:
        loader.scale = robot_visual_scale

    if "glb" not in robot_name:
        glb_filepath = filepath.with_name(f"{filepath.stem}_glb{filepath.suffix}")
        filepath = str(glb_filepath if glb_filepath.exists() else filepath)
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif is_x2_robot(robot_name):
        robot.set_pose(get_x2_upright_pose(-0.05, hand_type))
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

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)
    if is_x2_robot(robot_name) and hand_type == "Left" and joint_sign is None:
        logger.warning(
            "x2 left_mode is active. "
            "Using the same single rh_* URDF as right_mode; four-finger commands "
            "keep the same positive closing semantics as right_mode. "
            "The world -X closing direction comes from the robot root pose and "
            "the palm-frame left adapter. "
            f"Use --joint-sign {X2_LEFT_JOINT_SIGN} only for legacy diagnosis."
        )
    joint_signs = parse_joint_signs(joint_sign, retargeting_joint_names)
    joint_gains = parse_joint_gains(joint_gain, retargeting_joint_names)
    joint_limits = retargeting.optimizer.robot.joint_limits
    output_joint_limits = get_signed_output_joint_limits(joint_limits, joint_signs)
    x2_palm_flex_guard_enabled = is_x2_robot(robot_name) and joint_sign is None
    x2_left_signed_kinematics = False
    if is_x2_robot(robot_name) and hand_type == "Left" and joint_sign == X2_LEFT_JOINT_SIGN:
        enable_joint_sign_kinematics(retargeting, joint_signs)
        set_retargeting_optimizer_limits(retargeting, output_joint_limits)
        x2_left_signed_kinematics = True
        logger.warning(
            "x2 left-hand signed kinematics is active. "
            "The optimizer now solves in left-hand command qpos while the "
            "viewer displays the mapped x2 model qpos."
        )
    if is_x2_robot(robot_name) and joint_sign is None and hand_type != "Left":
        logger.warning(
            "x2 detected. If the fingers bend in the opposite direction, try: "
            "--joint-sign all=-1"
        )
    if joint_sign is not None:
        logger.warning(
            "--joint-sign is applied after optimization for visual diagnosis. "
            "If it fixes x2 direction, bake the sign into the URDF joint axis or config."
        )
    if joint_gain is not None:
        logger.warning(
            "--joint-gain is applied after optimization for visual tuning. "
            "If a gain causes near-limit joints, reduce the gain or adjust the retargeting config."
        )
    logger.info(f"Retargeting joints: {retargeting_joint_names}")
    logger.info(f"SAPIEN active joints: {sapien_joint_names}")
    logger.info(f"Retargeting -> SAPIEN index map: {retargeting_to_sapien.tolist()}")
    logger.info(f"Finger map: {finger_map or 'default'}")
    logger.info(f"Reference transform: {ref_transform or 'none'}")
    if is_x2_robot(robot_name):
        logger.info(f"x2 control mode: {x2_control_mode} (single rh_* URDF)")
        logger.info(
            "x2 landmark canonicalization: "
            + format_x2_landmark_canonicalization(
                {
                    "enabled": True,
                    "source": "palm_frame",
                    "palm_frame": True,
                    "left_adapter": x2_control_mode == "left_mode",
                    "mirror": tuple(float(value) for value in X2_LEFT_LANDMARK_MIRROR),
                    "semantic_map": X2_LANDMARK_SEMANTIC_MAP,
                }
            )
        )
        logger.info(
            "x2 palm flex guard: "
            + ("enabled" if x2_palm_flex_guard_enabled else "disabled")
        )
    logger.info("Target link mapping:\n" + format_link_human_mapping(retargeting))
    logger.info(
        "Joint signs: "
        + ", ".join(
            f"{name}:{sign:+.0f}" for name, sign in zip(retargeting_joint_names, joint_signs)
        )
    )
    logger.info(
        "Joint gains: "
        + ", ".join(
            f"{name}:{gain:.2f}" for name, gain in zip(retargeting_joint_names, joint_gains)
        )
    )
    logger.info(
        "Target vector gains: "
        + format_target_vector_gains(retargeting, target_vector_gains)
    )

    frame_id = 0
    lost_frames = 0
    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        _, joint_pos, keypoint_2d, _, detection_meta = detector.detect(
            rgb, return_meta=True
        )
        joint_pos, x2_landmark_info = canonicalize_x2_landmarks_for_control_mode(
            joint_pos,
            x2_control_mode,
            is_x2_robot(robot_name),
            detection_meta,
        )
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            lost_frames += 1
            logger.warning(f"{hand_type} hand is not detected.")
            if (
                reset_on_hand_lost
                and lost_frames == max(lost_reset_frames, 1)
            ):
                reset_retargeting_pose(
                    retargeting,
                    robot,
                    joint_signs,
                    joint_gains,
                    joint_limits,
                    retargeting_to_sapien,
                    output_joint_limits=output_joint_limits,
                )
                logger.warning(
                    f"{hand_type} hand was lost for {lost_frames} frames. "
                    "Retargeting state has been reset."
                )
        else:
            lost_frames = 0
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            raw_ref_value = ref_value.copy()
            ref_value = transform_ref_value(ref_value, ref_transform)
            ref_value = apply_target_vector_gains(ref_value, target_vector_gains)
            qpos = retargeting.retarget(ref_value)
            unguarded_qpos = qpos.copy()
            qpos = apply_x2_palm_flex_guard(
                qpos,
                retargeting_joint_names,
                x2_palm_flex_guard_enabled,
                x2_control_mode,
            )
            if x2_palm_flex_guard_enabled and not np.allclose(qpos, unguarded_qpos):
                retargeting.set_qpos(qpos)
            signed_qpos = apply_joint_output_transform(
                qpos, joint_signs, joint_gains, output_joint_limits
            )
            model_qpos = qpos if x2_left_signed_kinematics else signed_qpos
            if debug_retargeting and frame_id % max(debug_interval, 1) == 0:
                debug_print_retargeting(
                    frame_id,
                    joint_pos,
                    raw_ref_value,
                    ref_value,
                    model_qpos,
                    signed_qpos,
                    retargeting_joint_names,
                    joint_limits,
                    debug_limit_margin,
                    signed_joint_limits=output_joint_limits,
                )
                if x2_palm_flex_guard_enabled:
                    print(
                        "  x2 palm flex guard:",
                        format_x2_palm_flex_guard_changes(
                            unguarded_qpos, qpos, retargeting_joint_names
                        ),
                        flush=True,
                    )
                if is_x2_robot(robot_name):
                    print(
                        "  x2 landmark canonicalization:",
                        format_x2_landmark_canonicalization(x2_landmark_info),
                        flush=True,
                    )
            robot.set_qpos(model_qpos[retargeting_to_sapien])
            frame_id += 1

        if getattr(viewer, "window", None) is None:
            logger.error("SAPIEN viewer window is not available or has been closed.")
            return
        viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if is_realsense_camera(camera_path):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for --camera-path realsense. "
                "Install it with: pip install pyrealsense2"
            ) from exc

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                put_latest_frame(queue, np.asanyarray(color_frame.get_data()).copy())
        finally:
            pipeline.stop()
        return

    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        put_latest_frame(queue, image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    scaling_factor: Optional[float] = None,
    low_pass_alpha: Optional[float] = None,
    normal_delta: Optional[float] = None,
    robot_visual_scale: Optional[float] = None,
    joint_sign: Optional[str] = None,
    joint_gain: Optional[str] = None,
    target_vector_gain: Optional[str] = None,
    finger_map: Optional[str] = None,
    ref_transform: Optional[str] = None,
    debug_retargeting: bool = False,
    debug_interval: int = 15,
    debug_limit_margin: float = 0.03,
    reset_on_hand_lost: bool = True,
    lost_reset_frames: int = 5,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
            Use "realsense" to read the Intel RealSense color stream via pyrealsense2.
        scaling_factor: override the retargeting scaling factor from the config.
        low_pass_alpha: override the low-pass filter alpha from the config.
        normal_delta: override the temporal regularization. Use 0 while debugging sticky poses.
        robot_visual_scale: override the SAPIEN visual scale for the robot model.
        joint_sign: post-retargeting joint sign overrides. Examples:
            "all=-1", "fingers=-1,thumb=1", "rh_FFJ2=-1,rh_MFJ2=-1".
        joint_gain: post-retargeting joint gain overrides. Examples:
            "thumb=1.6", "rh_THJ2=2,rh_THJ1=2".
        target_vector_gain: pre-optimization target vector gain overrides. Examples:
            "thumb=1.5", "thumb_tip=1.6,thumb_ip=1.2".
        finger_map: remap MediaPipe finger landmarks before retargeting.
            Use "reverse" to mirror index/middle/ring/pinky, or
            "swap-index-pinky" to swap only index and pinky.
        ref_transform: transform reference vectors into the robot frame.
            x2 uses axis reorder without mirroring automatically. Use "none" to disable
            or "x2_mirror" to test the old mirrored spread.
        debug_retargeting: print human vectors and robot qpos in the terminal.
        debug_interval: print one debug block every N detected frames.
        debug_limit_margin: print a joint as near-limit if it is within this many radians.
        reset_on_hand_lost: reset optimizer and low-pass state after consecutive missed detections.
        lost_reset_frames: number of consecutive missed detections before reset.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    if config_path is None:
        raise ValueError(
            f"No config for {robot_name.name} {retargeting_type.name} {hand_type.name}."
        )
    robot_dir = get_robot_dir()

    queue = multiprocessing.Queue(maxsize=1)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(
            queue,
            str(robot_dir),
            str(config_path),
            scaling_factor,
            low_pass_alpha,
            normal_delta,
            robot_visual_scale,
            joint_sign,
            joint_gain,
            target_vector_gain,
            finger_map,
            ref_transform,
            debug_retargeting,
            debug_interval,
            debug_limit_margin,
            reset_on_hand_lost,
            lost_reset_frames,
        ),
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)

import csv
import multiprocessing
import os
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Optional


def configure_mujoco_gl_from_cli() -> None:
    """Choose a MuJoCo GL backend before importing mujoco."""
    viewer_backend = "cv2"
    for index, arg in enumerate(sys.argv):
        if arg == "--viewer-backend" and index + 1 < len(sys.argv):
            viewer_backend = sys.argv[index + 1].lower()
            break
        if arg.startswith("--viewer-backend="):
            viewer_backend = arg.split("=", 1)[1].lower()
            break

    if viewer_backend in {"cv2", "opencv", "egl"}:
        # The OpenCV display path renders MuJoCo offscreen, so prefer EGL.
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif viewer_backend in {"glfw", "mujoco"}:
        # Native MuJoCo viewer uses GLFW/GLX. A leftover EGL/OSMesa setting
        # from offscreen rendering can prevent the viewer from creating a window.
        if os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
            os.environ.pop("MUJOCO_GL")


configure_mujoco_gl_from_cli()

import cv2
import glfw
import mujoco
import numpy as np
import tyro
from loguru import logger

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector
from show_realtime_retargeting import (
    apply_joint_output_transform,
    apply_target_vector_gains,
    apply_x2_four_finger_control_sign,
    canonicalize_x2_landmarks_for_control_mode,
    debug_print_retargeting,
    format_x2_four_finger_sign_changes,
    format_x2_landmark_canonicalization,
    format_link_human_mapping,
    format_qpos_by_finger,
    format_target_vector_gains,
    get_x2_four_finger_command_sign,
    get_x2_control_mode,
    get_robot_dir,
    human_landmark_label,
    is_x2_robot,
    parse_joint_gains,
    parse_joint_signs,
    parse_target_vector_gains,
    produce_frame,
    remap_human_indices,
    transform_ref_value,
    X2_LANDMARK_SEMANTIC_MAP,
    X2_LEFT_PALM_FRAME_ADAPTER,
)


X2_ROOT_QUAT = (0.5, -0.5, -0.5, -0.5)
X2_RIGHT_CAMERA_AZIMUTH = 180.0
X2_INTERNAL_QPOS_SMOOTH_ALPHA = 0.65
X2_INTERNAL_MAX_DELTA = 0.12
X2_FOUR_FINGER_QPOS_DEADBAND = 0.025
X2_FOUR_FINGER_DELTA_DEADBAND = 0.01
X2_LEFT_TARGET_VECTOR_Y_AXIS_GAIN = 0.50


def resolve_mujoco_model_path(
    config_urdf_path: str, mujoco_model_path: Optional[Path]
) -> Path:
    if mujoco_model_path is not None:
        return mujoco_model_path

    config_urdf = Path(config_urdf_path)
    if is_x2_robot(config_urdf.stem):
        candidate = config_urdf.with_name("x2_mujoco.urdf")
        if candidate.exists():
            return candidate
        logger.warning(
            f"{candidate} does not exist. Run: "
            "python tools/prepare_x2_mujoco_assets.py"
        )
    return config_urdf


def set_free_root_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_joint_name: str,
    root_pos: tuple[float, float, float],
    root_quat: tuple[float, float, float, float],
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
    if joint_id < 0:
        return
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        logger.warning(f"{root_joint_name} exists but is not a free joint.")
        return

    qpos_addr = model.jnt_qposadr[joint_id]
    quat = np.asarray(root_quat, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(root_pos, dtype=np.float64)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat


def is_close_tuple(values, expected, atol: float = 1e-9) -> bool:
    return np.allclose(
        np.asarray(values, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        atol=atol,
    )


def x2_palm_normal_label(root_quat: tuple[float, float, float, float]) -> str:
    if is_close_tuple(root_quat, X2_ROOT_QUAT):
        return "world +X"
    return "custom"


X2_FOUR_FINGER_DISTAL_BODIES = {
    "FF": "rh_ffdistal",
    "MF": "rh_mfdistal",
    "RF": "rh_rfdistal",
    "LF": "rh_lfdistal",
}
X2_FINGER_TIP_LOCAL_OFFSET = np.asarray([0.034, 0.0, 0.0], dtype=np.float64)
X2_THUMB_JOINT_NAMES = ("rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1")
X2_JOINT_SWEEP_BODIES = {
    "rh_FFJ": "rh_ffdistal",
    "rh_MFJ": "rh_mfdistal",
    "rh_RFJ": "rh_rfdistal",
    "rh_LFJ": "rh_lfdistal",
    "rh_THJ": "rh_thdistal",
}
X2_FOUR_FINGER_PRIMARY_SWEEP_JOINTS = {
    "FF": "rh_FFJ3",
    "MF": "rh_MFJ3",
    "RF": "rh_RFJ3",
    "LF": "rh_LFJ3",
}


def x2_finger_tip_world_position(
    model: mujoco.MjModel, data: mujoco.MjData, distal_body_name: str
) -> Optional[np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, distal_body_name)
    if body_id < 0:
        return None
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ X2_FINGER_TIP_LOCAL_OFFSET


def compute_x2_joint_sweep_rows(
    model: mujoco.MjModel,
    root_joint_name: str,
    root_pos: tuple[float, float, float],
    root_quat: tuple[float, float, float, float],
    joint_names: list[str],
) -> list[dict]:
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
    if root_id < 0:
        return []
    root_addr = model.jnt_qposadr[root_id]
    root_quat_array = np.asarray(root_quat, dtype=np.float64)
    root_quat_array /= np.linalg.norm(root_quat_array)

    def set_root() -> None:
        data.qpos[:] = 0.0
        data.qpos[root_addr : root_addr + 3] = np.asarray(root_pos, dtype=np.float64)
        data.qpos[root_addr + 3 : root_addr + 7] = root_quat_array

    def tip_position(body_name: str) -> Optional[np.ndarray]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return None
        rotation = data.xmat[body_id].reshape(3, 3)
        return data.xpos[body_id] + rotation @ X2_FINGER_TIP_LOCAL_OFFSET

    rows = []
    for mode in ("right_mode", "left_mode"):
        set_root()
        mujoco.mj_forward(model, data)
        neutral = {
            body_name: tip_position(body_name)
            for body_name in set(X2_JOINT_SWEEP_BODIES.values())
        }
        for joint_name in joint_names:
            body_name = next(
                (
                    body
                    for prefix, body in X2_JOINT_SWEEP_BODIES.items()
                    if prefix in joint_name
                ),
                None,
            )
            if body_name is None or neutral.get(body_name) is None:
                continue
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                continue
            qaddr = model.jnt_qposadr[joint_id]
            for q_value in (0.5, -0.5):
                set_root()
                data.qpos[qaddr] = q_value
                mujoco.mj_forward(model, data)
                current = tip_position(body_name)
                if current is None:
                    continue
                delta = current - neutral[body_name]
                rows.append(
                    {
                        "mode": mode,
                        "joint": joint_name,
                        "q": q_value,
                        "tip_body": body_name,
                        "dx": float(delta[0]),
                        "dy": float(delta[1]),
                        "dz": float(delta[2]),
                    }
                )
    return rows


def format_x2_four_finger_joint_sweep(rows: list[dict]) -> str:
    if not rows:
        return "unavailable"
    parts = []
    for mode, q_value in (("right_mode", 0.5), ("left_mode", -0.5)):
        expected = "dx>0" if mode == "right_mode" else "dx<0"
        finger_parts = []
        for finger, joint_name in X2_FOUR_FINGER_PRIMARY_SWEEP_JOINTS.items():
            row = next(
                (
                    item
                    for item in rows
                    if item["mode"] == mode
                    and item["joint"] == joint_name
                    and np.isclose(item["q"], q_value)
                ),
                None,
            )
            if row is not None:
                finger_parts.append(f"{finger}:{joint_name} q={q_value:+.1f} dx={row['dx']:+.4f}")
        parts.append(f"{mode} expected {expected}; " + ", ".join(finger_parts))
    return " | ".join(parts)


def format_x2_thumb_joint_sweep(rows: list[dict]) -> str:
    if not rows:
        return "unavailable"
    parts = []
    for joint_name in X2_THUMB_JOINT_NAMES:
        for q_value in (0.5, -0.5):
            row = next(
                (
                    item
                    for item in rows
                    if item["mode"] == "right_mode"
                    and item["joint"] == joint_name
                    and np.isclose(item["q"], q_value)
                ),
                None,
            )
            if row is None:
                continue
            parts.append(
                f"{joint_name} q={q_value:+.1f}: "
                f"thumb_tip d=({row['dx']:+.4f},{row['dy']:+.4f},{row['dz']:+.4f})"
            )
    return "; ".join(parts) if parts else "unavailable"


def compute_x2_world_finger_tip_dx(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addrs: np.ndarray,
) -> dict[str, float]:
    neutral_data = mujoco.MjData(model)
    neutral_data.qpos[:] = data.qpos
    neutral_data.qpos[qpos_addrs] = 0.0
    mujoco.mj_forward(model, neutral_data)

    deltas = {}
    for finger_name, distal_body_name in X2_FOUR_FINGER_DISTAL_BODIES.items():
        current_tip = x2_finger_tip_world_position(model, data, distal_body_name)
        neutral_tip = x2_finger_tip_world_position(model, neutral_data, distal_body_name)
        if current_tip is None or neutral_tip is None:
            continue
        deltas[finger_name] = float(current_tip[0] - neutral_tip[0])
    return deltas


def x2_finger_motion_amount(
    finger_name: str, joint_names: list[str], qpos: np.ndarray
) -> float:
    token = f"_{finger_name}J"
    return float(
        sum(abs(float(value)) for name, value in zip(joint_names, qpos) if token in name)
    )


def format_x2_world_bend_direction(
    control_mode: str,
    world_dx: dict[str, float],
    joint_names: list[str],
    qpos: np.ndarray,
    tolerance: float = 1e-4,
) -> tuple[str, list[str]]:
    expected_sign = -1.0 if control_mode == "left_mode" else 1.0
    expected_text = (
        "dx<0 in left_mode"
        if control_mode == "left_mode"
        else "dx>0 in right_mode"
    )
    parts = []
    wrong_fingers = []
    for finger_name in X2_FOUR_FINGER_DISTAL_BODIES:
        if finger_name not in world_dx:
            continue
        dx = world_dx[finger_name]
        parts.append(f"{finger_name}:dx={dx:+.4f}")
        if (
            x2_finger_motion_amount(finger_name, joint_names, qpos) > 0.05
            and expected_sign * dx <= tolerance
        ):
            wrong_fingers.append(finger_name)
    return (
        f"mode={control_mode} expected {expected_text}; " + ", ".join(parts),
        wrong_fingers,
    )


def build_qpos_addr_map(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    qpos_addrs = []
    missing = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        qpos_addrs.append(model.jnt_qposadr[joint_id])

    if missing:
        raise ValueError(f"MuJoCo model is missing retargeting joints: {missing}")
    return np.asarray(qpos_addrs, dtype=np.int32)


def reset_retargeting_pose_mujoco(
    retargeting,
    data: mujoco.MjData,
    qpos_addrs: np.ndarray,
    joint_signs: np.ndarray,
    joint_gains: np.ndarray,
    joint_limits: np.ndarray,
    output_joint_limits: Optional[np.ndarray] = None,
    zero_reset: bool = False,
) -> None:
    if output_joint_limits is None:
        output_joint_limits = joint_limits

    retargeting.reset()
    if zero_reset:
        neutral_qpos = np.zeros_like(retargeting.get_qpos())
        neutral_qpos = np.clip(neutral_qpos, joint_limits[:, 0], joint_limits[:, 1])
        retargeting.set_qpos(neutral_qpos)
    else:
        neutral_qpos = retargeting.get_qpos()
    output_qpos = apply_joint_output_transform(
        neutral_qpos, joint_signs, joint_gains, output_joint_limits
    )
    data.qpos[qpos_addrs] = output_qpos


def get_signed_output_joint_limits(
    joint_limits: np.ndarray, joint_signs: np.ndarray
) -> np.ndarray:
    output_limits = joint_limits.copy()
    for index, sign in enumerate(joint_signs):
        if sign < 0:
            lower, upper = output_limits[index]
            output_limits[index] = [-upper, -lower]
    return output_limits


def build_x2_four_finger_deadband_mask(joint_names: list[str]) -> np.ndarray:
    tokens = ("rh_FFJ", "rh_MFJ", "rh_RFJ", "rh_LFJ")
    return np.asarray(
        [any(token in joint_name for token in tokens) for joint_name in joint_names],
        dtype=bool,
    )


class X2CommandSmoother:
    def __init__(
        self,
        alpha: float = X2_INTERNAL_QPOS_SMOOTH_ALPHA,
        max_delta: float = X2_INTERNAL_MAX_DELTA,
        deadband_mask: Optional[np.ndarray] = None,
        qpos_deadband: float = X2_FOUR_FINGER_QPOS_DEADBAND,
        delta_deadband: float = X2_FOUR_FINGER_DELTA_DEADBAND,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_delta = float(max_delta)
        self.deadband_mask = (
            np.asarray(deadband_mask, dtype=bool)
            if deadband_mask is not None
            else None
        )
        self.qpos_deadband = float(max(0.0, qpos_deadband))
        self.delta_deadband = float(max(0.0, delta_deadband))
        self.previous: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous = None

    def update(self, qpos: np.ndarray) -> tuple[np.ndarray, dict]:
        filtered = qpos.copy()
        deadband_mask = None
        if self.deadband_mask is not None and len(self.deadband_mask) == len(filtered):
            deadband_mask = self.deadband_mask
            zero_mask = deadband_mask & (np.abs(filtered) < self.qpos_deadband)
            filtered[zero_mask] = 0.0

        if self.previous is None:
            self.previous = filtered.copy()
            return filtered, {
                "enabled": True,
                "initialized": True,
                "alpha": self.alpha,
                "max_delta": self.max_delta,
                "qpos_deadband": self.qpos_deadband,
                "delta_deadband": self.delta_deadband,
                "dq_max_abs": 0.0,
                "clipped": False,
                "deadband_joints": int(np.sum(deadband_mask)) if deadband_mask is not None else 0,
            }

        if deadband_mask is not None and self.delta_deadband > 0:
            hold_mask = deadband_mask & (
                np.abs(filtered - self.previous) < self.delta_deadband
            )
            filtered[hold_mask] = self.previous[hold_mask]

        blended = self.alpha * filtered + (1.0 - self.alpha) * self.previous
        delta = blended - self.previous
        if self.max_delta > 0:
            clipped_delta = np.clip(delta, -self.max_delta, self.max_delta)
        else:
            clipped_delta = delta
        smoothed = self.previous + clipped_delta
        if deadband_mask is not None:
            final_zero_mask = (
                deadband_mask
                & (np.abs(filtered) < self.qpos_deadband)
                & (np.abs(smoothed) < self.qpos_deadband)
            )
            smoothed[final_zero_mask] = 0.0
            clipped_delta = smoothed - self.previous
        self.previous = smoothed.copy()
        return smoothed, {
            "enabled": True,
            "initialized": False,
            "alpha": self.alpha,
            "max_delta": self.max_delta,
            "qpos_deadband": self.qpos_deadband,
            "delta_deadband": self.delta_deadband,
            "dq_max_abs": float(np.max(np.abs(clipped_delta))) if len(clipped_delta) else 0.0,
            "clipped": bool(not np.allclose(delta, clipped_delta)),
            "deadband_joints": int(np.sum(deadband_mask)) if deadband_mask is not None else 0,
        }


def format_x2_command_smoothing(info: dict) -> str:
    if not info.get("enabled"):
        return "off"
    return (
        f"alpha={info.get('alpha', 0.0):.2f},"
        f"max_delta={info.get('max_delta', 0.0):.3f},"
        f"deadband={info.get('qpos_deadband', 0.0):.3f}/"
        f"{info.get('delta_deadband', 0.0):.3f},"
        f"dq_max={info.get('dq_max_abs', 0.0):.3f},"
        f"clipped={info.get('clipped', False)}"
    )


class X2DebugRecorder:
    LANDMARK_EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    )

    def __init__(
        self,
        debug_dir: Optional[Path],
        joint_names: list[str],
        joint_limits: np.ndarray,
        save_png: bool = False,
    ) -> None:
        self.enabled = debug_dir is not None
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.joint_names = joint_names
        self.joint_limits = joint_limits
        self.save_png = save_png
        self.previous_qpos: Optional[np.ndarray] = None
        if self.enabled:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "landmarks").mkdir(exist_ok=True)
            (self.debug_dir / "vectors").mkdir(exist_ok=True)

    def _append_csv(self, filename: str, fieldnames: list[str], rows: list[dict]) -> None:
        if not self.enabled or not rows:
            return
        path = self.debug_dir / filename
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_points_csv(self, frame_id: int, stage: str, points: Optional[np.ndarray]) -> None:
        if not self.enabled or points is None:
            return
        rows = [
            {"index": i, "x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
            for i, point in enumerate(points)
        ]
        path = self.debug_dir / "landmarks" / f"frame_{frame_id:06d}_{stage}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_points_png(self, frame_id: int, stage: str, points: Optional[np.ndarray]) -> None:
        if not self.enabled or not self.save_png or points is None:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Skip x2 debug landmark PNG: {exc}")
            return

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=14)
        for start, end in self.LANDMARK_EDGES:
            segment = points[[start, end]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], linewidth=1)
        ax.set_title(stage)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        max_range = float(np.max(np.ptp(points, axis=0))) if len(points) else 1.0
        center = np.mean(points, axis=0)
        radius = max(max_range * 0.55, 1e-3)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        fig.tight_layout()
        fig.savefig(self.debug_dir / "landmarks" / f"frame_{frame_id:06d}_{stage}.png")
        plt.close(fig)

    def _write_vectors_csv(
        self,
        frame_id: int,
        raw_ref_value: np.ndarray,
        retarget_ref_value: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        rows = []
        for index, (raw, target) in enumerate(zip(raw_ref_value, retarget_ref_value)):
            rows.append(
                {
                    "index": index,
                    "raw_x": float(raw[0]),
                    "raw_y": float(raw[1]),
                    "raw_z": float(raw[2]),
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "target_z": float(target[2]),
                    "target_norm": float(np.linalg.norm(target)),
                }
            )
        path = self.debug_dir / "vectors" / f"frame_{frame_id:06d}_target_vectors.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "raw_x",
                    "raw_y",
                    "raw_z",
                    "target_x",
                    "target_y",
                    "target_z",
                    "target_norm",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_vectors_png(self, frame_id: int, retarget_ref_value: np.ndarray) -> None:
        if not self.enabled or not self.save_png:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
        origin = np.zeros(3)
        for vector in retarget_ref_value:
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                vector[0],
                vector[1],
                vector[2],
                length=1.0,
                normalize=False,
            )
        ax.set_title("target vectors")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.tight_layout()
        fig.savefig(self.debug_dir / "vectors" / f"frame_{frame_id:06d}_target_vectors.png")
        plt.close(fig)

    def record_frame(
        self,
        frame_id: int,
        handedness: str,
        control_mode: str,
        detection_meta: dict,
        mode_points: Optional[np.ndarray],
        raw_ref_value: np.ndarray,
        retarget_ref_value: np.ndarray,
        qpos: np.ndarray,
        smoothing_info: dict,
        world_dx: dict[str, float],
    ) -> None:
        if not self.enabled:
            return

        self._write_points_csv(frame_id, "raw", detection_meta.get("raw_landmarks"))
        self._write_points_csv(
            frame_id, "palm_frame", detection_meta.get("palm_frame_landmarks")
        )
        self._write_points_csv(frame_id, "mode_transformed", mode_points)
        self._write_points_png(frame_id, "raw", detection_meta.get("raw_landmarks"))
        self._write_points_png(
            frame_id, "palm_frame", detection_meta.get("palm_frame_landmarks")
        )
        self._write_points_png(frame_id, "mode_transformed", mode_points)
        self._write_vectors_csv(frame_id, raw_ref_value, retarget_ref_value)
        self._write_vectors_png(frame_id, retarget_ref_value)

        qpos_rows = []
        for index, (name, value, limits) in enumerate(
            zip(self.joint_names, qpos, self.joint_limits)
        ):
            previous = 0.0 if self.previous_qpos is None else float(self.previous_qpos[index])
            lower, upper = limits
            near_limit = min(float(value) - lower, upper - float(value)) <= 0.03
            qpos_rows.append(
                {
                    "frame": frame_id,
                    "handedness": handedness,
                    "mode": control_mode,
                    "joint": name,
                    "qpos": float(value),
                    "dq": float(value) - previous,
                    "near_limit": near_limit,
                    "lower": float(lower),
                    "upper": float(upper),
                    "smooth_alpha": smoothing_info.get("alpha", 0.0),
                    "smooth_max_delta": smoothing_info.get("max_delta", 0.0),
                }
            )
        self.previous_qpos = qpos.copy()
        self._append_csv(
            "qpos.csv",
            [
                "frame",
                "handedness",
                "mode",
                "joint",
                "qpos",
                "dq",
                "near_limit",
                "lower",
                "upper",
                "smooth_alpha",
                "smooth_max_delta",
            ],
            qpos_rows,
        )

        self._append_csv(
            "fingertip_world_dx.csv",
            ["frame", "mode", "finger", "dx"],
            [
                {"frame": frame_id, "mode": control_mode, "finger": finger, "dx": dx}
                for finger, dx in world_dx.items()
            ],
        )

    def write_joint_sweep(
        self,
        model: mujoco.MjModel,
        root_joint_name: str,
        root_pos: tuple[float, float, float],
        root_quat: tuple[float, float, float, float],
        rows: Optional[list[dict]] = None,
    ) -> None:
        if not self.enabled:
            return
        path = self.debug_dir / "joint_sweep.csv"
        if rows is None:
            rows = compute_x2_joint_sweep_rows(
                model,
                root_joint_name,
                root_pos,
                root_quat,
                self.joint_names,
            )
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["mode", "joint", "q", "tip_body", "dx", "dy", "dz"]
            )
            writer.writeheader()
            writer.writerows(rows)


def produce_frame_with_error_report(
    queue: multiprocessing.Queue,
    error_queue: multiprocessing.Queue,
    camera_path: Optional[str],
):
    try:
        produce_frame(queue, camera_path)
    except Exception as exc:
        error_queue.put(
            f"{type(exc).__name__}: {exc}. "
            "If this is a RealSense camera, close realsense-viewer/other scripts "
            "or unplug/replug the camera."
        )
        raise


def check_camera_process_startup(
    producer: multiprocessing.Process,
    error_queue: multiprocessing.Queue,
    timeout: float = 1.5,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not error_queue.empty():
            raise RuntimeError(f"Camera producer failed: {error_queue.get()}")
        if not producer.is_alive():
            exit_code = producer.exitcode
            if not error_queue.empty():
                raise RuntimeError(f"Camera producer failed: {error_queue.get()}")
            raise RuntimeError(f"Camera producer exited early with code {exit_code}.")
        time.sleep(0.05)


def check_glfw_context_available() -> None:
    if not glfw.init():
        raise RuntimeError(
            "GLFW init failed. Check DISPLAY/Wayland/Xorg and NVIDIA OpenGL setup."
        )
    window = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(16, 16, "mujoco_glfw_preflight", None, None)
        if window is None:
            raise RuntimeError(
                "GLFW could not create an OpenGL window. "
                "This usually means GLX is unavailable in the current terminal/session."
            )
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def set_glfw_context_api(context_api: str) -> None:
    context_api = context_api.lower()
    if context_api in {"native", "default"}:
        return

    api_map = {
        "egl": glfw.EGL_CONTEXT_API,
        "osmesa": glfw.OSMESA_CONTEXT_API,
    }
    if context_api not in api_map:
        raise ValueError(
            f"Unknown glfw_context_api '{context_api}'. "
            "Use native, egl, or osmesa."
        )
    if not glfw.init():
        raise RuntimeError("GLFW init failed before setting context API.")
    glfw.window_hint(glfw.CONTEXT_CREATION_API, api_map[context_api])
    logger.warning(
        f"MuJoCo viewer will ask GLFW to create a {context_api.upper()} context. "
        "This is still the native MuJoCo viewer; it only avoids GLX context creation."
    )


def launch_mujoco_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    glfw_context_api: str,
):
    try:
        set_glfw_context_api(glfw_context_api)
        import mujoco.viewer

        return mujoco.viewer.launch_passive(model, data)
    except Exception as exc:
        logger.error(
            "Failed to create the MuJoCo GLFW viewer window. "
            "This is an OpenGL/GLX desktop context issue, not a retargeting error."
        )
        logger.error(
            "Display env: "
            f"DISPLAY={os.environ.get('DISPLAY')!r}, "
            f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}, "
            f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}, "
            f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')!r}"
        )
        logger.error(
            "Try from a local desktop terminal: "
            "unset MUJOCO_GL PYOPENGL_PLATFORM; "
            "unset __GLX_VENDOR_LIBRARY_NAME; "
            "python example/vector_retargeting/show_realtime_retargeting_mujoco.py ..."
        )
        raise exc


X2_NATURAL_VECTOR_GAIN = (
    "rh_thtip-rh_fftip=0.35,"
    "rh_thtip-rh_mftip=0.25,"
    "rh_lfmiddle-rh_lfproximal=0.75,"
    "rh_lfdistal-rh_lfmiddle=0.75,"
    "rh_lftip-rh_lfdistal=0.80,"
    "rh_lftip-rh_lfmiddle=0.65,"
    "rh_mfmiddle-rh_mfproximal=0.75,"
    "rh_mfdistal-rh_mfmiddle=0.75,"
    "rh_mftip-rh_mfdistal=0.80,"
    "rh_mftip-rh_mfmiddle=0.65,"
    "rh_rfmiddle-rh_rfproximal=0.75,"
    "rh_rfdistal-rh_rfmiddle=0.75,"
    "rh_rftip-rh_rfdistal=0.80,"
    "rh_rftip-rh_rfmiddle=0.65,"
    "rh_ffmiddle-rh_ffproximal=0.75,"
    "rh_ffdistal-rh_ffmiddle=0.75,"
    "rh_fftip-rh_ffdistal=0.80,"
    "rh_fftip-rh_ffmiddle=0.65"
)

X2_NATURAL_AXIS_GAIN = (
    "rh_thtip-rh_palm:y=0.55,"
    "rh_fftip-rh_palm:y=0.32,"
    "rh_ffmiddle-rh_palm:y=0.28,"
    "rh_mftip-rh_palm:y=0.31,"
    "rh_mfmiddle-rh_palm:y=0.27,"
    "rh_rftip-rh_palm:y=0.30,"
    "rh_rfmiddle-rh_palm:y=0.26,"
    "rh_lftip-rh_palm:y=0.28,"
    "rh_lfmiddle-rh_palm:y=0.24,"
    "rh_fftip-rh_palm:z=0.88,"
    "rh_ffmiddle-rh_palm:z=0.84,"
    "rh_mftip-rh_palm:z=0.88,"
    "rh_mfmiddle-rh_palm:z=0.84,"
    "rh_rftip-rh_palm:z=0.84,"
    "rh_rfmiddle-rh_palm:z=0.80,"
    "rh_lftip-rh_palm:z=0.78,"
    "rh_lfmiddle-rh_palm:z=0.74"
)

X2_NATURAL_JOINT_LIMITS = [
    "rh_FFJ3=-1.20:1.20",
    "rh_MFJ3=-1.20:1.20",
    "rh_RFJ3=-1.20:1.20",
    "rh_LFJ3=-1.20:1.20",
    "rh_THJ4=-1.74:1.65",
    "rh_THJ3=-1.31:1.31",
    "rh_THJ2=-1.31:1.31",
    "rh_THJ1=-1.31:1.31",
]

X2_PINCH_ACTIVATION_NEAR = 0.055
X2_PINCH_ACTIVATION_FAR = 0.12
X2_FINGER_STRAIGHT_CLOSED_RATIO = 1.08
X2_FINGER_STRAIGHT_OPEN_RATIO = 1.22
X2_THUMB_DIRECT_JOINTS = ("rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1")
X2_THUMB_DIRECT_BASELINE_FRAMES = 12
X2_THUMB_DIRECT_DEADZONE = {
    "rh_THJ4": 0.0,
    "rh_THJ3": 0.35,
    "rh_THJ2": 1.2,
    "rh_THJ1": 1.0,
}
X2_THUMB_DIRECT_GAIN = {
    "rh_THJ4": 1.0,
    "rh_THJ3": np.deg2rad(0.65),
    "rh_THJ2": np.deg2rad(1.15),
    "rh_THJ1": np.deg2rad(1.05),
}
X2_THUMB_DIRECT_BLEND = {
    "rh_THJ4": 1.0,
    "rh_THJ3": 1.0,
    "rh_THJ2": 1.0,
    "rh_THJ1": 1.0,
}
X2_THUMB_CONTROL_EMA_ALPHA = 0.65
X2_THUMB_FEATURE_EMA_ALPHA = 0.60
X2_THUMB_DIRECT_MIN_AMOUNT = {
    "rh_THJ4": 0.045,
    "rh_THJ3": 0.25,
    "rh_THJ2": 0.35,
    "rh_THJ1": 0.35,
}
X2_THUMB_DIRECT_MAX_MAGNITUDE = {
    "rh_THJ4": 1.10,
    "rh_THJ3": 0.25,
    "rh_THJ2": 0.60,
    "rh_THJ1": 0.45,
}
X2_THUMB_COORDINATION_PROGRESS_START = 0.45
X2_THUMB_COORDINATION_PROGRESS_END = 0.85
X2_THUMB_COORDINATION_EXTRA = {
    "rh_THJ3": 0.10,
    "rh_THJ2": 0.16,
    "rh_THJ1": 0.08,
}
X2_THJ4_SOFT_MAX_LOW_FLEXION = 0.90
X2_THJ4_SOFT_FLEXION_LOW_DEG = 10.0
X2_THJ4_SOFT_FLEXION_HIGH_DEG = 18.0
X2_THJ4_SOFT_PROGRESS_START = 0.60
X2_THJ4_SOFT_PROGRESS_END = 0.95
X2_THUMB_ROOT_PALM_PROJECTION_SCALE = 0.45
X2_FINGER_LANDMARKS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
X2_FINGER_DIRECT_LANDMARKS = {
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
X2_FINGER_DIRECT_JOINTS = {
    "index": ("rh_FFJ3", "rh_FFJ2", "rh_FFJ1"),
    "middle": ("rh_MFJ3", "rh_MFJ2", "rh_MFJ1"),
    "ring": ("rh_RFJ3", "rh_RFJ2", "rh_RFJ1"),
    "pinky": ("rh_LFJ3", "rh_LFJ2", "rh_LFJ1"),
}
X2_FINGER_DIRECT_BASELINE_FRAMES = 12
X2_FINGER_DIRECT_DEADZONE_DEG = {
    "mcp": 3.0,
    "pip": 4.0,
    "dip": 4.0,
}
X2_FINGER_CURL_INPUT_WEIGHT = {
    "mcp": 0.65,
    "pip": 0.85,
    "dip": 0.55,
}
X2_FINGER_CURL_STAGE_POINTS = (
    (0.35, {"mcp": 0.72, "pip": 0.24, "dip": 0.06}),
    (1.05, {"mcp": 0.52, "pip": 0.72, "dip": 0.34}),
    (1.65, {"mcp": 0.58, "pip": 0.66, "dip": 0.52}),
)
X2_FINGER_CURL_GAIN_RAD_PER_DEG = np.deg2rad(1.05)
X2_FINGER_NEAR_LIMIT_MARGIN = 0.12
X2_FINGER_DIRECT_QPOS_HYSTERESIS = 0.012
X2_FINGER_ANGLE_EMA_ALPHA = 0.60
X2_FINGER_COUPLING_PIP_PER_MCP = 0.8
X2_FINGER_COUPLING_DIP_PER_PIP = 0.5
X2_FINGER_MODE_GAIN = {
    "right_mode": {"mcp": 1.00, "pip": 1.05, "dip": 1.10},
    "left_mode": {"mcp": 1.10, "pip": 1.30, "dip": 1.35},
}
X2_THUMB_DIRECT_MIN_RELATIVE_IMPROVEMENT = 0.15
X2_THUMB_DIRECT_MIN_ABSOLUTE_IMPROVEMENT = 0.02
X2_THJ4_SIDE_DEADZONE = 0.010
X2_THJ4_PALM_DEADZONE = 0.010
X2_THJ4_OPPOSITION_GATE = 0.040
X2_THJ4_SIDE_HIGH = 0.120
X2_THJ4_PALM_HIGH = 0.120
X2_THJ4_BASE_SIDE_WEIGHT = 0.70
X2_THJ4_TIP_SIDE_WEIGHT = 0.30
X2_THJ4_PALM_WEIGHT = 0.55
X2_THJ4_PINCH_AUX_WEIGHT = 0.04
X2_THJ4_EARLY_PROGRESS = 0.300
X2_THJ4_EARLY_AMOUNT = 0.450
X2_THJ4_AMOUNT_BASE_WEIGHT = 0.80
X2_THJ4_AMOUNT_PALM_WEIGHT = 0.15
X2_THJ4_AMOUNT_TIP_WEIGHT = 0.05


def parse_x2_finger_direct_gain(gain_spec: Optional[str]) -> dict[str, float]:
    gains = {"mcp": 1.0, "pip": 1.0, "dip": 1.0}
    if gain_spec is None or gain_spec.lower() in {"", "default", "none"}:
        return gains

    aliases = {
        "all": tuple(gains),
        "finger": tuple(gains),
        "fingers": tuple(gains),
        "j3": ("mcp",),
        "mcp": ("mcp",),
        "j2": ("pip",),
        "pip": ("pip",),
        "j1": ("dip",),
        "dip": ("dip",),
    }
    for raw_item in gain_spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            key, value = "all", item
        key = key.strip().lower()
        if key not in aliases:
            raise ValueError(
                f"Unknown x2 finger direct gain target '{key}'. "
                "Use all, mcp/j3, pip/j2, or dip/j1."
            )
        gain = float(value.strip())
        if gain < 0:
            raise ValueError(f"x2 finger direct gain for {key} must be non-negative.")
        for joint_key in aliases[key]:
            gains[joint_key] = gain
    return gains


def format_x2_finger_direct_gain(gains: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in gains.items())


def get_x2_finger_mode_gain(control_mode: str) -> dict[str, float]:
    return X2_FINGER_MODE_GAIN.get(
        control_mode,
        {"mcp": 1.0, "pip": 1.0, "dip": 1.0},
    ).copy()


X2_PALM_VECTOR_MIN_X = {
    "rh_thtip-rh_palm": 0.082,
    "rh_fftip-rh_palm": 0.092,
    "rh_mftip-rh_palm": 0.096,
    "rh_rftip-rh_palm": 0.092,
    "rh_lftip-rh_palm": 0.086,
    "rh_ffmiddle-rh_palm": 0.105,
    "rh_mfmiddle-rh_palm": 0.108,
    "rh_rfmiddle-rh_palm": 0.105,
    "rh_lfmiddle-rh_palm": 0.100,
}

X2_PALM_VECTOR_MAX_X = {
    "rh_thtip-rh_palm": 0.115,
    "rh_fftip-rh_palm": 0.138,
    "rh_mftip-rh_palm": 0.142,
    "rh_rftip-rh_palm": 0.140,
    "rh_lftip-rh_palm": 0.135,
    "rh_ffmiddle-rh_palm": 0.120,
    "rh_mfmiddle-rh_palm": 0.122,
    "rh_rfmiddle-rh_palm": 0.120,
    "rh_lfmiddle-rh_palm": 0.116,
}

X2_PALM_VECTOR_MAX_Z = {
    "rh_thtip-rh_palm": 0.045,
}


def is_x2_profile_enabled(profile: Optional[str]) -> bool:
    if profile is None:
        return True
    return profile.lower() not in {"", "none", "off", "false", "0", "default"}


def merge_csv_overrides(defaults: str, overrides: Optional[str]) -> Optional[str]:
    if not defaults:
        return overrides
    if overrides is None or overrides.lower() in {"", "none", "default"}:
        return defaults
    return defaults + "," + overrides


def add_x2_local_shape_constraints(config: RetargetingConfig) -> list[str]:
    if config.type != "vector":
        return []
    if config.target_origin_link_names is None or config.target_task_link_names is None:
        return []
    if config.target_link_human_indices is None:
        return []

    local_specs = [
        ("rh_fftip", "rh_thtip", 8, 4),
        ("rh_mftip", "rh_thtip", 12, 4),
        ("rh_ffproximal", "rh_ffmiddle", 5, 6),
        ("rh_ffmiddle", "rh_ffdistal", 6, 7),
        ("rh_ffdistal", "rh_fftip", 7, 8),
        ("rh_ffmiddle", "rh_fftip", 6, 8),
        ("rh_mfproximal", "rh_mfmiddle", 9, 10),
        ("rh_mfmiddle", "rh_mfdistal", 10, 11),
        ("rh_mfdistal", "rh_mftip", 11, 12),
        ("rh_mfmiddle", "rh_mftip", 10, 12),
        ("rh_rfproximal", "rh_rfmiddle", 13, 14),
        ("rh_rfmiddle", "rh_rfdistal", 14, 15),
        ("rh_rfdistal", "rh_rftip", 15, 16),
        ("rh_rfmiddle", "rh_rftip", 14, 16),
        ("rh_lfproximal", "rh_lfmiddle", 17, 18),
        ("rh_lfmiddle", "rh_lfdistal", 18, 19),
        ("rh_lfdistal", "rh_lftip", 19, 20),
        ("rh_lfmiddle", "rh_lftip", 18, 20),
    ]
    existing = set(zip(config.target_origin_link_names, config.target_task_link_names))
    new_origin_links = []
    new_task_links = []
    new_human_origin = []
    new_human_task = []
    added = []
    for origin_link, task_link, human_origin, human_task in local_specs:
        if (origin_link, task_link) in existing:
            continue
        new_origin_links.append(origin_link)
        new_task_links.append(task_link)
        new_human_origin.append(human_origin)
        new_human_task.append(human_task)
        existing.add((origin_link, task_link))
        added.append(f"{task_link}-{origin_link}")

    if not new_origin_links:
        return []

    config.target_origin_link_names = config.target_origin_link_names + new_origin_links
    config.target_task_link_names = config.target_task_link_names + new_task_links
    local_indices = np.asarray([new_human_origin, new_human_task], dtype=np.int64)
    config.target_link_human_indices = np.concatenate(
        [config.target_link_human_indices, local_indices], axis=1
    )
    return added


def remove_x2_thumb_hard_match_objectives(config: RetargetingConfig) -> list[str]:
    if config.type != "vector":
        return []
    if config.target_origin_link_names is None or config.target_task_link_names is None:
        return []
    if config.target_link_human_indices is None:
        return []

    removed_pairs = {
        ("rh_palm", "rh_thbase"),
        ("rh_thbase", "rh_thproximal"),
        ("rh_thproximal", "rh_thmiddle"),
        ("rh_thmiddle", "rh_thtip"),
        ("rh_palm", "rh_thproximal"),
        ("rh_palm", "rh_thmiddle"),
    }
    removed = []
    keep = []
    for origin_link, task_link in zip(
        config.target_origin_link_names, config.target_task_link_names
    ):
        should_remove = (origin_link, task_link) in removed_pairs
        keep.append(not should_remove)
        if should_remove:
            removed.append(f"{task_link}-{origin_link}")

    if not removed:
        return []

    indices = np.asarray(config.target_link_human_indices)
    config.target_origin_link_names = [
        name for name, should_keep in zip(config.target_origin_link_names, keep) if should_keep
    ]
    config.target_task_link_names = [
        name for name, should_keep in zip(config.target_task_link_names, keep) if should_keep
    ]
    config.target_link_human_indices = indices[:, keep]
    return removed


def apply_x2_natural_ref_shape(
    ref_value: np.ndarray,
    retargeting,
    profile: Optional[str],
) -> np.ndarray:
    if not is_x2_profile_enabled(profile):
        return ref_value
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type != "VECTOR":
        return ref_value

    shaped = ref_value.copy()
    for i, (origin_link, task_link) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names)
    ):
        key = f"{task_link}-{origin_link}".lower()
        min_x = X2_PALM_VECTOR_MIN_X.get(key)
        if min_x is not None and shaped[i, 0] > 0:
            shaped[i, 0] = max(shaped[i, 0], min_x)
        max_x = X2_PALM_VECTOR_MAX_X.get(key)
        if max_x is not None and shaped[i, 0] > 0:
            shaped[i, 0] = min(shaped[i, 0], max_x)
        max_z = X2_PALM_VECTOR_MAX_Z.get(key)
        if max_z is not None and shaped[i, 2] > 0:
            shaped[i, 2] = min(shaped[i, 2], max_z)
    return shaped


def ratio_to_unit_interval(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def compute_x2_finger_straightness(joint_pos: Optional[np.ndarray]) -> dict[str, dict]:
    if joint_pos is None or len(joint_pos) <= 20:
        return {}

    info = {}
    wrist = joint_pos[0]
    for finger, (middle_index, tip_index) in X2_FINGER_LANDMARKS.items():
        middle_norm = float(np.linalg.norm(joint_pos[middle_index] - wrist))
        tip_norm = float(np.linalg.norm(joint_pos[tip_index] - wrist))
        ratio = tip_norm / max(middle_norm, 1e-6)
        straightness = ratio_to_unit_interval(
            ratio,
            X2_FINGER_STRAIGHT_CLOSED_RATIO,
            X2_FINGER_STRAIGHT_OPEN_RATIO,
        )
        info[finger] = {
            "ratio": ratio,
            "straightness": straightness,
            "curl_gate": 1.0 - straightness,
        }
    return info


def compute_x2_pinch_gate_info(
    joint_pos: Optional[np.ndarray], pinch_finger: str
) -> dict[str, dict]:
    finger_info = compute_x2_finger_straightness(joint_pos)
    return {
        finger: finger_info.get(finger, {"ratio": 0.0, "curl_gate": 1.0})
        for finger in parse_pinch_fingers(pinch_finger)
    }


def format_x2_pinch_gate_info(gate_info: dict[str, dict]) -> str:
    if not gate_info:
        return "default"
    return ", ".join(
        f"{finger}:ratio={values.get('ratio', 0.0):.2f},"
        f"gate={values.get('curl_gate', 1.0):.2f}"
        for finger, values in gate_info.items()
    )


def compute_x2_finger_raw_angles(joint_pos: np.ndarray) -> dict[str, dict[str, float]]:
    angles = {}
    for finger, (wrist, mcp, pip, dip, tip) in X2_FINGER_DIRECT_LANDMARKS.items():
        angles[finger] = {
            "mcp": thumb_flexion_angle_deg(joint_pos, wrist, mcp, pip),
            "pip": thumb_flexion_angle_deg(joint_pos, mcp, pip, dip),
            "dip": thumb_flexion_angle_deg(joint_pos, pip, dip, tip),
        }
    return angles


def x2_signed_joint_limit_magnitude(
    joint_limits: np.ndarray,
    joint_index: int,
    sign: float,
) -> float:
    lower, upper = joint_limits[joint_index]
    if sign >= 0:
        return max(0.0, float(upper))
    return max(0.0, float(-lower))


def interpolate_x2_finger_curl_distribution(curl: float) -> dict[str, float]:
    if curl <= 0.0:
        return {"mcp": 0.0, "pip": 0.0, "dip": 0.0}

    previous_limit, previous_distribution = X2_FINGER_CURL_STAGE_POINTS[0]
    if curl <= previous_limit:
        return previous_distribution.copy()

    for limit, distribution in X2_FINGER_CURL_STAGE_POINTS[1:]:
        if curl <= limit:
            ratio = (curl - previous_limit) / max(limit - previous_limit, 1e-6)
            return {
                key: (1.0 - ratio) * previous_distribution[key]
                + ratio * distribution[key]
                for key in ("mcp", "pip", "dip")
            }
        previous_limit = limit
        previous_distribution = distribution
    return previous_distribution.copy()


def add_x2_curl_excess_with_capacity(
    values: dict[str, float],
    caps: dict[str, float],
    excess: float,
    preferred_keys: tuple[str, ...],
) -> float:
    remaining = float(max(0.0, excess))
    for _ in range(2):
        available = [
            key for key in preferred_keys if caps.get(key, 0.0) - values.get(key, 0.0) > 1e-6
        ]
        if remaining <= 1e-6 or not available:
            break
        total_capacity = sum(caps[key] - values[key] for key in available)
        if total_capacity <= 1e-6:
            break
        used = 0.0
        for key in available:
            capacity = caps[key] - values[key]
            portion = remaining * capacity / total_capacity
            addition = min(capacity, portion)
            values[key] += addition
            used += addition
        remaining -= used
    return remaining


def redistribute_x2_finger_curl_near_limits(
    magnitudes: dict[str, float],
    joint_indices: dict[str, int],
    joint_limits: np.ndarray,
    sign: float,
) -> dict[str, float]:
    redistributed = magnitudes.copy()
    caps = {}
    for joint_key, joint_index in joint_indices.items():
        limit = x2_signed_joint_limit_magnitude(joint_limits, joint_index, sign)
        caps[joint_key] = max(0.0, limit - X2_FINGER_NEAR_LIMIT_MARGIN)

    if not caps:
        return redistributed

    pip_cap = caps.get("pip")
    if pip_cap is not None and redistributed["pip"] > pip_cap:
        excess = redistributed["pip"] - pip_cap
        redistributed["pip"] = pip_cap
        add_x2_curl_excess_with_capacity(
            redistributed, caps, excess, ("mcp", "dip")
        )

    for joint_key in ("mcp", "dip", "pip"):
        cap = caps.get(joint_key)
        if cap is None or redistributed[joint_key] <= cap:
            continue
        excess = redistributed[joint_key] - cap
        redistributed[joint_key] = cap
        preferred = tuple(key for key in ("mcp", "pip", "dip") if key != joint_key)
        add_x2_curl_excess_with_capacity(redistributed, caps, excess, preferred)

    for joint_key, cap in caps.items():
        redistributed[joint_key] = min(redistributed[joint_key], cap)
    return redistributed


class X2FingerDirectMappingState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.baseline: Optional[dict[str, dict[str, float]]] = None
        self.filtered_angles: Optional[dict[str, dict[str, float]]] = None
        self.previous_direct_qpos: dict[str, float] = {}
        self.samples = 0

    def update(
        self,
        joint_pos: Optional[np.ndarray],
        profile: Optional[str],
        joint_names: list[str],
        joint_limits: np.ndarray,
        control_mode: str,
        gain_scale: dict[str, float],
    ) -> dict:
        if (
            not is_x2_profile_enabled(profile)
            or joint_pos is None
            or len(joint_pos) <= 20
        ):
            return {"enabled": False}

        raw_angles = compute_x2_finger_raw_angles(joint_pos)
        if self.filtered_angles is None:
            self.filtered_angles = {
                finger: values.copy() for finger, values in raw_angles.items()
            }
        else:
            alpha = X2_FINGER_ANGLE_EMA_ALPHA
            for finger, values in raw_angles.items():
                for joint_key, value in values.items():
                    previous = self.filtered_angles[finger][joint_key]
                    self.filtered_angles[finger][joint_key] = (
                        alpha * value + (1.0 - alpha) * previous
                    )
        angles = {
            finger: values.copy() for finger, values in self.filtered_angles.items()
        }
        if self.baseline is None:
            self.baseline = {
                finger: values.copy() for finger, values in angles.items()
            }
            self.samples = 1
        elif self.samples < X2_FINGER_DIRECT_BASELINE_FRAMES:
            for finger, values in angles.items():
                for joint_key, value in values.items():
                    self.baseline[finger][joint_key] = min(
                        self.baseline[finger][joint_key], value
                    )
            self.samples += 1

        calibrated = self.samples >= X2_FINGER_DIRECT_BASELINE_FRAMES
        direct_qpos = {}
        flexion_deg = {}
        curl_rad = {}
        curl_rad_after_mode_gain = {}
        redistribution = {}
        finger_qpos_before_mode_gain = {}
        finger_qpos_after_mode_gain = {}
        clipped = {}
        sign = get_x2_four_finger_command_sign(control_mode)
        mode_gain = get_x2_finger_mode_gain(control_mode)
        for finger, values in angles.items():
            flexion_deg[finger] = {}
            mcp_joint, pip_joint, dip_joint = X2_FINGER_DIRECT_JOINTS[finger]
            joint_lookup = {
                "mcp": mcp_joint,
                "pip": pip_joint,
                "dip": dip_joint,
            }
            for joint_key, joint_name in [
                ("mcp", mcp_joint),
                ("pip", pip_joint),
                ("dip", dip_joint),
            ]:
                baseline = self.baseline[finger][joint_key]
                deadzone = X2_FINGER_DIRECT_DEADZONE_DEG[joint_key]
                flexion = max(0.0, values[joint_key] - baseline - deadzone)
                flexion_deg[finger][joint_key] = flexion

            curl_input_deg = sum(
                X2_FINGER_CURL_INPUT_WEIGHT[joint_key]
                * flexion_deg[finger][joint_key]
                for joint_key in ("mcp", "pip", "dip")
            )
            curl = X2_FINGER_CURL_GAIN_RAD_PER_DEG * curl_input_deg
            curl_rad[finger] = curl
            curl_distribution = interpolate_x2_finger_curl_distribution(curl)
            magnitudes_before_mode_gain = {
                joint_key: curl_distribution[joint_key]
                * curl
                * gain_scale.get(joint_key, 1.0)
                for joint_key in ("mcp", "pip", "dip")
            }
            magnitudes = {
                joint_key: magnitudes_before_mode_gain[joint_key]
                * mode_gain.get(joint_key, 1.0)
                for joint_key in ("mcp", "pip", "dip")
            }
            distribution_sum = max(
                sum(curl_distribution[joint_key] for joint_key in ("mcp", "pip", "dip")),
                1e-6,
            )
            curl_rad_after_mode_gain[finger] = sum(magnitudes.values()) / distribution_sum
            finger_qpos_before_mode_gain[finger] = {
                joint_lookup[joint_key]: sign * magnitudes_before_mode_gain[joint_key]
                for joint_key in ("mcp", "pip", "dip")
            }
            finger_qpos_after_mode_gain[finger] = {
                joint_lookup[joint_key]: sign * magnitudes[joint_key]
                for joint_key in ("mcp", "pip", "dip")
            }
            joint_indices = {
                joint_key: joint_names.index(joint_name)
                for joint_key, joint_name in joint_lookup.items()
                if joint_name in joint_names
            }
            redistributed = redistribute_x2_finger_curl_near_limits(
                magnitudes, joint_indices, joint_limits, sign
            )
            redistribution[finger] = {
                joint_key: redistributed[joint_key] - magnitudes[joint_key]
                for joint_key in ("mcp", "pip", "dip")
            }

            for joint_key, joint_name in joint_lookup.items():
                target = (
                    sign
                    * redistributed[joint_key]
                )
                if joint_name in joint_names:
                    index = joint_names.index(joint_name)
                    lower, upper = joint_limits[index]
                    clipped_target = float(np.clip(target, lower, upper))
                    if abs(clipped_target) < X2_FOUR_FINGER_QPOS_DEADBAND:
                        clipped_target = 0.0
                    previous = self.previous_direct_qpos.get(joint_name)
                    if (
                        previous is not None
                        and abs(clipped_target - previous)
                        < X2_FINGER_DIRECT_QPOS_HYSTERESIS
                    ):
                        clipped_target = previous
                    direct_qpos[joint_name] = clipped_target
                    clipped[joint_name] = not np.isclose(target, clipped_target)

        self.previous_direct_qpos = direct_qpos.copy()

        return {
            "enabled": True,
            "calibrated": calibrated,
            "samples": self.samples,
            "raw_angles": raw_angles,
            "filtered_angles": angles,
            "baseline": {
                finger: values.copy() for finger, values in self.baseline.items()
            },
            "flexion_deg": flexion_deg,
            "curl_rad": curl_rad,
            "curl_rad_after_mode_gain": curl_rad_after_mode_gain,
            "redistribution": redistribution,
            "direct_qpos": direct_qpos,
            "finger_qpos_before_mode_gain": finger_qpos_before_mode_gain,
            "finger_qpos_after_mode_gain": finger_qpos_after_mode_gain,
            "clipped": clipped,
            "gain_scale": gain_scale.copy(),
            "mode_finger_gain": mode_gain,
            "finger_sign": sign,
        }


def apply_x2_finger_direct_mapping(
    qpos: np.ndarray,
    joint_names: list[str],
    profile: Optional[str],
    direct_info: dict,
) -> np.ndarray:
    if (
        not is_x2_profile_enabled(profile)
        or not direct_info.get("enabled")
        or not direct_info.get("calibrated")
    ):
        return qpos

    output = qpos.copy()
    for joint_name, target in direct_info["direct_qpos"].items():
        if joint_name in joint_names:
            output[joint_names.index(joint_name)] = target
    return output


def format_x2_finger_direct_mapping(direct_info: dict) -> str:
    if not direct_info.get("enabled"):
        return "off"
    if not direct_info.get("calibrated"):
        return (
            f"calibrating {direct_info.get('samples', 0)}/"
            f"{X2_FINGER_DIRECT_BASELINE_FRAMES}"
        )

    parts = []
    gain_scale = direct_info.get("gain_scale", {})
    if gain_scale:
        parts.append(
            "gain("
            + ",".join(
                f"{key}={gain_scale.get(key, 1.0):.2f}"
                for key in ("mcp", "pip", "dip")
            )
            + ")"
        )
    mode_gain = direct_info.get("mode_finger_gain", {})
    if mode_gain:
        parts.append(
            f"mode_finger_gain_mcp={mode_gain.get('mcp', 1.0):.2f} "
            f"mode_finger_gain_pip={mode_gain.get('pip', 1.0):.2f} "
            f"mode_finger_gain_dip={mode_gain.get('dip', 1.0):.2f}"
        )
    parts.append(f"finger_sign={direct_info.get('finger_sign', 1.0):+.0f}")
    for finger in ("index", "middle", "ring", "pinky"):
        qpos_items = []
        for joint_name in X2_FINGER_DIRECT_JOINTS[finger]:
            value = direct_info["direct_qpos"].get(joint_name)
            if value is not None:
                qpos_items.append(f"{joint_name}:{value:+.3f}")
        flex = direct_info["flexion_deg"].get(finger, {})
        curl = direct_info.get("curl_rad", {}).get(finger, 0.0)
        curl_after_gain = direct_info.get("curl_rad_after_mode_gain", {}).get(
            finger, curl
        )
        before_gain = direct_info.get("finger_qpos_before_mode_gain", {}).get(
            finger, {}
        )
        after_gain = direct_info.get("finger_qpos_after_mode_gain", {}).get(
            finger, {}
        )
        parts.append(
            f"{finger}[curl={curl:.3f};"
            f"finger_curl_before_mode_gain={curl:.3f};"
            f"finger_curl_after_mode_gain={curl_after_gain:.3f};"
            f"mcp={flex.get('mcp', 0.0):.1f},"
            f"pip={flex.get('pip', 0.0):.1f},"
            f"dip={flex.get('dip', 0.0):.1f}; "
            "finger_qpos_before_mode_gain="
            + ",".join(
                f"{joint}:{value:+.3f}" for joint, value in before_gain.items()
            )
            + "; finger_qpos_after_mode_gain="
            + ",".join(
                f"{joint}:{value:+.3f}" for joint, value in after_gain.items()
            )
            + "; final="
            + ",".join(qpos_items)
            + "]"
        )
    return " ".join(parts)


def apply_x2_four_finger_coupling(
    qpos: np.ndarray,
    joint_names: list[str],
    joint_limits: np.ndarray,
    profile: Optional[str],
    strength: float,
) -> tuple[np.ndarray, dict]:
    if not is_x2_profile_enabled(profile) or strength <= 0:
        return qpos, {"enabled": False}

    output = qpos.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    details = {}

    def blend_if_more_flexed(current: float, desired: float) -> float:
        if abs(desired) <= abs(current):
            return current
        if abs(current) > 1e-6 and np.sign(current) != np.sign(desired):
            return current
        return (1.0 - strength) * current + strength * desired

    for finger, (mcp_joint, pip_joint, dip_joint) in X2_FINGER_DIRECT_JOINTS.items():
        if mcp_joint not in joint_names or pip_joint not in joint_names:
            continue
        mcp_index = joint_names.index(mcp_joint)
        pip_index = joint_names.index(pip_joint)
        if abs(output[mcp_index]) < X2_FOUR_FINGER_QPOS_DEADBAND:
            if dip_joint in joint_names:
                dip_index = joint_names.index(dip_joint)
                details[finger] = {
                    "mcp": float(output[mcp_index]),
                    "pip": float(output[pip_index]),
                    "dip": float(output[dip_index]),
                }
            else:
                details[finger] = {
                    "mcp": float(output[mcp_index]),
                    "pip": float(output[pip_index]),
                }
            continue

        desired_pip = X2_FINGER_COUPLING_PIP_PER_MCP * output[mcp_index]
        output[pip_index] = blend_if_more_flexed(
            float(output[pip_index]), float(desired_pip)
        )
        output[pip_index] = np.clip(
            output[pip_index], joint_limits[pip_index, 0], joint_limits[pip_index, 1]
        )

        if dip_joint in joint_names:
            dip_index = joint_names.index(dip_joint)
            desired_dip = X2_FINGER_COUPLING_DIP_PER_PIP * output[pip_index]
            output[dip_index] = blend_if_more_flexed(
                float(output[dip_index]), float(desired_dip)
            )
            output[dip_index] = np.clip(
                output[dip_index],
                joint_limits[dip_index, 0],
                joint_limits[dip_index, 1],
            )
            details[finger] = {
                "mcp": float(output[mcp_index]),
                "pip": float(output[pip_index]),
                "dip": float(output[dip_index]),
            }
        else:
            details[finger] = {
                "mcp": float(output[mcp_index]),
                "pip": float(output[pip_index]),
            }

    return output, {
        "enabled": True,
        "strength": strength,
        "pip_per_mcp": X2_FINGER_COUPLING_PIP_PER_MCP,
        "dip_per_pip": X2_FINGER_COUPLING_DIP_PER_PIP,
        "finger": details,
    }


def format_x2_four_finger_coupling(info: dict) -> str:
    if not info.get("enabled"):
        return "off"
    parts = [
        f"strength={info.get('strength', 0.0):.2f}",
        f"pip={info.get('pip_per_mcp', 0.0):.2f}*mcp",
        f"dip={info.get('dip_per_pip', 0.0):.2f}*pip",
    ]
    for finger, values in info.get("finger", {}).items():
        parts.append(
            f"{finger}[mcp={values.get('mcp', 0.0):+.3f},"
            f"pip={values.get('pip', 0.0):+.3f},"
            f"dip={values.get('dip', 0.0):+.3f}]"
        )
    return " ".join(parts)


def compute_x2_pinch_strength(
    retargeting,
    ref_value: np.ndarray,
    pinch_finger: str,
    profile: Optional[str],
    pinch_gate_info: Optional[dict[str, dict]] = None,
) -> float:
    if not is_x2_profile_enabled(profile):
        return 0.0
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type != "VECTOR":
        return 0.0

    finger_tip_links = {
        "index": "rh_fftip",
        "middle": "rh_mftip",
        "ring": "rh_rftip",
        "pinky": "rh_lftip",
    }
    strengths = []
    for finger in parse_pinch_fingers(pinch_finger):
        tip_link = finger_tip_links[finger]
        for i, (origin_link, task_link) in enumerate(
            zip(optimizer.origin_link_names, optimizer.task_link_names)
        ):
            if origin_link != "rh_thtip" or task_link != tip_link:
                continue
            distance = float(np.linalg.norm(ref_value[i]))
            strength = (X2_PINCH_ACTIVATION_FAR - distance) / (
                X2_PINCH_ACTIVATION_FAR - X2_PINCH_ACTIVATION_NEAR
            )
            gate = 1.0
            if pinch_gate_info is not None:
                gate = float(pinch_gate_info.get(finger, {}).get("curl_gate", 1.0))
            strengths.append(float(np.clip(strength, 0.0, 1.0) * gate))
    return max(strengths, default=0.0)


def thumb_flexion_angle_deg(
    points: np.ndarray,
    a: int,
    b: int,
    c: int,
) -> float:
    first = points[a] - points[b]
    second = points[c] - points[b]
    denom = np.linalg.norm(first) * np.linalg.norm(second)
    if denom < 1e-8:
        return 0.0
    cos = float(np.dot(first, second) / denom)
    angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return max(0.0, 180.0 - angle)


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denom = np.linalg.norm(first) * np.linalg.norm(second)
    if denom < 1e-8:
        return 0.0
    cos = float(np.dot(first, second) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def project_to_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    normal_norm = np.linalg.norm(plane_normal)
    if normal_norm < 1e-8:
        return vector
    normal = plane_normal / normal_norm
    return vector - float(np.dot(vector, normal)) * normal


def thumb_palm_projection_flexion_deg(points: np.ndarray) -> float:
    if len(points) <= 17:
        return 0.0

    thumb_root = points[2] - points[1]
    palm_forward = points[9] - points[0]
    palm_spread = points[5] - points[17]

    thumb_projected = project_to_plane(thumb_root, palm_spread)
    palm_projected = project_to_plane(palm_forward, palm_spread)
    if np.linalg.norm(thumb_projected) < 1e-8 or np.linalg.norm(palm_projected) < 1e-8:
        return 0.0
    angle = vector_angle_deg(thumb_projected, palm_projected)
    return max(0.0, 180.0 - angle) * X2_THUMB_ROOT_PALM_PROJECTION_SCALE


def safe_unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm < 1e-8:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        return fallback / fallback_norm
    return vector / norm


def compute_x2_thj4_lateral_features(points: np.ndarray) -> dict[str, float]:
    thumb_tip = points[4]
    thumb_cmc = points[1]
    thumb_mcp = points[2]
    index_tip = points[8]
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]
    wrist = points[0]

    index_side_raw = index_mcp - pinky_mcp
    palm_width = max(float(np.linalg.norm(index_side_raw)), 1e-6)
    index_side_axis = safe_unit_vector(index_side_raw, np.array([0.0, 1.0, 0.0]))
    finger_forward_axis = safe_unit_vector(
        middle_mcp - wrist, np.array([0.0, 0.0, 1.0])
    )
    palm_normal_axis = safe_unit_vector(
        np.cross(finger_forward_axis, index_side_axis),
        np.array([1.0, 0.0, 0.0]),
    )

    thumb_vec = thumb_tip - thumb_mcp
    thumb_mcp_from_cmc = thumb_mcp - thumb_cmc
    thumb_tip_from_cmc = thumb_tip - thumb_cmc
    side_proj = float(np.dot(thumb_vec, index_side_axis) / palm_width)
    palm_proj = float(np.dot(thumb_vec, palm_normal_axis) / palm_width)
    thumb_mcp_side_proj = float(np.dot(thumb_mcp_from_cmc, index_side_axis) / palm_width)
    thumb_tip_side_proj = float(np.dot(thumb_tip_from_cmc, index_side_axis) / palm_width)
    thumb_mcp_palm_proj = float(np.dot(thumb_mcp_from_cmc, palm_normal_axis) / palm_width)
    thumb_tip_palm_proj = float(np.dot(thumb_tip_from_cmc, palm_normal_axis) / palm_width)
    pinch_distance = float(np.linalg.norm(thumb_tip - index_tip) / palm_width)
    return {
        "side_proj": side_proj,
        "palm_proj": palm_proj,
        "thumb_mcp_side_proj": thumb_mcp_side_proj,
        "thumb_tip_side_proj": thumb_tip_side_proj,
        "thumb_mcp_palm_proj": thumb_mcp_palm_proj,
        "thumb_tip_palm_proj": thumb_tip_palm_proj,
        "pinch_distance": pinch_distance,
    }


def normalize_positive_delta(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def interpolate_scalar(
    value: float,
    in_low: float,
    in_high: float,
    out_low: float,
    out_high: float,
) -> float:
    if in_high <= in_low:
        return out_low
    ratio = float(np.clip((value - in_low) / (in_high - in_low), 0.0, 1.0))
    return out_low + ratio * (out_high - out_low)


def smoothstep_scalar(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        return 0.0
    x = float(np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def x2_thj4_opposition_amount(
    progress: float,
    max_amount: float = X2_THUMB_DIRECT_MAX_MAGNITUDE["rh_THJ4"],
) -> float:
    if progress < X2_THJ4_OPPOSITION_GATE:
        return 0.0
    if progress < X2_THJ4_EARLY_PROGRESS:
        ratio = (progress - X2_THJ4_OPPOSITION_GATE) / (
            X2_THJ4_EARLY_PROGRESS - X2_THJ4_OPPOSITION_GATE
        )
        return interpolate_scalar(
            max(0.0, ratio) ** 0.75,
            0.0,
            1.0,
            0.12,
            X2_THJ4_EARLY_AMOUNT,
        )
    ratio = (min(progress, 1.0) - X2_THJ4_EARLY_PROGRESS) / (
        1.0 - X2_THJ4_EARLY_PROGRESS
    )
    return interpolate_scalar(
        max(0.0, ratio) ** 0.45,
        0.0,
        1.0,
        X2_THJ4_EARLY_AMOUNT,
        max_amount,
    )


def compute_x2_thj4_opposition_progress(
    features: dict[str, float],
    baseline: dict[str, float],
    control_mode: str,
) -> dict[str, float | bool]:
    side_delta = features["side_proj"] - baseline["side_proj"]
    palm_delta = features["palm_proj"] - baseline["palm_proj"]
    base_side_delta = (
        features["thumb_mcp_side_proj"] - baseline["thumb_mcp_side_proj"]
    )
    tip_side_delta = (
        features["thumb_tip_side_proj"] - baseline["thumb_tip_side_proj"]
    )
    base_palm_delta = (
        features["thumb_mcp_palm_proj"] - baseline["thumb_mcp_palm_proj"]
    )
    tip_palm_delta = (
        features["thumb_tip_palm_proj"] - baseline["thumb_tip_palm_proj"]
    )
    pinch_delta = baseline.get("pinch_distance", features["pinch_distance"]) - features[
        "pinch_distance"
    ]
    side_polarity = -1.0 if control_mode == "left_mode" else 1.0
    weighted_palm_delta = (
        X2_THJ4_BASE_SIDE_WEIGHT * base_palm_delta
        + X2_THJ4_TIP_SIDE_WEIGHT * tip_palm_delta
    )
    effective_base_side_delta = side_polarity * base_side_delta
    effective_tip_side_delta = side_polarity * tip_side_delta
    effective_side_delta = (
        X2_THJ4_BASE_SIDE_WEIGHT * effective_base_side_delta
        + X2_THJ4_TIP_SIDE_WEIGHT * effective_tip_side_delta
    )
    effective_palm_delta = weighted_palm_delta
    side_abs_fallback = False
    palm_abs_fallback = False

    base_side_abs_fallback = False
    tip_side_abs_fallback = False
    if (
        effective_base_side_delta < X2_THJ4_SIDE_DEADZONE
        and abs(base_side_delta) >= X2_THJ4_SIDE_DEADZONE
    ):
        effective_base_side_delta = abs(base_side_delta)
        base_side_abs_fallback = True
    if (
        effective_tip_side_delta < X2_THJ4_SIDE_DEADZONE
        and abs(tip_side_delta) >= X2_THJ4_SIDE_DEADZONE
    ):
        effective_tip_side_delta = abs(tip_side_delta)
        tip_side_abs_fallback = True

    effective_side_delta = (
        X2_THJ4_BASE_SIDE_WEIGHT * effective_base_side_delta
        + X2_THJ4_TIP_SIDE_WEIGHT * effective_tip_side_delta
    )
    if (
        effective_side_delta < X2_THJ4_SIDE_DEADZONE
        and (base_side_abs_fallback or tip_side_abs_fallback)
    ):
        side_abs_fallback = True

    if (
        effective_palm_delta < X2_THJ4_PALM_DEADZONE
        and abs(weighted_palm_delta) >= X2_THJ4_PALM_DEADZONE
    ):
        effective_palm_delta = abs(weighted_palm_delta)
        palm_abs_fallback = True

    base_side_progress = normalize_positive_delta(
        effective_base_side_delta,
        X2_THJ4_SIDE_DEADZONE,
        X2_THJ4_SIDE_HIGH,
    )
    tip_side_progress = normalize_positive_delta(
        effective_tip_side_delta,
        X2_THJ4_SIDE_DEADZONE,
        X2_THJ4_SIDE_HIGH,
    )
    side_progress = normalize_positive_delta(
        effective_side_delta,
        X2_THJ4_SIDE_DEADZONE,
        X2_THJ4_SIDE_HIGH,
    )
    palm_progress = normalize_positive_delta(
        effective_palm_delta,
        X2_THJ4_PALM_DEADZONE,
        X2_THJ4_PALM_HIGH,
    )
    gate_progress_raw = (
        0.45 * base_side_progress
        + 0.40 * tip_side_progress
        + 0.15 * palm_progress
    )
    amount_progress_raw = (
        X2_THJ4_AMOUNT_BASE_WEIGHT * base_side_progress
        + X2_THJ4_AMOUNT_PALM_WEIGHT * palm_progress
        + X2_THJ4_AMOUNT_TIP_WEIGHT * tip_side_progress
    )
    pinch_aux = 0.0
    if gate_progress_raw > X2_THJ4_OPPOSITION_GATE * 0.5:
        pinch_progress = normalize_positive_delta(max(0.0, pinch_delta), 0.010, 0.120)
        pinch_aux = min(0.035, X2_THJ4_PINCH_AUX_WEIGHT * pinch_progress)
    gate_progress = float(np.clip(gate_progress_raw + pinch_aux, 0.0, 1.0))
    amount_progress_clamped = float(np.clip(amount_progress_raw, 0.0, 1.0))
    gate_open = gate_progress >= X2_THJ4_OPPOSITION_GATE
    return {
        "side_delta": float(side_delta),
        "palm_delta": float(palm_delta),
        "base_side_delta": float(base_side_delta),
        "tip_side_delta": float(tip_side_delta),
        "base_palm_delta": float(base_palm_delta),
        "tip_palm_delta": float(tip_palm_delta),
        "effective_side_delta": float(effective_side_delta),
        "effective_base_side_delta": float(effective_base_side_delta),
        "effective_tip_side_delta": float(effective_tip_side_delta),
        "effective_palm_delta": float(effective_palm_delta),
        "base_side_progress": float(base_side_progress),
        "tip_side_progress": float(tip_side_progress),
        "side_progress": float(side_progress),
        "palm_progress": float(palm_progress),
        "gate_progress_raw": float(gate_progress_raw),
        "gate_progress": float(gate_progress),
        "amount_progress_raw": float(amount_progress_raw),
        "amount_progress_clamped": float(amount_progress_clamped),
        "lateral_progress": float(gate_progress),
        "pinch_aux": float(pinch_aux),
        "opposition_progress": float(gate_progress),
        "gate_open": bool(gate_open),
        "side_abs_fallback": bool(side_abs_fallback),
        "base_side_abs_fallback": bool(base_side_abs_fallback),
        "tip_side_abs_fallback": bool(tip_side_abs_fallback),
        "palm_abs_fallback": bool(palm_abs_fallback),
    }


def compute_x2_thumb_raw_controls(
    joint_pos: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    thj3_basal = thumb_flexion_angle_deg(joint_pos, 0, 1, 2)
    thj3_palm_projection = thumb_palm_projection_flexion_deg(joint_pos)
    thj4_features = compute_x2_thj4_lateral_features(joint_pos)
    return (
        {
            "rh_THJ4": thj4_features["side_proj"],
            "rh_THJ3": max(thj3_basal, thj3_palm_projection),
            "rh_THJ2": thumb_flexion_angle_deg(joint_pos, 1, 2, 3),
            "rh_THJ1": thumb_flexion_angle_deg(joint_pos, 2, 3, 4),
        },
        thj4_features,
    )


def get_x2_thumb_sign_table(control_mode: str) -> dict[str, float]:
    return {
        "rh_THJ4": -1.0 if control_mode == "left_mode" else 1.0,
        "rh_THJ3": -1.0,
        "rh_THJ2": -1.0,
        "rh_THJ1": -1.0,
    }


def x2_thumb_qpos_map(qpos: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    return {
        joint_name: float(qpos[joint_names.index(joint_name)])
        for joint_name in X2_THUMB_DIRECT_JOINTS
        if joint_name in joint_names
    }


def x2_thumb_debug_prefix(joint_name: str) -> str:
    return joint_name.split("_")[-1].lower()


def get_x2_thumb_direct_max_magnitude(
    joint_name: str,
    joint_names: list[str],
    joint_limits: np.ndarray,
) -> float:
    default_max = X2_THUMB_DIRECT_MAX_MAGNITUDE[joint_name]
    if joint_name not in joint_names:
        return default_max
    joint_index = joint_names.index(joint_name)
    lower, upper = joint_limits[joint_index]
    limit_abs = min(abs(float(lower)), abs(float(upper)))
    if limit_abs <= 0.0:
        return default_max
    return min(default_max, 0.90 * limit_abs)


def compute_x2_thj4_dynamic_max(
    static_max: float,
    amount_progress: float,
    thumb_flexion_progress: float,
    thumb_distal_flexion_progress: float,
) -> float:
    opposition_weight = smoothstep_scalar(
        X2_THJ4_SOFT_PROGRESS_START,
        X2_THJ4_SOFT_PROGRESS_END,
        amount_progress,
    )
    flexion_progress = max(
        float(thumb_flexion_progress),
        0.65 * float(thumb_distal_flexion_progress),
    )
    flexion_weight = smoothstep_scalar(
        X2_THJ4_SOFT_FLEXION_LOW_DEG,
        X2_THJ4_SOFT_FLEXION_HIGH_DEG,
        flexion_progress,
    )
    soft_cap = min(static_max, X2_THJ4_SOFT_MAX_LOW_FLEXION)
    high_opposition_cap = soft_cap + flexion_weight * (static_max - soft_cap)
    return float(
        (1.0 - opposition_weight) * static_max
        + opposition_weight * high_opposition_cap
    )


def compute_x2_thumb_coordination_progress(amount_progress: float) -> float:
    return smoothstep_scalar(
        X2_THUMB_COORDINATION_PROGRESS_START,
        X2_THUMB_COORDINATION_PROGRESS_END,
        amount_progress,
    )


def project_x2_thumb_qpos_to_sign_convention(
    qpos: np.ndarray,
    joint_names: list[str],
    joint_limits: np.ndarray,
    sign_table: dict[str, float],
) -> np.ndarray:
    projected = qpos.copy()
    for joint_name, sign in sign_table.items():
        if joint_name not in joint_names:
            continue
        index = joint_names.index(joint_name)
        lower, upper = joint_limits[index]
        projected[index] = np.clip(sign * abs(float(projected[index])), lower, upper)
    return projected


class X2ThumbDirectMappingState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.baseline: Optional[dict[str, float]] = None
        self.filtered_controls: Optional[dict[str, float]] = None
        self.thj4_baseline_features: Optional[dict[str, float]] = None
        self.filtered_thj4_features: Optional[dict[str, float]] = None
        self.filtered_feature_values: Optional[dict[str, float]] = None
        self.feature_ema_valid = False
        self.left_hand_detected_previous = False
        self.last_feature_ema_reset = False
        self.samples = 0

    def smooth_feature_values(self, raw_values: dict[str, float]) -> dict[str, float]:
        if self.filtered_feature_values is None or not self.feature_ema_valid:
            self.filtered_feature_values = raw_values.copy()
            self.feature_ema_valid = True
            self.last_feature_ema_reset = True
        else:
            self.last_feature_ema_reset = False
            alpha = X2_THUMB_FEATURE_EMA_ALPHA
            for key, value in raw_values.items():
                previous = self.filtered_feature_values.get(key, value)
                self.filtered_feature_values[key] = (
                    alpha * value + (1.0 - alpha) * previous
                )
        return self.filtered_feature_values.copy()

    def mark_hand_lost(self) -> None:
        self.filtered_controls = None
        self.filtered_thj4_features = None
        self.filtered_feature_values = None
        self.feature_ema_valid = False
        self.left_hand_detected_previous = False
        self.last_feature_ema_reset = False

    def update(
        self,
        joint_pos: Optional[np.ndarray],
        profile: Optional[str],
        joint_names: list[str],
        joint_limits: np.ndarray,
        control_mode: str,
    ) -> dict:
        if (
            not is_x2_profile_enabled(profile)
            or joint_pos is None
            or len(joint_pos) <= 17
        ):
            return {"enabled": False}

        left_hand_detected_previous = self.left_hand_detected_previous
        left_hand_detected_current = True
        reacquired = not left_hand_detected_previous
        raw_controls, raw_thj4_features = compute_x2_thumb_raw_controls(joint_pos)
        self.filtered_controls = raw_controls.copy()
        self.filtered_thj4_features = raw_thj4_features.copy()
        controls = self.filtered_controls.copy()
        thj4_features = self.filtered_thj4_features.copy()
        if self.baseline is None:
            self.baseline = controls.copy()
            self.thj4_baseline_features = thj4_features.copy()
            self.samples = 1
        elif self.samples < X2_THUMB_DIRECT_BASELINE_FRAMES:
            for joint_name in X2_THUMB_DIRECT_JOINTS:
                if joint_name == "rh_THJ4":
                    continue
                self.baseline[joint_name] = min(
                    self.baseline[joint_name], controls[joint_name]
                )
            baseline_count = max(self.samples, 1)
            for key, value in thj4_features.items():
                self.thj4_baseline_features[key] = (
                    baseline_count * self.thj4_baseline_features[key] + value
                ) / (baseline_count + 1)
            self.samples += 1

        calibrated = self.samples >= X2_THUMB_DIRECT_BASELINE_FRAMES
        control_amount = {}
        direct_qpos = {}
        clipped = {}
        sign_table = get_x2_thumb_sign_table(control_mode)
        direct_max_magnitude = {
            joint_name: get_x2_thumb_direct_max_magnitude(
                joint_name, joint_names, joint_limits
            )
            for joint_name in X2_THUMB_DIRECT_JOINTS
        }
        thj4_progress = compute_x2_thj4_opposition_progress(
            thj4_features, self.thj4_baseline_features, control_mode
        )
        thumb_flexion_progress_for_leak = max(
            0.0,
            controls["rh_THJ2"] - self.baseline["rh_THJ2"],
            controls["rh_THJ1"] - self.baseline["rh_THJ1"],
        )
        thj4_base_motion = max(
            abs(thj4_progress.get("base_side_delta", 0.0)),
            abs(thj4_progress.get("base_palm_delta", 0.0)),
        )
        flexion_leak_check = (
            thumb_flexion_progress_for_leak > 8.0
            and thj4_base_motion < 0.018
        )
        if flexion_leak_check:
            thj4_progress["gate_open"] = False
            thj4_progress["gate_closed_by_flexion_leak"] = True
        else:
            thj4_progress["gate_closed_by_flexion_leak"] = False
        for joint_name in X2_THUMB_DIRECT_JOINTS:
            if joint_name == "rh_THJ4":
                amount = (
                    x2_thj4_opposition_amount(
                        float(thj4_progress["amount_progress_clamped"]),
                        direct_max_magnitude["rh_THJ4"],
                    )
                    if thj4_progress["gate_open"]
                    else 0.0
                )
                thj4_progress["amount_curve_output"] = float(amount)
            else:
                baseline = self.baseline[joint_name]
                deadzone = X2_THUMB_DIRECT_DEADZONE[joint_name]
                amount = max(0.0, controls[joint_name] - baseline - deadzone)
            control_amount[joint_name] = amount
            magnitude = min(
                X2_THUMB_DIRECT_GAIN[joint_name] * amount,
                direct_max_magnitude[joint_name],
            )
            target = sign_table[joint_name] * magnitude
            if joint_name in joint_names:
                index = joint_names.index(joint_name)
                lower, upper = joint_limits[index]
                clipped_target = float(np.clip(target, lower, upper))
                direct_qpos[joint_name] = clipped_target
                clipped[joint_name] = not np.isclose(target, clipped_target)
            else:
                direct_qpos[joint_name] = 0.0
                clipped[joint_name] = False

        raw_direct_qpos = direct_qpos.copy()
        raw_feature_values = {
            "thj4_gate_progress": float(thj4_progress.get("gate_progress", 0.0)),
            "thj4_amount_progress_clamped": float(
                thj4_progress.get("amount_progress_clamped", 0.0)
            ),
            "thumb_flexion_progress": float(control_amount.get("rh_THJ2", 0.0)),
            "thumb_distal_flexion_progress": float(
                control_amount.get("rh_THJ1", 0.0)
            ),
        }
        smooth_feature_values = self.smooth_feature_values(raw_feature_values)
        self.left_hand_detected_previous = True
        thj4_progress["gate_progress_raw"] = raw_feature_values[
            "thj4_gate_progress"
        ]
        thj4_progress["gate_progress_smooth"] = smooth_feature_values[
            "thj4_gate_progress"
        ]
        thj4_progress["amount_progress_smooth"] = smooth_feature_values[
            "thj4_amount_progress_clamped"
        ]
        thj4_progress["amount_progress_used"] = float(
            np.clip(smooth_feature_values["thj4_amount_progress_clamped"], 0.0, 1.0)
        )
        thj4_dynamic_max = compute_x2_thj4_dynamic_max(
            direct_max_magnitude["rh_THJ4"],
            thj4_progress["amount_progress_used"],
            smooth_feature_values["thumb_flexion_progress"],
            smooth_feature_values["thumb_distal_flexion_progress"],
        )
        thj4_progress["dynamic_max"] = float(thj4_dynamic_max)
        thj4_progress["static_max"] = float(direct_max_magnitude["rh_THJ4"])
        thj4_progress["gate_open_raw"] = bool(thj4_progress["gate_open"])
        thj4_progress["gate_open"] = (
            smooth_feature_values["thj4_gate_progress"] >= X2_THJ4_OPPOSITION_GATE
        )
        if flexion_leak_check:
            thj4_progress["gate_open"] = False

        if "rh_THJ4" in direct_qpos:
            amount = (
                x2_thj4_opposition_amount(
                    thj4_progress["amount_progress_used"],
                    thj4_dynamic_max,
                )
                if thj4_progress["gate_open"]
                else 0.0
            )
            thj4_progress["amount_curve_output"] = float(amount)
            control_amount["rh_THJ4"] = amount
            index = joint_names.index("rh_THJ4") if "rh_THJ4" in joint_names else None
            if index is not None:
                lower, upper = joint_limits[index]
                direct_qpos["rh_THJ4"] = float(
                    np.clip(sign_table["rh_THJ4"] * amount, lower, upper)
                )
        for joint_name, feature_key in (
            ("rh_THJ2", "thumb_flexion_progress"),
            ("rh_THJ1", "thumb_distal_flexion_progress"),
        ):
            if joint_name not in direct_qpos or joint_name not in joint_names:
                continue
            index = joint_names.index(joint_name)
            lower, upper = joint_limits[index]
            raw_amount = raw_feature_values[feature_key]
            smooth_amount = smooth_feature_values[feature_key]
            raw_qpos = sign_table[joint_name] * min(
                X2_THUMB_DIRECT_GAIN[joint_name] * raw_amount,
                direct_max_magnitude[joint_name],
            )
            smooth_qpos = sign_table[joint_name] * min(
                X2_THUMB_DIRECT_GAIN[joint_name] * smooth_amount,
                direct_max_magnitude[joint_name],
            )
            direct_qpos[joint_name] = float(
                np.clip(smooth_qpos, lower, upper)
            )
            prefix = x2_thumb_debug_prefix(joint_name)
            thj_debug_raw = float(np.clip(raw_qpos, lower, upper))
            thj_debug_smooth = direct_qpos[joint_name]
            raw_feature_values[f"{prefix}_direct_qpos_from_raw_progress"] = (
                thj_debug_raw
            )
            smooth_feature_values[f"{prefix}_direct_qpos_from_smooth_progress"] = (
                thj_debug_smooth
            )

        return {
            "enabled": True,
            "calibrated": calibrated,
            "samples": self.samples,
            "raw_controls": raw_controls,
            "filtered_controls": controls,
            "thj4_features": thj4_features,
            "thj4_baseline_features": self.thj4_baseline_features.copy(),
            "thj4_progress": thj4_progress,
            "thj4_flexion_leak_check": bool(flexion_leak_check),
            "thumb_flexion_progress": float(
                smooth_feature_values["thumb_flexion_progress"]
            ),
            "thumb_flexion_progress_raw": float(
                raw_feature_values["thumb_flexion_progress"]
            ),
            "thumb_flexion_progress_smooth": float(
                smooth_feature_values["thumb_flexion_progress"]
            ),
            "thumb_distal_flexion_progress": float(
                smooth_feature_values["thumb_distal_flexion_progress"]
            ),
            "thumb_distal_flexion_progress_raw": float(
                raw_feature_values["thumb_distal_flexion_progress"]
            ),
            "thumb_distal_flexion_progress_smooth": float(
                smooth_feature_values["thumb_distal_flexion_progress"]
            ),
            "thumb_feature_smooth_alpha": X2_THUMB_FEATURE_EMA_ALPHA,
            "thumb_feature_raw_values": raw_feature_values,
            "thumb_feature_smooth_values": smooth_feature_values,
            "direct_qpos_raw": raw_direct_qpos,
            "thumb_direct_max_magnitude": direct_max_magnitude,
            "left_hand_detected_current": left_hand_detected_current,
            "left_hand_detected_previous": left_hand_detected_previous,
            "x2_thumb_feature_ema_valid": self.feature_ema_valid,
            "thumb_feature_ema_reset": bool(reacquired or self.last_feature_ema_reset),
            "baseline": self.baseline.copy(),
            "control_amount": control_amount,
            "direct_qpos": direct_qpos,
            "thumb_sign_table": sign_table,
            "control_mode": control_mode,
            "clipped": clipped,
        }


def apply_x2_thumb_direct_mapping(
    qpos: np.ndarray,
    joint_names: list[str],
    profile: Optional[str],
    direct_info: dict,
    control_mode: str,
    retargeting=None,
    target_ref_value: Optional[np.ndarray] = None,
    joint_limits: Optional[np.ndarray] = None,
) -> np.ndarray:
    if (
        not is_x2_profile_enabled(profile)
        or not direct_info.get("enabled")
        or not direct_info.get("calibrated")
    ):
        return qpos

    if joint_limits is None:
        direct_info["selection"] = {
            "selected_source": "solver_raw",
            "reason": "missing_joint_limits",
        }
        return qpos

    sign_table = get_x2_thumb_sign_table(control_mode)
    solver_raw = qpos.copy()
    solver_projected = project_x2_thumb_qpos_to_sign_convention(
        qpos, joint_names, joint_limits, sign_table
    )
    thj4_gate_open = bool(
        direct_info.get("thj4_progress", {}).get("gate_open", True)
    )
    thj4_amount_progress_used = float(
        direct_info.get("thj4_progress", {}).get("amount_progress_used", 0.0)
    )
    thumb_coordination_progress = (
        compute_x2_thumb_coordination_progress(thj4_amount_progress_used)
        if thj4_gate_open
        else 0.0
    )
    direct_info["thumb_coordination_progress"] = float(thumb_coordination_progress)
    if not thj4_gate_open and "rh_THJ4" in joint_names:
        thj4_index = joint_names.index("rh_THJ4")
        solver_projected[thj4_index] = 0.0
    direct_signed = solver_projected.copy()
    active_joints = []
    for joint_name, target in direct_info["direct_qpos"].items():
        if joint_name == "rh_THJ4":
            continue
        if joint_name not in joint_names:
            continue
        if (
            direct_info["control_amount"].get(joint_name, 0.0)
            < X2_THUMB_DIRECT_MIN_AMOUNT.get(joint_name, 0.0)
        ):
            continue
        magnitude = abs(float(target))
        if magnitude < 1e-6:
            continue
        index = joint_names.index(joint_name)
        lower, upper = joint_limits[index]
        signed_target = float(np.clip(sign_table[joint_name] * magnitude, lower, upper))
        direct_signed[index] = signed_target
        active_joints.append((joint_name, index))

    def apply_thj4_channel(candidate_qpos: np.ndarray) -> np.ndarray:
        output = candidate_qpos.copy()
        if "rh_THJ4" not in joint_names:
            direct_info["thj4_source"] = "unavailable"
            return output

        thj4_index = joint_names.index("rh_THJ4")
        lower, upper = joint_limits[thj4_index]
        solver_thj4 = float(solver_projected[thj4_index])
        direct_thj4 = float(direct_info.get("direct_qpos", {}).get("rh_THJ4", 0.0))
        if thj4_gate_open:
            final_thj4 = float(np.clip(direct_thj4, lower, upper))
            thj4_source = "direct_opposition"
        else:
            final_thj4 = 0.0
            thj4_source = "gate_closed_zero"

        output[thj4_index] = final_thj4
        direct_info["thj4_source"] = thj4_source
        direct_info["thj4_solver_projected_qpos"] = solver_thj4
        direct_info["thj4_direct_qpos"] = direct_thj4
        direct_info["thj4_final_before_smoothing"] = final_thj4
        return output

    def apply_thumb_flex_channels(candidate_qpos: np.ndarray) -> np.ndarray:
        output = candidate_qpos.copy()
        for joint_name in ("rh_THJ3", "rh_THJ2", "rh_THJ1"):
            if joint_name not in joint_names:
                continue
            joint_index = joint_names.index(joint_name)
            lower, upper = joint_limits[joint_index]
            solver_value = float(solver_projected[joint_index])
            direct_value = float(direct_info.get("direct_qpos", {}).get(joint_name, 0.0))
            amount = float(direct_info.get("control_amount", {}).get(joint_name, 0.0))

            if joint_name == "rh_THJ1":
                force_zero = (
                    amount < X2_THUMB_DIRECT_MIN_AMOUNT[joint_name]
                    or abs(direct_value) < 0.025
                )
                if force_zero:
                    final_value = 0.0
                else:
                    final_value = 0.85 * direct_value + 0.15 * solver_value
            elif joint_name == "rh_THJ2":
                force_zero = (
                    amount < X2_THUMB_DIRECT_MIN_AMOUNT[joint_name]
                    or abs(direct_value) < 0.025
                )
                if force_zero:
                    final_value = 0.0
                else:
                    final_value = 0.70 * direct_value + 0.30 * solver_value
            else:
                force_zero = False
                final_value = 0.70 * direct_value + 0.30 * solver_value
                final_value = float(np.clip(final_value, -0.25, 0.0))

            final_before_coordination = final_value
            extra_magnitude = (
                thumb_coordination_progress
                * X2_THUMB_COORDINATION_EXTRA.get(joint_name, 0.0)
            )
            if extra_magnitude > 1e-6:
                final_value += sign_table[joint_name] * extra_magnitude
                force_zero = False
            if joint_name == "rh_THJ3":
                final_value = float(np.clip(final_value, -0.25, 0.0))
            final_value = float(np.clip(final_value, lower, upper))
            output[joint_index] = final_value
            prefix = x2_thumb_debug_prefix(joint_name)
            direct_info[f"{prefix}_direct_qpos"] = direct_value
            direct_info[f"{prefix}_solver_projected_qpos"] = solver_value
            direct_info[
                f"{prefix}_final_before_smoothing_before_coordination"
            ] = float(final_before_coordination)
            direct_info[f"{prefix}_extra_from_opposition"] = float(extra_magnitude)
            direct_info[
                f"{prefix}_final_before_smoothing_after_coordination"
            ] = final_value
            if joint_name in {"rh_THJ2", "rh_THJ1"}:
                direct_info[f"{prefix}_force_zero"] = bool(force_zero)
            direct_info[f"{prefix}_final_before_smoothing"] = final_value
        return output

    if not active_joints:
        selected = apply_thumb_flex_channels(solver_projected)
        selected = apply_thj4_channel(selected)
        direct_info["selection"] = {
            "selected_source": "solver_projected",
            "flex_source": "solver_projected",
            "reason": "no_active_direct_joints",
        }
        direct_info["solver_thumb_qpos_raw"] = x2_thumb_qpos_map(solver_raw, joint_names)
        direct_info["solver_thumb_qpos_projected"] = x2_thumb_qpos_map(
            solver_projected, joint_names
        )
        direct_info["direct_thumb_amount"] = direct_info.get("control_amount", {}).copy()
        direct_info["direct_thumb_qpos_signed"] = x2_thumb_qpos_map(
            direct_signed, joint_names
        )
        direct_info["selected_thumb_qpos_before_smoothing"] = x2_thumb_qpos_map(
            selected, joint_names
        )
        direct_info["thumb_sign_table"] = sign_table
        return selected

    blend = solver_projected.copy()
    for _, index in active_joints:
        blend[index] = 0.55 * solver_projected[index] + 0.45 * direct_signed[index]

    solver_score = compute_x2_thumb_candidate_score(
        retargeting, target_ref_value, solver_projected, solver_projected, joint_names
    )
    direct_score = compute_x2_thumb_candidate_score(
        retargeting, target_ref_value, direct_signed, solver_projected, joint_names
    )
    blend_score = compute_x2_thumb_candidate_score(
        retargeting, target_ref_value, blend, solver_projected, joint_names
    )

    selected = solver_projected
    selected_source = "solver_projected"
    selected_score = solver_score

    def clearly_better(candidate_score: Optional[dict], baseline_score: Optional[dict]) -> bool:
        if candidate_score is None or baseline_score is None:
            return False
        candidate = candidate_score["score"]
        baseline = baseline_score["score"]
        return (
            candidate < baseline * (1.0 - X2_THUMB_DIRECT_MIN_RELATIVE_IMPROVEMENT)
            or baseline - candidate > X2_THUMB_DIRECT_MIN_ABSOLUTE_IMPROVEMENT
        )

    for source, candidate, score in (
        ("direct_signed", direct_signed, direct_score),
        ("blend", blend, blend_score),
    ):
        if clearly_better(score, solver_score) and (
            selected_score is None or score["score"] < selected_score["score"]
        ):
            selected = candidate
            selected_source = source
            selected_score = score

    selected = apply_thumb_flex_channels(selected)
    selected = apply_thj4_channel(selected)

    direct_info["selection"] = {
        "selected_source": selected_source,
        "flex_source": selected_source,
        "solver_projected_err": None if solver_score is None else solver_score["score"],
        "solver_projected_angle": None
        if solver_score is None
        else solver_score.get("max_angle", 0.0),
        "direct_signed_err": None if direct_score is None else direct_score["score"],
        "direct_signed_angle": None
        if direct_score is None
        else direct_score.get("max_angle", 0.0),
        "blend_err": None if blend_score is None else blend_score["score"],
        "blend_angle": None if blend_score is None else blend_score.get("max_angle", 0.0),
        "selected_err": None if selected_score is None else selected_score["score"],
        "active_joints": [joint_name for joint_name, _ in active_joints],
    }
    direct_info["solver_thumb_qpos_raw"] = x2_thumb_qpos_map(solver_raw, joint_names)
    direct_info["solver_thumb_qpos_projected"] = x2_thumb_qpos_map(
        solver_projected, joint_names
    )
    direct_info["direct_thumb_amount"] = direct_info.get("control_amount", {}).copy()
    direct_info["direct_thumb_qpos_signed"] = x2_thumb_qpos_map(direct_signed, joint_names)
    direct_info["selected_thumb_qpos_before_smoothing"] = x2_thumb_qpos_map(
        selected, joint_names
    )
    direct_info["thumb_sign_table"] = sign_table
    return selected


def enforce_x2_thj4_gate_after_smoothing(
    qpos: np.ndarray,
    joint_names: list[str],
    direct_info: dict,
    smoother: Optional[X2CommandSmoother] = None,
) -> np.ndarray:
    output = qpos.copy()
    if "rh_THJ4" not in joint_names:
        return output

    thj4_index = joint_names.index("rh_THJ4")
    gate_open = bool(direct_info.get("thj4_progress", {}).get("gate_open", True))
    gate_closed_decay = False
    if (
        direct_info.get("enabled")
        and direct_info.get("calibrated")
        and not gate_open
    ):
        output[thj4_index] = 0.0
        gate_closed_decay = True
        if (
            smoother is not None
            and smoother.previous is not None
            and len(smoother.previous) == len(output)
        ):
            smoother.previous[thj4_index] = output[thj4_index]

    direct_info["thj4_gate_closed_decay"] = bool(gate_closed_decay)
    direct_info["thj4_after_gate_enforcement"] = float(output[thj4_index])
    direct_info["thj4_final_after_smoothing"] = float(output[thj4_index])
    for joint_name in ("rh_THJ3", "rh_THJ2", "rh_THJ1"):
        if joint_name not in joint_names:
            continue
        joint_index = joint_names.index(joint_name)
        enforced_value = float(output[joint_index])
        force_zero = False
        if joint_name == "rh_THJ3":
            enforced_value = float(np.clip(enforced_value, -0.25, 0.0))
        elif direct_info.get(f"{x2_thumb_debug_prefix(joint_name)}_force_zero", False):
            enforced_value = 0.0
            force_zero = True

        if force_zero or not np.isclose(enforced_value, output[joint_index]):
            output[joint_index] = enforced_value
            if (
                smoother is not None
                and smoother.previous is not None
                and len(smoother.previous) == len(output)
            ):
                smoother.previous[joint_index] = enforced_value

        direct_info[
            f"{x2_thumb_debug_prefix(joint_name)}_final_after_smoothing"
        ] = float(output[joint_index])
    return output


def reset_x2_thumb_smoother_state_to_command(
    smoother: X2CommandSmoother,
    command_qpos: np.ndarray,
    joint_names: list[str],
) -> None:
    if smoother.previous is None or len(smoother.previous) != len(command_qpos):
        return
    for joint_name in X2_THUMB_DIRECT_JOINTS:
        if joint_name in joint_names:
            joint_index = joint_names.index(joint_name)
            smoother.previous[joint_index] = command_qpos[joint_index]


def format_x2_thumb_direct_mapping(direct_info: dict) -> str:
    if not direct_info.get("enabled"):
        return "off"
    if not direct_info.get("calibrated"):
        return (
            f"calibrating {direct_info.get('samples', 0)}/"
            f"{X2_THUMB_DIRECT_BASELINE_FRAMES}"
        )

    def format_value(name: str, value: float) -> str:
        if name == "rh_THJ4":
            return f"{name}:{value:.3f}"
        return f"{name}:{value:.1f}deg"

    def format_amount(name: str, value: float) -> str:
        if name == "rh_THJ4":
            return f"{name}:{value:.3f}"
        return f"{name}:{value:.1f}deg"

    def format_qpos_map(values: dict) -> str:
        if not values:
            return "n/a"
        return ",".join(
            f"{name}:{values.get(name, 0.0):+.3f}"
            for name in X2_THUMB_DIRECT_JOINTS
            if name in values
        )

    def format_sign_table(values: dict) -> str:
        if not values:
            return "n/a"
        return ",".join(
            f"{name}:{values.get(name, 0.0):+.0f}"
            for name in X2_THUMB_DIRECT_JOINTS
            if name in values
        )

    selection = direct_info.get("selection", {})
    thj4_progress = direct_info.get("thj4_progress", {})
    thj4_features = direct_info.get("thj4_features", {})
    thj4_baseline = direct_info.get("thj4_baseline_features", {})
    direct_qpos = direct_info.get("direct_qpos", {})
    selection_parts = [
        f"flex_source={selection.get('flex_source', selection.get('selected_source', 'pending'))}",
    ]
    if selection.get("solver_projected_err") is not None:
        selection_parts.append(
            f"solver_projected_err={selection.get('solver_projected_err', 0.0):.4f}"
        )
    if selection.get("direct_signed_err") is not None:
        selection_parts.append(
            f"direct_signed_err={selection.get('direct_signed_err', 0.0):.4f}"
        )
    if selection.get("blend_err") is not None:
        selection_parts.append(f"blend_err={selection.get('blend_err', 0.0):.4f}")
    if selection.get("selected_err") is not None:
        selection_parts.append(f"selected_err={selection.get('selected_err', 0.0):.4f}")
    if selection.get("direct_signed_angle") is not None:
        selection_parts.append(
            f"direct_angle={selection.get('direct_signed_angle', 0.0):.1f}"
        )

    return " ".join(
        [
            f"mode={direct_info.get('control_mode', 'unknown')}",
            "thumb_sign_table="
            + format_sign_table(direct_info.get("thumb_sign_table", {})),
            "left_hand_detected_current="
            f"{direct_info.get('left_hand_detected_current', False)}",
            "left_hand_detected_previous="
            f"{direct_info.get('left_hand_detected_previous', False)}",
            "x2_thumb_feature_ema_valid="
            f"{direct_info.get('x2_thumb_feature_ema_valid', False)}",
            "thumb_feature_ema_reset="
            f"{direct_info.get('thumb_feature_ema_reset', False)}",
            f"thj4_source={direct_info.get('thj4_source', 'pending')}",
            "flex_select(" + ",".join(selection_parts) + ")",
            "thj4_gate_progress="
            f"{thj4_progress.get('gate_progress', 0.0):.3f}",
            "thj4_gate_progress_raw="
            f"{thj4_progress.get('gate_progress_raw', thj4_progress.get('gate_progress', 0.0)):.3f}",
            "thj4_gate_progress_smooth="
            f"{thj4_progress.get('gate_progress_smooth', thj4_progress.get('gate_progress', 0.0)):.3f}",
            "thj4_amount_progress_raw="
            f"{thj4_progress.get('amount_progress_raw', 0.0):.3f}",
            "thj4_amount_progress_smooth="
            f"{thj4_progress.get('amount_progress_smooth', thj4_progress.get('amount_progress_clamped', 0.0)):.3f}",
            "thj4_amount_progress_used="
            f"{thj4_progress.get('amount_progress_used', thj4_progress.get('amount_progress_clamped', 0.0)):.3f}",
            "thj4_base_side_progress="
            f"{thj4_progress.get('base_side_progress', 0.0):.3f}",
            "thj4_tip_side_progress="
            f"{thj4_progress.get('tip_side_progress', 0.0):.3f}",
            "thj4_palm_progress="
            f"{thj4_progress.get('palm_progress', 0.0):.3f}",
            "thj4_base_weight="
            f"{X2_THJ4_AMOUNT_BASE_WEIGHT:.2f}",
            "thj4_tip_weight="
            f"{X2_THJ4_AMOUNT_TIP_WEIGHT:.2f}",
            "thj4_amount_curve_output="
            f"{thj4_progress.get('amount_curve_output', 0.0):.3f}",
            "thj4_lateral_opposition_progress="
            f"{thj4_progress.get('lateral_progress', 0.0):.3f}",
            "thj4_side_proj="
            f"{thj4_features.get('side_proj', 0.0):+.3f}",
            "thj4_palm_proj="
            f"{thj4_features.get('palm_proj', 0.0):+.3f}",
            "thj4_baseline_side_proj="
            f"{thj4_baseline.get('side_proj', 0.0):+.3f}",
            "thj4_baseline_palm_proj="
            f"{thj4_baseline.get('palm_proj', 0.0):+.3f}",
            "thj4_side_delta_raw="
            f"{thj4_progress.get('side_delta', 0.0):+.3f}",
            "thj4_palm_delta_raw="
            f"{thj4_progress.get('palm_delta', 0.0):+.3f}",
            "thj4_effective_side_delta="
            f"{thj4_progress.get('effective_side_delta', 0.0):+.3f}",
            "thj4_effective_palm_delta="
            f"{thj4_progress.get('effective_palm_delta', 0.0):+.3f}",
            "thj4_base_side_delta="
            f"{thj4_progress.get('base_side_delta', 0.0):+.3f}",
            "thj4_tip_side_delta="
            f"{thj4_progress.get('tip_side_delta', 0.0):+.3f}",
            "thj4_flexion_leak_check="
            f"{direct_info.get('thj4_flexion_leak_check', False)}",
            "thj4_gate_open="
            f"{thj4_progress.get('gate_open', False)}",
            "thj4_gate_closed_decay="
            f"{direct_info.get('thj4_gate_closed_decay', False)}",
            "thj4_after_gate_enforcement="
            f"{direct_info.get('thj4_after_gate_enforcement', 0.0):+.3f}",
            "thj4_dynamic_max="
            f"{thj4_progress.get('dynamic_max', direct_info.get('thumb_direct_max_magnitude', {}).get('rh_THJ4', 0.0)):.3f}",
            "thj4_max="
            f"{direct_info.get('thumb_direct_max_magnitude', {}).get('rh_THJ4', 0.0):.3f}",
            "thj4_direct_qpos="
            f"{direct_qpos.get('rh_THJ4', 0.0):+.3f}",
            "thj4_solver_projected_qpos="
            f"{direct_info.get('thj4_solver_projected_qpos', 0.0):+.3f}",
            "thj4_final_before_smoothing="
            f"{direct_info.get('thj4_final_before_smoothing', 0.0):+.3f}",
            "thj4_final_after_smoothing="
            f"{direct_info.get('thj4_final_after_smoothing', 0.0):+.3f}",
            "thumb_flexion_progress="
            f"{direct_info.get('thumb_flexion_progress', 0.0):.1f}",
            "thumb_flexion_progress_raw="
            f"{direct_info.get('thumb_flexion_progress_raw', 0.0):.1f}",
            "thumb_flexion_progress_smooth="
            f"{direct_info.get('thumb_flexion_progress_smooth', 0.0):.1f}",
            "thumb_distal_flexion_progress="
            f"{direct_info.get('thumb_distal_flexion_progress', 0.0):.1f}",
            "thumb_distal_flexion_progress_raw="
            f"{direct_info.get('thumb_distal_flexion_progress_raw', 0.0):.1f}",
            "thumb_distal_flexion_progress_smooth="
            f"{direct_info.get('thumb_distal_flexion_progress_smooth', 0.0):.1f}",
            "thumb_coordination_progress="
            f"{direct_info.get('thumb_coordination_progress', 0.0):.3f}",
            "thj3_direct_qpos="
            f"{direct_info.get('thj3_direct_qpos', 0.0):+.3f}",
            "thj3_solver_projected_qpos="
            f"{direct_info.get('thj3_solver_projected_qpos', 0.0):+.3f}",
            "thj3_extra_from_opposition="
            f"{direct_info.get('thj3_extra_from_opposition', 0.0):.3f}",
            "thj3_final_before_smoothing_after_coordination="
            f"{direct_info.get('thj3_final_before_smoothing_after_coordination', direct_info.get('thj3_final_before_smoothing', 0.0)):+.3f}",
            "thj3_final_before_smoothing="
            f"{direct_info.get('thj3_final_before_smoothing', 0.0):+.3f}",
            "thj3_final_after_smoothing="
            f"{direct_info.get('thj3_final_after_smoothing', 0.0):+.3f}",
            "thj2_direct_qpos="
            f"{direct_info.get('thj2_direct_qpos', 0.0):+.3f}",
            "thj2_direct_qpos_from_raw_progress="
            f"{direct_info.get('thumb_feature_raw_values', {}).get('thj2_direct_qpos_from_raw_progress', 0.0):+.3f}",
            "thj2_direct_qpos_from_smooth_progress="
            f"{direct_info.get('thumb_feature_smooth_values', {}).get('thj2_direct_qpos_from_smooth_progress', 0.0):+.3f}",
            "thj2_direct_qpos_used="
            f"{direct_info.get('direct_qpos', {}).get('rh_THJ2', 0.0):+.3f}",
            "thj2_solver_projected_qpos="
            f"{direct_info.get('thj2_solver_projected_qpos', 0.0):+.3f}",
            "thj2_extra_from_opposition="
            f"{direct_info.get('thj2_extra_from_opposition', 0.0):.3f}",
            "thj2_final_before_smoothing_after_coordination="
            f"{direct_info.get('thj2_final_before_smoothing_after_coordination', direct_info.get('thj2_final_before_smoothing', 0.0)):+.3f}",
            "thj2_final_before_smoothing="
            f"{direct_info.get('thj2_final_before_smoothing', 0.0):+.3f}",
            "thj2_final_after_smoothing="
            f"{direct_info.get('thj2_final_after_smoothing', 0.0):+.3f}",
            "thj1_direct_qpos="
            f"{direct_info.get('thj1_direct_qpos', 0.0):+.3f}",
            "thj1_direct_qpos_from_raw_progress="
            f"{direct_info.get('thumb_feature_raw_values', {}).get('thj1_direct_qpos_from_raw_progress', 0.0):+.3f}",
            "thj1_direct_qpos_from_smooth_progress="
            f"{direct_info.get('thumb_feature_smooth_values', {}).get('thj1_direct_qpos_from_smooth_progress', 0.0):+.3f}",
            "thj1_direct_qpos_used="
            f"{direct_info.get('direct_qpos', {}).get('rh_THJ1', 0.0):+.3f}",
            "thj1_solver_projected_qpos="
            f"{direct_info.get('thj1_solver_projected_qpos', 0.0):+.3f}",
            "thj1_extra_from_opposition="
            f"{direct_info.get('thj1_extra_from_opposition', 0.0):.3f}",
            "thj1_final_before_smoothing_after_coordination="
            f"{direct_info.get('thj1_final_before_smoothing_after_coordination', direct_info.get('thj1_final_before_smoothing', 0.0)):+.3f}",
            "thj1_final_before_smoothing="
            f"{direct_info.get('thj1_final_before_smoothing', 0.0):+.3f}",
            "thj1_final_after_smoothing="
            f"{direct_info.get('thj1_final_after_smoothing', 0.0):+.3f}",
            "thj2_qpos="
            f"{direct_qpos.get('rh_THJ2', 0.0):+.3f}",
            "thj1_qpos="
            f"{direct_qpos.get('rh_THJ1', 0.0):+.3f}",
            "solver_raw=" + format_qpos_map(direct_info.get("solver_thumb_qpos_raw", {})),
            "solver_projected="
            + format_qpos_map(direct_info.get("solver_thumb_qpos_projected", {})),
            "raw="
            + ",".join(
                format_value(name, value)
                for name, value in direct_info["raw_controls"].items()
            ),
            "baseline="
            + ",".join(
                format_value(name, value)
                for name, value in direct_info["baseline"].items()
            ),
            "amount="
            + ",".join(
                format_amount(name, value)
                for name, value in direct_info["control_amount"].items()
            ),
            "direct_signed=" + format_qpos_map(direct_info.get("direct_qpos", {})),
            "selected_before_smoothing="
            + format_qpos_map(direct_info.get("selected_thumb_qpos_before_smoothing", {})),
            "selected_after_smoothing="
            + format_qpos_map(direct_info.get("selected_thumb_qpos_after_smoothing", {})),
            "final_mujoco_thumb_qpos="
            + format_qpos_map(direct_info.get("final_mujoco_thumb_qpos", {})),
        ]
    )


def as_numpy_indices(indices) -> np.ndarray:
    if hasattr(indices, "detach"):
        return indices.detach().cpu().numpy().astype(np.int64)
    return np.asarray(indices, dtype=np.int64)


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return float("nan")
    cos = float(np.dot(a, b) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_retargeting_robot_vectors(retargeting, robot_qpos: np.ndarray):
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type != "VECTOR":
        return None

    robot = optimizer.robot
    adapted_qpos = (
        optimizer.adaptor.forward_qpos(robot_qpos)
        if optimizer.adaptor is not None
        else robot_qpos
    )
    robot.compute_forward_kinematics(adapted_qpos)
    link_poses = [
        robot.get_link_pose(index) for index in optimizer.computed_link_indices
    ]
    body_pos = np.asarray([pose[:3, 3] for pose in link_poses])
    origin_indices = as_numpy_indices(optimizer.origin_link_indices)
    task_indices = as_numpy_indices(optimizer.task_link_indices)
    origin_pos = body_pos[origin_indices]
    task_pos = body_pos[task_indices]
    return task_pos - origin_pos, origin_pos, task_pos


def compute_x2_thumb_regularity_penalty(
    candidate_qpos: np.ndarray,
    reference_qpos: np.ndarray,
    joint_names: list[str],
) -> float:
    penalty = 0.0
    for joint_name in X2_THUMB_DIRECT_JOINTS:
        if joint_name not in joint_names:
            continue
        index = joint_names.index(joint_name)
        value = abs(float(candidate_qpos[index]))
        soft_limit = X2_THUMB_DIRECT_MAX_MAGNITUDE[joint_name]
        if value > soft_limit:
            penalty += 0.20 * (value - soft_limit) ** 2
        penalty += 0.012 * value
        penalty += 0.030 * abs(float(candidate_qpos[index] - reference_qpos[index]))

    thj4 = abs(float(candidate_qpos[joint_names.index("rh_THJ4")])) if "rh_THJ4" in joint_names else 0.0
    thj2 = abs(float(candidate_qpos[joint_names.index("rh_THJ2")])) if "rh_THJ2" in joint_names else 0.0
    thj1 = abs(float(candidate_qpos[joint_names.index("rh_THJ1")])) if "rh_THJ1" in joint_names else 0.0
    if thj4 > 0.50 and thj2 + thj1 < 0.20:
        penalty += 0.035
    if thj2 > 0.50 and thj1 > 0.40:
        penalty += 0.020
    return float(penalty)


def compute_x2_thumb_candidate_score(
    retargeting,
    target_ref_value: Optional[np.ndarray],
    robot_qpos: np.ndarray,
    reference_qpos: np.ndarray,
    joint_names: list[str],
) -> Optional[dict]:
    if retargeting is None or target_ref_value is None:
        return None

    vectors = compute_retargeting_robot_vectors(retargeting, robot_qpos)
    if vectors is None:
        return None

    robot_vec, _, _ = vectors
    optimizer = retargeting.optimizer
    target_vec = target_ref_value * optimizer.scaling
    preferred_norm_errors = []
    preferred_angles = []
    fallback_errors = []
    palm_side_penalty = 0.0
    for i, (origin_link, task_link) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names)
    ):
        if task_link != "rh_thtip":
            continue
        target = target_vec[i]
        robot = robot_vec[i]
        norm_error = float(np.linalg.norm(robot - target))
        angle = 0.0
        if np.linalg.norm(target) > 1e-6 and np.linalg.norm(robot) > 1e-6:
            angle = vector_angle_deg(target, robot)
        if origin_link in {"rh_fftip", "rh_mftip"}:
            preferred_norm_errors.append(norm_error)
            preferred_angles.append(angle)
        elif origin_link == "rh_palm":
            fallback_errors.append(norm_error + 0.002 * angle)
            if abs(target[1]) > 1e-6 and abs(robot[1]) > 1e-6:
                if np.sign(target[1]) != np.sign(robot[1]):
                    palm_side_penalty += 0.04
            palm_side_penalty += 0.20 * abs(float(robot[1] - target[1]))

    if preferred_norm_errors:
        norm_error = float(np.mean(preferred_norm_errors))
        mean_angle = float(np.mean(preferred_angles))
        max_angle = float(np.max(preferred_angles))
        angle_penalty = 0.0020 * mean_angle + 0.0040 * max(0.0, max_angle - 55.0)
        vector_score = norm_error + angle_penalty + palm_side_penalty
    elif fallback_errors:
        norm_error = float(np.mean(fallback_errors))
        mean_angle = 0.0
        max_angle = 0.0
        vector_score = norm_error + palm_side_penalty
    else:
        return None

    regularity_penalty = compute_x2_thumb_regularity_penalty(
        robot_qpos, reference_qpos, joint_names
    )
    return {
        "score": float(vector_score + regularity_penalty),
        "vector_score": float(vector_score),
        "norm_error": norm_error,
        "mean_angle": mean_angle,
        "max_angle": max_angle,
        "regularity_penalty": regularity_penalty,
        "palm_side_penalty": float(palm_side_penalty),
    }


def get_target_vector_groups(retargeting) -> dict[str, list[int]]:
    optimizer = retargeting.optimizer
    if optimizer.retargeting_type == "POSITION":
        raise ValueError("Target vector groups only support vector-style retargeting.")

    human_task_indices = np.asarray(optimizer.target_link_human_indices)[1]
    groups = {
        "all": list(range(len(optimizer.task_link_names))),
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
        key = link_name.lower()
        groups.setdefault(key, []).append(i)
    for i, (origin_link, task_link) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names)
    ):
        key = f"{task_link}-{origin_link}".lower()
        groups.setdefault(key, []).append(i)
        groups.setdefault(key.replace("-", "_"), []).append(i)
    return groups


def parse_target_vector_axis_gains(
    target_vector_axis_gain: Optional[str], retargeting
) -> np.ndarray:
    num_vectors = len(retargeting.optimizer.task_link_names)
    gains = np.ones((num_vectors, 3), dtype=np.float32)
    if (
        target_vector_axis_gain is None
        or target_vector_axis_gain.lower() in {"", "none", "default"}
    ):
        return gains

    groups = get_target_vector_groups(retargeting)
    axis_to_index = {
        "x": 0,
        "0": 0,
        "y": 1,
        "1": 1,
        "z": 2,
        "2": 2,
    }
    for raw_item in target_vector_axis_gain.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                "Invalid target vector axis gain item "
                f"'{item}'. Use e.g. y=0.5 or thumb:z=0.55."
            )

        key = key.strip().lower()
        gain = float(value.strip())
        if gain < 0:
            raise ValueError(f"Target vector axis gain for {key} must be non-negative.")

        if "." in key:
            group_key, axis_key = key.rsplit(".", 1)
        elif "/" in key:
            group_key, axis_key = key.rsplit("/", 1)
        elif ":" in key:
            group_key, axis_key = key.rsplit(":", 1)
        else:
            group_key, axis_key = "all", key

        group_key = group_key.strip().lower()
        axis_key = axis_key.strip().lower()
        if axis_key not in axis_to_index:
            raise ValueError(
                f"Unknown target vector axis '{axis_key}'. Use x/y/z or 0/1/2."
            )
        vector_indices = groups.get(group_key)
        if vector_indices is None or not vector_indices:
            if group_key.isdigit():
                vector_indices = [int(group_key)]
            else:
                raise ValueError(
                    f"Unknown target vector group '{group_key}'. "
                    "Use all, thumb, thumb_tip, thumb_ip, index, index_tip, "
                    "index_mid, a task link name, or a vector row index."
                )

        axis_index = axis_to_index[axis_key]
        for vector_index in vector_indices:
            if vector_index < 0 or vector_index >= num_vectors:
                raise ValueError(
                    f"Target vector index {vector_index} out of range [0, {num_vectors})."
                )
            gains[vector_index, axis_index] = gain
    return gains


def apply_target_vector_axis_gains(
    ref_value: np.ndarray, target_vector_axis_gains: np.ndarray
) -> np.ndarray:
    return ref_value * target_vector_axis_gains


def format_target_vector_axis_gains(retargeting, gains: np.ndarray) -> str:
    optimizer = retargeting.optimizer
    parts = []
    for i, (origin_link, task_link, gain) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names, gains)
    ):
        if np.allclose(gain, 1.0):
            continue
        parts.append(
            f"{i}:{task_link}-{origin_link}"
            f"[x={gain[0]:.2f},y={gain[1]:.2f},z={gain[2]:.2f}]"
        )
    return ", ".join(parts) if parts else "default"


def parse_limit_pair(value: str) -> tuple[float, float]:
    if ":" in value:
        lower_text, upper_text = value.split(":", 1)
    elif ".." in value:
        lower_text, upper_text = value.split("..", 1)
    else:
        raise ValueError(
            f"Invalid joint limit '{value}'. Use lower:upper, e.g. -1.31:0.0."
        )
    lower = float(lower_text.strip())
    upper = float(upper_text.strip())
    if lower > upper:
        raise ValueError(f"Invalid joint limit '{value}': lower > upper.")
    return lower, upper


def apply_retarget_joint_limit_overrides(
    retargeting, retarget_joint_limit: Optional[str]
) -> np.ndarray:
    full_limits = retargeting.optimizer.robot.joint_limits.copy()
    full_limits, applied = build_joint_limit_overrides(
        retargeting.joint_names, full_limits, retarget_joint_limit
    )
    if not applied:
        return full_limits

    target_limits = full_limits[retargeting.optimizer.idx_pin2target]
    retargeting.joint_limits = target_limits.copy()
    retargeting.optimizer.set_joint_limit(target_limits)
    retargeting.last_qpos = np.clip(
        retargeting.last_qpos, target_limits[:, 0], target_limits[:, 1]
    )
    logger.warning("Retarget joint limit overrides: " + ", ".join(applied))
    return full_limits


def build_joint_limit_overrides(
    joint_names: list[str],
    base_limits: np.ndarray,
    retarget_joint_limit: Optional[str],
) -> tuple[np.ndarray, list[str]]:
    limits = base_limits.copy()
    if (
        retarget_joint_limit is None
        or retarget_joint_limit.lower() in {"", "none", "default"}
    ):
        return limits, []

    groups = {
        "all": joint_names,
        "th": [name for name in joint_names if "_TH" in name],
        "thumb": [name for name in joint_names if "_TH" in name],
        "ff": [name for name in joint_names if "_FF" in name],
        "mf": [name for name in joint_names if "_MF" in name],
        "rf": [name for name in joint_names if "_RF" in name],
        "lf": [name for name in joint_names if "_LF" in name],
    }

    applied = []
    for raw_item in retarget_joint_limit.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            raise ValueError(
                f"Invalid retarget joint limit item '{item}'. "
                "Use e.g. rh_THJ3=-1.31:1.31."
            )

        key = key.strip()
        lower, upper = parse_limit_pair(value.strip())
        names = groups.get(key.lower(), [key])
        missing_names = [name for name in names if name not in joint_names]
        if missing_names:
            raise ValueError(f"Unknown retarget joint limit target(s): {missing_names}")
        for name in names:
            joint_index = joint_names.index(name)
            limits[joint_index] = [lower, upper]
            applied.append(f"{name}=[{lower:+.3f},{upper:+.3f}]")

    return limits, applied


def parse_pinch_fingers(pinch_finger: str) -> list[str]:
    finger_specs = {
        "index": ("rh_fftip", "rh_ffmiddle", 8, 6),
        "ff": ("rh_fftip", "rh_ffmiddle", 8, 6),
        "middle": ("rh_mftip", "rh_mfmiddle", 12, 10),
        "mfinger": ("rh_mftip", "rh_mfmiddle", 12, 10),
        "mf": ("rh_mftip", "rh_mfmiddle", 12, 10),
        "ring": ("rh_rftip", "rh_rfmiddle", 16, 14),
        "rf": ("rh_rftip", "rh_rfmiddle", 16, 14),
        "pinky": ("rh_lftip", "rh_lfmiddle", 20, 18),
        "little": ("rh_lftip", "rh_lfmiddle", 20, 18),
        "lf": ("rh_lftip", "rh_lfmiddle", 20, 18),
    }
    if pinch_finger.lower() in {"all", "any"}:
        return ["index", "middle", "ring", "pinky"]

    fingers = []
    for raw_item in pinch_finger.split(","):
        key = raw_item.strip().lower()
        if not key:
            continue
        if key not in finger_specs:
            raise ValueError(
                f"Unknown pinch finger '{key}'. "
                "Use index, middle, ring, pinky, all, or comma-separated values."
            )
        canonical = {
            "ff": "index",
            "mfinger": "middle",
            "mf": "middle",
            "rf": "ring",
            "little": "pinky",
            "lf": "pinky",
        }.get(key, key)
        if canonical not in fingers:
            fingers.append(canonical)
    return fingers or ["index"]


def get_x2_pinch_vector_gain_defaults(pinch_finger: str) -> str:
    finger_links = {
        "index": ("rh_fftip", "rh_ffmiddle"),
        "middle": ("rh_mftip", "rh_mfmiddle"),
        "ring": ("rh_rftip", "rh_rfmiddle"),
        "pinky": ("rh_lftip", "rh_lfmiddle"),
    }
    parts = []
    for finger in parse_pinch_fingers(pinch_finger):
        tip_link, middle_link = finger_links[finger]
        parts.append(f"{tip_link}-rh_thtip=0.55")
        parts.append(f"{middle_link}-rh_thmiddle=0.65")
    return ",".join(parts)


def get_x2_pinch_axis_gain_defaults(pinch_finger: str) -> str:
    finger_links = {
        "index": ("rh_fftip", "rh_ffmiddle"),
        "middle": ("rh_mftip", "rh_mfmiddle"),
        "ring": ("rh_rftip", "rh_rfmiddle"),
        "pinky": ("rh_lftip", "rh_lfmiddle"),
    }
    parts = []
    for finger in parse_pinch_fingers(pinch_finger):
        tip_link, middle_link = finger_links[finger]
        parts.append(f"{tip_link}-rh_thtip:y=0.20")
        parts.append(f"{middle_link}-rh_thmiddle:y=0.35")
        parts.append(f"{tip_link}-rh_thtip:z=0.85")
        parts.append(f"{middle_link}-rh_thmiddle:z=0.80")
    return ",".join(parts)


def add_x2_pinch_constraints(config: RetargetingConfig, pinch_finger: str) -> list[str]:
    if config.type != "vector":
        raise ValueError("--pinch-mode only supports vector retargeting.")
    if config.target_origin_link_names is None or config.target_task_link_names is None:
        raise ValueError("--pinch-mode needs vector target link names.")
    if config.target_link_human_indices is None:
        raise ValueError("--pinch-mode needs target_link_human_indices.")

    finger_specs = {
        "index": ("rh_fftip", "rh_ffmiddle", 8, 6),
        "middle": ("rh_mftip", "rh_mfmiddle", 12, 10),
        "ring": ("rh_rftip", "rh_rfmiddle", 16, 14),
        "pinky": ("rh_lftip", "rh_lfmiddle", 20, 18),
    }

    existing = set(zip(config.target_origin_link_names, config.target_task_link_names))
    new_origin_links = []
    new_task_links = []
    new_human_origin = []
    new_human_task = []
    added = []
    for finger in parse_pinch_fingers(pinch_finger):
        tip_link, middle_link, tip_human_index, middle_human_index = finger_specs[finger]
        for origin_link, task_link, human_origin, human_task in [
            ("rh_thtip", tip_link, 4, tip_human_index),
            ("rh_thmiddle", middle_link, 3, middle_human_index),
        ]:
            if (origin_link, task_link) in existing:
                continue
            new_origin_links.append(origin_link)
            new_task_links.append(task_link)
            new_human_origin.append(human_origin)
            new_human_task.append(human_task)
            existing.add((origin_link, task_link))
            added.append(f"{task_link}-{origin_link}")

    if not new_origin_links:
        return []

    config.target_origin_link_names = config.target_origin_link_names + new_origin_links
    config.target_task_link_names = config.target_task_link_names + new_task_links
    pinch_indices = np.asarray([new_human_origin, new_human_task], dtype=np.int64)
    config.target_link_human_indices = np.concatenate(
        [config.target_link_human_indices, pinch_indices], axis=1
    )
    return added


def merge_joint_limit_overrides(
    retarget_joint_limit: Optional[str], extra_limits: list[str]
) -> Optional[str]:
    limit_items = list(extra_limits)
    if retarget_joint_limit is not None and retarget_joint_limit.strip():
        limit_items.append(retarget_joint_limit.strip())
    return ",".join(limit_items) if limit_items else None


def format_mujoco_qpos_by_finger(
    joint_names: list[str],
    qpos_addrs: np.ndarray,
    output_qpos: np.ndarray,
    data: mujoco.MjData,
    joint_limits: np.ndarray,
    limit_margin: float,
) -> str:
    groups = [
        ("FF", "_FF"),
        ("MF", "_MF"),
        ("RF", "_RF"),
        ("LF", "_LF"),
        ("TH", "_TH"),
    ]
    parts = []
    for label, token in groups:
        values = []
        for i, name in enumerate(joint_names):
            if token not in name:
                continue
            lower, upper = joint_limits[i]
            value = float(data.qpos[qpos_addrs[i]])
            requested = float(output_qpos[i])
            near_limit = min(value - lower, upper - value) <= limit_margin
            marker = "!" if near_limit else ""
            values.append(
                f"{name}@q{int(qpos_addrs[i])}:{value:+.3f}"
                f"/req{requested:+.3f}{marker}"
            )
        if values:
            parts.append(f"{label}[{', '.join(values)}]")
    return " ".join(parts)


def debug_print_joint_retargeting_alignment(
    frame_id: int,
    retargeting,
    retarget_ref_value: np.ndarray,
    output_qpos: np.ndarray,
    joint_names: list[str],
    qpos_addrs: np.ndarray,
    data: mujoco.MjData,
    joint_limits: np.ndarray,
    limit_margin: float,
    target_vector_axis_gains: Optional[np.ndarray] = None,
    x2_left_y_axis_gain: float = 1.0,
) -> None:
    optimizer = retargeting.optimizer
    print("\n[joint-retarget-debug]", f"frame={frame_id}", flush=True)
    try:
        objective = optimizer.opt.last_optimum_value()
        print(f"  optimizer objective: {objective:.6f}", flush=True)
    except Exception:
        pass
    print(
        "  mujoco qpos:",
        format_mujoco_qpos_by_finger(
            joint_names, qpos_addrs, output_qpos, data, joint_limits, limit_margin
        ),
        flush=True,
    )

    vectors = compute_retargeting_robot_vectors(retargeting, output_qpos)
    if vectors is None:
        print("  robot vector debug: only available for vector retargeting.", flush=True)
        return

    robot_vec, _, _ = vectors
    target_vec = retarget_ref_value * optimizer.scaling
    errors = robot_vec - target_vec
    axis_mae = np.mean(np.abs(errors), axis=0)
    worst_axis = ["x", "y", "z"][int(np.argmax(axis_mae))]
    origin_human_indices = np.asarray(optimizer.target_link_human_indices)[0]
    task_human_indices = np.asarray(optimizer.target_link_human_indices)[1]

    print(
        f"  vector scaling: {optimizer.scaling:.3f}; "
        "angle near 180 means the robot vector points opposite to the target.",
        flush=True,
    )
    print(
        "  vector axis mean abs error: "
        f"x={axis_mae[0]:.4f}, y={axis_mae[1]:.4f}, z={axis_mae[2]:.4f} "
        f"(worst={worst_axis})",
        flush=True,
    )
    print(
        "  vector_axis_error_x="
        f"{axis_mae[0]:.4f} "
        "vector_axis_error_y="
        f"{axis_mae[1]:.4f} "
        "vector_axis_error_z="
        f"{axis_mae[2]:.4f}",
        flush=True,
    )
    print(f"  x2_left_y_axis_gain={x2_left_y_axis_gain:.2f}", flush=True)
    if target_vector_axis_gains is not None:
        print(
            "  target_vector_axis_gain: "
            + format_target_vector_axis_gains(retargeting, target_vector_axis_gains),
            flush=True,
        )
    if worst_axis == "y" and axis_mae[1] > 0.05:
        print(
            "  hint: y-axis target is larger than the robot can reach. "
            "Try --target-vector-axis-gain y=0.5; if J3 still hits limits, "
            "lower it to y=0.4.",
            flush=True,
        )
    for i, (origin_link, task_link) in enumerate(
        zip(optimizer.origin_link_names, optimizer.task_link_names)
    ):
        target = target_vec[i]
        robot = robot_vec[i]
        error = errors[i]
        print(
            f"  {i:02d} {task_link}-{origin_link} <- "
            f"{human_landmark_label(int(task_human_indices[i]))} - "
            f"{human_landmark_label(int(origin_human_indices[i]))}: "
            f"target_norm={np.linalg.norm(target):.4f}, "
            f"robot_norm={np.linalg.norm(robot):.4f}, "
            f"err={np.linalg.norm(error):.4f}, "
            f"angle={vector_angle_deg(target, robot):.1f}deg",
            flush=True,
        )
        print(
            "     "
            f"target={np.array2string(target, precision=3, suppress_small=True)} "
            f"robot={np.array2string(robot, precision=3, suppress_small=True)} "
            f"diff={np.array2string(error, precision=3, suppress_small=True)}",
            flush=True,
        )


def start_retargeting_mujoco(
    queue: multiprocessing.Queue,
    error_queue: Optional[multiprocessing.Queue],
    robot_dir: str,
    config_path: str,
    mujoco_model_path: Optional[Path] = None,
    scaling_factor: Optional[float] = None,
    low_pass_alpha: Optional[float] = None,
    normal_delta: Optional[float] = None,
    robot_visual_scale: Optional[float] = None,
    joint_sign: Optional[str] = None,
    joint_gain: Optional[str] = None,
    retarget_joint_limit: Optional[str] = None,
    target_vector_gain: Optional[str] = None,
    target_vector_axis_gain: Optional[str] = None,
    pinch_mode: bool = False,
    pinch_finger: str = "index",
    thumb_natural_limits: bool = False,
    x2_retargeting_profile: str = "natural",
    x2_finger_direct_gain: Optional[str] = None,
    x2_finger_coupling_strength: float = 0.35,
    x2_smooth_alpha: float = X2_INTERNAL_QPOS_SMOOTH_ALPHA,
    x2_max_delta: float = X2_INTERNAL_MAX_DELTA,
    x2_debug_dir: Optional[Path] = None,
    x2_debug_save_png: bool = False,
    finger_map: Optional[str] = None,
    ref_transform: Optional[str] = None,
    debug_retargeting: bool = False,
    debug_interval: int = 15,
    debug_limit_margin: float = 0.03,
    reset_on_hand_lost: bool = True,
    lost_reset_frames: int = 5,
    root_joint_name: str = "mujoco_root_joint",
    root_pos: tuple[float, float, float] = (0.0, 0.0, -0.05),
    root_quat: tuple[float, float, float, float] = X2_ROOT_QUAT,
    camera_distance: float = 0.65,
    camera_azimuth: float = X2_RIGHT_CAMERA_AZIMUTH,
    camera_elevation: float = -5.0,
    viewer_backend: str = "cv2",
    glfw_context_api: str = "egl",
    render_width: int = 480,
    render_height: int = 480,
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
    user_retarget_joint_limit = retarget_joint_limit
    default_joint_limit_overrides: list[str] = []
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    x2_finger_direct_gains = parse_x2_finger_direct_gain(x2_finger_direct_gain)
    if config.target_link_human_indices is not None:
        config.target_link_human_indices = remap_human_indices(
            config.target_link_human_indices, finger_map
        )
    is_x2_config = is_x2_robot(Path(config.urdf_path).stem)
    x2_profile_active = is_x2_config and is_x2_profile_enabled(x2_retargeting_profile)
    x2_default_vector_gain = ""
    x2_default_axis_gain = ""
    if x2_profile_active:
        removed_thumb_objectives = remove_x2_thumb_hard_match_objectives(config)
        added_local_constraints = add_x2_local_shape_constraints(config)
        logger.warning(
            "x2 natural retargeting profile is active. "
            "It adds local finger constraints, x2-specific vector shaping, "
            "and natural joint limits."
        )
        if removed_thumb_objectives:
            logger.warning(
                "x2 profile removed thumb hard-match objectives: "
                + ", ".join(removed_thumb_objectives)
            )
        if added_local_constraints:
            logger.warning(
                "x2 profile added local finger constraints: "
                + ", ".join(added_local_constraints)
            )
        x2_default_vector_gain = X2_NATURAL_VECTOR_GAIN
        x2_default_axis_gain = X2_NATURAL_AXIS_GAIN
        default_joint_limit_overrides.extend(X2_NATURAL_JOINT_LIMITS)
    if pinch_mode:
        added_pinch_constraints = add_x2_pinch_constraints(config, pinch_finger)
        logger.warning(
            "--pinch-mode added thumb relative constraints: "
            + (", ".join(added_pinch_constraints) if added_pinch_constraints else "none")
        )
        if x2_profile_active:
            x2_default_vector_gain = merge_csv_overrides(
                x2_default_vector_gain,
                get_x2_pinch_vector_gain_defaults(pinch_finger),
            )
            x2_default_axis_gain = merge_csv_overrides(
                x2_default_axis_gain,
                get_x2_pinch_axis_gain_defaults(pinch_finger),
            )
    if x2_profile_active:
        target_vector_gain = merge_csv_overrides(
            x2_default_vector_gain, target_vector_gain
        )
        target_vector_axis_gain = merge_csv_overrides(
            x2_default_axis_gain, target_vector_axis_gain
        )
    retargeting = config.build()
    target_vector_gains = parse_target_vector_gains(target_vector_gain, retargeting)
    target_vector_axis_gains = parse_target_vector_axis_gains(
        target_vector_axis_gain, retargeting
    )

    detector = SingleHandDetector(hand_type=hand_type, selfie=False)
    x2_control_mode = get_x2_control_mode(hand_type)
    x2_left_output_mode = is_x2_config and x2_control_mode == "left_mode"
    x2_left_y_axis_gain = 1.0
    if (
        is_x2_config
        and x2_control_mode == "left_mode"
        and retargeting.optimizer.retargeting_type == "VECTOR"
    ):
        target_vector_axis_gains[:, 1] *= X2_LEFT_TARGET_VECTOR_Y_AXIS_GAIN
        x2_left_y_axis_gain = X2_LEFT_TARGET_VECTOR_Y_AXIS_GAIN

    model_path = resolve_mujoco_model_path(config.urdf_path, mujoco_model_path)
    robot_name = model_path.stem
    if ref_transform is None and is_x2_robot(robot_name):
        # X2 left/right share root pose; mode direction is expressed by qpos sign.
        ref_transform = "x2"
    if robot_visual_scale is not None:
        logger.warning(
            "--robot-visual-scale is ignored by the MuJoCo viewer. "
            "Use --camera-distance for display size, or regenerate scaled meshes."
        )

    logger.info(f"MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    set_free_root_pose(model, data, root_joint_name, root_pos, root_quat)

    retargeting_joint_names = retargeting.joint_names
    mujoco_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    qpos_addrs = build_qpos_addr_map(model, retargeting_joint_names)
    x2_joint_sweep_rows = (
        compute_x2_joint_sweep_rows(
            model,
            root_joint_name,
            root_pos,
            root_quat,
            retargeting_joint_names,
        )
        if is_x2_config
        else []
    )
    joint_signs = parse_joint_signs(joint_sign, retargeting_joint_names)
    joint_gains = parse_joint_gains(joint_gain, retargeting_joint_names)
    if thumb_natural_limits and not x2_profile_active:
        default_joint_limit_overrides.extend(
            ["rh_THJ3=-1.31:1.31", "rh_THJ2=-1.31:1.31", "rh_THJ1=-1.31:1.31"]
        )
    retarget_joint_limit = merge_joint_limit_overrides(
        user_retarget_joint_limit, default_joint_limit_overrides
    )
    joint_limits = apply_retarget_joint_limit_overrides(
        retargeting, retarget_joint_limit
    )
    output_joint_limits = get_signed_output_joint_limits(joint_limits, joint_signs)
    x2_four_finger_sign_enabled = is_x2_config
    if joint_sign is not None:
        logger.warning(
            "--joint-sign is an explicit post-retargeting output transform. "
            "It is not part of the default x2 left/right mode convention."
        )
    if joint_gain is not None:
        logger.warning(
            "--joint-gain is applied after optimization for visual tuning. "
            "If a gain causes near-limit joints, reduce the gain or adjust the retargeting config."
        )

    logger.info(f"Retargeting joints: {retargeting_joint_names}")
    logger.info(f"MuJoCo joints: {mujoco_joint_names}")
    logger.info(f"Retargeting -> MuJoCo qpos addrs: {qpos_addrs.tolist()}")
    logger.info(f"Finger map: {finger_map or 'default'}")
    logger.info(f"Reference transform: {ref_transform or 'none'}")
    if is_x2_config:
        logger.info(
            "x2 simulation palm normal: "
            f"{x2_palm_normal_label(root_quat)} "
            f"(root_quat={tuple(float(v) for v in root_quat)}, "
            f"camera_azimuth={camera_azimuth:.1f})"
        )
    logger.info(
        "x2 retargeting profile: "
        + (x2_retargeting_profile if x2_profile_active else "off")
    )
    if is_x2_config:
        logger.info(f"x2 control mode: {x2_control_mode} (single rh_* URDF)")
        logger.info(
            "x2 control convention:\n"
            "  * use_same_root_pose: True\n"
            "  * use_left_root_mirror: False\n"
            "  * right_mode_finger_sign: +1\n"
            "  * left_mode_finger_sign: -1\n"
            f"  * left_adapter_enabled: {x2_control_mode == 'left_mode'}\n"
            "  * x2_flip_yz_enabled: False\n"
            "  * auto_command_sign_flip: False\n"
            f"  * joint_sign_arg_enabled: {joint_sign is not None}"
        )
        logger.info(
            "x2 landmark canonicalization: "
            + format_x2_landmark_canonicalization(
                {
                    "enabled": True,
                    "source": "palm_frame",
                    "palm_frame": True,
                    "left_adapter": x2_control_mode == "left_mode",
                    "adapter": tuple(float(value) for value in X2_LEFT_PALM_FRAME_ADAPTER),
                    "semantic_map": X2_LANDMARK_SEMANTIC_MAP,
                }
            )
        )
        logger.info(
            "x2 four-finger mode sign: "
            f"{get_x2_four_finger_command_sign(x2_control_mode):+.0f}"
        )
        logger.info(
            f"x2 command smoothing: alpha={x2_smooth_alpha:.2f}, "
            f"max_delta={x2_max_delta:.3f} rad/frame"
        )
        logger.info(
            "x2 joint sweep four fingers (print only, no sign correction): "
            + format_x2_four_finger_joint_sweep(x2_joint_sweep_rows)
        )
        logger.info(
            "x2 thumb joint sweep (print only, same root for both modes): "
            + format_x2_thumb_joint_sweep(x2_joint_sweep_rows)
        )
        if x2_profile_active and x2_control_mode == "left_mode":
            logger.info(
                "x2 left mode uses the common x2 natural joint limits; "
                "no image/example-specific left limits are applied."
            )
        if x2_profile_active:
            logger.info(
                "x2 finger direct gain scale: "
                + format_x2_finger_direct_gain(x2_finger_direct_gains)
            )
        if x2_profile_active:
            logger.info(
                f"x2 finger coupling: strength={x2_finger_coupling_strength:.2f}, "
                f"pip={X2_FINGER_COUPLING_PIP_PER_MCP:.2f}*mcp, "
                f"dip={X2_FINGER_COUPLING_DIP_PER_PIP:.2f}*pip"
            )
        logger.info(f"x2 left y-axis target vector gain: {x2_left_y_axis_gain:.2f}")
        if x2_debug_dir is not None:
            logger.info(
                f"x2 debug artifacts: {x2_debug_dir} "
                f"(png={'on' if x2_debug_save_png else 'off'})"
            )
    if is_x2_config:
        logger.info("x2 command mode: direct qpos")
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
    logger.info(
        "Target vector axis gains: "
        + format_target_vector_axis_gains(retargeting, target_vector_axis_gains)
    )

    reset_retargeting_pose_mujoco(
        retargeting,
        data,
        qpos_addrs,
        joint_signs,
        joint_gains,
        joint_limits,
        output_joint_limits=output_joint_limits,
        zero_reset=x2_profile_active,
    )
    mujoco.mj_forward(model, data)

    frame_id = 0
    lost_frames = 0
    x2_thumb_direct_state = X2ThumbDirectMappingState()
    x2_finger_direct_state = X2FingerDirectMappingState()
    x2_command_smoother = X2CommandSmoother(
        alpha=x2_smooth_alpha,
        max_delta=x2_max_delta,
        deadband_mask=(
            build_x2_four_finger_deadband_mask(retargeting_joint_names)
            if is_x2_config
            else None
        ),
    )
    x2_debug_recorder = X2DebugRecorder(
        Path(x2_debug_dir) if x2_debug_dir is not None else None,
        retargeting_joint_names,
        output_joint_limits,
        save_png=x2_debug_save_png,
    )
    if is_x2_config:
        x2_debug_recorder.write_joint_sweep(
            model,
            root_joint_name,
            root_pos,
            root_quat,
            rows=x2_joint_sweep_rows,
        )

    def process_frame() -> bool:
        nonlocal frame_id, lost_frames
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            if error_queue is not None and not error_queue.empty():
                logger.error(f"Camera producer failed: {error_queue.get()}")
                return False
            logger.error(
                "Fail to fetch image from camera in 5 secs. "
                "Please check your camera device."
            )
            return False

        _, joint_pos, keypoint_2d, _, detection_meta = detector.detect(
            rgb, return_meta=True
        )
        joint_pos, x2_landmark_info = canonicalize_x2_landmarks_for_control_mode(
            joint_pos,
            x2_control_mode,
            is_x2_config,
            detection_meta,
        )
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

        set_free_root_pose(model, data, root_joint_name, root_pos, root_quat)
        if joint_pos is None:
            lost_frames += 1
            logger.warning(f"{hand_type} hand is not detected.")
            if x2_profile_active:
                x2_thumb_direct_state.mark_hand_lost()
            if reset_on_hand_lost and lost_frames == max(lost_reset_frames, 1):
                reset_retargeting_pose_mujoco(
                    retargeting,
                    data,
                    qpos_addrs,
                    joint_signs,
                    joint_gains,
                    joint_limits,
                    output_joint_limits=output_joint_limits,
                    zero_reset=x2_profile_active,
                )
                logger.warning(
                    f"{hand_type} hand was lost for {lost_frames} frames. "
                    "Retargeting state has been reset."
                )
                x2_thumb_direct_state.reset()
                x2_finger_direct_state.reset()
                x2_command_smoother.reset()
        else:
            lost_frames = 0
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            raw_ref_value = ref_value.copy()
            ref_value = transform_ref_value(ref_value, ref_transform)
            ref_value = apply_target_vector_gains(ref_value, target_vector_gains)
            ref_value = apply_target_vector_axis_gains(
                ref_value, target_vector_axis_gains
            )
            ref_value = apply_x2_natural_ref_shape(
                ref_value, retargeting, x2_retargeting_profile if is_x2_config else None
            )
            x2_pinch_gate_info = (
                compute_x2_pinch_gate_info(joint_pos, pinch_finger)
                if x2_profile_active and pinch_mode
                else {}
            )
            x2_pinch_strength = (
                compute_x2_pinch_strength(
                    retargeting,
                    raw_ref_value,
                    pinch_finger,
                    x2_retargeting_profile,
                    x2_pinch_gate_info,
                )
                if x2_profile_active and pinch_mode
                else 0.0
            )
            x2_thumb_direct_info = (
                x2_thumb_direct_state.update(
                    joint_pos,
                    x2_retargeting_profile,
                    retargeting_joint_names,
                    joint_limits,
                    x2_control_mode,
                )
                if x2_profile_active
                else {"enabled": False}
            )
            x2_finger_direct_info = (
                x2_finger_direct_state.update(
                    joint_pos,
                    x2_retargeting_profile,
                    retargeting_joint_names,
                    joint_limits,
                    x2_control_mode,
                    x2_finger_direct_gains,
                )
                if x2_profile_active
                else {"enabled": False}
            )
            qpos = retargeting.retarget(ref_value)
            profiled_qpos = apply_x2_finger_direct_mapping(
                qpos,
                retargeting_joint_names,
                x2_retargeting_profile if is_x2_config else None,
                x2_finger_direct_info,
            )
            pre_sign_profiled_qpos = profiled_qpos.copy()
            profiled_qpos = apply_x2_four_finger_control_sign(
                profiled_qpos,
                retargeting_joint_names,
                x2_four_finger_sign_enabled,
                x2_control_mode,
            )
            profiled_qpos, x2_finger_coupling_info = apply_x2_four_finger_coupling(
                profiled_qpos,
                retargeting_joint_names,
                joint_limits,
                x2_retargeting_profile if is_x2_config else None,
                x2_finger_coupling_strength,
            )
            profiled_qpos = apply_x2_thumb_direct_mapping(
                profiled_qpos,
                retargeting_joint_names,
                x2_retargeting_profile if is_x2_config else None,
                x2_thumb_direct_info,
                x2_control_mode,
                retargeting=retargeting,
                target_ref_value=ref_value,
                joint_limits=joint_limits,
            )
            command_qpos = apply_joint_output_transform(
                profiled_qpos,
                joint_signs,
                joint_gains,
                output_joint_limits,
            )
            command_qpos = np.clip(
                command_qpos, output_joint_limits[:, 0], output_joint_limits[:, 1]
            )
            command_qpos_before_smoothing = command_qpos.copy()
            if x2_profile_active:
                if x2_thumb_direct_info.get("thumb_feature_ema_reset", False):
                    reset_x2_thumb_smoother_state_to_command(
                        x2_command_smoother,
                        command_qpos,
                        retargeting_joint_names,
                    )
                command_qpos, x2_smoothing_info = x2_command_smoother.update(
                    command_qpos
                )
                command_qpos = enforce_x2_thj4_gate_after_smoothing(
                    command_qpos,
                    retargeting_joint_names,
                    x2_thumb_direct_info,
                    x2_command_smoother,
                )
            else:
                x2_smoothing_info = {"enabled": False}
            model_qpos = command_qpos
            if x2_profile_active:
                x2_thumb_direct_info["selected_thumb_qpos_after_smoothing"] = (
                    x2_thumb_qpos_map(command_qpos, retargeting_joint_names)
                )
                x2_thumb_direct_info["final_mujoco_thumb_qpos"] = x2_thumb_qpos_map(
                    model_qpos, retargeting_joint_names
                )
                x2_thumb_direct_info["selected_thumb_qpos_before_output_smoothing"] = (
                    x2_thumb_qpos_map(command_qpos_before_smoothing, retargeting_joint_names)
                )
            if x2_profile_active:
                state_qpos = np.clip(
                    profiled_qpos, joint_limits[:, 0], joint_limits[:, 1]
                )
                retargeting.set_qpos(state_qpos)
            data.qpos[qpos_addrs] = model_qpos
            mujoco.mj_forward(model, data)
            if debug_retargeting and frame_id % max(debug_interval, 1) == 0:
                debug_print_retargeting(
                    frame_id,
                    joint_pos,
                    raw_ref_value,
                    ref_value,
                    model_qpos,
                    command_qpos,
                    retargeting_joint_names,
                    joint_limits,
                    debug_limit_margin,
                    signed_joint_limits=output_joint_limits,
                )
                debug_print_joint_retargeting_alignment(
                    frame_id,
                    retargeting,
                    ref_value,
                    model_qpos,
                    retargeting_joint_names,
                    qpos_addrs,
                    data,
                    output_joint_limits,
                    debug_limit_margin,
                    target_vector_axis_gains=target_vector_axis_gains,
                    x2_left_y_axis_gain=x2_left_y_axis_gain,
                )
                x2_world_dx = {}
                if is_x2_config:
                    x2_world_dx = compute_x2_world_finger_tip_dx(
                        model, data, qpos_addrs
                    )
                    bend_line, wrong_fingers = format_x2_world_bend_direction(
                        x2_control_mode,
                        x2_world_dx,
                        retargeting_joint_names,
                        model_qpos,
                    )
                    print("  x2 world bend direction:", bend_line, flush=True)
                    if wrong_fingers:
                        expected_axis = (
                            "left command bends toward world -X"
                            if x2_control_mode == "left_mode"
                            else "right command bends toward world +X"
                        )
                        print(
                            "  warning: x2 "
                            f"{x2_control_mode} finger(s) not matching "
                            f"{expected_axis}: "
                            + ", ".join(wrong_fingers),
                            flush=True,
                        )
                if x2_left_output_mode:
                    print(
                        "  left solver qpos:",
                        format_qpos_by_finger(
                            retargeting_joint_names, pre_sign_profiled_qpos
                        ),
                        flush=True,
                    )
                    print(
                        "  left command qpos:",
                        format_qpos_by_finger(retargeting_joint_names, command_qpos),
                        flush=True,
                    )
                if x2_four_finger_sign_enabled:
                    print(
                        "  x2 four-finger mode sign:",
                        format_x2_four_finger_sign_changes(
                            pre_sign_profiled_qpos,
                            profiled_qpos,
                            retargeting_joint_names,
                        ),
                        flush=True,
                    )
                if x2_profile_active and pinch_mode:
                    print(
                        f"  x2 thumb pinch strength: {x2_pinch_strength:.3f}",
                        flush=True,
                    )
                    print(
                        "  x2 pinch curl gate: "
                        + format_x2_pinch_gate_info(x2_pinch_gate_info),
                        flush=True,
                    )
                if x2_profile_active:
                    print(
                        "  x2 landmark canonicalization: "
                        + format_x2_landmark_canonicalization(x2_landmark_info),
                        flush=True,
                    )
                    print(
                        "  x2 finger direct mapping: "
                        + format_x2_finger_direct_mapping(x2_finger_direct_info),
                        flush=True,
                    )
                    print(
                        "  x2 finger coupling: "
                        + format_x2_four_finger_coupling(x2_finger_coupling_info),
                        flush=True,
                    )
                    print(
                        "  x2 command smoothing: "
                        + format_x2_command_smoothing(x2_smoothing_info),
                        flush=True,
                    )
                    print(
                        "  x2 thumb direct mapping: "
                        + format_x2_thumb_direct_mapping(x2_thumb_direct_info),
                        flush=True,
                    )
                    if (
                        x2_thumb_direct_info.get("calibrated")
                        and max(
                            x2_thumb_direct_info["control_amount"].get("rh_THJ2", 0.0),
                            x2_thumb_direct_info["control_amount"].get("rh_THJ1", 0.0),
                        )
                        > 8.0
                        and max(
                            abs(x2_thumb_direct_info["direct_qpos"].get("rh_THJ2", 0.0)),
                            abs(x2_thumb_direct_info["direct_qpos"].get("rh_THJ1", 0.0)),
                        )
                        < 0.03
                    ):
                        print(
                            "  hint: MediaPipe thumb flexion changed, but THJ1/THJ2 "
                            "direct qpos stayed near zero. Check calibration baseline "
                            "or joint limits.",
                            flush=True,
                        )
                    clipped_joints = [
                        name
                        for name, clipped in x2_thumb_direct_info.get("clipped", {}).items()
                        if clipped
                    ]
                    if clipped_joints:
                        print(
                            "  hint: thumb direct qpos clipped by limits: "
                            + ", ".join(clipped_joints),
                            flush=True,
                        )
                    x2_debug_recorder.record_frame(
                        frame_id,
                        str(detection_meta.get("handedness", hand_type)),
                        x2_control_mode,
                        detection_meta,
                        joint_pos,
                        raw_ref_value,
                        ref_value,
                        command_qpos,
                        x2_smoothing_info,
                        x2_world_dx,
                    )
            frame_id += 1

        mujoco.mj_forward(model, data)
        return True

    viewer_backend = viewer_backend.lower()
    if viewer_backend in {"glfw", "mujoco"}:
        with launch_mujoco_viewer(model, data, glfw_context_api) as viewer:
            viewer.cam.distance = camera_distance
            # X2 left/right modes share the same root orientation.
            viewer.cam.azimuth = camera_azimuth
            viewer.cam.elevation = camera_elevation

            while viewer.is_running():
                if not process_frame():
                    break
                viewer.sync()
                time.sleep(0.001)
    elif viewer_backend in {"cv2", "opencv", "egl"}:
        if render_width <= 0 or render_height <= 0:
            raise ValueError("--render-width and --render-height must be positive.")
        logger.info(
            "Using OpenCV/EGL MuJoCo renderer. "
            "This avoids GLFW/GLX window creation. "
            f"Render size: {render_width}x{render_height}."
        )
        renderer = mujoco.Renderer(model, height=render_height, width=render_width)
        render_camera = mujoco.MjvCamera()
        render_camera.lookat[:] = np.asarray(root_pos, dtype=np.float64)
        render_camera.distance = camera_distance
        render_camera.azimuth = camera_azimuth
        render_camera.elevation = camera_elevation
        try:
            while True:
                if not process_frame():
                    break
                renderer.update_scene(data, camera=render_camera)
                rgb_render = renderer.render()
                cv2.imshow(
                    "mujoco_retargeting",
                    cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.001)
        finally:
            if hasattr(renderer, "close"):
                renderer.close()
    else:
        raise ValueError(
            f"Unknown viewer_backend '{viewer_backend}'. Use cv2 or glfw."
        )

    cv2.destroyAllWindows()


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    mujoco_model_path: Optional[Path] = None,
    scaling_factor: Optional[float] = None,
    low_pass_alpha: Optional[float] = None,
    normal_delta: Optional[float] = None,
    robot_visual_scale: Optional[float] = None,
    joint_sign: Optional[str] = None,
    joint_gain: Optional[str] = None,
    retarget_joint_limit: Optional[str] = None,
    target_vector_gain: Optional[str] = None,
    target_vector_axis_gain: Optional[str] = None,
    pinch_mode: bool = False,
    pinch_finger: str = "index",
    thumb_natural_limits: bool = False,
    x2_retargeting_profile: str = "natural",
    x2_finger_direct_gain: Optional[str] = None,
    x2_finger_coupling_strength: float = 0.35,
    x2_smooth_alpha: float = X2_INTERNAL_QPOS_SMOOTH_ALPHA,
    x2_max_delta: float = X2_INTERNAL_MAX_DELTA,
    x2_debug_dir: Optional[Path] = None,
    x2_debug_save_png: bool = False,
    finger_map: Optional[str] = None,
    ref_transform: Optional[str] = None,
    debug_retargeting: bool = False,
    debug_interval: int = 15,
    debug_limit_margin: float = 0.03,
    reset_on_hand_lost: bool = True,
    lost_reset_frames: int = 5,
    root_pos: tuple[float, float, float] = (0.0, 0.0, -0.05),
    root_quat: tuple[float, float, float, float] = X2_ROOT_QUAT,
    camera_distance: float = 0.65,
    camera_azimuth: float = X2_RIGHT_CAMERA_AZIMUTH,
    camera_elevation: float = -5.0,
    viewer_backend: str = "cv2",
    glfw_context_api: str = "egl",
    render_width: int = 480,
    render_height: int = 480,
):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    if config_path is None:
        raise ValueError(
            f"No default config found for {robot_name}-{retargeting_type}-{hand_type}."
        )
    robot_dir = get_robot_dir()
    queue = multiprocessing.Queue(maxsize=1)
    error_queue = multiprocessing.Queue(maxsize=1)
    producer = multiprocessing.Process(
        target=produce_frame_with_error_report,
        args=(queue, error_queue, camera_path),
    )
    producer.start()
    try:
        check_camera_process_startup(producer, error_queue)
        start_retargeting_mujoco(
            queue,
            error_queue,
            str(robot_dir),
            str(config_path),
            mujoco_model_path=mujoco_model_path,
            scaling_factor=scaling_factor,
            low_pass_alpha=low_pass_alpha,
            normal_delta=normal_delta,
            robot_visual_scale=robot_visual_scale,
            joint_sign=joint_sign,
            joint_gain=joint_gain,
            retarget_joint_limit=retarget_joint_limit,
            target_vector_gain=target_vector_gain,
            target_vector_axis_gain=target_vector_axis_gain,
            pinch_mode=pinch_mode,
            pinch_finger=pinch_finger,
            thumb_natural_limits=thumb_natural_limits,
            x2_retargeting_profile=x2_retargeting_profile,
            x2_finger_direct_gain=x2_finger_direct_gain,
            x2_finger_coupling_strength=x2_finger_coupling_strength,
            x2_smooth_alpha=x2_smooth_alpha,
            x2_max_delta=x2_max_delta,
            x2_debug_dir=x2_debug_dir,
            x2_debug_save_png=x2_debug_save_png,
            finger_map=finger_map,
            ref_transform=ref_transform,
            debug_retargeting=debug_retargeting,
            debug_interval=debug_interval,
            debug_limit_margin=debug_limit_margin,
            reset_on_hand_lost=reset_on_hand_lost,
            lost_reset_frames=lost_reset_frames,
            root_pos=root_pos,
            root_quat=root_quat,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            viewer_backend=viewer_backend,
            glfw_context_api=glfw_context_api,
            render_width=render_width,
            render_height=render_height,
        )
    finally:
        producer.terminate()
        producer.join()


if __name__ == "__main__":
    tyro.cli(main)

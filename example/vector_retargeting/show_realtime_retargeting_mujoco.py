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
    apply_x2_palm_flex_guard,
    canonicalize_x2_landmarks_for_control_mode,
    debug_print_retargeting,
    format_x2_landmark_canonicalization,
    format_x2_palm_flex_guard_changes,
    format_link_human_mapping,
    format_qpos_by_finger,
    format_target_vector_gains,
    get_x2_control_mode,
    get_robot_dir,
    human_landmark_label,
    is_x2_robot,
    enable_joint_sign_kinematics,
    parse_joint_gains,
    parse_joint_signs,
    parse_target_vector_gains,
    produce_frame,
    remap_human_indices,
    set_retargeting_optimizer_limits,
    transform_ref_value,
    X2_LANDMARK_SEMANTIC_MAP,
    X2_LEFT_JOINT_SIGN,
    X2_LEFT_LANDMARK_MIRROR,
)


X2_RIGHT_ROOT_QUAT = (0.5, -0.5, -0.5, -0.5)
X2_LEFT_ROOT_QUAT = (0.5, 0.5, -0.5, 0.5)
X2_RIGHT_CAMERA_AZIMUTH = 180.0
X2_LEFT_CAMERA_AZIMUTH = 0.0


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
    if is_close_tuple(root_quat, X2_RIGHT_ROOT_QUAT):
        return "world +X"
    if is_close_tuple(root_quat, X2_LEFT_ROOT_QUAT):
        return "world -X"
    return "custom"


X2_FOUR_FINGER_DISTAL_BODIES = {
    "FF": "rh_ffdistal",
    "MF": "rh_mfdistal",
    "RF": "rh_rfdistal",
    "LF": "rh_lfdistal",
}
X2_FINGER_TIP_LOCAL_OFFSET = np.asarray([0.034, 0.0, 0.0], dtype=np.float64)


def x2_finger_tip_world_position(
    model: mujoco.MjModel, data: mujoco.MjData, distal_body_name: str
) -> Optional[np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, distal_body_name)
    if body_id < 0:
        return None
    rotation = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + rotation @ X2_FINGER_TIP_LOCAL_OFFSET


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


def infer_x2_world_bend_command_signs(
    model: mujoco.MjModel,
    root_joint_name: str,
    root_pos: tuple[float, float, float],
    root_quat: tuple[float, float, float, float],
    control_mode: str,
    joint_names: list[str],
    probe_qpos: float = 0.5,
) -> tuple[dict[str, float], dict[str, dict]]:
    expected_sign = -1.0 if control_mode == "left_mode" else 1.0
    data = mujoco.MjData(model)
    signs = {finger: 1.0 for finger in X2_FOUR_FINGER_DISTAL_BODIES}
    diagnostics = {}

    for finger, distal_body_name in X2_FOUR_FINGER_DISTAL_BODIES.items():
        token = f"_{finger}J"
        candidates = [name for name in joint_names if token in name]
        preferred = [name for name in candidates if name.endswith("J3")]
        joint_name = (preferred or candidates or [None])[0]
        if joint_name is None:
            continue

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qaddr = model.jnt_qposadr[joint_id]

        data.qpos[:] = 0.0
        set_free_root_pose(model, data, root_joint_name, root_pos, root_quat)
        mujoco.mj_forward(model, data)
        neutral_tip = x2_finger_tip_world_position(model, data, distal_body_name)
        if neutral_tip is None:
            continue

        data.qpos[qaddr] = probe_qpos
        mujoco.mj_forward(model, data)
        probe_tip = x2_finger_tip_world_position(model, data, distal_body_name)
        if probe_tip is None:
            continue

        dx = float(probe_tip[0] - neutral_tip[0])
        signs[finger] = 1.0 if expected_sign * dx > 1e-6 else -1.0
        diagnostics[finger] = {
            "joint": joint_name,
            "probe_qpos": probe_qpos,
            "dx": dx,
            "command_sign": signs[finger],
        }
    return signs, diagnostics


def apply_x2_world_bend_command_signs(
    qpos: np.ndarray,
    joint_names: list[str],
    command_signs: dict[str, float],
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return qpos

    output = qpos.copy()
    for index, joint_name in enumerate(joint_names):
        for finger, sign in command_signs.items():
            if f"_{finger}J" in joint_name:
                output[index] *= sign
                break
    return output


def format_x2_world_bend_command_signs(
    diagnostics: dict[str, dict],
    control_mode: str,
) -> str:
    if not diagnostics:
        return "unavailable"
    expected = "dx<0" if control_mode == "left_mode" else "dx>0"
    return "; ".join(
        f"{finger}:{values.get('joint')} +{values.get('probe_qpos', 0.0):.2f}"
        f" -> dx={values.get('dx', 0.0):+.4f},"
        f"cmd_sign={values.get('command_sign', 1.0):+.0f}"
        for finger, values in diagnostics.items()
    ) + f" (expected {expected})"


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


class X2CommandSmoother:
    def __init__(self, alpha: float = 0.65, max_delta: float = 0.18) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_delta = float(max_delta)
        self.previous: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous = None

    def update(self, qpos: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.previous is None:
            self.previous = qpos.copy()
            return qpos, {
                "enabled": True,
                "initialized": True,
                "alpha": self.alpha,
                "max_delta": self.max_delta,
                "dq_max_abs": 0.0,
                "clipped": False,
            }

        blended = self.alpha * qpos + (1.0 - self.alpha) * self.previous
        delta = blended - self.previous
        if self.max_delta > 0:
            clipped_delta = np.clip(delta, -self.max_delta, self.max_delta)
        else:
            clipped_delta = delta
        smoothed = self.previous + clipped_delta
        self.previous = smoothed.copy()
        return smoothed, {
            "enabled": True,
            "initialized": False,
            "alpha": self.alpha,
            "max_delta": self.max_delta,
            "dq_max_abs": float(np.max(np.abs(clipped_delta))) if len(clipped_delta) else 0.0,
            "clipped": bool(not np.allclose(delta, clipped_delta)),
        }


def format_x2_command_smoothing(info: dict) -> str:
    if not info.get("enabled"):
        return "off"
    return (
        f"alpha={info.get('alpha', 0.0):.2f},"
        f"max_delta={info.get('max_delta', 0.0):.3f},"
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
    ) -> None:
        if not self.enabled:
            return
        path = self.debug_dir / "joint_sweep.csv"
        data = mujoco.MjData(model)
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        if root_id < 0:
            return
        root_addr = model.jnt_qposadr[root_id]
        joint_to_body = {
            "rh_FFJ": "rh_ffdistal",
            "rh_MFJ": "rh_mfdistal",
            "rh_RFJ": "rh_rfdistal",
            "rh_LFJ": "rh_lfdistal",
            "rh_THJ": "rh_thdistal",
        }

        def set_root(quat):
            data.qpos[:] = 0.0
            data.qpos[root_addr : root_addr + 3] = np.asarray(root_pos, dtype=np.float64)
            data.qpos[root_addr + 3 : root_addr + 7] = quat

        def tip_position(body_name: str) -> Optional[np.ndarray]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                return None
            rotation = data.xmat[body_id].reshape(3, 3)
            return data.xpos[body_id] + rotation @ X2_FINGER_TIP_LOCAL_OFFSET

        rows = []
        for mode, quat in [
            ("right_mode", X2_RIGHT_ROOT_QUAT),
            ("left_mode", X2_LEFT_ROOT_QUAT),
        ]:
            set_root(quat)
            mujoco.mj_forward(model, data)
            neutral = {}
            for body_name in set(joint_to_body.values()):
                neutral[body_name] = tip_position(body_name)
            for joint_name in self.joint_names:
                body_name = next(
                    (body for prefix, body in joint_to_body.items() if prefix in joint_name),
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
                    set_root(quat)
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
    "rh_thbase-rh_palm=0.18,"
    "rh_thproximal-rh_palm=0.45,"
    "rh_thproximal-rh_thbase=0.70,"
    "rh_thmiddle-rh_thproximal=1.05,"
    "rh_thtip-rh_thmiddle=1.10,"
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
    "rh_thbase-rh_palm:x=0.18,"
    "rh_thbase-rh_palm:y=0.18,"
    "rh_thbase-rh_palm:z=0.18,"
    "rh_thproximal-rh_thbase:x=0.60,"
    "rh_thproximal-rh_thbase:y=0.95,"
    "rh_thproximal-rh_thbase:z=0.90,"
    "rh_thmiddle-rh_thproximal:x=0.80,"
    "rh_thmiddle-rh_thproximal:y=1.00,"
    "rh_thmiddle-rh_thproximal:z=0.95,"
    "rh_thtip-rh_thmiddle:x=0.80,"
    "rh_thtip-rh_thmiddle:y=1.00,"
    "rh_thtip-rh_thmiddle:z=0.95,"
    "rh_thtip-rh_palm:y=0.70,"
    "rh_thmiddle-rh_palm:y=0.70,"
    "rh_fftip-rh_palm:y=0.50,"
    "rh_ffmiddle-rh_palm:y=0.45,"
    "rh_mftip-rh_palm:y=0.48,"
    "rh_mfmiddle-rh_palm:y=0.43,"
    "rh_rftip-rh_palm:y=0.45,"
    "rh_rfmiddle-rh_palm:y=0.40,"
    "rh_lftip-rh_palm:y=0.42,"
    "rh_lfmiddle-rh_palm:y=0.38,"
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
    "rh_THJ3=-0.95:0.12",
    "rh_THJ2=-1.31:0.00",
    "rh_THJ1=-1.31:0.00",
]

X2_RIGHT_THUMB_PINCH_TARGET_QPOS = {
    "rh_THJ4": np.deg2rad(66.0),
    "rh_THJ3": np.deg2rad(-8.0),
    "rh_THJ2": np.deg2rad(-42.0),
    "rh_THJ1": np.deg2rad(-26.0),
}
X2_LEFT_THUMB_PINCH_TARGET_QPOS = {
    "rh_THJ4": np.deg2rad(-66.0),
    "rh_THJ3": np.deg2rad(-8.0),
    "rh_THJ2": np.deg2rad(-42.0),
    "rh_THJ1": np.deg2rad(-26.0),
}
X2_PINCH_ACTIVATION_NEAR = 0.055
X2_PINCH_ACTIVATION_FAR = 0.12
X2_FINGER_STRAIGHT_CLOSED_RATIO = 1.08
X2_FINGER_STRAIGHT_OPEN_RATIO = 1.22
X2_THUMB_STRAIGHT_CLOSED_RATIO = 1.25
X2_THUMB_STRAIGHT_OPEN_RATIO = 1.60
X2_THUMB_DIRECT_JOINTS = ("rh_THJ3", "rh_THJ2", "rh_THJ1")
X2_THUMB_DIRECT_BASELINE_FRAMES = 12
X2_THUMB_DIRECT_DEADZONE_DEG = {
    "rh_THJ3": 1.0,
    "rh_THJ2": 2.0,
    "rh_THJ1": 1.5,
}
X2_THUMB_DIRECT_GAIN_RAD_PER_DEG = {
    "rh_THJ3": np.deg2rad(0.85),
    "rh_THJ2": np.deg2rad(0.70),
    "rh_THJ1": np.deg2rad(0.75),
}
X2_THUMB_DIRECT_BLEND = {
    "rh_THJ3": 1.0,
    "rh_THJ2": 1.0,
    "rh_THJ1": 1.0,
}
X2_THUMB_ROOT_PALM_PROJECTION_SCALE = 0.45
X2_FINGER_LANDMARKS = {
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
X2_FINGER_JOINT_TOKENS = {
    "index": "_FF",
    "middle": "_MF",
    "ring": "_RF",
    "pinky": "_LF",
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
X2_FINGER_DIRECT_GAIN_RAD_PER_DEG = {
    "mcp": np.deg2rad(0.95),
    "pip": np.deg2rad(1.10),
    "dip": np.deg2rad(1.00),
}
X2_FINGER_DIRECT_SMOOTHING = 0.65
X2_FINGER_COUPLING_PIP_PER_MCP = 0.8
X2_FINGER_COUPLING_DIP_PER_PIP = 0.5


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


X2_PALM_VECTOR_MIN_X = {
    "rh_thtip-rh_palm": 0.082,
    "rh_fftip-rh_palm": 0.092,
    "rh_mftip-rh_palm": 0.096,
    "rh_rftip-rh_palm": 0.092,
    "rh_lftip-rh_palm": 0.086,
    "rh_thmiddle-rh_palm": 0.075,
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
    "rh_thmiddle-rh_palm": 0.095,
    "rh_ffmiddle-rh_palm": 0.120,
    "rh_mfmiddle-rh_palm": 0.122,
    "rh_rfmiddle-rh_palm": 0.120,
    "rh_lfmiddle-rh_palm": 0.116,
}

X2_PALM_VECTOR_MAX_Z = {
    "rh_thtip-rh_palm": 0.045,
    "rh_thmiddle-rh_palm": 0.040,
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
        ("rh_palm", "rh_thbase", 0, 1),
        ("rh_thbase", "rh_thproximal", 1, 2),
        ("rh_thproximal", "rh_thmiddle", 2, 3),
        ("rh_thmiddle", "rh_thtip", 3, 4),
        ("rh_palm", "rh_thproximal", 0, 2),
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


def get_x2_thumb_pinch_target_qpos(hand_type: str) -> dict[str, float]:
    if hand_type == "Left":
        return X2_LEFT_THUMB_PINCH_TARGET_QPOS
    return X2_RIGHT_THUMB_PINCH_TARGET_QPOS


def format_x2_thumb_pinch_target_qpos(target_qpos: dict[str, float]) -> str:
    return ", ".join(
        f"{joint_name}={np.rad2deg(value):+.1f}deg"
        for joint_name, value in target_qpos.items()
    )


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


def compute_x2_thumb_straightness(joint_pos: Optional[np.ndarray]) -> dict:
    if joint_pos is None or len(joint_pos) <= 4:
        return {}

    wrist = joint_pos[0]
    ip_norm = float(np.linalg.norm(joint_pos[3] - wrist))
    tip_norm = float(np.linalg.norm(joint_pos[4] - wrist))
    ratio = tip_norm / max(ip_norm, 1e-6)
    straightness = ratio_to_unit_interval(
        ratio,
        X2_THUMB_STRAIGHT_CLOSED_RATIO,
        X2_THUMB_STRAIGHT_OPEN_RATIO,
    )
    return {"ratio": ratio, "straightness": straightness}


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


def apply_x2_open_hand_relaxation(
    qpos: np.ndarray,
    joint_names: list[str],
    profile: Optional[str],
    joint_pos: Optional[np.ndarray],
    pinch_strength: float,
) -> tuple[np.ndarray, dict]:
    if not is_x2_profile_enabled(profile) or joint_pos is None:
        return qpos, {"enabled": False}

    output = qpos.copy()
    finger_info = compute_x2_finger_straightness(joint_pos)
    for finger, values in finger_info.items():
        token = X2_FINGER_JOINT_TOKENS[finger]
        straightness = float(values["straightness"])
        if straightness <= 0:
            continue
        scale = 1.0 - straightness
        for index, joint_name in enumerate(joint_names):
            if token in joint_name:
                output[index] *= scale

    thumb_info = compute_x2_thumb_straightness(joint_pos)
    thumb_straightness = float(thumb_info.get("straightness", 0.0))
    thumb_relax = thumb_straightness * (1.0 - float(np.clip(pinch_strength, 0.0, 1.0)))
    if thumb_relax > 0:
        scale = 1.0 - thumb_relax
        for index, joint_name in enumerate(joint_names):
            if "_TH" in joint_name:
                output[index] *= scale

    return output, {
        "enabled": True,
        "finger": finger_info,
        "thumb": thumb_info,
        "thumb_relax": thumb_relax,
    }


def format_x2_open_hand_relaxation(info: dict) -> str:
    if not info.get("enabled"):
        return "off"
    finger_parts = []
    for finger, values in info.get("finger", {}).items():
        finger_parts.append(
            f"{finger}:straight={values.get('straightness', 0.0):.2f},"
            f"ratio={values.get('ratio', 0.0):.2f}"
        )
    thumb = info.get("thumb", {})
    thumb_text = (
        f"thumb:straight={thumb.get('straightness', 0.0):.2f},"
        f"ratio={thumb.get('ratio', 0.0):.2f},"
        f"relax={info.get('thumb_relax', 0.0):.2f}"
    )
    return "; ".join(finger_parts + [thumb_text])


def compute_x2_finger_raw_angles(joint_pos: np.ndarray) -> dict[str, dict[str, float]]:
    angles = {}
    for finger, (wrist, mcp, pip, dip, tip) in X2_FINGER_DIRECT_LANDMARKS.items():
        angles[finger] = {
            "mcp": thumb_flexion_angle_deg(joint_pos, wrist, mcp, pip),
            "pip": thumb_flexion_angle_deg(joint_pos, mcp, pip, dip),
            "dip": thumb_flexion_angle_deg(joint_pos, pip, dip, tip),
        }
    return angles


class X2FingerDirectMappingState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.baseline: Optional[dict[str, dict[str, float]]] = None
        self.samples = 0
        self.smoothed_qpos: Optional[dict[str, float]] = None

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
            or control_mode != "left_mode"
            or joint_pos is None
            or len(joint_pos) <= 20
        ):
            return {"enabled": False}

        raw_angles = compute_x2_finger_raw_angles(joint_pos)
        if self.baseline is None:
            self.baseline = {
                finger: values.copy() for finger, values in raw_angles.items()
            }
            self.samples = 1
        elif self.samples < X2_FINGER_DIRECT_BASELINE_FRAMES:
            for finger, values in raw_angles.items():
                for joint_key, value in values.items():
                    self.baseline[finger][joint_key] = min(
                        self.baseline[finger][joint_key], value
                    )
            self.samples += 1
        else:
            for finger, values in raw_angles.items():
                for joint_key, value in values.items():
                    if value < self.baseline[finger][joint_key]:
                        self.baseline[finger][joint_key] = (
                            0.90 * self.baseline[finger][joint_key] + 0.10 * value
                        )

        calibrated = self.samples >= X2_FINGER_DIRECT_BASELINE_FRAMES
        direct_qpos = {}
        flexion_deg = {}
        clipped = {}
        sign = 1.0
        for finger, values in raw_angles.items():
            flexion_deg[finger] = {}
            mcp_joint, pip_joint, dip_joint = X2_FINGER_DIRECT_JOINTS[finger]
            for joint_key, joint_name in [
                ("mcp", mcp_joint),
                ("pip", pip_joint),
                ("dip", dip_joint),
            ]:
                baseline = self.baseline[finger][joint_key]
                deadzone = X2_FINGER_DIRECT_DEADZONE_DEG[joint_key]
                flexion = max(0.0, values[joint_key] - baseline - deadzone)
                flexion_deg[finger][joint_key] = flexion
                target = (
                    sign
                    * X2_FINGER_DIRECT_GAIN_RAD_PER_DEG[joint_key]
                    * gain_scale.get(joint_key, 1.0)
                    * flexion
                )
                if joint_name in joint_names:
                    index = joint_names.index(joint_name)
                    lower, upper = joint_limits[index]
                    clipped_target = float(np.clip(target, lower, upper))
                    direct_qpos[joint_name] = clipped_target
                    clipped[joint_name] = not np.isclose(target, clipped_target)

        if self.smoothed_qpos is None:
            self.smoothed_qpos = direct_qpos.copy()
        else:
            alpha = X2_FINGER_DIRECT_SMOOTHING
            for joint_name, value in direct_qpos.items():
                previous = self.smoothed_qpos.get(joint_name, value)
                self.smoothed_qpos[joint_name] = (
                    (1.0 - alpha) * previous + alpha * value
                )

        return {
            "enabled": True,
            "calibrated": calibrated,
            "samples": self.samples,
            "raw_angles": raw_angles,
            "baseline": {
                finger: values.copy() for finger, values in self.baseline.items()
            },
            "flexion_deg": flexion_deg,
            "direct_qpos": self.smoothed_qpos.copy(),
            "clipped": clipped,
            "gain_scale": gain_scale.copy(),
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
    for finger in ("index", "middle", "ring", "pinky"):
        qpos_items = []
        for joint_name in X2_FINGER_DIRECT_JOINTS[finger]:
            value = direct_info["direct_qpos"].get(joint_name)
            if value is not None:
                qpos_items.append(f"{joint_name}:{value:+.3f}")
        flex = direct_info["flexion_deg"].get(finger, {})
        parts.append(
            f"{finger}[mcp={flex.get('mcp', 0.0):.1f},"
            f"pip={flex.get('pip', 0.0):.1f},"
            f"dip={flex.get('dip', 0.0):.1f}; "
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
    for finger, (mcp_joint, pip_joint, dip_joint) in X2_FINGER_DIRECT_JOINTS.items():
        if mcp_joint not in joint_names or pip_joint not in joint_names:
            continue
        mcp_index = joint_names.index(mcp_joint)
        pip_index = joint_names.index(pip_joint)
        desired_pip = X2_FINGER_COUPLING_PIP_PER_MCP * output[mcp_index]
        output[pip_index] = (
            (1.0 - strength) * output[pip_index] + strength * desired_pip
        )
        output[pip_index] = np.clip(
            output[pip_index], joint_limits[pip_index, 0], joint_limits[pip_index, 1]
        )

        if dip_joint in joint_names:
            dip_index = joint_names.index(dip_joint)
            desired_dip = X2_FINGER_COUPLING_DIP_PER_PIP * output[pip_index]
            output[dip_index] = (
                (1.0 - strength) * output[dip_index] + strength * desired_dip
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


def apply_x2_thumb_pinch_synergy(
    qpos: np.ndarray,
    joint_names: list[str],
    joint_limits: np.ndarray,
    profile: Optional[str],
    pinch_strength: float,
    target_qpos: dict[str, float],
    preserve_joints: Optional[set[str]] = None,
) -> np.ndarray:
    if not is_x2_profile_enabled(profile) or pinch_strength <= 0:
        return qpos

    output = qpos.copy()
    preserve_joints = preserve_joints or set()
    strength = float(np.clip(pinch_strength, 0.0, 1.0))
    for joint_name, target in target_qpos.items():
        if joint_name in preserve_joints:
            continue
        if joint_name not in joint_names:
            continue
        index = joint_names.index(joint_name)
        lower, upper = joint_limits[index]
        target = float(np.clip(target, lower, upper))
        blended = (1.0 - strength) * output[index] + strength * target
        if joint_name == "rh_THJ4":
            output[index] = (
                max(output[index], blended)
                if target >= 0
                else min(output[index], blended)
            )
        elif joint_name == "rh_THJ2":
            output[index] = min(output[index], blended)
        else:
            output[index] = blended
    return output


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


def compute_x2_thumb_raw_angles(joint_pos: np.ndarray) -> dict[str, float]:
    thj3_basal = thumb_flexion_angle_deg(joint_pos, 0, 1, 2)
    thj3_palm_projection = thumb_palm_projection_flexion_deg(joint_pos)
    return {
        "rh_THJ3": max(thj3_basal, thj3_palm_projection),
        "rh_THJ2": thumb_flexion_angle_deg(joint_pos, 1, 2, 3),
        "rh_THJ1": thumb_flexion_angle_deg(joint_pos, 2, 3, 4),
    }


class X2ThumbDirectMappingState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.baseline: Optional[dict[str, float]] = None
        self.samples = 0

    def update(
        self,
        joint_pos: Optional[np.ndarray],
        profile: Optional[str],
        joint_names: list[str],
        joint_limits: np.ndarray,
    ) -> dict:
        if (
            not is_x2_profile_enabled(profile)
            or joint_pos is None
            or len(joint_pos) <= 4
        ):
            return {"enabled": False}

        raw_angles = compute_x2_thumb_raw_angles(joint_pos)
        if self.baseline is None:
            self.baseline = raw_angles.copy()
            self.samples = 1
        elif self.samples < X2_THUMB_DIRECT_BASELINE_FRAMES:
            for joint_name in X2_THUMB_DIRECT_JOINTS:
                self.baseline[joint_name] = min(
                    self.baseline[joint_name], raw_angles[joint_name]
                )
            self.samples += 1
        else:
            for joint_name in X2_THUMB_DIRECT_JOINTS:
                if raw_angles[joint_name] < self.baseline[joint_name]:
                    self.baseline[joint_name] = (
                        0.85 * self.baseline[joint_name]
                        + 0.15 * raw_angles[joint_name]
                    )

        calibrated = self.samples >= X2_THUMB_DIRECT_BASELINE_FRAMES
        flexion_deg = {}
        direct_qpos = {}
        clipped = {}
        for joint_name in X2_THUMB_DIRECT_JOINTS:
            baseline = self.baseline[joint_name]
            deadzone = X2_THUMB_DIRECT_DEADZONE_DEG[joint_name]
            flexion = max(0.0, raw_angles[joint_name] - baseline - deadzone)
            flexion_deg[joint_name] = flexion
            target = -X2_THUMB_DIRECT_GAIN_RAD_PER_DEG[joint_name] * flexion
            if joint_name in joint_names:
                index = joint_names.index(joint_name)
                lower, upper = joint_limits[index]
                clipped_target = float(np.clip(target, lower, upper))
                direct_qpos[joint_name] = clipped_target
                clipped[joint_name] = not np.isclose(target, clipped_target)
            else:
                direct_qpos[joint_name] = 0.0
                clipped[joint_name] = False

        return {
            "enabled": True,
            "calibrated": calibrated,
            "samples": self.samples,
            "raw_angles": raw_angles,
            "baseline": self.baseline.copy(),
            "flexion_deg": flexion_deg,
            "direct_qpos": direct_qpos,
            "clipped": clipped,
        }


def apply_x2_thumb_direct_mapping(
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
        if joint_name not in joint_names:
            continue
        index = joint_names.index(joint_name)
        blend = X2_THUMB_DIRECT_BLEND[joint_name]
        output[index] = (1.0 - blend) * output[index] + blend * target
    return output


def format_x2_thumb_direct_mapping(direct_info: dict) -> str:
    if not direct_info.get("enabled"):
        return "off"
    if not direct_info.get("calibrated"):
        return (
            f"calibrating {direct_info.get('samples', 0)}/"
            f"{X2_THUMB_DIRECT_BASELINE_FRAMES}"
        )
    return " ".join(
        [
            "raw="
            + ",".join(
                f"{name}:{value:.1f}"
                for name, value in direct_info["raw_angles"].items()
            ),
            "baseline="
            + ",".join(
                f"{name}:{value:.1f}"
                for name, value in direct_info["baseline"].items()
            ),
            "flex="
            + ",".join(
                f"{name}:{value:.1f}"
                for name, value in direct_info["flexion_deg"].items()
            ),
            "qpos="
            + ",".join(
                f"{name}:{value:+.3f}"
                for name, value in direct_info["direct_qpos"].items()
            ),
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
                "Use e.g. rh_THJ3=-0.95:0.12."
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
    if "rh_THJ2" in joint_names:
        thj2_value = float(output_qpos[joint_names.index("rh_THJ2")])
        if thj2_value > 0.03:
            print(
                f"  hint: rh_THJ2 is positive ({thj2_value:+.3f}). "
                "For a more natural x2 thumb pinch, try --thumb-natural-limits "
                "or --retarget-joint-limit rh_THJ2=-1.31:0.0.",
                flush=True,
            )
    if "rh_THJ3" in joint_names:
        thj3_value = float(output_qpos[joint_names.index("rh_THJ3")])
        if thj3_value > 0.03:
            print(
                f"  hint: rh_THJ3 is positive ({thj3_value:+.3f}). "
                "For right-hand x2 thumb basal flexion, palm-side bending should "
                "be negative, but a small positive extension is allowed to avoid "
                "stiff offsets. Try --retarget-joint-limit rh_THJ3=-0.95:0.12.",
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
    x2_smooth_alpha: float = 0.65,
    x2_max_delta: float = 0.18,
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
    root_quat: tuple[float, float, float, float] = X2_RIGHT_ROOT_QUAT,
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
        added_local_constraints = add_x2_local_shape_constraints(config)
        logger.warning(
            "x2 natural retargeting profile is active. "
            "It adds local finger constraints, x2-specific vector shaping, "
            "and natural joint limits."
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
    x2_left_output_mode = is_x2_config and hand_type == "Left"
    if x2_left_output_mode:
        if is_close_tuple(root_quat, X2_RIGHT_ROOT_QUAT):
            root_quat = X2_LEFT_ROOT_QUAT
        if abs(camera_azimuth - X2_RIGHT_CAMERA_AZIMUTH) < 1e-9:
            camera_azimuth = X2_LEFT_CAMERA_AZIMUTH

    model_path = resolve_mujoco_model_path(config.urdf_path, mujoco_model_path)
    robot_name = model_path.stem
    if ref_transform is None and is_x2_robot(robot_name):
        # X2 left/right use the same command-frame bending semantics; the
        # world-space hand direction is mirrored by the MuJoCo root pose.
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
    x2_world_bend_command_signs = {
        finger: 1.0 for finger in X2_FOUR_FINGER_DISTAL_BODIES
    }
    x2_world_bend_sign_diagnostics = {}
    if is_x2_config:
        (
            x2_world_bend_command_signs,
            x2_world_bend_sign_diagnostics,
        ) = infer_x2_world_bend_command_signs(
            model,
            root_joint_name,
            root_pos,
            root_quat,
            x2_control_mode,
            retargeting_joint_names,
        )
    x2_left_signed_kinematics = False
    x2_left_default_joint_signs = parse_joint_signs(
        X2_LEFT_JOINT_SIGN, retargeting_joint_names
    )
    if x2_left_output_mode and joint_sign is None:
        logger.warning(
            "x2 left_mode is active. "
            "Using direct MediaPipe finger-angle mapping for four fingers; "
            "thumb remains independently retargeted."
        )
    joint_signs = parse_joint_signs(joint_sign, retargeting_joint_names)
    joint_gains = parse_joint_gains(joint_gain, retargeting_joint_names)
    if thumb_natural_limits and not x2_profile_active:
        default_joint_limit_overrides.extend(
            ["rh_THJ2=-1.31:0.0", "rh_THJ3=-0.95:0.12"]
        )
    retarget_joint_limit = merge_joint_limit_overrides(
        user_retarget_joint_limit, default_joint_limit_overrides
    )
    joint_limits = apply_retarget_joint_limit_overrides(
        retargeting, retarget_joint_limit
    )
    output_joint_limits = get_signed_output_joint_limits(joint_limits, joint_signs)
    x2_palm_flex_guard_enabled = x2_profile_active and joint_sign is None
    if (
        x2_left_output_mode
        and joint_sign is not None
        and np.array_equal(joint_signs, x2_left_default_joint_signs)
    ):
        enable_joint_sign_kinematics(retargeting, joint_signs)
        set_retargeting_optimizer_limits(retargeting, output_joint_limits)
        x2_left_signed_kinematics = True
        logger.warning(
            "x2 left-hand signed kinematics is active. "
            "The optimizer now solves and MuJoCo displays the left-hand command qpos."
        )
    elif x2_left_output_mode and joint_sign is not None:
        logger.warning(
            "x2 left_mode is active, but custom --joint-sign does not "
            f"match the default left bending mode ({X2_LEFT_JOINT_SIGN}). "
            "Signed left-hand kinematics is disabled for this run."
        )
    x2_thumb_pinch_target_qpos = get_x2_thumb_pinch_target_qpos(hand_type)

    if is_x2_robot(robot_name) and joint_sign is None and not x2_left_output_mode:
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
        logger.info(
            f"x2 command smoothing: alpha={x2_smooth_alpha:.2f}, "
            f"max_delta={x2_max_delta:.3f} rad/frame"
        )
        logger.info(
            "x2 world bend command signs: "
            + format_x2_world_bend_command_signs(
                x2_world_bend_sign_diagnostics, x2_control_mode
            )
        )
        if x2_profile_active and x2_control_mode == "left_mode":
            logger.info(
                "x2 left mode uses the common x2 natural joint limits; "
                "no image/example-specific left limits are applied."
            )
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
        if x2_debug_dir is not None:
            logger.info(
                f"x2 debug artifacts: {x2_debug_dir} "
                f"(png={'on' if x2_debug_save_png else 'off'})"
            )
    if x2_profile_active:
        logger.info(
            "x2 thumb pinch target: "
            + format_x2_thumb_pinch_target_qpos(x2_thumb_pinch_target_qpos)
        )
    if is_x2_config:
        logger.info(
            "x2 command mode: "
            + (
                "legacy left signed joint-sign diagnosis"
                if x2_left_signed_kinematics
                else "direct qpos"
            )
        )
        if x2_left_signed_kinematics:
            logger.info(
                "x2 kinematic signs: "
                + ", ".join(
                    f"{name}:{sign:+.0f}"
                    for name, sign in zip(retargeting_joint_names, joint_signs)
                )
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
        alpha=x2_smooth_alpha, max_delta=x2_max_delta
    )
    x2_debug_recorder = X2DebugRecorder(
        Path(x2_debug_dir) if x2_debug_dir is not None else None,
        retargeting_joint_names,
        output_joint_limits,
        save_png=x2_debug_save_png,
    )
    if is_x2_config:
        x2_debug_recorder.write_joint_sweep(model, root_joint_name, root_pos)

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
            )
            profiled_qpos = apply_x2_thumb_pinch_synergy(
                profiled_qpos,
                retargeting_joint_names,
                joint_limits,
                x2_retargeting_profile if is_x2_config else None,
                x2_pinch_strength,
                x2_thumb_pinch_target_qpos,
                preserve_joints={"rh_THJ2", "rh_THJ1"}
                if x2_thumb_direct_info.get("calibrated")
                else None,
            )
            profiled_qpos, x2_open_relax_info = apply_x2_open_hand_relaxation(
                profiled_qpos,
                retargeting_joint_names,
                x2_retargeting_profile if is_x2_config else None,
                joint_pos,
                x2_pinch_strength,
            )
            unguarded_profiled_qpos = profiled_qpos.copy()
            profiled_qpos = apply_x2_palm_flex_guard(
                profiled_qpos,
                retargeting_joint_names,
                x2_palm_flex_guard_enabled,
                x2_control_mode,
            )
            if x2_left_signed_kinematics:
                command_qpos = np.clip(
                    profiled_qpos * joint_gains,
                    output_joint_limits[:, 0],
                    output_joint_limits[:, 1],
                )
            else:
                command_qpos = apply_joint_output_transform(
                    profiled_qpos,
                    joint_signs,
                    joint_gains,
                    output_joint_limits,
                )
            command_qpos = apply_x2_world_bend_command_signs(
                command_qpos,
                retargeting_joint_names,
                x2_world_bend_command_signs,
                x2_profile_active
                and joint_sign is None
                and not x2_left_signed_kinematics,
            )
            command_qpos = np.clip(
                command_qpos, output_joint_limits[:, 0], output_joint_limits[:, 1]
            )
            if x2_profile_active:
                command_qpos, x2_smoothing_info = x2_command_smoother.update(
                    command_qpos
                )
            else:
                x2_smoothing_info = {"enabled": False}
            model_qpos = command_qpos
            if x2_profile_active:
                state_qpos = np.clip(
                    profiled_qpos, joint_limits[:, 0], joint_limits[:, 1]
                )
                retargeting.set_qpos(
                    command_qpos if x2_left_signed_kinematics else state_qpos
                )
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
                            retargeting_joint_names, unguarded_profiled_qpos
                        ),
                        flush=True,
                    )
                    print(
                        "  left command qpos:",
                        format_qpos_by_finger(retargeting_joint_names, command_qpos),
                        flush=True,
                    )
                if x2_palm_flex_guard_enabled:
                    print(
                        "  x2 palm flex guard:",
                        format_x2_palm_flex_guard_changes(
                            unguarded_profiled_qpos,
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
                        "  x2 world bend command signs: "
                        + format_x2_world_bend_command_signs(
                            x2_world_bend_sign_diagnostics, x2_control_mode
                        ),
                        flush=True,
                    )
                    print(
                        "  x2 open hand relax: "
                        + format_x2_open_hand_relaxation(x2_open_relax_info),
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
                            x2_thumb_direct_info["flexion_deg"].get("rh_THJ2", 0.0),
                            x2_thumb_direct_info["flexion_deg"].get("rh_THJ1", 0.0),
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
            # x2 right-hand palm local +Y maps to world +X; left-hand maps to
            # world -X. The default camera azimuth follows that handedness.
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
    x2_smooth_alpha: float = 0.65,
    x2_max_delta: float = 0.18,
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
    root_quat: tuple[float, float, float, float] = X2_RIGHT_ROOT_QUAT,
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
    if robot_name == RobotName.x2 and hand_type == HandType.left:
        if is_close_tuple(root_quat, X2_RIGHT_ROOT_QUAT):
            root_quat = X2_LEFT_ROOT_QUAT
        if abs(camera_azimuth - X2_RIGHT_CAMERA_AZIMUTH) < 1e-9:
            camera_azimuth = X2_LEFT_CAMERA_AZIMUTH
    # X2 left mode uses world -X as both the palm normal and the expected
    # finger-closing direction.

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

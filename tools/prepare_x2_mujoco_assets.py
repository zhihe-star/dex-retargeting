#!/usr/bin/env python3
"""Prepare MuJoCo-friendly assets for the x2 hand URDF.

MuJoCo rejects STL meshes with more than 200000 faces. The original x2 palm
mesh is above that limit, so this script creates a separate mesh directory and
decimates only the meshes that need it. The original URDF and STL files are
left untouched.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_URDF = Path("original/x^2.urdf")
DEFAULT_OUTPUT_URDF = Path("original/x2_mujoco.urdf")
DEFAULT_SOURCE_MESH_DIR = Path("original/meshes")
DEFAULT_OUTPUT_MESH_DIR = Path("original/meshes_mujoco")


def read_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path} is too small to be a binary STL")

    face_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + 50 * face_count
    if expected_size != len(data):
        raise ValueError(
            f"{path} does not look like a standard binary STL: "
            f"size={len(data)}, expected={expected_size}, faces={face_count}"
        )

    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attr", "<u2"),
        ]
    )
    return np.frombuffer(data, dtype=dtype, offset=84, count=face_count)["vertices"].copy()


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    triangles = triangles.astype(np.float32, copy=False)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0

    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attr", "<u2"),
        ]
    )
    records = np.zeros(len(triangles), dtype=dtype)
    records["normal"] = normals
    records["vertices"] = triangles

    header = b"MuJoCo simplified STL".ljust(80, b"\0")
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        f.write(records.tobytes())


def binary_stl_face_count(path: Path) -> int:
    with path.open("rb") as f:
        f.seek(80)
        return struct.unpack("<I", f.read(4))[0]


def cluster_triangles(triangles: np.ndarray, voxel_size: float) -> np.ndarray:
    flat = triangles.reshape(-1, 3)
    mins = flat.min(axis=0)
    quantized = np.floor((flat - mins) / voxel_size).astype(np.int64)

    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    vertex_count = int(inverse.max()) + 1

    vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    np.add.at(vertices, inverse, flat)
    counts = np.bincount(inverse, minlength=vertex_count)
    vertices /= counts[:, None]

    faces = inverse.reshape(-1, 3)
    nondegenerate = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[nondegenerate]

    sorted_faces = np.sort(faces, axis=1)
    _, unique_indices = np.unique(sorted_faces, axis=0, return_index=True)
    faces = faces[np.sort(unique_indices)]

    return vertices[faces].astype(np.float32)


def simplify_stl(input_path: Path, output_path: Path, target_faces: int) -> int:
    triangles = read_binary_stl(input_path)
    if len(triangles) <= target_faces:
        shutil.copy2(input_path, output_path)
        return len(triangles)

    flat = triangles.reshape(-1, 3)
    span = float((flat.max(axis=0) - flat.min(axis=0)).max())
    if span <= 0:
        raise ValueError(f"{input_path} has an invalid bounding box")

    lo = span / 100000.0
    hi = span / 200.0
    simplified = cluster_triangles(triangles, hi)
    while len(simplified) > target_faces:
        lo = hi
        hi *= 1.5
        simplified = cluster_triangles(triangles, hi)

    best = simplified
    for _ in range(24):
        mid = (lo + hi) / 2.0
        current = cluster_triangles(triangles, mid)
        if len(current) > target_faces:
            lo = mid
        else:
            hi = mid
            best = current

    write_binary_stl(output_path, best)
    return len(best)


def make_mujoco_urdf(source_urdf: Path, output_urdf: Path) -> None:
    text = source_urdf.read_text()
    text = text.replace('filename="meshes/', 'filename="meshes_mujoco/')
    text = text.replace('name="x2_hand"', 'name="x2_hand_mujoco"', 1)
    if 'name="mujoco_root"' not in text:
        root_wrapper = """  <link name="mujoco_root" />
  <joint name="mujoco_root_joint" type="floating">
    <parent link="mujoco_root" />
    <child link="rh_palm" />
    <origin xyz="0 0 0" rpy="0 0 0" />
  </joint>

"""
        text, count = re.subn(r"(<robot\b[^>]*>\n)", r"\1" + root_wrapper, text, count=1)
        if count != 1:
            raise ValueError(f"Could not find <robot> tag in {source_urdf}")
    output_urdf.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument("--output-urdf", type=Path, default=DEFAULT_OUTPUT_URDF)
    parser.add_argument("--source-mesh-dir", type=Path, default=DEFAULT_SOURCE_MESH_DIR)
    parser.add_argument("--output-mesh-dir", type=Path, default=DEFAULT_OUTPUT_MESH_DIR)
    parser.add_argument("--target-faces", type=int, default=180000)
    args = parser.parse_args()

    args.output_mesh_dir.mkdir(parents=True, exist_ok=True)

    for input_path in sorted(args.source_mesh_dir.glob("*.stl")):
        output_path = args.output_mesh_dir / input_path.name
        source_faces = binary_stl_face_count(input_path)
        output_faces = simplify_stl(input_path, output_path, args.target_faces)
        action = "copied" if source_faces == output_faces else "simplified"
        print(f"{action}: {input_path.name}: {source_faces} -> {output_faces} faces")

    make_mujoco_urdf(args.source_urdf, args.output_urdf)
    print(f"wrote: {args.output_urdf}")


if __name__ == "__main__":
    main()

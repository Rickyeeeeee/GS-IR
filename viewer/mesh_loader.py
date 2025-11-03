from __future__ import annotations

import base64
import json
import logging
import math
import os
import struct
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,                          # DEBUG/INFO/WARNING/ERROR/CRITICAL
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


GLB_HEADER_STRUCT = struct.Struct("<4sII")
GLB_CHUNK_HEADER_STRUCT = struct.Struct("<II")
GLB_MAGIC = b"glTF"
GLB_CHUNK_TYPE_JSON = 0x4E4F534A
GLB_CHUNK_TYPE_BIN = 0x004E4942

logger = logging.getLogger(__name__)


@dataclass
class _Primitive:
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: Optional[np.ndarray]
    base_color_factor: np.ndarray
    metallic_factor: float
    roughness_factor: float
    base_color_texture: Optional[np.ndarray]
    metallic_roughness_texture: Optional[np.ndarray]


def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    rgb = arr[..., :3]
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((np.clip(rgb, 0.04045, None) + 0.055) / 1.055, 2.4),
    )
    result = arr.copy()
    result[..., :3] = linear
    return result


def _decode_image(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as handle:
        image = handle.convert("RGBA")
    arr = np.array(image, dtype=np.float32) / 255.0
    return arr.astype(np.float32)


def _load_images(gltf: Dict[str, Any], buffers: List[bytes], asset_dir: str) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for idx, image_dict in enumerate(gltf.get("images", [])):
        logger.info("Decoding image %d", idx)
        if "uri" in image_dict:
            uri = image_dict["uri"]
            if uri.startswith("data:"):
                comma_idx = uri.find(",")
                if comma_idx == -1:
                    raise ValueError("Malformed data URI in image.")
                encoded = uri[comma_idx + 1 :]
                image_bytes = base64.b64decode(encoded)
            else:
                image_path = os.path.join(asset_dir, uri)
                with open(image_path, "rb") as handle:
                    image_bytes = handle.read()
        elif "bufferView" in image_dict:
            buffer_view = gltf["bufferViews"][image_dict["bufferView"]]
            buffer_idx = buffer_view.get("buffer", 0)
            buffer_data = buffers[buffer_idx]
            offset = buffer_view.get("byteOffset", 0)
            length = buffer_view["byteLength"]
            image_bytes = buffer_data[offset : offset + length]
        else:
            raise ValueError("Image must reference either a uri or bufferView.")
        images.append(_decode_image(image_bytes))
    return images


def _get_texture_image(
    gltf: Dict[str, Any],
    images: List[np.ndarray],
    texture_idx: int,
    srgb: bool,
) -> np.ndarray:
    textures = gltf.get("textures", [])
    if texture_idx >= len(textures):
        raise ValueError(f"Texture index {texture_idx} out of range.")
    texture_info = textures[texture_idx]
    source_idx = texture_info.get("source")
    if source_idx is None or source_idx >= len(images):
        raise ValueError("Texture source missing or out of range.")
    image = images[source_idx]
    if srgb:
        return _srgb_to_linear(image)
    return image


def _quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = q
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _euler_deg_to_matrix(euler_deg: Sequence[float], order: str = "ZYX") -> np.ndarray:
    if len(euler_deg) != 3:
        raise ValueError("rotation_euler_deg must contain exactly 3 values.")
    order = order.upper()
    if len(order) != 3 or any(c not in "XYZ" for c in order):
        raise ValueError("Euler order must be a permutation of X, Y, Z.")
    angles = [math.radians(float(a)) for a in euler_deg]

    def _axis_matrix(axis: str, angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        if axis == "X":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        if axis == "Y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    R = np.eye(3, dtype=np.float32)
    for axis, angle in zip(order, angles):
        R = R @ _axis_matrix(axis, angle)
    return R


def _compose_trs(
    translation: Sequence[float],
    rotation_matrix: np.ndarray,
    scale: Sequence[float],
) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    R = np.eye(4, dtype=np.float32)
    R[:3, :3] = rotation_matrix.astype(np.float32)
    scale_arr = np.array(scale, dtype=np.float32)
    S = np.diag(np.append(scale_arr, 1.0)).astype(np.float32)
    return T @ R @ S


def _matrix_from_transform_spec(transform: Optional[Dict[str, Any]]) -> np.ndarray:
    if not transform:
        return np.eye(4, dtype=np.float32)

    if "matrix" in transform:
        matrix_values = np.array(transform["matrix"], dtype=np.float32)
        if matrix_values.size != 16:
            raise ValueError("Transform matrix must provide 16 values.")
        return matrix_values.reshape(4, 4)

    translation = transform.get("translation", [0.0, 0.0, 0.0])
    scale_spec = transform.get("scale", [1.0, 1.0, 1.0])
    if isinstance(scale_spec, (int, float)):
        scale = [float(scale_spec)] * 3
    else:
        if len(scale_spec) != 3:
            raise ValueError("scale must be a float or a sequence of length 3.")
        scale = [float(s) for s in scale_spec]

    if "rotation_quat" in transform:
        rotation_matrix = _quat_to_matrix(transform["rotation_quat"])
    elif "rotation" in transform and len(transform["rotation"]) == 4:
        rotation_matrix = _quat_to_matrix(transform["rotation"])
    elif "rotation_euler_deg" in transform:
        rotation_matrix = _euler_deg_to_matrix(transform["rotation_euler_deg"])
    else:
        rotation_matrix = np.eye(3, dtype=np.float32)

    return _compose_trs(translation, rotation_matrix, scale)


def _load_glb_bytes(path: str) -> Tuple[Dict[str, Any], List[bytes]]:
    with open(path, "rb") as handle:
        data = handle.read()

    if len(data) < GLB_HEADER_STRUCT.size:
        raise ValueError(f"GLB file '{path}' is too small to contain a valid header.")

    magic, version, length = GLB_HEADER_STRUCT.unpack_from(data, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"File '{path}' is not a valid GLB (invalid magic).")
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version} in '{path}'.")
    if length != len(data):
        raise ValueError(f"GLB length mismatch for '{path}' (header {length} vs actual {len(data)}).")

    offset = GLB_HEADER_STRUCT.size
    json_dict: Optional[Dict[str, Any]] = None
    bin_chunks: List[bytes] = []

    while offset < len(data):
        if offset + GLB_CHUNK_HEADER_STRUCT.size > len(data):
            raise ValueError(f"Corrupted GLB '{path}': chunk header truncated.")
        chunk_length, chunk_type = GLB_CHUNK_HEADER_STRUCT.unpack_from(data, offset)
        offset += GLB_CHUNK_HEADER_STRUCT.size
        chunk_data = data[offset : offset + chunk_length]
        if len(chunk_data) != chunk_length:
            raise ValueError(f"Corrupted GLB '{path}': chunk data truncated.")
        offset += chunk_length

        if chunk_type == GLB_CHUNK_TYPE_JSON:
            json_dict = json.loads(chunk_data.decode("utf-8"))
        elif chunk_type == GLB_CHUNK_TYPE_BIN:
            bin_chunks.append(chunk_data)

    if json_dict is None:
        raise ValueError(f"GLB '{path}' does not contain a JSON chunk.")
    if not bin_chunks:
        raise ValueError(f"GLB '{path}' does not contain any BIN chunks.")
    return json_dict, bin_chunks


def _accessor_to_numpy(
    gltf: Dict[str, Any],
    buffers: List[bytes],
    accessor_idx: int,
) -> np.ndarray:
    accessor = gltf["accessors"][accessor_idx]
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    buffer_idx = buffer_view.get("buffer", 0)
    buffer_data = buffers[buffer_idx]

    component_type = accessor["componentType"]
    accessor_type = accessor["type"]
    count = accessor["count"]

    component_type_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    num_components_map = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    if component_type not in component_type_map:
        raise NotImplementedError(f"Unsupported component type: {component_type}")
    if accessor_type not in num_components_map:
        raise NotImplementedError(f"Unsupported accessor type: {accessor_type}")

    dtype = component_type_map[component_type]
    num_components = num_components_map[accessor_type]

    byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    byte_stride = buffer_view.get("byteStride")

    expected_num_bytes = count * num_components * np.dtype(dtype).itemsize
    available_bytes = len(buffer_data) - byte_offset
    if byte_stride is None and expected_num_bytes > available_bytes:
        raise ValueError("Accessor exceeds buffer size.")

    if byte_stride is None:
        array = np.frombuffer(
            buffer_data,
            dtype=dtype,
            count=count * num_components,
            offset=byte_offset,
        )
        return array.reshape(count, num_components)

    stride = byte_stride
    out = np.zeros((count, num_components), dtype=dtype)
    for idx in range(count):
        start = byte_offset + idx * stride
        slice_bytes = buffer_data[start : start + num_components * np.dtype(dtype).itemsize]
        out[idx] = np.frombuffer(slice_bytes, dtype=dtype, count=num_components)
    return out


def _ensure_normals(positions: np.ndarray, indices: np.ndarray, normals: Optional[np.ndarray]) -> np.ndarray:
    if normals is not None:
        return normals.astype(np.float32)

    computed = np.zeros_like(positions, dtype=np.float32)
    tris = indices.reshape(-1, 3)
    for tri in tris:
        p0, p1, p2 = positions[tri]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            n /= norm
        computed[tri] += n
    norms = np.linalg.norm(computed, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    computed /= norms
    return computed


def _gather_primitives(
    gltf: Dict[str, Any],
    buffers: List[bytes],
    images: List[np.ndarray],
) -> List[Tuple[_Primitive, np.ndarray]]:
    scene_idx = gltf.get("scene", 0)
    scenes = gltf.get("scenes", [])
    if not scenes:
        raise ValueError("glTF file does not contain any scenes.")
    scene = scenes[scene_idx]
    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    materials = gltf.get("materials", [])

    def node_matrix(node_dict: Dict[str, Any]) -> np.ndarray:
        if "matrix" in node_dict:
            mat = np.array(node_dict["matrix"], dtype=np.float32)
            if mat.size != 16:
                raise ValueError("Node matrix must have 16 values.")
            return mat.reshape(4, 4).T
        translation = node_dict.get("translation", [0.0, 0.0, 0.0])
        scale = node_dict.get("scale", [1.0, 1.0, 1.0])
        rotation = node_dict.get("rotation", [0.0, 0.0, 0.0, 1.0])
        rotation_matrix = _quat_to_matrix(rotation)
        return _compose_trs(translation, rotation_matrix, scale)

    collected: List[Tuple[_Primitive, np.ndarray]] = []

    def traverse(node_index: int, parent_matrix: np.ndarray) -> None:
        node = nodes[node_index]
        local_matrix = node_matrix(node)
        world_matrix = parent_matrix @ local_matrix

        mesh_idx = node.get("mesh")
        if mesh_idx is not None:
            mesh = meshes[mesh_idx]
            for prim_idx, primitive_dict in enumerate(mesh.get("primitives", [])):
                pos_accessor = primitive_dict["attributes"].get("POSITION")
                if pos_accessor is None:
                    continue
                positions = _accessor_to_numpy(gltf, buffers, pos_accessor).astype(np.float32)
                normals_accessor = primitive_dict["attributes"].get("NORMAL")
                normals = None
                if normals_accessor is not None:
                    normals = _accessor_to_numpy(gltf, buffers, normals_accessor).astype(np.float32)
                uv_accessor = primitive_dict["attributes"].get("TEXCOORD_0")
                uvs = None
                if uv_accessor is not None:
                    uvs = _accessor_to_numpy(gltf, buffers, uv_accessor).astype(np.float32)

                indices_accessor = primitive_dict.get("indices")
                if indices_accessor is None:
                    continue
                indices_raw = _accessor_to_numpy(gltf, buffers, indices_accessor)
                indices = indices_raw.astype(np.int32).reshape(-1, 3)

                world_positions = (
                    np.c_[positions, np.ones(positions.shape[0], dtype=np.float32)] @ world_matrix.T
                )[:, :3]
                normal_matrix = np.linalg.inv(world_matrix[:3, :3]).T
                world_normals = (_ensure_normals(positions, indices, normals) @ normal_matrix).astype(np.float32)
                normal_lengths = np.linalg.norm(world_normals, axis=1, keepdims=True)
                normal_lengths[normal_lengths < 1e-12] = 1.0
                world_normals /= normal_lengths

                material_idx = primitive_dict.get("material")
                base_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                metallic_factor = 1.0
                roughness_factor = 1.0
                base_color_texture = None
                metallic_roughness_texture = None
                if material_idx is not None and material_idx < len(materials):
                    material = materials[material_idx]
                    pbr = material.get("pbrMetallicRoughness", {})
                    base = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
                    base_color = np.array(base[:3], dtype=np.float32)
                    metallic_factor = float(pbr.get("metallicFactor", metallic_factor))
                    roughness_factor = float(pbr.get("roughnessFactor", roughness_factor))

                    base_tex_info = pbr.get("baseColorTexture")
                    if base_tex_info is not None:
                        tex_idx = base_tex_info.get("index")
                        texcoord_set = base_tex_info.get("texCoord", 0)
                        if tex_idx is not None:
                            if texcoord_set != 0:
                                logger.warning("Only TEXCOORD_0 is supported for base color textures.")
                            elif uvs is None:
                                logger.warning("Base color texture specified but TEXCOORD_0 missing; ignoring texture.")
                            else:
                                base_color_texture = _get_texture_image(gltf, images, tex_idx, srgb=True)
                                logger.info(
                                    "Loaded baseColorTexture for primitive %d (mesh %d) size=%s",
                                    prim_idx,
                                    mesh_idx,
                                    base_color_texture.shape,
                                )

                    mr_tex_info = pbr.get("metallicRoughnessTexture")
                    if mr_tex_info is not None:
                        tex_idx = mr_tex_info.get("index")
                        texcoord_set = mr_tex_info.get("texCoord", 0)
                        if tex_idx is not None:
                            if texcoord_set != 0:
                                logger.warning("Only TEXCOORD_0 is supported for metallic-roughness textures.")
                            elif uvs is None:
                                logger.warning("Metallic-roughness texture specified but TEXCOORD_0 missing; ignoring texture.")
                            else:
                                metallic_roughness_texture = _get_texture_image(gltf, images, tex_idx, srgb=False)
                                logger.info(
                                    "Loaded metallicRoughnessTexture for primitive %d (mesh %d) size=%s",
                                    prim_idx,
                                    mesh_idx,
                                    metallic_roughness_texture.shape,
                                )

                logger.info(
                    "Primitive %d (mesh %d) vertices=%d triangles=%d base_color=%s metallic=%.3f roughness=%.3f",
                    prim_idx,
                    mesh_idx,
                    world_positions.shape[0],
                    indices.shape[0],
                    np.array2string(base_color, precision=3),
                    metallic_factor,
                    roughness_factor,
                )

                collected.append(
                    (
                        _Primitive(
                            positions=world_positions.astype(np.float32),
                            normals=world_normals.astype(np.float32),
                            indices=indices.astype(np.int32),
                            uvs=uvs.astype(np.float32) if uvs is not None else None,
                            base_color_factor=base_color.astype(np.float32),
                            metallic_factor=metallic_factor,
                            roughness_factor=roughness_factor,
                            base_color_texture=base_color_texture,
                            metallic_roughness_texture=metallic_roughness_texture,
                        ),
                        world_matrix,
                    )
                )

        for child_idx in node.get("children", []):
            traverse(child_idx, world_matrix)

    identity = np.eye(4, dtype=np.float32)
    for root_node in scene.get("nodes", []):
        traverse(root_node, identity)

    return collected


def load_glb_pbr_mesh(path: str, extra_transform: Optional[np.ndarray] = None) -> Optional[List[Dict[str, Any]]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mesh file not found: {path}")
    logger.info("Loading GLB mesh '%s'", path)
    gltf_json, buffer_chunks = _load_glb_bytes(path)
    images = _load_images(gltf_json, buffer_chunks, os.path.dirname(path))
    gathered = _gather_primitives(gltf_json, buffer_chunks, images)
    if not gathered:
        logger.warning("No primitives found in '%s'", path)
        return None

    segments: List[Dict[str, Any]] = []
    for prim, world_matrix in gathered:
        transform_matrix = extra_transform @ world_matrix if extra_transform is not None else world_matrix
        positions_h = np.c_[prim.positions, np.ones(prim.positions.shape[0], dtype=np.float32)]
        world_positions = (positions_h @ transform_matrix.T)[:, :3]
        normal_matrix = np.linalg.inv(transform_matrix[:3, :3]).T
        world_normals = (prim.normals @ normal_matrix).astype(np.float32)
        normal_lengths = np.linalg.norm(world_normals, axis=1, keepdims=True)
        normal_lengths[normal_lengths < 1e-12] = 1.0
        world_normals /= normal_lengths

        segments.append(
            {
                "positions": world_positions.astype(np.float32),
                "normals": world_normals.astype(np.float32),
                "indices": prim.indices.reshape(-1, 3).astype(np.int32),
                "uvs": prim.uvs.astype(np.float32) if prim.uvs is not None else None,
                "base_color_factor": prim.base_color_factor.astype(np.float32),
                "metallic_factor": float(prim.metallic_factor),
                "roughness_factor": float(prim.roughness_factor),
                "base_color_texture": prim.base_color_texture.copy() if prim.base_color_texture is not None else None,
                "metallic_roughness_texture": prim.metallic_roughness_texture.copy() if prim.metallic_roughness_texture is not None else None,
            }
        )

    return segments


def load_pbr_meshes(
    mesh_specs: Optional[Union[Sequence[Any], str]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor | Optional[torch.Tensor]]]:
    if mesh_specs is None:
        return []

    if isinstance(mesh_specs, (str, os.PathLike)):
        specs_iterable: List[Any] = [os.fspath(mesh_specs)]
    else:
        specs_iterable = list(mesh_specs)

    loaded_meshes: List[Dict[str, torch.Tensor | Optional[torch.Tensor]]] = []
    for raw_spec in specs_iterable:
        if isinstance(raw_spec, (str, os.PathLike)):
            mesh_path = os.fspath(raw_spec)
            extra = np.eye(4, dtype=np.float32)
        elif isinstance(raw_spec, dict):
            if "path" not in raw_spec:
                raise ValueError("Mesh spec dictionaries must include a 'path' key.")
            mesh_path = os.fspath(raw_spec["path"])
            extra = _matrix_from_transform_spec(raw_spec.get("transform"))
        else:
            raise ValueError(f"Unsupported mesh specification type: {type(raw_spec)}")

        logger.info("Preparing mesh spec for path '%s'", mesh_path)
        mesh_segments = load_glb_pbr_mesh(mesh_path, extra_transform=extra)
        if not mesh_segments:
            logger.warning("Skipping mesh '%s' because it returned no data.", mesh_path)
            continue

        for seg_idx, seg in enumerate(mesh_segments):
            logger.info(
                "Loaded mesh segment %d from '%s': vertices=%d triangles=%d",
                seg_idx,
                mesh_path,
                seg["positions"].shape[0],
                seg["indices"].shape[0],
            )
            entry: Dict[str, torch.Tensor | Optional[torch.Tensor]] = {
                "positions": torch.from_numpy(seg["positions"]).to(device=device, dtype=torch.float32).contiguous(),
                "normals": torch.from_numpy(seg["normals"]).to(device=device, dtype=torch.float32).contiguous(),
                "indices": torch.from_numpy(seg["indices"]).to(device=device, dtype=torch.int32).contiguous(),
                "base_color_factor": torch.from_numpy(seg["base_color_factor"]).to(device=device, dtype=torch.float32),
                "metallic_factor": torch.tensor(seg["metallic_factor"], device=device, dtype=torch.float32),
                "roughness_factor": torch.tensor(seg["roughness_factor"], device=device, dtype=torch.float32),
            }
            if seg["uvs"] is not None:
                entry["uvs"] = torch.from_numpy(seg["uvs"]).to(device=device, dtype=torch.float32).contiguous()
            else:
                entry["uvs"] = None

            if seg["base_color_texture"] is not None:
                entry["base_color_texture"] = torch.from_numpy(seg["base_color_texture"]).to(
                    device=device, dtype=torch.float32
                ).contiguous()
            else:
                entry["base_color_texture"] = None

            if seg["metallic_roughness_texture"] is not None:
                entry["metallic_roughness_texture"] = torch.from_numpy(seg["metallic_roughness_texture"]).to(
                    device=device, dtype=torch.float32
                ).contiguous()
            else:
                entry["metallic_roughness_texture"] = None

            loaded_meshes.append(entry)

    return loaded_meshes

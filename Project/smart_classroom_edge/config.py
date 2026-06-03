"""路径与运行参数。模型默认指向同仓库内的 Open Model Zoo 演示目录。"""
from __future__ import annotations

import os
from pathlib import Path

# smart_classroom_edge 目录
EDGE_DIR = Path(__file__).resolve().parent
# 与 smart_classroom_edge 同级的项目根（包含 open_model_zoo-master）
PROJECT_ROOT = EDGE_DIR.parent

# 相对 PROJECT_ROOT 的默认 IR 路径（与 OMZ smart_classroom_demo 示例一致）
DEFAULT_MODEL_RELATIVE = Path(
    "open_model_zoo-master",
    "demos",
    "smart_classroom_demo",
    "cpp",
    "models",
    "intel",
    "person-detection-action-recognition-0005",
    "FP32",
    "person-detection-action-recognition-0005.xml",
)

# 默认演示视频放在 assets 下，便于与代码分离
DEFAULT_VIDEO_RELATIVE = Path("assets", "test.mp4")

ROOM_CAPACITY = 40
CONF_THRESH = 0.25
NMS_THRESH = 0.4
MAX_MISSING_FRAMES = 15
HISTORY_LEN = 300

ENV_MODEL_XML = "SMART_CLASSROOM_MODEL_XML"
ENV_VIDEO = "SMART_CLASSROOM_VIDEO"


def default_model_xml() -> Path:
    return (PROJECT_ROOT / DEFAULT_MODEL_RELATIVE).resolve()


def default_video_path() -> Path:
    """优先与脚本同目录的 test.mp4，其次 assets/test.mp4（避免误用占位小文件挡住根目录真视频）。"""
    same_dir = (EDGE_DIR / "test.mp4").resolve()
    in_assets = (EDGE_DIR / DEFAULT_VIDEO_RELATIVE).resolve()
    if same_dir.is_file():
        return same_dir
    if in_assets.is_file():
        return in_assets
    return in_assets


def resolve_model_xml(override: str | os.PathLike[str] | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.environ.get(ENV_MODEL_XML)
    if env:
        return Path(env).expanduser().resolve()
    return default_model_xml()


def resolve_video_path(override: str | os.PathLike[str] | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    env = os.environ.get(ENV_VIDEO)
    if env:
        return Path(env).expanduser().resolve()
    return default_video_path()

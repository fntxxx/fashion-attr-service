from __future__ import annotations

from pathlib import Path


def preload_siglip() -> None:
    from fashion_attr_service.models.fashion_siglip_model import get_clip_model

    print('[preload] loading Marqo FashionSigLIP...', flush=True)
    get_clip_model()
    print('[preload] Marqo FashionSigLIP ready.', flush=True)


def preload_yolo() -> None:
    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parents[2]
    weights_dir = project_root / 'artifacts' / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_path = weights_dir / 'yolov8n.pt'

    print(f'[preload] loading YOLO weights into {target_path}...', flush=True)
    YOLO(str(target_path))
    print('[preload] YOLO weights ready.', flush=True)


if __name__ == '__main__':
    preload_siglip()
    preload_yolo()

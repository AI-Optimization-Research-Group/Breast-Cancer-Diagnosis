import os
from pathlib import Path
from typing import Sequence, Optional
import math

import torch
from PIL import Image
from ultralytics import YOLO





model_path = r"your path" 
input_root = Path(r"your path") 
output_root = Path(r"your path")  
subfolders = ["birads1", "birads2", "birads4", "birads5"]
img_exts: Sequence[str] = (".png", ".jpg", ".jpeg")

conf_threshold: float = 0.25
iou_threshold: float = 0.7
imgsz: int = 640
allowed_classes: Optional[Sequence[int]] = None


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_crop(img: Image.Image, box_xyxy, out_path: Path):
    W, H = img.size
    x1, y1, x2, y2 = box_xyxy
    x1 = int(clamp(math.floor(x1), 0, W - 1))
    y1 = int(clamp(math.floor(y1), 0, H - 1))
    x2 = int(clamp(math.ceil(x2), 1, W))
    y2 = int(clamp(math.ceil(y2), 1, H))
    if x2 <= x1 or y2 <= y1:
        return False
    crop = img.crop((x1, y1, x2, y2))
    ensure_dir(out_path.parent)
    crop.save(out_path)
    return True


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    ensure_dir(output_root)
    total_images = 0
    total_crops = 0

    for sub in subfolders:
        in_dir = input_root / sub
        out_dir = output_root / sub
        if not in_dir.exists():
            continue
        ensure_dir(out_dir)

        for img_path in sorted(in_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in img_exts:
                continue

            total_images += 1
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    results_list = model.predict(
                        source=str(img_path),
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                        save=False
                    )
                    if not results_list:
                        continue
                    results = results_list[0]
                    boxes = getattr(results, "boxes", None)
                    if boxes is None or boxes.data is None or len(boxes) == 0:
                        continue

                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else None

                    keep_indices = list(range(xyxy.shape[0]))
                    if allowed_classes is not None and cls_ids is not None:
                        keep_indices = [i for i in keep_indices if int(cls_ids[i]) in allowed_classes]
                    if not keep_indices:
                        continue

                    stem = img_path.stem
                    ext = img_path.suffix.lower()

                    if len(keep_indices) == 1:
                        i = keep_indices[0]
                        out_path = out_dir / f"{stem}{ext}"
                        if save_crop(img, xyxy[i], out_path):
                            total_crops += 1
                    else:
                        for rank, i in enumerate(keep_indices, start=1):
                            out_path = out_dir / f"{stem}_{rank}{ext}"
                            if save_crop(img, xyxy[i], out_path):
                                total_crops += 1
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

    print(f"[DONE] Processed images: {total_images}, Saved crops: {total_crops}")
    print(f"[OUTPUT] Root: {output_root}")


if __name__ == "__main__":
    run()

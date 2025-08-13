#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xml2yolo_samefolder.py
----------------------
Convert Pascal VOC-style XML annotations to YOLO format when **images and XMLs
are in the SAME directory**. Produces a train-only YOLO folder:

    <out_dir>/train/images
    <out_dir>/train/labels

Usage (example):
    python xml2yolo_samefolder.py \
        --in-dir "/path/to/folder_with_png_and_xml" \
        --out   "/path/to/out_yolo" \
        --image-exts .png \
        --classes meme \
        --make-data-yaml

Notes
-----
- Expects Pascal VOC-like XML:
  <annotation>
    <size><width>W</width><height>H</height>...</size>
    <object>
      <name>class_name</name>
      <bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax></bndbox>
    </object>
  </annotation>
- Writes empty .txt for images with no objects (YOLO-compatible).
- If --classes is given, unseen names are skipped. Otherwise classes discovered
  from XML are used (alphabetically) to build classes.txt.
- Optional: write data.yaml for Ultralytics with --make-data-yaml.
"""
import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Try Pillow for image-size fallback if XML has no <size>
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


def find_images_and_xmls(root: Path, exts: List[str]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    images: Dict[str, Path] = {}
    xmls: Dict[str, Path] = {}
    exts_lower = {e.lower() for e in exts}
    for p in root.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".xml":
            xmls[p.stem] = p
        elif suf in exts_lower:
            images[p.stem] = p
    return images, xmls


def parse_voc_xml(xml_path: Path) -> Tuple[Optional[int], Optional[int], List[Tuple[str, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    W = H = None
    size = root.find("size")
    if size is not None:
        try:
            W = int(float(size.findtext("width", default="0")))
            H = int(float(size.findtext("height", default="0")))
            if W <= 0 or H <= 0:
                W = H = None
        except Exception:
            W = H = None

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown").strip()
        bnd = obj.find("bndbox") or obj.find("bbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except Exception:
            continue
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin
        objects.append((name, xmin, ymin, xmax, ymax))
    return W, H, objects


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def voc_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, W: int, H: int):
    xmin = clamp(xmin, 0, W - 1)
    ymin = clamp(ymin, 0, H - 1)
    xmax = clamp(xmax, 0, W - 1)
    ymax = clamp(ymax, 0, H - 1)
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return (cx / W, cy / H, bw / W, bh / H)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_txt(p: Path, lines: List[str]):
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_class_index(given: List[str], discovered: List[str]) -> Dict[str, int]:
    if given:
        mapping = {n: i for i, n in enumerate(given)}
        unk = sorted(set(discovered) - set(given))
        if unk:
            print(f"[warn] {len(unk)} class name(s) not in --classes → skipped: {unk[:5]}{' ...' if len(unk)>5 else ''}")
        return mapping
    else:
        uniq = sorted(set(discovered))
        return {n: i for i, n in enumerate(uniq)}


def maybe_get_size(img_path: Path) -> Tuple[int, int]:
    if PIL_OK:
        try:
            with Image.open(img_path) as im:
                return im.size  # (W,H)
        except Exception:
            pass
    try:
        import cv2
        im = cv2.imread(str(img_path))
        if im is not None:
            H, W = im.shape[:2]
            return (W, H)
    except Exception:
        pass
    raise RuntimeError(f"Cannot determine size for: {img_path}")


def save_classes(out_labels: Path, cls2id: Dict[str, int]):
    items = sorted(cls2id.items(), key=lambda kv: kv[1])
    (out_labels / "classes.txt").write_text("\n".join([k for k, _ in items]) + "\n", encoding="utf-8")


def maybe_write_yaml(out_dir: Path, cls2id: Dict[str, int]):
    names = [k for k, _ in sorted(cls2id.items(), key=lambda kv: kv[1])]
    yaml = [
        "train: train/images",
        "val:   train/images  # adjust this to your real val split if you have one",
        f"nc: {len(names)}",
        "names: [" + ", ".join([f\"'{n}'\" for n in names]) + "]",
        ""
    ]
    (out_dir / "data.yaml").write_text("\n".join(yaml), encoding="utf-8")


def convert_same_folder(in_dir: Path, out_dir: Path, image_exts: List[str], given_classes: List[str], make_yaml: bool):
    out_images = out_dir / "train" / "images"
    out_labels = out_dir / "train" / "labels"
    ensure_dir(out_images); ensure_dir(out_labels)

    images, xmls = find_images_and_xmls(in_dir, image_exts)
    stems_img = set(images.keys()); stems_xml = set(xmls.keys())
    matched = sorted(stems_img & stems_xml)
    only_img = sorted(stems_img - stems_xml)
    only_xml = sorted(stems_xml - stems_img)

    print(f"[info] in-dir: {in_dir}")
    print(f"[info] images: {len(images)}, xmls: {len(xmls)}, matched: {len(matched)}")
    if only_img:
        print(f"[warn] images w/o xml: {len(only_img)} (e.g., {only_img[:3]})")
    if only_xml:
        print(f"[warn] xmls w/o image: {len(only_xml)} (e.g., {only_xml[:3]})")

    discovered: List[str] = []
    for s in matched:
        _, _, objs = parse_voc_xml(xmls[s])
        for (name, *_bb) in objs:
            discovered.append(name)
    cls2id = build_class_index(given_classes, discovered)
    save_classes(out_labels, cls2id)

    total = kept = zero = 0
    for s in matched:
        img_p = images[s]; xml_p = xmls[s]
        W, H, objs = parse_voc_xml(xml_p)
        if W is None or H is None:
            try:
                W, H = maybe_get_size(img_p)
            except Exception as e:
                print(f"[error] {e}. Skipping {img_p.name}")
                continue

        lines: List[str] = []
        for (name, xmin, ymin, xmax, ymax) in objs:
            total += 1
            if name not in cls2id:
                continue
            conv = voc_to_yolo(xmin, ymin, xmax, ymax, W, H)
            if conv is None:
                zero += 1
                continue
            cx, cy, bw, bh = conv
            lines.append(f"{cls2id[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            kept += 1

        write_txt(out_labels / f"{s}.txt", lines)
        shutil.copy2(img_p, out_images / img_p.name)

    print(f"[done] images → {out_images}")
    print(f"[done] labels → {out_labels}")
    print(f"[stats] total boxes: {total}, kept: {kept}, zero/invalid: {zero}")

    if make_yaml and cls2id:
        maybe_write_yaml(out_dir, cls2id)
        print(f"[info] data.yaml written at: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Convert VOC XMLs to YOLO when images & XMLs are in the same folder (train-only).")
    ap.add_argument("--in-dir", type=str, required=True, help="Directory containing both images (e.g., .png) and XMLs")
    ap.add_argument("--out", type=str, required=True, help="Output directory to write YOLO train/images and train/labels")
    ap.add_argument("--image-exts", type=str, nargs="+", default=[".png", ".jpg", ".jpeg", ".bmp"], help="Accepted image extensions")
    ap.add_argument("--classes", type=str, nargs="*", default=[], help="Optional class list to fix ID order")
    ap.add_argument("--make-data-yaml", action="store_true", help="Also create a simple Ultralytics-style data.yaml")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out).resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[error] Input folder does not exist or is not a directory: {in_dir}")
        sys.exit(1)

    convert_same_folder(
        in_dir=in_dir,
        out_dir=out_dir,
        image_exts=args.image_exts,
        given_classes=args.classes,
        make_yaml=args.make_data_yaml
    )


if __name__ == "__main__":
    main()

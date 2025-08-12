from pathlib import Path
import xml.etree.ElementTree as ET
import shutil, random, sys
from typing import List, Tuple, Optional

IN_DIR   = Path(r"your input path")
OUT_DIR  = Path(r"your output path")
IMG_EXTS = [".png"]
CLASS_NAME  = "meme"
TEST_RATIO  = 0.20                               
SEED        = 42                               

def _list_pairs(in_dir: Path, image_exts: List[str]) -> List[Tuple[Path, Path]]:
    images = {}
    xmls   = {}
    exts = {e.lower() for e in image_exts}
    for p in in_dir.iterdir():
        if not p.is_file():
            continue
        s = p.suffix.lower()
        if s == ".xml":
            xmls[p.stem] = p
        elif s in exts:
            images[p.stem] = p
    stems = sorted(set(images) & set(xmls))
    return [(images[s], xmls[s]) for s in stems]

def _parse_first_bbox(xml_path: Path, class_name: str) -> Tuple[int, int, Optional[Tuple[float,float,float,float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"<size> not found in {xml_path.name}")
    W = int(float(size.findtext("width")))
    H = int(float(size.findtext("height")))

    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name != class_name:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin")); ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax")); ymax = float(bb.findtext("ymax"))
        if xmax < xmin: xmin, xmax = xmax, xmin
        if ymax < ymin: ymin, ymax = ymax, ymin
        return W, H, (xmin, ymin, xmax, ymax)
    return W, H, None

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _voc_to_yolo(xmin, ymin, xmax, ymax, W, H) -> Tuple[float,float,float,float]:
    xmin = _clamp(xmin, 0, W - 1)
    ymin = _clamp(ymin, 0, H - 1)
    xmax = _clamp(xmax, 0, W - 1)
    ymax = _clamp(ymax, 0, H - 1)
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        raise ValueError("Invalid bbox: non-positive area")
    cx = xmin + bw/2.0
    cy = ymin + bh/2.0
    return (cx / W, cy / H, bw / W, bh / H)

def _ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def _write_label(path: Path, line: Optional[str]):
    path.write_text((line + "\n") if line else "", encoding="utf-8")

def _split(items: List[Tuple[Path,Path]], test_ratio: float, seed: int):
    rnd = random.Random(seed)
    items = items[:]
    rnd.shuffle(items)
    n_test = int(round(len(items) * test_ratio))
    return items[n_test:], items[:n_test] 

def main():
    if not IN_DIR.exists() or not IN_DIR.is_dir():
        print(f"[error] Input folder not found: {IN_DIR}")
        sys.exit(1)

    pairs = _list_pairs(IN_DIR, IMG_EXTS)
    print(f"[info] matched (image+xml): {len(pairs)}")
    if not pairs:
        print("[error] No matched pairs found. Check IN_DIR and IMG_EXTS.")
        sys.exit(1)

    train_set, test_set = _split(pairs, TEST_RATIO, SEED)
    print(f"[split] train={len(train_set)}  test={len(test_set)}")

    timg = OUT_DIR / "train" / "images"; tlab = OUT_DIR / "train" / "labels"
    vimg = OUT_DIR / "test"  / "images"; vlab = OUT_DIR / "test"  / "labels"
    for d in (timg, tlab, vimg, vlab): _ensure(d)

    def handle(img_p: Path, xml_p: Path, img_out: Path, lab_out: Path):
        W, H, bbox = _parse_first_bbox(xml_p, CLASS_NAME)
        if bbox is None:
            line = None
        else:
            xmin, ymin, xmax, ymax = bbox
            cx, cy, w, h = _voc_to_yolo(xmin, ymin, xmax, ymax, W, H)
            line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        _write_label(lab_out / f"{img_p.stem}.txt", line)
        shutil.copy2(img_p, img_out / img_p.name)

    for img_p, xml_p in train_set:
        handle(img_p, xml_p, timg, tlab)
    for img_p, xml_p in test_set:
        handle(img_p, xml_p, vimg, vlab)


    (tlab / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")
    (vlab / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")

    print(f"[done] train → {timg} / {tlab}")
    print(f"[done] test  → {vimg} / {vlab}")

if __name__ == "__main__":
    main()

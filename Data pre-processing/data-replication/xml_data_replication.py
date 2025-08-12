from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

def _read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text if obj.find("name") is not None else "object"
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        objects.append({"el": obj, "name": name, "bbox": (xmin, ymin, xmax, ymax)})
    return tree, root, W, H, objects

def _transform_bbox(bbox, W, H, mode="rot180"):
    xmin, ymin, xmax, ymax = bbox
    if mode == "hflip":
        nxmin = W - xmax
        nxmax = W - xmin
        nymin = ymin
        nymax = ymax
    elif mode == "rot180":
        nxmin = W - xmax
        nxmax = W - xmin
        nymin = H - ymax
        nymax = H - ymin
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    nxmin, nxmax = sorted((nxmin, nxmax))
    nymin, nymax = sorted((nymin, nymax))
    nxmin = max(0, min(W, int(round(nxmin))))
    nymin = max(0, min(H, int(round(nymin))))
    nxmax = max(0, min(W, int(round(nxmax))))
    nymax = max(0, min(H, int(round(nymax))))
    return nxmin, nymin, nxmax, nymax

def _update_xml(tree, root, W, H, objects_new, out_filename=None, out_path_str=None):
    if out_filename is not None:
        fn = root.find("filename")
        if fn is None:
            fn = ET.SubElement(root, "filename")
        fn.text = out_filename
    if out_path_str is not None:
        p = root.find("path")
        if p is None:
            p = ET.SubElement(root, "path")
        p.text = out_path_str
    size = root.find("size")
    size.find("width").text = str(W)
    size.find("height").text = str(H)
    
    for obj_el, new_bbox in zip(root.findall("object"), objects_new):
        bb = obj_el.find("bndbox")
        xmin, ymin, xmax, ymax = new_bbox
        bb.find("xmin").text = str(int(xmin))
        bb.find("ymin").text = str(int(ymin))
        bb.find("xmax").text = str(int(xmax))
        bb.find("ymax").text = str(int(ymax))
    return tree

def process_one(img_path, xml_path, out_dir, mode="rot180", suffix=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = Path(img_path)
    xml_path = Path(xml_path)
    if suffix is None:
        suffix = "_rot180" if mode == "rot180" else "_hflip"
    im = Image.open(img_path)
    if mode == "rot180":
        im_aug = im.rotate(180, expand=False)
    elif mode == "hflip":
        im_aug = im.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    out_img_name = img_path.stem + suffix + img_path.suffix
    out_img_path = out_dir / out_img_name
    im_aug.save(out_img_path)

    tree, root, W, H, objects = _read_xml(xml_path)
    transformed_bboxes = [_transform_bbox(o["bbox"], W, H, mode) for o in objects]

    out_xml_name = xml_path.stem + suffix + xml_path.suffix
    out_xml_path = out_dir / out_xml_name

    _update_xml(tree, root, W, H, transformed_bboxes, out_filename=out_img_name, out_path_str=str(out_img_path))
    tree.write(out_xml_path, encoding="utf-8", xml_declaration=False)

    return str(out_img_path), str(out_xml_path)

def process_folder(in_dir, out_dir, mode="rot180", img_exts=(".png", ".jpg", ".jpeg")):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_map = {p.stem: p for p in in_dir.glob("*.xml")}
    processed = []
    skipped = []

    for ext in img_exts:
        for img_path in in_dir.glob(f"*{ext}"):
            stem = img_path.stem
            xml_path = xml_map.get(stem)
            if not xml_path or not xml_path.exists():
                skipped.append(stem)
                continue
            out_img, out_xml = process_one(img_path, xml_path, out_dir, mode=mode)
            processed.append((img_path.name, Path(out_img).name, Path(out_xml).name))

    return processed, skipped

if __name__ == "__main__":
    process_folder("input path","output path")

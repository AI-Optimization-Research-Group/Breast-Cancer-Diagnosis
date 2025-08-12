import os
import cv2
import xml.etree.ElementTree as ET

input_dir  = r'input path' 
output_dir = r'output path' 

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith('.xml'):
        continue

    xml_path = os.path.join(input_dir, fname)
    tree     = ET.parse(xml_path)
    root     = tree.getroot()
    img_name = root.find('filename').text
    img_path = os.path.join(input_dir, img_name)
    if not os.path.isfile(img_path):
        print(f" image not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"image not found: {img_path}")
        continue

    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        xmin = int(bb.find('xmin').text)
        ymin = int(bb.find('ymin').text)
        xmax = int(bb.find('xmax').text)
        ymax = int(bb.find('ymax').text)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, img)
    print(f"save: {out_path}")

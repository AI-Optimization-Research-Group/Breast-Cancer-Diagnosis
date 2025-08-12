import os
import cv2

def draw_boxes_from_yolo_labels(image_path, label_path, output_path):
    
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        
        label, x_center, y_center, box_width, box_height = map(float, line.strip().split())

        
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  

   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def process_yolo_directory(yolo_root, output_root):
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(yolo_root, split, 'images')
        label_dir = os.path.join(yolo_root, split, 'labels')
        output_image_dir = os.path.join(output_root, split, 'images')
        output_label_dir = os.path.join(output_root, split, 'labels')
        
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        for file in os.listdir(label_dir):
            if file.endswith('.txt'):
                
                label_path = os.path.join(label_dir, file)
                image_path = os.path.join(image_dir, file.replace('.txt', '.png'))
                if not os.path.exists(image_path):
                    image_path = os.path.join(image_dir, file.replace('.txt', '.jpg'))
                if not os.path.exists(image_path):
                    continue  

                
                output_image_path = os.path.join(output_image_dir, file.replace('.txt', '.png'))

                
                draw_boxes_from_yolo_labels(image_path, label_path, output_image_path)

yolo_root = r'Your Path'
output_root = r'Your Path'

process_yolo_directory(yolo_root, output_root)

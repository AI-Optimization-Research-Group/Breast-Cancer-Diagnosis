import os
import xml.etree.ElementTree as ET
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def parse_xml(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes, labels = [], []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bnd = obj.find('bndbox')
        xmin = float(bnd.find('xmin').text)
        ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text)
        ymax = float(bnd.find('ymax').text)
        # basit validasyon
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)
    return boxes, labels


class VOCDataset(Dataset):
    def __init__(self, root_dir, class2idx, transforms=None):
        self.root = root_dir
        self.transforms = transforms
        self.class2idx = class2idx

        # sadece .xml dosyaları
        all_xmls = [f for f in os.listdir(self.root) if f.lower().endswith('.xml')]

        self.xml_files = []
        skipped_no_img = 0
        skipped_empty = 0
        for f in all_xmls:
            xml_path = os.path.join(self.root, f)
            try:
                root = ET.parse(xml_path).getroot()
                img_name_node = root.find('filename')
                if img_name_node is None or img_name_node.text is None:
                    continue
                img_name = img_name_node.text.strip()
                img_path = os.path.join(self.root, img_name)
                if not os.path.exists(img_path):
                    skipped_no_img += 1
                    continue
                boxes, labels = parse_xml(xml_path)
                if len(boxes) == 0:
                    skipped_empty += 1
                    continue
                # sınıf isimleri doğrulama
                if any(l not in self.class2idx for l in labels):
                    continue
                self.xml_files.append(f)
            except Exception:
                # bozuk xml vs.
                continue

        if len(self.xml_files) == 0:
            raise RuntimeError("XML Not Found")

        print(f"[VOCDataset] Toplam XML: {len(all_xmls)} | "
              f"Kullanılan: {len(self.xml_files)} | "
              f"Görüntüsü eksik: {skipped_no_img} | Boş anotasyon: {skipped_empty}")

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_name = self.xml_files[idx]
        xml_path = os.path.join(self.root, xml_name)

        root = ET.parse(xml_path).getroot()
        img_name = root.find('filename').text.strip()
        img_path = os.path.join(self.root, img_name)

        img = Image.open(img_path).convert('RGB')
        boxes, labels = parse_xml(xml_path)

        labels_idx = [self.class2idx[lbl] for lbl in labels]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels_idx, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transforms:
            img = self.transforms(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_data_loaders(root_dir, class2idx, transforms, test_split=0.1, batch_size=2, num_workers=0):

    dataset = VOCDataset(root_dir, class2idx, transforms)
    n_test = max(1, int(len(dataset) * test_split))
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader

def get_model(num_classes, device, lr):
    # torchvision sürümüne göre 'pretrained=True' çalışır; yeni sürümlerde 'weights' argümanı var.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    return model, optimizer


def train_one_epoch(model, optimizer, loader, device):
    model.train()
    for images, targets in loader:
        # emniyet kemeri: boş target'lı örnekleri at
        kept_images, kept_targets = [], []
        for img, tgt in zip(images, targets):
            if 'boxes' in tgt and isinstance(tgt['boxes'], torch.Tensor) and tgt['boxes'].numel() >= 4:
                kept_images.append(img)
                kept_targets.append(tgt)
        if not kept_images:
            continue

        images = [img.to(device) for img in kept_images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in kept_targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(model, loader, device, iou_thresh=0.5, score_thresh=0.5):
    model.eval()
    all_preds, all_gts = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for tgt, out in zip(targets, outputs):
            gt_boxes = tgt['boxes']       
            gt_labels = tgt['labels']      

            preds = out
            keep = preds['scores'] >= score_thresh
            pb = preds['boxes'][keep].cpu()
            pl = preds['labels'][keep].cpu()

            matched = set()
            for b, label in zip(pb, pl):
                ious = box_iou(b.unsqueeze(0), gt_boxes)
                max_iou, idx = torch.max(ious, 1)

                if max_iou.item() >= iou_thresh and label.item() == gt_labels[idx].item() and idx.item() not in matched:
                    all_preds.append(1); all_gts.append(1)
                    matched.add(idx.item())
                else:

                    all_preds.append(1); all_gts.append(0)


            for i in range(len(gt_boxes)):
                if i not in matched:
                    all_preds.append(0); all_gts.append(1)

    prec, rec, f1, _ = precision_recall_fscore_support(all_gts, all_preds, average='binary', zero_division=0)
    acc = accuracy_score(all_gts, all_preds)
    return acc, prec, rec, f1


if __name__ == "__main__":

    ROOT_DIR    = r'C:\Users\34mam\memesegment'  
    OUTPUT_DIR  = r'models'
    CLASS_NAMES = ['_background_', 'meme']      
    TEST_SPLIT  = 0.2
    BATCH_SIZE  = 2
    NUM_WORKERS = 0                             
    LR          = 0.005
    NUM_EPOCHS  = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class2idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    num_classes = len(CLASS_NAMES)

    transforms = T.Compose([T.ToTensor()])


    train_loader, test_loader = get_data_loaders(
        ROOT_DIR, class2idx, transforms,
        test_split=TEST_SPLIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    model, optimizer = get_model(num_classes, device, LR)

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} tamamlandı.")


    acc, prec, rec, f1 = evaluate(model, test_loader, device)
    print(f"Test -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'faster_rcnn.pth'))
    print(f"Model kaydedildi: {os.path.join(OUTPUT_DIR, 'faster_rcnn.pth')}")

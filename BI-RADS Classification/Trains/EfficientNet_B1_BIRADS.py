import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import random
import time  
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is:", device)

NUM_EPOCHS = 1
BATCH_SIZE = 64
lr = 0.0001

root =  r"C:\memekanser\memeveri"

base_dataset = ImageFolder(root=str(root))
targets = np.array(base_dataset.targets)
indices = np.arange(len(base_dataset))

sss = StratifiedShuffleSplit(n_splits=1, test_size= 0.15, random_state=42)
train_idx, test_idx = next(sss.split(indices, targets))
print(f"[INFO] Train: {len(train_idx)} | Test: {len(test_idx)} | Toplam: {len(base_dataset)}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = T.Compose([
    T.Resize((16, 16)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


train_dataset = ImageFolder(root=str(root), transform=train_transform)
test_dataset  = ImageFolder(root=str(root), transform=test_transform)
train_subset = Subset(train_dataset, train_idx.tolist())
test_subset  = Subset(test_dataset,  test_idx.tolist())

class_to_idx: Dict[str, int] = train_dataset.class_to_idx
idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)
print("[INFO] class_to_idx:", class_to_idx)


pin_mem = (device.type == "cuda")
train_loader = DataLoader(train_subset, batch_size= 16, shuffle=True,
                          num_workers=0)
test_loader  = DataLoader(test_subset,  batch_size=16, shuffle=False,
                          num_workers= 0)

weights = EfficientNet_B1_Weights.IMAGENET1K_V2
model = efficientnet_b1(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

COUNTER = 1

acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for run in range(COUNTER):
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    t0 = time.perf_counter()


    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        f1_train = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}, F1 Score: {f1_train:.4f}")
        
        
    elapsed = time.perf_counter() - t0
    m, s = divmod(elapsed, 60)
    print(f"\n[Time] {NUM_EPOCHS} epoch : {int(m):02d}:{s:04.1f} dk:sn")


    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(targets.squeeze().long().cpu().numpy())
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)


    precision = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    recall    = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    f1        = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    acc = accuracy_score(test_labels, test_preds)

    print(f"--> {run+1}. Eğitim Sonu Test Metrikleri:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")


    acc_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)



print(f" Accuracy:  {np.mean(acc_scores):.4f}")
print(f" Precision: {np.mean(precision_scores):.4f}")
print(f" Recall:    {np.mean(recall_scores):.4f}")
print(f" F1 Score:  {np.mean(f1_scores):.4f}")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import random
import time
import os
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

DATA_DIR = './kendi_meme_kanseri_veri_seti_klasor_yolu' 


NUM_EPOCHS = 50
BATCH_SIZE = 32
lr = 0.001
COUNTER = 1

n_channels = 3  
n_classes = 4   
IMG_SIZE = 260 





data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=data_transform)
    test_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=data_transform)
    
    print(f"Eğitim Setindeki Sınıf İsimleri: {train_dataset.classes}")
    
    if len(train_dataset.classes) != n_classes:
         print(f"[UYARI] Tanımlanan sınıf sayısı ({n_classes}) ile veri setindeki sınıf sayısı ({len(train_dataset.classes)}) uyuşmuyor.")


except FileNotFoundError:
    print(f"\n[HATA] Veri setleri '{DATA_DIR}/train' ve '{DATA_DIR}/test' klasörlerinde bulunamadı.")
    print("Lütfen DATA_DIR değişkenini ve klasör yapısını kontrol edin.")
    exit()

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)



class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        
        in_features = self.model._fc.in_features
        
        self.model._fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


for run in range(COUNTER):
    
    model = EfficientNetModel(num_classes=n_classes).to(device)    

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

            loss = criterion(outputs, targets.long())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        f1_train = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss / len(train_loader):.4f}, F1 Score (Train Weighted): {f1_train:.4f}")
        
        
    elapsed = time.perf_counter() - t0
    m, s = divmod(elapsed, 60)
    print(f"\n[Çalışma Süresi] {NUM_EPOCHS} epoch toplam: {int(m):02d}:{s:04.1f} dk:sn")


    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(targets.long().cpu().numpy())
            

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)


    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall    = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1_        = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    acc = accuracy_score(test_labels, test_preds)

    print(f"--> {run+1}. Eğitim Sonu Test Metrikleri:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {precision:.4f} (Weighted)")
    print(f"    Recall:    {recall:.4f} (Weighted)")
    print(f"    F1 Score:  {f1_:.4f} (Weighted)")


    acc_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1_)


print("\n====== Tüm Eğitimlerin Ortalama Test Metrikleri ======")
print(f"Ortalama Accuracy:  {np.mean(acc_scores):.4f}")
print(f"Ortalama Precision: {np.mean(precision_scores):.4f} (Weighted)")
print(f"Ortalama Recall:    {np.mean(recall_scores):.4f} (Weighted)")
print(f"Ortalama F1 Score:  {np.mean(f1_scores):.4f} (Weighted)")

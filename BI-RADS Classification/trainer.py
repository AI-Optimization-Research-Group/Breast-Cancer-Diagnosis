import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from config import NUM_EPOCHS, LEARNING_RATE


def train_epoch(model, train_loader, criterion, optimizer, device):
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

    avg_loss = running_loss / len(train_loader)
    f1_train = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, f1_train


def evaluate_model(model, test_loader, device):
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


    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n{'='*60}")
    print(f"Eğitim Başlıyor: {num_epochs} Epoch")
    print(f"{'='*60}")
    
    t0 = time.perf_counter()

    for epoch in range(num_epochs):
        train_loss, f1_train = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, F1 Score (Train Weighted): {f1_train:.4f}")

    elapsed = time.perf_counter() - t0
    m, s = divmod(elapsed, 60)
    print(f"\n[Çalışma Süresi] {num_epochs} epoch toplam: {int(m):02d}:{s:04.1f} dk:sn")

    
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Test Metrikleri:")
    print(f"{'='*60}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f} (Weighted)")
    print(f"Recall:    {test_metrics['recall']:.4f} (Weighted)")
    print(f"F1 Score:  {test_metrics['f1_score']:.4f} (Weighted)")
    print(f"{'='*60}\n")
    
    return test_metrics

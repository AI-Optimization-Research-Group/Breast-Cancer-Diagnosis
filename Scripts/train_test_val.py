import os
import shutil
import random

# ================= AYARLAR =================
# Veri setinizin olduğu ana klasör yolu
DATA_DIR = r'C:\memekanser\birads\data' 

# Test veri seti oranı (%25 test, %75 train)
TEST_RATIO = 0.25 
# ===========================================

def split_dataset_by_copying(root_path, test_ratio):
    """
    Orijinal klasörlere dokunmadan, dosyaları KOPYALAYARAK
    train ve test klasörleri oluşturur.
    """
    
    train_root = os.path.join(root_path, 'train')
    test_root = os.path.join(root_path, 'test')

    # Eğer train veya test klasörü zaten varsa, üzerine yazmamak için durdur.
    if os.path.exists(train_root) or os.path.exists(test_root):
        print(f"[UYARI] '{train_root}' veya '{test_root}' zaten mevcut.")
        print("Lütfen önce eski 'train' ve 'test' klasörlerini silin, sonra bu kodu çalıştırın.")
        return

    # Sadece klasörleri listele (train ve test isimli olanlar hariç)
    classes = [d for d in os.listdir(root_path) 
               if os.path.isdir(os.path.join(root_path, d)) 
               and d not in ['train', 'test']]

    if not classes:
        print("[HATA] Ayrıştırılacak sınıf klasörü bulunamadı.")
        return

    print(f"Tespit edilen sınıflar: {classes}")
    print("Kopyalama işlemi başlıyor, bu işlem disk hızınıza göre biraz sürebilir...\n")

    for class_name in classes:
        class_dir = os.path.join(root_path, class_name)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Resimleri karıştır
        random.shuffle(images)
        
        # Bölme noktasını belirle
        split_idx = int(len(images) * test_ratio)
        test_images = images[:split_idx]
        train_images = images[split_idx:]
        
        # Yeni klasör yollarını oluştur
        train_class_dir = os.path.join(train_root, class_name)
        test_class_dir = os.path.join(test_root, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # --- TRAIN İÇİN KOPYALA ---
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst) # copy2 meta veriyi de korur
            
        # --- TEST İÇİN KOPYALA ---
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
            
        print(f" -> {class_name}: {len(train_images)} adet Train, {len(test_images)} adet Test klasörüne KOPYALANDI.")

    print("\n[TAMAMLANDI] İşlem bitti. Orijinal dosyalarınız olduğu yerde duruyor.")
    print(f"Yeni klasörler şurada: \n - {train_root} \n - {test_root}")

if __name__ == "__main__":
    split_dataset_by_copying(DATA_DIR, TEST_RATIO)
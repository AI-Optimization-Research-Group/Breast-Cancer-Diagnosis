
import os
import pydicom
import numpy as np
from PIL import Image

def dcm_to_png_preserve_size(
    dcm_path: str,
    png_path: str,
    anonymize: bool = False
):
    """
    1) DICOM'u okur; JPEG2000/RLE gibi sıkıştırılmışsa açar.
    2) Ham pixel_array'i alır; MONOCHROME1 ise tersler.
    3) BitsAllocated & SamplesPerPixel okur.
    4) 8-bit veya 16-bit, gri veya RGB olarak kaydeder.
    5) (Opsiyonel) Hasta verilerini siler.
    """

    # --- 1) Oku ve dekompresyon ---
    ds = pydicom.dcmread(dcm_path, force=True)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()

    # --- 2) Ham pikselleri al & fotometrik düzeltme ---
    arr = ds.pixel_array
    # MONOCHROME1: 0 → beyaz, max → siyah olduğundan tersle
    if ds.PhotometricInterpretation == "MONOCHROME1":
        arr = np.max(arr) - arr

    # --- 3) Bit derinliği & kanal sayısı ---
    bits     = int(ds.BitsAllocated)                   # genellikle 8 veya 16
    channels = int(getattr(ds, "SamplesPerPixel", 1))  # genellikle 1 veya 3

    # --- 4) Doğru dtype & PIL mode seçimi ---
    if bits <= 8:
        arr = arr.astype(np.uint8)
        mode = "L"    if channels == 1 else "RGB"
    else:
     # 16-bit → 8-bit'e basitçe map et (clip+scale)
     # # aralıktaki min–max'ı al, 0–255'e ölçekle
     vmin, vmax = arr.min(), arr.max()
     arr8 = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
     img = Image.fromarray(arr8)   # otomatik "L" veya "RGB"
     img.save(png_path)
     return


    # --- 5) (İsteğe bağlı) Anonimleştir ---
    if anonymize:
        ds.remove_private_tags()
        for tag in ("PatientName","PatientID",
                    "PatientBirthDate","PatientSex"):
            ds.pop(tag, None)

    # --- 6) Kaydet ---
    Image.fromarray(arr, mode=mode).save(png_path)



# === Kullanım Örneği ===
if __name__ == "__main__":
    src = r"/Users/melekaltun/Downloads/RCC.dcm"
    dst = r"/Users/melekaltun/Downloads/test.png"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    dcm_to_png_preserve_size(src, dst, anonymize=False)
    print("Dönüştürüldü:", dst)

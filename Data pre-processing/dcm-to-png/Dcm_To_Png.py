
import os
import pydicom
import numpy as np
from PIL import Image

def dcm_to_png_preserve_size(
    dcm_path: str,
    png_path: str,
    anonymize: bool = False
):

    ds = pydicom.dcmread(dcm_path, force=True)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()

    arr = ds.pixel_array
    if ds.PhotometricInterpretation == "MONOCHROME1":
        arr = np.max(arr) - arr

    bits     = int(ds.BitsAllocated)                  
    channels = int(getattr(ds, "SamplesPerPixel", 1))  

    if bits <= 8:
        arr = arr.astype(np.uint8)
        mode = "L"    if channels == 1 else "RGB"
    else:
     vmin, vmax = arr.min(), arr.max()
     arr8 = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)
     img = Image.fromarray(arr8) 
     img.save(png_path)
     return

    if anonymize:
        ds.remove_private_tags()
        for tag in ("PatientName","PatientID",
                    "PatientBirthDate","PatientSex"):
            ds.pop(tag, None)

    Image.fromarray(arr, mode=mode).save(png_path)


if __name__ == "__main__":
    src = r"/Users/melekaltun/Downloads/RCC.dcm"
    dst = r"/Users/melekaltun/Downloads/test.png"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    dcm_to_png_preserve_size(src, dst, anonymize=False)
    print("Dönüştürüldü:", dst)

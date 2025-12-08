import os
import random
import shutil
from pathlib import Path

DATASET_ROOT = Path("/Users/melekaltun/Desktop/dataset")

OUTPUT_ROOT = Path("binary_birads")

CLASSES = ["birads1", "birads2", "birads4", "birads5"]

VALID_EXTS = {".png"}


def list_images(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )


def create_binary_for_target(split_root: Path, split_name: str, target_class: str):
    print(f"\n=== Split: {split_name} | Target: {target_class} ===")

    pos_dir_in = split_root / target_class
    pos_files = list_images(pos_dir_in)
    n_pos = len(pos_files)

    if n_pos == 0:
        print(f"Uyarı: {split_name}/{target_class} içinde görüntü yok, atlanıyor.")
        return

    print(f"Pozitif sınıf ({target_class}) görüntü sayısı: {n_pos}")

    other_classes = [c for c in CLASSES if c != target_class]
    other_files_dict = {c: list_images(split_root / c) for c in other_classes}

    for c in other_classes:
        print(f"{split_name}/{c} görüntü sayısı: {len(other_files_dict[c])}")

    num_other_classes = len(other_classes) 
    per_class_base = n_pos // num_other_classes
    remainder = n_pos % num_other_classes

    for c in other_classes:
        required_min = per_class_base 
        if len(other_files_dict[c]) < required_min:
            raise ValueError(
                f"{split_name}/{c} klasöründe yeterli görüntü yok.\n"
                f"Hedef sınıf: {target_class}, Pozitif sayısı: {n_pos}\n"
                f"Her negatif sınıftan en az {required_min} görüntü gerekiyor, "
                f"mevcut: {len(other_files_dict[c])}"
            )

    sampled_neg_files = []
    for i, c in enumerate(other_classes):
        n_to_sample = per_class_base + (1 if i < remainder else 0)

        if len(other_files_dict[c]) < n_to_sample:
            raise ValueError(
                f"{split_name}/{c} klasöründe istenenden az görüntü var.\n"
                f"İstenen: {n_to_sample}, mevcut: {len(other_files_dict[c])}"
            )

        sampled = random.sample(other_files_dict[c], n_to_sample)
        sampled_neg_files.extend(sampled)
        print(f"{split_name}/{c} -> seçilen negatif sayısı: {len(sampled)}")

    print(
        f"Toplam negatif (non_{target_class}) sayısı: "
        f"{len(sampled_neg_files)} (pozitif ile eşit olmalı: {n_pos})"
    )

    split_out_root = OUTPUT_ROOT / split_name
    pos_out_dir = split_out_root / target_class
    neg_out_dir = split_out_root / f"non_{target_class}"

    pos_out_dir.mkdir(parents=True, exist_ok=True)
    neg_out_dir.mkdir(parents=True, exist_ok=True)

    for src_path in pos_files:
        dst_path = pos_out_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    for src_path in sampled_neg_files:
        parent_class = src_path.parent.name
        new_name = f"{src_path.stem}_{parent_class}{src_path.suffix}"
        dst_path = neg_out_dir / new_name
        shutil.copy2(src_path, dst_path)

    print(
        f"{split_name} için {target_class} / non_{target_class} oluşturuldu. "
        f"Pozitif: {n_pos}, Negatif: {len(sampled_neg_files)}"
    )


def main():
    for split_name in ["train", "test"]:
        split_root = DATASET_ROOT / split_name

        if not split_root.exists():
            print(f"Uyarı: {split_root} bulunamadı, bu split atlanıyor.")
            continue

        for target_class in CLASSES:
            create_binary_for_target(split_root, split_name, target_class)


if __name__ == "__main__":
    main()
    print(f"Output: {OUTPUT_ROOT.resolve()}")

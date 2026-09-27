"""
Converte anotações do LabelMe (.json, polígonos/retângulos) para o formato
de rótulo do YOLO/Darknet:  <classe> <x_center> <y_center> <largura> <altura>
(todos os valores normalizados entre 0 e 1).

Uso:
    python labelme_to_yolo.py --labelme_dir data/raw_labelme --out_dir data/yolo_dataset
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_classes(classes_path: str):
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def convert_one(json_path: Path, classes: list, images_out: Path, labels_out: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    img_name = Path(data["imagePath"]).name

    src_img = json_path.parent / img_name
    if src_img.exists():
        shutil.copy(src_img, images_out / img_name)

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in classes:
            # classe não cadastrada em classes.txt -> ignora com aviso
            print(f"[aviso] classe '{label}' não está em classes.txt, ignorando shape em {json_path.name}")
            continue
        class_id = classes.index(label)
        x_min, y_min, x_max, y_max = polygon_to_bbox(shape["points"])

        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_file = labels_out / (Path(img_name).stem + ".txt")
    with open(label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelme_dir", required=True, help="pasta com os .json do LabelMe")
    parser.add_argument("--out_dir", required=True, help="pasta de saída no formato YOLO")
    parser.add_argument("--classes", default="data/classes.txt")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    classes = load_classes(args.classes)

    labelme_dir = Path(args.labelme_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    json_files = sorted(labelme_dir.glob("*.json"))
    if not json_files:
        print(f"Nenhum .json encontrado em {labelme_dir}")
        return

    n_val = max(1, int(len(json_files) * args.val_split))
    val_files = set(json_files[:n_val])

    for jf in json_files:
        split = "val" if jf in val_files else "train"
        convert_one(
            jf,
            classes,
            out_dir / "images" / split,
            out_dir / "labels" / split,
        )

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: c for i, c in enumerate(classes)},
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"Convertidos {len(json_files)} arquivos. data.yaml gerado em {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

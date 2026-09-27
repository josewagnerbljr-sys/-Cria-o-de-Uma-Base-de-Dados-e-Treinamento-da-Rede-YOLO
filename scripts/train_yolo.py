"""
Treinamento por transfer learning: parte de um YOLOv8 já pré-treinado no COCO
(80 classes) e retreina incluindo as 2 classes novas (capacete, extintor),
sem perder o conhecimento das classes originais.

Uso:
    python train_yolo.py --data data/yolo_dataset/data.yaml --epochs 100 --base yolov8n.pt
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="caminho para data.yaml")
    parser.add_argument("--base", default="yolov8n.pt", help="pesos pré-treinados (transfer learning)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--freeze", type=int, default=10,
                         help="nº de camadas iniciais (backbone) a congelar para preservar o conhecimento do COCO")
    args = parser.parse_args()

    # Modelo pré-treinado no COCO (80 classes) -> ponto de partida do transfer learning
    model = YOLO(args.base)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        freeze=args.freeze,   # congela o backbone -> mantém features já aprendidas no COCO
        patience=20,
        pretrained=True,
        verbose=True,
    )

    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print(f"Pesos finais salvos em {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

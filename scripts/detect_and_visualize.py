"""
Roda a detecção com o modelo treinado e desenha as caixas delimitadoras
(reprodução da "Figura 1" do enunciado do desafio).

Uso:
    python detect_and_visualize.py --weights runs/train/exp/weights/best.pt --source amostra.jpg
"""
import argparse

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="imagem, vídeo ou 0 para webcam")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", default="outputs/deteccao_resultado.jpg")
    args = parser.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source == "0" else args.source

    results = model.predict(source=source, conf=args.conf, verbose=False)

    for r in results:
        annotated = r.plot()  # já desenha bbox + label + score, como na Figura 1
        cv2.imwrite(args.out, annotated)
        print(f"Resultado salvo em {args.out}")
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"  -> {cls_name}: {conf:.2f}")
        break  # apenas a primeira imagem no modo simples


if __name__ == "__main__":
    main()

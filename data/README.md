# Dataset

## Opção A — Rotular do zero com LabelMe
1. Instale o LabelMe: `pip install labelme`
2. Rotule as imagens desenhando retângulos/polígonos em torno de `capacete` e `extintor`.
3. O LabelMe salva um `.json` por imagem em `data/raw_labelme/`.
4. Rode `scripts/labelme_to_yolo.py` para converter para o formato YOLO (`.txt`).

## Opção B — Usar o COCO já rotulado
Baixe um subconjunto do COCO (ex.: classe `person`) com a própria API do Ultralytics:

```python
from ultralytics.utils.downloads import download
download(['https://ultralytics.com/assets/coco8.zip'])
```

O `coco8` já vem pronto no formato YOLO (imagem + `.txt`), ideal para testar o pipeline rapidamente antes de subir para o dataset completo.

## Mesclando COCO + classes novas
`data/classes.txt` lista as 80 classes originais do COCO (índices 0–79) seguidas das 2 classes novas
(índice 80 = `capacete`, 81 = `extintor`). O `data.yaml` gerado por `train_yolo.py` aponta para essa
lista combinada, permitindo transfer learning sem esquecer o que já foi aprendido.

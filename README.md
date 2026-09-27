# Desafio DIO — Rotulação de Dataset + Treinamento YOLO (com evoluções avançadas)

Projeto de conclusão do desafio **"Criação de uma base de dados e treinamento da rede YOLO"**.

## 1. O que o desafio pedia

- Rotular uma base de imagens (LabelMe) **ou** usar o COCO já rotulado.
- Treinar/retreinar a rede **YOLO** (Darknet) ou aplicar **transfer learning** (Colab) para quem não tem GPU local.
- Ter **no mínimo 2 classes novas**, além das classes já pré-treinadas.

## 2. O que foi entregue aqui

Implementação moderna usando **YOLOv8 (Ultralytics)** — sucessor direto do Darknet/YOLO original citado no desafio, mantendo total compatibilidade conceitual (mesma lógica de anchors/grid, formato de rótulo `.txt` idêntico ao Darknet: `classe x_center y_center largura altura` normalizados).

Duas classes novas foram escolhidas com cenário real de aplicação — **segurança do trabalho industrial**:

| ID | Classe nova       | Motivo                                              |
|----|--------------------|------------------------------------------------------|
| 80 | `capacete`         | Detectar uso de EPI (equipamento de proteção)        |
| 81 | `extintor`         | Detectar presença/ausência de equipamento de segurança |

Essas 2 classes são treinadas **junto** com um subconjunto das 80 classes originais do COCO (ex.: `person`), preservando o conhecimento pré-treinado (transfer learning real, não catastrophic forgetting).

### Estrutura do projeto

```
desafio-dio-yolo-deteccao-objetos/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md              # como organizar/baixar o dataset
│   └── classes.txt            # lista de classes (80 COCO + 2 novas)
├── scripts/
│   ├── labelme_to_yolo.py     # converte anotações do LabelMe -> formato YOLO
│   ├── train_yolo.py          # transfer learning (YOLOv8, GPU local ou Colab)
│   └── detect_and_visualize.py# inferência + desenho das bounding boxes (Figura 1)
├── advanced/
│   ├── vision_2d_3d_4d_adapter.py   # EVOLUÇÃO 1
│   └── automation_safety_system.py  # EVOLUÇÃO 2
└── notebook/
    └── colab_transfer_learning.ipynb
```

## 3. Como rodar (resumo rápido)

```bash
pip install -r requirements.txt

# 1) Converter rótulos do LabelMe para o formato YOLO
python scripts/labelme_to_yolo.py --labelme_dir data/raw_labelme --out_dir data/yolo_dataset

# 2) Treinar (transfer learning a partir do yolov8n.pt pré-treinado no COCO)
python scripts/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 100 --base yolov8n.pt

# 3) Rodar detecção numa imagem/vídeo (gera a "Figura 1" do enunciado)
python scripts/detect_and_visualize.py --weights runs/train/exp/weights/best.pt --source amostra.jpg
```

Para quem não tem GPU (situação prevista no próprio enunciado): usar `notebook/colab_transfer_learning.ipynb`, que roda 100% no Google Colab gratuito.

---

## 4. Evoluções de alto nível (muito além do exigido)

O desafio pedia apenas detecção 2D com 2 classes novas. Foram entregues **duas camadas adicionais** de engenharia que elevam o projeto de "exercício de laboratório" para "produto":

### 🚀 Evolução 1 — Adaptador de Visão 2D → 3D → 4D
Arquivo: `advanced/vision_2d_3d_4d_adapter.py`

Pipeline que pega a saída 2D do YOLO (bounding boxes) e adiciona progressivamente novas dimensões:

- **2D**: bounding boxes clássicas do YOLO (x, y, largura, altura).
- **3D**: estimativa de **profundidade monocular** (modelo MiDaS via `torch.hub`) aplicada dentro de cada bbox, convertendo cada objeto detectado em uma posição aproximada `(X, Y, Z)` no espaço, usando um modelo de câmera pinhole simplificado.
- **4D**: rastreamento temporal (tracking) de cada objeto entre frames com um filtro de Kalman leve, adicionando a **dimensão tempo** — calcula velocidade, direção de deslocamento e prediz a posição `(X, Y, Z)` no frame seguinte (útil para robótica, veículos autônomos, análise de trajetória de pessoas em chão de fábrica).

Ou seja: o mesmo YOLO 2D passa a alimentar um mapa espaço-temporal 3D dinâmico, sem precisar de câmera estéreo ou LiDAR.

### 🚀 Evolução 2 — Sistema de Automação de Segurança Orientado a Eventos
Arquivo: `advanced/automation_safety_system.py`

Motor de regras que transforma detecções brutas em **ações automáticas**:

- Regras configuráveis (ex.: `person` detectada sem `capacete` próximo por N frames consecutivos → gera alerta `ALTA` severidade).
- **Debounce/anti-flood**: evita alertas duplicados a cada frame.
- Sistema de notificação plugável: console, webhook (HTTP POST simulado) e log estruturado em JSON/CSV pronto para alimentar um dashboard (Grafana/Power BI/Metabase).
- Máquina de estados por objeto rastreado (novo → observado → confirmado → alerta → resolvido).

Isso transforma o projeto de "detector de imagem" em um **sistema de automação industrial em tempo real** (visão computacional + regras de negócio + automação de resposta).

---

## 5. Créditos
Baseado no material da trilha DIO (Deep Learning / YOLO), com dataset de referência COCO (https://cocodataset.org) e ferramenta de rotulação LabelMe (http://labelme.csail.mit.edu/Release3.0/).

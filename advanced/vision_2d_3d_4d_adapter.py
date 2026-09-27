"""
EVOLUÇÃO 1 — Adaptador de Visão 2D -> 3D -> 4D
================================================
Vai muito além do que o desafio pede (apenas bbox 2D). Este módulo:

  2D  -> recebe as detecções do YOLO (bounding boxes)
  3D  -> estima profundidade monocular (MiDaS, via torch.hub) e converte
         cada bbox em uma posição aproximada (X, Y, Z) no espaço, usando
         um modelo de câmera pinhole simplificado (sem precisar de câmera
         estéreo nem LiDAR)
  4D  -> rastreia cada objeto ao longo do tempo com um filtro de Kalman
         leve (posição + velocidade em X, Y, Z), permitindo prever a
         posição futura -> a 4ª dimensão é o tempo

Uso típico:
    from ultralytics import YOLO
    from vision_2d_3d_4d_adapter import Vision2D3D4DAdapter

    yolo = YOLO("runs/train/exp/weights/best.pt")
    adapter = Vision2D3D4DAdapter(focal_length_px=800)

    for frame in video_frames:
        result = yolo.predict(frame, verbose=False)[0]
        tracked_objects = adapter.process_frame(frame, result)
        for obj in tracked_objects:
            print(obj.class_name, obj.position_3d, obj.velocity_3d, obj.predicted_position_3d)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # torch é obrigatório em produção; deixado leve para leitura do código
    torch = None


@dataclass
class TrackedObject3D:
    track_id: int
    class_name: str
    confidence: float
    bbox_2d: Tuple[float, float, float, float]      # x1, y1, x2, y2
    position_3d: Tuple[float, float, float]         # X, Y, Z estimados (metros aprox.)
    velocity_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    predicted_position_3d: Tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    age_frames: int = 0


class _KalmanTrack3D:
    """Filtro de Kalman simples para (X, Y, Z) com modelo de velocidade constante.
    Implementado manualmente (sem dependência externa) para deixar a lógica 4D explícita."""

    def __init__(self, initial_pos: np.ndarray, dt: float = 1 / 30):
        self.dt = dt
        # estado: [X, Y, Z, Vx, Vy, Vz]
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
        self.P = np.eye(6) * 10.0

        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt

        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        self.Q = np.eye(6) * 0.05      # ruído do processo
        self.R = np.eye(3) * 0.3       # ruído de medição

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, measured_pos: np.ndarray):
        y = measured_pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]


class Vision2D3D4DAdapter:
    def __init__(self, focal_length_px: float = 800.0, max_track_age: int = 30):
        self.focal_length_px = focal_length_px
        self.max_track_age = max_track_age
        self._depth_model = None
        self._depth_transform = None
        self._tracks: Dict[int, _KalmanTrack3D] = {}
        self._track_meta: Dict[int, dict] = {}
        self._next_id = 0
        self._last_time = time.time()

    # ---------- Camada 3D: profundidade monocular ----------
    def _load_depth_model(self):
        if self._depth_model is not None:
            return
        if torch is None:
            raise RuntimeError("PyTorch é necessário para a estimativa de profundidade (MiDaS).")
        self._depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self._depth_model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._depth_transform = midas_transforms.small_transform

    def _estimate_depth_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._load_depth_model()
        input_batch = self._depth_transform(frame_bgr)
        with torch.no_grad():
            prediction = self._depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_bgr.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    def _bbox_to_3d(self, bbox, depth_map, frame_shape) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        cx_px, cy_px = (x1 + x2) / 2, (y1 + y2) / 2

        region = depth_map[int(y1):int(y2), int(x1):int(x2)]
        rel_depth = float(np.median(region)) if region.size else float(np.median(depth_map))
        # MiDaS retorna profundidade relativa (inverso); normalizamos para uma
        # pseudo-distância em metros com uma calibração simples e configurável
        z_meters = 10.0 / (rel_depth + 1e-6)
        z_meters = float(np.clip(z_meters, 0.3, 50.0))

        # Projeção pinhole inversa: X = (u - cx)*Z/f , Y = (v - cy)*Z/f
        principal_x, principal_y = w / 2, h / 2
        x_meters = (cx_px - principal_x) * z_meters / self.focal_length_px
        y_meters = (cy_px - principal_y) * z_meters / self.focal_length_px
        return (x_meters, y_meters, z_meters)

    # ---------- Camada 4D: tracking temporal ----------
    def _match_or_create_track(self, position_3d: np.ndarray) -> int:
        best_id, best_dist = None, 1.5  # limiar em metros
        for tid, track in self._tracks.items():
            dist = float(np.linalg.norm(track.position - position_3d))
            if dist < best_dist:
                best_dist, best_id = dist, tid
        if best_id is not None:
            return best_id
        new_id = self._next_id
        self._next_id += 1
        self._tracks[new_id] = _KalmanTrack3D(position_3d)
        self._track_meta[new_id] = {"age": 0}
        return new_id

    def process_frame(self, frame_bgr: np.ndarray, yolo_result) -> List[TrackedObject3D]:
        now = time.time()
        dt = max(now - self._last_time, 1 / 60)
        self._last_time = now

        depth_map = self._estimate_depth_map(frame_bgr)
        output: List[TrackedObject3D] = []
        seen_ids = set()

        for box in yolo_result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            class_name = yolo_result.names[int(box.cls[0])]

            pos_3d = np.array(self._bbox_to_3d((x1, y1, x2, y2), depth_map, frame_bgr.shape))

            track_id = self._match_or_create_track(pos_3d)
            track = self._tracks[track_id]
            track.F[:3, 3:] = np.eye(3) * dt  # atualiza dt no modelo de movimento
            track.predict()
            track.update(pos_3d)
            self._track_meta[track_id]["age"] = 0
            seen_ids.add(track_id)

            predicted_next = track.position + track.velocity * dt

            output.append(TrackedObject3D(
                track_id=track_id,
                class_name=class_name,
                confidence=conf,
                bbox_2d=(x1, y1, x2, y2),
                position_3d=tuple(track.position),
                velocity_3d=tuple(track.velocity),
                predicted_position_3d=tuple(predicted_next),
                age_frames=self._track_meta[track_id]["age"],
            ))

        # limpeza de tracks não vistos (evita memória infinita / IDs fantasmas)
        for tid in list(self._tracks.keys()):
            if tid not in seen_ids:
                self._track_meta[tid]["age"] += 1
                if self._track_meta[tid]["age"] > self.max_track_age:
                    del self._tracks[tid]
                    del self._track_meta[tid]

        return output

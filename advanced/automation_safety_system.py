"""
EVOLUÇÃO 2 — Sistema de Automação de Segurança Orientado a Eventos
====================================================================
Transforma detecções brutas do YOLO em AÇÕES automáticas: alertas, webhooks
e logs estruturados prontos para dashboard. Muito além do "apenas detectar
e mostrar a caixa" pedido no desafio.

Componentes:
  - Regra (Rule): condição lógica sobre as classes detectadas
  - Máquina de estados por objeto: novo -> observado -> confirmado -> alerta -> resolvido
  - Debounce: evita repetir o mesmo alerta a cada frame
  - Notificador plugável: console / webhook (HTTP) / arquivo JSON-lines (para BI)

Uso típico:
    system = SafetyAutomationSystem()
    system.add_rule(Rule(
        name="pessoa_sem_capacete",
        required_classes={"person"},
        forbidden_nearby_classes={"capacete"},
        severity="ALTA",
        min_consecutive_frames=5,
    ))

    for frame_detections in stream_de_deteccoes:
        events = system.process(frame_detections)
        for ev in events:
            print(ev)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

try:
    import requests
except ImportError:
    requests = None


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    track_id: Optional[int] = None


@dataclass
class Rule:
    name: str
    required_classes: Set[str]
    forbidden_nearby_classes: Set[str] = field(default_factory=set)
    proximity_px: float = 250.0
    severity: str = "MEDIA"          # BAIXA | MEDIA | ALTA
    min_consecutive_frames: int = 3   # debounce: só dispara após N frames seguidos
    cooldown_seconds: float = 15.0    # tempo mínimo entre alertas repetidos


@dataclass
class SafetyEvent:
    rule_name: str
    severity: str
    message: str
    timestamp: float
    detections_envolvidas: List[Detection]

    def to_dict(self):
        return {
            "rule": self.rule_name,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "detections": [
                {"class": d.class_name, "confidence": d.confidence, "bbox": d.bbox, "track_id": d.track_id}
                for d in self.detections_envolvidas
            ],
        }


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class _RuleState:
    """Máquina de estados simples por regra: novo -> observado -> confirmado -> alerta -> resolvido"""

    def __init__(self):
        self.consecutive_hits = 0
        self.last_alert_time = 0.0
        self.state = "novo"


class SafetyAutomationSystem:
    def __init__(self, log_path: str = "outputs/safety_events.jsonl", webhook_url: Optional[str] = None):
        self.rules: List[Rule] = []
        self._rule_states: Dict[str, _RuleState] = {}
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.webhook_url = webhook_url
        self._notifiers: List[Callable[[SafetyEvent], None]] = [self._notify_console, self._notify_log_file]
        if webhook_url:
            self._notifiers.append(self._notify_webhook)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        self._rule_states[rule.name] = _RuleState()

    # ---------- avaliação das regras (motor de regras) ----------
    def _rule_triggers(self, rule: Rule, detections: List[Detection]) -> Optional[List[Detection]]:
        found_required = [d for d in detections if d.class_name in rule.required_classes]
        if not found_required:
            return None

        if not rule.forbidden_nearby_classes:
            return found_required

        forbidden_present = [d for d in detections if d.class_name in rule.forbidden_nearby_classes]

        # regra do tipo "required SEM forbidden por perto" -> ex.: pessoa sem capacete
        violating = []
        for req in found_required:
            near_ok = any(
                _distance(_bbox_center(req.bbox), _bbox_center(f.bbox)) <= rule.proximity_px
                for f in forbidden_present
            )
            if not near_ok:
                violating.append(req)

        return violating or None

    def process(self, detections: List[Detection]) -> List[SafetyEvent]:
        events = []
        now = time.time()

        for rule in self.rules:
            state = self._rule_states[rule.name]
            triggered_detections = self._rule_triggers(rule, detections)

            if triggered_detections:
                state.consecutive_hits += 1
                state.state = "observado" if state.consecutive_hits < rule.min_consecutive_frames else "confirmado"
            else:
                state.consecutive_hits = 0
                state.state = "resolvido" if state.state in ("confirmado", "alerta") else "novo"
                continue

            if state.consecutive_hits < rule.min_consecutive_frames:
                continue  # ainda não confirmado o suficiente (debounce)

            if now - state.last_alert_time < rule.cooldown_seconds:
                continue  # anti-flood: dentro do cooldown, não repete o alerta

            state.state = "alerta"
            state.last_alert_time = now

            event = SafetyEvent(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Regra '{rule.name}' violada: {[d.class_name for d in triggered_detections]}",
                timestamp=now,
                detections_envolvidas=triggered_detections,
            )
            events.append(event)
            for notifier in self._notifiers:
                notifier(event)

        return events

    # ---------- notificadores plugáveis ----------
    def _notify_console(self, event: SafetyEvent):
        print(f"[{event.severity}] {event.message}")

    def _notify_log_file(self, event: SafetyEvent):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def _notify_webhook(self, event: SafetyEvent):
        if requests is None or not self.webhook_url:
            return
        try:
            requests.post(self.webhook_url, json=event.to_dict(), timeout=3)
        except Exception as exc:  # nunca deve derrubar o pipeline de vídeo por causa de rede
            print(f"[aviso] falha ao enviar webhook: {exc}")


# ---------- exemplo de configuração pronta para o cenário deste desafio ----------
def build_default_safety_system() -> SafetyAutomationSystem:
    system = SafetyAutomationSystem()
    system.add_rule(Rule(
        name="pessoa_sem_capacete",
        required_classes={"person"},
        forbidden_nearby_classes={"capacete"},
        proximity_px=200.0,
        severity="ALTA",
        min_consecutive_frames=5,
        cooldown_seconds=20.0,
    ))
    system.add_rule(Rule(
        name="area_sem_extintor_visivel",
        required_classes={"person"},
        forbidden_nearby_classes={"extintor"},
        proximity_px=400.0,
        severity="MEDIA",
        min_consecutive_frames=10,
        cooldown_seconds=60.0,
    ))
    return system

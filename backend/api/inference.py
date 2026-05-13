"""Lightweight inference module for the FastAPI service.

Decoupled from evaluator.py to avoid matplotlib/sklearn/seaborn imports and
to keep memory and startup footprint small. Loads the model + tokenizer once
and exposes synchronous predict methods over plain Python types.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# backend/ 를 sys.path 에 추가 — uvicorn 을 어디서 실행하든 형제 모듈 import 가능
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import BASE_DIR, resolve_path  # noqa: E402
from model import MultiLabelElectraClassifier, load_tokenizer  # noqa: E402


DEFAULT_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
DEFAULT_MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.5

# 환경변수로 override 가능
MODEL_CHECKPOINT_PATH = os.environ.get(
    "MODEL_CHECKPOINT_PATH",
    str(BASE_DIR / "output" / "koelectra_default_20260513_145614" / "checkpoints" / "best_model.pth"),
)
LABEL_INFO_PATH = os.environ.get(
    "LABEL_INFO_PATH",
    str(BASE_DIR / "korean_unsmile_csv" / "label_info.json"),
)


class InferenceEngine:
    """Singleton-style inference engine — load once, reuse for every request."""

    def __init__(
        self,
        model_checkpoint_path: str = MODEL_CHECKPOINT_PATH,
        label_info_path: str = LABEL_INFO_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.model_checkpoint_path = resolve_path(model_checkpoint_path)
        self.label_info_path = resolve_path(label_info_path)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.label_columns: List[str] = []
        self.label_descriptions: Dict[str, str] = {}
        self.model_name: str = DEFAULT_MODEL_NAME
        self.max_length: int = DEFAULT_MAX_LENGTH
        self.model: Optional[MultiLabelElectraClassifier] = None
        self.tokenizer = None
        self.loaded = False

    # ---------- lifecycle ----------

    def load(self) -> None:
        if self.loaded:
            return

        self._load_label_info()
        self._load_checkpoint_and_model()
        self.tokenizer = load_tokenizer(self.model_name)
        self.loaded = True

    def _load_label_info(self) -> None:
        with open(self.label_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        self.label_columns = info["label_columns"]
        self.label_descriptions = info.get("label_descriptions", {})

    def _load_checkpoint_and_model(self) -> None:
        try:
            ckpt = torch.load(
                self.model_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(self.model_checkpoint_path, map_location=self.device)

        cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
        if cfg:
            self.model_name = cfg.get("model", {}).get("model_name", DEFAULT_MODEL_NAME)
            self.max_length = cfg.get("model", {}).get("max_length", DEFAULT_MAX_LENGTH)

        model = MultiLabelElectraClassifier(
            model_name=self.model_name,
            num_labels=len(self.label_columns),
        )

        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    # ---------- inference ----------

    @torch.inference_mode()
    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Dict:
        if not self.loaded:
            raise RuntimeError("InferenceEngine.load() 가 먼저 호출되어야 합니다.")

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs["logits"]).squeeze(0).cpu().tolist()

        labels = [
            {
                "name": name,
                "probability": float(prob),
                "predicted": bool(prob >= threshold),
            }
            for name, prob in zip(self.label_columns, probs)
        ]
        top = max(labels, key=lambda x: x["probability"])

        return {
            "text": text,
            "threshold": threshold,
            "labels": labels,
            "top": top,
        }


_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.load()
    return _engine

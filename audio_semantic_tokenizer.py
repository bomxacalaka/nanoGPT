from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audio_codec import encode_waveform
from audio_codec import load_audio
from audio_codec import load_encodec_model


@dataclass
class SemanticTokenBatch:
    tokens: torch.Tensor
    sample_rate: int
    semantic_rate_hz: float
    vocab_size: int
    backend_name: str
    aux: dict[str, Any] | None = None
    normalized_wav: torch.Tensor | None = None


class SemanticTokenizerBackend(ABC):
    backend_name: str

    @abstractmethod
    def encode_waveform(
        self,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> SemanticTokenBatch:
        raise NotImplementedError

    @abstractmethod
    def to_metadata(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class EncodecCoarseSemanticConfig:
    model_name: str = "encodec_24khz"
    bandwidth: float = 6.0
    codebook_size: int = 1024
    semantic_codebook_index: int = 0
    frame_stride: int = 3
    device: str = "cpu"


class EncodecCoarseSemanticBackend(SemanticTokenizerBackend):
    backend_name = "encodec_coarse_v1"

    def __init__(self, config: EncodecCoarseSemanticConfig):
        if config.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if config.semantic_codebook_index < 0:
            raise ValueError("semantic_codebook_index must be >= 0")
        self.config = config
        self.codec = load_encodec_model(
            model_name=config.model_name,
            bandwidth=config.bandwidth,
            device=config.device,
        )

    def encode_waveform(
        self,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> SemanticTokenBatch:
        codec_batch = encode_waveform(
            self.codec,
            wav,
            sample_rate,
            device=self.config.device,
            codebook_size=self.config.codebook_size,
        )
        if self.config.semantic_codebook_index >= codec_batch.num_codebooks:
            raise ValueError(
                f"semantic_codebook_index={self.config.semantic_codebook_index} exceeds "
                f"num_codebooks={codec_batch.num_codebooks}"
            )

        semantic_tokens = codec_batch.codes[self.config.semantic_codebook_index].contiguous()
        semantic_tokens = semantic_tokens[:: self.config.frame_stride].clone()

        duration_seconds = max(
            codec_batch.normalized_wav.size(-1) / float(codec_batch.sample_rate),
            1e-8,
        )
        semantic_rate_hz = float(semantic_tokens.numel()) / duration_seconds

        return SemanticTokenBatch(
            tokens=semantic_tokens.to(torch.long),
            sample_rate=codec_batch.sample_rate,
            semantic_rate_hz=semantic_rate_hz,
            vocab_size=self.config.codebook_size,
            backend_name=self.backend_name,
            aux={
                "num_codec_codebooks": codec_batch.num_codebooks,
                "semantic_codebook_index": self.config.semantic_codebook_index,
                "frame_stride": self.config.frame_stride,
                "codec_model_name": self.config.model_name,
                "codec_bandwidth": self.config.bandwidth,
            },
            normalized_wav=codec_batch.normalized_wav,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "config": asdict(self.config),
        }


@dataclass
class HubertKmeansSemanticConfig:
    bundle_name: str = "HUBERT_BASE"
    centroids_path: str = ""
    layer: int = -1
    frame_stride: int = 2
    device: str = "cpu"


class HubertKmeansSemanticBackend(SemanticTokenizerBackend):
    backend_name = "hubert_kmeans_v1"

    def __init__(self, config: HubertKmeansSemanticConfig):
        if not config.centroids_path:
            raise ValueError("hubert_kmeans_v1 requires a non-empty centroids_path")
        if config.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        self.config = config

        import torchaudio

        try:
            bundle = getattr(torchaudio.pipelines, config.bundle_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torchaudio pipeline bundle: {config.bundle_name}") from exc

        centroids = np.load(config.centroids_path)
        if centroids.ndim != 2:
            raise ValueError(
                f"Expected centroids shaped [num_units, feature_dim], got {tuple(centroids.shape)}"
            )

        self.bundle = bundle
        self.sample_rate = int(bundle.sample_rate)
        self.model = bundle.get_model().to(config.device)
        self.model.eval()
        self.centroids = torch.from_numpy(centroids.astype(np.float32, copy=False)).to(config.device)
        self.vocab_size = int(self.centroids.size(0))
        self.feature_dim = int(self.centroids.size(1))

    def _extract_layer_features(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        import torchaudio

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        with torch.no_grad():
            features_out = self.model.extract_features(wav.to(self.config.device))

        if isinstance(features_out, tuple):
            feature_layers = features_out[0]
        else:
            feature_layers = features_out
        if not isinstance(feature_layers, (list, tuple)):
            raise ValueError("Expected HuBERT extract_features() to return a feature-layer list/tuple")

        layer_index = self.config.layer if self.config.layer >= 0 else len(feature_layers) + self.config.layer
        if layer_index < 0 or layer_index >= len(feature_layers):
            raise ValueError(
                f"Requested HuBERT layer {self.config.layer}, but only {len(feature_layers)} feature layers exist"
            )
        features = feature_layers[layer_index]
        if features.dim() != 3:
            raise ValueError(f"Expected HuBERT features shaped [batch, frames, dim], got {tuple(features.shape)}")
        return features

    def _quantize(self, features: torch.Tensor) -> torch.Tensor:
        batch, frames, dim = features.shape
        if dim != self.feature_dim:
            raise ValueError(
                f"Centroid feature dim {self.feature_dim} does not match HuBERT feature dim {dim}"
            )
        flat = features.reshape(batch * frames, dim)
        distances = torch.cdist(flat, self.centroids)
        units = torch.argmin(distances, dim=1)
        return units.view(batch, frames)

    def encode_waveform(
        self,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> SemanticTokenBatch:
        features = self._extract_layer_features(wav, sample_rate)
        if self.config.frame_stride > 1:
            features = features[:, :: self.config.frame_stride, :]
        semantic_tokens = self._quantize(features).squeeze(0).to(torch.long)
        duration_seconds = max(wav.size(-1) / float(sample_rate), 1e-8)
        semantic_rate_hz = float(semantic_tokens.numel()) / duration_seconds
        return SemanticTokenBatch(
            tokens=semantic_tokens,
            sample_rate=sample_rate,
            semantic_rate_hz=semantic_rate_hz,
            vocab_size=self.vocab_size,
            backend_name=self.backend_name,
            aux={
                "bundle_name": self.config.bundle_name,
                "centroids_path": self.config.centroids_path,
                "layer": self.config.layer,
                "frame_stride": self.config.frame_stride,
                "feature_dim": self.feature_dim,
            },
            normalized_wav=None,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "config": asdict(self.config),
        }


def build_backend(metadata: dict[str, Any]) -> SemanticTokenizerBackend:
    backend_name = metadata["backend_name"]
    config = metadata.get("config", {})
    if backend_name == EncodecCoarseSemanticBackend.backend_name:
        return EncodecCoarseSemanticBackend(EncodecCoarseSemanticConfig(**config))
    if backend_name == HubertKmeansSemanticBackend.backend_name:
        return HubertKmeansSemanticBackend(HubertKmeansSemanticConfig(**config))
    raise ValueError(f"Unsupported semantic tokenizer backend: {backend_name}")


class SemanticTokenizer:
    def __init__(self, backend: SemanticTokenizerBackend):
        self.backend = backend

    def encode_waveform(self, wav: torch.Tensor, sample_rate: int) -> SemanticTokenBatch:
        return self.backend.encode_waveform(wav, sample_rate)

    def encode_audio_file(self, path: str | Path) -> SemanticTokenBatch:
        wav, sample_rate = load_audio(str(path))
        return self.encode_waveform(wav, sample_rate)

    def to_metadata(self) -> dict[str, Any]:
        return self.backend.to_metadata()

    def save_metadata(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_metadata(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "SemanticTokenizer":
        return cls(build_backend(metadata))

    @classmethod
    def load_metadata(cls, path: str | Path) -> "SemanticTokenizer":
        with Path(path).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cls.from_metadata(metadata)


def build_default_semantic_tokenizer(
    *,
    semantic_backend: str = EncodecCoarseSemanticBackend.backend_name,
    device: str = "cpu",
    frame_stride: int = 3,
    semantic_codebook_index: int = 0,
    model_name: str = "encodec_24khz",
    bandwidth: float = 6.0,
    codebook_size: int = 1024,
    hubert_bundle_name: str = "HUBERT_BASE",
    hubert_centroids_path: str = "",
    hubert_layer: int = -1,
    hubert_frame_stride: int = 2,
) -> SemanticTokenizer:
    if semantic_backend == EncodecCoarseSemanticBackend.backend_name:
        backend = EncodecCoarseSemanticBackend(
            EncodecCoarseSemanticConfig(
                model_name=model_name,
                bandwidth=bandwidth,
                codebook_size=codebook_size,
                semantic_codebook_index=semantic_codebook_index,
                frame_stride=frame_stride,
                device=device,
            )
        )
    elif semantic_backend == HubertKmeansSemanticBackend.backend_name:
        backend = HubertKmeansSemanticBackend(
            HubertKmeansSemanticConfig(
                bundle_name=hubert_bundle_name,
                centroids_path=hubert_centroids_path,
                layer=hubert_layer,
                frame_stride=hubert_frame_stride,
                device=device,
            )
        )
    else:
        raise ValueError(f"Unsupported semantic_backend={semantic_backend!r}")
    return SemanticTokenizer(backend)

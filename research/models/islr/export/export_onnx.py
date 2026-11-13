"""Export a phrase (CTC) checkpoint to ONNX for the browser (onnxruntime-web).

Writes to --out-dir:
  sign_ctc.onnx     input  "features" (1, T, IN_FEAT) float32, 8 ≤ T ≤ 512
                    output "logits"   (1, T, 251) — 250 signs + blank (last)
  labels.json       index → sign gloss (blank = len(labels))
  preprocess.json   the feature layout the browser must reproduce (landmark
                    order, origin, hold-fill, Δ1, presence flags) + constants

The graph has no mask input: every frame of a browser buffer is valid, so the
all-True mask is built inside. Verified against PyTorch on real LMDB clips
(single and stitched) before writing — the script exits if logits differ.

Usage:
  PYTHONPATH=research/models uv run --with onnx --with onnxruntime --with onnxscript \
    python -m islr.export.export_onnx \
    --checkpoint results/phrases/phrases04/best_phrases.pth \
    --out-dir exports/phrases04
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..core import config as C
from ..core.data._cache_keys import lmdb_key
from ..core.data.preprocessing import hand_presence
from ..core.model import GraphConformer, LandmarkConformer


class _AllValid(nn.Module):
    """Wraps the CTC model with an all-True mask (batch 1 browser buffers)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        return self.model(features, mask)


def load_ctc_model(checkpoint: Path, meta: dict, n_classes: int) -> nn.Module:
    """Phrase checkpoint → model with n_classes signs + blank (last). A checkpoint trained with
    a wider vocabulary (train_phrases --vocab; meta["blank"] > n_classes) is cut to its first
    n_classes rows + blank, i.e. exactly the 250-sign model the browser runs."""
    trained = meta.get("blank", n_classes)
    graph_kw = (dict(graph_hidden=meta["graph_hidden"], graph_layers=meta["graph_layers"])
                if meta["arch"] == "graph" else {})
    model = (GraphConformer if meta["arch"] == "graph" else LandmarkConformer)(
        **graph_kw,
        num_classes=trained,  # classification head, unused in CTC mode but sized like the vocab
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        n_layers=meta["n_layers"],
        n_signers=0,
        ctc_vocab_size=trained,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if trained != n_classes:
        old, keep = model.ctc_head, list(range(n_classes)) + [trained]
        model.ctc_head = nn.Linear(old.in_features, n_classes + 1)
        with torch.no_grad():
            model.ctc_head.weight.copy_(old.weight[keep])
            model.ctc_head.bias.copy_(old.bias[keep])
    return model.eval()


def clip_features(coords: torch.Tensor) -> torch.Tensor:
    """Stored LMDB coords (T, COORD_FEAT) → model input (T, IN_FEAT), as at eval."""
    coords, presence = hand_presence(coords)
    vel = torch.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    return torch.cat([coords, vel, presence], dim=-1)


def verification_inputs(lmdb_path: str, csv_path: str, n: int, seed: int, max_frames: int) -> list:
    """n single clips + n stitched 2–5-clip inputs from the IS LMDB."""
    import lmdb
    import pandas as pd

    paths = pd.read_csv(csv_path)["path"].sample(n * 5, random_state=seed).tolist()
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                    map_size=os.path.getsize(lmdb_path) + (256 << 20))
    clips = []
    with env.begin() as txn:
        for p in paths:
            v = txn.get(lmdb_key(p))
            if v is not None:
                clips.append(torch.load(io.BytesIO(v), weights_only=True))
    rng = np.random.default_rng(seed)
    inputs = [clip_features(c) for c in clips[:n]]
    for _ in range(n):
        k = int(rng.integers(2, 6))
        picks = [clips[int(i)] for i in rng.choice(len(clips), k, replace=False)]
        inputs.append(clip_features(torch.cat(picks))[:max_frames])
    return inputs


def preprocess_spec(max_frames: int) -> dict:
    landmarks = (
        [["left_hand", i] for i in range(C.N_LH)]
        + [["pose", i] for i in range(C.N_POSE)]
        + [["right_hand", i] for i in range(C.N_RH)]
        + [["face", i] for i in C.SELECTED_FACE_INDICES]
    )
    return {
        "in_feat": C.IN_FEAT,
        "coords_per_landmark": C.COORDS_PER_LM,
        "landmarks": landmarks,  # order of the position block; x, y, z each
        "layout": "[positions (n_landmarks*3) | delta1 (n_landmarks*3) | presence (left_hand, right_hand)]",
        "origin": "per frame: pose 0 (nose) → mean(pose 11, 12) → mean(pose 23, 24) → 0; "
                  "subtracted from x, y and z of every landmark",
        "image": "unmirrored camera frames; x, y normalised to [0, 1] (MediaPipe)",
        "hand_slots": "left_hand = the signer's own left hand (nearest pose 15), "
                      "right_hand = nearest pose 16 — not a handedness label",
        "missing_hand": "presence 0; coordinates hold the nearest detection "
                        "(previous one, or the next for leading frames); 0 if never detected",
        "missing_pose_or_face": "hold the previous frame (next for leading frames); 0 if never seen",
        "delta1": "frame t minus frame t-1 of the position block (after origin and hold-fill); 0 at t = 0",
        "max_frames": max_frames,
        "max_positional_frames": 512,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--meta", type=Path, help="default: phrases_meta.json next to the checkpoint")
    p.add_argument("--vocab", default="data/asl-is-lmdb/sign_to_prediction_index_map.json")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--lmdb-path", default="data/asl-is-lmdb/is.lmdb.mdb")
    p.add_argument("--csv", default="data/asl-is-lmdb/train.csv")
    p.add_argument("--n-verify", type=int, default=16)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import onnxruntime as ort

    meta = json.loads((args.meta or args.checkpoint.with_name("phrases_meta.json")).read_text())
    if meta.get("research_only"):
        raise SystemExit("this checkpoint was trained on research-only data (ASL Citizen) and must "
                         "not be exported for the app")
    sign2idx = json.loads(Path(args.vocab).read_text())
    labels = [s for s, _ in sorted(sign2idx.items(), key=lambda kv: kv[1])]
    model = load_ctc_model(args.checkpoint, meta, len(labels))
    wrapped = _AllValid(model).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "sign_ctc.onnx"
    example = torch.zeros(1, 64, C.IN_FEAT)
    torch.onnx.export(
        wrapped, (example,), str(onnx_path),
        input_names=["features"], output_names=["logits"],
        dynamic_shapes={"features": {1: torch.export.Dim("T", min=8, max=512)}},
        opset_version=args.opset, dynamo=True, external_data=False,
    )
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MiB)")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst, agree, n = 0.0, 0, 0
    for feats in verification_inputs(args.lmdb_path, args.csv, args.n_verify, args.seed, args.max_frames):
        x = feats[None].float()
        with torch.no_grad():
            ref = wrapped(x).numpy()
        out = sess.run(None, {"features": x.numpy()})[0]
        worst = max(worst, float(np.abs(out - ref).max()))
        agree += int((out.argmax(-1) == ref.argmax(-1)).all())
        n += 1
    print(f"Verified {n} inputs (T up to {args.max_frames}): max |Δlogit| {worst:.2e}, "
          f"identical frame argmax {agree}/{n}")
    if worst > args.atol:
        raise SystemExit(f"ONNX logits differ from PyTorch by {worst:.2e} > {args.atol}")

    (args.out_dir / "labels.json").write_text(json.dumps(labels))
    (args.out_dir / "preprocess.json").write_text(
        json.dumps({**preprocess_spec(args.max_frames), "blank": len(labels),
                    "checkpoint_meta": meta}, indent=2))
    print(f"Wrote labels.json, preprocess.json → {args.out_dir}")


if __name__ == "__main__":
    main()

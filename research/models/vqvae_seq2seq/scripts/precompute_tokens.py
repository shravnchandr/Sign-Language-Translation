"""
Pre-tokenize the dataset once using a trained VQ-VAE.

Run this after Phase 1 completes. Saves per-sample token indices as .pt files.
Phase 2 training then loads these directly — no VQ-VAE in memory, no re-encoding.

Usage:
    uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
        --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
        --data-dir data/Isolated_ASL_Recognition \
        --token-dir data/tokens

Output structure mirrors the data directory:
    data/tokens/train_landmark_files/{signer_id}/{sequence_id}.pt
Each file contains:
    {"pose": LongTensor(n_chunks,), "motion": ..., "dynamics": ..., "face": ...,
     "n_chunks": int}
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import VQVAEDataset, collate_vqvae
from ..data.preprocessing import LandmarkConfig, FACE_LANDMARK_SUBSETS
from ..data.signer_split import create_signer_splits
from ..vqvae.config import ImprovedVQVAEConfig
from ..vqvae.vqvae_model import ImprovedVQVAE


def _load_vqvae(checkpoint_path: str, device: torch.device) -> ImprovedVQVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ImprovedVQVAEConfig(**checkpoint["config"])
    model = ImprovedVQVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.train(False)  # inference mode
    return model


@torch.no_grad()
def precompute(
    vqvae: ImprovedVQVAE,
    dataset: VQVAEDataset,
    df,
    token_dir: Path,
    batch_size: int,
    device: torch.device,
    split_name: str,
    num_workers: int = 4,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_vqvae,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )

    sample_idx = 0
    skipped = 0

    for batch in tqdm(loader, desc=f"Tokenizing {split_name}"):
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        indices = vqvae.tokenize(landmarks, mask)
        B = landmarks.shape[0]

        for i in range(B):
            row = df.iloc[sample_idx]
            rel_path = row["path"] if "path" in df.columns else str(sample_idx)
            out_path = token_dir / Path(rel_path).with_suffix(".pt")

            if out_path.exists():
                sample_idx += 1
                skipped += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            sample_tokens = {name: idx[i].cpu() for name, idx in indices.items()}
            first_factor = next(iter(sample_tokens.values()))
            sample_tokens["n_chunks"] = torch.tensor(
                first_factor.shape[0], dtype=torch.long
            )

            tmp = out_path.with_suffix(".tmp")
            torch.save(sample_tokens, tmp)
            tmp.rename(out_path)

            sample_idx += 1

    total = len(dataset)
    new_files = total - skipped
    print(f"  {split_name}: {new_files} new, {skipped} cached, {total} total")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset with trained VQ-VAE"
    )
    parser.add_argument("--vqvae-checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/Isolated_ASL_Recognition")
    parser.add_argument("--token-dir", type=str, default="data/tokens")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Landmark cache dir (speeds up parquet loading)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for CPU preprocessing (try 8-12 on fast SSDs)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    token_dir = Path(args.token_dir)
    token_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VQ-VAE from {args.vqvae_checkpoint}...")
    vqvae = _load_vqvae(args.vqvae_checkpoint, device)
    print(f"VQ-VAE loaded ({sum(p.numel() for p in vqvae.parameters()):,} params)")

    csv_path = f"{args.data_dir}/train.csv"
    splits, _ = create_signer_splits(csv_path, args.data_dir)

    landmark_config = LandmarkConfig(
        include_z=True,
        face_subset=FACE_LANDMARK_SUBSETS["compact"],
    )

    for split_name in args.splits:
        df = splits[split_name]
        dataset = VQVAEDataset(
            df=df,
            base_path=args.data_dir,
            config=landmark_config,
            augment=False,
            cache_dir=args.cache_dir,
        )
        print(f"\n{split_name}: {len(dataset)} samples")
        precompute(
            vqvae,
            dataset,
            df,
            token_dir,
            args.batch_size,
            device,
            split_name,
            args.num_workers,
        )

    # Save VQ-VAE config alongside tokens for reference
    vqvae_ckpt = torch.load(
        args.vqvae_checkpoint, map_location="cpu", weights_only=True
    )
    meta = {
        "vqvae_checkpoint": args.vqvae_checkpoint,
        "vqvae_config": vqvae_ckpt["config"],
        "codebook_names": list(vqvae.quantizers.quantizers.keys()),
    }
    with open(token_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta written to {token_dir}/meta.json")
    print("Pre-tokenization complete.")


if __name__ == "__main__":
    main()

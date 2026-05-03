"""
Merge multiple ASL landmark datasets into a single train.csv that the
AnatomicalConformer training pipeline can consume without code changes.

Supported datasets (all optional except Google ASL):
  - Google ASL Signs   (primary, defines the 250-class label space)
  - WLASL              (after landmark extraction via preprocess_landmarks.py)
  - MS-ASL             (after landmark extraction via preprocess_landmarks.py)

Label strategy
--------------
Google ASL's sign_to_prediction_index_map.json is the canonical label space.
Secondary dataset glosses are mapped to canonical Google ASL glosses via the
pre-built cross-dataset maps.  Samples whose gloss has no mapping are dropped.

Participant ID strategy
-----------------------
Google ASL IDs are large integers (~16000–65000).
WLASL signer IDs are small integers (0–119)  → no offset needed.
MS-ASL signer IDs are small integers (0–221) → offset by MSASL_SIGNER_OFFSET
to avoid collisions with WLASL.

Output
------
  <out-dir>/train.csv                      — merged, absolute paths
  <out-dir>/sign_to_prediction_index_map.json — copied from Google ASL dir

Usage
-----
  python research/tools/merge_datasets.py \\
      --google-asl-dir data/Isolated_ASL_Recognition \\
      --wlasl-dir      data/WLASL_Landmarks \\
      --msasl-dir      data/MSASL_Landmarks \\
      --out-dir        data/merged_asl

  # Dry-run: print overlap stats without writing
  python research/tools/merge_datasets.py \\
      --google-asl-dir data/Isolated_ASL_Recognition \\
      --wlasl-dir      data/WLASL_Landmarks \\
      --dry-run
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

# MS-ASL signer IDs start from 0 — offset to avoid collision with WLASL (0–119).
MSASL_SIGNER_OFFSET = 10_000


def _load_google_asl(data_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (df, sign_map, idx_to_canonical_gloss)."""
    with open(data_dir / "sign_to_prediction_index_map.json") as f:
        sign_map = json.load(f)                        # gloss → int index
    idx_to_gloss = {v: k for k, v in sign_map.items()}  # int index → canonical gloss

    df = pd.read_csv(data_dir / "train.csv")
    df["path"] = df["path"].apply(lambda p: str((data_dir / p).resolve()))
    df = df[["path", "sign", "participant_id"]].copy()
    return df, sign_map, idx_to_gloss


def _build_wlasl_signer_map(wlasl_json: Path) -> dict[str, int]:
    """Build video_id → signer_id lookup from WLASL_v0.3.json."""
    if not wlasl_json.exists():
        return {}
    with open(wlasl_json) as f:
        data = json.load(f)
    mapping = {}
    for entry in data:
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id", ""))
            sid = int(inst.get("signer_id", -1))
            if vid:
                mapping[vid] = sid
    return mapping


def _remap_secondary(
    data_dir: Path,
    cross_map_path: Path,
    idx_to_gloss: dict,
    signer_offset: int = 0,
    dataset_name: str = "dataset",
    wlasl_json: Path | None = None,
) -> pd.DataFrame:
    """
    Load a secondary dataset's train.csv, remap signs to canonical Google ASL
    glosses, and make paths absolute.

    cross_map_path: JSON mapping secondary_gloss → Google ASL int index.
    wlasl_json: optional path to WLASL_v0.3.json for video_id → signer_id lookup
                when participant_id column is absent (pre-tool CSV format).
    """
    if not data_dir.exists():
        print(f"  {dataset_name}: directory not found, skipping")
        return pd.DataFrame(columns=["path", "sign", "participant_id"])

    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        print(f"  {dataset_name}: train.csv not found in {data_dir}, skipping")
        return pd.DataFrame(columns=["path", "sign", "participant_id"])

    if not cross_map_path.exists():
        print(f"  {dataset_name}: cross-map {cross_map_path} not found, skipping")
        return pd.DataFrame(columns=["path", "sign", "participant_id"])

    with open(cross_map_path) as f:
        cross_map = json.load(f)   # secondary_gloss → Google ASL index
    # Build: secondary_gloss → canonical Google ASL gloss
    gloss_remap = {g: idx_to_gloss[idx] for g, idx in cross_map.items() if idx in idx_to_gloss}

    df = pd.read_csv(csv_path)
    before = len(df)
    df["sign"] = df["sign"].str.lower().str.strip().map(gloss_remap)
    df = df.dropna(subset=["sign"])
    after = len(df)

    df["path"] = df["path"].apply(lambda p: str((data_dir / p).resolve()))

    # Resolve participant_id: prefer the column if present; otherwise fall back
    # to video_id → signer_id from the WLASL JSON (pre-tool CSV format).
    if "participant_id" in df.columns:
        df["participant_id"] = df["participant_id"].astype(int) + signer_offset
    elif "video_id" in df.columns and wlasl_json is not None:
        signer_map = _build_wlasl_signer_map(wlasl_json)
        df["participant_id"] = (
            df["video_id"].astype(str).map(signer_map).fillna(-1).astype(int) + signer_offset
        )
        n_unknown = (df["participant_id"] == signer_offset - 1).sum()
        if n_unknown:
            print(f"  {dataset_name}: {n_unknown} samples have unknown signer_id (will get -1)")
    else:
        df["participant_id"] = -1
        print(f"  {dataset_name}: no participant_id column — signer tracking disabled for this dataset")

    print(f"  {dataset_name}: {after}/{before} samples kept "
          f"({before - after} dropped — no label mapping), "
          f"{df['sign'].nunique()} classes, "
          f"{df['participant_id'].nunique()} signers")
    return df[["path", "sign", "participant_id"]]


def merge(
    google_asl_dir: Path,
    wlasl_dir: Path | None,
    msasl_dir: Path | None,
    out_dir: Path,
    dry_run: bool,
) -> None:
    print("Loading Google ASL Signs...")
    google_df, sign_map, idx_to_gloss = _load_google_asl(google_asl_dir)
    print(f"  Google ASL: {len(google_df)} samples, "
          f"{google_df['sign'].nunique()} classes, "
          f"{google_df['participant_id'].nunique()} signers")

    parts = [google_df]

    if wlasl_dir is not None:
        print("Loading WLASL...")
        wlasl_cross = wlasl_dir / "wlasl_to_google_isolated_asl_index_map.json"
        wlasl_json  = wlasl_dir / "WLASL_v0.3.json"
        wlasl_df = _remap_secondary(
            wlasl_dir,
            cross_map_path=wlasl_cross,
            idx_to_gloss=idx_to_gloss,
            signer_offset=0,
            dataset_name="WLASL",
            wlasl_json=wlasl_json if wlasl_json.exists() else None,
        )
        parts.append(wlasl_df)

    if msasl_dir is not None:
        print("Loading MS-ASL...")
        msasl_df = _remap_secondary(
            msasl_dir,
            cross_map_path=msasl_dir / "msasl_to_google_index.json",
            idx_to_gloss=idx_to_gloss,
            signer_offset=MSASL_SIGNER_OFFSET,
            dataset_name="MS-ASL",
        )
        parts.append(msasl_df)

    merged = pd.concat(parts, ignore_index=True)

    # Verify no NaN labels snuck through
    assert merged["sign"].notna().all(), "BUG: NaN signs in merged DataFrame"

    # Summary
    print(f"\nMerged totals:")
    print(f"  Samples    : {len(merged):,}")
    print(f"  Classes    : {merged['sign'].nunique()} / {len(sign_map)}")
    print(f"  Signers    : {merged['participant_id'].nunique()}")

    # Class coverage
    uncovered = set(sign_map.keys()) - set(merged["sign"].unique())
    if uncovered:
        print(f"  Classes with zero samples: {len(uncovered)} — {sorted(uncovered)}")

    # Signer breakdown per dataset
    print(f"\n  Signer breakdown:")
    print(f"    Google ASL : {google_df['participant_id'].nunique()}")
    for name, df_part in zip(
        ["WLASL", "MS-ASL"],
        [p for p in parts[1:] if len(p) > 0],
    ):
        print(f"    {name:<10} : {df_part['participant_id'].nunique()}")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "train.csv", index=False)
    shutil.copy(
        google_asl_dir / "sign_to_prediction_index_map.json",
        out_dir / "sign_to_prediction_index_map.json",
    )
    print(f"\nWritten to {out_dir}/")
    print(f"  train.csv ({len(merged):,} rows)")
    print(f"  sign_to_prediction_index_map.json (copied from Google ASL)")
    print(f"\nNext steps:")
    print(f"  1. Rebuild LMDB:  python -m cnn_transformer.data.build_lmdb \\")
    print(f"                        --data-dir {out_dir} \\")
    print(f"                        --lmdb-path data/cache/merged_asl/asl.lmdb")
    print(f"  2. Train:         python -m cnn_transformer.train \\")
    print(f"                        --data-dir {out_dir} \\")
    print(f"                        --lmdb-path data/cache/merged_asl/asl.lmdb")


def main():
    p = argparse.ArgumentParser(
        description="Merge ASL landmark datasets into a single training CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--google-asl-dir", required=True,
                   help="data/Isolated_ASL_Recognition")
    p.add_argument("--wlasl-dir", default=None,
                   help="data/WLASL_Landmarks (after landmark extraction)")
    p.add_argument("--msasl-dir", default=None,
                   help="data/MSASL_Landmarks (after landmark extraction)")
    p.add_argument("--out-dir", default="data/merged_asl",
                   help="Output directory for merged train.csv and sign map")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats without writing any files")
    args = p.parse_args()

    merge(
        google_asl_dir=Path(args.google_asl_dir),
        wlasl_dir=Path(args.wlasl_dir) if args.wlasl_dir else None,
        msasl_dir=Path(args.msasl_dir) if args.msasl_dir else None,
        out_dir=Path(args.out_dir),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

# Sign Language Recognition Training Pipeline

## Overview

This pipeline creates a universal sign language tokenizer using VQ-VAE, then trains a classifier on the tokenized sequences.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Google ASL Signs    │  ASL Fingerspelling  │  WLASL (processed)   │
│  (94k videos)        │  (large dataset)     │  (21k videos)        │
│  [Kaggle]            │  [Kaggle]            │  [needs MediaPipe]   │
└──────────┬───────────┴──────────┬───────────┴──────────┬────────────┘
           │                      │                      │
           └──────────────────────┼──────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: PRE-TRAINING                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            Factorized Temporal VQ-VAE                        │   │
│  │                                                               │   │
│  │   Video Chunks ──► [Pose Encoder] ──► Pose Codebook (512)    │   │
│  │                ──► [Motion Encoder] ──► Motion Codebook (256)│   │
│  │                ──► [Dynamics Encoder] ──► Dynamics CB (128)  │   │
│  │                                                               │   │
│  │   Output: 3 tokens per chunk (pose_id, motion_id, dyn_id)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Unsupervised - uses ALL data without labels                        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: CLASSIFICATION                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │            Token Sequence Classifier                         │   │
│  │                                                               │   │
│  │   Video ──► Tokenizer ──► [(p1,m1,d1), (p2,m2,d2), ...]     │   │
│  │                                  │                           │   │
│  │                                  ▼                           │   │
│  │                        Transformer Encoder                   │   │
│  │                                  │                           │   │
│  │                                  ▼                           │   │
│  │                          Classification                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Supervised - uses Google ASL Signs labels (250 classes)            │
└──────────────────────────────────────────────────────────────────────┘
```

## Datasets

### 1. Google ASL Signs (Primary - Competition Data)
- **Kaggle**: `kaggle competitions download -c asl-signs`
- **Format**: Parquet with MediaPipe landmarks
- **Size**: ~94,000 videos, 250 signs
- **Status**: Ready to use

### 2. ASL Fingerspelling (Additional Pre-training)
- **Kaggle**: `kaggle competitions download -c asl-fingerspelling`
- **Format**: Parquet with MediaPipe landmarks (same format!)
- **Size**: Very large, character-level
- **Status**: Ready to use

### 3. WLASL (Additional Pre-training)
- **Kaggle**: Search for "WLASL" or download videos separately
- **Format**: Raw videos (need MediaPipe processing)
- **Size**: ~21,000 videos, 2000 signs
- **Status**: Needs preprocessing with `preprocess-wlasl-mediapipe.ipynb`

## Landmark Consistency

**CRITICAL**: All datasets must use the same MediaPipe landmark format:

| Type | Landmarks | Indices |
|------|-----------|---------|
| pose | 33 | 0-32 |
| left_hand | 21 | 0-20 |
| right_hand | 21 | 0-20 |
| face | 468 | 0-467 |

**Total**: 543 landmarks per frame

## Notebooks

### Pre-training Phase
1. `preprocess-wlasl-mediapipe.ipynb` - Extract landmarks from WLASL videos
2. `sign-vqvae-temporal.ipynb` - Train factorized VQ-VAE with temporal dynamics

### Classification Phase
3. `sign-token-classifier.ipynb` - Train transformer on tokenized sequences

### Baseline (for comparison)
4. `asl-improved-transformer-v2.ipynb` - Direct transformer without tokenization

## What the Factorized Codebooks Capture

| Codebook | Size | Input | What it learns |
|----------|------|-------|----------------|
| **Pose** | 512 | Positions | Handshapes, body positions, locations |
| **Motion** | 256 | Velocities | Speed, direction, movement type |
| **Dynamics** | 128 | Accelerations | Sharp stops, smooth transitions, rhythm |

### Example Token Sequences

**Sign: "STOP"** (sharp hand forward then stop)
```
Chunk 1: (pose=42, motion=15, dynamics=3)   # Hand starting position, moving forward, accelerating
Chunk 2: (pose=42, motion=89, dynamics=112) # Same shape, slowing down, sharp deceleration
Chunk 3: (pose=42, motion=0, dynamics=0)    # Same shape, stopped, no motion
```

**Sign: "HELLO"** (waving)
```
Chunk 1: (pose=100, motion=45, dynamics=20)  # Open hand, moving right, smooth
Chunk 2: (pose=100, motion=67, dynamics=20)  # Same shape, moving left, smooth
Chunk 3: (pose=100, motion=45, dynamics=20)  # Same shape, moving right, smooth
...repeating pattern...
```

## Training Steps

### Step 1: Prepare WLASL Data (Optional but Recommended)
```bash
# On Kaggle, run preprocess-wlasl-mediapipe.ipynb
# This creates landmarks in the same format as competition
```

### Step 2: Pre-train VQ-VAE
```bash
# Run sign-vqvae-temporal.ipynb
# Uses ALL available landmark data (no labels needed)
# Outputs: temporal_sign_tokenizer.pth
```

### Step 3: Train Classifier
```bash
# Run sign-token-classifier.ipynb
# Uses Google ASL Signs with labels
# Outputs: token_classifier_best.pth
```

## Expected Improvements

| Approach | Expected Accuracy | Notes |
|----------|-------------------|-------|
| Baseline Transformer | ~78% | Direct landmark input |
| Single VQ-VAE + Transformer | ~82% | Basic tokenization |
| Factorized VQ-VAE + Transformer | ~85%+ | Temporal dynamics |
| + More pre-training data | ~87%+ | WLASL + Fingerspelling |
| + Ensemble | ~89%+ | Multiple models |

## Key Insights

1. **Temporal dynamics matter**: Sign language encodes meaning in speed/acceleration, not just positions
2. **Factorization helps**: Separating pose from motion allows better generalization
3. **More data helps**: Unsupervised pre-training on diverse data improves representations
4. **Consistency is critical**: All data must use identical MediaPipe processing

# TRD-BOT: Unsupervised Robotic Trajectory Segmentation via Temporal Relational Distillation and Boundary-Guided Optimal Transport

## Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
  - [TRD Training](#trd-training)
  - [Feature Extraction](#feature-extraction)
  - [Clustering](#clustering)
  - [BOT Refinement](#bot-refinement)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)

## Installation

Clone this repository and install the required dependencies.

```bash
git clone https://github.com/yueli0827/TRD-BOT.git
cd TRD-BOT

conda create -n TRD_BOT python=3.9 -y
conda activate TRD_BOT

pip install -r requirements.txt
```

## Datasets

TRD-BOT is evaluated on six robotic manipulation tasks from three datasets:

- [REASSEMBLE](https://tuwien-asl.github.io/REASSEMBLE_page/)
  - Grasping
  - Insertion
- [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
  - Suturing
  - Passing
- NursBot
  - Transfer
  - Pouring

The NursBot dataset used in this work is included in this repository.

## Getting Started

The TRD-BOT pipeline consists of four main stages:

1. TRD student training
2. Frame-level feature extraction
3. Initial clustering
4. Boundary-guided optimal transport refinement

### TRD Training

`train_trd.py` trains one TRD or Basic student on the unlabeled videos of a dataset.

Example:

```bash
cd feature_extraction/

mkdir -p checkpoints

python train_trd.py \
    --dataset_dir PATH_TO_YOUR_DATASET \
    --output checkpoints/trd_student.pt \
    --dataset_name YOUR_DATASET_NAME \
    --beta 0.10 \
    --learning_rate 2e-6 \
    --training_steps 1000 \
    --gradient_accumulation_steps 4 \
    --warmup_steps 50 \
    --teacher_timesteps 5 \
    --seed 0
```

For the Basic configuration, set:

```bash
--beta 0
```

### Feature Extraction

After training, use the saved student checkpoint to extract frame-level features.

Example:

```bash
cd feature_extraction/

python feature_extraction.py \
    --video_path PATH_TO_YOUR_VIDEO \
    --ckpt_path checkpoints/trd_student.pt \
    --batch_size 4 \
    --output_base_path PATH_TO_FEATURE_OUTPUT
```

The output includes:

```text
video_features.npy
frame_indices.npy
```

### Clustering

The extracted frame features can be segmented using different clustering backends.

The paper evaluates:

- K-means
- Gaussian Mixture Model (GMM)
- Spectral Clustering
- Graph Clustering
- Hierarchical Clustering

DeepDPM is used to estimate the number of clusters for each video.

Example using spectral clustering:

```bash
cd clustering/

python spectral_clustering.py \
    --input_path YOUR_FEATURE_DIR \
    --out_dir YOUR_OUTPUT_DIR \
    --k_mode deepdpm \
    --k_min 2 \
    --k_max 20
```

The same estimated cluster count is used when comparing the Basic and TRD-BOT configurations for a given video.

### BOT Refinement

BOT refines the initial clustering result using boundary-aware temporal consistency in a fused Gromov-Wasserstein optimal transport framework.

Example:

```bash
cd postpromoting/

python postpromoting.py \
    --features PATH_TO/video_features.npy \
    --labels PATH_TO/initial_labels.npy \
    --output PATH_TO/bot_result.npz \
    --kappa 0.15 \
    --structure-weight 0.65 \
    --entropy-weight 0.03 \
    --windows 5 10 20
```

The main hyperparameters used in the paper are:

```text
beta   = 0.10
kappa  = 0.15
lambda = 0.65
eta    = 0.03
W      = {5, 10, 20}
```

## Evaluation

The paper reports the following metrics:

- Silhouette Coefficient (SC)
- Adjusted Rand Index (ARI)
- Segmentation Accuracy (SA)
- Segmental F1 at IoU threshold 0.50 (F1@50)
- Boundary F1


## Acknowledgement

The clustering and cluster-number estimation components are based on or related to:

- Spectral Clustering
- Gaussian Mixture Model
- K-means
- Graph Clustering
- Hierarchical Clustering
- [DeepDPM](https://github.com/BGU-CS-VIL/DeepDPM)

The experiments use the REASSEMBLE, JIGSAWS, and NursBot datasets. We sincerely thank the authors and maintainers of these datasets and open-source projects for making their work publicly available.

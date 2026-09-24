# TRD-BOT: Unsupervised Robotic Trajectory Segmentation via Temporal Relational Distillation and Boundary-Guided Optimal Transport

## Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
  - [Temporal Relational Distillation and Feature Extraction](#temporal-relational-distillation-and-feature-extraction)
  - [Clustering](#clustering)
  - [Boundary-Guided Optimal Transport Refinement](#boundary-guided-optimal-transport-refinement)
- [Default Settings](#default-settings)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Overview

TRD-BOT is an unsupervised robotic trajectory segmentation framework that combines temporal relational distillation (TRD) with boundary-guided optimal transport (BOT).

TRD enhances multi-timestep diffusion feature alignment by transferring local inter-frame similarity relationships from a frozen diffusion teacher to a student feature extractor. This encourages the learned frame representations to preserve local temporal structure while remaining suitable for downstream clustering.

BOT refines the initial clustering result through a fused Gromov-Wasserstein optimal transport formulation. Boundary reliability estimated from local feature changes is used to adaptively modulate temporal consistency, reducing within-action fragmentation while preserving meaningful action transitions.

The framework is evaluated on six robotic manipulation tasks from three datasets and with five clustering backends: K-means, Gaussian mixture model (GMM), spectral clustering, graph clustering, and hierarchical clustering.

## Installation

Clone this repository and install the required dependencies.

```bash
git clone https://github.com/yueli0827/TRE-BOT.git
cd TRE-BOT

conda create -n TRD_BOT python=3.9
conda activate TRD_BOT

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Datasets

TRD-BOT is evaluated on three datasets.

### REASSEMBLE

REASSEMBLE is a multimodal dataset for contact-rich robotic assembly and disassembly.

Dataset page:
https://tuwien-asl.github.io/REASSEMBLE_page/

The experiments in the paper use:
- Grasping
- Insertion

### JIGSAWS

JIGSAWS is a benchmark dataset for robotic surgical activity analysis.

Dataset page:
https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/

The experiments in the paper use:
- Suturing
- Passing

### NursBot

The NursBot dataset used in the paper is included in this repository.

The experiments use:
- Transfer
- Pouring

Please organize the dataset paths according to the directory structure expected by the corresponding scripts before running feature extraction or evaluation.

## Getting Started

The TRD-BOT pipeline consists of three main stages:

1. Temporal relational distillation and frame-level feature extraction
2. Initial clustering
3. Boundary-guided optimal transport refinement

### Temporal Relational Distillation and Feature Extraction

TRD is built on multi-timestep diffusion feature alignment. A frozen Stable Diffusion 2.1 teacher provides intermediate diffusion features, while a trainable student receives clean latent inputs.

For each adjacent frame pair, TRD additionally matches the local inter-frame similarity relationship between teacher and student features. Teacher relations are averaged across multiple diffusion timesteps, while the relational loss is applied directly to the unprojected student features.

After training, only the student network is retained for feature extraction.

An example feature-extraction command is:

```bash
cd feature_extraction/

python feature_extraction.py \
    --video_path PATH_TO_YOUR_VIDEO \
    --num_frames 1000 \
    --batch_size 8 \
    --feat_key "mid"
```

For reproducing the paper settings, use the TRD weight and training configuration listed in the Default Settings section.

### Clustering

The extracted frame features can be segmented with different clustering backends.

The paper evaluates:
- K-means
- Gaussian Mixture Model (GMM)
- Spectral Clustering
- Graph Clustering
- Hierarchical Clustering

DeepDPM is used to estimate the number of clusters for each video.

An example using spectral clustering is:

```bash
cd clustering/

python spectral_clustering.py \
    --input_path YOUR_FEATURE_DIR \
    --out_dir YOUR_OUTPUT_DIR \
    --k_mode deepdpm \
    --k_min 2 \
    --k_max 20
```

The same estimated cluster count should be used when comparing the Basic and TRD-BOT configurations for a given video.

### Boundary-Guided Optimal Transport Refinement

BOT refines the initial frame-wise clustering result.

The refinement process includes:
1. Constructing initial temporal segments from consecutive frames with the same cluster label
2. Computing frame-to-segment visual matching costs
3. Estimating multi-scale boundary reliability from local feature changes
4. Modulating temporal consistency according to boundary confidence
5. Optimizing a fused Gromov-Wasserstein objective
6. Decoding the optimized soft assignment into the final segmentation

The current repository retains the post-processing entry point under the `postpromoting/` directory.

Example:

```bash
cd postpromoting/

python postpromoting.py \
    -d YOUR_DATA_SET_NAME \
    -clustering-model YOUR_CLUSTERING_MODEL_NAME
```

Please set the BOT hyperparameters according to the values used in the paper or your own experimental configuration.

## Default Settings

The core settings used in the paper are:

| Parameter | Value |
|---|---|
| Selected feature maps K | 13 |
| Teacher timestep samples M | 5 |
| Optimizer | Adam |
| Learning rate | 2e-6 |
| Effective batch size | 4 frame pairs |
| Training updates | 1000 |
| TRD weight beta | 0.10 |
| BOT boundary scale kappa | 0.15 |
| Structure weight lambda | 0.65 |
| Entropy weight eta | 0.03 |
| Boundary window half-widths | {5, 10, 20} frames |
| Temporal neighborhood | w = max(1, ceil(0.02 * J)) |

The sensitivity experiments in the paper use:

TRD weight beta:
```text
{0.025, 0.05, 0.10, 0.20, 0.30, 0.40}
```

BOT boundary scale kappa:
```text
{0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
```

Selected values:
```text
beta = 0.10
kappa = 0.15
```

## Evaluation

The paper reports:
- Silhouette Coefficient (SC)
- Adjusted Rand Index (ARI)
- Segmentation Accuracy (SA)
- Segmental F1 at IoU threshold 0.50 (F1@50)
- Boundary F1

For Boundary F1, a predicted boundary is matched to a ground-truth boundary when their temporal distance is within 5 frames, using one-to-one matching.

The experiments are conducted on six tasks:

| Dataset | Task |
|---|---|
| REASSEMBLE | Grasping |
| REASSEMBLE | Insertion |
| JIGSAWS | Suturing |
| JIGSAWS | Passing |
| NursBot | Transfer |
| NursBot | Pouring |

## Citation

If you find this project useful, please consider citing the corresponding paper after publication.


## Acknowledgement

This project benefits from prior work on diffusion representations, optimal transport, robotic trajectory segmentation, and unsupervised clustering.

The clustering and cluster-number estimation components are related to or implemented with reference to:
- Spectral Clustering
- Gaussian Mixture Model (GMM)
- K-means
- Graph Clustering
- Hierarchical Clustering
- DeepDPM: https://github.com/BGU-CS-VIL/DeepDPM

The experiments use the REASSEMBLE, JIGSAWS, and NursBot datasets. We sincerely thank the authors and maintainers of these datasets and open-source projects for making their work publicly available.

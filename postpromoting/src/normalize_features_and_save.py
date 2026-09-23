import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from triton.language import dtype
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from video_dataset import VideoDataset
import tvc
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics


layer_sizes = [46080]
output_npy_path = 'video_features_normalize.npy'

def fit_clusters(dataloader):
    with torch.no_grad():
        features_full = []
        for features_raw, _, _, _, _ in dataloader:
            B, T, _ = features_raw.shape
            print('features_raw.shape:', features_raw.shape)
            D = layer_sizes[-1]
            features = F.normalize(features_raw.reshape(-1, features_raw.shape[-1]).reshape(B, T, D), dim=-1)
            features_full.append(features)
        features_full = torch.cat(features_full, dim=0).reshape(-1, features.shape[2]).cpu().numpy()
        print('features_full.shape:', features_full.shape)
        # 转换并输出结果
        np.save(output_npy_path, features_full)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")
    parser.add_argument('--n-frames-test', '-ft', type=int, default=512,
                        help='number of frames sampled per video for test')
    args = parser.parse_args()

    std_feats = True
    dataset = 'ROBOT'
    activity = 'robot_task'
    data_test = VideoDataset('/data/dataset/tvc/data', dataset, args.n_frames_test,
                             standardise=std_feats, random=False, action_class=activity)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)
    fit_clusters(test_loader)
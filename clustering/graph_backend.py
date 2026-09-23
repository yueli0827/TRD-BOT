import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, features):
        return self.ln2(self.fc2(F.gelu(self.ln1(self.fc1(features)))))


def build_knn_neighbors(features, n_neighbors=20):
    count = len(features)
    neighbors = min(int(n_neighbors), count - 1)
    if neighbors < 1:
        return np.empty((count, 0), dtype=np.int64)
    index = NearestNeighbors(n_neighbors=neighbors + 1, metric="cosine")
    candidates = index.fit(features).kneighbors(features, return_distance=False)
    return np.stack([row[row != frame][:neighbors] for frame, row in enumerate(candidates)])


def info_nce_loss(anchor, positive, temperature=0.2):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative_logits = anchor @ anchor.T / temperature
    diagonal = torch.eye(len(anchor), device=anchor.device, dtype=torch.bool)
    negative_logits = negative_logits.masked_fill(diagonal, -torch.inf)
    positive_logits = (anchor * positive).sum(dim=1, keepdim=True) / temperature
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    return F.cross_entropy(logits, torch.zeros(len(anchor), dtype=torch.long, device=anchor.device))


def train_graph_embeddings(features, seed=0, epochs=200, batch_size=1024, device=None):
    if epochs < 1 or batch_size < 2:
        raise ValueError("Graph training requires positive epochs and a batch size of at least two.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    neighbors = build_knn_neighbors(features)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        encoder = MLPEncoder(features.shape[1]).to(device)
        projection = nn.Linear(128, 128).to(device)
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(projection.parameters()), lr=1e-3, weight_decay=1e-5)
        data = torch.from_numpy(np.asarray(features, dtype=np.float32))
        for _ in range(epochs):
            order = torch.randperm(len(data), generator=generator)
            for offset in range(0, len(data), batch_size):
                anchors = order[offset:offset + batch_size]
                if len(anchors) < 2:
                    continue
                selection = rng.integers(neighbors.shape[1], size=len(anchors))
                positives = torch.from_numpy(neighbors[anchors.numpy(), selection])
                anchor_input = F.dropout(data[anchors].to(device), p=0.2, training=True)
                positive_input = F.dropout(data[positives].to(device), p=0.2, training=True)
                anchor_output = projection(encoder(anchor_input))
                positive_output = projection(encoder(positive_input))
                loss = info_nce_loss(anchor_output, positive_output)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        encoder.eval()
        projection.eval()
        output = []
        with torch.no_grad():
            for chunk in data.split(4096):
                output.append(F.normalize(projection(encoder(chunk.to(device))), dim=1).cpu().numpy())
    return np.concatenate(output, axis=0)


def cluster_graph(features, n_clusters, seed=0):
    embeddings = train_graph_embeddings(features, seed=seed)
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20, max_iter=300).fit_predict(embeddings).astype(np.int64)

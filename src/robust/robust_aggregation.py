import torch
import numpy as np

def krum_aggregation(client_weights, num_adv):
    """
    Krum aggregation: Select the model closest to the majority of clients
    client_weights: List[dict], each dict is model parameters
    num_adv: Estimated number of adversarial clients
    """
    num_clients = len(client_weights)
    # Flatten each state_dict into a vector
    weight_vectors = [torch.nn.utils.parameters_to_vector([v.cpu().float() for v in w.values()]) for w in client_weights]
    scores = []
    for i, w_i in enumerate(weight_vectors):
        distances = [torch.norm(w_i - w_j).item() for j, w_j in enumerate(weight_vectors) if i != j]
        distances.sort()
        scores.append(sum(distances[:num_clients - num_adv - 2]))
    krum_idx = scores.index(min(scores))
    return client_weights[krum_idx]

def median_aggregation(client_weights):
    """
    Median aggregation: Take the median of all clients for each parameter
    """
    keys = client_weights[0].keys()
    median_weights = {}
    for key in keys:
        stacked = torch.stack([w[key].cpu().float() for w in client_weights], dim=0)
        median_weights[key] = torch.median(stacked, dim=0)[0]
    return median_weights

def trimmed_mean_aggregation(client_weights, trim_ratio=0.1):
    """
    Trimmed Mean aggregation: Remove extreme values for each parameter and then average
    trim_ratio: The ratio to remove (e.g., 0.1 means removing 10% max and min)
    """
    keys = client_weights[0].keys()
    trimmed_mean_weights = {}
    n = len(client_weights)
    trim_k = int(n * trim_ratio)
    for key in keys:
        stacked = torch.stack([w[key].cpu().float() for w in client_weights], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[trim_k:n-trim_k, ...] if n - 2*trim_k > 0 else sorted_vals
        trimmed_mean_weights[key] = trimmed.mean(dim=0)
    return trimmed_mean_weights 
import numpy as np
import torch


def get_mitigation_method(name):
    if name == 'fedavg':
        return apply_fedavg
    if name == "fairfed":
        return apply_fairfed
    elif name == "fairfate":
        return apply_fairfate
    elif name == "fairguard":
        return apply_fairguard
    elif name == "krum":
        return aggregate_with_krum
    elif name == "fltrust":
        return aggregate_with_fltrust
    else:
        raise ValueError(f"Unknown mitigation method: {name}")

def apply_fedavg(local_models, *args, **kwargs):
    print("[FedAvg] Standard averaging.")
    return simple_average(local_models)

def apply_fairfed(local_models, global_model, fairness_gaps, alpha=0.5):
    print("[FairFed] Aggregating with fairness gap reweighting.")
    if fairness_gaps is None or len(fairness_gaps) != len(local_models):
        print("  Warning: fairness_gaps not provided, using simple average.")
        return simple_average(local_models)
    # print(fairness_gaps)
    weights = np.array([np.exp(-alpha * g) for g in fairness_gaps])
    weights /= np.sum(weights)

    agg = []
    for layers in zip(*local_models):
        weighted_sum = sum(w * l for w, l in zip(weights, layers))
        agg.append(weighted_sum)
    return agg

def apply_fairfate(local_models, global_model, momentum_buf, momentum=0.9, *args, **kwargs):
    print("[FAIR-FATE] Momentum-based fairness-aware aggregation.")
    avg = simple_average(local_models)
    if momentum_buf is None:
        return avg
    return [momentum * m + (1 - momentum) * a for m, a in zip(momentum_buf, avg)]

def apply_fairguard(local_models, global_model, synthetic_data, model_class, criterion, device="cpu", threshold=0.1):
    print("[FairGuard] Filtering clients based on fairness deviation on synthetic samples.")
    gap_scores = []
    for model_params in local_models:
        model = model_class()
        for param, new in zip(model.parameters(), model_params):
            param.data = torch.tensor(new).to(param.device)
        model.eval()
        with torch.no_grad():
            X_syn, A_syn = synthetic_data
            preds = model(torch.tensor(X_syn).float().to(device)).argmax(dim=1).cpu().numpy()
        # Estimate gap: dummy logic (std of predictions per group)
        groups = np.unique(A_syn)
        means = [np.mean(preds[A_syn == g]) for g in groups]
        gap = np.max(means) - np.min(means)
        gap_scores.append(gap)

    # Downweight clients with high gap
    weights = np.array([np.exp(-gap) for gap in gap_scores])
    weights /= np.sum(weights)

    agg = []
    for layers in zip(*local_models):
        weighted_sum = sum(w * l for w, l in zip(weights, layers))
        agg.append(weighted_sum)
    return agg

def aggregate_with_krum(local_models, *args, **kwargs):
    print("[Krum] Byzantine-robust aggregation.")
    distances = []
    for i, model_i in enumerate(local_models):
        dists = []
        for j, model_j in enumerate(local_models):
            if i != j:
                dist = sum(np.sum((a - b)**2) for a, b in zip(model_i, model_j))
                dists.append(dist)
        # 求 n-2 个最近邻的总距离
        krum_score = sum(sorted(dists)[:len(local_models) - 2])
        distances.append((i, krum_score))
    selected_idx = sorted(distances, key=lambda x: x[1])[0][0]
    return local_models[selected_idx]


def aggregate_with_fltrust(local_models, server_update, *args, **kwargs):
    print("[FLTrust] Aggregating using cosine similarity to server update.")
    def cosine_sim(a, b):
        return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    weights = []
    for model in local_models:
        sims = [cosine_sim(p, s) for p, s in zip(model, server_update)]
        weights.append(np.mean(sims))
    weights = np.maximum(weights, 0)
    if np.sum(weights) == 0:
        weights = np.ones(len(local_models)) / len(local_models)
    else:
        weights = weights / np.sum(weights)

    agg = []
    for layers in zip(*local_models):
        weighted_sum = sum(w * l for w, l in zip(weights, layers))
        agg.append(weighted_sum)
    return agg

def simple_average(local_models):
    return [np.mean(params, axis=0) for params in zip(*local_models)]
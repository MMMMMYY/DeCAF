import numpy as np
import torch

def compute_distance_to_boundary(model, X, y, criterion):
    model.eval()
    X_tensor = torch.tensor(X).float()
    X_tensor.requires_grad = True
    outputs = model(X_tensor)
    probs = torch.softmax(outputs, dim=1)
    loss = criterion(outputs, torch.tensor(y).long())
    loss.backward(torch.ones_like(loss))
    gradients = X_tensor.grad
    distances = gradients.norm(p=2, dim=1).detach().numpy()
    return distances

def concealed_target_class(X, y, model, criterion, num_classes):
    class_distances = []
    for c in range(num_classes):
        indices = np.where(y == c)[0]
        if len(indices) == 0:
            class_distances.append(float("inf"))
            continue
        dist = compute_distance_to_boundary(model, X[indices], y[indices], criterion)
        class_distances.append(np.mean(dist))
    target_class = np.argmin(class_distances)
    return target_class

def select_vulnerable_samples(X_class, y_class, model, criterion, alpha=0.3):
    distances = compute_distance_to_boundary(model, X_class, y_class, criterion)
    k = int(len(X_class) * alpha)
    sorted_indices = np.argsort(distances)[:k]
    return sorted_indices, distances[sorted_indices]

def apply_model_poisoning(X_poisoned, y_poisoned, X_clean, y_clean, model, criterion, optimizer, lambda_attack=0.9, max_norm=1e-5):
    model.train()
    Xp = torch.tensor(X_poisoned).float()
    yp = torch.tensor(y_poisoned).long()
    Xc = torch.tensor(X_clean).float()
    yc = torch.tensor(y_clean).long()

    # poisoned loss
    optimizer.zero_grad()
    model.eval()
    loss_p = criterion(model(Xp), yp)
    loss_p.backward()
    grads_poison = [p.grad.clone() for p in model.parameters()]

    # clean loss
    optimizer.zero_grad()
    model.eval()
    loss_c = criterion(model(Xc), yc)
    loss_c.backward()
    grads_clean = [p.grad.clone() for p in model.parameters()]

    # only apply poisoning on last layer (assume last layer is -1)
    for i, (p, gp, gc) in enumerate(zip(model.parameters(), grads_poison, grads_clean)):
        if i == len(grads_poison) - 1:
            # normalize and clip poisoned gradient
            gp = gp / (gp.norm() + 1e-6)
            gp = torch.clamp(gp, -max_norm, max_norm)
            combined = (1 - lambda_attack) * gc + lambda_attack * gp
            p.grad = combined
        else:
            p.grad = gc

    optimizer.step()

def apply_data_poisoning(y, selected_indices):
    y_poisoned = y.copy()
    # unique_classes = np.unique(y)

    # if len(unique_classes) == 2:
    y_poisoned[selected_indices] = 1 - y_poisoned[selected_indices]
    # else:
    #     for idx in selected_indices:
    #         current_label = y_poisoned[idx]
    #         other_classes = list(set(unique_classes) - {current_label})
    #         y_poisoned[idx] = np.random.choice(other_classes)

    return y_poisoned

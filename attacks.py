import numpy as np

def label_flip_attack(y, A, flip_ratio, target_group):
    y_poisoned = y.copy()
    unique_classes = np.unique(y)
    indices = np.where(A == target_group)[0]
    num_to_flip = int(len(indices) * flip_ratio)
    if num_to_flip > 0:
        flip_indices = np.random.choice(indices, size=num_to_flip, replace=False)
        for i in flip_indices:
            current = y_poisoned[i]
            if len(unique_classes) == 2:
                y_poisoned[i] = 1 - current
            else:
                other_classes = list(set(unique_classes) - {current})
                y_poisoned[i] = np.random.choice(other_classes)
    return y_poisoned


def attribute_flip_attack(A, y, flip_ratio, target_group):
    A_poisoned = A.copy()
    indices = np.where(y == target_group)[0]
    num_to_flip = int(len(indices) * flip_ratio)
    if num_to_flip > 0:
        flip_indices = np.random.choice(indices, size=num_to_flip, replace=False)
        A_poisoned[flip_indices] = 1 - A_poisoned[flip_indices]
    return A_poisoned

def hybrid_flip_attack(y, A, flip_ratio, target_group):
    y_poisoned = y.copy()
    A_poisoned = A.copy()
    unique_classes = np.unique(y)
    indices = np.where(A == target_group)[0]
    num_to_flip = int(len(indices) * flip_ratio)
    if num_to_flip > 0:
        flip_indices = np.random.choice(indices, size=num_to_flip, replace=False)
        for i in flip_indices:
            current = y_poisoned[i]
            if len(unique_classes) == 2:
                y_poisoned[i] = 1 - current
            else:
                other_classes = list(set(unique_classes) - {current})
                y_poisoned[i] = np.random.choice(other_classes)
    return y_poisoned, A_poisoned

def double_flip_attack(y, A, flip_ratio, target_group):
    y_poisoned, A_poisoned = label_flip_attack(y, A, flip_ratio, target_group), A.copy()
    A_poisoned = attribute_flip_attack(A_poisoned, y, flip_ratio, target_group)
    return y_poisoned, A_poisoned

def poison_data(y, A, attack, flip_ratio, target_group):
    if attack == "label_flip":
        return label_flip_attack(y, A, flip_ratio, target_group), A
    elif attack == "attribute_flip":
        return y, attribute_flip_attack(A, y, flip_ratio, target_group)
    elif attack == "hybrid_flip":
        return hybrid_flip_attack(y, A, flip_ratio, target_group)
    elif attack == "double_flip":
        return double_flip_attack(y, A, flip_ratio, target_group)
    else:
        raise ValueError(f"Unknown attack type: {attack}")
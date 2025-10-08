import numpy as np
import torch

def split_data_among_clients(X, y, A, num_clients):
    data_size = len(y)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, num_clients)
    client_data = [(X[idx], y[idx], A[idx]) for idx in split_indices]
    return client_data

def get_model_parameters(model):
    return [param.data.cpu().numpy() for param in model.parameters()]

def set_model_parameters(model, parameters):
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param).to(param.device)

def average_model_parameters(param_list):
    avg_params = []
    for params in zip(*param_list):
        avg_params.append(np.mean(params, axis=0))
    return avg_params


def train_model(model, criterion, optimizer, X_train, y_train, epochs=20):
    model.train()

    if isinstance(X_train, torch.Tensor):
        X_train_tensor = X_train.clone().detach().float()
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float)

    if isinstance(y_train, torch.Tensor):
        y_train_tensor = y_train.clone().detach().long().view(-1)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).view(-1)

    # print("X shape:", X_train_tensor.shape)
    # print("y shape:", y_train_tensor.shape)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        # print("Model output shape:", outputs.shape)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()




def test_model(model, X_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float()
        predictions = model(X_test_tensor).argmax(dim=1).numpy()
    return predictions
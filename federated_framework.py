import torch
import torch.nn as nn
import numpy as np
import argparse

from models import get_model
from datasets_utils import load_dataset_by_name
from attacks import poison_data
from evaluation import evaluate_fairness_binary, evaluate_fairness_multiclass
from mitigation import get_mitigation_method
from concealed_attack import concealed_target_class, select_vulnerable_samples, apply_data_poisoning, apply_model_poisoning
from utils import (
    split_data_among_clients, get_model_parameters, set_model_parameters,
    average_model_parameters, train_model, test_model
)
import random


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def federated_training(args):
    X_train, y_train, A_train, X_test, y_test, A_test = load_dataset_by_name(args.dataset, args.sensitive_attr)
    clients = list(split_data_among_clients(X_train, y_train, A_train, args.num_clients))  # ✅ 转为 list

    num_poisoned = int(args.num_clients * args.poisoned_ratio)
    poisoned_clients = set(np.random.choice(args.num_clients, size=num_poisoned, replace=False))
    print(f"Poisoned Clients: {sorted(poisoned_clients)}")

    global_model, criterion, _ = get_model(args.model, input_dim=X_train.shape[1], num_classes=args.num_classes)
    aggregate_fn = get_mitigation_method(args.mitigation)
    momentum_buffer = None

    for rnd in range(args.rounds):
        print(f"\nCommunication Round {rnd+1}/{args.rounds}")
        local_params = []

        for cid, (X_c, y_c, A_c) in enumerate(clients):
            assert len(X_c) == len(y_c), f"[Client {cid}] X_c: {X_c.shape}, y_c: {y_c.shape}"  # ✅ Debug check

            local_model, _, optimizer = get_model(args.model, input_dim=X_train.shape[1], num_classes=args.num_classes)
            set_model_parameters(local_model, get_model_parameters(global_model))

            if args.attack == "concealed" and cid in poisoned_clients:
                group_idx = np.where(A_c == args.target_group)[0]
                if len(group_idx) > 0:
                    X_cg, y_cg = X_c[group_idx], y_c[group_idx]
                    target_cls = concealed_target_class(X_cg, y_cg, local_model, criterion, args.num_classes)
                    idx_cls = group_idx[np.where(y_c[group_idx] == target_cls)[0]]
                    if len(idx_cls) > 0:
                        X_cls, y_cls = X_c[idx_cls], y_c[idx_cls]
                        sel_idx, _ = select_vulnerable_samples(X_cls, y_cls, local_model, criterion, alpha=args.flip_ratio)
                        final_sel_idx = idx_cls[sel_idx]
                        if args.concealed_mode == "data":
                            y_c = apply_data_poisoning(y_c, final_sel_idx)
                        elif args.concealed_mode == "model":
                            clean_idx = np.array([i for i in range(len(X_c)) if i not in final_sel_idx])
                            apply_model_poisoning(X_c[final_sel_idx], y_c[final_sel_idx],
                                                  X_c[clean_idx], y_c[clean_idx],
                                                  local_model, criterion, optimizer)
            elif args.attack != 'none' and cid in poisoned_clients:
                y_c, A_c = poison_data(y_c, A_c, args.attack, args.flip_ratio, args.target_group)

            if args.concealed_mode != "model":
                train_model(local_model, criterion, optimizer, X_c, y_c, epochs=args.local_epochs)

            local_params.append(get_model_parameters(local_model))

        if args.mitigation == "fltrust":
            server_update = get_model_parameters(global_model)
            averaged = aggregate_fn(local_params, server_update)
        elif args.mitigation == "fairguard":
            X_syn = np.random.normal(0, 1, size=(200, X_train.shape[1]))
            A_syn = np.random.choice(np.unique(A_train), size=200)
            synthetic_data = (X_syn, A_syn)
            model_class = lambda: get_model(args.model, input_dim=X_train.shape[1], num_classes=args.num_classes)[0]
            averaged = aggregate_fn(local_params, global_model, synthetic_data, model_class, criterion)
        elif args.mitigation == "fairfed":
            fairness_gaps = []
            for i, model_param in enumerate(local_params):
                model = get_model(args.model, input_dim=X_train.shape[1], num_classes=args.num_classes)[0]
                set_model_parameters(model, model_param)

                # ✅ 重新取出每个客户端自己的数据
                X_c, y_c, A_c = clients[i]

                preds_i = test_model(model, X_c)
                if args.num_classes == 2:
                    metric_i = evaluate_fairness_binary(y_c, preds_i, A_c)
                else:
                    metric_i = evaluate_fairness_multiclass(y_c, preds_i, A_c, average="macro")
                fairness_gaps.append(abs(metric_i["EOD"]))
            averaged = aggregate_fn(local_params, global_model, fairness_gaps)
        else:
            averaged = aggregate_fn(local_params, global_model, momentum_buffer)

        set_model_parameters(global_model, averaged)
        momentum_buffer = averaged

    # Final Evaluation
    preds = test_model(global_model, X_test)
    print("Label distribution in test set:", np.bincount(y_test))

    if args.num_classes == 2:
        metrics = evaluate_fairness_binary(y_test, preds, A_test)
    else:
        metrics = evaluate_fairness_multiclass(y_test, preds, A_test, average="macro")

    print("Evaluation Results after Federated Learning:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="utkface")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--num_clients", type=int, default=50)
    parser.add_argument("--poisoned_ratio", type=float, default=0.1)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--flip_ratio", type=float, default=0.3)
    parser.add_argument("--attack", type=str, default="label_flip")
    parser.add_argument("--target_group", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sensitive_attr", type=str, default="race")
    parser.add_argument("--mitigation", type=str, default="fedavg")
    parser.add_argument("--concealed_mode", type=str, default="data", help="data or model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # 42 for all,
    args = parser.parse_args()
    set_random_seed(args.seed)
    output = federated_training(args)
    # Save results
    with open(
            f"federated_{args.mitigation}_{args.attack}_{args.concealed_mode}_{args.dataset}_{args.sensitive_attr}.txt",
            "w") as f_out:
        f_out.write("Experiment Configuration:\n")
        for arg in vars(args):
            f_out.write(f"{arg}: {getattr(args, arg)}\n")
        f_out.write("\nEvaluation Results:\n")
        for key, value in output.items():
            f_out.write(f"{key}: {value:.4f}\n")

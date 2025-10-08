import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision import datasets, transforms
import torch
from transformers import BertTokenizer
from datasets import load_dataset
from PIL import Image


DATA_DIR = "data"

def download_file(url, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path

def load_adult(sensitive_attr='sex'):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    filename = "adult.csv"
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    if not os.path.exists(filename):
        df = pd.read_csv(url, header=None, names=column_names, skipinitialspace=True)
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)

    df = df.replace('?', pd.NA).dropna()

    y = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    A = df[sensitive_attr]
    X = df.drop(columns=['income'])

    X_encoded = pd.get_dummies(X.drop(columns=[sensitive_attr]), drop_first=True)

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X_encoded, y, A, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train.values, A_train.values, X_test, y_test.values, A_test.values

def load_drug(sensitive_attr="Gender"):
    path = download_file(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data",
        "drug_consumption.data"
    )
    df = pd.read_csv(path, header=None)
    df.columns = ["ID", "Age", "Gender", "Education", "Country", "Ethnicity",
                  "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS",
                  "Alcohol", "Amphet", "Amyl", "Benzos", "Caffeine", "Cannabis",
                  "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
                  "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"]

    df.drop(columns=["ID"], inplace=True)

    # 只保留合法 Cannabis 标签
    valid_levels = ["CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]
    df = df[df["Cannabis"].isin(valid_levels)]
    df["label"] = df["Cannabis"].map({lvl: i for i, lvl in enumerate(valid_levels)})

    # 去掉缺失值的敏感属性和目标变量
    df = df[df[sensitive_attr].notna()]
    df = df[df["label"].notna()]

    # 分离 label 和敏感属性
    y = df["label"].values
    A = LabelEncoder().fit_transform(df[sensitive_attr].values)

    df.drop(columns=["Cannabis", "label", sensitive_attr], inplace=True)

    # 对所有非数值列做 One-Hot 编码，数值列保留
    df = pd.get_dummies(df)

    # 标准化
    X = StandardScaler().fit_transform(df)

    # Train-test split
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A, test_size=0.2, random_state=42
    )
    return X_train, y_train, A_train, X_test, y_test, A_test




def load_utkface(sensitive_attr="race"):
    data_path = os.path.join(DATA_DIR, "UTKFace")
    if not os.path.exists(data_path):
        print(f"⚠️ Please manually download and unzip UTKFace into: {data_path}")

    images = []
    labels = []
    sens_attrs = []

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    for filename in os.listdir(data_path):
        if filename.endswith(".jpg"):
            try:
                age, gender, race, _ = filename.split("_")
                img_path = os.path.join(data_path, filename)
                image = Image.open(img_path).convert("RGB")
                image = transform(image)
                # label: age group, sensitive: race or gender
                age = int(age)
                if age < 20:
                    age_label = 0
                elif age < 40:
                    age_label = 1
                else:
                    age_label = 2
                images.append(image)
                labels.append(age_label)
                if sensitive_attr == "race":
                    sens_attrs.append(int(race))
                elif sensitive_attr == "gender":
                    sens_attrs.append(int(gender))
                else:
                    raise ValueError("sensitive_attr must be 'race' or 'gender'")
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")
                continue

    X = torch.stack(images)
    y = torch.tensor(labels)
    A = torch.tensor(sens_attrs)

    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42)
    return X_train, y_train, A_train, X_test, y_test, A_test

def load_fairlex(sensitive_attr="gender"):
    dataset = load_dataset("lex_glue", "ecthr_a")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        texts = examples["text"]
        if isinstance(texts[0], list):
            texts = [" ".join(t) for t in texts]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    tokenized = dataset.map(tokenize_function, batched=True)

    label_column = "labels"
    y_train = torch.tensor(np.array(dataset["train"][label_column])).flatten()
    X_train = torch.tensor(tokenized["train"]["input_ids"], dtype=torch.long)
    y_test = torch.tensor(np.array(dataset["test"][label_column])).flatten()
    X_test = torch.tensor(tokenized["test"]["input_ids"], dtype=torch.long)

    # Add dummy sensitive attribute if not present
    if sensitive_attr not in dataset["train"].column_names:
        dataset["train"] = dataset["train"].add_column(sensitive_attr, [0] * len(dataset["train"]))
        dataset["test"] = dataset["test"].add_column(sensitive_attr, [0] * len(dataset["test"]))

    A_train = torch.tensor(dataset["train"][sensitive_attr], dtype=torch.long)
    A_test = torch.tensor(dataset["test"][sensitive_attr], dtype=torch.long)

    return X_train, y_train, A_train, X_test, y_test, A_test



def load_dataset_by_name(name, sensitive_attr="sex"):
    if name == "adult":
        return load_adult(sensitive_attr)
    elif name == "drug":
        return load_drug(sensitive_attr)
    elif name == "utkface":
        return load_utkface(sensitive_attr)
    elif name == "fairlex":
        return load_fairlex(sensitive_attr)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

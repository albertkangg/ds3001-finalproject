#!/usr/bin/env python3

import os
import random
import multiprocessing
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition    import PCA
from sklearn.preprocessing   import StandardScaler, PowerTransformer, normalize
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics         import accuracy_score, classification_report, confusion_matrix
from scipy.stats             import randint
import matplotlib.pyplot as plt

DATASET_ROOT   = "Indian Food Images/Indian Food Images"
TRAIN_FRACTION = 0.8
SEED           = 42
BATCH_SIZE     = 64
NUM_WORKERS    = 0
PCA_DIM        = 128
RF_ITER        = 20
RF_CV          = 3

class FolderImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path,label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def extract_features(model, loader, device):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            feats.append(out.cpu().numpy())
            labs.append(labels.numpy())
    return np.vstack(feats), np.hstack(labs)

def main():
    multiprocessing.freeze_support()

    # 1) Discover and select half the class folders
    classes = sorted(
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    )
    if not classes:
        raise RuntimeError(f"No class folders found in {DATASET_ROOT}")
    random.seed(SEED)
    selected = random.sample(classes, k=len(classes)//2)
    print(f"Selected {len(selected)}/{len(classes)} classes")

    # 2) Build (image_path, label_idx) list
    samples = []
    for idx, cls in enumerate(selected):
        folder = os.path.join(DATASET_ROOT, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                samples.append((os.path.join(folder, fn), idx))
    if not samples:
        raise RuntimeError("No images found in selected classes!")
    print("Total samples:", len(samples))

    # 3) Split into train/test (stratified)
    paths, labels = zip(*samples)
    tr_p, te_p, tr_l, te_l = train_test_split(
        paths, labels,
        train_size=TRAIN_FRACTION,
        stratify=labels,
        random_state=SEED
    )
    train_samples = list(zip(tr_p, tr_l))
    test_samples  = list(zip(te_p, te_l))
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # 4) Data transforms & loaders
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225],
        ),
    ])
    train_loader = DataLoader(
        FolderImageDataset(train_samples, transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        FolderImageDataset(test_samples, transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # 5) Load ResNet-50 and strip classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    ).to(device)
    resnet.fc = nn.Identity()
    for p in resnet.parameters():
        p.requires_grad = False

    # 6) Extract embeddings
    print("Extracting train features...")
    X_train, y_train = extract_features(resnet, train_loader, device)
    print("Extracting test features...")
    X_test,  y_test  = extract_features(resnet, test_loader,  device)
    print("Raw feature shapes:", X_train.shape, X_test.shape)

    # 7) Preprocess embeddings
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(X_train)
    X_train = pca.transform(X_train)
    X_test  = pca.transform(X_test)

    pt = PowerTransformer(method="yeo-johnson").fit(X_train)
    X_train = pt.transform(X_train)
    X_test  = pt.transform(X_test)

    X_train = normalize(X_train, norm="l2", axis=1)
    X_test  = normalize(X_test,  norm="l2", axis=1)

    # 8) Baseline Random Forest
    print("Training baseline RF (100 trees)...")
    rf0 = RandomForestClassifier(
        n_estimators=100, random_state=SEED, n_jobs=-1
    )
    rf0.fit(X_train, y_train)
    acc0 = accuracy_score(y_test, rf0.predict(X_test))
    print(f"Baseline RF accuracy: {acc0*100:.2f}%")

    # 9) Hyperparameter search
    print("Running RandomizedSearchCV on RF...")
    param_dist = {
        "n_estimators": randint(100, 1001),
        "max_depth":    [None, 10, 20, 50],
        "max_features": ["sqrt", "log2", 0.5],
        "min_samples_leaf": randint(1,5),
        "bootstrap":    [True, False],
    }
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=RF_ITER, cv=RF_CV,
        scoring="accuracy",
        random_state=SEED,
        n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"Best CV accuracy: {search.best_score_*100:.2f}%")
    print("Best parameters:", search.best_params_)

    # 10) Evaluate best RF on test
    best_rf = search.best_estimator_
    y_pred  = best_rf.predict(X_test)
    acc_best= accuracy_score(y_test, y_pred)
    print(f"Test accuracy (best RF): {acc_best*100:.2f}%\n")

    print("Classification report:")
    names = [selected[i] for i in sorted(set(y_test))]
    print(classification_report(
        y_test, y_pred,
        target_names=names,
        zero_division=0
    ))

    # 11) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Random Forest Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

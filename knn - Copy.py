#!/usr/bin/env python3
import os, random, multiprocessing
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition   import PCA
from sklearn.preprocessing  import StandardScaler, PowerTransformer, normalize
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.metrics        import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

DATASET_ROOT = "Indian Food Images/Indian Food Images"
TRAIN_FRAC   = 0.8
SEED         = 42
BATCH_SIZE   = 64
NUM_WORKERS  = 0
PCA_DIM      = 128
K_NEIGHBORS  = 5

class FolderImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path,label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
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

    # 1) Discover class folders & pick half
    classes = sorted(d for d in os.listdir(DATASET_ROOT)
                     if os.path.isdir(os.path.join(DATASET_ROOT, d)))
    random.seed(SEED)
    selected = random.sample(classes, k=len(classes)//2)
    print(f"Selected {len(selected)}/{len(classes)} classes")

    # 2) Build (path,label) list
    samples = []
    for idx,cls in enumerate(selected):
        folder = os.path.join(DATASET_ROOT, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png",".jpg","jpeg",".bmp")):
                samples.append((os.path.join(folder,fn), idx))
    if not samples:
        raise RuntimeError("No images found!")
    print("Total:", len(samples))

    # 3) Stratified train/test split
    paths, labels = zip(*samples)
    tr_p, te_p, tr_l, te_l = train_test_split(
        paths, labels,
        train_size=TRAIN_FRAC,
        stratify=labels,
        random_state=SEED
    )
    train_samples = list(zip(tr_p, tr_l))
    test_samples  = list(zip(te_p, te_l))
    print(f"Train {len(train_samples)}, Test {len(test_samples)}")

    # 4) DataLoader setup
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225]),
    ])
    train_ds = FolderImageDataset(train_samples, tf)
    test_ds  = FolderImageDataset(test_samples,  tf)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True)

    # 5) Load ResNet-50 & strip classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    resnet = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    ).to(device)
    resnet.fc = nn.Identity()
    for p in resnet.parameters(): p.requires_grad = False

    # 6) Extract 512-dim features
    print("Extracting train features…")
    X_train, y_train = extract_features(resnet, train_ld, device)
    print("Extracting test features…")
    X_test,  y_test  = extract_features(resnet,  test_ld,  device)

    # 7) Standard scale → PCA → power transform → normalize
    print("Preprocessing embeddings…")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    pca = PCA(n_components=PCA_DIM, whiten=False, random_state=SEED)
    X_train = pca.fit_transform(X_train)
    X_test  = pca.transform(X_test)

    pt = PowerTransformer(method="yeo-johnson")
    X_train = pt.fit_transform(X_train)
    X_test  = pt.transform(X_test)

    X_train = normalize(X_train, norm="l2", axis=1)
    X_test  = normalize(X_test,  norm="l2", axis=1)

    # 8) Train & evaluate KNN with cosine metric
    print(f"Training KNN (k={K_NEIGHBORS}, metric=cosine)…")
    knn = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights="distance",
        metric="cosine",
        n_jobs=-1
    )
    knn.fit(X_train, y_train)

    print("Predicting test set…")
    y_pred = knn.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy: {acc*100:.2f}%\n")

    names = [selected[i] for i in sorted(set(y_test))]
    print(classification_report(
        y_test, y_pred,
        target_names=names,
        zero_division=0
    ))
    
    # confusion matrix

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(cax)

    classes = [selected[i] for i in range(len(cm))]
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

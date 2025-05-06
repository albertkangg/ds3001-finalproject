#!/usr/bin/env python3
import os, random, multiprocessing
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition    import PCA
from sklearn.preprocessing   import StandardScaler, PowerTransformer, normalize
from sklearn.cluster         import KMeans
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATASET_ROOT   = "Indian Food Images/Indian Food Images"
TRAIN_FRACTION = 0.8
SEED           = 42
BATCH_SIZE     = 64
NUM_WORKERS    = 0
PCA_DIM        = 128

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

    # 1) select half the classes
    classes = sorted(d for d in os.listdir(DATASET_ROOT)
                     if os.path.isdir(os.path.join(DATASET_ROOT, d)))
    random.seed(SEED)
    selected = random.sample(classes, k=len(classes)//2)
    print(f"Selected {len(selected)} classes")

    # 2) gather samples
    samples = []
    for idx,cls in enumerate(selected):
        for fn in os.listdir(os.path.join(DATASET_ROOT,cls)):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp")):
                samples.append((os.path.join(DATASET_ROOT,cls,fn), idx))
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

    # 3) DataLoaders
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ld = DataLoader(FolderImageDataset(train_samples, tf),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_ld  = DataLoader(FolderImageDataset(test_samples,  tf),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # 4) ResNet-50 feature extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet.fc = nn.Identity()
    for p in resnet.parameters(): p.requires_grad = False

    print("Extracting train features...")
    X_train, y_train = extract_features(resnet, train_ld, device)
    print("Extracting test features...")
    X_test,  y_test  = extract_features(resnet,  test_ld,  device)

    # 5) Preprocess embeddings
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

    # 6) Sweep cluster counts
    cluster_range = list(range(2, len(selected)+1))
    accuracies = []
    print("Sweeping KMeans cluster counts...")
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        train_cl = kmeans.fit_predict(X_train)
        mapping = {}
        for c in range(k):
            lbls = y_train[train_cl==c]
            mapping[c] = np.bincount(lbls).argmax() if len(lbls)>0 else -1
        test_cl = kmeans.predict(X_test)
        y_pred = np.array([mapping[c] for c in test_cl])
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"clusters={k:2d}, accuracy={acc*100:5.2f}%")

    plt.figure(figsize=(8,5))
    plt.plot(cluster_range, accuracies, marker='o')
    plt.title("KMeans Clusters vs Accuracy")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Accuracy")
    plt.xticks(cluster_range, rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

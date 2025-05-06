from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

# ——— User configuration ———
ROOT_DIR  = Path("/Users/nathansuh/Downloads/sp25/ds3002/project/archive (3)/Indian Food Images/Indian Food Images") 
HIST_BINS = 32                                        # bins per channel
# ——————————————————————————

if not (ROOT_DIR.exists() and ROOT_DIR.is_dir()):
    print(f"ERROR: {ROOT_DIR} is not a valid directory", file=sys.stderr)
    sys.exit(1)

# Gather class directories
class_dirs  = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir()])
class_names = [d.name for d in class_dirs]
n = len(class_names)

# Compute per-class grayscale and color histograms
gray_hists = []
color_hists = []
for cls in class_dirs:
    gh_list, ch_list = [], []
    for img_path in cls.glob("*.jpg"):
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            # grayscale histogram
            gray = np.array(img.convert("L"))
            gh, _ = np.histogram(gray, bins=HIST_BINS, range=(0,255))
            gh_list.append(gh / gh.sum())
            # color histogram (R, G, B)
            rh, _ = np.histogram(arr[:,:,0], bins=HIST_BINS, range=(0,255))
            ghc, _ = np.histogram(arr[:,:,1], bins=HIST_BINS, range=(0,255))
            bh, _ = np.histogram(arr[:,:,2], bins=HIST_BINS, range=(0,255))
            ch = np.concatenate([rh, ghc, bh]).astype(float)
            ch /= ch.sum()
            ch_list.append(ch)
        except Exception:
            continue
    gray_hists.append(np.mean(gh_list, axis=0) if gh_list else np.zeros(HIST_BINS))
    color_hists.append(np.mean(ch_list, axis=0) if ch_list else np.zeros(3*HIST_BINS))

# Build distance matrices
gray_dist  = np.zeros((n,n))
color_dist = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        gray_dist[i,j]  = np.linalg.norm(gray_hists[i] - gray_hists[j])
        color_dist[i,j] = np.linalg.norm(color_hists[i] - color_hists[j])

# 1) Grayscale heatmap (moderate size)
plt.figure(figsize=(12,10))
plt.imshow(gray_dist, cmap='viridis', aspect='auto')
plt.colorbar(label='Euclidean Distance', fraction=0.046, pad=0.04)
plt.xticks(np.arange(n), class_names, rotation=90, fontsize=8)
plt.yticks(np.arange(n), class_names, fontsize=8)
plt.title("Grayscale Histogram Distance", fontsize=14)
plt.tight_layout()
plt.show()

# 2) Color heatmap (moderate size)
plt.figure(figsize=(12,10))
plt.imshow(color_dist, cmap='plasma', aspect='auto')
plt.colorbar(label='Euclidean Distance', fraction=0.046, pad=0.04)
plt.xticks(np.arange(n), class_names, rotation=90, fontsize=8)
plt.yticks(np.arange(n), class_names, fontsize=8)
plt.title("Color Histogram Distance", fontsize=14)
plt.tight_layout()
plt.show()








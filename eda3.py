from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ——— set this to the same base that worked above ———
root = Path("/Users/nathansuh/Downloads/sp25/ds3002/project/archive (3)/Indian Food Images/Indian Food Images")

# Class-balance
counts = {}
for cls in root.iterdir():
    if cls.is_dir():
        jpgs = list(cls.glob("*.jpg")) + list(cls.glob("*.png"))
        counts[cls.name] = len(jpgs)
names, vals = zip(*sorted(counts.items(), key=lambda x:-x[1]))
plt.figure(figsize=(12,4))
plt.bar(names, vals)
plt.xticks(rotation=90, fontsize=8)
plt.title("Images per Dish Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Size & aspect-ratio
widths, heights = [], []
for cls in root.iterdir():
    if not cls.is_dir(): continue
    for img in cls.glob("*.jpg"):
        try:
            w,h = Image.open(img).size
            widths.append(w); heights.append(h)
        except:
            pass

aspect = np.array(widths) / np.array(heights)
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
ax1.hist(widths, bins=20)
ax1.set(title="Width Distribution", xlabel="Width (px)", ylabel="Count")
ax2.hist(aspect, bins=20)
ax2.set(title="Aspect Ratio (w/h)", xlabel="Ratio")
plt.tight_layout()
plt.show()

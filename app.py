import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from pathlib import Path

print("Current folder =", Path.cwd())
# ======================================
# USER SETTINGS
# ======================================

IMAGE_FOLDER = r"C:\Users\PC01\Desktop\images"

PIXEL_SIZE = 1.3886  # um/pixel

LOCAL_WINDOW = 21

MIN_WOUND_WIDTH = 50

CENTER_SEARCH_RATIO = 0.5

# ======================================
# local variance
# ======================================

def local_variance(img, size=21):

    img = img.astype(np.float32)

    mean = uniform_filter(img, size=size)

    mean_sq = uniform_filter(img**2, size=size)

    var = mean_sq - mean**2

    return var

# ======================================
# analyze image
# ======================================

def analyze_image(image_path):

    img = tiff.imread(str(image_path))

    if len(img.shape) == 3:
        img = img[:,:,0]

    img = img.astype(np.float32)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # =====================================
    # CLAHE
    # =====================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img2 = clahe.apply(img)

    # =====================================
    # local variance
    # =====================================

    varmap = local_variance(
        img2,
        31
    )

    varmap = cv2.normalize(
        varmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # =====================================
    # wound mask
    # =====================================

    thresh = np.percentile(varmap, 35)

    wound_mask = (
        varmap < thresh
    ).astype(np.uint8)

    kernel = np.ones((9,9),np.uint8)

    wound_mask = cv2.morphologyEx(
        wound_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    wound_mask = cv2.morphologyEx(
        wound_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =====================================
    # 找最大中央區域
    # =====================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wound_mask,
        connectivity=8
    )

    h, w = wound_mask.shape

    center_x = w // 2

    best_label = None
    best_score = 0

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        if x < center_x < x + width:

            if area > best_score:
                best_score = area
                best_label = i

    final_mask = np.zeros_like(wound_mask)

    if best_label is not None:
        final_mask[labels == best_label] = 1

    # =====================================
    # width measurement
    # =====================================

    widths = []

    left_points = []
    right_points = []

    for y in range(h):

        row = final_mask[y]

        wound_idx = np.where(row > 0)[0]

        if len(wound_idx) < 20:
            continue

        left_edge = wound_idx.min()
        right_edge = wound_idx.max()

        widths.append(
            right_edge - left_edge
        )

        left_points.append(
            (left_edge,y)
        )

        right_points.append(
            (right_edge,y)
        )

    widths = np.array(widths)

    mean_width_px = np.mean(widths)
    median_width_px = np.median(widths)

    result = {

        "mean_width_px": mean_width_px,
        "median_width_px": median_width_px,
        "mean_width_um": mean_width_px * PIXEL_SIZE,
        "median_width_um": median_width_px * PIXEL_SIZE,
        "wound_area_px2": np.sum(final_mask)
    }

    overlay = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    overlay[:,:,1] = np.maximum(
        overlay[:,:,1],
        final_mask * 255
    )

    for x,y in left_points:

        cv2.circle(
            overlay,
            (x,y),
            1,
            (0,255,0),
            -1
        )

    for x,y in right_points:

        cv2.circle(
            overlay,
            (x,y),
            1,
            (0,0,255),
            -1
        )

    return result, overlay

# ======================================
# batch
# ======================================

folder = Path(IMAGE_FOLDER)
print("Folder =", folder)
print("Exists =", folder.exists())
print("Files =", list(folder.glob("*.tif")))
output_dir = folder / "overlay"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)
results = []

for file in sorted(folder.glob("*.tif")):

    print(f"Processing {file.name}")

    result, overlay = analyze_image(file)

    result["image"] = file.name

    results.append(result)

    outname = (
        output_dir /
        f"{file.stem}_overlay.png"
    )

    success = cv2.imwrite(
    str(outname),
    overlay
)

print("Output path =", outname)
print("Save success =", success)

df = pd.DataFrame(results)

# ======================================
# closure
# ======================================

baseline = df.iloc[0]["median_width_px"]

df["closure_percent"] = (
    (baseline - df["median_width_px"])
    / baseline
    * 100
)

df.to_csv(
    folder / "results.csv",
    index=False
)

print(df)

# ======================================
# plot
# ======================================

plt.figure(figsize=(6,4))

plt.plot(
    df["image"],
    df["closure_percent"],
    marker="o"
)

plt.ylabel("Closure (%)")

plt.xlabel("Image")

plt.tight_layout()

plt.savefig(
    folder / "closure_curve.png",
    dpi=300
)

plt.show()

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

# ======================================
# PAGE
# ======================================

st.set_page_config(
    page_title="Scratch Assay Analyzer",
    layout="wide"
)

st.title("Scratch Assay Analyzer")

# ======================================
# USER SETTINGS
# ======================================

PIXEL_SIZE = st.number_input(
    "Pixel Size (um/pixel)",
    value=1.3886,
    format="%.4f"
)

LOCAL_WINDOW = st.slider(
    "Local Variance Window",
    5,
    101,
    31,
    step=2
)

PERCENTILE_THRESHOLD = st.slider(
    "Variance Percentile",
    1,
    99,
    35
)

KERNEL_SIZE = st.slider(
    "Morphology Kernel Size",
    3,
    25,
    5,
    step=2
)

# ======================================
# local variance
# ======================================

def local_variance(img, size):

    img = img.astype(np.float32)

    mean = uniform_filter(
        img,
        size=size
    )

    mean_sq = uniform_filter(
        img**2,
        size=size
    )

    return mean_sq - mean**2

# ======================================
# analyze image
# ======================================

def analyze_image(
    uploaded_file,
    pixel_size,
    local_window,
    percentile_threshold,
    kernel_size
):

    img = tiff.imread(uploaded_file)

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

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    img2 = clahe.apply(img)

    varmap = local_variance(
        img2,
        local_window
    )

    varmap = cv2.normalize(
        varmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 尋找低方差閥值（即傷口與平坦右側細胞的集合）
    thresh = np.percentile(
        varmap,
        percentile_threshold
    )

    # 1 代表潛在的平坦區域（包含傷口與無紋理細胞）
    wound_mask = (varmap < thresh).astype(np.uint8)

    # 形態學基礎去噪
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, kernel)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel)

    h, w = wound_mask.shape

    # ========================================================
    # 新改良：動態尋找左右細胞群的真實邊界，避免右側細胞被誤吞
    # ========================================================
    
    # 建立一個高方差（充滿細胞纹理）的遮罩
    cell_texture_mask = (varmap >= thresh).astype(np.uint8)
    
    # 用較大的形態學閉運算把細胞連成一塊，方便找邊界
    cell_kernel = np.ones((15, 15), np.uint8)
    cell_texture_mask = cv2.morphologyEx(cell_texture_mask, cv2.MORPH_CLOSE, cell_kernel)
    
    # 逐行動態尋找左側細胞的「右邊緣」與右側細胞的「左邊緣」
    final_wound_mask = np.zeros_like(wound_mask)
    
    center_x = w // 2
    
    for y in range(h):
        # 找左半邊有細胞纹理的最右點（即左側細胞群的邊界）
        left_side_cells = np.where(cell_texture_mask[y, :center_x] > 0)[0]
        left_bound = left_side_cells.max() if left_side_cells.size > 0 else 0
        
        # 找右半邊有細胞纹理的最左點（即右側細胞群的邊界）
        right_side_cells = np.where(cell_texture_mask[y, center_x:] > 0)[0]
        right_bound = (right_side_cells.min() + center_x) if right_side_cells.size > 0 else w
        
        # 真正的傷口必然夾在左右細胞群邊界之間
        # 我們只保留這個區間內、原本被判定為低方差的區域
        final_wound_mask[y, left_bound:right_bound] = wound_mask[y, left_bound:right_bound]

    # ========================================================
    # 填補中間主傷口內部的局部孔洞 (使用連通域防呆)
    # ========================================================
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        final_wound_mask,
        connectivity=8
    )

    best_label = None
    best_score = 0

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        area = stats[i, cv2.CC_STAT_AREA]

        # 只要這個區塊有跨越或者非常接近中央，且面積最大，就是我們的主傷口
        if x < center_x < x + width or (x <= center_x and x + width >= center_x - 50):
            if area > best_score:
                best_score = area
                best_label = i

    final_mask = np.zeros_like(final_wound_mask)
    if best_label is not None:
        final_mask[labels == best_label] = 1
        
        # 填補主傷口內部的小孔洞 (FloodFill 防呆版：只填補 final_mask 內部的孔洞)
        flood = (final_mask * 255).astype(np.uint8)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        holes = (flood_inv > 0).astype(np.uint8)
        final_mask = final_mask | holes

    # ========================================================
    # 計算寬度與點位
    # ========================================================
    widths = []
    left_points = []
    right_points = []

    for y in range(h):
        row = final_mask[y]
        wound_idx = np.where(row > 0)[0]

        if len(wound_idx) < 10:  # 容忍度調低，避免漏掉過窄的列
            continue

        left_edge = wound_idx.min()
        right_edge = wound_idx.max()

        widths.append(right_edge - left_edge)
        left_points.append((left_edge, y))
        right_points.append((right_edge, y))

    widths = np.array(widths)

    if widths.size == 0:
        return {
            "mean_width_px": 0,
            "median_width_px": 0,
            "mean_width_um": 0,
            "median_width_um": 0,
            "wound_area_px2": 0
        }, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    mean_width_px = np.mean(widths)
    median_width_px = np.median(widths)

    result = {
        "mean_width_px": mean_width_px,
        "median_width_px": median_width_px,
        "mean_width_um": mean_width_px * pixel_size,
        "median_width_um": median_width_px * pixel_size,
        "wound_area_px2": np.sum(final_mask)
    }

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 綠色半透明塗滿傷口
    overlay[:,:,1] = np.maximum(
        overlay[:,:,1],
        (final_mask * 100).astype(np.uint8)
    )

    for x,y in left_points:
        cv2.circle(overlay, (x,y), 1, (0,255,0), -1)

    for x,y in right_points:
        cv2.circle(overlay, (x,y), 1, (0,0,255), -1)

    return result, overlay

# ======================================
# upload
# ======================================

uploaded_files = st.file_uploader(
    "Upload TIFF images",
    type=["tif","tiff"],
    accept_multiple_files=True
)

# ======================================
# run
# ======================================

if uploaded_files:

    results = []
    overlays = {}

    for file in uploaded_files:

        result, overlay = analyze_image(
            file,
            PIXEL_SIZE,
            LOCAL_WINDOW,
            PERCENTILE_THRESHOLD,
            KERNEL_SIZE
        )

        result["image"] = file.name

        results.append(result)

        overlays[file.name] = overlay

    df = pd.DataFrame(results)

    baseline = df.iloc[0]["median_width_px"]

    if baseline > 0:
        df["closure_percent"] = (
            (baseline - df["median_width_px"])
            / baseline
            * 100
        )
    else:
        df["closure_percent"] = 0.0

    st.subheader("Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "Download CSV",
        csv,
        file_name="results.csv",
        mime="text/csv"
    )

    st.subheader("Closure Curve")

    fig, ax = plt.subplots(
        figsize=(6,4)
    )

    ax.plot(
        df["image"],
        df["closure_percent"],
        marker="o"
    )

    ax.set_ylabel(
        "Closure (%)"
    )

    ax.set_xlabel(
        "Image"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    st.subheader(
        "Overlay Preview"
    )

    for name, overlay in overlays.items():

        st.write(name)

        st.image(
            overlay,
            use_container_width=True
        )

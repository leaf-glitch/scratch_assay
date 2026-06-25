import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

# ======================================
# 網頁頁面設定
# ======================================
st.set_page_config(page_title="細胞劃痕傷口癒合分析工具", layout="wide")
st.title("🔬 傷口癒合自動化分析系統 (Scratch Assay)")
st.write("請在上傳區上傳實驗的 `.tif` 顯微鏡影像，系統將自動計算傷口寬度與癒合率。")

# 側邊欄：使用者參數設定
st.sidebar.header("⚙️ 參數設定")
PIXEL_SIZE = st.sidebar.number_input("像素大小 (um/pixel)", value=1.3886, format="%.4f")
LOCAL_WINDOW = st.sidebar.slider("局部變異數視窗大小 (Window Size)", 11, 51, 31, step=2)
THRES_PERCENTILE = st.sidebar.slider("傷口閾值百分比 (Percentile)", 10, 90, 35)

# ======================================
# 核心影像處理函數 (不變，僅將參數彈性化)
# ======================================
def local_variance(img, size=21):
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img**2, size=size)
    var = mean_sq - mean**2
    return var

def analyze_image(uploaded_file):
    # Streamlit 上傳的是位元組串流，用 tiff.imread 直接讀取
    img = tiff.imread(uploaded_file)
    
    if len(img.shape) == 3:
        img = img[:,:,0]

    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)

    # 局部變異數 (使用側邊欄參數)
    varmap = local_variance(img2, LOCAL_WINDOW)
    varmap = cv2.normalize(varmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 傷口遮罩 (使用側邊欄參數)
    thresh = np.percentile(varmap, THRES_PERCENTILE)
    wound_mask = (varmap < thresh).astype(np.uint8)

    kernel = np.ones((9,9), np.uint8)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, kernel)
    wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel)

    # 找最大中央區域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wound_mask, connectivity=8)
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

    # 寬度測量
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
        widths.append(right_edge - left_edge)
        left_points.append((left_edge, y))
        right_points.append((right_edge, y))

    widths = np.array(widths)
    if len(widths) == 0:
        return None, None

    mean_width_px = np.mean(widths)
    median_width_px = np.median(widths)

    result = {
        "mean_width_px": mean_width_px,
        "median_width_px": median_width_px,
        "mean_width_um": mean_width_px * PIXEL_SIZE,
        "median_width_um": median_width_px * PIXEL_SIZE,
        "wound_area_px2": np.sum(final_mask)
    }

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[:,:,1] = np.maximum(overlay[:,:,1], final_mask * 255)

    for x, y in left_points:
        cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1)
    for x, y in right_points:
        cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

    return result, overlay

# ======================================
# 網頁互動邏輯
# ======================================
uploaded_files = st.file_uploader(
    "請選擇上傳多張傷口照片 (.tif)", 
    type=["tif", "tiff"], 
    accept_multiple_files=True
)

if uploaded_files:
    # 確保檔案排序正確（依檔名）
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    
    results = []
    
    st.header("📸 影像辨識預覽 (按順序顯示)")
    
    # 使用網頁網格（Grid）來並排顯示影像結果
    cols = st.columns(2) 
    
    for idx, file in enumerate(uploaded_files):
        result, overlay = analyze_image(file)
        
        if result is None:
            st.warning(f"無法從 {file.name} 中辨識出有效傷口區域。")
            continue
            
        result["image"] = file.name
        results.append(result)
        
        # 顯示在網頁上
        with cols[idx % 2]:
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                caption=f"處理結果: {file.name}", 
                use_container_width=True
            )

    if results:
        df = pd.DataFrame(results)

        # 計算癒合率 (Closure %)
        baseline = df.iloc[0]["median_width_px"]
        df["closure_percent"] = ((baseline - df["median_width_px"]) / baseline * 100)

        # 重新調整欄位順序方便人類閱讀
        df = df[["image", "closure_percent", "mean_width_um", "median_width_um", "wound_area_px2"]]

        # 數據與圖表呈現
        st.write("---")
        st.header("📊 分析數據結果")
        
        col_table, col_chart = st.columns([1, 1])
        
        with col_table:
            st.subheader("📋 數據表格")
            st.dataframe(df)
            
            # 提供 CSV 下載按鈕
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下載資料 (CSV)",
                data=csv,
                file_name='wound_healing_results.csv',
                mime='text/csv',
            )

        with col_chart:
            st.subheader("📈 傷口癒合曲線 (Closure %)")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["image"], df["closure_percent"], marker="o", color="#1f77b4", linewidth=2)
            ax.set_ylabel("Closure (%)")
            ax.set_xlabel("Image")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

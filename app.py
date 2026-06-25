import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import uniform_filter
from PIL import Image
import io

# ======================================
# PAGE
# ======================================

st.set_page_config(
    page_title="MSC Wound Healing Analyzer",
    layout="wide"
)

st.title("MSC Wound Healing Analyzer")

# ======================================
# SETTINGS
# ======================================

pixel_size = st.number_input(
    "Pixel size (μm/pixel)",
    min_value=0.0001,
    value=1.3886,
    step=0.0001,
    format="%.4f"
)

LOCAL_WINDOW = st.number_input(
    "Local variance window",
    value=21,
    step=2
)

MIN_WOUND_WIDTH = st.number_input(
    "Minimum wound width (pixel)",
    value=50
)

CENTER_SEARCH_RATIO = st.slider(
    "Center search ratio",
    0.1,
    1.0,
    0.5
)

uploaded_files = st.file_uploader(
    "Upload TIFF files",
    type=["tif", "tiff"],
    accept_multiple_files=True
)

# ======================================
# FUNCTIONS
# ======================================

def local_variance(img, size=21):

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


def analyze_image(
    uploaded_file,
    pixel_size,
    local_window,
    min_wound_width,
    center_search_ratio
):

    img = tiff.imread(uploaded_file)

    if len(img.shape) == 3:
        img = img[:, :, 0]

    img = img.astype(np.float32)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    img = img.astype(np.uint8)

    # --------------------------
    # CLAHE
    # --------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img2 = clahe.apply(img)

    # --------------------------
    # variance
    # --------------------------

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
    )

    varmap = varmap.astype(np.uint8)

    varmap = cv2.medianBlur(
        varmap,
        5
    )

    h, w = varmap.shape

    center = w // 2

    search_half = int(
        w * center_search_ratio / 2
    )

    widths = []

    left_points = []
    right_points = []

    for y in range(h):

        row = varmap[y]

        threshold = np.percentile(
            row,
            70
        )

        cell = row > threshold

        kernel = np.ones(5)

        cell = np.convolve(
            cell.astype(np.uint8),
            kernel,
            mode="same"
        )

        cell = cell >= 5

        left_region = cell[
            center-search_half:center
        ]

        right_region = cell[
            center:center+search_half
        ]

        left_idx = np.where(
            left_region
        )[0]

        right_idx = np.where(
            right_region
        )[0]

        if len(left_idx) == 0:
            continue

        if len(right_idx) == 0:
            continue

        left_edge = (
            center-search_half
            + left_idx.max()
        )

        right_edge = (
            center
            + right_idx.min()
        )

        width = right_edge - left_edge

        if width < min_wound_width:
            continue

        widths.append(width)

        left_points.append(
            (left_edge, y)
        )

        right_points.append(
            (right_edge, y)
        )

    widths = np.array(widths)

    if len(widths) == 0:
        return None, None

    mean_width_px = np.mean(widths)

    median_width_px = np.median(widths)

    wound_area_px = np.sum(widths)

    overlay = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    for x, y in left_points:

        cv2.circle(
            overlay,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    for x, y in right_points:

        cv2.circle(
            overlay,
            (x, y),
            1,
            (0, 0, 255),
            -1
        )

    result = {

        "image": uploaded_file.name,

        "mean_width_px":
            round(mean_width_px, 2),

        "median_width_px":
            round(median_width_px, 2),

        "mean_width_um":
            round(
                mean_width_px
                * pixel_size,
                2
            ),

        "median_width_um":
            round(
                median_width_px
                * pixel_size,
                2
            ),

        "wound_area_px2":
            int(wound_area_px)
    }

    return result, overlay

# ======================================
# ANALYSIS
# ======================================

if uploaded_files:

    if st.button("Analyze"):

        results = []

        overlays = {}

        progress = st.progress(0)

        for i, file in enumerate(
            uploaded_files
        ):

            result, overlay = analyze_image(
                file,
                pixel_size,
                LOCAL_WINDOW,
                MIN_WOUND_WIDTH,
                CENTER_SEARCH_RATIO
            )

            if result is not None:

                results.append(result)

                overlays[
                    file.name
                ] = overlay

            progress.progress(
                (i + 1)
                / len(uploaded_files)
            )

        df = pd.DataFrame(results)

        st.subheader("Results")

        st.dataframe(
            df,
            use_container_width=True
        )

        baseline_image = st.selectbox(
            "Select baseline image (0 hr)",
            df["image"]
        )

        baseline = df.loc[
            df["image"]
            == baseline_image,
            "median_width_px"
        ].iloc[0]

        df["closure_percent"] = (
            (baseline
             - df["median_width_px"])
            / baseline
            * 100
        ).round(2)

        st.subheader("Closure")

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
            "results.csv",
            "text/csv"
        )

        st.subheader("Overlay Preview")

        for name, overlay in overlays.items():

            st.image(
                overlay,
                caption=name,
                use_container_width=True
            )
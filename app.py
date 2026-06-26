import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from io import BytesIO

# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Scratch Assay Analyzer",
    layout="wide"
)

st.title("Scratch Assay Analyzer")

st.write(
    "Automatic wound healing assay analysis based on image segmentation."
)

# =====================================================
# USER PARAMETERS
# =====================================================

pixel_size = st.number_input(
    "Pixel size (µm/pixel)",
    min_value=0.0001,
    value=1.3886,
    format="%.4f"
)

sobel_threshold = st.slider(
    "Edge threshold percentile",
    50,
    99,
    80
)

kernel_size = st.slider(
    "Morphology kernel size",
    3,
    21,
    7,
    step=2
)

minimum_width = st.slider(
    "Minimum wound width (pixel)",
    10,
    500,
    50
)

uploaded_files = st.file_uploader(
    "Upload TIFF images",
    type=["tif", "tiff"],
    accept_multiple_files=True
)

# =====================================================
# IMAGE PREPROCESS
# =====================================================

def preprocess_image(img):

    if img.ndim == 3:
        img = img[:, :, 0]

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
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    return img

# =====================================================
# SOBEL EDGE
# =====================================================

def sobel_gradient(img):

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    grad = cv2.magnitude(
        gx,
        gy
    )

    grad = cv2.normalize(
        grad,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return grad

# =====================================================
# ANALYZE IMAGE
# =====================================================

def analyze_image(
    uploaded_file,
    pixel_size,
    threshold_percentile,
    kernel_size,
    minimum_width
):

    img = tiff.imread(uploaded_file)

    img = preprocess_image(img)

    gradient = sobel_gradient(img)

    threshold = np.percentile(
        gradient,
        threshold_percentile
    )

    edge = gradient > threshold

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    edge = cv2.morphologyEx(
        edge.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    )

    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_OPEN,
        kernel
    )

    h, w = edge.shape

    center = w // 2
        # =====================================================
    # FIND LEFT / RIGHT EDGE
    # =====================================================

    left_edge_points = []
    right_edge_points = []

    widths = []

    wound_mask = np.zeros(
        edge.shape,
        dtype=np.uint8
    )

    for y in range(h):

        row = edge[y]

        left = np.where(
            row[:center] > 0
        )[0]

        right = np.where(
            row[center:] > 0
        )[0]

        if len(left) == 0:
            continue

        if len(right) == 0:
            continue

        left_edge = left.max()

        right_edge = center + right.min()

        width = right_edge - left_edge

        if width < minimum_width:
            continue

        widths.append(width)

        left_edge_points.append(
            (left_edge, y)
        )

        right_edge_points.append(
            (right_edge, y)
        )

        wound_mask[
            y,
            left_edge:right_edge
        ] = 1

    # =====================================================
    # FILL SMALL GAPS
    # =====================================================

    wound_mask = binary_fill_holes(
        wound_mask
    ).astype(np.uint8)

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    wound_mask = cv2.morphologyEx(
        wound_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    wound_mask = cv2.morphologyEx(
        wound_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # =====================================================
    # KEEP LARGEST COMPONENT
    # =====================================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        wound_mask,
        connectivity=8
    )

    largest = 0
    largest_area = 0

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area > largest_area:

            largest_area = area
            largest = i

    final_mask = np.zeros_like(
        wound_mask
    )

    if largest > 0:

        final_mask[
            labels == largest
        ] = 1

    # =====================================================
    # RECALCULATE WIDTH
    # =====================================================

    widths = []

    left_edge_points = []

    right_edge_points = []

    for y in range(h):

        idx = np.where(
            final_mask[y] > 0
        )[0]

        if len(idx) < minimum_width:
            continue

        left = idx.min()

        right = idx.max()

        widths.append(
            right - left
        )

        left_edge_points.append(
            (left, y)
        )

        right_edge_points.append(
            (right, y)
        )

    widths = np.array(widths)

    if len(widths) == 0:

        return None, None

    mean_width_px = np.mean(
        widths
    )

    median_width_px = np.median(
        widths
    )

    wound_area = np.sum(
        final_mask
    )

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = cv2.cvtColor(
        img,
        cv2.COLOR_GRAY2BGR
    )

    green = np.zeros_like(
        overlay
    )

    green[:, :, 1] = 255

    alpha = 0.35

    mask = final_mask.astype(bool)

    overlay[mask] = cv2.addWeighted(
        overlay[mask],
        1 - alpha,
        green[mask],
        alpha,
        0
    )

    for x, y in left_edge_points:

        cv2.circle(
            overlay,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )

    for x, y in right_edge_points:

        cv2.circle(
            overlay,
            (x, y),
            1,
            (0, 0, 255),
            -1
        )

    result = {

        "mean_width_px": mean_width_px,

        "median_width_px": median_width_px,

        "mean_width_um": mean_width_px * pixel_size,

        "median_width_um": median_width_px * pixel_size,

        "wound_area_px2": wound_area

    }

    return result, overlay
# =====================================================
# RUN ANALYSIS
# =====================================================

if uploaded_files:

    results = []

    overlays = {}

    progress = st.progress(0)

    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):

        result, overlay = analyze_image(
            file,
            pixel_size,
            sobel_threshold,
            kernel_size,
            minimum_width
        )

        if result is None:
            continue

        result["image"] = file.name

        results.append(result)

        overlays[file.name] = overlay

        progress.progress((i + 1) / total)

    progress.empty()

    if len(results) == 0:

        st.error("No wound could be detected.")

        st.stop()

    df = pd.DataFrame(results)

# =====================================================
# SORT FILE NAME
# =====================================================

    df = df.sort_values(
        "image"
    ).reset_index(drop=True)

# =====================================================
# CLOSURE
# =====================================================

    baseline = df.loc[
        0,
        "median_width_px"
    ]

    df["closure_percent"] = (
        (baseline - df["median_width_px"])
        / baseline
        * 100
    )

# =====================================================
# RESULTS
# =====================================================

    st.subheader("Results")

    st.dataframe(
        df,
        use_container_width=True
    )

# =====================================================
# CSV DOWNLOAD
# =====================================================

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Results CSV",
        csv,
        file_name="results.csv",
        mime="text/csv"
    )

# =====================================================
# CLOSURE CURVE
# =====================================================

    st.subheader("Closure Curve")

    fig, ax = plt.subplots(
        figsize=(7,4)
    )

    ax.plot(
        df["image"],
        df["closure_percent"],
        "-o",
        linewidth=2
    )

    ax.set_ylabel(
        "Closure (%)"
    )

    ax.set_xlabel(
        "Image"
    )

    ax.grid(True)

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

# =====================================================
# FIGURE DOWNLOAD
# =====================================================

    fig_buffer = BytesIO()

    fig.savefig(
        fig_buffer,
        dpi=300,
        format="png"
    )

    fig_buffer.seek(0)

    st.download_button(
        "Download Closure Curve",
        fig_buffer,
        file_name="closure_curve.png",
        mime="image/png"
    )

# =====================================================
# OVERLAY
# =====================================================

    st.subheader("Overlay Preview")

    cols = st.columns(2)

    i = 0

    for name, overlay in overlays.items():

        rgb = cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )

        with cols[i % 2]:

            st.image(
                rgb,
                caption=name,
                use_container_width=True
            )

        i += 1
# =====================================================
# ANALYSIS PARAMETERS
# =====================================================

st.subheader("Analysis Parameters")

parameter_df = pd.DataFrame({

    "Parameter":[
        "Pixel Size (µm/pixel)",
        "Edge Threshold Percentile",
        "Morphology Kernel Size",
        "Minimum Wound Width (pixel)"
    ],

    "Value":[
        pixel_size,
        sobel_threshold,
        kernel_size,
        minimum_width
    ]

})

st.dataframe(
    parameter_df,
    use_container_width=True,
    hide_index=True
)

# =====================================================
# SUMMARY
# =====================================================

st.subheader("Summary")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Images",
        len(df)
    )

with col2:

    st.metric(
        "Initial Width (µm)",
        f"{df.iloc[0]['median_width_um']:.1f}"
    )

with col3:

    st.metric(
        "Final Closure (%)",
        f"{df.iloc[-1]['closure_percent']:.1f}"
    )

# =====================================================
# DOWNLOAD OVERLAY IMAGES
# =====================================================

import zipfile

zip_buffer = BytesIO()

with zipfile.ZipFile(
    zip_buffer,
    "w",
    zipfile.ZIP_DEFLATED
) as zip_file:

    for name, overlay in overlays.items():

        success, png = cv2.imencode(
            ".png",
            overlay
        )

        if success:

            zip_file.writestr(
                name.replace(".tif", "_overlay.png").replace(".tiff","_overlay.png"),
                png.tobytes()
            )

zip_buffer.seek(0)

st.download_button(

    "Download All Overlay Images",

    zip_buffer,

    file_name="overlay_images.zip",

    mime="application/zip"

)

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Scratch Assay Analyzer | Wound Healing Assay"
)

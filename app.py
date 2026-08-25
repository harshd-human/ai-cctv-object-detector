"""
Multifunctional AI/ML CCTV Camera — Object & Person Detection
Built by Harsh Dixit | Vadodara Hackathon 4.0 follow-up build

A lightweight real-time object & person detection app using OpenCV's DNN
module with a pretrained MobileNet-SSD model. Upload an image, snap a
webcam photo, or (locally) run on a live webcam feed.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="AI/ML CCTV Object Detector", page_icon="🎥", layout="centered")

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
COLORS = np.random.default_rng(42).uniform(0, 255, size=(len(CLASSES), 3))

PROTOTXT = "model/MobileNetSSD_deploy.prototxt"
WEIGHTS = "model/MobileNetSSD_deploy.caffemodel"


@st.cache_resource
def load_model():
    return cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)


def detect(image_bgr, net, conf_threshold=0.4):
    (h, w) = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    counts = {}
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx] if idx < len(CLASSES) else "object"
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        color = COLORS[idx % len(COLORS)]
        cv2.rectangle(image_bgr, (startX, startY), (endX, endY), color, 2)
        text = f"{label}: {confidence * 100:.1f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 20
        cv2.putText(image_bgr, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        counts[label] = counts.get(label, 0) + 1

    return image_bgr, counts


def main():
    st.title("🎥 AI/ML CCTV Object & Person Detector")
    st.caption(
        "Built on the object/person detection work from **Vadodara Hackathon 4.0** "
        "(Multifunctional AI/ML CCTV Camera). Runs a MobileNet-SSD model in the browser "
        "via OpenCV's DNN module — no internet round-trip to an external API."
    )

    net = load_model()
    conf_threshold = st.slider("Detection confidence threshold", 0.1, 0.9, 0.4, 0.05)

    tab1, tab2 = st.tabs(["📤 Upload an image", "📷 Use your camera"])

    image_bgr = None
    with tab1:
        uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with tab2:
        camera_img = st.camera_input("Take a photo")
        if camera_img is not None:
            pil_img = Image.open(camera_img).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if image_bgr is not None:
        with st.spinner("Running detection..."):
            output_bgr, counts = detect(image_bgr.copy(), net, conf_threshold)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        st.image(output_rgb, caption="Detection result", use_container_width=True)

        if counts:
            st.subheader("Detected objects")
            cols = st.columns(len(counts))
            for col, (label, count) in zip(cols, counts.items()):
                col.metric(label.capitalize(), count)
        else:
            st.info("No objects detected above the confidence threshold. Try lowering it.")
    else:
        st.info("Upload an image or take a photo above to run detection.")

    st.divider()
    st.caption(
        "Model: MobileNet-SSD (pretrained on Pascal VOC, 20 object classes including 'person'). "
        "Detects objects/people in a static frame — the same detection logic used in the "
        "hackathon CCTV prototype, adapted here into a deployable web demo."
    )


if __name__ == "__main__":
    main()

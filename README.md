# AI/ML CCTV Camera — Object & Person Detection

A real-time object and person detection web app, built on the detection work
from **Vadodara Hackathon 4.0** (Multifunctional AI/ML CCTV Camera prototype).

Upload an image or use your camera, and the app draws bounding boxes around
detected people and objects (20 classes: person, car, dog, chair, etc.) with
confidence scores.

## How it works

- Uses **OpenCV's DNN module** with a pretrained **MobileNet-SSD** model
  (trained on Pascal VOC).
- The image is resized to 300×300, passed through the network, and the
  output detections are filtered by a confidence threshold and drawn back
  onto the original frame.
- The UI is built with **Streamlit** — supports both file upload and live
  camera snapshot input.

This mirrors the core detection logic from the original hackathon build
(model inference → bounding boxes → labeled output), packaged here as a
deployable web demo instead of a standalone camera pipeline.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy for free (Streamlit Community Cloud) — ~10 minutes

1. Create a new **public** GitHub repo, e.g. `ai-cctv-object-detector`.
2. Upload all files in this folder to that repo, **including** the `model/`
   folder (the `.caffemodel` file is ~23MB — GitHub allows files up to
   100MB, so this is fine).
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
4. Select your repo, branch `main`, and set the main file path to `app.py`.
5. Click **Deploy**. Streamlit installs `requirements.txt` and gives you a
   live public URL like `https://your-app-name.streamlit.app`.

That URL is what you put on your resume / LinkedIn / application as the
**live project link**. The GitHub repo itself is your **project link** /
code reference.

## Files

```
app.py                                  # Streamlit app
requirements.txt                        # Python dependencies
model/MobileNetSSD_deploy.prototxt      # Model architecture
model/MobileNetSSD_deploy.caffemodel    # Pretrained weights (~23MB)
README.md                               # This file
```

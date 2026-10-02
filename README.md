# Sorting Lab: which bin does it go in?

A live waste-sorting assistant by **Omir Gebreel Abdallteif**.

Hold an item up to the camera. A custom-trained YOLO11n names the material and points to the right recycling bin
(Organic, Paper & Cardboard, Plastic & Metal, Glass). A second YOLO11n finds text. The app then takes the scene apart into
items, text and background, and shows classical edge detection, segmentation and feature extraction next to the deep model.

| Model | Test mAP50 | Test mAP50-95 |
|---|---|---|
| Waste detector, 6 classes (GARBAGE CLASSIFICATION 3, CC BY 4.0) | 64.6% | 44.9% |
| Text detector (Total-Text, BSD-3) | 74.8% | 44.6% |

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The full project (FastAPI app, training scripts, reports) lives in the parent folder; `sync_assets.py` copies the engine,
models and demo images into this folder.

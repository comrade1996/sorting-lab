"""Sorting Lab, Streamlit edition (for Streamlit Community Cloud).

Same engine (vision.py) and same two trained YOLO11n models as the FastAPI app.
Live video uses WebRTC; a single camera photo works everywhere as a fallback.
"""
import base64
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SORTING_LAB_ROOT", str(HERE))
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import av  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

from vision import BINS, CLASS_INFO, Engine, draw_label, hex_to_bgr  # noqa: E402

st.set_page_config(page_title="Sorting Lab: which bin does it go in?", page_icon="♻️", layout="wide")

AUTHOR = "Omir Gebreel Abdallteif"
SAMPLES = sorted((HERE / "samples").glob("*.jpg"))
DARK_TEXT = {"Plastic & Metal"}
MODES = {"Detected items": "detect", "Item masks": "segment", "Canny edges": "edges",
         "Sobel inside items": "sobel_objects", "Colour clusters": "kmeans", "Text": "text"}

# ----------------------------------------------------------------------------- style
st.markdown("""<style>
@font-face { font-family: "Archivo"; src: url("app/static/Archivo.ttf") format("truetype"); font-weight: 100 900; font-stretch: 62% 125%; }
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button p, .stApp [data-baseweb="tab"] p, .stApp [data-testid="stCaptionContainer"] { font-family: "Archivo", "Helvetica Neue", Arial, sans-serif !important; }
.block-container { padding-top: 3.4rem; max-width: 1360px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.sl-brand { display:flex; align-items:center; gap:12px; font-weight:800; font-stretch:125%; font-size:1.25rem; }
.sl-lids { display:grid; grid-template-columns:repeat(4, 8px); gap:3px; height:24px; }
.sl-lids i { border-radius:2px 2px 0 0; }
.sl-h1 { font-size:3rem; line-height:1.02; font-weight:820; font-stretch:125%; letter-spacing:-0.015em; max-width:14ch; margin:0.4rem 0 0.8rem; }
.sl-lede { font-size:1.15rem; line-height:1.55; max-width:62ch; color:#2b3a32; }
.sl-bins { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:1.6rem 0 0.4rem; }
.sl-bin { position:relative; min-height:220px; padding:30px 16px 14px; border-radius:10px 10px 3px 3px; color:#fff; display:flex; flex-direction:column; }
.sl-bin::before { content:""; position:absolute; left:22%; right:22%; top:12px; height:8px; border-radius:4px; background:rgba(0,0,0,.22); }
.sl-bin.dark { color:#1d2b24; }
.sl-bin b { font-size:1.6rem; line-height:1.05; font-weight:820; font-stretch:125%; }
.sl-bin p { font-size:.92rem; margin:.4rem 0 0; opacity:.92; }
.sl-bin span { margin-top:auto; font-size:.8rem; font-weight:650; }
.sl-lid { position:relative; border-radius:14px 14px 6px 6px; padding:32px 20px 18px; color:#fff; min-height:170px; }
.sl-lid::before { content:""; position:absolute; left:30%; right:30%; top:13px; height:9px; border-radius:5px; background:rgba(0,0,0,.25); }
.sl-lid.dark { color:#1d2b24; }
.sl-lid small { font-size:.95rem; opacity:.85; font-weight:600; }
.sl-lid b { display:block; font-size:2.1rem; line-height:1.02; font-weight:850; font-stretch:125%; margin:6px 0 10px; }
.sl-row { display:grid; grid-template-columns:1fr auto; padding:8px 10px; margin-bottom:6px; background:#fafbf8; border-left:6px solid var(--c); border-radius:0 6px 6px 0; }
.sl-row b { font-size:1.3rem; font-weight:800; font-stretch:125%; }
.sl-row.zero { opacity:.55; }
.sl-steps { display:grid; grid-template-columns:repeat(6,1fr); gap:14px; }
.sl-steps div { border-top:4px solid #1d2b24; padding-top:10px; font-size:.9rem; color:#4a5a51; }
.sl-steps b { display:block; color:#1d2b24; font-size:1rem; }
.sl-steps em { display:block; font-style:normal; font-size:1.7rem; font-weight:820; font-stretch:125%; color:#1d2b24; line-height:1; margin-bottom:6px; }
.sl-score b { display:block; font-size:2.4rem; line-height:1; font-weight:820; font-stretch:125%; }
.sl-score span { color:#4a5a51; font-size:.9rem; }
@media (max-width: 900px) { .sl-bins, .sl-steps { grid-template-columns:repeat(2,1fr); } .sl-h1 { font-size:2.2rem; } }
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading the two YOLO models")
def engine() -> Engine:
    return Engine()


def uri_bytes(uri: str) -> bytes:
    return base64.b64decode(uri.split(",", 1)[1])


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_upload(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def tally(counts):
    return "".join(f'<div class="sl-row{"" if counts.get(n) else " zero"}" style="--c:{b["color"]}"><span>{n}</span><b>{counts.get(n, 0)}</b></div>'
                   for n, b in BINS.items())


def decision(dets):
    top = max(dets, key=lambda d: d["conf"]) if dets else None
    if not top:
        return '<div class="sl-lid" style="background:#1d2b24"><small>Put it in</small><b style="font-size:1.4rem">Nothing found yet</b>Hold an item in front of the camera.</div>'
    b = BINS[top["bin"]]
    return (f'<div class="sl-lid{" dark" if top["bin"] in DARK_TEXT else ""}" style="background:{b["color"]}"><small>Put it in</small>'
            f'<b>{top["bin"]}</b>Looks like {top["cls"]}, {round(top["conf"] * 100)}% sure. {b["hint"]}</div>')


def det_table(dets):
    if not dets:
        st.caption("No items found. Lower the minimum confidence or move closer.")
        return
    st.dataframe(pd.DataFrame([{"Item": d["id"], "Material": d["cls"], "Sure": f"{round(d['conf'] * 100)}%", "Bin": d["bin"]} for d in dets]),
                 hide_index=True, use_container_width=True)


def banner(frame, dets):
    """Draw the bin decision onto the video frame itself (WebRTC frames can't update Streamlit widgets)."""
    if not dets:
        return frame
    top = max(dets, key=lambda d: d["conf"])
    col = hex_to_bgr(BINS[top["bin"]]["color"])
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 54), (w, h), col, -1)
    ink = (36, 43, 29) if top["bin"] in DARK_TEXT else (255, 255, 255)
    cv2.putText(frame, f"Put it in: {top['bin']}", (14, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.9, ink, 2, cv2.LINE_AA)
    return frame


# ----------------------------------------------------------------------------- header
lids = "".join(f'<i style="background:{b["color"]}"></i>' for b in BINS.values())
st.markdown(f'<div class="sl-brand"><span class="sl-lids">{lids}</span>Sorting Lab</div>', unsafe_allow_html=True)
eng = engine()
tab_over, tab_cam, tab_photo, tab_inside, tab_train = st.tabs(["Overview", "Camera", "Analyze a photo", "Inside YOLO", "Training"])

# ----------------------------------------------------------------------------- overview
with tab_over:
    st.markdown('<div class="sl-h1">Which bin does it go in?</div>'
                '<p class="sl-lede">Hold an item up to the camera. A YOLO model trained on 10,464 photos of waste names the material and points '
                'to the right bin. Then the app takes the scene apart: each object, each piece of text and the background on its own, with '
                'the edge, segmentation and feature steps shown along the way.</p>', unsafe_allow_html=True)
    classes = {}
    for c, i in CLASS_INFO.items():
        classes.setdefault(i["bin"], []).append(c)
    st.markdown('<div class="sl-bins">' + "".join(
        f'<div class="sl-bin{" dark" if n in DARK_TEXT else ""}" style="background:{b["color"]}"><b>{n}</b><p>{b["hint"]}</p>'
        f'<span>{", ".join(classes[n])}</span></div>' for n, b in BINS.items()) + "</div>", unsafe_allow_html=True)
    st.caption(f"Made by {AUTHOR}")
    st.subheader("What happens to each frame")
    steps = [("Capture", "Webcam, video or photo"), ("Prepare", "Letterbox 416 × 416, tensor [1, 3, 416, 416]"),
             ("Detect", "Waste model finds items; text model finds words"), ("Segment", "Each box seeds GrabCut; Otsu, K-Means, Watershed, SLIC"),
             ("Describe", "Colour, hue histogram, GLCM, HOG, ORB, shape"), ("Decide", "One label per item, one bin per label")]
    st.markdown('<div class="sl-steps">' + "".join(f"<div><em>{i + 1}</em><b>{t}</b>{d}</div>" for i, (t, d) in enumerate(steps)) + "</div>",
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------- camera
with tab_cam:
    c1, c2 = st.columns([3, 2])
    with c1:
        source = st.segmented_control("Camera", ["Live video", "Take a photo"], default="Live video")
    with c2:
        live_conf = st.slider("Minimum confidence", 0.1, 0.9, 0.4, 0.05, key="live_conf")
    mode_label = st.segmented_control("What to show", list(MODES), default="Detected items", key="mode")
    mode = MODES[mode_label or "Detected items"]

    if source == "Take a photo":
        shot = st.camera_input("Take a photo of an item")
        if shot:
            r = eng.live_frame(decode_upload(shot.getvalue()), mode, live_conf)
            left, right = st.columns([3, 2])
            left.image(rgb(r["frame"]), use_container_width=True)
            with right:
                st.markdown(decision(r["detections"]), unsafe_allow_html=True)
                st.markdown("#### Items in this photo")
                st.markdown(tally(r["counts"]), unsafe_allow_html=True)
                det_table(r["detections"])
    else:
        st.caption("Each frame goes to the model and comes back annotated. The bar at the bottom of the video is the bin decision. "
                   "If the video does not connect on your network, switch to Take a photo.")

        settings = st.session_state.setdefault("_live", {})  # plain dict: safe to read from the WebRTC worker thread
        settings.update(mode=mode, conf=live_conf)

        def callback(frame: av.VideoFrame) -> av.VideoFrame:
            r = eng.live_frame(frame.to_ndarray(format="bgr24"), settings["mode"], settings["conf"])
            return av.VideoFrame.from_ndarray(banner(r["frame"], r["detections"]), format="bgr24")

        webrtc_streamer(key="sorting-lab-live", mode=WebRtcMode.SENDRECV, video_frame_callback=callback,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]},
                        media_stream_constraints={"video": {"width": {"ideal": 960}}, "audio": False}, async_processing=True)

# ----------------------------------------------------------------------------- analyze a photo
with tab_photo:
    a, b = st.columns([3, 2])
    with a:
        up = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp"])
    with b:
        an_conf = st.slider("Minimum confidence", 0.1, 0.9, 0.35, 0.05, key="an_conf")
    names = [p.name for p in SAMPLES]
    pick = st.selectbox("Or try one from the test set", names, index=names.index("mixed_scene_4.jpg") if "mixed_scene_4.jpg" in names else 0,
                        format_func=lambda n: n.replace(".jpg", "").replace("_", " "))
    data = up.getvalue() if up else (HERE / "samples" / pick).read_bytes()
    st.session_state["photo"] = data

    with st.spinner("Detecting items, then building masks and features"):
        r = eng.analyze(decode_upload(data), conf=an_conf)
    I = {k: uri_bytes(v) for k, v in r["images"].items()}

    left, right = st.columns([3, 2])
    left.image(I["detected"], use_container_width=True)
    with right:
        st.markdown("#### Items per bin")
        st.markdown(tally({k: v["count"] for k, v in r["bins"].items()}), unsafe_allow_html=True)
        st.markdown("#### Detections")
        det_table(r["detections"])
        st.caption("Time spent: " + ", ".join(f"{k.replace('_ms', '').replace('_', ' ')} {round(v)} ms" for k, v in r["timings"].items()))

    st.subheader("The photo, taken apart")
    cols = st.columns(4)
    for col, (k, t, d) in zip(cols, [("objects_only", "Items only", "YOLO box refined by GrabCut"), ("text_only", "Text only", r["text_method"]),
                                     ("background", "Background only", "Items removed, Telea inpainting"), ("panoptic", "Panoptic view", "Items, background and text")]):
        col.image(I[k], use_container_width=True)
        col.markdown(f"**{t}**  \n{d}")

    st.subheader("Each item and what describes it")
    for row_start in range(0, len(r["objects"]), 3):
        cols = st.columns(3)
        for col, o in zip(cols, r["objects"][row_start:row_start + 3]):
            f = o["features"]
            with col:
                st.markdown(f'<div style="border-top:6px solid {CLASS_INFO.get(o["cls"], {}).get("color", "#e4572e")};padding-top:6px">'
                            f'<b>{o["id"]}. {o["cls"].capitalize()}</b> &nbsp; {round(o["conf"] * 100)}% sure, {o["bin"]}</div>', unsafe_allow_html=True)
                p1, p2 = st.columns(2)
                p1.image(uri_bytes(o["crop"]), caption="As detected", use_container_width=True)
                p2.image(uri_bytes(o["cutout"]), caption="Cut out", use_container_width=True)
                st.markdown('<div style="display:flex;height:18px;border-radius:3px;overflow:hidden">' + "".join(
                    f'<div style="flex:{dc["pct"]};background:{dc["hex"]}"></div>' for dc in f["dominant"]) + "</div>", unsafe_allow_html=True)
                t = f["texture"]
                st.caption(f"GLCM contrast {t['contrast']}, homogeneity {t['homogeneity']}, energy {t['energy']}, correlation {t['correlation']}. "
                           f"Edge density {f['edge_density']}, ORB points {f['orb_keypoints']}, solidity {f['shape']['solidity']}.")
                m1, m2, m3 = st.columns(3)
                m1.image(uri_bytes(f["edges"]), caption="Canny", use_container_width=True)
                m2.image(uri_bytes(f["hog"]), caption="HOG", use_container_width=True)
                m3.image(uri_bytes(f["orb"]), caption="ORB", use_container_width=True)

    st.subheader("Each piece of text")
    if r["texts"]:
        cols = st.columns(6)
        for i, t in enumerate(r["texts"][:18]):
            cols[i % 6].image(uri_bytes(t["crop"]), caption=f"Text {t['id']}" + (f", {round(t['conf'] * 100)}%" if t["conf"] else ""),
                              use_container_width=True)
    else:
        st.caption("No text in this photo.")
    c1, c2 = st.columns(2)
    c1.image(I["text"], caption="Found by the YOLO text model", use_container_width=True)
    c2.image(I["text_classical"], caption="Found by the classical method (connected components, stroke width, line grouping)", use_container_width=True)

    st.subheader("Filtering and edges")
    grid = [("gaussian", "Gaussian smoothing, 7 × 7, σ = 2"), ("sobel_x", "Sobel Gx, vertical edges"), ("sobel_y", "Sobel Gy, horizontal edges"),
            ("sobel_mag", "Gradient magnitude √(Gx² + Gy²)"), ("canny", "Canny, hysteresis 50 / 150"), ("edges_in_objects", "Edges inside items only")]
    for i in range(0, 6, 3):
        for col, (k, cap) in zip(st.columns(3), grid[i:i + 3]):
            col.image(I[k], caption=cap, use_container_width=True)

    st.subheader("Segmentation: classical methods next to YOLO")
    cl = r["classical"]
    grid = [("otsu", f"Otsu threshold, T = {cl['otsu_threshold']:.0f}"), ("kmeans", "K-Means, k = 4 in Lab colour"), ("distance", "Distance transform"),
            ("watershed", f"Watershed, {cl['watershed_regions']} regions"), ("slic", f"SLIC, {cl['slic_segments']} superpixels"),
            ("yolo_mask", f"YOLO + GrabCut, IoU with Otsu {cl['iou_otsu_vs_yolo']}")]
    for i in range(0, 6, 3):
        for col, (k, cap) in zip(st.columns(3), grid[i:i + 3]):
            col.image(I[k], caption=cap, use_container_width=True)

# ----------------------------------------------------------------------------- inside YOLO
with tab_inside:
    st.markdown("This follows the photo from **Analyze a photo** through the network: the tensor that goes in, the grids that score it, "
                "the boxes NMS throws away and what the layers learned to look for.")
    if st.button("Look inside", type="primary"):
        ins = eng.inside(decode_upload(st.session_state["photo"]))
        t, h = ins["tensor"], ins["head"]
        c1, c2, c3 = st.columns(3)
        c1.image(uri_bytes(ins["letterbox"]), caption="Input after letterboxing", use_container_width=True)
        c2.markdown(f"**Input tensor**  \nShape [{', '.join(map(str, t['shape']))}] (batch, channels, height, width)  \nType {t['dtype']}, "
                    f"device {t['device']}  \nValues {t['min']} to {t['max']}, mean {t['mean']}, std {t['std']}  \n"
                    f"Parameters {ins['params']:,}, layers {ins['layers']}")
        c2.dataframe(pd.DataFrame(t["sample"]), hide_index=True)
        c3.markdown(f"**Detection head**  \nRaw output [{', '.join(map(str, h['output_shape']))}]  \n4 box values and {h['classes']} class scores per cell  \n"
                    f"{h['predictions']:,} candidate boxes (52² + 26² + 13²)  \n{h['above_conf']} above {h['conf']}, {h['after_nms']} kept by NMS")
        st.subheader("Three grids, three object sizes")
        for col, g in zip(st.columns(3), ins["grids"]):
            col.image(uri_bytes(g["image"]), caption=f"{g['cells']} cells, stride {g['stride']}, highest score {g['max_score']}", use_container_width=True)
        st.subheader("Non-maximum suppression")
        c1, c2 = st.columns(2)
        c1.image(uri_bytes(ins["nms_before"]), caption=f"{h['above_conf']} candidate boxes before NMS", use_container_width=True)
        c2.image(uri_bytes(ins["nms_after"]), caption=f"{h['after_nms']} boxes after NMS", use_container_width=True)
        st.subheader("What the layers learned")
        for col, f in zip(st.columns(2) * 3, ins["feature_maps"]):
            col.image(uri_bytes(f["image"]), caption=f"Layer {f['layer']}, {f['module']}: {f['shape'][0]} channels of {f['shape'][1]} × {f['shape'][2]}",
                      use_container_width=True)

# ----------------------------------------------------------------------------- training
with tab_train:
    kind = st.segmented_control("Model", ["Waste model", "Text model"], default="Waste model")
    k = "text" if kind == "Text model" else "waste"
    m = json.loads((HERE / "models" / f"{k}_metrics.json").read_text())
    res = pd.read_csv(HERE / "training" / f"{k}_results.csv")
    res.columns = [c.strip() for c in res.columns]
    cols = st.columns(5)
    for col, (v, l) in zip(cols, [(len(res), "Epochs trained"), (f"{100 * m['mAP50']:.1f}%", "mAP50 on the test set"),
                                  (f"{100 * m['mAP50_95']:.1f}%", "mAP50-95 on the test set"), (f"{100 * m['precision']:.1f}%", "Precision"),
                                  (f"{100 * m['recall']:.1f}%", "Recall")]):
        col.markdown(f'<div class="sl-score"><b>{v}</b><span>{l}</span></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Validation scores after each epoch")
        chart = res.set_index("epoch")[["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]]
        chart.columns = ["mAP50", "mAP50-95", "Precision", "Recall"]
        st.line_chart(chart, color=["#1d2b24", "#4c8c2b", "#1f5fae", "#a0703c"])
    with c2:
        st.markdown("#### Results per class on the test set")
        st.dataframe(pd.DataFrame([{"Class": c, "Precision": f"{100 * v['precision']:.1f}%", "Recall": f"{100 * v['recall']:.1f}%",
                                    "mAP50": f"{100 * v['mAP50']:.1f}%", "mAP50-95": f"{100 * v['mAP50_95']:.1f}%"} for c, v in m["per_class"].items()]),
                     hide_index=True, use_container_width=True)
    plots = sorted((HERE / "training").glob(f"{k}_*.png")) + sorted((HERE / "training").glob(f"{k}_*.jpg"))
    for i in range(0, len(plots), 2):
        for col, p in zip(st.columns(2), plots[i:i + 2]):
            col.image(str(p), caption=p.stem.replace(f"{k}_", "").replace("_", " "), use_container_width=True)

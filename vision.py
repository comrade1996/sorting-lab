"""
vision.py - Core engine of the YOLO Smart Waste Sorting Lab.

Deep learning : a single custom-trained YOLO11 detector (6 waste classes).
Classical CV  : filtering, Sobel / Canny edges, Otsu, K-Means, Watershed, SLIC, GrabCut,
                morphological text-region detection, and handcrafted features
                (color, HSV histogram, GLCM texture, HOG, ORB, shape descriptors).
"""
import base64
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries, slic
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(os.environ.get("SORTING_LAB_ROOT", Path(__file__).resolve().parents[1]))

CLASS_INFO = {
    "biodegradable": {"bin": "Organic", "color": "#4c8c2b"},
    "cardboard": {"bin": "Paper & Cardboard", "color": "#a0703c"},
    "paper": {"bin": "Paper & Cardboard", "color": "#1f5fae"},
    "plastic": {"bin": "Plastic & Metal", "color": "#f2b705"},
    "metal": {"bin": "Plastic & Metal", "color": "#8a949c"},
    "glass": {"bin": "Glass", "color": "#2a9d8f"},
}
BINS = {
    "Organic": {"color": "#4c8c2b", "hint": "Food scraps and anything else that rots"},
    "Paper & Cardboard": {"color": "#1f5fae", "hint": "Paper, boxes and cartons, kept dry"},
    "Plastic & Metal": {"color": "#f2b705", "hint": "Bottles, cans and packaging, rinsed"},
    "Glass": {"color": "#2a9d8f", "hint": "Bottles and jars, lids off"},
}
TEXT_BGR = (46, 87, 228)  # signal red-orange #e4572e: text is not a bin, so it never borrows a bin colour
FEATURE_LAYERS = {0: "P1/2 Conv", 2: "P2/4 C3k2", 4: "P3/8 C3k2", 6: "P4/16 C3k2", 9: "P5/32 SPPF"}


# ----------------------------------------------------------------------------- helpers
def hex_to_bgr(h: str):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def class_color(name: str):
    return hex_to_bgr(CLASS_INFO.get(name, {"color": "#f43f5e"})["color"])


def bin_of(name: str) -> str:
    return CLASS_INFO.get(name, {"bin": "Other"})["bin"]


def encode(img: np.ndarray, fmt: str = ".jpg", quality: int = 88) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, img, params)
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf).decode()}"


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def limit_size(img: np.ndarray, max_side: int = 960) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def to_u8(x: np.ndarray) -> np.ndarray:
    x = np.abs(x.astype(np.float32))
    return (255 * x / (x.max() + 1e-6)).astype(np.uint8)


def gray_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def draw_label(img, text, x, y, color, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.5 * scale, max(1, int(scale)))
    y0 = max(y - th - base - 6, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + th + base + 6), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(img, text, (x + 4, y0 + th + 3), font, 0.5 * scale, (20, 20, 20) if lum > 140 else (255, 255, 255),
                max(1, int(scale)), cv2.LINE_AA)


def draw_detections(img, dets, show_bin=True):
    out = img.copy()
    scale = max(0.8, max(img.shape[:2]) / 640)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        c = class_color(d["cls"])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, max(2, int(2 * scale)), cv2.LINE_AA)
        label = f"{d['id']} {d['cls']} {d['conf']:.2f}" + (f", {d['bin']}" if show_bin else "")
        draw_label(out, label, x1, y1, c, scale)
    return out


# ----------------------------------------------------------------------------- classical CV
def gaussian_denoise(img, k=5, sigma=1.2):
    return cv2.GaussianBlur(img, (k, k), sigma)


def sobel_maps(gray):
    g = gaussian_denoise(gray, 3, 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return to_u8(gx), to_u8(gy), to_u8(cv2.magnitude(gx, gy))


def canny(gray, lo=50, hi=150):
    return cv2.Canny(gaussian_denoise(gray), lo, hi)


def otsu(gray):
    t, b = cv2.threshold(gaussian_denoise(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (b > 0).mean() > 0.5:  # assume objects are the minority -> make them white
        b = 255 - b
    return b, float(t)


def kmeans_labels(img, k=4, work_side=256):
    small = limit_size(img, work_side)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2]).astype(np.uint8)
    labels = cv2.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    centers_bgr = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(-1, 3)
    return labels, centers_bgr


def kmeans_image(img, k=4):
    labels, centers = kmeans_labels(img, k)
    return centers[labels]


def watershed_seg(img):
    binary, _ = otsu(gray_of(img))
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img.copy(), markers)
    rng = np.random.default_rng(7)
    palette = rng.integers(60, 255, (n + 2, 3), dtype=np.uint8)
    palette[1] = (35, 35, 35)
    vis = palette[np.clip(markers, 0, n + 1)]
    vis = cv2.addWeighted(img, 0.45, vis, 0.55, 0)
    vis[markers == -1] = (0, 0, 255)
    return vis, int(max(n - 1, 0)), to_u8(dist)


def slic_seg(img, n_segments=180):
    small = limit_size(img, 480)
    seg = slic(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), n_segments=n_segments, compactness=12, sigma=1, start_label=0)
    mean = np.zeros_like(small)
    for lbl in np.unique(seg):
        m = seg == lbl
        mean[m] = small[m].mean(axis=0)
    mean[find_boundaries(seg, mode="thick")] = (255, 255, 0)
    return cv2.resize(mean, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST), int(seg.max() + 1)


def grabcut_mask(img, box, iters=3, work_side=200):
    """Box-prompted GrabCut: turns a YOLO bounding box into a pixel mask (YOLO -> segmentation)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(0.1 * bw) + 2, int(0.1 * bh) + 2
    X1, Y1, X2, Y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    roi = img[Y1:Y2, X1:X2]
    s = min(1.0, work_side / max(roi.shape[:2]))
    roi_s = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else roi
    rh, rw = roi_s.shape[:2]
    rx1, ry1 = int((x1 - X1) * s), int((y1 - Y1) * s)
    rx2, ry2 = int((x2 - X1) * s), int((y2 - Y1) * s)
    rect = (max(1, rx1), max(1, ry1), max(2, min(rw - 2, rx2) - max(1, rx1)), max(2, min(rh - 2, ry2) - max(1, ry1)))
    mask = np.zeros((rh, rw), np.uint8)
    try:
        cv2.grabCut(roi_s, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), iters,
                    cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        m = np.zeros((rh, rw), np.uint8)
    full = np.zeros((h, w), np.uint8)
    full[Y1:Y2, X1:X2] = cv2.resize(m, (X2 - X1, Y2 - Y1), interpolation=cv2.INTER_NEAREST)
    box_area = max(bw * bh, 1)
    if (full[y1:y2, x1:x2] > 0).sum() < 0.08 * box_area:  # GrabCut failed -> ellipse prior
        full[:] = 0
        cv2.ellipse(full, ((x1 + x2) // 2, (y1 + y2) // 2), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
    full[:y1], full[y2:], full[:, :x1], full[:, x2:] = 0, 0, 0, 0
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(full)
    if n > 2:  # keep the largest component
        full = np.where(lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    return full


def classical_text_regions(img, max_regions=12):
    """Classical text localisation (baseline for the YOLO text model):
    adaptive threshold -> connected components (character candidates) -> stroke-width consistency
    (distance transform on the skeleton) -> link neighbours with similar height / stroke / intensity
    (union-find) -> keep collinear groups (PCA residual) with a uniform background."""
    gray0 = gray_of(img)
    s = min(2.2, 900 / max(gray0.shape))
    gray = cv2.resize(gray0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = gray.shape
    cands = []
    for mode in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):  # dark-on-light and light-on-dark text
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, 25, 15)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not (6 <= h <= 0.08 * max(H, W) and w <= 2.5 * h and a >= 15 and 0.12 <= a / (w * h) <= 0.92):
                continue
            m = np.pad((lbl[y:y + h, x:x + w] == i).astype(np.uint8), 1)
            sw = cv2.distanceTransform(m, cv2.DIST_L2, 3)[skeletonize(m > 0)]
            if len(sw) < 3 or sw.std() / (sw.mean() + 1e-6) > 0.45 or 2 * sw.mean() > 0.6 * h:
                continue
            cands.append([x, y, w, h, sw.mean(), float(gray[y:y + h, x:x + w][m[1:-1, 1:-1] > 0].mean())])
    if not cands:
        return []
    c = np.array(cands, np.float32)
    n = len(c)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cx, cy, hh = c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2, c[:, 3]
    for i in range(n):
        d = np.hypot(cx - cx[i], cy - cy[i])
        hr = np.maximum(hh, hh[i]) / np.minimum(hh, hh[i])
        swr = np.maximum(c[:, 4], c[i, 4]) / np.maximum(np.minimum(c[:, 4], c[i, 4]), 0.5)
        ok = (d <= 1.15 * np.maximum(hh, hh[i])) & (hr <= 1.7) & (swr <= 1.8) & (np.abs(c[:, 5] - c[i, 5]) <= 45)
        ok[: i + 1] = False
        for j in np.nonzero(ok)[0]:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) < 3:
            continue
        gc = c[g]
        x1, y1 = int(gc[:, 0].min()), int(gc[:, 1].min())
        x2, y2 = int((gc[:, 0] + gc[:, 2]).max()), int((gc[:, 1] + gc[:, 3]).max())
        hcv = gc[:, 3].std() / gc[:, 3].mean()
        density = (gc[:, 2] * gc[:, 3]).sum() / max((x2 - x1) * (y2 - y1), 1)
        patch = gray[y1:y2, x1:x2].astype(np.float32)
        char_m = np.zeros(patch.shape, bool)
        for x, y, w, h in gc[:, :4].astype(int):
            char_m[y - y1:y - y1 + h, x - x1:x - x1 + w] = True
        bgstd = float(patch[~char_m].std()) if (~char_m).sum() > 20 else 0.0
        pts = np.stack([cx[g], cy[g]], 1)
        pts = pts - pts.mean(0)
        _, evec = np.linalg.eigh(np.cov(pts.T))
        resid = float(np.abs(pts @ evec[:, 0]).mean() / gc[:, 3].mean())
        major = pts @ evec[:, 1]
        length = float((major.max() - major.min()) / gc[:, 3].mean())
        if hcv > 0.35 or bgstd > 35 or density < 0.2 or resid > 0.22 or length < 1.2:
            continue
        out.append((len(g), [int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)]))
    out.sort(key=lambda b: -b[0])
    return [b for _, b in out[:max_regions]]


# ----------------------------------------------------------------------------- features
def object_features(img, mask, box):
    x1, y1, x2, y2 = box
    crop, m = img[y1:y2, x1:x2], mask[y1:y2, x1:x2] > 0
    pix = crop[m] if m.sum() > 30 else crop.reshape(-1, 3)
    # Dominant colours (K-Means in BGR space)
    sample = pix[np.random.default_rng(0).choice(len(pix), min(len(pix), 3000), replace=False)].astype(np.float32)
    k = min(3, len(sample))
    _, lbl, cen = cv2.kmeans(sample, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2,
                             cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k)
    order = np.argsort(-counts)
    dominant = [{"hex": "#%02x%02x%02x" % tuple(int(v) for v in cen[i][::-1]), "pct": round(100 * counts[i] / counts.sum(), 1)}
                for i in order]
    # HSV hue histogram (only coloured pixels)
    hsv = cv2.cvtColor(pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    colored = hsv[hsv[:, 1] > 40]
    hist = np.bincount((colored[:, 0] // 10).astype(int), minlength=18)[:18] if len(colored) else np.zeros(18)
    hist = (hist / max(hist.sum(), 1)).round(3).tolist()
    # Texture: GLCM on a 32-level grayscale crop
    g = cv2.resize(gray_of(crop), (96, 96), interpolation=cv2.INTER_AREA) // 8
    glcm = graycomatrix(g, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    texture = {p: round(float(graycoprops(glcm, p).mean()), 3) for p in ("contrast", "homogeneity", "energy", "correlation")}
    # Edges & shape
    edges = canny(gray_of(crop))
    edge_density = float((edges > 0).mean())
    cnts, _ = cv2.findContours(mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(m.sum())
    solidity = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        solidity = float(cv2.contourArea(c) / hull) if hull > 0 else 0.0
    # HOG visualisation
    hog_in = cv2.resize(gray_of(crop), (128, 128))
    _, hog_img = hog(hog_in, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_img = cv2.applyColorMap(to_u8(np.sqrt(hog_img)), cv2.COLORMAP_INFERNO)
    # ORB keypoints (FAST corners + rotated BRIEF descriptors)
    orb = cv2.ORB_create(nfeatures=150, edgeThreshold=8, patchSize=15)
    vis = cv2.resize(crop, (200, max(1, int(200 * crop.shape[0] / max(crop.shape[1], 1)))))
    kps = orb.detect(gray_of(vis), None)
    orb_img = vis.copy()
    for kp in kps:
        cv2.circle(orb_img, tuple(int(v) for v in kp.pt), 3, (0, 255, 0), 1, cv2.LINE_AA)
    return {
        "dominant": dominant, "hue_hist": hist, "texture": texture,
        "edge_density": round(edge_density, 3), "orb_keypoints": len(kps),
        "shape": {"area_px": int(area), "aspect": round((x2 - x1) / max(y2 - y1, 1), 2),
                  "extent": round(area / max((x2 - x1) * (y2 - y1), 1), 2), "solidity": round(solidity, 2)},
        "hog": encode(hog_img), "orb": encode(orb_img), "edges": encode(edges),
    }


# ----------------------------------------------------------------------------- YOLO engine
def find_weights(kind: str = "waste"):
    for p in [ROOT / "models" / f"{kind}_best.pt", ROOT / "runs" / f"{kind}_yolo11n" / "weights" / "best.pt"]:
        if p.exists():
            return p
    return ROOT / "weights" / "yolo11n.pt" if kind == "waste" else None


class Engine:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.load()

    def load(self):
        self.weights, self.text_weights = find_weights("waste"), find_weights("text")
        self.model = YOLO(str(self.weights))
        self.text_model = YOLO(str(self.text_weights)) if self.text_weights else None
        self.names = self.model.names

    def maybe_reload(self):
        if (find_weights("waste"), find_weights("text")) != (self.weights, self.text_weights):
            with self.lock:
                self.load()

    def detect_text(self, img, conf=0.3, imgsz=640, max_det=40):
        """YOLO text model if trained, otherwise the classical baseline."""
        if self.text_model is None:
            return [{"box": b, "conf": None} for b in classical_text_regions(img)], "classical (CC + stroke width)"
        with self.lock:
            r = self.text_model.predict(img, conf=conf, iou=0.4, imgsz=imgsz, device=self.device, max_det=max_det,
                                        verbose=False)[0]
        out = [{"box": [int(round(v)) for v in xyxy], "conf": round(float(s), 3)}
               for xyxy, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        return out, "YOLO11n-text (trained on Total-Text)"

    def detect(self, img, conf=0.35, iou=0.5, imgsz=416, max_det=50):
        with self.lock:
            # class-agnostic NMS: one physical item -> one label -> one bin
            r = self.model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=self.device, max_det=max_det,
                                   agnostic_nms=True, verbose=False)[0]
        b = r.boxes
        dets = []
        for i, (xyxy, s, c) in enumerate(zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))):
            name = self.names[c]
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            dets.append({"id": i + 1, "cls": name, "conf": round(float(s), 3), "bin": bin_of(name), "box": [x1, y1, x2, y2]})
        return dets, {k: round(v, 1) for k, v in r.speed.items()}

    # ------------------------------------------------------------------ full image analysis
    def analyze(self, img, conf=0.35, iou=0.5, imgsz=416, max_objects=12):
        t0 = time.time()
        img = limit_size(img)
        h, w = img.shape[:2]
        gray = gray_of(img)
        dets, speed = self.detect(img, conf, iou, imgsz)
        timings = {"yolo_ms": round(sum(speed.values()), 1)}

        # Objects only: YOLO box -> GrabCut mask -> transparent cut-out
        t = time.time()
        objects, union = [], np.zeros((h, w), np.uint8)
        inst_map = np.zeros((h, w), np.int32)
        for d in dets[:max_objects]:
            mask = grabcut_mask(img, d["box"])
            union |= mask
            inst_map[mask > 0] = d["id"]
            x1, y1, x2, y2 = d["box"]
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[..., 3] = mask
            objects.append({**d, "crop": encode(img[y1:y2, x1:x2]), "cutout": encode(rgba[y1:y2, x1:x2], ".png"),
                            "features": object_features(img, mask, d["box"])})
        timings["segmentation_ms"] = round(1000 * (time.time() - t), 1)

        # Text only: YOLO text detector (+ classical baseline for comparison)
        t = time.time()
        text_dets, text_method = self.detect_text(img)
        timings["text_ms"] = round(1000 * (time.time() - t), 1)
        texts, scale = [], max(0.8, max(h, w) / 640)
        text_vis, text_only = (img * 0.3).astype(np.uint8), np.full_like(img, 255)
        for i, td in enumerate(text_dets):
            x1, y1, x2, y2 = td["box"]
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            crop = img[y1:y2, x1:x2]
            binar = cv2.threshold(gray_of(crop), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            if (binar > 0).mean() < 0.5:
                binar = 255 - binar
            texts.append({"id": i + 1, "box": td["box"], "conf": td["conf"], "crop": encode(crop),
                          "binary": encode(binar)})
            text_vis[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            text_only[y1:y2, x1:x2] = img[y1:y2, x1:x2]
            cv2.rectangle(text_vis, (x1, y1), (x2, y2), TEXT_BGR, 2)
            draw_label(text_vis, f"T{i + 1}" + (f" {td['conf']:.2f}" if td["conf"] else ""), x1, y1, TEXT_BGR, scale)
        classical_vis = img.copy()
        for x1, y1, x2, y2 in classical_text_regions(img):
            cv2.rectangle(classical_vis, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Background only: remove objects + inpaint (Telea)
        hole = cv2.dilate(union, np.ones((9, 9), np.uint8))
        small = limit_size(img, 480)
        small_hole = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        bg_small = cv2.inpaint(small, small_hole, 5, cv2.INPAINT_TELEA)
        bg = img.copy()
        bg[hole > 0] = cv2.resize(bg_small, (w, h))[hole > 0]

        # Panoptic view: "things" = YOLO instances, "stuff" = K-Means regions of the background
        k_labels, k_centers = kmeans_labels(img, 4)
        stuff = (0.55 * k_centers[k_labels] + 0.45 * 90).astype(np.uint8)
        panoptic = stuff.copy()
        objects_only = np.zeros_like(img)
        for d in dets[:max_objects]:
            m = inst_map == d["id"]
            col = np.array(class_color(d["cls"]), np.uint8)
            panoptic[m] = (0.35 * img[m] + 0.65 * col).astype(np.uint8)
            objects_only[m] = img[m]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panoptic, cnts, -1, (255, 255, 255), 2)
        for d in dets[:max_objects]:
            draw_label(panoptic, f"{d['cls']} #{d['id']}", d["box"][0], d["box"][1], class_color(d["cls"]),
                       max(0.8, max(h, w) / 640))

        # Edges
        t = time.time()
        gx, gy, mag = sobel_maps(gray)
        edges_canny = canny(gray)
        edges_in_objects = np.zeros_like(img)
        edges_in_objects[union > 0] = cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)[union > 0]
        timings["edges_ms"] = round(1000 * (time.time() - t), 1)

        # Classical segmentation comparison
        t = time.time()
        bin_otsu, thr = otsu(gray)
        ws_vis, ws_count, dist = watershed_seg(img)
        slic_vis, slic_count = slic_seg(img)
        timings["classical_seg_ms"] = round(1000 * (time.time() - t), 1)

        # IoU of each classical foreground vs YOLO+GrabCut foreground (agreement score)
        def iou_with(a):
            a, b = a > 0, union > 0
            return round(float((a & b).sum() / max((a | b).sum(), 1)), 3)

        bins = {name: {"count": 0, **info} for name, info in BINS.items()}
        for d in dets:
            bins.setdefault(d["bin"], {"count": 0, "color": "#f43f5e", "hint": "Unknown"})["count"] += 1
        for t_ in texts:
            x1, y1, x2, y2 = t_["box"]
            cv2.rectangle(panoptic, (x1, y1), (x2, y2), TEXT_BGR, 2)
        timings["total_ms"] = round(1000 * (time.time() - t0), 1)
        return {
            "size": [w, h], "weights": self.weights.name, "detections": dets, "bins": bins, "timings": timings,
            "text_method": text_method,
            "images": {
                "original": encode(img), "detected": encode(draw_detections(img, dets)),
                "objects_only": encode(objects_only), "background": encode(bg), "text": encode(text_vis),
                "text_only": encode(text_only), "text_classical": encode(classical_vis),
                "panoptic": encode(panoptic), "sobel_x": encode(gx), "sobel_y": encode(gy),
                "sobel_mag": encode(cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)), "canny": encode(edges_canny),
                "edges_in_objects": encode(edges_in_objects), "gaussian": encode(gaussian_denoise(img, 7, 2)),
                "otsu": encode(bin_otsu), "kmeans": encode(k_centers[k_labels]), "watershed": encode(ws_vis),
                "distance": encode(cv2.applyColorMap(dist, cv2.COLORMAP_JET)), "slic": encode(slic_vis),
                "yolo_mask": encode(union),
            },
            "objects": objects, "texts": texts,
            "classical": {"otsu_threshold": thr, "watershed_regions": ws_count, "slic_segments": slic_count,
                          "iou_otsu_vs_yolo": iou_with(bin_otsu), "kmeans_k": 4},
        }

    # ------------------------------------------------------------------ live frames
    def live(self, img, mode="detect", conf=0.4):
        r = self.live_frame(img, mode, conf)
        return {**r, "image": encode(r.pop("frame"), quality=80)}

    def live_frame(self, img, mode="detect", conf=0.4):
        """Same as live() but returns the annotated BGR frame (used by the Streamlit WebRTC stream)."""
        t0 = time.time()
        img = limit_size(img, 640)
        dets, speed = self.detect(img, conf=conf, imgsz=416)
        gray = gray_of(img)
        if mode == "edges":
            e = canny(gray)
            base = (img * 0.35).astype(np.uint8)
            base[e > 0] = (255, 255, 0)
        elif mode == "sobel_objects":
            _, _, mag = sobel_maps(gray)
            base = (img * 0.3).astype(np.uint8)
            heat = cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                base[y1:y2, x1:x2] = heat[y1:y2, x1:x2]
        elif mode == "segment":
            base = img.copy()
            for d in dets[:6]:
                m = grabcut_mask(img, d["box"], iters=1, work_side=120) > 0
                base[m] = (0.45 * base[m] + 0.55 * np.array(class_color(d["cls"]))).astype(np.uint8)
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(base, cnts, -1, (255, 255, 255), 2)
        elif mode == "kmeans":
            base = kmeans_image(img, 5)
        elif mode == "text":
            base = (img * 0.45).astype(np.uint8)
            n_text = 0
            for td in self.detect_text(img, imgsz=480)[0]:
                x1, y1, x2, y2 = td["box"]
                base[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                cv2.rectangle(base, (x1, y1), (x2, y2), TEXT_BGR, 2)
                n_text += 1
            draw_label(base, f"Text regions: {n_text}", 8, 34, TEXT_BGR, 1.2)
        else:
            base = img
        out = draw_detections(base, dets)
        counts = {name: 0 for name in BINS}
        for d in dets:
            counts[d["bin"]] = counts.get(d["bin"], 0) + 1
        return {"frame": out, "detections": dets, "counts": counts,
                "yolo_ms": round(sum(speed.values()), 1), "total_ms": round(1000 * (time.time() - t0), 1)}

    # ------------------------------------------------------------------ inside YOLO
    def inside(self, img, conf=0.35, iou=0.5, imgsz=416):
        img = limit_size(img)
        lb = LetterBox((imgsz, imgsz), auto=False)(image=img)
        rgb = np.ascontiguousarray(lb[..., ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        with self.lock:
            net = self.model.model
            net.eval()
            device = next(net.parameters()).device
            feats, hooks = {}, []
            for idx in FEATURE_LAYERS:
                hooks.append(net.model[idx].register_forward_hook(
                    lambda _m, _i, o, idx=idx: feats.__setitem__(idx, o.detach().float().cpu())))
            with torch.no_grad():
                out = net(tensor.to(device))
            for hk in hooks:
                hk.remove()
        preds = (out[0] if isinstance(out, (list, tuple)) else out)[0].float().cpu()  # [4+nc, N]
        boxes_xywh, cls_scores = preds[:4].T, preds[4:].T
        scores, cls = cls_scores.max(1)
        n_total = preds.shape[1]

        # grid heat-maps per stride (YOLO divides the image into SxS cells at 3 scales)
        grids, start = [], 0
        for s in (8, 16, 32):
            g = imgsz // s
            sm = scores[start:start + g * g].reshape(g, g).numpy()
            start += g * g
            heat = cv2.applyColorMap((255 * sm / max(sm.max(), 1e-6)).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.resize(heat, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
            vis = cv2.addWeighted(lb, 0.5, heat, 0.5, 0)
            for k in range(1, g):
                if g <= 26:
                    cv2.line(vis, (k * s, 0), (k * s, imgsz), (255, 255, 255), 1)
                    cv2.line(vis, (0, k * s), (imgsz, k * s), (255, 255, 255), 1)
            grids.append({"stride": s, "cells": f"{g}x{g}", "count": g * g, "max_score": round(float(sm.max()), 3),
                          "image": encode(vis)})

        # NMS: before vs after
        keep_mask = scores > conf
        xyxy = torch.cat([boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2, boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2], 1)
        cand, cand_s, cand_c = xyxy[keep_mask], scores[keep_mask], cls[keep_mask]
        keep = torchvision.ops.nms(cand, cand_s, iou) if len(cand) else torch.zeros(0, dtype=torch.long)
        before, after = lb.copy(), lb.copy()
        for b, c in zip(cand.numpy().astype(int), cand_c.numpy()):
            cv2.rectangle(before, tuple(b[:2]), tuple(b[2:]), class_color(self.names[int(c)]), 1)
        for i in keep.numpy():
            b, c, s = cand[i].numpy().astype(int), int(cand_c[i]), float(cand_s[i])
            cv2.rectangle(after, tuple(b[:2]), tuple(b[2:]), class_color(self.names[c]), 2)
            draw_label(after, f"{self.names[c]} {s:.2f}", int(b[0]), int(b[1]), class_color(self.names[c]), 0.8)

        # Feature maps (top-8 most active channels per layer)
        fmaps = []
        for idx, name in FEATURE_LAYERS.items():
            f = feats[idx][0]
            top = torch.argsort(f.mean((1, 2)), descending=True)[:8]
            tiles = []
            for ch in top:
                a = f[ch].numpy()
                a = (255 * (a - a.min()) / (a.max() - a.min() + 1e-6)).astype(np.uint8)
                tiles.append(cv2.resize(cv2.applyColorMap(a, cv2.COLORMAP_VIRIDIS), (104, 104), interpolation=cv2.INTER_NEAREST))
            mosaic = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])])
            fmaps.append({"layer": idx, "name": name, "module": type(net.model[idx]).__name__,
                          "shape": list(f.shape), "image": encode(mosaic)})

        return {
            "letterbox": encode(lb), "tensor": {
                "shape": list(tensor.shape), "dtype": str(tensor.dtype), "min": round(float(tensor.min()), 3),
                "max": round(float(tensor.max()), 3), "mean": round(float(tensor.mean()), 3),
                "std": round(float(tensor.std()), 3), "device": str(device),
                "sample": (tensor[0, 0, imgsz // 2:imgsz // 2 + 5, imgsz // 2:imgsz // 2 + 5] * 255).round().int().tolist()},
            "head": {"output_shape": list((out[0] if isinstance(out, (list, tuple)) else out).shape),
                     "predictions": n_total, "classes": len(self.names),
                     "above_conf": int(keep_mask.sum()), "after_nms": int(len(keep)), "conf": conf, "iou": iou},
            "grids": grids, "nms_before": encode(before), "nms_after": encode(after), "feature_maps": fmaps,
            "params": int(sum(p.numel() for p in net.parameters())), "layers": len(net.model),
        }

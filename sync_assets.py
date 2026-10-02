"""Copy the engine, trained models, demo images, fonts and training results from the main project into this
deploy folder (the GitHub repo that Streamlit Community Cloud builds from). Run from the main project:
    source env.sh && python streamlit_deploy/sync_assets.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    copy(ROOT / "app" / "vision.py", HERE / "vision.py")
    for kind in ("waste", "text"):
        copy(ROOT / "models" / f"{kind}_best.pt", HERE / "models" / f"{kind}_best.pt")
        copy(ROOT / "models" / f"{kind}_metrics.json", HERE / "models" / f"{kind}_metrics.json")
        run, test = ROOT / "runs" / f"{kind}_yolo11n", ROOT / "runs" / f"{kind}_yolo11n_test"
        copy(run / "results.csv", HERE / "training" / f"{kind}_results.csv")
        for src, name in [(run / "results.png", "1_curves.png"), (test / "confusion_matrix_normalized.png", "2_confusion_matrix.png"),
                          (test / "BoxPR_curve.png", "3_precision_recall.png"), (test / "val_batch0_pred.jpg", "4_test_predictions.jpg")]:
            if src.exists():
                copy(src, HERE / "training" / f"{kind}_{name}")
    for p in (ROOT / "app" / "static" / "samples").glob("*.jpg"):
        copy(p, HERE / "samples" / p.name)
    for name in ("Archivo.ttf", "OFL-Archivo.txt"):
        copy(ROOT / "app" / "static" / "fonts" / name, HERE / "static" / name)
    print("synced into", HERE)


if __name__ == "__main__":
    main()

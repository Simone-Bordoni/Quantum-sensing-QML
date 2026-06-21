"""Regenerate all dashboard PDFs in the results folder with the confusion matrix included."""
import os
import glob
import matplotlib.pyplot as plt
from qsopt import OptimizationCallback
from qsopt.utils.visualization import plot_optimization_dashboard

results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

npz_files = sorted(glob.glob(os.path.join(results_dir, "*.npz")))
print(f"Found {len(npz_files)} NPZ files in {results_dir}")

for npz_path in npz_files:
    base = npz_path[:-4]  # strip .npz
    pdf_path = base + "_dashboard.pdf"
    print(f"  Regenerating: {os.path.basename(pdf_path)} ...", end=" ", flush=True)
    try:
        callback = OptimizationCallback.load_callback(npz_path)
        plot_optimization_dashboard(
            optimization_callback=callback,
            save_path=pdf_path,
            show_confusion_matrix_summary=True,
        )
        plt.close("all")
        print("done")
    except Exception as exc:
        print(f"FAILED: {exc}")

print("All dashboards regenerated.")

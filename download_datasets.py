import kaggle
import os

# ============================================================
# TARGET FOLDER — change this if your path is different
# ============================================================
BASE_DEST = r"C:\dev\ai_for_engineering\dataset"

# ============================================================
# DATASETS TO DOWNLOAD
# ============================================================
DATASETS = [
    {
        "id":   "erkandevvecii/wall-crack-hole-normal-dataset",
        "name": "wall-crack-hole-normal",
    },
    {
        "id":   "programmer3/concrete-structural-defect-imaging-dataset",
        "name": "concrete-structural-defect",
    },
    {
        "id":   "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes",
        "name": "road-damage-potholes-cracks",
    },
]

# ============================================================
# DOWNLOAD + EXTRACT
# ============================================================
def download_and_extract(dataset_id, folder_name, base_dest):
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_id}")
    print(f"{'='*60}")

    dest = os.path.join(base_dest, folder_name)
    os.makedirs(dest, exist_ok=True)

    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_id, path=dest, unzip=True)
        print(f"Downloaded and extracted to: {dest}")
    except Exception as e:
        print(f"ERROR downloading {dataset_id}: {e}")


if __name__ == "__main__":
    print("Starting dataset downloads...")
    print(f"All datasets will be saved to: {BASE_DEST}\n")

    for ds in DATASETS:
        download_and_extract(ds["id"], ds["name"], BASE_DEST)

    print(f"\n{'='*60}")
    print("All downloads complete.")
    print(f"Check your datasets at: {BASE_DEST}")
    print(f"{'='*60}")

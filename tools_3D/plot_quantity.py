
import os
import csv

import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict


DATA_DIR = "/home/vivianea/projects/BrainInnov/data/npy_3D_augmented"
SPLITS = "train", "val", "test"
SIZE = (256, 256, 256)



def extract_patient_id(patient_id):
    # Extracts patient ID assuming filename format: "LIDC-IDRI-XXXX_sliceX.npy" or "LIDC-IDRI-XXXX_X_augY.npy"
    if patient_id.find('_') != -1:
        return patient_id.split('_')[0]
    else:
        return patient_id


for split in SPLITS:
    index_path = os.path.join(DATA_DIR, f"{split}_index.csv")
    summary = defaultdict(list)  # label -> [patient_ids]

    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            label_str = "cancer" if label == 1 else "non-cancer"
            patient_id = extract_patient_id(row["patient_id"])
            summary[label_str].append(patient_id)
    # Count and deduplicate
    stat_path = os.path.join(DATA_DIR, f"{split}_summary.csv")
    with open(stat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Num Samples", "Num Unique Patients", 'size'])
        for cls, patients in summary.items():
            writer.writerow([cls, len(patients), len(set(patients)), SIZE])

    print(f"[✓] Saved summary for {split} to {stat_path}")

data = []

for split in SPLITS:
    stat_path = os.path.join(DATA_DIR, f"{split}_summary.csv")
    if not os.path.exists(stat_path):
        print(f"❌ Missing: {stat_path}")
        continue

    df = pd.read_csv(stat_path)
    for _, row in df.iterrows():
        data.append({
            "Split": split,
            "Class": row["Class"],
            "Samples": row["Num Samples"],
            "Unique Patients": row["Num Unique Patients"]
        })

# Create DataFrame
summary_df = pd.DataFrame(data)

# ---- Plot: Total Samples ----
plt.figure(figsize=(10, 6))
bar_width = 0.35
splits = summary_df["Split"].unique()
classes = summary_df["Class"].unique()
x = range(len(splits))

for i, cls in enumerate(classes):
    cls_data = summary_df[summary_df["Class"] == cls].set_index("Split").reindex(splits)
    values = cls_data["Samples"].values
    bars = plt.bar(
        [pos + i * bar_width for pos in x],
        values,
        width=bar_width,
        label=cls
    )
    # Add values
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{int(height)}",
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.title("Total Augmented Samples per Class and Split")
plt.xlabel("Data Split")
plt.ylabel("Number of Samples")
plt.xticks([r + bar_width / 2 for r in x], ["Train", "Val", "Test"])
plt.legend(title="Class")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
filepath = os.path.join(DATA_DIR, f"total_augmented.png")
plt.savefig(filepath)
plt.show()

# ---- Plot: Unique Patients ----
plt.figure(figsize=(10, 6))
for i, cls in enumerate(classes):
    cls_data = summary_df[summary_df["Class"] == cls].set_index("Split").reindex(splits)
    values = cls_data["Unique Patients"].values
    bars = plt.bar(
        [pos + i * bar_width for pos in x],
        values,
        width=bar_width,
        label=cls
    )
    # Add values
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{int(height)}",
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.title("Unique Patients per Class and Split")
plt.xlabel("Data Split")
plt.ylabel("Number of Unique Patients")
plt.xticks([r + bar_width / 2 for r in x], ["Train", "Val", "Test"])
plt.legend(title="Class")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
filepath = os.path.join(DATA_DIR, f"Unique_patients.png")
plt.savefig(filepath)
plt.show()
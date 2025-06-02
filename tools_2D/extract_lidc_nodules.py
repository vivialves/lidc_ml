import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm
import cv2

# ---------------- CONFIG ----------------
CANCER_THRESH = 4.5
NONCANCER_THRESH = 2.5
PATCH_SIZE = 256
SAVE_DIR = '/home/vivianea/projects/BrainInnov/data/LIDC_ready'
# ---------------------------------------

os.makedirs(os.path.join(SAVE_DIR, 'cancer'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'non-cancer'), exist_ok=True)

def load_scan(dicom_dir):
    slices = []
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            if f.endswith('.dcm'):
                path = os.path.join(root, f)
                try:
                    dcm = pydicom.dcmread(path)
                    if hasattr(dcm, 'ImagePositionPatient'):
                        slices.append(dcm)
                except:
                    continue
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices

def get_voxel_array(slices):
    return np.stack([s.pixel_array for s in slices])

def normalize_image(img):
    img = np.clip(img, -1000, 400)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return (img * 255).astype(np.uint8)

def extract_patch(scan_array, center, size=256):
    x, y, z = center
    half = size // 2
    try:
        patch = scan_array[z, y-half:y+half, x-half:x+half]
        if patch.shape != (size, size):
            return None
        return normalize_image(patch)
    except:
        return None

def parse_all_nodules(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    nodules = []

    for session in root.findall(".//{http://www.nih.gov}readingSession"):
        for nodule in session.findall("{http://www.nih.gov}unblindedReadNodule"):
            malignancy = None
            coords = []

            for characteristic in nodule.findall("{http://www.nih.gov}characteristics"):
                mal_tag = characteristic.find("{http://www.nih.gov}malignancy")
                if mal_tag is not None and mal_tag.text and mal_tag.text.isdigit():
                    malignancy = int(mal_tag.text)

            for roi in nodule.findall("{http://www.nih.gov}roi"):
                z_pos = roi.find("{http://www.nih.gov}imageZposition")
                if z_pos is None:
                    continue
                z = float(z_pos.text)
                for edge in roi.findall("{http://www.nih.gov}edgeMap"):
                    x = int(edge.find("{http://www.nih.gov}xCoord").text)
                    y = int(edge.find("{http://www.nih.gov}yCoord").text)
                    coords.append((x, y, z))

            if coords and malignancy:
                center = np.mean(coords, axis=0)
                nodules.append({'center': center, 'malignancy': malignancy})
    return nodules

def group_nodules(nodule_list, dist_threshold=5):
    groups = []
    for nodule in nodule_list:
        added = False
        for group in groups:
            for existing in group:
                dist = np.linalg.norm(np.array(existing['center']) - np.array(nodule['center']))
                if dist < dist_threshold:
                    group.append(nodule)
                    added = True
                    break
            if added:
                break
        if not added:
            groups.append([nodule])
    return groups

# ---------------- MAIN SCRIPT ----------------

# ---------- Step 1: Build path dataframe ----------
df_path = pd.DataFrame(columns=['filename', 'dirname'])
for dirname, _, filenames in os.walk('/home/vivianea/projects/BrainInnov/data/LIDC'):
    if filenames:  # skip empty folders
        df_path.loc[len(df_path)] = {'filename': filenames, 'dirname': dirname}

# Keep only entries that contain .xml files
def contains_xml(file_list):
    return any(f.endswith('.xml') for f in file_list) if isinstance(file_list, list) else False

df_fil = df_path[df_path['filename'].apply(contains_xml)]
print(f"Found {len(df_fil)} patients with XML annotations.")

# ---------- Step 2: Loop through and process each patient ----------

for _, row in tqdm(df_fil.iterrows(), total=len(df_fil)):
    filenames = row['filename']
    dirname = row['dirname']

    # Find XML file
    xml_file = next((f for f in filenames if f.endswith('.xml')), None)
    if not xml_file:
        continue

    xml_path = os.path.join(dirname, xml_file)

    # Find DICOM directory (parent folder containing .dcm files)
    parent_dir = Path(dirname).parent
    dicom_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    dicom_dir = None
    for d in dicom_dirs:
        if any(f.endswith('.dcm') for f in os.listdir(d)):
            dicom_dir = d
            break

    if not dicom_dir:
        continue

    patient_id = Path(parent_dir).name

    # ---------- Step 3: Load scan ----------

    slices = load_scan(dicom_dir)
    if not slices:
        continue

    try:
        scan_array = get_voxel_array(slices)
    except:
        continue

    # ---------- Step 4: Parse and group nodules ----------
    nodules = parse_all_nodules(xml_path)
    groups = group_nodules(nodules)
    
    # ---------- Step 5: Extract patches ----------
    for i, group in enumerate(groups):
        avg_center = np.mean([n['center'] for n in group], axis=0).astype(int)
        avg_malignancy = np.mean([n['malignancy'] for n in group])

        label = None
        if avg_malignancy >= CANCER_THRESH:
            label = 'cancer'
        elif avg_malignancy <= NONCANCER_THRESH:
            label = 'non-cancer'
        else:
            continue  # skip uncertain

        patch = extract_patch(scan_array, avg_center, PATCH_SIZE)
        if patch is None:
            continue

        print(f"Center: {avg_center}, Patch: {patch.shape if patch is not None else 'None'}")

        filename = f"{patient_id}_group{i}.png"
        Image.fromarray(patch).save(os.path.join(SAVE_DIR, label, filename))

print("Done: Nodules grouped, labeled, and saved!")
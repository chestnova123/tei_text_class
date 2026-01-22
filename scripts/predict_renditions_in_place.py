#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
import joblib

XML_NS = "http://www.w3.org/XML/1998/namespace"

# === CONFIG: change this to your root folder with XMLs ===
INPUT_FOLDER = Path(
    r"C:\Users\elena\Documents\GitHub\semper-tei_new\druckfahnen\253\253_9"
)

# Model + class
MODEL_PATH = Path(
    r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\Q_rendition_automation\model\rendition_rf_model.pkl"
)
CLASS_NAMES_PATH = Path(
    r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\Q_rendition_automation\model\rendition_class_names.npy"
)
# Feature columns expected by the model
FEATURE_COLS = [
    "num_lb",
    "num_quote",
    "num_rs",
    "num_add",
    "num_del",
    "num_hi",
    "num_handShift",
    "num_metamark",
    "num_anchor",
    "text_length",
    "num_ptr",
    "zone_area_px2",
]


# ---------- helpers  ----------


def get_clean_text(element):
    """Extract plain text from an XML element, removing all markup."""
    text = etree.tostring(element, method="text", encoding="unicode")
    text = " ".join(text.split())  # normalize whitespace
    return text.strip()


def parse_points(points_str):
    points = []
    for pair in points_str.strip().split():
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


# ---------- prediction for a single file ----------


def predict_in_file(xml_path: Path, model, class_names) -> int:
    """
    Parse one XML file, find all <p> without @rendition,
    predict renditions, write them into the same file.
    Returns the number of <p> elements modified.
    """
    parser = etree.XMLParser(load_dtd=True, resolve_entities=True, recover=True)
    try:
        tree = etree.parse(str(xml_path), parser)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to parse {xml_path}: {e}\n")
        return 0

    root = tree.getroot()

    # Build map: zone xml:id -> zone element
    zone_by_id = {}
    zone_elements = root.xpath("//*[local-name()='zone']")
    for z in zone_elements:
        zid = z.get(f"{{{XML_NS}}}id") or z.get("xml:id") or z.get("id")
        if zid:
            zone_by_id[zid] = z

    # Find all <p> WITHOUT an existing rendition attribute
    p_elements = root.xpath("//*[local-name()='p' and not(@rendition)]")

    rows = []
    elems = []

    for p in p_elements:
        # --- feature extraction must match training script ---
        text_content = get_clean_text(p)
        num_lb = len(p.xpath(".//*[local-name()='lb']"))
        num_quote = len(p.xpath(".//*[local-name()='quote']"))
        num_rs = len(p.xpath(".//*[local-name()='rs']"))
        num_add = len(p.xpath(".//*[local-name()='add']"))
        num_del = len(p.xpath(".//*[local-name()='del']"))
        num_hi = len(p.xpath(".//*[local-name()='hi']"))
        num_handShift = len(p.xpath(".//*[local-name()='handShift']"))
        num_metamark = len(p.xpath(".//*[local-name()='metamark']"))
        num_anchor = len(p.xpath(".//*[local-name()='anchor']"))
        text_length = len(text_content)
        num_ptr = len(p.xpath(".//*[local-name()='ptr']"))

        # area information via facs -> zone
        facs = p.get("facs")
        zone_area_px2 = 0.0

        if facs:
            zid = facs.lstrip("#")
            zone = zone_by_id.get(zid)
            if zone is not None:
                pts_str = (zone.get("points") or "").strip()
                if pts_str:
                    pts = parse_points(pts_str)
                    zone_area_px2 = polygon_area(pts)

        rows.append(
            {
                "num_lb": num_lb,
                "num_quote": num_quote,
                "num_rs": num_rs,
                "num_add": num_add,
                "num_del": num_del,
                "num_hi": num_hi,
                "num_handShift": num_handShift,
                "num_metamark": num_metamark,
                "num_anchor": num_anchor,
                "text_length": text_length,
                "num_ptr": num_ptr,
                "zone_area_px2": zone_area_px2,
            }
        )
        elems.append(p)

    if not rows:
        return 0  # nothing to predict in this file

    # Build DataFrame and ensure correct column order
    X_new = pd.DataFrame(rows)
    X_new = X_new[FEATURE_COLS]

    # Predict class codes and map to names
    y_pred_codes = model.predict(X_new)
    y_pred_labels = [class_names[c] for c in y_pred_codes]

    # Write back @rendition only where it was missing
    for p, label in zip(elems, y_pred_labels):
        if p.get("rendition") is None:  # extra safety
            p.set("rendition", f"#{label}")

    # Overwrite the original file (in-place)
    tree.write(
        str(xml_path),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
    )

    return len(elems)


# ---------- main: walk folder & subfolders, in-place edit ----------


def main():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        sys.stderr.write(
            f"[ERROR] Model or class-names file not found:\n"
            f"  {MODEL_PATH}\n  {CLASS_NAMES_PATH}\n"
        )
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()

    folder = INPUT_FOLDER
    if not folder.exists() or not folder.is_dir():
        sys.stderr.write(f"[ERROR] Folder not found or not a directory: {folder}\n")
        sys.exit(1)

    xml_files = sorted(folder.rglob("*.xml"))
    if not xml_files:
        sys.stderr.write(f"[WARN] No .xml files found under {folder}\n")
        return

    total_pred = 0
    for xp in xml_files:
        n = predict_in_file(xp, model, class_names)
        total_pred += n
        if n > 0:
            print(f"✅ {xp}: added renditions to {n} <p> elements (in place)")

    print(f"\n✅ Done. Predicted renditions for {total_pred} paragraphs total.")


if __name__ == "__main__":
    main()

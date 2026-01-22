#!/usr/bin/env python3

import csv
import sys
from pathlib import Path
from lxml import etree

XML_NS = "http://www.w3.org/XML/1998/namespace"
INPUT_FOLDER = Path(
    r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\Q_rendition_automation\data_xmls"
)
OUTPUT_FILE = Path(
    r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\Q_rendition_automation\training_data.csv"
)


# get plain text out of <p> elements
def get_clean_text(element):
    """Extract plain text from an XML element, removing all markup."""
    text = etree.tostring(element, method="text", encoding="unicode")
    text = " ".join(text.split())  # normalize whitespace
    return text.strip()


# get the rendition value
def get_rendition_value(p_el):
    rend = p_el.get("rendition")
    if not rend:
        return ""
    tokens = [tok.lstrip("#") for tok in rend.split()]
    return " ".join(tokens)


# Convert coordinates into list of float tuples
def parse_points(points_str):
    points = []
    for pair in points_str.strip().split():
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    return points


# calculate area of each zone
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


def extract_from_file(xml_path):
    try:
        parser = etree.XMLParser(load_dtd=True, resolve_entities=True, recover=True)
        tree = etree.parse(str(xml_path), parser)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to parse {xml_path}: {e}\n")
        return []

    rows = []

    # look up zones
    zone_by_id = {}
    zone_elements = tree.xpath("//*[local-name()='zone']")
    for z in zone_elements:
        zid = z.get(f"{{{XML_NS}}}id") or z.get("xml:id") or z.get("id")
        if zid:
            zone_by_id[zid] = z

    # extract features
    p_elements = tree.xpath("//*[local-name()='p']")
    for p in p_elements:
        rendition = get_rendition_value(p)
        if not rendition:  # skip unlabeled paragraphs
            continue

        xml_id = p.get(f"{{{XML_NS}}}id") or p.get("xml:id") or p.get("id") or ""
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

        # area information
        facs = p.get("facs")
        zone_id = ""
        zone_points = ""
        zone_area_px2 = ""

        if facs:
            zid = facs.lstrip("#")
            zone = zone_by_id.get(zid)
            if zone is not None:
                pts_str = (zone.get("points") or "").strip()
                zone_id = zid
                zone_points = pts_str
                if pts_str:
                    pts = parse_points(pts_str)
                    zone_area_px2 = polygon_area(pts)

        rows.append(
            {
                "file_path": str(xml_path),
                "xml_id": xml_id,
                "rendition": rendition,
                "text_content": text_content,
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
                "zone_id": zone_id,
                "zone_points": zone_points,
                "zone_area_px2": zone_area_px2,
            }
        )
    return rows


def main():
    folder = INPUT_FOLDER
    if not folder.exists() or not folder.is_dir():
        sys.stderr.write(f"[ERROR] Folder not found or not a directory: {folder}\n")
        sys.exit(1)

    xml_files = sorted(folder.rglob("*.xml"))
    if not xml_files:
        sys.stderr.write(f"[WARN] No .xml files found under {folder}\n")

    all_rows = []
    for xp in xml_files:
        rows = extract_from_file(xp)
        all_rows.extend(rows)

    out_path = OUTPUT_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_path",
        "xml_id",
        "rendition",
        "text_content",
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
        "zone_id",
        "zone_points",
        "zone_area_px2",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"✅ Wrote {len(all_rows)} labeled paragraphs to {out_path}")


if __name__ == "__main__":
    main()

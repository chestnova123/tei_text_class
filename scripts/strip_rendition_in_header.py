import os
from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
NSMAP = {"tei": TEI_NS}


def remove_rendition_in_header(path):
    parser = etree.XMLParser(remove_blank_text=False)

    try:
        tree = etree.parse(path, parser)
    except etree.XMLSyntaxError as e:
        print(f"[ERR] Cannot parse {path}: {e}")
        return  # skip this file

    root = tree.getroot()

    tei_header = root.find(".//tei:teiHeader", namespaces=NSMAP)
    if tei_header is None:
        print(f"[SKIP] No teiHeader in {path}")
        return

    removed = 0
    for elem in tei_header.iter():
        if "rendition" in elem.attrib:
            del elem.attrib["rendition"]
            removed += 1

    if removed > 0:
        tree.write(
            path,
            encoding="UTF-8",
            xml_declaration=True,
            pretty_print=True,
        )
        print(f"[OK] {path}: removed {removed} @rendition attributes")
    else:
        print(f"[OK] {path}: nothing to remove")


def process_folder(base_dir):
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.lower().endswith(".xml"):
                full_path = os.path.join(dirpath, filename)
                remove_rendition_in_header(full_path)


# change this:
base_directory = r"c:\Users\elena\Documents\GitHub\semper-tei_new\druckfahnen\253"
process_folder(base_directory)



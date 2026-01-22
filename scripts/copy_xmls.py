import os
import shutil


def copy_xml_files(source_root, destination_folder):
    """
    Copies all .xml files from child directories of `source_root`
    into `destination_folder`.
    """
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    copied_count = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(".xml"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(destination_folder, file)

                # If duplicate file names exist, add a number suffix
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(
                        destination_folder, f"{base}_{counter}{ext}"
                    )
                    counter += 1

                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied: {src_path} → {dest_path}")
                    copied_count += 1
                except Exception as e:
                    print(f"Error moving {src_path}: {e}")

    print(f"\n✅ Done! {copied_count} .xml files copied to: {destination_folder}")


if __name__ == "__main__":
    source_root = r"C:\Users\elena\Documents\GitHub\semper-tei_new\druckfahnen\252"
    destination_folder = r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\Q_rendition_automation\data_xmls"

    copy_xml_files(source_root, destination_folder)

import yaml
import os

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def load_yaml_config(file_path: str):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def ocr_pdf_to_markdown(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not exists: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Only support for pdf files.")

    converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
    rendered = converter(file_path)
    text, _, images = text_from_rendered(rendered)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(os.path.dirname(file_path), f"{base_name}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Output path: {output_path}")


def ocr_folder_to_markdown(folder_path: str):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not exists: {folder_path}")

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in folder.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        try:
            ocr_pdf_to_markdown(pdf_path)
        except Exception as e:
            print(f"Failed to process {pdf_file}: {e}")

ocr_pdf_to_markdown("/mnt/PACS/Data/thinhnh/phamngocdo/AI-Agent-System-in-Telecom/data/pretrain_data/raw/books/5G-MOBILE-AND-WIRELESS-COMMUNICATIONS-TECHNOLOGY.pdf")
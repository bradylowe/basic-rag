import os

from PIL import Image

from utils import image_to_txt
from images_and_pdfs import pdf_to_txt
from microsoft_office import docx_to_txt, pptx_to_txt

from utils import calculate_output_filename_from_input_file


# Define supported file types and map to converters
SUPPORTED_TYPES = {
    '.docx': docx_to_txt,
    '.pptx': pptx_to_txt,
    '.pdf': pdf_to_txt,
}


# We can support any image type that Image supports
# Must initialize before grabbing supported file types
Image.init()
for ext in Image.EXTENSION:
    if ext not in SUPPORTED_TYPES:
        SUPPORTED_TYPES[ext] = image_to_txt


def convert_file_to_txt(file_path: str, convert_type_to_txt, output_dir: str = ''):
    """Convert a single file to .txt format using the appropriate converter."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in convert_type_to_txt:
        txt_path = calculate_output_filename_from_input_file(file_path, '.txt', output_dir)
        if os.path.exists(txt_path):
            print(f"Skipping {txt_path}. File already exists.")
            return
        
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        converter = convert_type_to_txt[ext]
        converter(file_path, txt_path)
        print(f"Converted {file_path} to {txt_path}")


def convert_files_to_txt(input_dir: str, convert_type_to_txt, output_dir: str= '', recursive: bool = False):
    """Convert all supported files in a directory to .txt format."""
    for root, _, files in os.walk(input_dir):
        for name in files:
            file_path = os.path.join(root, name)
            convert_file_to_txt(file_path, convert_type_to_txt, output_dir)
        if not recursive:
            break


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Convert all supported files in a directory to .txt format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--recursive", action="store_true", help="Recursively explore the directory structure to convert all files")
    args = parser.parse_args()

    convert_files_to_txt(args.input_dir, SUPPORTED_TYPES, args.output_dir, recursive=args.recursive)

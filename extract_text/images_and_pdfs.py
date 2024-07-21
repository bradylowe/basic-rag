import os

from pdf2image import convert_from_path

from utils import image_object_to_str, convert_file_type_to_txt


def pdf_to_images(pdf_path: str, output_folder: str = ''):
    """
    Convert PDF pages to images and save them to the specified output folder.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder where the images will be saved. Defaults to the source path.
    
    Returns:
        list: List of paths to the saved images.
    """
    if not output_folder:
        output_folder = os.path.dirname(pdf_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    
    return image_paths


def pdf_to_str(pdf_path: str, sep: str = '\n\n') -> str:
    """Convert a .pdf file to a string of text"""
    return sep.join([image_object_to_str(img) for img in convert_from_path(pdf_path)])


def pdf_to_txt(pdf_path: str, txt_path: str = ''):
    """Convert a .pdf file to a .txt file"""
    return convert_file_type_to_txt(pdf_to_str, pdf_path, txt_path)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Convert .pdf file to .txt file")
    parser.add_argument("--input", type=str, required=True, help="Path to the .pdf file")
    parser.add_argument("--output", type=str, help="Path to the output .txt file")
    args = parser.parse_args()

    pdf_to_txt(args.input, args.output)

"""
Useful functions shared among the text extractors
"""
import os

from PIL import Image
import pytesseract


def image_object_to_str(image: Image, sep: str = '') -> str:
    """Convert an image object to a string of text"""
    return sep.join(pytesseract.image_to_string(image))


def image_to_str(image_path: str) -> str:
    """Convert an image file to a string of text"""
    return image_object_to_str(Image.open(image_path))


def change_extension(src_path: str, new_ext: str) -> str:
    """Replace the file extension on a file path"""
    name = os.path.splitext(src_path)[0]
    return f"{name}.{new_ext.replace('.', '')}"


def image_to_txt(image_path: str, txt_path: str = ''):
    """Convert an image file to a .txt file"""
    return convert_file_type_to_txt(image_to_str, image_path, txt_path)


def calculate_output_filename_from_input_file(file_path: str, new_ext: str, output_dir: str = '') -> str:
    """
    Calculate the output file name based on the input file path, 
    the new extension, and an optional output_dir.
    """
    name = os.path.splitext(file_path)[0]
    new_ext = new_ext.split('.')[-1]
    if output_dir:
        name = os.path.basename(file_path)
        name = os.path.join(output_dir, name)
    return f"{name}.{new_ext}"


def convert_file_type_to_txt(file_converter, src_path: str, txt_path: str = ''):
    """Create a .txt file based on the contents of the source file and using the given converter"""
    if not txt_path:
        txt_path = change_extension(src_path, 'txt')
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(file_converter(src_path))

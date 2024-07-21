from routing import(
    SUPPORTED_TYPES,
    convert_files_to_txt,
)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Convert all supported files in a directory to .txt format")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--recursive", action="store_true", help="Recursively explore the directory structure to convert all files")
    args = parser.parse_args()

    convert_files_to_txt(args.input_dir, SUPPORTED_TYPES, args.output_dir, recursive=args.recursive)

import os
import re

def remove_comments(file_path):
    try:
        # Read the contents of the file with the specified encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()

        # Use regex to remove single-line and multi-line comments and their corresponding spaces
        file_contents = re.sub(r'\s*#.*', '', file_contents)  # Remove single-line comments
        file_contents = re.sub(r'\s*\'\'\'[\s\S]*?\'\'\'', '', file_contents)  # Remove multi-line comments
        file_contents = re.sub(r'\s*\"\"\"[\s\S]*?\"\"\"', '', file_contents)  # Remove multi-line comments

        # Write the updated contents back to the file with the same encoding
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_contents)

        print(f"Comments and spaces removed from {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def remove_comments_from_files(file_list):
    for file_path in file_list:
        if os.path.isfile(file_path):
            remove_comments(file_path)
        else:
            print(f"File not found: {file_path}")

# Example usage
file_list = ['form_mask.py']
remove_comments_from_files(file_list)

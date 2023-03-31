import os
import shutil
path='/workspaces/Covid-detection-FL/after_preprocessing'
def delete_all(directory):
    """
    Deletes all files and folders in the specified directory.

    Args:
    - directory (str): The path to the directory to be emptied.

    Returns:
    - None
    """
    # Loop over all files and subdirectories in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # If the file is a directory, delete it recursively
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            print(f"Deleted directory '{file_path}'")

        # If the file is a regular file, delete it
        else:
            os.remove(file_path)
            print(f"Deleted file '{file_path}'")

    print(f"All files and folders in '{directory}' have been deleted.")

# Example usage
delete_all(path)

import os
import shutil

def delete_files_in_directory(directory):
    """Delete all files in the specified directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"Directory not found: {directory}")


delete_files_in_directory("models/")
delete_files_in_directory("processed_data/")
print("Cleanup complete!")
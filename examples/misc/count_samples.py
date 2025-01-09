import os
import pyarrow.parquet as pq

root_folder = 'geexhp-main/parallel'

num_samples = 0

for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue  # Skip non-directory entries

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not file_name.endswith(".parquet"):
            continue  # Skip non-parquet files

        # Use PyArrow to get row count
        table = pq.ParquetFile(file_path)
        num_samples += table.metadata.num_rows

print(num_samples)

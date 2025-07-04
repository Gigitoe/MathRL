import os
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

def erase_extra_info_from_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Process each parquet file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.parquet'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Read the Parquet file
            table = pq.read_table(input_path)
            
            # Remove the 'extra_info' column if it exists
            if 'extra_info' in table.column_names:
                columns_to_keep = [name for name in table.column_names if name != 'extra_info']
                modified_table = table.select(columns_to_keep)
            else:
                modified_table = table
            
            # Write the modified table to the output folder
            pq.write_table(modified_table, output_path)
            
            print(f"Processed {filename}")

# Example usage:
erase_extra_info_from_folder('data', 'data_modified')
import json
import pandas as pd

def parse_opensignals_txt(file_path):
    """
    Parse an OpenSignals .txt file into metadata (dict) and a pandas DataFrame.
    
    Args:
        file_path (str): Path to the OpenSignals .txt file.
        
    Returns:
        metadata (dict): Header metadata extracted from JSON.
        df (pd.DataFrame): Data section as a pandas DataFrame.
    """
    header_lines = []
    data_lines = []
    in_header = True

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("# EndOfHeader"):
                    in_header = False
                elif line.startswith("#"):
                    header_lines.append(line)
            else:
                if line:  # skip blank lines
                    data_lines.append(line)

    # Extract JSON metadata from header
    metadata = {}
    for line in header_lines:
        if line.startswith("# {"):
            try:
                metadata = json.loads(line[2:])  # remove "# "
            except json.JSONDecodeError:
                pass

    # Extract column names
    device_info = list(metadata.values())[0] if metadata else {}
    columns = device_info.get("column", [])

    # Parse numeric data
    df = pd.DataFrame(
        [list(map(float, row.split())) for row in data_lines],
        columns=columns if columns else None
    )

    return metadata, df


# Example usage:
if __name__ == "__main__":
    file_path = "eog_data.txt"  # Replace with your file
    metadata, df = parse_opensignals_txt(file_path)

    print("=== Metadata ===")
    print(json.dumps(metadata, indent=2))
    print("\n=== Data (first 5 rows) ===")
    print(df.head())

import json
import csv
import os
import numpy as np

# use glob
import glob
logs_path = [f for f in glob.glob("./layer_*/database_tuning_record.json")]

def export_times_to_csv(json_path):
    """
    Extract time data from a JSON file and export it to a CSV file.
    """
    # Generate the CSV file path (in the same directory as the JSON file)
    csv_path = json_path.replace("database_tuning_record.json", "times.csv")
    
    times_data = []
    
    with open(json_path, "r", encoding="utf-8") as json_file:
        for line_num, line in enumerate(json_file.readlines(), 1):
            try:
                data = json.loads(line)
                params = data[1]
                time = params[1]  # Time array
                
                # Add the time data to the list
                times_data.append({
                    'line_number': line_num,
                    'mean_time': np.mean(time),
                    'min_time': np.min(time),
                    'max_time': np.max(time),
                    'std_time': np.std(time),
                    'times': time  # Original time array
                })
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing line {line_num} in {json_path}: {e}")
                continue
    
    # Write to the CSV file
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        if times_data:
            # Get the maximum length of the time arrays
            max_time_len = max(len(item['times']) for item in times_data)
            
            fieldnames = ['line_number', 'mean_time', 'min_time', 'max_time', 'std_time']
            # Add columns for each time measurement
            fieldnames.extend([f'time_{i}' for i in range(max_time_len)])
            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in times_data:
                row = {
                    'line_number': item['line_number'],
                    'mean_time': item['mean_time'],
                    'min_time': item['min_time'],
                    'max_time': item['max_time'],
                    'std_time': item['std_time'],
                }
                # Add each time measurement value
                for i, t in enumerate(item['times']):
                    row[f'time_{i}'] = t
                
                writer.writerow(row)
    
    return csv_path, len(times_data)


# Main program
if __name__ == "__main__":
    print("Starting to export time data to CSV files...\n")
    
    for log_path in logs_path:
        if os.path.exists(log_path):
            csv_path, row_count = export_times_to_csv(log_path)
            print(f"Processed: {log_path}")
            print(f"  Output file: {csv_path}")
            print(f"  Number of rows: {row_count}")
            print()
        else:
            print(f"File does not exist: {log_path}\n")
    
    print("Done!")

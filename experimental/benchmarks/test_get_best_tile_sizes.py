import json
import numpy as np

# use glob
import glob
logs_path = [f for f in glob.glob("./layer_*/database_tuning_record.json")]

def get_best_tile_sizes(log):
    best_time = [9999]
    best_tile_sizes = None
    
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            params = data[1]
            tile_sizes = params[0][-1]
            time = params[1]
            
            if np.mean(best_time) > np.mean(time):
                best_time = time
                best_tile_sizes = tile_sizes
    
    return best_tile_sizes

for log in logs_path:
    tile_sizes = get_best_tile_sizes(log)
    print(f"{log}:")
    print(json.dumps(tile_sizes, indent=2))
    print()
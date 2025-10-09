import json
import numpy as np

# ./layer_6/database_tuning_record.json
# ./layer_0/database_tuning_record.json
# ./layer_4/database_tuning_record.json
# ./layer_1/database_tuning_record.json
# ./layer_7/database_tuning_record.json
# ./layer_5/database_tuning_record.json
# ./layer_9/database_tuning_record.json
# ./layer_8/database_tuning_record.json
# ./layer_3/database_tuning_record.json
# ./layer_2/database_tuning_record.json`

logs_path = [
"./layer_0/database_tuning_record.json",
"./layer_1/database_tuning_record.json",
"./layer_2/database_tuning_record.json",
"./layer_3/database_tuning_record.json",
"./layer_4/database_tuning_record.json",
"./layer_5/database_tuning_record.json",
"./layer_6/database_tuning_record.json",
"./layer_7/database_tuning_record.json",
"./layer_8/database_tuning_record.json",
"./layer_9/database_tuning_record.json",
]


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

# log_file = "/workspace/tvm-dev/bennu/experimental/benchmarks/layer_6/database_tuning_record.json"

# tile_sizes = get_best_tile_sizes(log_file)
# print(json.dumps(tile_sizes, indent=2))


for log in logs_path:
    tile_sizes = get_best_tile_sizes(log)
    print(f"{log}:")
    print(json.dumps(tile_sizes, indent=2))
    print()
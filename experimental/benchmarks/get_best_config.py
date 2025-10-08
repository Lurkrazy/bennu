import json
import numpy as np

def get_best_config(log):
    best_time = [9999]
    best_tile_sizes = None
    
    with open(log, "r", encoding="utf-8") as log_file:
        for line in log_file.readlines():
            data = json.loads(line)
            params = data[1]
            tile_sizes = params[0][-1]  # 获取 tile size 配置
            time = params[1]  # 获取执行时间
            
            if np.mean(best_time) > np.mean(time):
                best_time = time
                best_tile_sizes = tile_sizes
    
    return best_tile_sizes, best_time

# 使用示例
log_file = "/workspace/tvm-dev/bennu/experimental/benchmarks/layer_6/database_tuning_record.json"
tile_sizes, execution_time = get_best_config(log_file)

print("Best execution time:", execution_time)
print("Corresponding tile sizes:")
print(json.dumps(tile_sizes, indent=2))
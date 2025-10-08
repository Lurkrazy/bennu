import json
import csv
import os
import numpy as np

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


def export_times_to_csv(json_path):
    """
    从JSON文件中提取时间数据并导出到CSV文件
    """
    # 生成CSV文件路径（与JSON文件在同一目录）
    csv_path = json_path.replace("database_tuning_record.json", "times.csv")
    
    times_data = []
    
    with open(json_path, "r", encoding="utf-8") as json_file:
        for line_num, line in enumerate(json_file.readlines(), 1):
            try:
                data = json.loads(line)
                params = data[1]
                time = params[1]  # 时间数组
                
                # 将时间数据添加到列表
                times_data.append({
                    'line_number': line_num,
                    'mean_time': np.mean(time),
                    'min_time': np.min(time),
                    'max_time': np.max(time),
                    'std_time': np.std(time),
                    'times': time  # 原始时间数组
                })
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing line {line_num} in {json_path}: {e}")
                continue
    
    # 写入CSV文件
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        if times_data:
            # 获取时间数组的最大长度
            max_time_len = max(len(item['times']) for item in times_data)
            
            fieldnames = ['line_number', 'mean_time', 'min_time', 'max_time', 'std_time']
            # 为每个时间测量添加列
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
                # 添加各个时间测量值
                for i, t in enumerate(item['times']):
                    row[f'time_{i}'] = t
                
                writer.writerow(row)
    
    return csv_path, len(times_data)


# 主程序
if __name__ == "__main__":
    print("开始导出时间数据到CSV文件...\n")
    
    for log_path in logs_path:
        if os.path.exists(log_path):
            csv_path, row_count = export_times_to_csv(log_path)
            print(f"已处理: {log_path}")
            print(f"  输出文件: {csv_path}")
            print(f"  行数: {row_count}")
            print()
        else:
            print(f"文件不存在: {log_path}\n")
    
    print("完成!")

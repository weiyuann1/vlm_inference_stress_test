#!/usr/bin/env python3
import glob
import csv
import re

def merge_results():
    # 找到所有 results_*_stats.csv 文件
    files = glob.glob("results_*_stats.csv")
    
    if not files:
        print("未找到 results_*_stats.csv 文件")
        return
    
    merged_data = []
    headers = None
    
    # 先过滤有效文件
    valid_files = []
    for file in files:
        match = re.search(r'results_(\d+)_stats\.csv', file)
        if match:
            valid_files.append((file, int(match.group(1))))
    
    # 按并发数排序
    valid_files.sort(key=lambda x: x[1])
    
    for file, concurrency in valid_files:
        
        # 读取CSV文件
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['Name'] == 'Aggregated':
                    row['Concurrency'] = concurrency
                    if headers is None:
                        headers = ['Concurrency'] + [k for k in row.keys() if k != 'Concurrency']
                    merged_data.append(row)
                    print(f"处理文件: {file}, 并发数: {concurrency}")
                    break
    
    if merged_data:
        # 保存到文件
        output_file = "merged_benchmark_results.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(merged_data)
        
        print(f"合并结果已保存到: {output_file}")
        print(f"共处理 {len(merged_data)} 个文件")
    else:
        print("未找到有效的数据")

if __name__ == "__main__":
    merge_results()
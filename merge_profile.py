import json
import gzip
import os
import glob
import re

def merge_traces(input_folder, output_file):
    all_events = []
    
    # 获取文件夹内所有的 .trace.json.gz 文件
    search_path = os.path.join(input_folder, "*.trace.json.gz")
    files = glob.glob(search_path)
    files.sort() # 排序以保证 Rank 0, 1, 2... 顺序排列
    
    print(f"找到 {len(files)} 个 Trace 文件，开始合并...")

    for filename in files:
        # 从文件名中提取 Rank ID (这里假设用 TP-x 中的 x 作为唯一标识)
        # 文件名示例: ...TP-0-DP-0-EP-0.trace.json.gz
        match = re.search(r'TP-(\d+)', filename)
        if match:
            rank_id = int(match.group(1))
        else:
            rank_id = 0 # 如果没找到，默认设为0（根据你的文件名，应该都能找到）
            
        print(f"处理文件: {os.path.basename(filename)} | 映射为 PID: {rank_id}")

        try:
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                
                # 获取该文件内的所有事件
                if isinstance(data, dict):
                    events = data.get('traceEvents', [])
                elif isinstance(data, list):
                    events = data
                else:
                    continue

                # 【关键步骤】修改 PID，让每个文件在可视化中独占一行
                # 我们把 Rank ID 赋值给 PID
                for event in events:
                    event['pid'] = f"Rank {rank_id}" 
                
                all_events.extend(events)
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")

    # 截取 00:00:00 到 00:00:27 的数据
    if all_events:
        # 收集所有事件的时间戳以确定起始时间
        timestamps = [e['ts'] for e in all_events if 'ts' in e]
        if timestamps:
            start_ts = min(timestamps)
            # Chrome Tracing 时间戳通常为微秒 (microseconds)
            # 27秒 = 27 * 1,000,000 微秒
            end_ts = start_ts + 27 * 1000000
            
            print(f"正在截取前 27 秒数据 (从 {start_ts} 到 {end_ts})...")
            original_count = len(all_events)
            
            # 过滤：保留没有 'ts' 的 metadata 事件，或 'ts' 在范围内的事件
            all_events = [e for e in all_events if 'ts' not in e or e['ts'] <= end_ts]
            
            print(f"截取完成: 事件数量从 {original_count} 减少到 {len(all_events)}")
        else:
            print("警告: 未在事件中找到时间戳，跳过时间截取。")

    # 构建最终的 JSON 结构
    final_trace = {
        "traceEvents": all_events,
        "displayTimeUnit": "ms"
    }

    # 写入合并后的文件
    print(f"正在写入 {output_file} ...")
    try:
        if output_file.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open
            
        with opener(output_file, 'wt', encoding='utf-8') as f:
            json.dump(final_trace, f)
        print("合并完成！")
    except Exception as e:
        print(f"写入文件失败: {e}")

# ================= 使用配置 =================
# 修改这里的路径为你图片中的文件夹路径
input_folder_path = "."  
output_file_name = "merged_timeline.trace.json.gz"

if __name__ == "__main__":
    # 确保文件夹存在
    if os.path.exists(input_folder_path):
        merge_traces(input_folder_path, output_file_name)
    else:
        print(f"错误: 找不到文件夹 {input_folder_path}")
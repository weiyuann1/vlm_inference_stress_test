#!/usr/bin/env python3
import os
import glob
import statistics

def analyze_video_frames():
    base_path = "./processed_videos"
    
    if not os.path.exists(base_path):
        print(f"目录不存在: {base_path}")
        return
    
    frame_counts = []
    video_count = 0
    
    # 遍历所有视频目录 (category/video_name/)
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if not os.path.isdir(category_path):
            continue
            
        for video_name in os.listdir(category_path):
            video_path = os.path.join(category_path, video_name)
            if not os.path.isdir(video_path):
                continue
                
            # 统计该视频的帧数
            frame_files = glob.glob(os.path.join(video_path, "*.jpg"))
            frame_count = len(frame_files)
            
            if frame_count > 0:
                frame_counts.append(frame_count)
                video_count += 1
                
                if video_count % 100 == 0:
                    print(f"已处理 {video_count} 个视频...")
    
    if not frame_counts:
        print("未找到任何视频帧")
        return
    
    # 计算统计信息
    total_frames = sum(frame_counts)
    avg_frames = total_frames / len(frame_counts)
    min_frames = min(frame_counts)
    max_frames = max(frame_counts)
    median_frames = statistics.median(frame_counts)
    
    print(f"\n视频帧数统计:")
    print(f"总视频数: {len(frame_counts)}")
    print(f"总帧数: {total_frames}")
    print(f"平均帧数: {avg_frames:.2f}")
    print(f"最小帧数: {min_frames}")
    print(f"最大帧数: {max_frames}")
    print(f"中位数帧数: {median_frames}")
    
    # 帧数分布
    ranges = [(1, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
    print(f"\n帧数分布:")
    for min_r, max_r in ranges:
        if max_r == float('inf'):
            count = sum(1 for x in frame_counts if x >= min_r)
            print(f"{min_r}+ 帧: {count} 个视频")
        else:
            count = sum(1 for x in frame_counts if min_r <= x <= max_r)
            print(f"{min_r}-{max_r} 帧: {count} 个视频")

if __name__ == "__main__":
    analyze_video_frames()
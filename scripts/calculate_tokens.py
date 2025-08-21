#!/usr/bin/env python3
"""
计算 processed_videos 中每个视频的平均输入token数量
"""

import os
import glob
import json
import base64
import argparse
from pathlib import Path
from PIL import Image
import requests
import statistics

def count_text_tokens(text, model_name="Qwen2.5-VL"):
    """估算文本token数量（简单估算：1个token约4个字符）"""
    return len(text) // 4 + 1

def get_image_info(image_path):
    """获取图像信息"""
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "size": os.path.getsize(image_path)
            }
    except Exception as e:
        return None

def estimate_image_tokens(width, height):
    """估算图像token数量（基于Qwen2.5-VL的压缩机制）"""
    # Qwen2.5-VL使用Vision Transformer，通常将图像分成patches
    # 估算公式：(width/patch_size) * (height/patch_size) * compression_ratio
    patch_size = 14  # 常见的patch大小
    compression_ratio = 0.3  # 经验压缩比
    
    patches_w = width // patch_size
    patches_h = height // patch_size
    base_tokens = patches_w * patches_h * compression_ratio
    
    # 添加特殊token（CLS, SEP等）
    return int(base_tokens + 10)

def calculate_video_tokens(video_dir, prompt_text):
    """计算单个视频的token数量"""
    frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    
    if not frame_files:
        return None
    
    # 文本token
    text_tokens = count_text_tokens(prompt_text)
    
    # 图像token
    image_tokens = 0
    valid_frames = 0
    
    for frame_file in frame_files:
        img_info = get_image_info(frame_file)
        if img_info:
            tokens = estimate_image_tokens(img_info["width"], img_info["height"])
            image_tokens += tokens
            valid_frames += 1
    
    if valid_frames == 0:
        return None
    
    return {
        "video_dir": video_dir,
        "num_frames": valid_frames,
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "total_tokens": text_tokens + image_tokens,
        "avg_tokens_per_frame": image_tokens / valid_frames if valid_frames > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description="计算processed_videos中每个视频的token数量")
    parser.add_argument("--video_dir", type=str, default="processed_videos",
                       help="预处理视频目录 (默认: processed_videos)")
    parser.add_argument("--prompt", type=str, 
                       default="Please describe the content of the video.",
                       help="输入提示文本")
    parser.add_argument("--sample_size", type=int, default=50,
                       help="采样视频数量 (默认: 50, 0表示全部)")
    parser.add_argument("--output", type=str, default="token_analysis.json",
                       help="输出文件名 (默认: token_analysis.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        print(f"错误: 目录 {args.video_dir} 不存在")
        return
    
    # 收集所有视频目录
    video_dirs = []
    for category in os.listdir(args.video_dir):
        category_path = os.path.join(args.video_dir, category)
        if os.path.isdir(category_path):
            for video_name in os.listdir(category_path):
                video_path = os.path.join(category_path, video_name)
                if os.path.isdir(video_path):
                    video_dirs.append(video_path)
    
    print(f"找到 {len(video_dirs)} 个视频目录")
    
    if not video_dirs:
        print("没有找到视频目录")
        return
    
    # 采样
    if args.sample_size > 0 and len(video_dirs) > args.sample_size:
        import random
        video_dirs = random.sample(video_dirs, args.sample_size)
        print(f"随机采样 {len(video_dirs)} 个视频进行分析")
    
    # 分析每个视频
    results = []
    total_tokens = []
    frame_counts = []
    
    print("开始分析...")
    for i, video_dir in enumerate(video_dirs):
        result = calculate_video_tokens(video_dir, args.prompt)
        if result:
            results.append(result)
            total_tokens.append(result["total_tokens"])
            frame_counts.append(result["num_frames"])
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(video_dirs)} 个视频")
    
    if not results:
        print("没有成功分析的视频")
        return
    
    # 统计分析
    stats = {
        "total_videos": len(results),
        "prompt_text": args.prompt,
        "text_tokens": results[0]["text_tokens"],
        "token_statistics": {
            "min_tokens": min(total_tokens),
            "max_tokens": max(total_tokens),
            "avg_tokens": statistics.mean(total_tokens),
            "median_tokens": statistics.median(total_tokens),
            "std_tokens": statistics.stdev(total_tokens) if len(total_tokens) > 1 else 0
        },
        "frame_statistics": {
            "min_frames": min(frame_counts),
            "max_frames": max(frame_counts),
            "avg_frames": statistics.mean(frame_counts),
            "median_frames": statistics.median(frame_counts)
        },
        "detailed_results": results[:10]  # 只保存前10个详细结果
    }
    
    # 输出结果
    print("\n" + "="*60)
    print("TOKEN 分析结果")
    print("="*60)
    print(f"分析视频数量: {stats['total_videos']}")
    print(f"提示文本: '{stats['prompt_text']}'")
    print(f"文本tokens: {stats['text_tokens']}")
    print("\nToken统计:")
    print(f"  最小tokens: {stats['token_statistics']['min_tokens']}")
    print(f"  最大tokens: {stats['token_statistics']['max_tokens']}")
    print(f"  平均tokens: {stats['token_statistics']['avg_tokens']:.1f}")
    print(f"  中位数tokens: {stats['token_statistics']['median_tokens']:.1f}")
    print(f"  标准差: {stats['token_statistics']['std_tokens']:.1f}")
    print("\n帧数统计:")
    print(f"  最小帧数: {stats['frame_statistics']['min_frames']}")
    print(f"  最大帧数: {stats['frame_statistics']['max_frames']}")
    print(f"  平均帧数: {stats['frame_statistics']['avg_frames']:.1f}")
    print(f"  中位数帧数: {stats['frame_statistics']['median_frames']:.1f}")
    
    # 推荐配置
    recommended_max_len = int(stats['token_statistics']['avg_tokens'] * 1.5)
    print(f"\n推荐配置:")
    print(f"  --max-model-len {recommended_max_len}")
    print("="*60)
    
    # 保存详细结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
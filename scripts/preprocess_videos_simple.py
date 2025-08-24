#!/usr/bin/env python3
"""
简化的视频预处理脚本
将 videos_directory 下的视频抽帧、resize、编码等预处理
"""

import os
import math
import argparse
import glob
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# 导入必要的库
from qwen_vl_utils import process_vision_info
from PIL import Image
from PIL.Image import Image as ImageObject


class VideoPreprocessor:
    """视频预处理器，复制LLaMA-Factory中的视频处理逻辑"""
    
    def __init__(self, 
                 video_max_pixels: int = 602112,  # 对应推理脚本中的video_max_pixels
                 video_min_pixels: int = 784,     # 16*16，最小像素数
                 video_fps: float = 2.0,          # 抽帧帧率
                 video_maxlen: int = 16):         # 对应训练脚本中的video_maxlen
        self.video_max_pixels = video_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_fps = video_fps
        self.video_maxlen = video_maxlen
    
    # 使用qwen_vl_utils处理，不需要手动实现预处理逻辑
    
    def process_video(self, video_path: str, output_dir: str, video_base_dir: str) -> dict:
        """
        使用LLaMA-Factory Qwen-VL处理逻辑处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            video_base_dir: 视频基础目录
            
        Returns:
            包含处理结果的字典
        """
        try:
            # 创建输出目录，保持原始目录结构
            video_path_obj = Path(video_path)
            video_name = video_path_obj.stem
            
            # 保持相对于 video_base_dir 的目录结构
            rel_path = video_path_obj.relative_to(Path(video_base_dir))
            video_output_dir = Path(output_dir) / rel_path.parent / video_name
            
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 使用qwen_vl_utils处理视频
            video_message = [{
                'content': [{
                    "type": "video",
                    "video": video_path,
                    "min_pixels": self.video_min_pixels,
                    "max_pixels": self.video_max_pixels,
                    "fps": self.video_fps,
                    "video_maxlen": self.video_maxlen
                }]
            }]
            
            # 使用process_vision_info处理视频
            image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
            
            if video_inputs is None:
                raise ValueError("Failed to process video with qwen_vl_utils")
            
            # 获取处理后的视频帧
            video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            selected_frames = video_input[:self.video_maxlen] if len(video_input) > self.video_maxlen else video_input
            
            # 保存处理后的帧
            frame_paths = []
            for i, frame in enumerate(selected_frames):
                img = Image.fromarray(frame)
                frame_filename = f"frame_{i:04d}.jpg"
                frame_path = video_output_dir / frame_filename
                img.save(frame_path, "JPEG", quality=95)
                frame_paths.append(str(frame_path))
                del img  # 立即释放内存
            
            # 清理大对象
            del video_input, selected_frames, video_inputs
            gc.collect()  # 强制垃圾回收
            
            return {
                "video_path": video_path,
                "video_name": video_name,
                "frame_paths": frame_paths,
                "num_frames": len(frame_paths),
                "fps_per_video": self.video_fps,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "video_path": video_path,
                "video_name": Path(video_path).stem,
                "frame_paths": [],
                "num_frames": 0,
                "fps_per_video": 0.0,
                "success": False,
                "error": str(e)
            }


def main():
    parser = argparse.ArgumentParser(description="简化的视频预处理脚本")
    parser.add_argument("--video_dir", type=str, default="group_stand",
                       help="视频目录路径 (默认: videos_directory)")
    parser.add_argument("--output_dir", type=str, default="processed_videos_512_512",
                       help="输出目录路径 (默认: processed_videos)")
    parser.add_argument("--video_max_pixels", type=int, default=262144,
                       help="视频最大像素数 (默认: 262144)")
    parser.add_argument("--video_min_pixels", type=int, default=784,
                       help="视频最小像素数 (默认: 256)")
    parser.add_argument("--video_fps", type=float, default=2.0,
                       help="视频抽帧帧率 (默认: 2.0)")
    parser.add_argument("--video_maxlen", type=int, default=16,
                       help="视频最大帧数 (默认: 16)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="并行处理的线程数 (默认: 2)")
    
    args = parser.parse_args()
    
    # 检查视频目录是否存在
    if not os.path.exists(args.video_dir):
        print(f"错误: 视频目录 {args.video_dir} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有视频文件（兼容两种目录结构）
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    
    # 递归查找所有视频文件
    for ext in video_extensions:
        pattern = os.path.join(args.video_dir, '**', ext)
        found_files = glob.glob(pattern, recursive=True)
        video_files.extend(found_files)
    
    # 检查直接在根目录下的视频文件
    for ext in video_extensions:
        pattern = os.path.join(args.video_dir, ext)
        found_files = glob.glob(pattern)
        video_files.extend(found_files)
    
    # 去重
    video_files = list(set(video_files))
    
    print(f"在 {args.video_dir} 中找到 {len(video_files)} 个视频文件")
    
    if not video_files:
        print("没有找到视频文件")
        return
    
    # 检查哪些视频已经处理过
    unprocessed_videos = []
    processed_count = 0
    
    for video_path in video_files:
        video_path_obj = Path(video_path)
        video_name = video_path_obj.stem
        rel_path = video_path_obj.relative_to(Path(args.video_dir))
        video_output_dir = Path(args.output_dir) / rel_path.parent / video_name
        
        if video_output_dir.exists() and any(video_output_dir.glob("frame_*.jpg")):
            processed_count += 1
        else:
            unprocessed_videos.append(video_path)
    
    print(f"已处理: {processed_count} 个视频")
    print(f"未处理: {len(unprocessed_videos)} 个视频")
    
    if not unprocessed_videos:
        print("所有视频都已处理完成")
        return
    
    # 创建预处理器
    preprocessor = VideoPreprocessor(
        video_max_pixels=args.video_max_pixels,
        video_min_pixels=args.video_min_pixels,
        video_fps=args.video_fps,
        video_maxlen=args.video_maxlen
    )
    
    # 并行处理未处理的视频
    results = []
    successful_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 只提交未处理的视频任务
        future_to_video = {
            executor.submit(preprocessor.process_video, video_path, args.output_dir, args.video_dir): video_path
            for video_path in unprocessed_videos
        }
        
        # 处理结果
        with tqdm(total=len(unprocessed_videos), desc="处理视频") as pbar:
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        successful_count += 1
                        pbar.set_postfix({
                            "成功": successful_count,
                            "失败": failed_count,
                            "当前": Path(video_path).name
                        })
                    else:
                        failed_count += 1
                        print(f"\n处理失败: {video_path} - {result['error']}")
                    
                    # 每处理完10个视频就清理一次内存
                    if (successful_count + failed_count) % 10 == 0:
                        gc.collect()
                        
                except Exception as e:
                    failed_count += 1
                    print(f"\n处理异常: {video_path} - {str(e)}")
                    results.append({
                        "video_path": video_path,
                        "success": False,
                        "error": str(e)
                    })
                    gc.collect()  # 异常时也清理内存
                
                pbar.update(1)
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"  - 新处理: {successful_count} 个视频")
    print(f"  - 处理失败: {failed_count} 个视频")
    print(f"  - 之前已处理: {processed_count} 个视频")
    print(f"  - 总计: {processed_count + successful_count} 个视频")
    print(f"  - 输出目录: {args.output_dir}")
    
    # 输出失败的视频列表
    if failed_count > 0:
        print(f"\n失败的视频:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['video_path']}: {result['error']}")
    



if __name__ == "__main__":
    main()
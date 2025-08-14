#!/usr/bin/env python3
"""
Debug version of locustfile_video_net.py to identify why requests are 0
"""
from locust import HttpUser, task, between
from locust.exception import StopUser
import base64
import json
import os
import glob
import time
import threading
import numpy as np
from PIL import Image
from io import BytesIO
from qwen_vl_utils import process_vision_info
import argparse
import sys

# Default values
VIDEO_BASE_PATH = "videos_directory"
_max_requests = 584
prompt_text = "Please describe the content of the video."
max_tokens = 50

def prepare_message_for_vllm(content_messages):
    """Convert video frames to individual image_url messages for vLLM compatibility"""
    # print("[DEBUG] prepare_message_for_vllm called")
    vllm_messages = []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                # print(f"[DEBUG] Processing video: {part_message.get('video', 'unknown')}")
                video_message = [{'content': [part_message]}]
                try:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                    assert video_inputs is not None, "video_inputs should not be None"
                    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    # Limit frames to match server configuration (image=8)
                    max_frames = 8
                    selected_frames = video_input[:max_frames] if len(video_input) > max_frames else video_input
                    print(f"[DEBUG] Limited to {len(selected_frames)} frames (from {len(video_input)} total frames)")
                    
                    # Convert each frame to individual image_url messages
                    for i, frame in enumerate(selected_frames):
                        img = Image.fromarray(frame)
                        output_buffer = BytesIO()
                        img.save(output_buffer, format="jpeg")
                        byte_data = output_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        
                        # Add each frame as a separate image_url
                        new_content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                        })
                        # print(f"[DEBUG] Added frame {i+1} as image_url")
                except Exception as e:
                    print(f"[ERROR] Failed to process video: {e}")
                    return [], {}
            else:
                new_content_list.append(part_message)
                
        message["content"] = new_content_list
        vllm_messages.append(message)
    
    # print(f"[DEBUG] Returning {len(vllm_messages)} messages")
    return vllm_messages, {}

class VLLMUser(HttpUser):
    wait_time = between(0, 0)
    
    # Class variables to track total requests across all users
    _request_count = 0
    _request_lock = threading.Lock()
    _max_requests = _max_requests
    _stop_sending = False
    
    # Shared preloaded data across all users
    _preloaded_payloads = []
    _preload_lock = threading.Lock()
    _preload_done = False
    
    # Global video index to ensure each request uses a different video
    _global_video_index = 0
    _video_index_lock = threading.Lock()
    
    # Real request timing (excluding preload time)
    _first_request_time = None
    _last_request_time = None
    _timing_lock = threading.Lock()

    @classmethod
    def _preload_videos(cls):
        """Preload videos once for all users"""
        with cls._preload_lock:
            if cls._preload_done:
                print("[DEBUG] Videos already preloaded")
                return
            
            print("[INFO] Starting video preloading (shared across all users)...")
            start_time = time.time()
            
            # Load video files recursively from subdirectories
            if os.path.exists(VIDEO_BASE_PATH):
                video_files = []
                for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                    # Search recursively in subdirectories
                    found_files = glob.glob(os.path.join(VIDEO_BASE_PATH, '**', ext), recursive=True)
                    video_files.extend(found_files)
                    print(f"[DEBUG] Found {len(found_files)} {ext} files")
            else:
                video_files = []
                print(f"[ERROR] Video directory does not exist: {VIDEO_BASE_PATH}")
            
            print(f"[DEBUG] Total video files found: {len(video_files)}")
            
            if not video_files:
                print(f"[ERROR] No video files found in {VIDEO_BASE_PATH}")
                cls._preload_done = True
                return
            
            # Preload and encode video messages (limit to avoid long startup time)
            max_preload = len(video_files) # Limit to 50 videos for faster startup
            # max_preload=10 
            print(f"[DEBUG] Will preload {max_preload} videos out of {len(video_files)} total")
            
            successful_loads = 0
            for i, video_file in enumerate(video_files[:max_preload]):
                print(f"[DEBUG] Processing video {i+1}/{max_preload}: {video_file}")
                try:
                    messages = cls._prepare_video_message_static(video_file)
                    if messages:
                        # print(f"[DEBUG] Prepared messages for {video_file}")
                        processed_messages, video_kwargs = prepare_message_for_vllm(messages)
                        if processed_messages:
                            payload = {
                                "model": "Qwen2.5-VL",
                                "messages": processed_messages,
                                "max_tokens": max_tokens,
                                "temperature": 0.2
                            }
                            cls._preloaded_payloads.append(payload)
                            successful_loads += 1
                            # print(f"[DEBUG] Successfully preloaded {video_file}")
                        else:
                            print(f"[ERROR] No processed messages for {video_file}")
                    else:
                        print(f"[ERROR] No messages prepared for {video_file}")
                except Exception as e:
                    print(f"[ERROR] Failed to preload video {video_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            load_time = time.time() - start_time
            print(f"[INFO] Preloaded {len(cls._preloaded_payloads)} video payloads in {load_time:.2f}s")
            print(f"[DEBUG] Success rate: {successful_loads}/{max_preload}")
            cls._preload_done = True

    @staticmethod
    def _prepare_video_message_static(video_file):
        """Static method to prepare video message for processing"""
        try:
            # print(f"[DEBUG] Preparing message for {video_file}")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "video",
                            "video": video_file,
                            "total_pixels": 20480 * 28 * 28,
                            "min_pixels": 16 * 28 * 2,
                            "fps": 1.0,
                            "video_maxlen": 16
                        }
                    ]
                }
            ]
            # print(f"[DEBUG] Messages prepared for {video_file}") 
            return messages
        except Exception as e:
            print(f"[ERROR] Failed to prepare video message for {video_file}: {e}")
            return None

    def on_start(self):
        user_id = getattr(self, 'user_id', 'unknown')
        print(f"[INFO] User {user_id} starting...")
        
        # Ensure videos are preloaded (only happens once)
        self._preload_videos()
        
        # Check if we have any preloaded payloads
        if not self._preloaded_payloads:
            print(f"[ERROR] User {user_id} has no preloaded payloads - stopping")
            raise StopUser()
        
        # Initialize per-user index for sequential selection
        self.current_index = 0
        
        print(f"[INFO] User {user_id} ready with {len(self._preloaded_payloads)} preloaded payloads")

    @task
    def send_chat_completion(self):
        print(f"[DEBUG] send_chat_completion called")
        
        # Check if we should stop sending requests and increment counter atomically
        should_send = False
        current_count = 0
        is_final_request = False
        
        with VLLMUser._request_lock:
            if VLLMUser._stop_sending:
                print(f"[INFO] User stopping - already reached {VLLMUser._request_count} requests")
                raise StopUser()
                
            if VLLMUser._request_count < VLLMUser._max_requests:
                # Increment request count and allow this request
                VLLMUser._request_count += 1
                current_count = VLLMUser._request_count
                should_send = True
                
                # Check if this is the last request
                if current_count >= VLLMUser._max_requests:
                    VLLMUser._stop_sending = True
                    is_final_request = True
                    print(f"[INFO] Request #{current_count}: This is the final request")
            else:
                print(f"[INFO] Already reached maximum requests ({VLLMUser._max_requests}), stopping user")
                raise StopUser()
        
        # Only send request if we got permission
        if not should_send:
            print(f"[DEBUG] Request not allowed, stopping user")
            raise StopUser()
            
        if not self._preloaded_payloads:
            print(f"[WARNING] No preloaded payloads available for request #{current_count}")
            raise StopUser()
        
        # Get unique video index for this request
        video_index = 0
        with VLLMUser._video_index_lock:
            video_index = VLLMUser._global_video_index % len(self._preloaded_payloads)
            VLLMUser._global_video_index += 1
        
        # Use the unique video index to select payload
        payload = self._preloaded_payloads[video_index]
        # print(f"[INFO] Request #{current_count} using video index {video_index}")
        
        headers = {"Content-Type": "application/json"}

        # Record timing for real req/s calculation
        request_start_time = time.time()
        
        # Send the request
        try:
            print(f"[DEBUG] Sending request #{current_count}")
            response = self.client.post(
                "/v1/chat/completions",
                data=json.dumps(payload),
                headers=headers,
                name="vllm_video_completion"
            )
            
            # Record request completion time
            request_end_time = time.time()
            
            # Update timing statistics
            with VLLMUser._timing_lock:
                if VLLMUser._first_request_time is None:
                    VLLMUser._first_request_time = request_start_time
                    print(f"[INFO] First request started at {request_start_time}")
                VLLMUser._last_request_time = request_end_time
            
            # Print error details for debugging
            if response.status_code != 200:
                print(f"[ERROR] Request #{current_count} failed with status {response.status_code}")
                print(f"[ERROR] Response: {response.text}")
            else:
                print(f"[SUCCESS] Request #{current_count} completed in {request_end_time - request_start_time:.2f}s")
                    
        except Exception as e:
            print(f"[ERROR] Request #{current_count} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Stop this user after completing the request
        if is_final_request:
            print(f"[INFO] Final request completed. Total requests sent: {current_count}")
            # Calculate and display real req/s
            VLLMUser._display_real_stats()
        
        # Always stop the user if we've reached the limit
        if VLLMUser._stop_sending:
            raise StopUser()

    @classmethod
    def _display_real_stats(cls):
        """Display real request statistics excluding preload time"""
        with cls._timing_lock:
            if cls._first_request_time is not None and cls._last_request_time is not None:
                real_test_duration = cls._last_request_time - cls._first_request_time
                total_requests = cls._request_count
                
                print("\n" + "="*60)
                print("🚀 REAL REQUEST PERFORMANCE STATISTICS")
                print("="*60)
                print(f"📊 Total requests sent: {total_requests}")
                print(f"⏱️  First request started at: {time.strftime('%H:%M:%S', time.localtime(cls._first_request_time))}")
                print(f"⏱️  Last request completed at: {time.strftime('%H:%M:%S', time.localtime(cls._last_request_time))}")
                print(f"⏰ Real testing duration: {real_test_duration:.2f} seconds")
                
                if real_test_duration > 0:
                    real_req_per_sec = total_requests / real_test_duration
                    print(f"🎯 REAL REQ/S (excluding preload): {real_req_per_sec:.2f}")
                else:
                    print(f"🎯 REAL REQ/S: Unable to calculate (duration too short)")
                
                print("="*60)
                print("📝 Note: This excludes video preloading time and shows")
                print("   actual request processing performance only.")
                print("="*60 + "\n")
            else:
                print("\n[WARNING] Unable to calculate real req/s - no timing data available\n")
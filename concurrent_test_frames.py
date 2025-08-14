from locust import HttpUser, task, between
from locust.exception import StopUser
import base64
import json
import os
import glob
import random
import time
import threading
import argparse
import sys


# Default values
VIDEO_BASE_PATH = "./testset"
_max_requests = 1024
prompt_text = "Please describe the content of the video."
max_tokens = 50

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

    @classmethod
    def _preload_videos(cls):
        """Preload videos once for all users"""
        with cls._preload_lock:
            if cls._preload_done:
                return
            
            print("[INFO] Starting video preloading (shared across all users)...")
            start_time = time.time()
            total_files = 0
            total_bytes = 0
            
            # Load video directories
            if os.path.exists(VIDEO_BASE_PATH):
                video_dirs = [d for d in os.listdir(VIDEO_BASE_PATH) 
                             if os.path.isdir(os.path.join(VIDEO_BASE_PATH, d))]
            else:
                video_dirs = []
            
            if not video_dirs:
                print(f"Error: No video directories found in {VIDEO_BASE_PATH}")
                cls._preload_done = True
                return
            
            # Preload and encode all video frames
            max_preload = len(video_dirs)  # Load all available videos
            # max_preload=10
            for i, video_dir in enumerate(video_dirs[:max_preload]):
                try:
                    content, file_count, byte_count = cls._load_and_encode_video_frames_static(video_dir)
                    total_files += file_count
                    total_bytes += byte_count
                    
                    if len(content) > 1:  # Has images
                        payload = {
                            "model": "Qwen2.5-VL",
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": max_tokens,
                            "temperature": 0.2
                        }
                        cls._preloaded_payloads.append(payload)
                        
                    # Progress indicator every 100 videos
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"[INFO] Processed {i+1}/{max_preload} videos in {elapsed:.2f}s")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to preload video {video_dir}: {e}")
                    continue
            
            load_time = time.time() - start_time
            avg_speed = total_bytes / load_time / (1024 * 1024)  # MB/s
            print(f"[INFO] Preloaded {len(cls._preloaded_payloads)} video payloads in {load_time:.2f}s")
            print(f"[INFO] Processed {total_files} files, {total_bytes/(1024*1024):.1f}MB at {avg_speed:.1f}MB/s")
            cls._preload_done = True

    @staticmethod
    def _load_and_encode_video_frames_static(video_dir):
        """Static method to load and encode up to 8 frames from a video directory"""
        video_path = os.path.join(VIDEO_BASE_PATH, video_dir)
        frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        
        file_count = 0
        total_bytes = 0
        
        if not frame_files:
            return [{"type": "text", "text": prompt_text}], 0, 0
        
        # Load up to 8 frames
        content = [{"type": "text", "text": prompt_text}]
        
        for frame_file in frame_files[:8]:
            try:
                with open(frame_file, "rb") as f:
                    image_bytes = f.read()
                    file_count += 1
                    total_bytes += len(image_bytes)
                    image_base64 = base64.b64encode(image_bytes).decode()

                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })
            except Exception as e:
                print(f"[ERROR] Failed to load frame {frame_file}: {e}")
                continue
        
        return content, file_count, total_bytes

    def on_start(self):
        user_id = getattr(self, 'user_id', 'unknown')
        print(f"[INFO] User {user_id} starting...")
        
        # Ensure videos are preloaded (only happens once)
        self._preload_videos()
        
        # Initialize per-user index for sequential selection
        self.current_index = 0
        
        print(f"[INFO] User {user_id} ready with {len(self._preloaded_payloads)} preloaded payloads")



    @task
    def send_chat_completion(self):
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
        print(f"[INFO] Request #{current_count} using video index {video_index}")
        
        headers = {"Content-Type": "application/json"}

        # Send the request
        try:
            response = self.client.post(
                "/v1/chat/completions",
                data=json.dumps(payload),
                headers=headers,
                name="vllm_video_completion"
            )
            print(f"[INFO] Request #{current_count} completed with status: {response.status_code}")
            
            # # Debug: Print model inference results
            # if response.status_code == 200:
            #     try:
            #         response_data = response.json()
            #         print(f"[DEBUG] Request #{current_count} - Full Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
                    
            #         # Extract and print the model's response content
            #         if 'choices' in response_data and len(response_data['choices']) > 0:
            #             model_response = response_data['choices'][0].get('message', {}).get('content', '')
            #             print(f"[DEBUG] Request #{current_count} - Model Response: {model_response}")
            #         else:
            #             print(f"[DEBUG] Request #{current_count} - No choices found in response")
                        
            #     except json.JSONDecodeError as json_err:
            #         print(f"[ERROR] Request #{current_count} - Failed to parse JSON response: {json_err}")
            #         print(f"[DEBUG] Request #{current_count} - Raw response text: {response.text}")
            # else:
            #     print(f"[ERROR] Request #{current_count} - HTTP Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] Request #{current_count} failed: {e}")
        
        # Stop this user after completing the request
        if is_final_request:
            print(f"[INFO] Final request completed. Total requests sent: {current_count}")
        
        # Always stop the user if we've reached the limit
        if VLLMUser._stop_sending:
            raise StopUser()

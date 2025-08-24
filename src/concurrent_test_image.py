from locust import HttpUser, task, between
from locust.exception import StopUser
import base64
import json
import os
import glob
import time
import threading
from PIL import Image
import io

IMAGE_BASE_PATH = "./cc_ocr_data"
prompt_text = "what is the text in the image?"
max_tokens = 64
_max_requests = 500

# Preload images at module level (before any test starts)
print("[INFO] Starting image preloading at module initialization...")
preload_start_time = time.time()

# Load all image files recursively
image_files = glob.glob(os.path.join(IMAGE_BASE_PATH, "**", "*.jpg"), recursive=True) + \
             glob.glob(os.path.join(IMAGE_BASE_PATH, "**", "*.png"), recursive=True) + \
             glob.glob(os.path.join(IMAGE_BASE_PATH, "**", "*.jpeg"), recursive=True)

# Limit to maximum 500 images
if len(image_files) > 500:
    image_files = image_files[:500]
    print(f"[DEBUG] Limited to 500 images out of total available")

print(f"[DEBUG] Will process {len(image_files)} image files")

_preloaded_payloads = []
if image_files:
    total_files = 0
    total_bytes = 0
    
    for i, image_file in enumerate(image_files):
        try:
            # Load and resize image to 1024x1024
            with Image.open(image_file) as img:
                img = img.convert('RGB')
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                # Convert to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=95)
                image_bytes = img_buffer.getvalue()
                
                total_files += 1
                total_bytes += len(image_bytes)
                image_base64 = base64.b64encode(image_bytes).decode()
            
            content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
            
            payload = {
                "model": "Qwen2.5-VL",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.2
            }
            _preloaded_payloads.append(payload)
            
            # Progress indicator every 100 images
            if (i + 1) % 100 == 0:
                elapsed = time.time() - preload_start_time
                print(f"[INFO] Processed {i+1}/{len(image_files)} images in {elapsed:.2f}s")
                
        except Exception as e:
            print(f"[ERROR] Failed to preload image {image_file}: {e}")
            continue
    
    load_time = time.time() - preload_start_time
    avg_speed = total_bytes / load_time / (1024 * 1024)  # MB/s
    print(f"[INFO] Preloaded {len(_preloaded_payloads)} image payloads in {load_time:.2f}s")
    print(f"[INFO] Processed {total_files} files, {total_bytes/(1024*1024):.1f}MB at {avg_speed:.1f}MB/s")
else:
    print(f"[ERROR] No image files found in {IMAGE_BASE_PATH}")

print("[INFO] Image preloading completed. Test can start without including loading time.")

class VLLMUser(HttpUser):
    wait_time = between(0, 0)
    
    # Class variables for request limiting
    _request_count = 0
    _request_lock = threading.Lock()
    _max_requests = _max_requests
    _stop_sending = False
    
    # Global image index to ensure each request uses a different image
    _global_image_index = 0
    _image_index_lock = threading.Lock()
    
    # Active users tracking
    _active_users = set()
    _users_lock = threading.Lock()

    def on_start(self):
        user_id = id(self)
        self.user_id = user_id
        
        with VLLMUser._users_lock:
            VLLMUser._active_users.add(user_id)
            print(f"[INFO] User {user_id} starting... Active users: {len(VLLMUser._active_users)}")
        
        print(f"[INFO] User {user_id} ready with {len(_preloaded_payloads)} preloaded payloads")
    
    def on_stop(self):
        user_id = getattr(self, 'user_id', 'unknown')
        
        with VLLMUser._users_lock:
            VLLMUser._active_users.discard(user_id)
            remaining_users = len(VLLMUser._active_users)
            print(f"[INFO] User {user_id} stopped. Remaining active users: {remaining_users}")
            
            if remaining_users == 0:
                print(f"[INFO] *** ALL USERS STOPPED *** Total requests: {VLLMUser._request_count}")

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
            
        if not _preloaded_payloads:
            print(f"[WARNING] No preloaded payloads available for request #{current_count}")
            raise StopUser()
        
        # Get unique image index for this request
        image_index = 0
        with VLLMUser._image_index_lock:
            image_index = VLLMUser._global_image_index % len(_preloaded_payloads)
            VLLMUser._global_image_index += 1
        
        # Use the unique image index to select payload
        payload = _preloaded_payloads[image_index]
        print(f"[INFO] Request #{current_count} using image index {image_index}")
        
        headers = {"Content-Type": "application/json"}

        # Send the request
        request_start_time = time.time()
        try:
            response = self.client.post(
                "/v1/chat/completions",
                data=json.dumps(payload),
                headers=headers,
                name="vllm_single_image_completion",
                timeout=300  # 5 minutes timeout
            )
            request_duration = time.time() - request_start_time
            
            if response.status_code == 400:
                print(f"[ERROR] Request #{current_count} failed with 400. Response: {response.text}")
            else:
                print(f"[INFO] Request #{current_count} completed in {request_duration:.2f}s with status: {response.status_code}")
                
        except Exception as e:
            request_duration = time.time() - request_start_time
            print(f"[ERROR] Request #{current_count} failed after {request_duration:.2f}s: {e}")
        
        # Stop this user after completing the request
        if is_final_request:
            print(f"[INFO] Final request completed. Total requests sent: {current_count}")
            print(f"[INFO] All requests sent, stopping user {getattr(self, 'user_id', 'unknown')}")
        
        # Always stop the user if we've reached the limit
        if VLLMUser._stop_sending:
            with VLLMUser._users_lock:
                active_count = len(VLLMUser._active_users)
            print(f"[INFO] User {getattr(self, 'user_id', 'unknown')} stopping. Active users: {active_count}")
            raise StopUser()

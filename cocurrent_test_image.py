from locust import HttpUser, task, between
import base64
import json


IMAGE_PATH = "test-640.jpg"

class VLLMUser(HttpUser):
    wait_time = between(0.5, 1)

    def on_start(self):
        with open(IMAGE_PATH, "rb") as f:
            self.image_base64 = base64.b64encode(f.read()).decode("utf-8")

    @task
    def send_chat_completion(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self.image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "<OD>"
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "skip_special_tokens": False
        }

        headers = {"Content-Type": "application/json"}

        self.client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            headers=headers,
            name="vllm_chat_completion"
        )

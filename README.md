# vlm_inference_stress_test
VLM model inference stress test using locust

# 使用Docker启动vLLM，并加载模型, 进行压测
参数示例, 适用于G6e.2xlarge
模型服务启动:
# Florence-2
docker run --gpus all --cpus=4 --rm -it \
  -p 8080:8080 \
  vllm/vllm-openai:v0.9.1 \
  --model microsoft/Florence-2-base-ft \
  --tokenizer facebook/bart-large \
  --port 8080 \
  --dtype float16 \
  --model-impl vllm \
  --gpu-memory-utilization 0.95 \
  --served-model-name florence-2 \
  --trust-remote-code \
  --tokenizer-pool-size 16 \
  --max-model-len 1024 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 64 \
  --swap-space 8 \
  --tensor-parallel-size 1 \
  --disable-log-stats \
  --disable-log-requests \
  --enable-prefix-caching 
  <!-- --enable-chunked-prefill -->

# Qwen/Qwen2.5-VL-3B-Instruct
docker run --gpus all --cpus=4 --rm -it \
  -p 8080:8080 \
  vllm/vllm-openai:v0.9.1 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --limit-mm-per-prompt.image 8 \
  --port 8080 \
  --dtype bfloat16 \
  --model-impl vllm \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen2.5-VL \
  --trust-remote-code \
  --tokenizer-pool-size 16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 16 \
  --swap-space 8 \
  --tensor-parallel-size 1 \
  --disable-log-stats \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill

# Qwen/Qwen2.5-VL-7B-Instruct
docker run --gpus all --cpus=4 --rm -it \
  -e CUDA_VISIBLE_DEVICES=7 \
  -p 8080:8080 \
  vllm/vllm-openai:v0.9.1 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit-mm-per-prompt.image 8 \
  --port 8080 \
  --dtype bfloat16 \
  --model-impl vllm \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen2.5-VL \
  --trust-remote-code \
  --tokenizer-pool-size 16 \
  --max-model-len 3300 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 64 \
  --swap-space 8 \
  --tensor-parallel-size 1 \
  --disable-log-stats \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill

# 在线并发压测
使用locust进行压测。
输入image: concurrent_test_image.py
输入frames: concurrent_test_frames.py
输入video: concurrent_test_video.py

* 安装locust

```shell
pip install locust
```

* 执行压测

1. 生成性能报告
# 无头模式运行并生成报告
python -m locust -f concurrent_test_frames.py -u 10 -r 5 -t 5m --host http://localhost:8080 --headless --html report.html --csv results
生成的文件：
report.html - HTML格式详细报告
results_stats.csv - 统计数据CSV
results_failures.csv - 失败请求CSV
results_stats_history.csv - 历史统计CSV

2. 启动Web界面
不使用--headless参数：
# 启动Web界面模式
python -m locust -f concurrent_test_frames.py --host http://localhost:8080
然后访问：http://localhost:8089

Web界面功能
实时监控：请求数、失败率、响应时间
图表展示：RPS、响应时间分布图
用户控制：动态调整并发用户数和启动速率
实时日志：查看请求状态和错误信息


python -m locust -f concurrent_test_image.py -u 200 -r 50 -t 30m --host http://localhost:8080 --headless --html report.html --csv results

python -m locust -f concurrent_test_video.py -u 200 -r 50 -t 10m --host http://localhost:8080 --headless --html report.html --csv results




# vlm_inference_stress_test
VLM model inference stress test using locust

# 使用Docker启动vLLM，并加载模型, 进行压测
参数示例, 适用于G6e.2xlarge

# 1. 启动VLM模型
## Qwen/Qwen2.5-VL-7B-Instruct
docker run --gpus all --cpus=8 --rm -it \
  -p 8080:8080 \
  vllm/vllm-openai:v0.10.0 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --limit-mm-per-prompt.image 16 \
  --port 8080 \
  --dtype bfloat16 \
  --model-impl vllm \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen2.5-VL \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --swap-space 8 \
  --tensor-parallel-size 1 \
  --disable-log-stats \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill

## Qwen/Qwen2.5-VL-3B-Instruct
docker run --gpus all --cpus=8 --rm -it \
  -p 8080:8080 \
  vllm/vllm-openai:v0.10.0 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --limit-mm-per-prompt.image 16 \
  --port 8080 \
  --dtype bfloat16 \
  --model-impl vllm \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen2.5-VL \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --swap-space 8 \
  --tensor-parallel-size 1 \
  --disable-log-stats \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill

# 2. 在线并发压测
使用locust进行压测。
输入image, 使用: concurrent_test_image.py
输入抽帧后的frames, 使用: concurrent_test_frames.py
输入video, 使用: concurrent_test_video.py

## 执行压测运行方式:

## 1. 生成性能报告
无头模式运行并生成报告
python -m locust -f concurrent_test_frames.py -u 200 -r 200 -t 20m --host http://localhost:8080 --headless --html report.html --csv results
生成的文件：
report.html - HTML格式详细报告
results_stats.csv - 统计数据CSV
results_failures.csv - 失败请求CSV
results_stats_history.csv - 历史统计CSV

## 2. 启动Web界面
不使用--headless参数：
启动Web界面模式
python -m locust -f concurrent_test_frames.py --host http://localhost:8080
然后访问：http://localhost:8080

Web界面功能
实时监控：请求数、失败率、响应时间
图表展示：RPS、响应时间分布图
用户控制：动态调整并发用户数和启动速率
实时日志：查看请求状态和错误信息





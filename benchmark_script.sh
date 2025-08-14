#!/bin/bash

# 梯度压测脚本 - 找到性能上限
echo "开始梯度压测，寻找性能上限..."

# 测试不同并发数
concurrency_levels=(50 100 200 400 800 1600 3200)

for concurrency in "${concurrency_levels[@]}"; do
    echo "========================================="
    echo "测试并发数: $concurrency"
    echo "========================================="
    
    # 运行压测
    python -m locust -f locustfile.py \
        -u $concurrency \
        -r 50 \
        -t 2m \
        --host http://localhost:8080 \
        --headless \
        --csv=results_${concurrency}
    
    echo "并发数 $concurrency 测试完成"
    sleep 10  # 让系统恢复
done

echo "所有测试完成，请查看 results_*.csv 文件分析结果"
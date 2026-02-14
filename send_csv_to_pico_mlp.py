import pandas as pd
import requests
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======= 配置 =======
csv_path = r"F:\\dataset\\merged csv\\dataset_balance.csv"
pico_ip = "http://192.168.178.162"
batch_size = 1

features = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count', 
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight'
]

# ======= 数据加载与预处理 =======
df = pd.read_csv(csv_path)
df = df[df['label'].isin(['DDoS-SlowLoris', 'BenignTraffic'])].copy()
df['label'] = df['label'].map({'DDoS-SlowLoris': 1, 'BenignTraffic': 0})
X = df[features]
y = df['label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 20% 作为测试集
start = int(len(X_scaled) * 0.8)
X_test = X_scaled[start:]
y_test = y[start:]

# ======= 发送数据 =======
y_pred = []
total_time = 0
total_samples = 0
mem_records = []

for i in range(0, len(X_test), batch_size):
    batch_X_raw = X_test[i:i+batch_size]
    batch_y = y_test[i:i+batch_size]
    batch_X = [[round(x, 6) for x in sample] for sample in batch_X_raw]

    try:
        start_time = time.time()
        res = requests.post(f"{pico_ip}/", json={"samples": batch_X}, timeout=10)
        elapsed = time.time() - start_time
        results = res.json()

        # 提取并记录 memory usage 信息
        if isinstance(results[-1], dict) and "mem_usage" in results[-1]:
            mem_info = results.pop()
            mem_info["mem_usage"]["timestamp"] = time.strftime("%H:%M:%S")
            mem_records.append(mem_info["mem_usage"])
            print(f"🧠 Pico Memory: {mem_info['mem_usage']}")
        else:
            print("⚠️ No memory info found.")
            mem_records.append({
                "allocated": 0,
                "free": 0,
                "usage_percent": 0.0,
                "timestamp": time.strftime("%H:%M:%S")
            })

        total_time += elapsed
        total_samples += len(results)

        for r in results:
            if isinstance(r, dict) and r.get("label") in [0, 1]:
                y_pred.append(r["label"])
            else:
                y_pred.append(0)

        print(f"Batch {i // batch_size}: {len(results)} samples, time: {elapsed:.2f}s")

    except Exception as e:
        print(f"Batch {i // batch_size} failed:", e)
        y_pred.extend([0] * len(batch_y))
        total_samples += len(batch_y)
        mem_records.append({
            "allocated": 0,
            "free": 0,
            "usage_percent": 0.0,
            "timestamp": time.strftime("%H:%M:%S")
        })

# ======= 评估 =======
y_true_filtered = []
y_pred_filtered = []
for yt, yp in zip(y_test[:len(y_pred)], y_pred):
    if yp in [0, 1]:
        y_true_filtered.append(yt)
        y_pred_filtered.append(yp)

if len(set(y_true_filtered + y_pred_filtered)) < 2:
    print("\n⚠️ Only one class present in prediction or label. Metrics defaulted to accuracy.")
    precision = recall = f1 = accuracy_score(y_true_filtered, y_pred_filtered)
else:
    precision = precision_score(y_true_filtered, y_pred_filtered)
    recall = recall_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered)

accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
avg_case_time_ms = (total_time / total_samples) * 1000 if total_samples else 0

# ======= 输出模型性能 =======
print("\n📊 Final Evaluation:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Total detection time : {total_time:.2f}s")
print(f"Avg case detection time: {avg_case_time_ms:.2f} ms")

# ======= 输出内存使用趋势 =======
print("\n📈 Pico Memory Trend (per batch):")
for idx, m in enumerate(mem_records):
    print(f"Batch {idx:03d} @ {m['timestamp']}: Used {m['usage_percent']:.2f}% | Alloc={m['allocated']} | Free={m['free']}")

# ======= 保存为 CSV 文件 =======
pd.DataFrame(mem_records).to_csv("memory_usage_log.csv", index_label="batch")
print("\n✅ Memory usage log saved to 'memory_usage_log.csv'")


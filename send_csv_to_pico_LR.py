import pandas as pd
import requests
import json
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

csv_path = r"F:\dataset\merged csv\dataset_balance.csv"
pico_ip = "http://192.168.178.162"
batch_size = 1  # 设置为1以统计单个 case 的耗时

# Features
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

# === 加载并预处理数据 ===
df = pd.read_csv(csv_path)
df = df[df['label'].isin(['DDoS-SlowLoris', 'BenignTraffic'])].copy()
df['label'] = df['label'].map({'DDoS-SlowLoris': 1, 'BenignTraffic': 0})
X = df[features]
y = df['label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 取后 20%
start = int(len(X_scaled) * 0.8)
X_test = X_scaled[start:]
y_test = y[start:]

# ==== 初始化 ====
y_pred = []
per_case_time_list = []
flow_duration_list = []

# ==== 遍历每个样本并发送到 Pico ====
for i in range(len(X_test)):
    x = [round(val, 6) for val in X_test[i]]
    true_label = y_test[i]
    original_flow_duration = df['flow_duration'].iloc[start + i]

    try:
        case_start = time.time()
        res = requests.post(f"{pico_ip}/", json={"samples": [x]}, timeout=10)
        case_time = (time.time() - case_start) * 1000  # ms

        result = res.json()[0]
        y_pred.append(result["label"])
        per_case_time_list.append(case_time)
        flow_duration_list.append(original_flow_duration)

        print(f"Sample {i}: Time={case_time:.2f} ms, Label={result['label']}")

    except Exception as e:
        print(f"Sample {i} failed:", e)
        y_pred.append(0)
        per_case_time_list.append(0)
        flow_duration_list.append(original_flow_duration)

# ==== 性能评估 ====
acc = accuracy_score(y_test[:len(y_pred)], y_pred)
prec = precision_score(y_test[:len(y_pred)], y_pred)
rec = recall_score(y_test[:len(y_pred)], y_pred)
f1 = f1_score(y_test[:len(y_pred)], y_pred)
avg_case_time_ms = sum(per_case_time_list) / len(per_case_time_list)

print("\n📊 Overall Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Avg prediction time per case: {avg_case_time_ms:.2f} ms")

# ==== 对比分析与绘图 ====
comparison_df = pd.DataFrame({
    'PredictionTime_ms': per_case_time_list,
    'FlowDuration': flow_duration_list
})
comparison_df.to_csv("pico_case_time_vs_flow_duration.csv", index=False)
print("✅ Saved comparison data to pico_case_time_vs_flow_duration.csv")

avg_flow_duration = sum(flow_duration_list) / len(flow_duration_list)

print(f"\n📈 Average Prediction Time (ms): {avg_case_time_ms:.2f}")
print(f"📈 Average Flow Duration        : {avg_flow_duration:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(per_case_time_list)), per_case_time_list, label='Prediction Time (ms)', alpha=0.6)
plt.scatter(range(len(flow_duration_list)), flow_duration_list, label='Flow Duration', alpha=0.6)
plt.title("Per-case Prediction Time vs. Flow Duration (Pico)")
plt.xlabel("Sample Index")
plt.ylabel("Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pico_prediction_vs_flowduration.png")
plt.show()
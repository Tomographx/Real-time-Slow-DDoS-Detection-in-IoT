import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import psutil
import datetime  # ✅ 用于生成时间戳

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

csv_path = '/home/chen/dataset_balance.csv'
data = pd.read_csv(csv_path)
dict_2classes = {
    'DDoS-SlowLoris': 'Attack', 
    'BenignTraffic': 'Benign'
}
data['label'] = data['label'].map(dict_2classes)
data = data.dropna(subset=['label'])
data['label'] = data['label'].map({'Benign': 0, 'Attack': 1})
X_columns = [
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
X = data[X_columns].astype('float32')
y = data['label'].astype('int32')
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X_columns)
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
class LightweightMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size):
        super(LightweightMLP, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        temp = F.relu(self.linear_1(x))
        return self.linear_2(temp)

features_num = X_train.shape[1]
hidden_neurons_num = 3
output_neurons_num = 1
model = LightweightMLP(features_num, hidden_neurons_num, output_neurons_num).to(device)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = torch.tensor([1.0, class_weights[1]], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(torch.FloatTensor([weights[1] / weights[0]])).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
batch_size = 128
X_train_tensor = torch.tensor(X_train.values).float().to(device)
y_train_tensor = torch.tensor(y_train.values).float().to(device)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_epochs = 100
pbar = tqdm(total=num_epochs)
loss_list = [None] * num_epochs
acc_list = [None] * num_epochs
epoch_time_list = [None] * num_epochs  

start_train_time = time.time()  

for epoch in range(num_epochs):
    epoch_start_time = time.time()  

    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    times = 0

    for inputs, labels in train_loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            y_true = labels.cpu().numpy()
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += accuracy_score(y_true, preds)
            times += 1

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_accuracy / times
    loss_list[epoch] = epoch_loss
    acc_list[epoch] = epoch_acc

    epoch_end_time = time.time()  
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_time_list[epoch] = epoch_duration

   
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_duration:.4f}s")

    pbar.update(1)

end_train_time = time.time()  
training_duration = end_train_time - start_train_time

pbar.reset()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list, label='Training Accuracy')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
start_pred_time = time.time()  

X_test_tensor = torch.tensor(X_test.values).float().to(device)
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor).squeeze()
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().cpu().numpy()

end_pred_time = time.time()  
prediction_duration = end_pred_time - start_pred_time
print("Calculating per-case prediction time...")
per_case_times = []
with torch.no_grad():
    for i in tqdm(range(len(X_test))):
        x_single = torch.tensor(X_test.iloc[i].values).float().to(device).unsqueeze(0)
        start_case = time.time()
        _ = model(x_single)
        end_case = time.time()
        elapsed_ms = (end_case - start_case) * 1000
        per_case_times.append(elapsed_ms)
per_case_df = pd.DataFrame({'CaseIndex': range(len(per_case_times)), 'PredictionTime_ms': per_case_times})
per_case_df.to_csv("per_case_prediction_time_MLP.csv", index=False)
print("Per-case prediction time saved to per_case_prediction_time_MLP.csv.")

# 还原原始 flow_duration 值（未标准化前）在测试集中的部分
original_flow_duration_test = data['flow_duration'].iloc[split_idx:].reset_index(drop=True)

# 合并 prediction time 和 flow_duration 到一个 DataFrame
comparison_df = pd.DataFrame({
    'PredictionTime_ms': per_case_times,
    'FlowDuration': original_flow_duration_test
})

# 保存数据
comparison_df.to_csv("flow_vs_prediction_time_MLP.csv", index=False)
print("Saved comparison data to flow_vs_prediction_time_MLP.csv.")

# 计算并输出平均值对比
avg_pred_time = np.mean(comparison_df['PredictionTime_ms'])
avg_flow_duration = np.mean(comparison_df['FlowDuration'])
print(f"\n📊 avg Prediction Time: {avg_pred_time:.4f} ms")
print(f"📊 avg Flow Duration: {avg_flow_duration:.2f} ms")

# 绘图：每个样本的 prediction time 与 flow_duration 对比
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.boxplot(
    [comparison_df['PredictionTime_ms'], comparison_df['FlowDuration']],
    labels=['Prediction Time (ms)', 'Flow Duration']
)
plt.yscale('log')  # 设置对数坐标轴
plt.title('Prediction Time vs. Flow Duration——MLP baseline')
plt.ylabel('Time (ms, log scale)')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("boxplot_logscale_prediction_vs_flowduration.png")
plt.show()






average_case_time_ms = np.mean(per_case_times)
print(f"Average Prediction Time per Case: {average_case_time_ms:.4f} ms")
acc = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("===== Final Evaluation on Test Set =====")
print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(f"Training Time: {training_duration:.4f} seconds")
print(f"Prediction Time: {prediction_duration:.4f} seconds")
result = {
    "Model": "MLP",
    "Accuracy": float(acc),
    "Precision": float(precision),
    "Recall": float(recall),
    "F1": float(f1),
    "Train Time (s)": float(training_duration),
    "Predict Time (s)": float(prediction_duration),
    "Avg Case Predict Time (ms)": float(average_case_time_ms)
}
with open("result_MLP.json", "w") as f:
    json.dump(result, f, indent=4)

print("Model evaluation results have been saved to result_MLP.json.")
W1 = model.linear_1.weight.detach().cpu().numpy()  # shape: [hidden_size, input_dim]
W1 = W1.T.tolist()  # 转置为 [input_dim, hidden_size]

B1 = model.linear_1.bias.detach().cpu().numpy().tolist()  # shape: [hidden_size]

W2 = model.linear_2.weight.detach().cpu().numpy()  # shape: [output_size, hidden_size]
W2 = W2.T.tolist()  # 转置为 [hidden_size, output_size]

B2 = model.linear_2.bias.detach().cpu().numpy().tolist()  # shape: [output_size]

# Step 2: 保存为 MicroPython 可用的 Python 文件
with open("mlp_weights_for_pico.py", "w") as f:
    f.write("W1 = " + json.dumps(W1, separators=(',', ':'), indent=None) + "\n")
    f.write("B1 = " + json.dumps(B1, separators=(',', ':'), indent=None) + "\n")
    f.write("W2 = " + json.dumps(W2, separators=(',', ':'), indent=None) + "\n")
    f.write("B2 = " + json.dumps(B2, separators=(',', ':'), indent=None) + "\n")
print("✅ W1 shape (should be 46 x hidden):", len(W1), "x", len(W1[0]))
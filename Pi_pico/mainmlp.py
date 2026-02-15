import network
import socket
import ujson as json
import math
import time
import gc

# ===== WiFi Configuration =====
SSID = "Hoomie"
PASSWORD = "Rubdown-Bats4-Luxury"

# ===== Load Trained Weights =====
from mlp_weights_for_pico import W1, B1, W2, B2

EXPECTED_FEATURES = len(W1)  # 46 features

# ===== Activation and Math =====
def relu(x):
    return max(0.0, x)

def matmul(v, m):
    return [sum(v[i] * m[i][j] for i in range(len(v))) for j in range(len(m[0]))]

def add_bias(v, b):
    return [v[i] + b[i] for i in range(len(v))]

def sigmoid(x):
    if x < -100: return 0.0
    if x > 100: return 1.0
    return 1 / (1 + math.exp(-x))

def predict(sample):
    try:
        if len(sample) != EXPECTED_FEATURES:
            raise ValueError(f"Input feature length {len(sample)} != expected {EXPECTED_FEATURES}")
        h1 = matmul(sample, W1)
        h1 = add_bias(h1, B1)
        h1 = [relu(x) for x in h1]
        out = matmul(h1, W2)
        out = add_bias(out, B2)
        return sigmoid(out[0])
    except Exception as e:
        print("Sample error:", e)
        return -1

# ===== WiFi Setup =====
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
print("Connecting to WiFi...")
while not wlan.isconnected():
    time.sleep(1)
print("Connected:", wlan.ifconfig())

# ===== Socket Setup =====
addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(addr)
server.listen(5)
print("Listening on http://" + wlan.ifconfig()[0])

# ===== Main Loop =====
while True:
    try:
        client, addr = server.accept()
        print("Client connected:", addr)
        client.settimeout(5)

        request = b""
        while b"\r\n\r\n" not in request:
            chunk = client.recv(1024)
            if not chunk:
                break
            request += chunk

        header_data, _, remaining = request.partition(b"\r\n\r\n")
        header_text = header_data.decode()

        content_length = 0
        for line in header_text.split("\r\n"):
            if line.lower().startswith("content-length"):
                content_length = int(line.split(":")[1].strip())
                break

        body = remaining
        while len(body) < content_length:
            chunk = client.recv(1024)
            if not chunk:
                break
            body += chunk

        # ===== Parse & Predict =====
        data = json.loads(body.decode())
        samples = data.get("samples", [])
        results = []

        for s in samples:
            score = predict(s)
            if score == -1:
                results.append({"label": 0, "score": 0.0})
            else:
                label = 1 if score > 0.5 else 0
                results.append({"label": label, "score": round(score, 4)})

        # ===== 内存监控信息 =====
        alloc = gc.mem_alloc()
        free = gc.mem_free()
        total = alloc + free
        usage_percent = (alloc / total) * 100
        results.append({
            "mem_usage": {
                "allocated": alloc,
                "free": free,
                "usage_percent": round(usage_percent, 2)
            }
        })

        response = json.dumps(results)
        client.send("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
        client.send(response)

        print(f"📦 Memory Usage: {alloc}B alloc / {free}B free / {usage_percent:.2f}% used")

    except Exception as e:
        print("Error:", e)
    finally:
        try:
            client.close()
        except:
            pass

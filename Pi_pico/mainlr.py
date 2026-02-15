import network
import socket
import ujson as json
import math
import time

# ====== WiFi Configuration ======
SSID = "Hoomie"
PASSWORD = "Rubdown-Bats4-Luxury"

# ====== Logistic Regression Parameters ======
weights = [0.18523664763097186, -4.042390770532113, -0.04874000581903087, -0.7833697520131284, 0.07024568317678621, 0.07024568317678621, 0.0, 0.23538050627825816, 0.0, 0.3096618349201937, 0.1665315130533298, 0.42892122181780606, 0.0, 0.0, -0.19492369073479468, 1.0054340894395597, 0.036530686916815054, -1.2422942481372732, 3.4616015808697234, 0.7773984653063625, -0.04210945827787974, -0.2170304285127187, 0.0, 0.0, 0.0, 0.0, 0.3824338758627083, 0.18662545828832683, 0.0, -0.014681136130136399, 0.0, 0.04146822627025031, 0.04146822627025031, 0.1708584404822021, 0.06896421715913988, -0.5031727499463667, 0.4571588870273641, -0.604369767921662, -0.6738708976030999, 0.14966312709625784, -0.20590076700748144, -0.603046665575227, 0.7443832802521675, 0.17646697760229488, -0.1778777008733265, 0.16713282659574497]
bias = 0.12958748663656916

# ====== Network Setup ======
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
print("Connecting to WiFi...")
while not wlan.isconnected():
    time.sleep(1)
print("Connected:", wlan.ifconfig())

addr = socket.getaddrinfo("0.0.0.0", 80)[0][-1]
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(addr)
server.listen(5)
print("Listening on http://" + wlan.ifconfig()[0])

# ====== ML Prediction Functions ======
def sigmoid(x):
    if x < -100: return 0.0
    if x > 100: return 1.0
    return 1 / (1 + math.exp(-x))

def predict(sample):
    z = sum(sample[i] * weights[i] for i in range(len(sample))) + bias
    return sigmoid(z)

# ====== Request Handling Loop ======
while True:
    try:
        client, addr = server.accept()
        print("Client connected:", addr)
        request = b""
        client.settimeout(5)

        # Receive headers
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

        data = json.loads(body.decode())
        samples = data.get("samples", [])
        results = []

        for s in samples:
            score = predict(s)
            label = 1 if score > 0.5 else 0
            results.append({"label": label, "score": score})

        response = json.dumps(results)
        client.send("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
        client.send(response)

    except Exception as e:
        print("Error:", e)
    finally:
        try:
            client.close()
        except:
            pass

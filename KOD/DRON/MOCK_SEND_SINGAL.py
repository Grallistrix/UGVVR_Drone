import socket
import time
import random

IP = "127.0.0.1"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    negX = random.randint(1,2)
    negY = random.randint(1,2)
    if negX%2 == 0:
        x = x*(-1)
    if negY%2 == 0:
        y = y*(-1)
        
    msg = f"{x:.3f};{y:.3f}"

    data = msg.encode("utf-8")

    sock.sendto(data, (IP, PORT))

    print(f"Sent: {msg}")

    time.sleep(1)
import socket


class UDPReceiver:
    def __init__(self, ip="127.0.0.1", port=5005, buffer_size=1024):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

        print(f"[UDP] Listening on {self.ip}:{self.port}")

    def receive_raw(self):
        data, addr = self.sock.recvfrom(self.buffer_size)

        msg = data.decode("utf-8")

        return msg, addr

    def receive_vector2(self):
        msg, addr = self.receive_raw()

        try:
            x, y = map(float, msg.split(";"))
            return x, y, addr

        except Exception as e:
            print(f"[UDP] Parse error: {e}")
            return None, None, addr
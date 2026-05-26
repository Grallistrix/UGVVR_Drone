import io
import socketserver
import threading
from http import server
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PORT = 80

# Nowa klasa bufora, która dziedziczy po io.BytesIO
class ThreadSafeBytesIO(io.BytesIO):
    def __init__(self):
        super().__init__()
        self.condition = threading.Condition()
        self.frame = b''

    def write(self, buf):
        # Sprawdzamy czy to początek nowej klatki JPEG
        if buf.startswith(b'\xff\xd8'):
            with self.condition:
                self.frame = self.getvalue()  # Pobieramy całą klatkę ze strumienia
                self.seek(0)
                self.truncate()               # Czyścimy bufor pod następną klatkę
                self.condition.notify_all()
        return super().write(buf)

print("Inicjalizacja kamery...")
picam = Picamera2()

config = picam.create_video_configuration(main={'size': (640, 480)})
config["fps"] = 30
picam.configure(config)

# Teraz przekazujemy obiekt zgodny z io.BufferedIOBase
buffer = ThreadSafeBytesIO()
stream_output = FileOutput(buffer)

encoder = MJPEGEncoder()
picam.start_recording(encoder, stream_output)
print("Kamera działa w trybie MJPEG.")

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()

        try:
            while True:
                with buffer.condition:
                    buffer.condition.wait()
                    frame = buffer.frame

                # Szybkie wysyłanie klatki
                self.wfile.write(b'--FRAME\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
        except Exception as e:
            print(f"Rozłączono klienta: {e}")

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

print(f"Start serwera na porcie {PORT}...")
try:
    StreamingServer(('0.0.0.0', PORT), StreamingHandler).serve_forever()
finally:
    print("Zamykanie kamery...")
    picam.stop_recording()
    picam.close()
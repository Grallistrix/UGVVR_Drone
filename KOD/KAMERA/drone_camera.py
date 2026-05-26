import io
import socketserver
import threading
import time
from http import server
import cv2  # Zamiast picamera2 używamy OpenCV do kamer USB

PORT = 80

class CameraBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = b''
        self.is_running = True

    def update_frame(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()

# Globalny bufor na klatki
buffer = CameraBuffer()

def camera_capture_thread():
    """Wątek odpowiedzialny za przechwytywanie obrazu z kamerki USB."""
    print("Inicjalizacja kamery USB...")
    
    # 0 zazwyczaj odpowiada urządzeniu /dev/video0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("BŁĄD: Nie można otworzyć kamerki USB (/dev/video0)!")
        return

    # Ustawienie rozdzielczości (takiej jak chciałeś: 640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Wymuszenie formatu MJPEG z kamery, jeśli sprzęt to wspiera (zwiększa FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Kamera USB działa. Rozpoczynam przechwytywanie.")

    try:
        while buffer.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Błąd odczytu klatki z kamery, ponawiam...")
                time.sleep(0.1)
                continue

            # Kompresujemy surową klatkę do formatu JPEG (jakość 80 dla balansu jakość/płynność)
            ret, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            if ret:
                # Zamieniamy spakowany obraz na bajty i wysyłamy do bufora
                buffer.update_frame(encoded_img.tobytes())
                
            # Krótki sleep, aby dopasować się do ~30 FPS i nie palić procesora w 100%
            time.sleep(1 / 30)
            
    finally:
        print("Zwalnianie zasobów kamery...")
        cap.release()

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

                # Szybkie wysyłanie klatki do przeglądarki/Unity
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

# Uruchomienie wątku kamery w tle
cap_thread = threading.Thread(target=camera_capture_thread)
cap_thread.daemon = True
cap_thread.start()

print(f"Start serwera HTTP na porcie {PORT}...")
try:
    server_address = ('0.0.0.0', PORT)
    httpd = StreamingServer(server_address, StreamingHandler)
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nZatrzymywanie serwera...")
finally:
    buffer.is_running = False
    cap_thread.join(timeout=2)
    print("Serwer zamknięty.")
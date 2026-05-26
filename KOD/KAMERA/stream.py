from picamera2 import Picamera2
import io, socketserver
from http import server

print("inicjuje kamere")
PORT = 80
picam = Picamera2()
picam.start()
print("kamera dziala")

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        #print("START - KOMENDA")
        if self.path != '/':
            print("ERR 404")
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        stream = io.BytesIO()
        #print("STREAM GOTOWY")
        while True:
            stream.seek(0)
            stream.truncate()
            picam.capture_file(stream, format='jpeg')
            self.wfile.write(b'--FRAME\r\n')
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(stream.tell()))
            self.end_headers()
            self.wfile.write(stream.getvalue())
            self.wfile.write(b'\r\n')

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    
print("Startuje strema")
StreamingServer(('0.0.0.0', PORT), StreamingHandler).serve_forever()
print("KONIEC")
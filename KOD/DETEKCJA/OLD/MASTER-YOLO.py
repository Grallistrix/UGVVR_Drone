import cv2
import numpy as np
from ultralytics import YOLO
import time
import websockets
import threading
import queue
import asyncio
import threading
import cv2
import io
import socketserver
from http import server


# CONFIG

WS_URL = "ws://localhost:8765"
MIN_CONTOUR_AREA = 1500
THRESHOLD_VALUE = 60
BLUR_SIZE = (11, 11)

latest_frame_vis = None
frame_lock = threading.Lock()


class VisStreamingHandler(server.BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path != "/":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=FRAME"
        )
        self.end_headers()

        global latest_frame_vis

        while True:

            with frame_lock:
                frame = latest_frame_vis

            if frame is None:
                continue

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            self.wfile.write(b"--FRAME\r\n")
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(buffer)))
            self.end_headers()
            self.wfile.write(buffer.tobytes())
            self.wfile.write(b"\r\n")
            
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
# MODEL
model = YOLO("yolo26n-seg.pt")

# CAMERA
# Foreign
#cap = cv2.VideoCapture("http://camera2:80")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open webcam!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def start_vis_server(port=8080):

    print(f"[VIS SERVER] running on :{port}")

    server = StreamingServer(
        ("0.0.0.0", port),
        VisStreamingHandler
    )

    server.serve_forever()
    
threading.Thread(
    target=start_vis_server,
    daemon=True
).start()
# ============================================
# BACKGROUND SUBTRACTOR
# ============================================
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=600,
    varThreshold=50,
    detectShadows=False
)

# ============================================
# ALERT SYSTEM (THREAD SAFE)
# ============================================
msg_queue = queue.Queue()

async def ws_sender():
    while True:
        msg = await asyncio.to_thread(msg_queue.get)

        try:
            async with websockets.connect(WS_URL) as websocket:
                await websocket.send(msg)
                print(f"[WS] sent: {msg}")

        except Exception as e:
            print(f"[WS ERROR] {e}")


def ws_thread():
    asyncio.run(ws_sender())


threading.Thread(target=ws_thread, daemon=True).start()

def send(msg: str):
    msg_queue.put(msg)

# ============================================
# STATE
# ============================================
anomaly_start_time = None
alert_state = "OFF"  # OFF / ON

# ============================================
# MAIN LOOP
# ============================================
while True:
 
    ret, frame_raw = cap.read()
    if not ret:
        break

    frame_vis = frame_raw.copy()


    person_mask = np.zeros((frame_raw.shape[0], frame_raw.shape[1]), dtype=np.uint8)

    results = model.predict(frame_raw, verbose=False)

    anomaly_detected = False

    for result in results:

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                if int(cls) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for mask, cls in zip(masks, classes):
                if int(cls) != 0:
                    continue

                mask = cv2.resize(mask, (frame_raw.shape[1], frame_raw.shape[0]))
                mask = (mask > 0.5).astype(np.uint8) * 255

                person_mask = cv2.bitwise_or(person_mask, mask)

    fg_mask = back_sub.apply(frame_raw)

    fg_mask = cv2.GaussianBlur(fg_mask, BLUR_SIZE, 0)

    _, fg_mask = cv2.threshold(fg_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_time = time.time()
    intruder_present = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        contour_mask = np.zeros_like(person_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        overlap = cv2.bitwise_and(contour_mask, person_mask)
        if cv2.countNonZero(overlap) > 0:
            continue

        intruder_present = True

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame_vis, "INTRUDER", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ============================================
    # STATE MACHINE
    # ============================================
    if intruder_present:
        if anomaly_start_time is None:
            anomaly_start_time = current_time

        duration = current_time - anomaly_start_time

        cv2.putText(frame_vis, f"{duration:.1f}s", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if duration >= 2 and alert_state == "OFF":
            send("Alert ON")
            alert_state = "ON"

    else:
        anomaly_start_time = None

        if alert_state == "ON":
            send("Alert OFF")
            alert_state = "OFF"

    cv2.imshow("Frame", frame_vis)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Person Mask", person_mask)

    
    if cv2.waitKey(1) == 27:
        break
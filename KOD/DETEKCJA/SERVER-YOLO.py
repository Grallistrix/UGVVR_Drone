import cv2
import numpy as np
from ultralytics import YOLO
import time
import asyncio
import websockets
from aiohttp import web
import threading

# ============================================
# CONFIG
# ============================================
HOST = "0.0.0.0"
PORT_WS = 8765
PORT_HTTP = 80 

MIN_CONTOUR_AREA = 1500
THRESHOLD_VALUE = 60
BLUR_SIZE = (11, 11)

# ============================================
# MODEL
# ============================================
print("Loading model")
model = YOLO("yolo26n-seg.pt")

# ============================================
# CAMERA
# ============================================
print("Opening Camera")
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://camera2")

if not cap.isOpened():
    raise Exception("Cannot open webcam!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================
# BACKGROUND SUBTRACTOR
# ============================================
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=600,
    varThreshold=50,
    detectShadows=False
)

# ============================================
# WEBSOCKET SERVER STATE
# ============================================
clients = set()
latest_frame = None
frame_lock = threading.Lock()

class AIState:
    def __init__(self):

        self.anomaly_start_time = None
        self.last_intruder_seen = None

        self.intruder_timer = 0.0
        self.alert_state = "OFF"
        self.intruder_present = False

        # POI = (x, y, w, h)
        self.poi = None
       
state = AIState()
main_loop = None

def detect_persons(model, frame):

    person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    person_boxes = []

    results = model.predict(frame, verbose=False)

    for result in results:

        if result.masks is None or result.boxes is None:
            continue

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for mask, cls in zip(masks, classes):

            # YOLO class 0 = person
            if int(cls) != 0:
                continue

            mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0])
            )

            mask = (mask > 0.5).astype(np.uint8) * 255

            person_mask = cv2.bitwise_or(
                person_mask,
                mask
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:

                if cv2.contourArea(c) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(c)

                person_boxes.append((x, y, w, h))

    return person_mask, person_boxes
def compute_motion_mask(back_sub, frame):

    fg_mask = back_sub.apply(frame)

    fg_mask = cv2.GaussianBlur(fg_mask, BLUR_SIZE, 0)

    _, fg_mask = cv2.threshold(
        fg_mask,
        THRESHOLD_VALUE,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    fg_mask = cv2.morphologyEx(
        fg_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    fg_mask = cv2.morphologyEx(
        fg_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return fg_mask, contours

def analyze_contours(
    contours,
    person_mask,
    frame_vis,
    poi
):

    intruder_present = False

    px, py, pw, ph = poi

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # =========================================
        # CHECK IF INSIDE POI
        # =========================================

        center_x = x + w // 2
        center_y = y + h // 2

        inside_poi = (
            px <= center_x <= px + pw and
            py <= center_y <= py + ph
        )

        if not inside_poi:
            continue

        contour_mask = np.zeros_like(person_mask)

        cv2.drawContours(
            contour_mask,
            [contour],
            -1,
            255,
            -1
        )

        overlap = cv2.bitwise_and(
            contour_mask,
            person_mask
        )

        # =========================================
        # PERSON DETECTED
        # =========================================

        if cv2.countNonZero(overlap) > 0:

            cv2.rectangle(
                frame_vis,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            draw_label(
                frame_vis,
                "IGNORED",
                x,
                y - 10,
                (255, 0, 0)
            )

            continue

        # =========================================
        # INTRUDER DETECTED
        # =========================================

        intruder_present = True

        cv2.rectangle(
            frame_vis,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        draw_label(
            frame_vis,
            "INTRUDER",
            x,
            y - 10,
            (0, 0, 255)
        )

    return intruder_present

def update_state(
    state: AIState,
    intruder_present,
    current_time
):

    GRACE_PERIOD = 0.2

    # =========================================
    # INTRUDER DETECTED
    # =========================================

    if intruder_present:

        state.last_intruder_seen = current_time

        if state.anomaly_start_time is None:
            state.anomaly_start_time = current_time

        state.intruder_timer = (
            current_time - state.anomaly_start_time
        )

        if (
            state.intruder_timer >= 2 and
            state.alert_state == "OFF"
        ):

            print("[ALERT] ON")

            state.alert_state = "ON"

            return "ALERT_ON"

    # =========================================
    # TEMPORARY LOST DETECTION
    # =========================================

    else:

        if (
            state.last_intruder_seen is not None and
            current_time - state.last_intruder_seen < GRACE_PERIOD
        ):

            intruder_present = True

            state.intruder_timer = (
                current_time - state.anomaly_start_time
            )

        else:

            state.anomaly_start_time = None
            state.last_intruder_seen = None
            state.intruder_timer = 0.0

            if state.alert_state == "ON":

                print("[ALERT] OFF")

                state.alert_state = "OFF"

                return "ALERT_OFF"

    state.intruder_present = intruder_present

    return None

def render(
    frame_vis,
    fg_mask,
    person_mask,
    person_boxes,
    state
):

    # =========================================
    # DRAW POI
    # =========================================

    if state.poi is not None:

        x, y, w, h = state.poi

        cv2.rectangle(
            frame_vis,
            (x, y),
            (x + w, y + h),
            (255, 0, 255),
            2
        )

    # =========================================
    # DRAW PERSONS
    # =========================================

    for (x, y, w, h) in person_boxes:

        cv2.rectangle(
            frame_vis,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        draw_label(
            frame_vis,
            "PERSON",
            x,
            y - 10,
            (0, 255, 255)
        )

    # =========================================
    # DRAW TIMER
    # =========================================

    if state.intruder_present:

        text = (
            f"INTRUDER TIME: "
            f"{state.intruder_timer:.2f}s"
        )

        cv2.putText(
            frame_vis,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    draw_debug(
        frame_vis,
        fg_mask,
        person_mask
    )
    
async def handler(websocket):

    global state

    clients.add(websocket)

    print("[WS] Client connected")

    try:

        async for msg in websocket:

            print("[WS RX]", msg)

            # POI [WIDTH] [HEIGHT] [CENTERX] [CENTERY] - POI 400 300 640 360
            if msg.startswith("POI"):

                parts = msg.split()

                if len(parts) == 5:

                    pw = int(parts[1])
                    ph = int(parts[2])

                    cx = int(parts[3])
                    cy = int(parts[4])

                    x = int(cx - pw / 2)
                    y = int(cy - ph / 2)

                    state.poi = (
                        x,
                        y,
                        pw,
                        ph
                    )

                    print("[POI UPDATED]", state.poi)

    except Exception as e:

        print("[WS ERROR]", e)

    finally:

        clients.remove(websocket)

        print("[WS] Client disconnected")
        
async def broadcast(msg: str):
    if clients:
        await asyncio.gather(*[c.send(msg) for c in clients])

def draw_label(frame, text, x, y, color):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
    
async def video_stream_handler(request):
    """Generuje strumień MJPEG pobierając klatki zabezpieczone Lockiem."""
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache, private',
            'Pragma': 'no-cache'
        }
    )
    await response.prepare(request)

    while True:
        local_frame = None
        # Bezpiecznie pobieramy klatkę z wątku AI przy użyciu blokady
        with frame_lock:
            if latest_frame is not None:
                local_frame = latest_frame.copy()

        if local_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', local_frame)
            if not ret:
                await asyncio.sleep(0.03)
                continue
            
            frame_bytes = jpeg.tobytes()
            header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame_bytes)}\r\n\r\n".encode('utf-8')
            footer = b"\r\n"
            
            try:
                await response.write(header + frame_bytes + footer)
            except (ConnectionResetError, RuntimeError):
                break
        
        await asyncio.sleep(0.03) # Ok. 25 FPS dla przeglądarki
        
    return response

def ai_loop_thread():
    """Tradycyjna, synchroniczna pętla AI działająca na osobnym wątku."""
    global state, latest_frame, main_loop

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_vis = frame_raw.copy()

        if state.poi is None:
            h, w = frame_raw.shape[:2]
            state.poi = (0, 0, w, h)

        current_time = time.time()

        # Przetwarzanie obrazu przez AI i OpenCV
        
        person_mask, person_boxes = detect_persons(model, frame_raw)
        fg_mask, contours = compute_motion_mask(back_sub, frame_raw)
        intruder_present = analyze_contours(contours, person_mask, frame_vis, state.poi)

        event = update_state(state, intruder_present, current_time)

        # Wywołanie asynchronicznego broadcastu z poziomu zwykłego wątku thread
        if event in ["ALERT_ON", "ALERT_OFF"] and main_loop is not None:
            msg = "Alert ON" if event == "ALERT_ON" else "Alert OFF"
            asyncio.run_coroutine_threadsafe(broadcast(msg), main_loop)

        render(frame_vis, fg_mask, person_mask, person_boxes, state)
        
        # Bezpiecznie zapisujemy klatkę dla serwera HTTP
        with frame_lock:
            latest_frame = frame_vis
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.001)

# ============================================
# MAIN
# ============================================
async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()

    # 1. Uruchomienie serwera WebSocket (Port 8765)
    ws_server = await websockets.serve(handler, HOST, PORT_WS)
    print(f"[WS] Server running on {HOST}:{PORT_WS}")

    # 2. Uruchomienie serwera HTTP (Port 80)
    app = web.Application()
    app.router.add_get('/video', video_stream_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT_HTTP)
    await site.start()
    print(f"[HTTP] Video stream available at http://localhost:{PORT_HTTP}/video")

    # 3. Odpalenie pętli AI w osobnym, dedykowanym wątku systemowym
    ai_thread = threading.Thread(target=ai_loop_thread, daemon=True)
    ai_thread.start()

    # Trzymamy serwer asynchroniczny przy życiu
    await ws_server.wait_closed()


def draw_debug(frame_vis, fg_mask, person_mask):
    cv2.imshow("Frame", frame_vis)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Person Mask", person_mask)

    if cv2.waitKey(1) == 27:
        import os
        os._exit(0)
        
asyncio.run(main())
import cv2
import numpy as np
import time
import asyncio
import websockets
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
# ============================================
# CONFIG
# ============================================
HOST = "0.0.0.0"
PORT = 8765

MIN_CONTOUR_AREA = 1500
THRESHOLD_VALUE = 60
BLUR_SIZE = (11, 11)

# ============================================
# MODEL
# ============================================
print("Loading Mask R-CNN model...")
# Wybieramy domyślne, najlepsze wagi dla COCO dataset
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

# Przełączamy model w tryb ewaluacji (wyłączamy dropout, batchnorm itp.)
model_rcnn.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rcnn.to(device)
print(f"Model loaded successfully on device: {device}")

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
 
def detect_persons(model, frame):
    person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    person_boxes = []

    # 1. Przygotowanie obrazu dla PyTorcha (konwersja BGR -> RGB, HWC -> CHW, Normalizacja)
    # OpenCV używa BGR, a torchvision oczekuje RGB sprowadzonego do zakresu [0.0, 1.0]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    # 2. Inferencja (Model oczekuje listy obiektów batch, dodajemy [img_tensor])
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # 3. Wyciąganie wyników z tensora
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    masks = predictions["masks"].cpu().numpy()

    # W zestawie danych COCO (na którym uczony jest Mask R-CNN), klasa 'person' ma ID = 1
    PERSON_CLASS_ID = 1
    CONFIDENCE_THRESHOLD = 0.5  # Próg pewności wykrycia

    for i in range(len(labels)):
        if labels[i] != PERSON_CLASS_ID or scores[i] < CONFIDENCE_THRESHOLD:
            continue

        # Wyciągamy maskę dla danego obiektu (ma kształt [1, H, W])
        mask = masks[i][0]
        
        # Mask R-CNN zwraca wartości prawdopodobieństwa (0.0 do 1.0).
        # Tworzymy maskę binarną (0 lub 255)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # Łączymy z główną maską ludzi
        person_mask = cv2.bitwise_or(person_mask, binary_mask)

        # Pobieramy współrzędne boxa [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = boxes[i].astype(int)
        w = xmax - xmin
        h = ymax - ymin

        # Dodajemy do listy, odrzucając miniaturowe artefakty
        if w * h > 500:
            person_boxes.append((xmin, ymin, w, h))

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

            # =========================================
            # POI WIDTH HEIGHT CENTERX CENTERY
            # EXAMPLE:
            # POI 400 300 640 360
            # =========================================

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

async def ai_loop():

    global state

    while True:

        ret, frame_raw = cap.read()

        if not ret:
            continue

        frame_vis = frame_raw.copy()

        # =========================================
        # INIT DEFAULT POI
        # =========================================

        if state.poi is None:
            h, w = frame_raw.shape[:2]

            state.poi = (0, 0, w, h)

        current_time = time.time()

        # =========================================
        # PERSON DETECTION
        # =========================================

        person_mask, person_boxes = detect_persons(
            model_rcnn,
            frame_raw
        )

        # =========================================
        # MOTION DETECTION
        # =========================================

        fg_mask, contours = compute_motion_mask(
            back_sub,
            frame_raw
        )

        # =========================================
        # ANALYZE CONTOURS
        # =========================================

        intruder_present = analyze_contours(
            contours,
            person_mask,
            frame_vis,
            state.poi
        )

        # =========================================
        # UPDATE STATE
        # =========================================

        event = update_state(
            state,
            intruder_present,
            current_time
        )

        # =========================================
        # BROADCAST EVENTS
        # =========================================

        if event == "ALERT_ON":
            await broadcast("Alert ON")

        elif event == "ALERT_OFF":
            await broadcast("Alert OFF")

        # =========================================
        # RENDER
        # =========================================

        render(
            frame_vis,
            fg_mask,
            person_mask,
            person_boxes,
            state
        )
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        await asyncio.sleep(0.01)   

# ============================================
# MAIN
# ============================================
async def main():
    server = await websockets.serve(handler, HOST, PORT)

    print(f"[WS] Server running on {HOST}:{PORT}")

    await asyncio.gather(
        server.wait_closed(),
        ai_loop()
    )

def draw_debug(frame_vis, fg_mask, person_mask):
    cv2.imshow("Frame", frame_vis)
    cv2.imshow("Foreground Mask", fg_mask)
    cv2.imshow("Person Mask", person_mask)

    if cv2.waitKey(1) == 27:
        import os
        os._exit(0)
        
asyncio.run(main())
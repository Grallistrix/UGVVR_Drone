import asyncio
import websockets
from datetime import datetime
import time

WS_URL = "ws://localhost:8765"

async def client():
    async with websockets.connect(WS_URL) as ws:
        print("[CLIENT] Connected")

        start_time = time.time()
        poi_sent = False

        while True:
            # =====================================
            # RECEIVE
            # =====================================
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{now}] RECEIVED: {msg}")
            except asyncio.TimeoutError:
                pass

            # =====================================
            # SEND POI AFTER 20s
            # =====================================
            if not poi_sent and (time.time() - start_time >= 20):

                width = 100
                height = 100

                center_x = 640 // 2
                center_y = 480 // 2

                cmd = f"POI {width} {height} {center_x} {center_y}"

                await ws.send(cmd)

                now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{now}] SENT: {cmd}")

                poi_sent = True

asyncio.run(client())
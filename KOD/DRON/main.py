from udp_receiver import UDPReceiver
from track_control import TrackController


receiver = UDPReceiver(
    ip="127.0.0.1",
    port=5005
)

controller = TrackController(deadzone=0.2)

while True:
    x, y, addr = receiver.receive_vector2()

    if x is None:
        continue

    print(f"[INPUT] X={x:.3f} | Y={y:.3f}")

    direction = controller.joystick_to_direction(x, y)

    controller.send_to_driver(direction)
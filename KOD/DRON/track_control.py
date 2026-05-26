class TrackController:
    def __init__(self, deadzone=0.2):
        self.deadzone = deadzone

    def joystick_to_direction(self, x, y):
        """
        Zamienia joystick X/Y na kierunek:
        F - Forward
        B - Backward
        L - Left
        R - Right
        """

        # Deadzone
        if abs(x) < self.deadzone:
            x = 0.0

        if abs(y) < self.deadzone:
            y = 0.0

        # Brak ruchu
        if x == 0.0 and y == 0.0:
            return None

        # Priorytet osi Y
        if abs(y) >= abs(x):
            if y > 0:
                return "F"
            else:
                return "B"

        # Skręty
        else:
            if x > 0:
                return "R"
            else:
                return "L"

    def send_to_driver(self, direction):
        """
        Mock sterownika gąsienic.
        Docelowo tutaj można wrzucić GPIO / UART / CAN / itd.
        """

        if direction is None:
            print("[DRIVER] STOP")
            return

        match direction:
            case "F":
                print("[DRIVER] FORWARD")

            case "B":
                print("[DRIVER] BACKWARD")

            case "L":
                print("[DRIVER] LEFT")

            case "R":
                print("[DRIVER] RIGHT")
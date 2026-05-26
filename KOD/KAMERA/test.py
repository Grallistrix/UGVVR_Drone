import socket
import json
import sys

# ── Konfiguracja połączenia ──────────────────────────────────────────────────
DEFAULT_ROBOT_IP = "10.220.64.21"  # Zmień na IP swojego BananaPi
UDP_PORT = 5005
COMMAND = "STATUS"

def main():
    # Pobieranie IP z argumentu linii poleceń, jeśli zostało podane
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]
    else:
        robot_ip = DEFAULT_ROBOT_IP

    print(f"[UDP] Wysyłanie zapytania o STATUS do {robot_ip}:{UDP_PORT}...")

    # Tworzymy gniazdo UDP
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            # Bardzo ważne: ustawiamy timeout (np. 3 sekundy).
            # Jeśli robot nie odpowie, skrypt nie zawiesi się w nieskończoność.
            sock.settimeout(3.0)
            
            # 1. Wysyłanie prośby o status
            sock.sendto(COMMAND.encode('utf-8'), (robot_ip, UDP_PORT))
            
            # 2. Oczekiwanie na odpowiedź (bufor 4096 bajtów wystarczy na cały JSON)
            data, addr = sock.recvfrom(4096)
            
            print(f"[UDP] Otrzymano odpowiedź z adresu {addr}!")
            print("-" * 50)
            
            # 3. Dekodowanie bajtów na tekst i konwersja na obiekt JSON (słownik)
            status_json = json.loads(data.decode('utf-8'))
            
            # 4. Ładne wypisanie JSON-a na ekranie (wcięcie 4 spacje)
            print(json.dumps(status_json, indent=4, ensure_ascii=False))
            print("-" * 50)
            
            # Przykład: jak wyciągnąć konkretną wartość z odebranych danych?
            print("\nSzybki podgląd kluczowych parametrów:")
            print(f"  • Bateria: {status_json['adc']['batt_v']} V")
            print(f"  • Temp. Procesora: {status_json['sys']['cpu_temp_c']} °C")
            print(f"  • Odległość z sonaru: {status_json['pico']['sonar_mm']} mm")

        except socket.timeout:
            print("[BŁĄD] Przekroczono czas oczekiwania! Robot nie odpowiedział (sprawdź IP lub połączenie).")
        except json.JSONDecodeError:
            print("[BŁĄD] Otrzymano dane, ale nie są one poprawnym formatem JSON.")
        except Exception as e:
            print(f"[BŁĄD] Coś poszło nie tak: {e}")

if __name__ == "__main__":
    main()
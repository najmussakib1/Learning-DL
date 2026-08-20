import sys
import time

def typewriter(text, color_code, delay=0.25):
    for char in text:
        sys.stdout.write(f"{color_code}{char}")
        sys.stdout.flush()
        time.sleep(delay)
    print("\033[0m")
    time.sleep(1) 

GREEN = "\033[32m"
BLUE = "\033[34m"

typewriter("Ha-mim Tasnim", GREEN)
typewriter("Md. Najmus Sakib Thoha", BLUE)
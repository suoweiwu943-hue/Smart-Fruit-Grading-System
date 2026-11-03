# Run only after test_ir.py confirms polarity. Stop test_ir first.
from pynq import Overlay
import time
BLOCKED_LEVEL = 0  # change to 1 if test shows blocking=1
ol = Overlay("fruit_trigger.bit")
gpio = ol.axi_gpio_0
gpio.write(4, 0xFFFFFFFF)
gpio.write(12, 0)
ctrl = 1 | (BLOCKED_LEVEL << 3)
gpio.write(8, ctrl)
print("Clear the beam first, then block it. Ctrl+C stops.")
try:
    while True:
        s = gpio.read(0)
        if s & 4:
            gpio.write(8, ctrl | 4)       # busy
            gpio.write(8, ctrl | 4 | 2)   # acknowledge
            deadline = time.monotonic() + 1
            while gpio.read(0) & 4:
                if time.monotonic() > deadline:
                    raise RuntimeError("PL acknowledge timeout")
                time.sleep(.001)
            gpio.write(8, ctrl | 4)
            print("Trigger count:", s >> 8, flush=True)
            # Replace this delay with capture_and_send() after capture code is separated
            # from its old BaseOverlay/button initialization and loop.
            time.sleep(1)
            gpio.write(8, ctrl)
        time.sleep(.01)
finally:
    gpio.write(8, 0)

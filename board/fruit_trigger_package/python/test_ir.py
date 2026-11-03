# Run on PYNQ in the same directory as fruit_trigger.bit and .hwh.
from pynq import Overlay
import time
ol = Overlay("fruit_trigger.bit")
gpio = ol.axi_gpio_0
# AXI GPIO channel1 input, channel2 output; use MMIO registers explicitly.
gpio.write(0x04, 0xFFFFFFFF)
gpio.write(0x0C, 0)
gpio.write(0x08, 0)  # disabled: observe only
print("Observe clear and blocked values first; Ctrl+C stops.")
previous = None
try:
    while True:
        s = gpio.read(0)
        values = (s & 1, (s >> 1) & 1, (s >> 4) & 1)
        if values != previous:
            print("raw=%d filtered=%d valid=%d" % values, flush=True)
            previous = values
        time.sleep(0.02)
finally:
    gpio.write(0x08, 0)

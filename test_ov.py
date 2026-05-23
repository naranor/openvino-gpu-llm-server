import openvino as ov
import numpy as np

core = ov.Core()
devices = core.available_devices
print(f"Available devices: {devices}")

for device in devices:
    print(f"\nDevice: {device}")
    for param in core.get_property(device, 'SUPPORTED_PROPERTIES'):
        try:
            val = core.get_property(device, param)
            print(f"  {param}: {val}")
        except:
            pass

#!/usr/bin/env python3
"""Adapter Info - Print information about available GPUs.

Usage:
    python adapter_info.py
"""

import goldy


def main():
    print("Goldy GPU Adapter Information")
    print("=" * 50)
    
    # Create instance
    instance = goldy.Instance()
    print(f"Backend Type: {instance.backend_type}")
    print()
    
    # Enumerate adapters
    adapters = instance.enumerate_adapters()
    
    if not adapters:
        print("No GPU adapters found!")
        return
    
    print(f"Found {len(adapters)} adapter(s):\n")
    
    for i, adapter in enumerate(adapters):
        print(f"Adapter {i}:")
        print(f"  ID:     {adapter.id}")
        print(f"  Name:   {adapter.name}")
        print(f"  Vendor: {adapter.vendor}")
        print(f"  Type:   {adapter.device_type}")
        print()
    
    # Create a device on the first discrete GPU (or first available)
    print("Creating device...")
    try:
        discrete = next(
            (a for a in adapters if a.device_type == goldy.DeviceType.DISCRETE_GPU),
            None,
        )
        device = (discrete or adapters[0]).request_device()
        print(f"Created device on adapter {device.adapter_id}")
        print(f"Device valid: {device.is_valid()}")
        print(f"Shader libraries: {device.list_libraries()}")
    except goldy.GoldyError as e:
        print(f"Failed to create device: {e}")


if __name__ == '__main__':
    main()


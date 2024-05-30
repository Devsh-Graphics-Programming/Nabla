import json
import os
import argparse

def buildPath(file_path):
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), file_path))

def loadJSON(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception as ex:
        print(f"Error while reading file: {file_path}\nException: {ex}")

def buildDeviceHeader(device_json):
    emptyline = f""
    res = []

    # Header Guard
    res.append(f"#ifndef {device_json['headerGuard']}")
    res.append(f"#define {device_json['headerGuard']}")
    res.append(emptyline)
    
    # Includes
    for include in device_json['includes']:
        res.append(f"#include \"{device_json['includePath']}{include}\"")
    for include in device_json['stlIncludes']:
        res.append(f"#include <{include}>")
    res.append(emptyline)

    # Namespace
    res.append(f"namespace {device_json['namespace']}")
    res.append("{")
    res.append(emptyline)

    # Close Namespace
    res.append("}" + f" //{device_json['namespace']}")
    res.append(emptyline)

    # Close Header Guard
    res.append(f"#endif")

    return res

def writeDeviceHeader(file_path, device_json):
    try:
        with open(file_path, mode="w") as file:
            device_header = buildDeviceHeader(device_json)
            file.write('\n'.join(device_header))
    except Exception as ex:
        print(f"Error while writing to file: {file_path}\nException: {ex}")

if __name__ == "__main__":
    limits = loadJSON(buildPath(os.path.join('..', '..', 'src', 'nbl', 'video', 'device_limits.json')))
    writeDeviceHeader(
        buildPath(os.path.join('test_device_limits.h')),
        limits
    )
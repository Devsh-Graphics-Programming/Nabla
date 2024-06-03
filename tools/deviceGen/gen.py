from DeviceGen import *
from argparse import ArgumentParser
from sys import exit

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Header Files")

    parser.add_argument("limits_json_path", type=str, help="The path to the device_limits.json file")
    parser.add_argument("features_json_path", type=str, help="The path to the device_features.json file")
    parser.add_argument("limits_output_path", type=str, help="The output path for the test_device_limits.h file")
    parser.add_argument("features_output_path", type=str, help="The output path for the test_device_features.h file")

    args = parser.parse_args()

    limits = loadJSON(args.limits_json_path)
    writeDeviceHeader(
        args.limits_output_path,
        limits
    )
    features = loadJSON(args.features_json_path)
    writeDeviceHeader(
        args.features_output_path,
        features
    )

    exit(0)
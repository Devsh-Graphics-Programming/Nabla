from DeviceGen import *
from argparse import ArgumentParser
from sys import exit

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Header Files")

    parser.add_argument("limits_json_path", type=str, help="The path to the device_limits.json file")
    parser.add_argument("features_json_path", type=str, help="The path to the device_features.json file")
    parser.add_argument("limits_output_path", type=str, help="The output path for SPhysicalDeviceLimits_")
    parser.add_argument("features_output_path", type=str, help="The output path for SPhysicalDeviceFeatures_")
    parser.add_argument("traits_output_path", type=str, help="The output path for the device_capabilities_traits_members.h")

    args = parser.parse_args()

    limits = loadJSON(args.limits_json_path)
    writeHeader(
        args.limits_output_path + "members.h",
        buildDeviceHeader,
        json=limits
    )
    writeHeader(
        args.limits_output_path + "subset.h",
        buildSubsetMethod,
        json=limits
    )
    features = loadJSON(args.features_json_path)
    writeHeader(
        args.features_output_path + "members.h",
        buildDeviceHeader,
        json=features
    )
    writeHeader(
        args.features_output_path + "union.h",
        buildFeaturesMethod,
        json=features,
        op="|"
    )
    writeHeader(
        args.features_output_path + "union.h",
        buildFeaturesMethod,
        json=features,
        op="&"
    )
    writeHeader(
        args.traits_output_path,
        buildTraitsHeader,
        limits_json=limits,
        features_json=features
    )

    exit(0)
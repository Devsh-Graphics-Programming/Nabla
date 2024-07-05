from DeviceGen import *
from argparse import ArgumentParser
from sys import exit

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Header Files")

    parser.add_argument("--limits_json_path", type=str, help="The path to the device_limits.json file")
    parser.add_argument("--features_json_path", type=str, help="The path to the device_features.json file")
    parser.add_argument("--limits_output_members_path", type=str, help="The output path for SPhysicalDeviceLimits_members.h")
    parser.add_argument("--limits_output_subset_path", type=str, help="The output path for SPhysicalDeviceLimits_subset.h")
    parser.add_argument("--features_output_members_path", type=str, help="The output path for SPhysicalDeviceFeatures_members.h")
    parser.add_argument("--features_output_union_path", type=str, help="The output path for SPhysicalDeviceFeatures_union.h")
    parser.add_argument("--features_output_intersect_path", type=str, help="The output path for SPhysicalDeviceFeatures_intersect.h")
    parser.add_argument("--traits_output_members_path", type=str, help="The output path for the device_capabilities_traits_members.hlsl")
    parser.add_argument("--traits_output_testers_path", type=str, help="The output path for the device_capabilities_traits_testers.hlsl")
    parser.add_argument("--traits_output_defaults_path", type=str, help="The output path for the device_capabilities_traits_defaults.hlsl")
    parser.add_argument("--traits_output_floats_path", type=str, help="The output path for the device_capabilities_traits_floats.hlsl")
    parser.add_argument("--jit_traits_output_path", type=str, help="The output path for the device_capabilities_jit_traits.h")

    args = parser.parse_args()

    features = loadJSON(args.features_json_path)
    writeHeader(
        args.features_output_members_path,
        buildDeviceHeader,
        json=features
    )
    writeHeader(
        args.features_output_union_path,
        buildFeaturesMethod,
        json=features,
        op="|"
    )
    writeHeader(
        args.features_output_intersect_path,
        buildFeaturesMethod,
        json=features,
        op="&"
    )

    limits = loadJSON(args.limits_json_path)
    limits['core10'] = [
        {
            "comment": [
                "Always enabled if available, reported as limits",
                "Core 1.0 Features"
            ],
            "entries": []
        }
    ]

    for limit in MovedLimits:
        limits['core10'][0]['entries'].append(
            dict(
                type=limit['type'],
                name=limit['name'],
                value=limit['value']
            )
        )

    writeHeader(
        args.limits_output_members_path,
        buildDeviceHeader,
        json=limits
    )
    writeHeader(
        args.limits_output_subset_path,
        buildSubsetMethod,
        json=limits
    )
    writeHeader(
        args.traits_output_testers_path,
        buildTraitsHeader,
        type="Testers",
        template="NBL_GENERATE_MEMBER_TESTER({});",
        limits_json=limits,
        features_json=features,
        format_params=["name"]
    )
    writeHeader(
        args.traits_output_defaults_path,
        buildTraitsHeader,
        type="Defaults",
        template="NBL_GENERATE_GET_OR_DEFAULT({}, {}, {});",
        limits_json=limits,
        features_json=features,
        format_params=["name", "type", "value"],
        enable_constexprs=True
    )
    writeHeader(
        args.traits_output_floats_path,
        buildTraitsFloatHeader,
        type="Floats",
        template="static float {}() {{ return asfloat({}); }}",
        limits_json=limits,
        features_json=features,
        format_params=["name", "cpp_name"]
    )
    writeHeader(
        args.traits_output_members_path,
        buildTraitsHeader,
        type="Members",
        template="NBL_CONSTEXPR_STATIC_INLINE {} {} = impl::get_or_default_{}<device_capabilities>::value;",
        limits_json=limits,
        features_json=features,
        format_params=["type", "name", "name"]
    )
    writeHeader(
        args.jit_traits_output_path,
        buildTraitsHeader,
        type="JIT Members",
        template="NBL_CONSTEXPR_STATIC_INLINE {} {} = )===\" + CJITIncludeLoader::to_string({}.{}) + R\"===(;",
        limits_json=limits,
        features_json=features,
        format_params=["type", "name", "json_type", "cpp_name"],
        enable_jit=True
    )
    exit(0)
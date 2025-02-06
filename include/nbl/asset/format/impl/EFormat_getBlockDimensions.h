switch (_fclass)
{
    case EFC_8_BIT: [[fallthrough]];
    case EFC_16_BIT: [[fallthrough]];
    case EFC_24_BIT: [[fallthrough]];
    case EFC_32_BIT: [[fallthrough]];
    case EFC_48_BIT: [[fallthrough]];
    case EFC_64_BIT: [[fallthrough]];
    case EFC_96_BIT: [[fallthrough]];
    case EFC_128_BIT: [[fallthrough]];
    case EFC_192_BIT: [[fallthrough]];
    case EFC_256_BIT:
        return hlsl::uint32_t3(1u, 1u, 1u);
    case EFC_BC1_RGB: [[fallthrough]];
    case EFC_BC1_RGBA: [[fallthrough]];
    case EFC_BC2: [[fallthrough]];
    case EFC_BC3: [[fallthrough]];
    case EFC_BC4: [[fallthrough]];
    case EFC_BC5: [[fallthrough]];
    case EFC_BC6: [[fallthrough]];
    case EFC_BC7: [[fallthrough]];
    case EFC_ETC2_RGB: [[fallthrough]];
    case EFC_ETC2_RGBA: [[fallthrough]];
    case EFC_ETC2_EAC_RGBA: [[fallthrough]];
    case EFC_ETC2_EAC_R: [[fallthrough]];
    case EFC_ETC2_EAC_RG: [[fallthrough]];
    case EFC_ASTC_4X4:
        return hlsl::uint32_t3(4u, 4u, 1u);
    case EFC_ASTC_5X4:
        return hlsl::uint32_t3(5u, 4u, 1u);
    case EFC_ASTC_5X5:
        return hlsl::uint32_t3(5u, 5u, 1u);
    case EFC_ASTC_6X5:
        return hlsl::uint32_t3(6u, 5u, 1u);
    case EFC_ASTC_6X6:
        return hlsl::uint32_t3(6u, 6u, 1u);
    case EFC_ASTC_8X5:
        return hlsl::uint32_t3(8u, 5u, 1u);
    case EFC_ASTC_8X6:
        return hlsl::uint32_t3(8u, 6u, 1u);
    case EFC_ASTC_8X8:
        return hlsl::uint32_t3(8u, 8u, 1u);
    case EFC_ASTC_10X5:
        return hlsl::uint32_t3(10u, 5u, 1u);
    case EFC_ASTC_10X6:
        return hlsl::uint32_t3(10u, 6u, 1u);
    case EFC_ASTC_10X8:
        return hlsl::uint32_t3(10u, 8u, 1u);
    case EFC_ASTC_10X10:
        return hlsl::uint32_t3(10u, 10u, 1u);
    case EFC_ASTC_12X10:
        return hlsl::uint32_t3(12u, 10u, 1u);
    case EFC_ASTC_12X12:
        return hlsl::uint32_t3(12u, 12u, 1u);
    default:
        _NBL_DEBUG_BREAK_IF(true);
        return hlsl::uint32_t3(0u);
}

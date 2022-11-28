switch (_fclass)
{
    case EFC_8_BIT: [[fallthrough]];
    case EFC_16_BIT: [[fallthrough]];
    case EFC_24_BIT: [[fallthrough]];
    case EFC_32_BIT:
        return _fclass+1u;
    case EFC_48_BIT:
        return 6u;
    case EFC_BC1_RGB: [[fallthrough]];
    case EFC_BC1_RGBA: [[fallthrough]];
    case EFC_BC4: [[fallthrough]];
    case EFC_64_BIT:
        return 8u;
    case EFC_96_BIT:
        return 12u;
    case EFC_BC2: [[fallthrough]];
    case EFC_BC3: [[fallthrough]];
    case EFC_BC5: [[fallthrough]];
    case EFC_BC6: [[fallthrough]];
    case EFC_BC7: [[fallthrough]];
    case EFC_ASTC_4X4: [[fallthrough]];
    case EFC_ASTC_5X4: [[fallthrough]];
    case EFC_ASTC_5X5: [[fallthrough]];
    case EFC_ASTC_6X5: [[fallthrough]];
    case EFC_ASTC_6X6: [[fallthrough]];
    case EFC_ASTC_8X5: [[fallthrough]];
    case EFC_ASTC_8X6: [[fallthrough]];
    case EFC_ASTC_8X8: [[fallthrough]];
    case EFC_ASTC_10X5: [[fallthrough]];
    case EFC_ASTC_10X6: [[fallthrough]];
    case EFC_ASTC_10X8: [[fallthrough]];
    case EFC_ASTC_10X10: [[fallthrough]];
    case EFC_ASTC_12X10: [[fallthrough]];
    case EFC_ASTC_12X12: [[fallthrough]];
    case EFC_128_BIT:
        return 16u;
    case EFC_192_BIT:
        return 24u;
    case EFC_256_BIT:
        return 32u;
    default:
        _NBL_DEBUG_BREAK_IF(true);
        return 0u;
}
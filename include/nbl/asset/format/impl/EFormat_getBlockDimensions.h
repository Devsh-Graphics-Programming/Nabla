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
        return core::vector3du32_SIMD(1u,1u,1u);
    default:
        _NBL_DEBUG_BREAK_IF(true);
        return core::vector3du32_SIMD(0u);
}

switch(_fclass)
{
    case EFC_8_BIT: [[fallthrough]];
    case EFC_16_BIT: [[fallthrough]];
    case EFC_24_BIT: [[fallthrough]];
    case EFC_32_BIT:
        return _fclass + 1u;
    case EFC_48_BIT:
        return 6u;
    case EFC_64_BIT:
        return 8u;
    case EFC_96_BIT:
        return 12u;
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
#include <string>
#include <cstdint>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <iostream>

// Generator assumes naming convention 

enum E_FMT_SPECS
{
    EFS_SIGNED = 1<<0,
    EFS_UNSIGNED = 1<<1,
    EFS_FLOAT = 1<<2,
    EFS_NORMALIZED = 1<<3,
    EFS_PACKED = 1<<4,
    EFS_SCALED = 1<<5,
    EFS_SRGB = 1<<6
};

static const std::unordered_map<char, uint32_t> chToIx = { {'R', 0u}, {'G', 1u}, {'B', 2u}, {'A', 3u} };

uint32_t parseFmtSpec(std::string _fmtName, uint32_t _outBitsPerCh[4], uint32_t _outChOrder[4])
{
    auto isChannelChar = [](char c) { c = std::toupper(c); return c == 'R' || c == 'G' || c == 'B' || c == 'A'; };

    memset(_outBitsPerCh, 0, 16);
    memset(_outChOrder, 0xff, 16);

    uint32_t flags = 0u;
    _fmtName.erase(0u, 4u); //erase "ECF_"
    
    uint32_t orderIx = 0u;
    uint32_t chIx = 4u;
    std::string str;
    for (char c : _fmtName)
    {
        if (c == '_')
            break;
        if (isChannelChar(c))
        {
            if (chIx < 4u) // if valid channel idx
            {
                _outBitsPerCh[chIx] = std::stoul(str);
                str.clear();
            }
            _outChOrder[orderIx++] = chIx = chToIx.at(c);
        }
        else
            str += c;
    }
    _outBitsPerCh[chIx] = std::stoul(str);
    str.clear();
    if (_fmtName.find("UINT") != std::string::npos || _fmtName.find("UFLOAT") != std::string::npos || _fmtName.find("USCALED") != std::string::npos || _fmtName.find("UNORM") != std::string::npos)
        flags |= EFS_UNSIGNED;
    if (_fmtName.find("SINT") != std::string::npos || _fmtName.find("SFLOAT") != std::string::npos || _fmtName.find("SSCALED") != std::string::npos || _fmtName.find("SNORM") != std::string::npos)
        flags |= EFS_SIGNED;
    if (_fmtName.find("SCALED") != std::string::npos)
    {
        flags |= EFS_SCALED;
        flags |= EFS_NORMALIZED;
    }
    if (_fmtName.find("NORM") != std::string::npos)
        flags |= EFS_NORMALIZED;
    if (_fmtName.find("FLOAT") != std::string::npos)
        flags |= EFS_FLOAT;
    if (_fmtName.find("SRGB") != std::string::npos)
    {
        flags |= EFS_SRGB;
        flags |= EFS_UNSIGNED;
        flags |= EFS_NORMALIZED;
    }
    if (_fmtName.find("PACK") != std::string::npos)
        flags |= EFS_PACKED;

    return flags;
}

std::string getCastType(const uint32_t _flags, const uint32_t* _bitsPerCh, bool& _outLoopNeeded)
{
    _outLoopNeeded = false;
    const uint32_t fmtBitSz = std::accumulate(_bitsPerCh, _bitsPerCh + 4, (uint32_t)0u);

    std::string casttype = "int8_t";
    if (!(_flags & EFS_FLOAT))
    {
        if (fmtBitSz > 32u)
        {
            if (fmtBitSz > 64u)
            {
                if (_bitsPerCh[0] == 32u)
                    casttype = "int32_t";
                else if (_bitsPerCh[0] == 64u)
                    casttype = "int64_t";
                _outLoopNeeded = true;
            }
            else
                casttype = "int64_t";
        }
        else if (fmtBitSz > 16u)
            casttype = "int32_t";
        else if (fmtBitSz > 8u)
            casttype = "int16_t";

        if (_flags & EFS_UNSIGNED)
            casttype = "u" + casttype;
    }
    else return "";

    return casttype;
}

std::string genSizeFunc(const std::vector<std::string>& _fmtNames)
{
    std::string code = 
R"(uint32_t getTexelOrBlockSize(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        parseFmtSpec(fmt, bits, dummy);
        const uint32_t sz = std::accumulate(bits, bits + 4u, (uint32_t)0u) >> 3;
        code += "\tcase " + fmt + ": return " + std::to_string(sz) + ";\n";
    }

    code += "\tdefault: return 0;\n\t}\n}\n";

    return code;
}

std::string getChannelCountFunc(const std::vector<std::string>& _fmtNames)
{
    std::string code = 
R"(uint32_t getChannelCount(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        parseFmtSpec(fmt, bits, dummy);
        const uint32_t chCnt = [bits]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bits[i]; return r; }();
        code += "\tcase " + fmt + ": return " + std::to_string(chCnt) + ";\n";
    }

    code += "\tdefault: return 0;\n\t}\n}\n";

    return code;
}

std::string getIsSignedFunc(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(bool isSignedFormat(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_SIGNED)
            code += "\tcase " + fmt + ":\n";
    }
    code += "\t\treturn true;\n";
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
std::string getIsSignedFuncCT(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(template<ECOLOR_FORMAT cf>
constexpr bool isSignedFormat()
{
    return is_any_of<
        cf,
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_SIGNED)
            code += "\t" + fmt + ",\n";
    }
    *(code.rbegin() + 1) = ' '; //remove last comma
    code += "\t>::value;\n";
    code += "}\n";

    return code;
}

std::string getIsIntegerFormat(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(bool isIntegerFormat(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        bool isint = !(flags & EFS_NORMALIZED) && !(flags & EFS_FLOAT);
        if (isint)
            code += "\tcase " + fmt + ":\n";
    }
    code += "\t\treturn true;\n";
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
std::string getIsIntegerFormatCT(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(template<ECOLOR_FORMAT cf>
constexpr bool isIntegerFormat()
{
    return is_any_of<
        cf,
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (!(flags & EFS_NORMALIZED) && !(flags & EFS_FLOAT))
            code += "\t" + fmt + ",\n";
    }
    *(code.rbegin() + 1) = ' '; //remove last comma
    code += "\t>::value;\n";
    code += "}\n";

    return code;
}

std::string getIsFloatingPointFormat(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(bool isFloatingPointFormat(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_FLOAT)
            code += "\tcase " + fmt + ":\n";
    }
    code += "\t\treturn true;\n";
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
std::string getIsFloatingPointFormatCT(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(template<ECOLOR_FORMAT cf>
constexpr bool isFloatingPointFormat()
{
    return is_any_of<
        cf,
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_FLOAT)
            code += "\t" + fmt + ",\n";
    }
    *(code.rbegin() + 1) = ' '; //remove last comma
    code += "\t>::value;\n";
    code += "}\n";

    return code;
}

std::string getIsScaledFormat(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(//! SCALED implies NORMALIZED!
bool isScaledFormat(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_SCALED)
            code += "\tcase " + fmt + ":\n";
    }
    code += "\t\treturn true;\n";
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
std::string getIsScaledFormatCT(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(template<ECOLOR_FORMAT cf>
constexpr bool isScaledFormat()
{
    return is_any_of<
        cf,
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_SCALED)
            code += "\t" + fmt + ",\n";
    }
    *(code.rbegin() + 1) = ' '; //remove last comma
    code += "\t>::value;\n";
    code += "}\n";

    return code;
}

std::string getIsNormalizedFormat(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(bool isNormalizedFormat(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_NORMALIZED)
            code += "\tcase " + fmt + ":\n";
    }
    code += "\t\treturn true;\n";
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
std::string getIsNormalizedFormatCT(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(template<ECOLOR_FORMAT cf>
constexpr bool isNormalizedFormat()
{
    return is_any_of<
        cf,
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_NORMALIZED)
            code += "\t" + fmt + ",\n";
    }
    *(code.rbegin() + 1) = ' '; //remove last comma
    code += "\t>::value;\n";
    code += "}\n";

    return code;
}

std::string getIsSRGBFormat(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(bool isSRGBFormat(ECOLOR_FORMAT _fmt)
{
    switch (_fmt)
    {
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_SRGB)
            code += "\tcase " + fmt + ":\n";
    }
    code += "\t\treturn true;\n";
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
std::string getIsSRGBFormatCT(const std::vector<std::string>& _fmtNames)
{
    std::string code =
        R"(template<ECOLOR_FORMAT cf>
constexpr bool isSRGBFormat()
{
    return is_any_of<
        cf,
)";

    for (const std::string& fmt : _fmtNames)
    {
        uint32_t bits[4], dummy[4];
        const uint32_t flags = parseFmtSpec(fmt, bits, dummy);
        if (flags & EFS_SRGB)
            code += "\t" + fmt + ",\n";
    }
    *(code.rbegin() + 1) = ' '; //remove last comma
    code += "\t>::value;\n";
    code += "}\n";

    return code;
}

std::string genDecodeFunc(const std::string& _fmtName, std::vector<std::string>& _compatibleNames, std::vector<std::string>& _scaledNames)
{
    uint32_t bitsPerCh[4];
    uint32_t order[4];
    const uint32_t flags = parseFmtSpec(_fmtName, bitsPerCh, order);
    const uint32_t fmtBitSz = std::accumulate(bitsPerCh, bitsPerCh+4u, (uint32_t)0u);
    const uint32_t validChCnt = [bitsPerCh]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bitsPerCh[i]; return r; }();

    if (!(flags & EFS_FLOAT) && !(flags & EFS_NORMALIZED))
        return "";

    if (flags & EFS_SCALED)
        _scaledNames.push_back(_fmtName);
    else
        _compatibleNames.push_back(_fmtName);

    if (flags & EFS_PACKED)
    {
        uint32_t o[4];
        for (uint32_t i = 0u; i < validChCnt; ++i)
            o[i] = order[validChCnt - 1u - i];
        memcpy(order, o, 16);
    }
    bool loop = false;
    const std::string casttype = getCastType(flags, bitsPerCh, loop);
    if (!casttype.size())
        return "";

    char buf[0xff];
    constexpr const char* PROTO_fmt = 
        "template<>\n"
        "inline void decodePixels<%s, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY";
    std::sprintf(buf, PROTO_fmt, _fmtName.c_str());

    std::string code = buf;
    if (flags & EFS_SCALED)
        code += ", uint64_t _scale)";
    else
        code += ")";
    code += "\n{\n";

    if (!loop)
    {
        code += "\tconst " + casttype + "& pix = reinterpret_cast<const " + casttype + "*>(_pix[0])[0];\n";
        uint32_t sum = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) //packed only here, srgb also only here (8bit per ch)
        {
            if (order[i] > 3u)
                break;
            const uint32_t bits = bitsPerCh[order[i]];
            std::stringstream ss;
            ss << std::hex << uint64_t{ (1ull<<bits)-1ull };
            const std::string bitsHex = "0x" + ss.str() + ((flags & EFS_UNSIGNED) ? "ULL" : "LL");
            code += std::string("\t_output[") + char{ (/*flags & EFS_PACKED*/false ? char(validChCnt - 1u - order[i]) : char(order[i])) + '0' } +"] = ((pix>>" + std::to_string(sum) + ") & " + bitsHex + ")";
            if ((flags & EFS_NORMALIZED) || (flags & EFS_SRGB))
            {
                code += " / " + std::to_string((1ull << (bits - (!!(flags & EFS_SIGNED))*1u)) - 1u) + ".";
            }
            if (flags & EFS_SRGB)
            {
                code += ";\n";
                code += std::string("\t{\n\tdouble& lin = _output[") + char((/*flags & EFS_PACKED*/false ? char(validChCnt - 1u - order[i]) : char(order[i])) + '0') + "];\n";
                code += "\tif (lin <= 0.4045) lin /= 12.92;\n\telse lin = pow((lin + 0.055)/1.055, 2.4);\n\t}";
            }
            if (flags & EFS_SCALED) // SCALED implies NORMALIZED so it's already divided
            {
                code += " * _scale";
            }
            code += ";";
            if (i < 3u && order[i+1u] < 4u)
                code += "\n";
            sum += bits;
        }
    }
    else
    {
        code += "\tconst " + casttype + "* pix = reinterpret_cast<const " + casttype + "*>(_pix);\n";
        code += "\tfor (uint32_t i = 0u; i < " + std::to_string(validChCnt) + "u; ++i)\n";
        code += "\t\t_output[i] = pix[i];";
    }

    code += "\n}\n";

    return code;
}
std::string genDecodeFuncI64(const std::string& _fmtName, std::vector<std::string>& _compatibleNames)
{
    uint32_t bitsPerCh[4];
    uint32_t order[4];
    const uint32_t flags = parseFmtSpec(_fmtName, bitsPerCh, order);
    const uint32_t fmtBitSz = std::accumulate(bitsPerCh, bitsPerCh + 4u, (uint32_t)0u);
    const uint32_t validChCnt = [bitsPerCh]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bitsPerCh[i]; return r; }();

    if (!(flags & EFS_SIGNED) || (flags & EFS_FLOAT) || (flags & EFS_NORMALIZED))
        return "";

    _compatibleNames.push_back(_fmtName);

    if (flags & EFS_PACKED)
    {
        uint32_t o[4];
        for (uint32_t i = 0u; i < validChCnt; ++i)
            o[i] = order[validChCnt - 1u - i];
        memcpy(order, o, 16);
    }
    bool loop = false;
    const std::string casttype = getCastType(flags, bitsPerCh, loop);
    if (!casttype.size())
        return "";

    char buf[0xff];
    constexpr const char* PROTO_fmt =
        "template<>\n"
        "inline void decodePixels<%s, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)";
    std::sprintf(buf, PROTO_fmt, _fmtName.c_str());

    std::string code = buf;
    code += "\n{\n";

    if (!loop)
    {
        code += "\tconst " + casttype + "& pix = reinterpret_cast<const " + casttype + "*>(_pix[0])[0];\n";
        uint32_t sum = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) //packed only here, srgb also only here (8bit per ch)
        {
            if (order[i] > 3u)
                break;
            const uint32_t bits = bitsPerCh[order[i]];
            std::stringstream ss;
            ss << std::hex << uint64_t{ (1ull << bits) - 1ull };
            const std::string bitsHex = "0x" + ss.str() + ((flags & EFS_UNSIGNED) ? "ULL" : "LL");
            code += std::string("\t_output[") + char(order[i] + '0') +"] = ((pix>>" + std::to_string(sum) + ") & " + bitsHex + ")";
            if (flags & EFS_SRGB)
            {
                code += ";\n";
                code += std::string("\t{\n\tdouble lin = _output[") + char(order[i] + '0') + "] / 255.;\n";
                code += "\tif (lin <= 0.4045) lin /= 12.92;\n\telse lin = pow((lin + 0.055)/1.055, 2.4);\n";
                code += std::string("\t_output[") + char(order[i] + '0') + "] = lin * 255.;\n\t}";
            }
            code += ";";
            if (i < 3u && order[i + 1u] < 4u)
                code += "\n";
            sum += bits;
        }
    }
    else
    {
        code += "\tconst " + casttype + "* pix = reinterpret_cast<const " + casttype + "*>(_pix);\n";
        code += "\tfor (uint32_t i = 0u; i < " + std::to_string(validChCnt) + "u; ++i)\n";
        code += "\t\t_output[i] = pix[i];";
    }

    code += "\n}\n";

    return code;
}
std::string genDecodeFuncU64(const std::string& _fmtName, std::vector<std::string>& _compatibleNames)
{
    uint32_t bitsPerCh[4];
    uint32_t order[4];
    const uint32_t flags = parseFmtSpec(_fmtName, bitsPerCh, order);
    const uint32_t fmtBitSz = std::accumulate(bitsPerCh, bitsPerCh + 4u, (uint32_t)0u);
    const uint32_t validChCnt = [bitsPerCh]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bitsPerCh[i]; return r; }();

    if ((flags & EFS_SIGNED) || (flags & EFS_FLOAT) || (flags & EFS_NORMALIZED))
        return "";

    _compatibleNames.push_back(_fmtName);

    if (flags & EFS_PACKED)
    {
        uint32_t o[4];
        for (uint32_t i = 0u; i < validChCnt; ++i)
            o[i] = order[validChCnt - 1u - i];
        memcpy(order, o, 16);
    }
    bool loop = false;
    const std::string casttype = getCastType(flags, bitsPerCh, loop);
    if (!casttype.size())
        return "";

    char buf[0xff];
    constexpr const char* PROTO_fmt =
        "template<>\n"
        "inline void decodePixels<%s, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)";
    std::sprintf(buf, PROTO_fmt, _fmtName.c_str());

    std::string code = buf;
    code += "\n{\n";

    if (!loop)
    {
        code += "\tconst " + casttype + "& pix = reinterpret_cast<const " + casttype + "*>(_pix[0])[0];\n";
        uint32_t sum = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) //packed only here, srgb also only here (8bit per ch)
        {
            if (order[i] > 3u)
                break;
            const uint32_t bits = bitsPerCh[order[i]];
            std::stringstream ss;
            ss << std::hex << uint64_t{ (1ull << bits) - 1ull };
            const std::string bitsHex = "0x" + ss.str() + ((flags & EFS_UNSIGNED) ? "ULL" : "LL");
            code += std::string("\t_output[") + char(order[i] + '0') + "] = ((pix>>" + std::to_string(sum) + ") & " + bitsHex + ")";
            if (flags & EFS_SRGB)
            {
                code += ";\n";
                code += std::string("\t{\n\tdouble lin = _output[") + char(order[i] + '0') + "] / 255.;\n";
                code += "\tif (lin <= 0.4045) lin /= 12.92;\n\telse lin = pow((lin + 0.055)/1.055, 2.4);\n";
                code += std::string("\t_output[") + char(order[i] + '0') + "] = lin * 255.;\n\t}";
            }
            code += ";";
            if (i < 3u && order[i + 1u] < 4u)
                code += "\n";
            sum += bits;
        }
    }
    else
    {
        code += "\tconst " + casttype + "* pix = reinterpret_cast<const " + casttype + "*>(_pix);\n";
        code += "\tfor (uint32_t i = 0u; i < " + std::to_string(validChCnt) + "u; ++i)\n";
        code += "\t\t_output[i] = pix[i];";
    }

    code += "\n}\n";

    return code;
}

std::string genEncodeFunc(const std::string& _fmtName, std::vector<std::string>& _compatibleNames, std::vector<std::string>& _scaledNames)
{
    uint32_t bitsPerCh[4];
    uint32_t order[4];
    const uint32_t flags = parseFmtSpec(_fmtName, bitsPerCh, order);
    const uint32_t fmtBitSz = std::accumulate(bitsPerCh, bitsPerCh + 4u, (uint32_t)0u);
    const uint32_t validChCnt = [bitsPerCh]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bitsPerCh[i]; return r; }();

    if (!(flags & EFS_FLOAT) && !(flags & EFS_NORMALIZED))
        return "";

    if (flags & EFS_SCALED)
        _scaledNames.push_back(_fmtName);
    else
        _compatibleNames.push_back(_fmtName);

    if (flags & EFS_PACKED)
    {
        uint32_t o[4];
        for (uint32_t i = 0u; i < validChCnt; ++i)
            o[i] = order[validChCnt - 1u - i];
        memcpy(order, o, 16);
    }
    bool loop = false;
    const std::string casttype = getCastType(flags, bitsPerCh, loop);
    if (!casttype.size())
        return "";

    char buf[0xff];
    constexpr const char* PROTO_fmt =
        "template<>\n"
        "inline void encodePixels<%s, double>(void* _pix, const double* _input";
    std::sprintf(buf, PROTO_fmt, _fmtName.c_str());

    std::string code = buf;
    if (flags & EFS_SCALED)
        code += ", uint64_t _scale)";
    else
        code += ")";
    code += "\n{\n";

    if (!loop)
    {
        code += "\t" + casttype + "& pix = reinterpret_cast<" + casttype + "*>(_pix)[0];\n";
        uint32_t sum = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) //packed only here, srgb also only here (8bit per ch)
        {
            if (order[i] > 3u)
                break;
            const uint32_t bits = bitsPerCh[order[i]];
            std::stringstream ss;
            ss << std::hex << ((1ull<<bits)-1ull);
            const std::string maskHex = "0x" + ss.str() + ((flags & EFS_UNSIGNED) ? "U" : "") + "LL";
            code += std::string("\t{\n\tconst ") + casttype + " mask = " + maskHex + ";\n";
            code += "\tpix &= (~(mask<<" + std::to_string(sum) + "));\n";
            code += std::string("\tdouble inp = _input[") + char(char(order[i]) + '0') + "];\n";
            if (flags & EFS_SCALED)
                code += "\tinp /= _scale;\n";
            if (flags & EFS_SRGB)
            {
                code += "\tif (inp <= 0.0031308) inp *= 12.92;\n\telse inp = 1.055 * pow(inp, 1./2.4) - 0.055;\n";
            }
            if ((flags & EFS_SRGB) || (flags & EFS_NORMALIZED))
                code += "\tinp *= " + std::to_string((1ull << (bits - (!!(flags & EFS_SIGNED)) * 1u)) - 1ull) + ".;\n";
            code += std::string("\tpix |= ((uint64_t(inp) & mask) << ") + std::to_string(sum) + ");\n\t}\n";

            sum += bits;
        }
    }
    else
    {
        code += "\t" + casttype + "* pix = reinterpret_cast<" + casttype + "*>(_pix);\n";
        code += "\tfor (uint32_t i = 0u; i < " + std::to_string(validChCnt) + "u; ++i)\n";
        code += "\t\tpix[i] = _input[i];";
    }

    code += "\n}\n";

    return code;
}
std::string genEncodeFuncI64(const std::string& _fmtName, std::vector<std::string>& _compatibleNames)
{
    uint32_t bitsPerCh[4];
    uint32_t order[4];
    const uint32_t flags = parseFmtSpec(_fmtName, bitsPerCh, order);
    const uint32_t fmtBitSz = std::accumulate(bitsPerCh, bitsPerCh + 4u, (uint32_t)0u);
    const uint32_t validChCnt = [bitsPerCh]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bitsPerCh[i]; return r; }();

    if (!(flags & EFS_SIGNED) || (flags & EFS_FLOAT) || (flags & EFS_NORMALIZED))
        return "";

    _compatibleNames.push_back(_fmtName);

    if (flags & EFS_PACKED)
    {
        uint32_t o[4];
        for (uint32_t i = 0u; i < validChCnt; ++i)
            o[i] = order[validChCnt - 1u - i];
        memcpy(order, o, 16);
    }
    bool loop = false;
    const std::string casttype = getCastType(flags, bitsPerCh, loop);
    if (!casttype.size())
        return "";

    char buf[0xff];
    constexpr const char* PROTO_fmt =
        "template<>\n"
        "inline void encodePixels<%s, int64_t>(void* _pix, const int64_t* _input)";
    std::sprintf(buf, PROTO_fmt, _fmtName.c_str());

    std::string code = buf;
    code += "\n{\n";

    if (!loop)
    {
        code += "\t" + casttype + "& pix = reinterpret_cast<" + casttype + "*>(_pix)[0];\n";
        uint32_t sum = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) //packed only here, srgb also only here (8bit per ch)
        {
            if (order[i] > 3u)
                break;
            const uint32_t bits = bitsPerCh[order[i]];
            std::stringstream ss;
            ss << std::hex << ((1ull << bits) - 1ull);
            const std::string maskHex = "0x" + ss.str() + ((flags & EFS_UNSIGNED) ? "U" : "") + "LL";
            code += std::string("\t{\n\tconst ") + casttype + " mask = " + maskHex + ";\n";
            code += "\tpix &= (~(mask<<" + std::to_string(sum) + "));\n";
            code += std::string("\tint64_t inp = _input[") + char(char(order[i]) + '0') + "];\n";
            if (flags & EFS_SRGB)
            {
                code += "\tdouble s = inp / 255.;\n";
                code += "\tif (s <= 0.0031308) s *= 12.92;\n\telse s = 1.055 * pow(s, 1./2.4) - 0.055;\n";
                code += "\tinp = s * 255;\n";
            }
            code += std::string("\tpix |= ((inp & mask) << ") + std::to_string(sum) + ");\n\t}\n";

            sum += bits;
        }
    }
    else
    {
        code += "\t" + casttype + "* pix = reinterpret_cast<" + casttype + "*>(_pix);\n";
        code += "\tfor (uint32_t i = 0u; i < " + std::to_string(validChCnt) + "u; ++i)\n";
        code += "\t\tpix[i] = _input[i];";
    }

    code += "\n}\n";

    return code;
}
std::string genEncodeFuncU64(const std::string& _fmtName, std::vector<std::string>& _compatibleNames)
{
    uint32_t bitsPerCh[4];
    uint32_t order[4];
    const uint32_t flags = parseFmtSpec(_fmtName, bitsPerCh, order);
    const uint32_t fmtBitSz = std::accumulate(bitsPerCh, bitsPerCh + 4u, (uint32_t)0u);
    const uint32_t validChCnt = [bitsPerCh]() { uint32_t r = 0u; for (uint32_t i = 0u; i < 4u; ++i) r += !!bitsPerCh[i]; return r; }();

    if ((flags & EFS_SIGNED) || (flags & EFS_FLOAT) || (flags & EFS_NORMALIZED))
        return "";

    _compatibleNames.push_back(_fmtName);

    if (flags & EFS_PACKED)
    {
        uint32_t o[4];
        for (uint32_t i = 0u; i < validChCnt; ++i)
            o[i] = order[validChCnt - 1u - i];
        memcpy(order, o, 16);
    }
    bool loop = false;
    const std::string casttype = getCastType(flags, bitsPerCh, loop);
    if (!casttype.size())
        return "";

    char buf[0xff];
    constexpr const char* PROTO_fmt =
        "template<>\n"
        "inline void encodePixels<%s, uint64_t>(void* _pix, const uint64_t* _input)";
    std::sprintf(buf, PROTO_fmt, _fmtName.c_str());

    std::string code = buf;
    code += "\n{\n";

    if (!loop)
    {
        code += "\t" + casttype + "& pix = reinterpret_cast<" + casttype + "*>(_pix)[0];\n";
        uint32_t sum = 0u;
        for (uint32_t i = 0u; i < 4u; ++i) //packed only here, srgb also only here (8bit per ch)
        {
            if (order[i] > 3u)
                break;
            const uint32_t bits = bitsPerCh[order[i]];
            std::stringstream ss;
            ss << std::hex << ((1ull << bits) - 1ull);
            const std::string maskHex = "0x" + ss.str() + ((flags & EFS_UNSIGNED) ? "U" : "") + "LL";
            code += std::string("\t{\n\tconst ") + casttype + " mask = " + maskHex + ";\n";
            code += "\tpix &= (~(mask<<" + std::to_string(sum) + "));\n";
            code += std::string("\tuint64_t inp = _input[") + char(char(order[i]) + '0') + "];\n";
            if (flags & EFS_SRGB)
            {
                code += "\tdouble s = inp / 255.;\n";
                code += "\tif (s <= 0.0031308) s *= 12.92;\n\telse s = 1.055 * pow(s, 1./2.4) - 0.055;\n";
                code += "\tinp = s * 255;\n";
            }
            code += std::string("\tpix |= ((inp & mask) << ") + std::to_string(sum) + ");\n\t}\n";

            sum += bits;
        }
    }
    else
    {
        code += "\t" + casttype + "* pix = reinterpret_cast<" + casttype + "*>(_pix);\n";
        code += "\tfor (uint32_t i = 0u; i < " + std::to_string(validChCnt) + "u; ++i)\n";
        code += "\t\tpix[i] = _input[i];";
    }

    code += "\n}\n";

    return code;
}

std::string genRuntimeConvertFunc(const std::vector<std::string>& _fmts)
{
    std::string code = "inline void convertColor(ECOLOR_FORMAT _sfmt, ECOLOR_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, uint64_t _scale, size_t _pixCnt, core::vector3d<uint32_t>& _imgSize)\n{";
    code += "\tswitch (_sfmt)\n\t{\n";
    for (const std::string& fmt : _fmts)
    {
        code += "\tcase " + fmt + ": return impl::convertColor_RTimpl<" + fmt + ">(_dfmt, _srcPix, _dstPix, _scale, _pixCnt, _imgSize);\n";
    }
    code += "\t}\n}\n";

    return code;
}

std::string gen_convertColor_RTimpl(const std::vector<std::string>& _fmts)
{
    std::string code =
        "namespace impl {\n"
        "template<ECOLOR_FORMAT sF>\n"
        "inline void convertColor_RTimpl(ECOLOR_FORMAT _dfmt, const void* _srcPix[4], void* _dstPix, uint64_t _scale, size_t _pixCnt, core::vector3d<uint32_t>& _imgSize)\n{";
    code += "\tswitch (_dfmt)\n\t{\n";
    for (const std::string& fmt : _fmts)
    {
        code += "\tcase " + fmt + ": return convertColor<sF, " + fmt + ">(_srcPix, _dstPix, _scale, _pixCnt, _imgSize);\n";
    }
    code += "\t}\n}\n}//namespace impl\n";

    return code;
}
namespace impl
{
std::string genRuntimeDecodeEncodeFunc(const std::vector<std::string>& _names, const std::string& _funcName, const std::string& _outputType, const std::string& _paramList, const std::string& _fwParamList)
{
    std::string code =
        "template<>\n"
        "bool " + _funcName + "<" + _outputType + ">(ECOLOR_FORMAT _fmt, " + _paramList + ")\n"
        "{\n"
        "\tswitch(_fmt)\n"
        "\t{\n";
    for (const auto& nm : _names)
    {
        code += "\tcase " + nm + ": " + _funcName + "<" + nm + ", " + _outputType + ">(" + _fwParamList + "); return true;\n";
    }
    code += "\tdefault: return false;\n\t}\n}\n";

    return code;
}
}

std::string genRuntimeDecodeF(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "decodePixels",
        "double",
        "const void* _pix, double* _output",
        "_pix, _output"
    );
}
std::string genRuntimeDecodeScaled(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "decodePixels",
        "double",
        "const void* _pix, double* _output, uint64_t _scale",
        "_pix, _output, _scale"
    );
}
std::string genRuntimeDecodeI(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "decodePixels",
        "int64_t",
        "const void* _pix, int64_t* _output",
        "_pix, _output"
    );
}
std::string genRuntimeDecodeU(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "decodePixels",
        "uint64_t",
        "const void* _pix, uint64_t* _output",
        "_pix, _output"
    );
}

std::string genRuntimeEncodeF(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "encodePixels",
        "double",
        "void* _pix, const double* _input",
        "_pix, _input"
    );
}
std::string genRuntimeEncodeScaled(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "encodePixels",
        "double",
        "void* _pix, const double* _input, uint64_t _scale",
        "_pix, _input, _scale"
    );
}
std::string genRuntimeEncodeI(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "encodePixels",
        "int64_t",
        "void* _pix, const int64_t* _input",
        "_pix, _input"
    );
}
std::string genRuntimeEncodeU(const std::vector<std::string>& _names)
{
    return impl::genRuntimeDecodeEncodeFunc(
        _names,
        "encodePixels",
        "uint64_t",
        "void* _pix, const uint64_t* _input",
        "_pix, _input"
    );
}

int main(int _cnt, char** _args)
{
    --_cnt;
    ++_args;
    std::vector<std::string> fmts;
    for (uint32_t i = 0u; i < _cnt; ++i)
        fmts.emplace_back(_args[i]);

    std::vector<std::string> decF, decScale, decI, decU, encF, encScale, encI, encU;


    for (int i = 0; i < _cnt; ++i)
    {
        std::cout << genDecodeFunc(_args[i], decF, decScale) << genDecodeFuncI64(_args[i], decI) << genDecodeFuncU64(_args[i], decU);
        std::cout << genEncodeFunc(_args[i], encF, encScale) << genEncodeFuncI64(_args[i], encI) << genEncodeFuncU64(_args[i], encU);
    }
    /*
    std::cout << gen_convertColor_RTimpl(fmts);
    std::cout << genRuntimeConvertFunc(fmts);

    std::cout << genSizeFunc(fmts);

    std::cout << getIsIntegerFormat(fmts);
    std::cout << getIsIntegerFormatCT(fmts);

    std::cout << getIsSignedFuncCT(fmts);
    std::cout << getIsFloatingPointFormatCT(fmts);
    std::cout << getIsNormalizedFormatCT(fmts);
    std::cout << getIsScaledFormatCT(fmts);
    std::cout << getIsSRGBFormatCT(fmts);
    */

    /*for (int i = 0; i < _cnt; ++i)
    {
        std::cout << genDecodeFunc(_args[i], decF, decScale) << genDecodeFuncI64(_args[i], decI) << genDecodeFuncU64(_args[i], decU);
        std::cout << genEncodeFunc(_args[i], encF, encScale) << genEncodeFuncI64(_args[i], encI) << genEncodeFuncU64(_args[i], encU);
    }
    std::cout << genRuntimeDecodeF(decF);
    std::cout << genRuntimeDecodeI(decI);
    std::cout << genRuntimeDecodeU(decU);
    std::cout << genRuntimeDecodeScaled(decScale);
    std::cout << genRuntimeEncodeF(encF);
    std::cout << genRuntimeEncodeI(encI);
    std::cout << genRuntimeEncodeU(encU);
    std::cout << genRuntimeEncodeScaled(encScale);*/
}
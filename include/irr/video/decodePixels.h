#ifndef __IRR_DECODE_PIXELS_H_INCLUDED__
#define __IRR_DECODE_PIXELS_H_INCLUDED__

#include <type_traits>
#include <cstdint>

#include "coreutil.h"
#include "irr/asset/EFormat.h"

namespace irr { namespace video
{
	template<asset::E_FORMAT fmt, typename T>
    inline typename
    std::enable_if<
        true,//std::is_same<T, double>::value && isScaledFormat<fmt>(),
        void
    >::type
    decodePixels(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale);

	template<asset::E_FORMAT fmt, typename T>
    inline typename
    std::enable_if<
        true,//std::is_same<T, double>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value,
        void
    >::type
    decodePixels(const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY);


	template<>
    inline void decodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = (pix & 0x1fu);
        _output[1] = ((pix>>5) & 0x1fu);
        _output[2] = ((pix>>10) & 0x1fu);
        _output[3] = pix >> 15;
    }

	template<>
    inline void decodePixels<asset::EF_B5G6R5_UNORM_PACK16, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x1fULL);
        _output[1] = ((pix >> 5) & 0x3fULL);
        _output[2] = ((pix >> 11) & 0x1fULL);
    }

    template<>
    inline void decodePixels<asset::EF_R4G4_UNORM_PACK8, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix[0])[0];
        _output[1] = ((pix >> 0) & 0xfULL) / 15.;
        _output[0] = ((pix >> 4) & 0xfULL) / 15.;
    }

    template<>
    inline void decodePixels<asset::EF_R4G4B4A4_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[3] = ((pix >> 0) & 0xfULL) / 15.;
        _output[2] = ((pix >> 4) & 0xfULL) / 15.;
        _output[1] = ((pix >> 8) & 0xfULL) / 15.;
        _output[0] = ((pix >> 12) & 0xfULL) / 15.;
    }

    template<>
    inline void decodePixels<asset::EF_B4G4R4A4_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[3] = ((pix >> 0) & 0xfULL) / 15.;
        _output[0] = ((pix >> 4) & 0xfULL) / 15.;
        _output[1] = ((pix >> 8) & 0xfULL) / 15.;
        _output[2] = ((pix >> 12) & 0xfULL) / 15.;
    }

    template<>
    inline void decodePixels<asset::EF_R5G6B5_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 5) & 0x3fULL) / 63.;
        _output[0] = ((pix >> 11) & 0x1fULL) / 31.;
    }

    template<>
    inline void decodePixels<asset::EF_B5G6R5_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 5) & 0x3fULL) / 63.;
        _output[2] = ((pix >> 11) & 0x1fULL) / 31.;
    }

    template<>
    inline void decodePixels<asset::EF_R5G5B5A1_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[3] = ((pix >> 0) & 0x1ULL) / 1.;
        _output[2] = ((pix >> 1) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 6) & 0x1fULL) / 31.;
        _output[0] = ((pix >> 11) & 0x1fULL) / 31.;
    }

    template<>
    inline void decodePixels<asset::EF_B5G5R5A1_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[3] = ((pix >> 0) & 0x1ULL) / 1.;
        _output[0] = ((pix >> 1) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 6) & 0x1fULL) / 31.;
        _output[2] = ((pix >> 11) & 0x1fULL) / 31.;
    }

    template<>
    inline void decodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x1fULL) / 31.;
        _output[1] = ((pix >> 5) & 0x1fULL) / 31.;
        _output[0] = ((pix >> 10) & 0x1fULL) / 31.;
        _output[3] = ((pix >> 15) & 0x1ULL) / 1.;
    }

    template<>
    inline void decodePixels<asset::EF_R8_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_R8_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int8_t& pix = reinterpret_cast<const int8_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_R8_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int8_t& pix = reinterpret_cast<const int8_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R8_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int8_t& pix = reinterpret_cast<const int8_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[2] = ((pix >> 16) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[2] = ((pix >> 16) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[2] = ((pix >> 16) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[2] = ((pix >> 16) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[2] = ((pix >> 16) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[2] = ((pix >> 16) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[0] = ((pix >> 16) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[0] = ((pix >> 16) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[0] = ((pix >> 16) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[0] = ((pix >> 16) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[0] = ((pix >> 16) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[0] = ((pix >> 16) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[2] = ((pix >> 16) & 0xffULL) / 255.;
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[2] = ((pix >> 16) & 0xffLL) / 127.;
        _output[3] = ((pix >> 24) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[2] = ((pix >> 16) & 0xffULL) / 255. * _scale;
        _output[3] = ((pix >> 24) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[2] = ((pix >> 16) & 0xffLL) / 127. * _scale;
        _output[3] = ((pix >> 24) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[2] = ((pix >> 16) & 0xffULL);
        _output[3] = ((pix >> 24) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[2] = ((pix >> 16) & 0xffLL);
        _output[3] = ((pix >> 24) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[0] = ((pix >> 16) & 0xffULL) / 255.;
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[0] = ((pix >> 16) & 0xffLL) / 127.;
        _output[3] = ((pix >> 24) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[0] = ((pix >> 16) & 0xffULL) / 255. * _scale;
        _output[3] = ((pix >> 24) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[0] = ((pix >> 16) & 0xffLL) / 127. * _scale;
        _output[3] = ((pix >> 24) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[0] = ((pix >> 16) & 0xffULL);
        _output[3] = ((pix >> 24) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[0] = ((pix >> 16) & 0xffLL);
        _output[3] = ((pix >> 24) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_UNORM_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255.;
        _output[1] = ((pix >> 8) & 0xffULL) / 255.;
        _output[2] = ((pix >> 16) & 0xffULL) / 255.;
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_SNORM_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127.;
        _output[1] = ((pix >> 8) & 0xffLL) / 127.;
        _output[2] = ((pix >> 16) & 0xffLL) / 127.;
        _output[3] = ((pix >> 24) & 0xffLL) / 127.;
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_USCALED_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL) / 255. * _scale;
        _output[1] = ((pix >> 8) & 0xffULL) / 255. * _scale;
        _output[2] = ((pix >> 16) & 0xffULL) / 255. * _scale;
        _output[3] = ((pix >> 24) & 0xffULL) / 255. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_SSCALED_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL) / 127. * _scale;
        _output[1] = ((pix >> 8) & 0xffLL) / 127. * _scale;
        _output[2] = ((pix >> 16) & 0xffLL) / 127. * _scale;
        _output[3] = ((pix >> 24) & 0xffLL) / 127. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_UINT_PACK32, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffULL);
        _output[1] = ((pix >> 8) & 0xffULL);
        _output[2] = ((pix >> 16) & 0xffULL);
        _output[3] = ((pix >> 24) & 0xffULL);
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_SINT_PACK32, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffLL);
        _output[1] = ((pix >> 8) & 0xffLL);
        _output[2] = ((pix >> 16) & 0xffLL);
        _output[3] = ((pix >> 24) & 0xffLL);
    }

    template<>
    inline void decodePixels<asset::EF_A2R10G10B10_UNORM_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x3ffULL) / 1023.;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023.;
        _output[0] = ((pix >> 20) & 0x3ffULL) / 1023.;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3.;
    }

    template<>
    inline void decodePixels<asset::EF_A2R10G10B10_SNORM_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x3ffLL) / 511.;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511.;
        _output[0] = ((pix >> 20) & 0x3ffLL) / 511.;
        _output[3] = ((pix >> 30) & 0x3LL) / 1.;
    }

    template<>
    inline void decodePixels<asset::EF_A2R10G10B10_USCALED_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x3ffULL) / 1023. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023. * _scale;
        _output[0] = ((pix >> 20) & 0x3ffULL) / 1023. * _scale;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_A2R10G10B10_SSCALED_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x3ffLL) / 511. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511. * _scale;
        _output[0] = ((pix >> 20) & 0x3ffLL) / 511. * _scale;
        _output[3] = ((pix >> 30) & 0x3LL) / 1. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_A2R10G10B10_UINT_PACK32, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x3ffULL);
        _output[1] = ((pix >> 10) & 0x3ffULL);
        _output[0] = ((pix >> 20) & 0x3ffULL);
        _output[3] = ((pix >> 30) & 0x3ULL);
    }

    template<>
    inline void decodePixels<asset::EF_A2R10G10B10_SINT_PACK32, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[2] = ((pix >> 0) & 0x3ffLL);
        _output[1] = ((pix >> 10) & 0x3ffLL);
        _output[0] = ((pix >> 20) & 0x3ffLL);
        _output[3] = ((pix >> 30) & 0x3LL);
    }

    template<>
    inline void decodePixels<asset::EF_A2B10G10R10_UNORM_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x3ffULL) / 1023.;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023.;
        _output[2] = ((pix >> 20) & 0x3ffULL) / 1023.;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3.;
    }

    template<>
    inline void decodePixels<asset::EF_A2B10G10R10_SNORM_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x3ffLL) / 511.;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511.;
        _output[2] = ((pix >> 20) & 0x3ffLL) / 511.;
        _output[3] = ((pix >> 30) & 0x3LL) / 1.;
    }

    template<>
    inline void decodePixels<asset::EF_A2B10G10R10_USCALED_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x3ffULL) / 1023. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffULL) / 1023. * _scale;
        _output[2] = ((pix >> 20) & 0x3ffULL) / 1023. * _scale;
        _output[3] = ((pix >> 30) & 0x3ULL) / 3. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_A2B10G10R10_SSCALED_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x3ffLL) / 511. * _scale;
        _output[1] = ((pix >> 10) & 0x3ffLL) / 511. * _scale;
        _output[2] = ((pix >> 20) & 0x3ffLL) / 511. * _scale;
        _output[3] = ((pix >> 30) & 0x3LL) / 1. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_A2B10G10R10_UINT_PACK32, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x3ffULL);
        _output[1] = ((pix >> 10) & 0x3ffULL);
        _output[2] = ((pix >> 20) & 0x3ffULL);
        _output[3] = ((pix >> 30) & 0x3ULL);
    }

    template<>
    inline void decodePixels<asset::EF_A2B10G10R10_SINT_PACK32, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0x3ffLL);
        _output[1] = ((pix >> 10) & 0x3ffLL);
        _output[2] = ((pix >> 20) & 0x3ffLL);
        _output[3] = ((pix >> 30) & 0x3LL);
    }

    template<>
    inline void decodePixels<asset::EF_R16_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
    }

    template<>
    inline void decodePixels<asset::EF_R16_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
    }

    template<>
    inline void decodePixels<asset::EF_R16_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R16_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int16_t& pix = reinterpret_cast<const int16_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R16G16_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535.;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767.;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
        _output[1] = ((pix >> 16) & 0xffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R16G16_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
        _output[1] = ((pix >> 16) & 0xffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535.;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535.;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767.;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767.;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535. * _scale;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767. * _scale;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
        _output[1] = ((pix >> 16) & 0xffffULL);
        _output[2] = ((pix >> 32) & 0xffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
        _output[1] = ((pix >> 16) & 0xffffLL);
        _output[2] = ((pix >> 32) & 0xffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16A16_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535.;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535.;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535.;
        _output[3] = ((pix >> 48) & 0xffffULL) / 65535.;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16A16_SNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767.;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767.;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767.;
        _output[3] = ((pix >> 48) & 0xffffLL) / 32767.;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16A16_USCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL) / 65535. * _scale;
        _output[1] = ((pix >> 16) & 0xffffULL) / 65535. * _scale;
        _output[2] = ((pix >> 32) & 0xffffULL) / 65535. * _scale;
        _output[3] = ((pix >> 48) & 0xffffULL) / 65535. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16A16_SSCALED, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL) / 32767. * _scale;
        _output[1] = ((pix >> 16) & 0xffffLL) / 32767. * _scale;
        _output[2] = ((pix >> 32) & 0xffffLL) / 32767. * _scale;
        _output[3] = ((pix >> 48) & 0xffffLL) / 32767. * _scale;
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16A16_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffULL);
        _output[1] = ((pix >> 16) & 0xffffULL);
        _output[2] = ((pix >> 32) & 0xffffULL);
        _output[3] = ((pix >> 48) & 0xffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R16G16B16A16_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffLL);
        _output[1] = ((pix >> 16) & 0xffffLL);
        _output[2] = ((pix >> 32) & 0xffffLL);
        _output[3] = ((pix >> 48) & 0xffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R32_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R32_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t& pix = reinterpret_cast<const int32_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R32G32_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffffffULL);
        _output[1] = ((pix >> 32) & 0xffffffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R32G32_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffffffLL);
        _output[1] = ((pix >> 32) & 0xffffffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R32G32B32_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t* pix = reinterpret_cast<const uint32_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R32G32B32_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t* pix = reinterpret_cast<const int32_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R32G32B32A32_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t* pix = reinterpret_cast<const uint32_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R32G32B32A32_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int32_t* pix = reinterpret_cast<const int32_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R64_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffffffffffffffULL);
    }

    template<>
    inline void decodePixels<asset::EF_R64_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t& pix = reinterpret_cast<const int64_t*>(_pix[0])[0];
        _output[0] = ((pix >> 0) & 0xffffffffffffffffLL);
    }

    template<>
    inline void decodePixels<asset::EF_R64G64_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t* pix = reinterpret_cast<const uint64_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 2u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R64G64_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t* pix = reinterpret_cast<const int64_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 2u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R64G64B64_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t* pix = reinterpret_cast<const uint64_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R64G64B64_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t* pix = reinterpret_cast<const int64_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 3u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R64G64B64A64_UINT, uint64_t>(const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint64_t* pix = reinterpret_cast<const uint64_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }

    template<>
    inline void decodePixels<asset::EF_R64G64B64A64_SINT, int64_t>(const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const int64_t* pix = reinterpret_cast<const int64_t*>(_pix[0]);
        for (uint32_t i = 0u; i < 4u; ++i)
            _output[i] = pix[i];
    }

    namespace impl
    {
        inline double srgb2lin(double _s)
        {
            if (_s <= 0.04045) return _s / 12.92;
            return pow((_s + 0.055) / 1.055, 2.4);
        }
    }

    template<>
    inline void decodePixels<asset::EF_R8_SRGB, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint8_t& pix = reinterpret_cast<const uint8_t*>(_pix[0])[0];
        _output[0] = impl::srgb2lin(((pix >> 0) & 0xffULL) / 255.);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8_SRGB, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint16_t& pix = reinterpret_cast<const uint16_t*>(_pix[0])[0];
        _output[0] = impl::srgb2lin(((pix >> 0) & 0xffULL) / 255.);
        _output[1] = impl::srgb2lin(((pix >> 8) & 0xffULL) / 255.);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8_SRGB, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = impl::srgb2lin(((pix >> 0) & 0xffULL) / 255.);
        _output[1] = impl::srgb2lin(((pix >> 8) & 0xffULL) / 255.);
        _output[2] = impl::srgb2lin(((pix >> 16) & 0xffULL) / 255.);
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8_SRGB, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = impl::srgb2lin(((pix >> 0) & 0xffULL) / 255.);
        _output[1] = impl::srgb2lin(((pix >> 8) & 0xffULL) / 255.);
        _output[0] = impl::srgb2lin(((pix >> 16) & 0xffULL) / 255.);
    }

    template<>
    inline void decodePixels<asset::EF_R8G8B8A8_SRGB, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[0] = impl::srgb2lin(((pix >> 0) & 0xffULL) / 255.);
        _output[1] = impl::srgb2lin(((pix >> 8) & 0xffULL) / 255.);
        _output[2] = impl::srgb2lin(((pix >> 16) & 0xffULL) / 255.);
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_B8G8R8A8_SRGB, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];
        _output[2] = impl::srgb2lin(((pix >> 0) & 0xffULL) / 255.);
        _output[1] = impl::srgb2lin(((pix >> 8) & 0xffULL) / 255.);
        _output[0] = impl::srgb2lin(((pix >> 16) & 0xffULL) / 255.);
        _output[3] = ((pix >> 24) & 0xffULL) / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_A8B8G8R8_SRGB_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        decodePixels<asset::EF_R8G8B8A8_SRGB, double>(_pix, _output, _blockX, _blockY);
    }


    //Floating point formats
	namespace impl
    {
        template<typename T>
        inline void decode_r11g11b10f(const void* _pix, T* _output)
        {
            using fptr = float(*)(uint32_t);
            fptr f[3]{ &core::unpack11bitFloat, &core::unpack11bitFloat, &core::unpack10bitFloat };

            const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix)[0];
            for (uint32_t i = 0u; i < 3u; ++i)
                _output[i] = f[i](pix >> 11 * i);
        }
	}
	template<>
    inline void decodePixels<asset::EF_B10G11R11_UFLOAT_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decode_r11g11b10f<double>(_pix[0], _output);
    }

	namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void decodef16(const void* _pix, T* _output)
        {
            const uint64_t& pix = reinterpret_cast<const uint64_t*>(_pix)[0];
            for (uint32_t i = 0u; i < chCnt; ++i)
                _output[i] = core::Float16Compressor::decompress(pix >> i*16);
        }
    }
	    template<>
    inline void decodePixels<asset::EF_R16_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef16<double, 1u>(_pix[0], _output);
    }
    template<> // asset::EF_R16G16_SFLOAT gets mapped to GL_RG
    inline void decodePixels<asset::EF_R16G16_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef16<double, 2u>(_pix[0], _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<asset::EF_R16G16B16_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef16<double, 3u>(_pix[0], _output);
    }
    template<> // mapped to GL_RGBA
    inline void decodePixels<asset::EF_R16G16B16A16_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef16<double, 4u>(_pix[0], _output);
    }

    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void decodef32(const void* _pix, T* _output)
        {
            const float* pix = reinterpret_cast<const float*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                _output[i] = pix[i];
        }
    }

	template<>
    inline void decodePixels<asset::EF_R32_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef32<double, 1u>(_pix[0], _output);
    }
    template<>
    inline void decodePixels<asset::EF_R32G32_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef32<double, 2u>(_pix[0], _output);
    }
    template<>
    inline void decodePixels<asset::EF_R32G32B32_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef32<double, 3u>(_pix[0], _output);
    }
    template<>
    inline void decodePixels<asset::EF_R32G32B32A32_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef32<double, 4u>(_pix[0], _output);
    }

	namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void decodef64(const void* _pix, T* _output)
        {
            const double* pix = reinterpret_cast<const double*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                _output[i] = pix[i];
        }
    }
    template<>
    inline void decodePixels<asset::EF_R64_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef64<double, 1u>(_pix[0], _output);
    }
    template<>
    inline void decodePixels<asset::EF_R64G64_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef64<double, 2u>(_pix[0], _output);
    }
    template<>
    inline void decodePixels<asset::EF_R64G64B64_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef64<double, 3u>(_pix[0], _output);
    }
    template<>
    inline void decodePixels<asset::EF_R64G64B64A64_SFLOAT, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        impl::decodef64<double, 4u>(_pix[0], _output);
    }

    template<>
    inline void decodePixels<asset::EF_E5B9G9R9_UFLOAT_PACK32, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint32_t& pix = reinterpret_cast<const uint32_t*>(_pix[0])[0];

        uint64_t exp = static_cast<uint64_t>(pix >> 27) + (1023ull - 15ull);
        exp <<= 52;
        for (uint32_t i = 0u; i < 3u; ++i)
        {
            uint64_t out = 0u;
            out |= uint64_t((pix >> (9*i)) & 0x1ffu) << (52-9);
            out |= exp;
            memcpy(_output+i, &out, 8);
        }
    }

    // Block Compression formats
    namespace impl
    {
        template<typename T>
        inline void decodeBC1(const void* _pix, T* _output, uint32_t _x, uint32_t _y, bool _alpha)
        {
#include "irr/irrpack.h"
            struct {
                uint16_t c0, c1;
                uint32_t lut;
            } PACK_STRUCT col;
#include "irr/irrunpack.h"
            memcpy(&col, _pix, 8u);

            struct {
                union {
                    struct { uint64_t r, g, b, a; };
                    uint64_t c[4];
                };
            } p[4];

            uint16_t r0, g0, b0, r1, g1, b1;

            const void* input = &col.c0;
            decodePixels<asset::EF_B5G6R5_UNORM_PACK16, uint64_t>(&input, p[0].c, 0u, 0u);
            r0 = p[0].r;
            g0 = p[0].g;
            b0 = p[0].b;
            input = &col.c1;
            decodePixels<asset::EF_B5G6R5_UNORM_PACK16, uint64_t>(&input, p[1].c, 0u, 0u);
            r1 = p[1].r;
            g1 = p[1].g;
            b1 = p[1].b;
            if (col.c0 > col.c1)
            {
                p[2].r = (2 * r0 + 1 * r1) / 3;
                p[2].g = (2 * g0 + 1 * g1) / 3;
                p[2].b = (2 * b0 + 1 * b1) / 3;
                p[2].a = 0xff;
                p[3].r = (1 * r0 + 2 * r1) / 3;
                p[3].g = (1 * g0 + 2 * g1) / 3;
                p[3].b = (1 * b0 + 2 * b1) / 3;
                p[3].a = 0xff;
            }
            else
            {
                p[2].r = (r0 + r1) / 2;
                p[2].g = (g0 + g1) / 2;
                p[2].b = (b0 + b1) / 2;
                p[2].a = 0xff;
                p[3].r = 0;
                p[3].g = 0;
                p[3].b = 0;
                p[3].a = 0;
            }

            const uint32_t idx = 4u*_y + _x;
            const uint32_t cw = 3u & (col.lut >> (2u * idx));
            for (uint32_t i = 0u; i < (_alpha ? 4u : 3u); ++i)
                _output[i] = p[cw].c[i];
        }
        template<typename T>
        inline void decodeBC2(const void* _pix, T* _output, uint32_t _x, uint32_t _y)
        {
            const uint8_t* pix = reinterpret_cast<const uint8_t*>(_pix);
            decodeBC1(pix+8, _output, _x, _y, false);

            const uint32_t idx = 4u*_y + _x;
            const uint32_t bitI = idx * 4;
            const uint32_t byI = bitI / 8u;
            const uint32_t av = 0xfu & (pix[byI] >> (bitI & 7u));
            _output[3] = av;
        }
        template<typename T>
        inline void decodeBC4(const void* _pix, T* _output, int _offset, uint32_t _x, uint32_t _y)
        {
            struct
            {
                uint8_t a0, a1;
                uint8_t lut[6];
            } b;
            uint16_t a0, a1;
            uint8_t a[8];
            memcpy(&b, _pix, sizeof(b));

            a0 = b.a0;
            a1 = b.a1;
            a[0] = (uint8_t)a0;
            a[1] = (uint8_t)a1;
            if (a0 > a1)
            {
                a[2] = (6 * a0 + 1 * a1) / 7;
                a[3] = (5 * a0 + 2 * a1) / 7;
                a[4] = (4 * a0 + 3 * a1) / 7;
                a[5] = (3 * a0 + 4 * a1) / 7;
                a[6] = (2 * a0 + 5 * a1) / 7;
                a[7] = (1 * a0 + 6 * a1) / 7;
            }
            else
            {
                a[2] = (4 * a0 + 1 * a1) / 5;
                a[3] = (3 * a0 + 2 * a1) / 5;
                a[4] = (2 * a0 + 3 * a1) / 5;
                a[5] = (1 * a0 + 4 * a1) / 5;
                a[6] = 0;
                a[7] = 0xff;
            }

            const uint32_t idx = 4u*_y + _x;

            if (idx < 8u)
            {
                int lut = int(b.lut[0]) | int(b.lut[1] << 8) | int(b.lut[2] << 16);
                int aw = 7 & (lut >> (3 * idx));
                _output[_offset] = a[aw];
            }
            else
            {
                int lut = int(b.lut[3]) | int(b.lut[4] << 8) | int(b.lut[5] << 16);
                int aw = 7 & (lut >> (3 * idx));
                _output[_offset] = a[aw];
            }
        }

        template<typename T>
        inline void SRGB2lin(T _srgb[3]);
		
        template<typename T>
        inline void lin2SRGB(T _lin[3]);

        template<>
        inline void SRGB2lin<double>(double _srgb[3])
        {
            for (uint32_t i = 0u; i < 3u; ++i)
            {
                double& s = _srgb[i];
                if (s <= 0.04045) s /= 12.92;
                else s = std::pow((s + 0.055) / 1.055, 2.4);
            }
        }
		
        template<>
        inline void lin2SRGB<double>(double _lin[3])
        {
            for (uint32_t i = 0u; i < 3u; ++i)
            {
                double& s = _lin[i];
                if (s <= 0.0031308) s *= 12.92;
                else s = 1.055 * std::pow(s, 1./2.4) - 0.055;
            }
        }

        template<typename T>// T is int64_t or uint64_t
        inline void SRGB2lin(T _srgb[3])
        {
            double s[3] { _srgb[0]/255., _srgb[1]/255., _srgb[2]/255. };
            SRGB2lin<double>(s);
            T* lin = _srgb;
            for (uint32_t i = 0; i < 3u; ++i)
                lin[i] = s[i] * 255.;
        }

        template<typename T>
        inline void lin2SRGB(T _lin[3])
        {
            double s[3] { _lin[0]/255., _lin[1]/255., _lin[2]/255. };
            lin2SRGB<double>(s);
            T* srgb = _lin;
            for (uint32_t i = 0; i < 3u; ++i)
                srgb[i] = s[i] * 255.;
        }
    }

    template<>
    inline void decodePixels<asset::EF_BC1_RGB_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        impl::decodeBC1<double>(_pix[0], _output, _x, _y, false);
        _output[0] /= 31.;
        _output[1] /= 63.;
        _output[2] /= 31.;
    }

    template<>
    inline void decodePixels<asset::EF_BC1_RGB_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<asset::EF_BC1_RGB_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<asset::EF_BC1_RGBA_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        impl::decodeBC1<double>(_pix[0], _output, _x, _y, true);
        _output[0] /= 31.;
        _output[1] /= 63.;
        _output[2] /= 31.;
    }

    template<>
    inline void decodePixels<asset::EF_BC1_RGBA_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<asset::EF_BC1_RGBA_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<asset::EF_BC2_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        impl::decodeBC2<double>(_pix[0], _output, _x, _y);
        _output[0] /= 31.;
        _output[1] /= 63.;
        _output[2] /= 31.;
        _output[3] /= 15.;
    }

    template<>
    inline void decodePixels<asset::EF_BC2_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<asset::EF_BC2_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<asset::EF_BC3_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        const void* pix[4];
        memcpy(pix, _pix, sizeof(pix));
        pix[0] = reinterpret_cast<const uint8_t*>(pix[0])+8;
        decodePixels<asset::EF_BC1_RGBA_UNORM_BLOCK, double>(pix, _output, _x, _y);
        impl::decodeBC4(_pix, _output, 3, _x, _y);
        _output[3] /= 255.;
    }

    template<>
    inline void decodePixels<asset::EF_BC3_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        decodePixels<asset::EF_BC3_UNORM_BLOCK, double>(_pix, _output, _x, _y);
        impl::SRGB2lin(_output);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_4x4_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_5x4_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_5x5_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_6x5_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_6x6_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_8x5_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_8x6_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_8x8_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x5_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x6_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x8_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x10_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_12x10_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_12x12_UNORM_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_4x4_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_5x4_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_5x5_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_6x5_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_6x6_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_8x5_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_8x6_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_8x8_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x5_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x6_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x8_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_10x10_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_12x10_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_ASTC_12x12_SRGB_BLOCK, double>(const void* _pix[4], double* _output, uint32_t _x, uint32_t _y)
    {
        assert(0);
    }

    template<>
    inline void decodePixels<asset::EF_G8_B8_R8_3PLANE_420_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint8_t** pix = reinterpret_cast<const uint8_t**>(_pix);
        _output[0] = pix[2][0] / 255.;
        _output[1] = pix[0][0] / 255.;
        _output[2] = pix[1][0] / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_G8_B8R8_2PLANE_420_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        const uint8_t** pix = reinterpret_cast<const uint8_t**>(_pix);
        _output[0] = pix[1][1] / 255.;
        _output[1] = pix[0][0] / 255.;
        _output[2] = pix[1][0] / 255.;
    }

    template<>
    inline void decodePixels<asset::EF_G8_B8_R8_3PLANE_422_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        // decoding is same as for asset::EF_G8_B8_R8_3PLANE_420_UNORM, but iterating through pixels will look differently
        // (420 is both X and Y dimensions halved for R and B planes, in 422 we have only X dimension halved also for R and B planes)
        decodePixels<asset::EF_G8_B8_R8_3PLANE_420_UNORM, double>(_pix, _output, _blockX, _blockY);
    }

    template<>
    inline void decodePixels<asset::EF_G8_B8R8_2PLANE_422_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        // analogous thing as between asset::EF_G8_B8_R8_3PLANE_420_UNORM and asset::EF_G8_B8_R8_3PLANE_420_UNORM
        decodePixels<asset::EF_G8_B8R8_2PLANE_420_UNORM, double>(_pix, _output, _blockX, _blockY);
    }

    template<>
    inline void decodePixels<asset::EF_G8_B8_R8_3PLANE_444_UNORM, double>(const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        // also same decoding, but different pixel iteration (444 is all planes same size)
        decodePixels<asset::EF_G8_B8_R8_3PLANE_420_UNORM, double>(_pix, _output, _blockX, _blockY);
    }

	//! Runtime-given format decode
    template<typename T>
    bool decodePixels(asset::E_FORMAT _fmt, const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY);
    template<typename T>
    bool decodePixels(asset::E_FORMAT _fmt, const void* _pix[4], T* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale);

    template<>
    inline bool decodePixels<double>(asset::E_FORMAT _fmt, const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY)
    {
        switch (_fmt)
        {
        case asset::EF_R4G4_UNORM_PACK8: decodePixels<asset::EF_R4G4_UNORM_PACK8, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R4G4B4A4_UNORM_PACK16: decodePixels<asset::EF_R4G4B4A4_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B4G4R4A4_UNORM_PACK16: decodePixels<asset::EF_B4G4R4A4_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R5G6B5_UNORM_PACK16: decodePixels<asset::EF_R5G6B5_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B5G6R5_UNORM_PACK16: decodePixels<asset::EF_B5G6R5_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R5G5B5A1_UNORM_PACK16: decodePixels<asset::EF_R5G5B5A1_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B5G5R5A1_UNORM_PACK16: decodePixels<asset::EF_B5G5R5A1_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A1R5G5B5_UNORM_PACK16: decodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8_UNORM: decodePixels<asset::EF_R8_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8_SNORM: decodePixels<asset::EF_R8_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8_UNORM: decodePixels<asset::EF_R8G8_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8_SNORM: decodePixels<asset::EF_R8G8_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8_UNORM: decodePixels<asset::EF_R8G8B8_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8_SNORM: decodePixels<asset::EF_R8G8B8_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8_UNORM: decodePixels<asset::EF_B8G8R8_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8_SNORM: decodePixels<asset::EF_B8G8R8_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8A8_UNORM: decodePixels<asset::EF_R8G8B8A8_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8A8_SNORM: decodePixels<asset::EF_R8G8B8A8_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8A8_UNORM: decodePixels<asset::EF_B8G8R8A8_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8A8_SNORM: decodePixels<asset::EF_B8G8R8A8_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A8B8G8R8_UNORM_PACK32: decodePixels<asset::EF_A8B8G8R8_UNORM_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A8B8G8R8_SNORM_PACK32: decodePixels<asset::EF_A8B8G8R8_SNORM_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2R10G10B10_UNORM_PACK32: decodePixels<asset::EF_A2R10G10B10_UNORM_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2R10G10B10_SNORM_PACK32: decodePixels<asset::EF_A2R10G10B10_SNORM_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2B10G10R10_UNORM_PACK32: decodePixels<asset::EF_A2B10G10R10_UNORM_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2B10G10R10_SNORM_PACK32: decodePixels<asset::EF_A2B10G10R10_SNORM_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16_UNORM: decodePixels<asset::EF_R16_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16_SNORM: decodePixels<asset::EF_R16_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16_UNORM: decodePixels<asset::EF_R16G16_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16_SNORM: decodePixels<asset::EF_R16G16_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16_UNORM: decodePixels<asset::EF_R16G16B16_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16_SNORM: decodePixels<asset::EF_R16G16B16_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16A16_UNORM: decodePixels<asset::EF_R16G16B16A16_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16A16_SNORM: decodePixels<asset::EF_R16G16B16A16_SNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8_SRGB: decodePixels<asset::EF_R8_SRGB, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8_SRGB: decodePixels<asset::EF_R8G8_SRGB, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8_SRGB: decodePixels<asset::EF_R8G8B8_SRGB, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8_SRGB: decodePixels<asset::EF_B8G8R8_SRGB, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8A8_SRGB: decodePixels<asset::EF_R8G8B8A8_SRGB, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8A8_SRGB: decodePixels<asset::EF_B8G8R8A8_SRGB, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A8B8G8R8_SRGB_PACK32: decodePixels<asset::EF_A8B8G8R8_SRGB_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16_SFLOAT: decodePixels<asset::EF_R16_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16_SFLOAT: decodePixels<asset::EF_R16G16_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16_SFLOAT: decodePixels<asset::EF_R16G16B16_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16A16_SFLOAT: decodePixels<asset::EF_R16G16B16A16_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32_SFLOAT: decodePixels<asset::EF_R32_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32_SFLOAT: decodePixels<asset::EF_R32G32_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32B32_SFLOAT: decodePixels<asset::EF_R32G32B32_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32B32A32_SFLOAT: decodePixels<asset::EF_R32G32B32A32_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64_SFLOAT: decodePixels<asset::EF_R64_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64_SFLOAT: decodePixels<asset::EF_R64G64_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64B64_SFLOAT: decodePixels<asset::EF_R64G64B64_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64B64A64_SFLOAT: decodePixels<asset::EF_R64G64B64A64_SFLOAT, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B10G11R11_UFLOAT_PACK32: decodePixels<asset::EF_B10G11R11_UFLOAT_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_E5B9G9R9_UFLOAT_PACK32: decodePixels<asset::EF_E5B9G9R9_UFLOAT_PACK32, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC1_RGB_UNORM_BLOCK: decodePixels<asset::EF_BC1_RGB_UNORM_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC1_RGB_SRGB_BLOCK: decodePixels<asset::EF_BC1_RGB_SRGB_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC1_RGBA_UNORM_BLOCK: decodePixels<asset::EF_BC1_RGBA_UNORM_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC1_RGBA_SRGB_BLOCK: decodePixels<asset::EF_BC1_RGBA_SRGB_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC2_UNORM_BLOCK: decodePixels<asset::EF_BC2_UNORM_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC2_SRGB_BLOCK: decodePixels<asset::EF_BC2_SRGB_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC3_UNORM_BLOCK: decodePixels<asset::EF_BC3_UNORM_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_BC3_SRGB_BLOCK: decodePixels<asset::EF_BC3_SRGB_BLOCK, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_G8_B8_R8_3PLANE_420_UNORM: decodePixels<asset::EF_G8_B8_R8_3PLANE_420_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_G8_B8R8_2PLANE_420_UNORM: decodePixels<asset::EF_G8_B8R8_2PLANE_420_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_G8_B8_R8_3PLANE_422_UNORM: decodePixels<asset::EF_G8_B8_R8_3PLANE_422_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_G8_B8R8_2PLANE_422_UNORM: decodePixels<asset::EF_G8_B8R8_2PLANE_422_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_G8_B8_R8_3PLANE_444_UNORM: decodePixels<asset::EF_G8_B8_R8_3PLANE_444_UNORM, double>(_pix, _output, _blockX, _blockY); return true;
        default: return false;
        }
    }

    template<>
    inline bool decodePixels<int64_t>(asset::E_FORMAT _fmt, const void* _pix[4], int64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        switch (_fmt)
        {
        case asset::EF_R8_SINT: decodePixels<asset::EF_R8_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8_SINT: decodePixels<asset::EF_R8G8_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8_SINT: decodePixels<asset::EF_R8G8B8_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8_SINT: decodePixels<asset::EF_B8G8R8_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8A8_SINT: decodePixels<asset::EF_R8G8B8A8_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8A8_SINT: decodePixels<asset::EF_B8G8R8A8_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A8B8G8R8_SINT_PACK32: decodePixels<asset::EF_A8B8G8R8_SINT_PACK32, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2R10G10B10_SINT_PACK32: decodePixels<asset::EF_A2R10G10B10_SINT_PACK32, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2B10G10R10_SINT_PACK32: decodePixels<asset::EF_A2B10G10R10_SINT_PACK32, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16_SINT: decodePixels<asset::EF_R16_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16_SINT: decodePixels<asset::EF_R16G16_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16_SINT: decodePixels<asset::EF_R16G16B16_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16A16_SINT: decodePixels<asset::EF_R16G16B16A16_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32_SINT: decodePixels<asset::EF_R32_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32_SINT: decodePixels<asset::EF_R32G32_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32B32_SINT: decodePixels<asset::EF_R32G32B32_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32B32A32_SINT: decodePixels<asset::EF_R32G32B32A32_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64_SINT: decodePixels<asset::EF_R64_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64_SINT: decodePixels<asset::EF_R64G64_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64B64_SINT: decodePixels<asset::EF_R64G64B64_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64B64A64_SINT: decodePixels<asset::EF_R64G64B64A64_SINT, int64_t>(_pix, _output, _blockX, _blockY); return true;
        default: return false;
        }
    }
    template<>
    inline bool decodePixels<uint64_t>(asset::E_FORMAT _fmt, const void* _pix[4], uint64_t* _output, uint32_t _blockX, uint32_t _blockY)
    {
        switch (_fmt)
        {
        case asset::EF_R8_UINT: decodePixels<asset::EF_R8_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8_UINT: decodePixels<asset::EF_R8G8_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8_UINT: decodePixels<asset::EF_R8G8B8_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8_UINT: decodePixels<asset::EF_B8G8R8_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R8G8B8A8_UINT: decodePixels<asset::EF_R8G8B8A8_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_B8G8R8A8_UINT: decodePixels<asset::EF_B8G8R8A8_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A8B8G8R8_UINT_PACK32: decodePixels<asset::EF_A8B8G8R8_UINT_PACK32, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2R10G10B10_UINT_PACK32: decodePixels<asset::EF_A2R10G10B10_UINT_PACK32, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_A2B10G10R10_UINT_PACK32: decodePixels<asset::EF_A2B10G10R10_UINT_PACK32, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16_UINT: decodePixels<asset::EF_R16_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16_UINT: decodePixels<asset::EF_R16G16_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16_UINT: decodePixels<asset::EF_R16G16B16_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R16G16B16A16_UINT: decodePixels<asset::EF_R16G16B16A16_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32_UINT: decodePixels<asset::EF_R32_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32_UINT: decodePixels<asset::EF_R32G32_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32B32_UINT: decodePixels<asset::EF_R32G32B32_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R32G32B32A32_UINT: decodePixels<asset::EF_R32G32B32A32_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64_UINT: decodePixels<asset::EF_R64_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64_UINT: decodePixels<asset::EF_R64G64_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64B64_UINT: decodePixels<asset::EF_R64G64B64_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        case asset::EF_R64G64B64A64_UINT: decodePixels<asset::EF_R64G64B64A64_UINT, uint64_t>(_pix, _output, _blockX, _blockY); return true;
        default: return false;
        }
    }
    template<>
    inline bool decodePixels<double>(asset::E_FORMAT _fmt, const void* _pix[4], double* _output, uint32_t _blockX, uint32_t _blockY, uint64_t _scale)
    {
        switch (_fmt)
        {
        case asset::EF_R8_USCALED: decodePixels<asset::EF_R8_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8_SSCALED: decodePixels<asset::EF_R8_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8G8_USCALED: decodePixels<asset::EF_R8G8_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8G8_SSCALED: decodePixels<asset::EF_R8G8_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8G8B8_USCALED: decodePixels<asset::EF_R8G8B8_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8G8B8_SSCALED: decodePixels<asset::EF_R8G8B8_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_B8G8R8_USCALED: decodePixels<asset::EF_B8G8R8_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_B8G8R8_SSCALED: decodePixels<asset::EF_B8G8R8_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8G8B8A8_USCALED: decodePixels<asset::EF_R8G8B8A8_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R8G8B8A8_SSCALED: decodePixels<asset::EF_R8G8B8A8_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_B8G8R8A8_USCALED: decodePixels<asset::EF_B8G8R8A8_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_B8G8R8A8_SSCALED: decodePixels<asset::EF_B8G8R8A8_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_A8B8G8R8_USCALED_PACK32: decodePixels<asset::EF_A8B8G8R8_USCALED_PACK32, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_A8B8G8R8_SSCALED_PACK32: decodePixels<asset::EF_A8B8G8R8_SSCALED_PACK32, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_A2R10G10B10_USCALED_PACK32: decodePixels<asset::EF_A2R10G10B10_USCALED_PACK32, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_A2R10G10B10_SSCALED_PACK32: decodePixels<asset::EF_A2R10G10B10_SSCALED_PACK32, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_A2B10G10R10_USCALED_PACK32: decodePixels<asset::EF_A2B10G10R10_USCALED_PACK32, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_A2B10G10R10_SSCALED_PACK32: decodePixels<asset::EF_A2B10G10R10_SSCALED_PACK32, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16_USCALED: decodePixels<asset::EF_R16_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16_SSCALED: decodePixels<asset::EF_R16_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16G16_USCALED: decodePixels<asset::EF_R16G16_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16G16_SSCALED: decodePixels<asset::EF_R16G16_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16G16B16_USCALED: decodePixels<asset::EF_R16G16B16_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16G16B16_SSCALED: decodePixels<asset::EF_R16G16B16_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16G16B16A16_USCALED: decodePixels<asset::EF_R16G16B16A16_USCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        case asset::EF_R16G16B16A16_SSCALED: decodePixels<asset::EF_R16G16B16A16_SSCALED, double>(_pix, _output, _blockX, _blockY, _scale); return true;
        default: return false;
        }
    }

}}//irr::video

#endif //__IRR_DECODE_PIXELS_H_INCLUDED__

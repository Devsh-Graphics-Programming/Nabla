// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_ENCODE_PIXELS_H_INCLUDED__
#define __NBL_ASSET_ENCODE_PIXELS_H_INCLUDED__

#include <type_traits>
#include <cstdint>

#include "nbl/core/declarations.h"
#include "nbl/asset/format/EFormat.h"

namespace nbl
{
namespace asset
{	

    // TODO: @Crisspl move this to EFormat and give it better names
	template<typename T>
	inline constexpr uint64_t getRangeValueOfVariable(bool maxValue = true)
	{
		if (std::is_same<T, uint8_t>::value)
			return 255ull * (maxValue ? 1 : 0);
		else if (std::is_same<T, uint16_t>::value)
			return 	65535ull * (maxValue ? 1 : 0);
		else if (std::is_same<T, uint32_t>::value)
			return 	4294967295ull * (maxValue ? 1 : 0);

		else if (std::is_same<T, int8_t>::value)
			return 127ull * (maxValue ? 1 : -1);
		else if (std::is_same<T, int16_t>::value)
			return 32767ull * (maxValue ? 1 : -1);
		else if (std::is_same<T, int32_t>::value)
			return 2147483647ull * (maxValue ? 1 : -1);
		else
			return -1; // handle an error
	}

    // Only some formats use this, so its pointless kind-of
	template<asset::E_FORMAT format, typename T>
	inline void clampVariableProperly(T& variableToAssignClampingTo, const double& variableToClamp)
	{
		constexpr uint64_t max = getRangeValueOfVariable<T>(true);
		constexpr uint64_t min = getRangeValueOfVariable<T>(false);
		constexpr float epsilon = 0.4f;
		constexpr float epsilonToAddToMin = (min < 0 ? -epsilon : epsilon);

		if (nbl::asset::isNormalizedFormat(format))                                             
			variableToAssignClampingTo = static_cast<T>(core::clamp(variableToClamp * static_cast<double>(max), min + epsilonToAddToMin, max + epsilon));
		else
			variableToAssignClampingTo = static_cast<T>(core::clamp(variableToClamp, min + epsilonToAddToMin, max + epsilon));
	}

    template<asset::E_FORMAT fmt, typename T>
    inline typename 
    std::enable_if<
        true,//std::is_same<T, double>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value,
        void
    >::type
    encodePixels(void* _pix, const T* _input);
	
    template<>
    inline void encodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        pix = _input[0] & 0x1fu;
        pix |= ((_input[1] & 0x1fu) << 5);
        pix |= ((_input[2] & 0x1fu) << 10);
        pix |= uint16_t(_input[3])<<15;
    }
	
    template<>
    inline void encodePixels<asset::EF_B5G6R5_UNORM_PACK16, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        pix = _input[0] & 0x1fu;
        pix |= ((_input[1] & 0x3fu) << 5);
        pix |= ((_input[2] & 0x1fu) << 11);
    }
	
    template<>
    inline void encodePixels<asset::EF_R4G4_UNORM_PACK8, double>(void* _pix, const double* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xfULL;
            pix &= (~(mask << 0));
            double inp = _input[1];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint8_t mask = 0xfULL;
            pix &= (~(mask << 4));
            double inp = _input[0];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 4);
        }
    }
	
    template<>
    inline void encodePixels<asset::EF_R4G4B4A4_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 4));
            double inp = _input[2];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 4);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 12));
            double inp = _input[0];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 12);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B4G4R4A4_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 4));
            double inp = _input[0];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 4);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint16_t mask = 0xfULL;
            pix &= (~(mask << 12));
            double inp = _input[2];
            inp *= 15.;
            pix |= ((uint64_t(inp) & mask) << 12);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R5G6B5_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x3fULL;
            pix &= (~(mask << 5));
            double inp = _input[1];
            inp *= 63.;
            pix |= ((uint64_t(inp) & mask) << 5);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B5G6R5_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x3fULL;
            pix &= (~(mask << 5));
            double inp = _input[1];
            inp *= 63.;
            pix |= ((uint64_t(inp) & mask) << 5);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R5G5B5A1_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1ULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 1));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 1);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 6));
            double inp = _input[1];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 6);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B5G5R5A1_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1ULL;
            pix &= (~(mask << 0));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 1));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 1);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 6));
            double inp = _input[1];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 6);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 11));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 11);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 5));
            double inp = _input[1];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 5);
        }
        {
            const uint16_t mask = 0x1fULL;
            pix &= (~(mask << 10));
            double inp = _input[0];
            inp *= 31.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint16_t mask = 0x1ULL;
            pix &= (~(mask << 15));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 15);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8_UNORM, double>(void* _pix, const double* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8_SNORM, double>(void* _pix, const double* _input)
    {
        int8_t& pix = reinterpret_cast<int8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8_USCALED, double>(void* _pix, const double* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8_SSCALED, double>(void* _pix, const double* _input)
    {
        int8_t& pix = reinterpret_cast<int8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int8_t& pix = reinterpret_cast<int8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8_UNORM, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8_SNORM, double>(void* _pix, const double* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8_USCALED, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8_SSCALED, double>(void* _pix, const double* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }

    }
	
	template<>
    inline void encodePixels<asset::EF_R8G8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8_UNORM, double>(void* _pix, const double* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R8G8B8_UNORM>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8_SNORM, double>(void* _pix, const double* _input)
    {
        int8_t* pix = reinterpret_cast<int8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R8G8B8_SNORM>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8_USCALED, double>(void* _pix, const double* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R8G8B8_USCALED>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8_SSCALED, double>(void* _pix, const double* _input)
    {
        int8_t* pix = reinterpret_cast<int8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R8G8B8_SSCALED>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = static_cast<int8_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int8_t* pix = reinterpret_cast<int8_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = static_cast<int8_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8_UNORM, double>(void* _pix, const double* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_B8G8R8_UNORM>(pix[2u - i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8_SNORM, double>(void* _pix, const double* _input)
    {
        int8_t* pix = reinterpret_cast<int8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_B8G8R8_SNORM>(pix[2u - i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8_USCALED, double>(void* _pix, const double* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_B8G8R8_USCALED>(pix[2u - i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8_SSCALED, double>(void* _pix, const double* _input)
    {
        int8_t* pix = reinterpret_cast<int8_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_B8G8R8_SSCALED>(pix[2u - i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[2u-i] = static_cast<uint8_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int8_t* pix = reinterpret_cast<int8_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[2u-i] = static_cast<int8_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_USCALED, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_SSCALED, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_USCALED, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_SSCALED, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_UNORM_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_SNORM_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            inp *= 127.;
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_USCALED_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_SSCALED_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_UINT_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_SINT_PACK32, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 8));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 8);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 16);
        }
        {
            const int32_t mask = 0xffLL;
            pix &= (~(mask << 24));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 24);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2R10G10B10_UNORM_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 3.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2R10G10B10_SNORM_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2R10G10B10_USCALED_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2R10G10B10_SSCALED_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2R10G10B10_UINT_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2R10G10B10_SINT_PACK32, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2B10G10R10_UNORM_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            inp *= 1023.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 3.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2B10G10R10_SNORM_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            inp *= 511.;
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            inp *= 1.;
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2B10G10R10_USCALED_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2B10G10R10_SSCALED_PACK32, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2B10G10R10_UINT_PACK32, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 10));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const uint32_t mask = 0x3ffULL;
            pix &= (~(mask << 20));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 20);
        }
        {
            const uint32_t mask = 0x3ULL;
            pix &= (~(mask << 30));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_A2B10G10R10_SINT_PACK32, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 10));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 10);
        }
        {
            const int32_t mask = 0x3ffLL;
            pix &= (~(mask << 20));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 20);
        }
        {
            const int32_t mask = 0x3LL;
            pix &= (~(mask << 30));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 30);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16_UNORM, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16_SNORM, double>(void* _pix, const double* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16_USCALED, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16_SSCALED, double>(void* _pix, const double* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int16_t& pix = reinterpret_cast<int16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16_UNORM, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16_SNORM, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16_USCALED, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16_SSCALED, double>(void* _pix, const double* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 16);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint32_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int32_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16_UNORM, double>(void* _pix, const double* _input)
    {
        uint16_t* pix = reinterpret_cast<uint16_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R16G16B16_UNORM>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16_SNORM, double>(void* _pix, const double* _input)
    {
        int16_t* pix = reinterpret_cast<int16_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R16G16B16_SNORM>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16_USCALED, double>(void* _pix, const double* _input)
    {
        uint16_t* pix = reinterpret_cast<uint16_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R16G16B16_USCALED>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16_SSCALED, double>(void* _pix, const double* _input)
    {
        int16_t* pix = reinterpret_cast<int16_t*>(_pix);
		for (uint32_t i = 0u; i < 3u; ++i)
			clampVariableProperly<asset::EF_R16G16B16_SSCALED>(pix[i], _input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint16_t* pix = reinterpret_cast<uint16_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = static_cast<uint16_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int16_t* pix = reinterpret_cast<int16_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = static_cast<int16_t>(_input[i] * 65535.);
    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16A16_UNORM, double>(void* _pix, const double* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            inp *= 65535.;
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16A16_SNORM, double>(void* _pix, const double* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            inp *= 32767.;
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16A16_USCALED, double>(void* _pix, const double* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16A16_SSCALED, double>(void* _pix, const double* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            double inp = _input[1];
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            double inp = _input[2];
            pix |= ((uint64_t(inp) & mask) << 32);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 48));
            double inp = _input[3];
            pix |= ((uint64_t(inp) & mask) << 48);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16A16_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 16));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 32));
            uint64_t inp = _input[2];
            pix |= ((inp & mask) << 32);
        }
        {
            const uint64_t mask = 0xffffULL;
            pix &= (~(mask << 48));
            uint64_t inp = _input[3];
            pix |= ((inp & mask) << 48);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R16G16B16A16_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 16));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 16);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 32));
            int64_t inp = _input[2];
            pix |= ((inp & mask) << 32);
        }
        {
            const int64_t mask = 0xffffLL;
            pix &= (~(mask << 48));
            int64_t inp = _input[3];
            pix |= ((inp & mask) << 48);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t& pix = reinterpret_cast<int32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffffffffULL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R32G32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const uint64_t mask = 0xffffffffULL;
            pix &= (~(mask << 32));
            uint64_t inp = _input[1];
            pix |= ((inp & mask) << 32);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R32G32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const int64_t mask = 0xffffffffLL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }
        {
            const int64_t mask = 0xffffffffLL;
            pix &= (~(mask << 32));
            int64_t inp = _input[1];
            pix |= ((inp & mask) << 32);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R32G32B32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t* pix = reinterpret_cast<uint32_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = static_cast<uint32_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R32G32B32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t* pix = reinterpret_cast<int32_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = static_cast<int32_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R32G32B32A32_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint32_t* pix = reinterpret_cast<uint32_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = static_cast<uint32_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R32G32B32A32_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int32_t* pix = reinterpret_cast<int32_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = static_cast<int32_t>(_input[i]);
    }
	
    template<>
    inline void encodePixels<asset::EF_R64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t& pix = reinterpret_cast<uint64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffffffffffffffULL;
            pix &= (~(mask << 0));
            uint64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t& pix = reinterpret_cast<int64_t*>(_pix)[0];
        {
            const uint64_t mask = 0xffffffffffffffffULL;
            pix &= (~(mask << 0));
            int64_t inp = _input[0];
            pix |= ((inp & mask) << 0);
        }

    }
	
    template<>
    inline void encodePixels<asset::EF_R64G64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t* pix = reinterpret_cast<uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 2u; ++i)
            pix[i] = _input[i];
    }
	
    template<>
    inline void encodePixels<asset::EF_R64G64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t* pix = reinterpret_cast<int64_t*>(_pix);
        for (uint32_t i = 0u; i < 2u; ++i)
            pix[i] = _input[i];
    }
	
    template<>
    inline void encodePixels<asset::EF_R64G64B64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t* pix = reinterpret_cast<uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = _input[i];
    }
	
    template<>
    inline void encodePixels<asset::EF_R64G64B64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t* pix = reinterpret_cast<int64_t*>(_pix);
        for (uint32_t i = 0u; i < 3u; ++i)
            pix[i] = _input[i];
    }
	
    template<>
    inline void encodePixels<asset::EF_R64G64B64A64_UINT, uint64_t>(void* _pix, const uint64_t* _input)
    {
        uint64_t* pix = reinterpret_cast<uint64_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = _input[i];
    }
	
    template<>
    inline void encodePixels<asset::EF_R64G64B64A64_SINT, int64_t>(void* _pix, const int64_t* _input)
    {
        int64_t* pix = reinterpret_cast<int64_t*>(_pix);
        for (uint32_t i = 0u; i < 4u; ++i)
            pix[i] = _input[i];
    }

    template<>
    inline void encodePixels<asset::EF_R8_SRGB, double>(void* _pix, const double* _input)
    {
        uint8_t& pix = reinterpret_cast<uint8_t*>(_pix)[0];
        {
            const uint8_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            if (inp <= 0.0031308) inp *= 12.92;
            else inp = 1.055 * pow(inp, 1. / 2.4) - 0.055;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }

    }

    template<>
    inline void encodePixels<asset::EF_R8G8_SRGB, double>(void* _pix, const double* _input)
    {
        uint16_t& pix = reinterpret_cast<uint16_t*>(_pix)[0];
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = _input[0];
            if (inp <= 0.0031308) inp *= 12.92;
            else inp = 1.055 * pow(inp, 1. / 2.4) - 0.055;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint16_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = _input[1];
            if (inp <= 0.0031308) inp *= 12.92;
            else inp = 1.055 * pow(inp, 1. / 2.4) - 0.055;
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }

    }

    template<>
    inline void encodePixels<asset::EF_R8G8B8_SRGB, double>(void* _pix, const double* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
        {
            double inp = core::lin2srgb(_input[0]);
            inp *= 255.;
            pix[0] = static_cast<uint8_t>(inp);
        }
        {
            double inp = core::lin2srgb(_input[1]);
            inp *= 255.;
            pix[1] = static_cast<uint8_t>(inp);
        }
        {
            double inp = core::lin2srgb(_input[2]);
            inp *= 255.;
            pix[2] = static_cast<uint8_t>(inp);
        }
    }

    template<>
    inline void encodePixels<asset::EF_B8G8R8_SRGB, double>(void* _pix, const double* _input)
    {
        uint8_t* pix = reinterpret_cast<uint8_t*>(_pix);
        {
            double inp = core::lin2srgb(_input[2]);
            inp *= 255.;
            pix[0] = static_cast<uint8_t>(inp);
        }
        {
            double inp = core::lin2srgb(_input[1]);
            inp *= 255.;
            pix[1] = static_cast<uint8_t>(inp);
        }
        {
            double inp = core::lin2srgb(_input[0]);
            inp *= 255.;
            pix[2] = static_cast<uint8_t>(inp);
        }
    }

    template<>
    inline void encodePixels<asset::EF_R8G8B8A8_SRGB, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = core::lin2srgb(_input[0]);
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = core::lin2srgb(_input[1]);
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = core::lin2srgb(_input[2]);
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            pix |= ((uint64_t(_input[3]*255.) & mask) << 24);
        }

    }

    template<>
    inline void encodePixels<asset::EF_B8G8R8A8_SRGB, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 0));
            double inp = core::lin2srgb(_input[2]);
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 0);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 8));
            double inp = core::lin2srgb(_input[1]);
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 8);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 16));
            double inp = core::lin2srgb(_input[0]);
            inp *= 255.;
            pix |= ((uint64_t(inp) & mask) << 16);
        }
        {
            const uint32_t mask = 0xffULL;
            pix &= (~(mask << 24));
            pix |= ((uint64_t(_input[3]*255.) & mask) << 24);
        }

    }

    template<>
    inline void encodePixels<asset::EF_A8B8G8R8_SRGB_PACK32, double>(void* _pix, const double* _input)
    {
        encodePixels<asset::EF_R8G8B8A8_SRGB, double>(_pix, _input);
    }
	
    //Floating point formats
    namespace impl
    {
        template<typename T>
        inline void encode_r11g11b10f(void* _pix, const T* _input)
        {
            using fptr = uint32_t(*)(float);
            fptr f[3]{ &core::to11bitFloat, &core::to11bitFloat, &core::to10bitFloat};

            uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
            pix = 0u;
            for (uint32_t i = 0u; i < 3u; ++i)
                pix |= (f[i](static_cast<float>(_input[i])) << (11*i));
        }
    }
	
    template<>
    inline void encodePixels<asset::EF_B10G11R11_UFLOAT_PACK32, double>(void* _pix, const double* _input)
    {
        impl::encode_r11g11b10f<double>(_pix, _input);
    }
	
    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void encodef16(void* _pix, const T* _input)
        {
            uint16_t* pix = reinterpret_cast<uint16_t*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
            {
                pix[i] = core::Float16Compressor::compress(_input[i]);
            }
        }
    }
    template<>
    inline void encodePixels<asset::EF_R16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 1u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R16G16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 2u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<asset::EF_R16G16B16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 3u>(_pix, _input);
    }
    template<> // mapped to GL_RGBA
    inline void encodePixels<asset::EF_R16G16B16A16_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef16<double, 4u>(_pix, _input);
    }
	
    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void encodef32(void* _pix, const T* _input)
        {
            float* pix = reinterpret_cast<float*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                pix[i] = static_cast<float>(_input[i]);
        }
    }
    template<>
    inline void encodePixels<asset::EF_R32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 1u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R32G32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 2u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R32G32B32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 3u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R32G32B32A32_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef32<double, 4u>(_pix, _input);
    }
	
    namespace impl
    {
        template<typename T, uint32_t chCnt>
        inline void encodef64(void* _pix, const T* _input)
        {
            double* pix = reinterpret_cast<double*>(_pix);
            for (uint32_t i = 0u; i < chCnt; ++i)
                pix[i] = _input[i];
        }
    }
    template<>
    inline void encodePixels<asset::EF_R64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 1u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R64G64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 2u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R64G64B64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 3u>(_pix, _input);
    }
    template<>
    inline void encodePixels<asset::EF_R64G64B64A64_SFLOAT, double>(void* _pix, const double* _input)
    {
        impl::encodef64<double, 4u>(_pix, _input);
    }

    template<>
    inline void encodePixels<asset::EF_E5B9G9R9_UFLOAT_PACK32, double>(void* _pix, const double* _input)
    {
        uint32_t& pix = reinterpret_cast<uint32_t*>(_pix)[0];
        pix = 0u;
        uint32_t exp;
        {
            uint64_t inp;
            memcpy(&inp, _input, 8);
            inp >>= 52;
            inp &= 0x7ffull;
            inp -= (1023ull - 15ull);
            // TODO: this is wrong, need to get maximum exponent across all 3 input values
            exp = (static_cast<uint32_t>(inp) << 27);
        }
        for (uint32_t i = 0u; i < 3u; ++i)
        {
            uint64_t inp;
            memcpy(&inp, _input+i, 8);
            uint32_t m = (inp >> (52-9)) & 0x1ffu;
            pix |= (m << (9*i));
        }
        pix |= exp;
    }
	
    template<typename T>
    inline bool encodePixels(asset::E_FORMAT _fmt, void* _pix, const T* _input);
	
    template<>
    inline bool encodePixels<double>(asset::E_FORMAT _fmt, void* _pix, const double* _input)
    {
        switch (_fmt)
        {
        case asset::EF_R4G4_UNORM_PACK8: encodePixels<asset::EF_R4G4_UNORM_PACK8, double>(_pix, _input); return true;
        case asset::EF_R4G4B4A4_UNORM_PACK16: encodePixels<asset::EF_R4G4B4A4_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_B4G4R4A4_UNORM_PACK16: encodePixels<asset::EF_B4G4R4A4_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_R5G6B5_UNORM_PACK16: encodePixels<asset::EF_R5G6B5_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_B5G6R5_UNORM_PACK16: encodePixels<asset::EF_B5G6R5_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_R5G5B5A1_UNORM_PACK16: encodePixels<asset::EF_R5G5B5A1_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_B5G5R5A1_UNORM_PACK16: encodePixels<asset::EF_B5G5R5A1_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_A1R5G5B5_UNORM_PACK16: encodePixels<asset::EF_A1R5G5B5_UNORM_PACK16, double>(_pix, _input); return true;
        case asset::EF_R8_UNORM: encodePixels<asset::EF_R8_UNORM, double>(_pix, _input); return true;
        case asset::EF_R8_SNORM: encodePixels<asset::EF_R8_SNORM, double>(_pix, _input); return true;
        case asset::EF_R8G8_UNORM: encodePixels<asset::EF_R8G8_UNORM, double>(_pix, _input); return true;
        case asset::EF_R8G8_SNORM: encodePixels<asset::EF_R8G8_SNORM, double>(_pix, _input); return true;
        case asset::EF_R8G8B8_UNORM: encodePixels<asset::EF_R8G8B8_UNORM, double>(_pix, _input); return true;
        case asset::EF_R8G8B8_SNORM: encodePixels<asset::EF_R8G8B8_SNORM, double>(_pix, _input); return true;
        case asset::EF_B8G8R8_UNORM: encodePixels<asset::EF_B8G8R8_UNORM, double>(_pix, _input); return true;
        case asset::EF_B8G8R8_SNORM: encodePixels<asset::EF_B8G8R8_SNORM, double>(_pix, _input); return true;
        case asset::EF_R8G8B8A8_UNORM: encodePixels<asset::EF_R8G8B8A8_UNORM, double>(_pix, _input); return true;
        case asset::EF_R8G8B8A8_SNORM: encodePixels<asset::EF_R8G8B8A8_SNORM, double>(_pix, _input); return true;
        case asset::EF_B8G8R8A8_UNORM: encodePixels<asset::EF_B8G8R8A8_UNORM, double>(_pix, _input); return true;
        case asset::EF_B8G8R8A8_SNORM: encodePixels<asset::EF_B8G8R8A8_SNORM, double>(_pix, _input); return true;
        case asset::EF_A8B8G8R8_UNORM_PACK32: encodePixels<asset::EF_A8B8G8R8_UNORM_PACK32, double>(_pix, _input); return true;
        case asset::EF_A8B8G8R8_SNORM_PACK32: encodePixels<asset::EF_A8B8G8R8_SNORM_PACK32, double>(_pix, _input); return true;
        case asset::EF_A2R10G10B10_UNORM_PACK32: encodePixels<asset::EF_A2R10G10B10_UNORM_PACK32, double>(_pix, _input); return true;
        case asset::EF_A2R10G10B10_SNORM_PACK32: encodePixels<asset::EF_A2R10G10B10_SNORM_PACK32, double>(_pix, _input); return true;
        case asset::EF_A2B10G10R10_UNORM_PACK32: encodePixels<asset::EF_A2B10G10R10_UNORM_PACK32, double>(_pix, _input); return true;
        case asset::EF_A2B10G10R10_SNORM_PACK32: encodePixels<asset::EF_A2B10G10R10_SNORM_PACK32, double>(_pix, _input); return true;
        case asset::EF_R16_UNORM: encodePixels<asset::EF_R16_UNORM, double>(_pix, _input); return true;
        case asset::EF_R16_SNORM: encodePixels<asset::EF_R16_SNORM, double>(_pix, _input); return true;
        case asset::EF_R16G16_UNORM: encodePixels<asset::EF_R16G16_UNORM, double>(_pix, _input); return true;
        case asset::EF_R16G16_SNORM: encodePixels<asset::EF_R16G16_SNORM, double>(_pix, _input); return true;
        case asset::EF_R16G16B16_UNORM: encodePixels<asset::EF_R16G16B16_UNORM, double>(_pix, _input); return true;
        case asset::EF_R16G16B16_SNORM: encodePixels<asset::EF_R16G16B16_SNORM, double>(_pix, _input); return true;
        case asset::EF_R16G16B16A16_UNORM: encodePixels<asset::EF_R16G16B16A16_UNORM, double>(_pix, _input); return true;
        case asset::EF_R16G16B16A16_SNORM: encodePixels<asset::EF_R16G16B16A16_SNORM, double>(_pix, _input); return true;
        case asset::EF_R8_SRGB: encodePixels<asset::EF_R8_SRGB, double>(_pix, _input); return true;
        case asset::EF_R8G8_SRGB: encodePixels<asset::EF_R8G8_SRGB, double>(_pix, _input); return true;
        case asset::EF_R8G8B8_SRGB: encodePixels<asset::EF_R8G8B8_SRGB, double>(_pix, _input); return true;
        case asset::EF_B8G8R8_SRGB: encodePixels<asset::EF_B8G8R8_SRGB, double>(_pix, _input); return true;
        case asset::EF_R8G8B8A8_SRGB: encodePixels<asset::EF_R8G8B8A8_SRGB, double>(_pix, _input); return true;
        case asset::EF_B8G8R8A8_SRGB: encodePixels<asset::EF_B8G8R8A8_SRGB, double>(_pix, _input); return true;
        case asset::EF_A8B8G8R8_SRGB_PACK32: encodePixels<asset::EF_A8B8G8R8_SRGB_PACK32, double>(_pix, _input); return true;
        case asset::EF_R16_SFLOAT: encodePixels<asset::EF_R16_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R16G16_SFLOAT: encodePixels<asset::EF_R16G16_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R16G16B16_SFLOAT: encodePixels<asset::EF_R16G16B16_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R16G16B16A16_SFLOAT: encodePixels<asset::EF_R16G16B16A16_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R32_SFLOAT: encodePixels<asset::EF_R32_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R32G32_SFLOAT: encodePixels<asset::EF_R32G32_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R32G32B32_SFLOAT: encodePixels<asset::EF_R32G32B32_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R32G32B32A32_SFLOAT: encodePixels<asset::EF_R32G32B32A32_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R64_SFLOAT: encodePixels<asset::EF_R64_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R64G64_SFLOAT: encodePixels<asset::EF_R64G64_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R64G64B64_SFLOAT: encodePixels<asset::EF_R64G64B64_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_R64G64B64A64_SFLOAT: encodePixels<asset::EF_R64G64B64A64_SFLOAT, double>(_pix, _input); return true;
        case asset::EF_B10G11R11_UFLOAT_PACK32: encodePixels<asset::EF_B10G11R11_UFLOAT_PACK32, double>(_pix, _input); return true;
        case asset::EF_E5B9G9R9_UFLOAT_PACK32: encodePixels<asset::EF_E5B9G9R9_UFLOAT_PACK32, double>(_pix, _input); return true;
        default: return false;
        }
    }
    template<>
    inline bool encodePixels<int64_t>(asset::E_FORMAT _fmt, void* _pix, const int64_t* _input)
    {
        switch (_fmt)
        {
        case asset::EF_R8_SINT: encodePixels<asset::EF_R8_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R8G8_SINT: encodePixels<asset::EF_R8G8_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R8G8B8_SINT: encodePixels<asset::EF_R8G8B8_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_B8G8R8_SINT: encodePixels<asset::EF_B8G8R8_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R8G8B8A8_SINT: encodePixels<asset::EF_R8G8B8A8_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_B8G8R8A8_SINT: encodePixels<asset::EF_B8G8R8A8_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_A8B8G8R8_SINT_PACK32: encodePixels<asset::EF_A8B8G8R8_SINT_PACK32, int64_t>(_pix, _input); return true;
        case asset::EF_A2R10G10B10_SINT_PACK32: encodePixels<asset::EF_A2R10G10B10_SINT_PACK32, int64_t>(_pix, _input); return true;
        case asset::EF_A2B10G10R10_SINT_PACK32: encodePixels<asset::EF_A2B10G10R10_SINT_PACK32, int64_t>(_pix, _input); return true;
        case asset::EF_R16_SINT: encodePixels<asset::EF_R16_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R16G16_SINT: encodePixels<asset::EF_R16G16_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R16G16B16_SINT: encodePixels<asset::EF_R16G16B16_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R16G16B16A16_SINT: encodePixels<asset::EF_R16G16B16A16_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R32_SINT: encodePixels<asset::EF_R32_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R32G32_SINT: encodePixels<asset::EF_R32G32_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R32G32B32_SINT: encodePixels<asset::EF_R32G32B32_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R32G32B32A32_SINT: encodePixels<asset::EF_R32G32B32A32_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R64_SINT: encodePixels<asset::EF_R64_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R64G64_SINT: encodePixels<asset::EF_R64G64_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R64G64B64_SINT: encodePixels<asset::EF_R64G64B64_SINT, int64_t>(_pix, _input); return true;
        case asset::EF_R64G64B64A64_SINT: encodePixels<asset::EF_R64G64B64A64_SINT, int64_t>(_pix, _input); return true;
        default: return false;
        }
    }
    template<>
    inline bool encodePixels<uint64_t>(asset::E_FORMAT _fmt, void* _pix, const uint64_t* _input)
    {
        switch (_fmt)
        {
        case asset::EF_R8_UINT: encodePixels<asset::EF_R8_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R8G8_UINT: encodePixels<asset::EF_R8G8_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R8G8B8_UINT: encodePixels<asset::EF_R8G8B8_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_B8G8R8_UINT: encodePixels<asset::EF_B8G8R8_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R8G8B8A8_UINT: encodePixels<asset::EF_R8G8B8A8_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_B8G8R8A8_UINT: encodePixels<asset::EF_B8G8R8A8_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_A8B8G8R8_UINT_PACK32: encodePixels<asset::EF_A8B8G8R8_UINT_PACK32, uint64_t>(_pix, _input); return true;
        case asset::EF_A2R10G10B10_UINT_PACK32: encodePixels<asset::EF_A2R10G10B10_UINT_PACK32, uint64_t>(_pix, _input); return true;
        case asset::EF_A2B10G10R10_UINT_PACK32: encodePixels<asset::EF_A2B10G10R10_UINT_PACK32, uint64_t>(_pix, _input); return true;
        case asset::EF_R16_UINT: encodePixels<asset::EF_R16_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R16G16_UINT: encodePixels<asset::EF_R16G16_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R16G16B16_UINT: encodePixels<asset::EF_R16G16B16_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R16G16B16A16_UINT: encodePixels<asset::EF_R16G16B16A16_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R32_UINT: encodePixels<asset::EF_R32_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R32G32_UINT: encodePixels<asset::EF_R32G32_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R32G32B32_UINT: encodePixels<asset::EF_R32G32B32_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R32G32B32A32_UINT: encodePixels<asset::EF_R32G32B32A32_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R64_UINT: encodePixels<asset::EF_R64_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R64G64_UINT: encodePixels<asset::EF_R64G64_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R64G64B64_UINT: encodePixels<asset::EF_R64G64B64_UINT, uint64_t>(_pix, _input); return true;
        case asset::EF_R64G64B64A64_UINT: encodePixels<asset::EF_R64G64B64A64_UINT, uint64_t>(_pix, _input); return true;
        default: return false;
        }
    }
    

    inline void encodePixelsRuntime(asset::E_FORMAT _fmt, void* _pix, const void* _input)
    {
        if (isIntegerFormat(_fmt))
        {
            if (isSignedFormat(_fmt))
                encodePixels<int64_t>(_fmt, _pix, reinterpret_cast<const int64_t*>(_input));
            else
                encodePixels<uint64_t>(_fmt, _pix, reinterpret_cast<const uint64_t*>(_input));
        }
        else
            encodePixels<double>(_fmt, _pix, reinterpret_cast<const double*>(_input));
    }


}
}

#endif
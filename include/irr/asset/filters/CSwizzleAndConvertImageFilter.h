// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "irr/asset/filters/CSwizzleableAndDitherableFilterBase.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/format/convertColor.h"


namespace irr
{
namespace asset
{
namespace impl
{
	template<bool Clamp, typename Swizzle, typename Dither>
	class CSwizzleAndConvertImageFilterBase : public CSwizzleableAndDitherableFilterBase<Clamp, Swizzle, Dither>, public CMatchedSizeInOutImageFilterCommon
	{
		public:
			class CState : public CSwizzleableAndDitherableFilterBase<Clamp, Swizzle, Dither>::state_type, public CMatchedSizeInOutImageFilterCommon::state_type
			{
				public:
					CState() {}
					virtual ~CState() {}
			};

			using state_type = CState;

			static inline bool validate(state_type* state)
			{
				if (!CSwizzleableAndDitherableFilterBase<Clamp, Swizzle, Dither>::validate(state))
					return false;

				if (!CMatchedSizeInOutImageFilterCommon::validate(state))
					return false;

				// TODO: need to triple check it works when we finally enable this feature
				if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
					return false;

				return true;
			}
	};
}

//! Compile time CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting
*/

template<E_FORMAT inFormat=EF_UNKNOWN, E_FORMAT outFormat=EF_UNKNOWN, typename Swizzle=DefaultSwizzle, bool Clamp = false, typename Dither = IdentityDither>
class CSwizzleAndConvertImageFilter : public CImageFilter<CSwizzleAndConvertImageFilter<inFormat,outFormat,Swizzle,Clamp,Dither>>, public impl::CSwizzleAndConvertImageFilterBase<Clamp,Swizzle,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Clamp, Swizzle,Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Clamp, Swizzle, Dither>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=inFormat)
				return false;

			if (state->outImage->getCreationParameters().format!=outFormat)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			const auto blockDims = asset::getBlockDimensions(inFormat);
			#ifdef _IRR_DEBUG
				assert(blockDims.z==1u);
				assert(blockDims.w==1u);
			#endif

			typedef std::conditional<asset::isIntegerFormat<inFormat>(), uint64_t, double>::type decodeBufferType;
			typedef std::conditional<asset::isIntegerFormat<outFormat>(), uint64_t, double>::type encodeBufferType;
			
			auto perOutputRegion = [&blockDims,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				constexpr uint32_t outChannelsAmount = asset::getFormatChannelCount<outFormat>();

				auto swizzle = [&commonExecuteData,&blockDims,&state,&outChannelsAmount](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY=0u; blockY<blockDims.y; blockY++)
					for (auto blockX=0u; blockX<blockDims.x; blockX++)
					{
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY),commonExecuteData.outByteStrides);
						
						constexpr auto maxChannels = 4;
						decodeBufferType decodeBuffer[maxChannels] = {};
						encodeBufferType encodeBuffer[maxChannels] = {};

						impl::CSwizzleAndConvertImageFilterBase<Clamp, Swizzle, Dither>::onDecode<inFormat>(state, srcPix, decodeBuffer, encodeBuffer, blockX, blockY);
						impl::CSwizzleAndConvertImageFilterBase<Clamp, Swizzle, Dither>::onEncode<outFormat>(state, dstPix, encodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
					}
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
};

/*
	Runtime specialization of CSwizzleAndConvertImageFilter
*/

template<typename Swizzle, bool Clamp, typename Dither>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither> : public CImageFilter<CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>>, public impl::CSwizzleAndConvertImageFilterBase<Clamp,Swizzle,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Clamp,Swizzle,Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			return impl::CSwizzleAndConvertImageFilterBase<Clamp,Swizzle,Dither>::validate(state);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			const auto inFormat = state->inImage->getCreationParameters().format;
			const auto outFormat = state->outImage->getCreationParameters().format;
			const auto blockDims = asset::getBlockDimensions(inFormat);
			#ifdef _IRR_DEBUG
				assert(blockDims.z==1u);
				assert(blockDims.w==1u);
			#endif
			auto perOutputRegion = [&blockDims,inFormat,outFormat,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				auto swizzle = [&commonExecuteData,&blockDims,inFormat,outFormat,&state](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY=0u; blockY<blockDims.y; blockY++)
					for (auto blockX=0u; blockX<blockDims.x; blockX++)
					{
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY),commonExecuteData.outByteStrides);
						/*

						TODO: 
						Make it look like this bellow
						Move In/Out format from template params to args of function


						constexpr auto maxChannels = 4;
						decodeBufferType decodeBuffer[maxChannels] = {};
						encodeBufferType encodeBuffer[maxChannels] = {};

						impl::CSwizzleAndConvertImageFilterBase<Clamp, Swizzle, Dither>::onDecode(inFormat, state, srcPix, decodeBuffer, encodeBuffer, blockX, blockY);
						impl::CSwizzleAndConvertImageFilterBase<Clamp, Swizzle, Dither>::onEncode(outFormat, state, dstPix, encodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);

						*/
					}
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
};

template<E_FORMAT outFormat, typename Swizzle, bool Clamp, typename Dither>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,outFormat,Swizzle,Clamp,Dither> : public CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Clamp,Swizzle,Dither>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=outFormat)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			// TODO: improve later
			return CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>::execute(state);
		}
};

template<E_FORMAT inFormat, typename Swizzle, bool Clamp, typename Dither>
class CSwizzleAndConvertImageFilter<inFormat,EF_UNKNOWN,Swizzle,Clamp,Dither> : public CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Clamp,Swizzle,Dither>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=inFormat)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			// TODO: improve later
			return CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Clamp,Dither>::execute(state);
		}
};


} // end namespace asset
} // end namespace irr

#endif
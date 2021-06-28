// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/core.h"

#include <type_traits>

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleableAndDitherableFilterBase.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/format/convertColor.h"


namespace nbl
{
namespace asset
{
namespace impl
{
	template<bool Normalize, bool Clamp, typename Swizzle, typename Dither>
	class CSwizzleAndConvertImageFilterBase : public CSwizzleableAndDitherableFilterBase<Normalize, Clamp, Swizzle, Dither>, public CMatchedSizeInOutImageFilterCommon
	{
		public:
			class CState : public CSwizzleableAndDitherableFilterBase<Normalize, Clamp, Swizzle, Dither>::state_type, public CMatchedSizeInOutImageFilterCommon::state_type
			{
				public:
					CState() {}
					virtual ~CState() {}
			};

			using state_type = CState;

			static inline bool validate(state_type* state)
			{
				if (!CSwizzleableAndDitherableFilterBase<Normalize, Clamp, Swizzle, Dither>::validate(state))
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

//! Compile-time CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting
*/

template<E_FORMAT inFormat=EF_UNKNOWN, E_FORMAT outFormat=EF_UNKNOWN, typename Swizzle=DefaultSwizzle, bool Normalize = false, bool Clamp = false, typename Dither = IdentityDither>
class CSwizzleAndConvertImageFilter : public CImageFilter<CSwizzleAndConvertImageFilter<inFormat,outFormat,Swizzle,Normalize,Clamp,Dither>>, public impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp, Swizzle,Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp, Swizzle, Dither>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=inFormat)
				return false;

			if (state->outImage->getCreationParameters().format!=outFormat)
				return false;

			return true;
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			const auto blockDims = asset::getBlockDimensions(inFormat);
			#ifdef _NBL_DEBUG
				assert(blockDims.z==1u);
				assert(blockDims.w==1u);
			#endif

			typedef std::conditional<asset::isIntegerFormat<inFormat>(), uint64_t, double>::type decodeBufferType;
			typedef std::conditional<asset::isIntegerFormat<outFormat>(), uint64_t, double>::type encodeBufferType;
			
			auto perOutputRegion = [policy,&blockDims,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
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

						impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onDecode<inFormat>(state, srcPix, decodeBuffer, encodeBuffer, blockX, blockY);
						impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onEncode<outFormat>(state, dstPix, encodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
					}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}
};

//! Full-runtime specialization of CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting
*/

template<typename Swizzle, bool Normalize, bool Clamp, typename Dither>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Normalize,Clamp,Dither> : public CImageFilter<CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Normalize,Clamp,Dither>>, public impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			return impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>::validate(state);
		}
		
		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			const auto inFormat = state->inImage->getCreationParameters().format;
			const auto outFormat = state->outImage->getCreationParameters().format;
			const auto blockDims = asset::getBlockDimensions(inFormat);
			const uint32_t outChannelsAmount = asset::getFormatChannelCount(outFormat);
			#ifdef _NBL_DEBUG
				assert(blockDims.z==1u);
				assert(blockDims.w==1u);
			#endif
			auto perOutputRegion = [policy,&blockDims,inFormat,outFormat,outChannelsAmount,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				auto swizzle = [&commonExecuteData,&blockDims,inFormat,outFormat,outChannelsAmount,&state](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY=0u; blockY<blockDims.y; blockY++)
					for (auto blockX=0u; blockX<blockDims.x; blockX++)
					{
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY),commonExecuteData.outByteStrides);
				
						constexpr auto maxChannels = 4;
						double decodeBuffer[maxChannels] = {};
						double encodeBuffer[maxChannels] = {};

						impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onDecode(inFormat, state, srcPix, decodeBuffer, encodeBuffer, blockX, blockY);
						impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onEncode(outFormat, state, dstPix, encodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
					}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}
};

//! Half-runtime specialization of CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting.
	Out format compile-time template parameter provided.
*/

template<E_FORMAT outFormat, typename Swizzle, bool Normalize, bool Clamp, typename Dither>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,outFormat,Swizzle,Normalize,Clamp,Dither> : public CImageFilter<CSwizzleAndConvertImageFilter<EF_UNKNOWN,outFormat,Swizzle,Normalize,Clamp,Dither>>, public impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::validate(state))
				return false;

			if (state->outImage->getCreationParameters().format!=outFormat)
				return false;

			return true;
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			const auto inFormat = state->inImage->getCreationParameters().format;
			const auto blockDims = asset::getBlockDimensions(inFormat);
			#ifdef _NBL_DEBUG
			assert(blockDims.z == 1u);
			assert(blockDims.w == 1u);
			#endif

			typedef std::conditional<asset::isIntegerFormat<outFormat>(), uint64_t, double>::type encodeBufferType;

			auto perOutputRegion = [policy,&blockDims,inFormat,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				constexpr uint32_t outChannelsAmount = asset::getFormatChannelCount<outFormat>();

				auto swizzle = [&commonExecuteData,&blockDims,inFormat,&outChannelsAmount,&state](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData + readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY = 0u; blockY < blockDims.y; blockY++)
						for (auto blockX = 0u; blockX < blockDims.x; blockX++)
						{
							auto localOutPos = readBlockPos * blockDims + commonExecuteData.offsetDifference;
							uint8_t* dstPix = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY), commonExecuteData.outByteStrides);

							constexpr auto maxChannels = 4;
							double decodeBuffer[maxChannels] = {};
							encodeBufferType encodeBuffer[maxChannels] = {};

							impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onDecode(inFormat, state, srcPix, decodeBuffer, encodeBuffer, blockX, blockY);
							impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onEncode<outFormat>(state, dstPix, encodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
						}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state, perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}
};

//! Half-runtime specialization of CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting.
	In format compile-time template parameter provided.
*/

template<E_FORMAT inFormat, typename Swizzle, bool Normalize, bool Clamp, typename Dither>
class CSwizzleAndConvertImageFilter<inFormat,EF_UNKNOWN,Swizzle,Normalize,Clamp,Dither> : public CImageFilter<CSwizzleAndConvertImageFilter<inFormat,EF_UNKNOWN,Swizzle,Normalize,Clamp,Dither>>, public impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Normalize,Clamp,Swizzle,Dither>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=inFormat)
				return false;

			return true;
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			const auto outFormat = state->outImage->getCreationParameters().format;
			const auto blockDims = asset::getBlockDimensions(inFormat);
			const uint32_t outChannelsAmount = asset::getFormatChannelCount(outFormat);
			#ifdef _NBL_DEBUG
			assert(blockDims.z == 1u);
			assert(blockDims.w == 1u);
			#endif

			typedef std::conditional<asset::isIntegerFormat<inFormat>(), uint64_t, double>::type decodeBufferType;

			auto perOutputRegion = [policy,&blockDims,&outFormat,outChannelsAmount,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				const uint32_t outChannelsAmount = asset::getFormatChannelCount(outFormat);

				auto swizzle = [&commonExecuteData,&blockDims,&outFormat,&outChannelsAmount,&state](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData + readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY = 0u; blockY < blockDims.y; blockY++)
						for (auto blockX = 0u; blockX < blockDims.x; blockX++)
						{
							auto localOutPos = readBlockPos * blockDims + commonExecuteData.offsetDifference;
							uint8_t* dstPix = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY), commonExecuteData.outByteStrides);

							constexpr auto maxChannels = 4;
							decodeBufferType decodeBuffer[maxChannels] = {};
							double encodeBuffer[maxChannels] = {};

							impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onDecode<inFormat>(state, srcPix, decodeBuffer, encodeBuffer, blockX, blockY);
							impl::CSwizzleAndConvertImageFilterBase<Normalize, Clamp, Swizzle, Dither>::onEncode(outFormat, state, dstPix, encodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
						}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state, perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}
};


} // end namespace asset
} // end namespace nbl

#endif
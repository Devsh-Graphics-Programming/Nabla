// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED_
#define _NBL_ASSET_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include <type_traits>

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleableAndDitherableFilterBase.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/format/convertColor.h"


namespace nbl::asset
{

namespace impl
{

template<typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CSwizzleAndConvertImageFilterBase : public CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>, public CMatchedSizeInOutImageFilterCommon
{
	public:
		using base_t = CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>;
		class CState : public base_t::state_type, public CMatchedSizeInOutImageFilterCommon::state_type
		{
			public:
				CState() {}
				virtual ~CState() {}
		};

		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!base_t::validate(state))
				return false;

			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			// TODO: need to triple check it works when we finally enable this feature
			if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
				return false;

			return true;
		}

	protected:
		template<E_FORMAT kInFormat, class ExecutionPolicy, typename decodeBufferType, typename encodeBufferType>
		static inline void normalizationPrepass(E_FORMAT rInFormat, const ExecutionPolicy& policy, state_type* state, const core::vectorSIMDu32& blockDims)
		{
			if constexpr (!std::is_void_v<Normalization>)
			{			
				assert(kInFormat==EF_UNKNOWN || rInFormat==EF_UNKNOWN);
				state->normalization.template initialize<encodeBufferType>();

				auto perOutputRegion = [policy,&blockDims,&state,rInFormat](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
				{
					auto normalizePrepass = [&commonExecuteData,&blockDims,&state,rInFormat](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
					{
						constexpr auto MaxPlanes = 4;
						const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

						for (auto blockY=0u; blockY<blockDims.y; blockY++)
						for (auto blockX=0u; blockX<blockDims.x; blockX++)
						{						
							constexpr auto maxChannels = 4;
							decodeBufferType decodeBuffer[maxChannels] = {};

							if constexpr (kInFormat!=EF_UNKNOWN)
								base_t::template onDecode<kInFormat>(state, srcPix, decodeBuffer, blockX, blockY, maxChannels);
							else
								base_t::template onDecode<encodeBufferType>(rInFormat, state, srcPix, decodeBuffer, blockX, blockY, maxChannels);

							state->normalization.prepass(decodeBuffer,readBlockPos*blockDims+commonExecuteData.offsetDifferenceInTexels,blockX,blockY,4u/*TODO: figure this out*/);
						}
					};
					CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, normalizePrepass, commonExecuteData.inRegions, clip);
					return true;
				};
				CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
				state->normalization.finalize<encodeBufferType>();
			}
		}
};

}


//! Compile-time CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting
*/
template<E_FORMAT inFormat=EF_UNKNOWN, E_FORMAT outFormat=EF_UNKNOWN, typename Swizzle=DefaultSwizzle, typename Dither=IdentityDither, typename Normalization=void, bool Clamp=false>
class CSwizzleAndConvertImageFilter : public CImageFilter<CSwizzleAndConvertImageFilter<inFormat,outFormat,Swizzle,Dither,Normalization,Clamp>>, public impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>
{
	private:
		using base_t = impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>;
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename base_t::state_type;

		static inline bool validate(state_type* state)
		{
			if (!base_t::validate(state))
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

			typedef typename std::conditional<asset::isIntegerFormat<inFormat>(), uint64_t, double>::type decodeBufferType;
			typedef typename std::conditional<asset::isIntegerFormat<outFormat>(), uint64_t, double>::type encodeBufferType;
			base_t::template normalizationPrepass<inFormat,ExecutionPolicy,decodeBufferType,encodeBufferType>(EF_UNKNOWN,policy,state,blockDims);
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
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifferenceInTexels;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY),commonExecuteData.outByteStrides);
						
						constexpr auto maxChannels = 4;
						decodeBufferType decodeBuffer[maxChannels] = {};

						base_t::template onDecode<inFormat>(state, srcPix, decodeBuffer, blockX, blockY, maxChannels);
						base_t::template onEncode<outFormat>(state, dstPix, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
					}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions, clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}
};

//! Full-runtime specialization of CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting
*/
template<typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Dither,Normalization,Clamp> : public CImageFilter<CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle,Dither,Normalization,Clamp>>, public impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>
{
	private:
		using base_t = impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>;
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename base_t::state_type;

		static inline bool validate(state_type* state)
		{
			return base_t::validate(state);
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
			base_t::template normalizationPrepass<EF_UNKNOWN,ExecutionPolicy,double,double>(inFormat,policy,state,blockDims);
			auto perOutputRegion = [policy,&blockDims,inFormat,outFormat,outChannelsAmount,&state](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				auto swizzle = [&commonExecuteData,&blockDims,inFormat,outFormat,outChannelsAmount,&state](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY=0u; blockY<blockDims.y; blockY++)
					for (auto blockX=0u; blockX<blockDims.x; blockX++)
					{
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifferenceInTexels;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY),commonExecuteData.outByteStrides);
				
						constexpr auto maxChannels = 4;
						double decodeBuffer[maxChannels] = {};
						
						base_t::template onDecode(inFormat, state, srcPix, decodeBuffer, blockX, blockY, maxChannels);
						base_t::template onEncode(outFormat, state, dstPix, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
					}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions, clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}
};

//! Half-runtime specialization of CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting.
	Out format compile-time template parameter provided.
*/
template<E_FORMAT outFormat, typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,outFormat,Swizzle,Dither,Normalization,Clamp> : public CImageFilter<CSwizzleAndConvertImageFilter<EF_UNKNOWN,outFormat,Swizzle,Dither,Normalization,Clamp>>, public impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>
{
	private:
		using base_t = impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>;
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename base_t::state_type;

		static inline bool validate(state_type* state)
		{
			if (!base_t::validate(state))
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

			typedef typename std::conditional<asset::isIntegerFormat<outFormat>(), uint64_t, double>::type encodeBufferType;
			normalizationPrepass<EF_UNKNOWN,ExecutionPolicy,double,encodeBufferType>(inFormat,policy,state,blockDims);
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
						auto localOutPos = readBlockPos * blockDims + commonExecuteData.offsetDifferenceInTexels;
						uint8_t* dstPix = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY), commonExecuteData.outByteStrides);

						constexpr auto maxChannels = 4;
						double decodeBuffer[maxChannels] = {};

						base_t::template onDecode(inFormat, state, srcPix, decodeBuffer, blockX, blockY, maxChannels);
						base_t::template onEncode<outFormat>(state, dstPix, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
					}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions, clip);
				return true;
			};
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state, perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}
};

//! Half-runtime specialization of CSwizzleAndConvertImageFilter
/*
	Do a per-pixel recombination of image channels while converting.
	In format compile-time template parameter provided.
*/
template<E_FORMAT inFormat, typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CSwizzleAndConvertImageFilter<inFormat,EF_UNKNOWN,Swizzle,Dither,Normalization,Clamp> : public CImageFilter<CSwizzleAndConvertImageFilter<inFormat,EF_UNKNOWN,Swizzle,Dither,Normalization,Clamp>>, public impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>
{
	private:
		using base_t = impl::CSwizzleAndConvertImageFilterBase<Swizzle,Dither,Normalization,Clamp>;
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename base_t::state_type;

		static inline bool validate(state_type* state)
		{
			if (!base_t::validate(state))
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

			typedef typename std::conditional<asset::isIntegerFormat<inFormat>(), uint64_t, double>::type decodeBufferType;
			normalizationPrepass<inFormat,ExecutionPolicy,decodeBufferType,double>(EF_UNKNOWN,policy,state,blockDims);
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
							auto localOutPos = readBlockPos * blockDims + commonExecuteData.offsetDifferenceInTexels;
							uint8_t* dstPix = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos + core::vectorSIMDu32(blockX, blockY), commonExecuteData.outByteStrides);

							constexpr auto maxChannels = 4;
							decodeBufferType decodeBuffer[maxChannels] = {};

							base_t::template onDecode<inFormat>(state, srcPix, decodeBuffer, blockX, blockY, maxChannels);
							base_t::template onEncode(outFormat, state, dstPix, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
						}
				};
				CBasicImageFilterCommon::executePerRegion(policy, commonExecuteData.inImg, swizzle, commonExecuteData.inRegions, clip);
				return true;
			};

			state->outImage->setContentHash(IPreHashed::INVALID_HASH);

			return CMatchedSizeInOutImageFilterCommon::commonExecute(state, perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}
};


} // end namespace nbl::asset

#endif
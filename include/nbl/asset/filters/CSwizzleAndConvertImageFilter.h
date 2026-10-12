// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED_
#define _NBL_ASSET_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include <optional>
#include <type_traits>

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleableAndDitherableFilterBase.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/format/convertColor.h"


namespace nbl::asset
{

//! Moves texels between 8 bit UNORM or sRGB formats that differ only in channel count or order, every copied channel keeps its encoding so the result is exact
struct STexelReshuffle
{
		//! False when a copied channel would change its encoding, or `mapping` reads a channel `inFormat` does not have
		static inline bool create(const E_FORMAT inFormat, const E_FORMAT outFormat, const ICPUImageView::SComponentMapping& mapping, STexelReshuffle& out)
		{
			int8_t inChannelByte[4], outChannelByte[4];
			bool inSRGB, outSRGB;
			if (!getByteChannels(inFormat,inChannelByte,inSRGB) || !getByteChannels(outFormat,outChannelByte,outSRGB))
				return false;
			out.inBlockByteSize = getTexelOrBlockBytesize(inFormat);
			out.outBlockByteSize = getTexelOrBlockBytesize(outFormat);
			const auto* components = &mapping.r;
			for (uint32_t c=0u; c<4u; c++)
			{
				if (outChannelByte[c]<0)
					continue;
				const uint8_t outByte = outChannelByte[c];
				auto component = components[c];
				if (component==ICPUImageView::SComponentMapping::ES_IDENTITY)
					component = static_cast<ICPUImageView::SComponentMapping::E_SWIZZLE>(ICPUImageView::SComponentMapping::ES_R+c);
				out.index[outByte] = 0u;
				out.mask[outByte] = 0u;
				switch (component)
				{
					case ICPUImageView::SComponentMapping::ES_ZERO:
						out.constant[outByte] = 0u;
						break;
					case ICPUImageView::SComponentMapping::ES_ONE:
						out.constant[outByte] = 0xffu;
						break;
					default:
					{
						const uint32_t s = component-ICPUImageView::SComponentMapping::ES_R;
						if (s>3u || inChannelByte[s]<0 || (inSRGB && s<3u)!=(outSRGB && c<3u))
							return false;
						out.index[outByte] = inChannelByte[s];
						out.mask[outByte] = 0xffu;
						out.constant[outByte] = 0u;
						break;
					}
				}
			}
			return true;
		}

		//! Reshuffles a box of texels (`w` is the layer count) between two layouts with `IImage::SBufferCopy::getBlockStrides` strides, @see `CBasicImageFilterCommon::executePerRun`
		template<class ExecutionPolicy>
		inline void execute(ExecutionPolicy&& policy, uint8_t* const dst, const core::vectorSIMDu32& dstStrides, const uint8_t* const src, const core::vectorSIMDu32& srcStrides, const core::vectorSIMDu32& extent) const
		{
			bool identity = inBlockByteSize==outBlockByteSize;
			for (uint32_t b=0u; identity && b<outBlockByteSize; b++)
				identity = index[b]==b && mask[b]==0xffu;
			if (identity)
				return CBasicImageFilterCommon::copyBlocks(std::forward<ExecutionPolicy>(policy),dst,dstStrides,src,srcStrides,inBlockByteSize,extent);

			CBasicImageFilterCommon::executePerRun(std::forward<ExecutionPolicy>(policy),extent,srcStrides,inBlockByteSize,dstStrides,outBlockByteSize,[&](const size_t srcByteOffset, const size_t dstByteOffset, const size_t texelCount) -> void
			{
				auto dispatch = [&]<uint32_t InSize>() -> void
				{
					switch (outBlockByteSize)
					{
						case 1u: reshuffle<InSize,1u>(src+srcByteOffset,dst+dstByteOffset,texelCount); break;
						case 2u: reshuffle<InSize,2u>(src+srcByteOffset,dst+dstByteOffset,texelCount); break;
						case 3u: reshuffle<InSize,3u>(src+srcByteOffset,dst+dstByteOffset,texelCount); break;
						default: reshuffle<InSize,4u>(src+srcByteOffset,dst+dstByteOffset,texelCount); break;
					}
				};
				switch (inBlockByteSize)
				{
					case 1u: dispatch.template operator()<1u>(); break;
					case 2u: dispatch.template operator()<2u>(); break;
					case 3u: dispatch.template operator()<3u>(); break;
					default: dispatch.template operator()<4u>(); break;
				}
			});
		}

		uint32_t inBlockByteSize = 0u;
		uint32_t outBlockByteSize = 0u;
		// per output byte, the input byte it keeps under `mask`, or `constant`
		uint8_t index[4] = {};
		uint8_t mask[4] = {};
		uint8_t constant[4] = {};

	private:
		static inline bool getByteChannels(const E_FORMAT format, int8_t (&channelByte)[4], bool& srgb)
		{
			switch (format)
			{
				case EF_R8_UNORM: case EF_R8_SRGB:
					channelByte[0] = 0; channelByte[1] = -1; channelByte[2] = -1; channelByte[3] = -1;
					break;
				case EF_R8G8_UNORM: case EF_R8G8_SRGB:
					channelByte[0] = 0; channelByte[1] = 1; channelByte[2] = -1; channelByte[3] = -1;
					break;
				case EF_R8G8B8_UNORM: case EF_R8G8B8_SRGB:
					channelByte[0] = 0; channelByte[1] = 1; channelByte[2] = 2; channelByte[3] = -1;
					break;
				case EF_B8G8R8_UNORM: case EF_B8G8R8_SRGB:
					channelByte[0] = 2; channelByte[1] = 1; channelByte[2] = 0; channelByte[3] = -1;
					break;
				case EF_R8G8B8A8_UNORM: case EF_R8G8B8A8_SRGB:
					channelByte[0] = 0; channelByte[1] = 1; channelByte[2] = 2; channelByte[3] = 3;
					break;
				case EF_B8G8R8A8_UNORM: case EF_B8G8R8A8_SRGB:
					channelByte[0] = 2; channelByte[1] = 1; channelByte[2] = 0; channelByte[3] = 3;
					break;
				default:
					return false;
			}
			srgb = isSRGBFormat(format);
			return true;
		}

		template<uint32_t InSize, uint32_t OutSize>
		inline void reshuffle(const uint8_t* src, uint8_t* dst, const size_t texelCount) const
		{
			uint8_t texelIndex[OutSize], texelMask[OutSize], texelConstant[OutSize];
			std::copy_n(index,OutSize,texelIndex);
			std::copy_n(mask,OutSize,texelMask);
			std::copy_n(constant,OutSize,texelConstant);
			constexpr uint32_t LineTexels = 64u/OutSize;
			size_t t = 0u;
			for (; t+LineTexels<=texelCount; t+=LineTexels, src+=LineTexels*InSize, dst+=LineTexels*OutSize)
			{
				uint8_t line[LineTexels*OutSize];
				for (uint32_t l=0u; l<LineTexels; l++)
				for (uint32_t b=0u; b<OutSize; b++)
					line[l*OutSize+b] = (src[l*InSize+texelIndex[b]]&texelMask[b])|texelConstant[b];
				memcpy(dst,line,sizeof(line));
			}
			for (; t<texelCount; t++, src+=InSize, dst+=OutSize)
			{
				uint8_t texel[OutSize];
				for (uint32_t b=0u; b<OutSize; b++)
					texel[b] = (src[texelIndex[b]]&texelMask[b])|texelConstant[b];
				memcpy(dst,texel,OutSize);
			}
		}
};

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

		static inline void storeEncoded(void* dst, const uint64_t (&encoded)[4], const uint32_t texelByteSize)
		{
			switch (texelByteSize)
			{
				case 1u: memcpy(dst,encoded,1u); break;
				case 2u: memcpy(dst,encoded,2u); break;
				case 3u: memcpy(dst,encoded,3u); break;
				case 4u: memcpy(dst,encoded,4u); break;
				case 6u: memcpy(dst,encoded,6u); break;
				case 8u: memcpy(dst,encoded,8u); break;
				case 12u: memcpy(dst,encoded,12u); break;
				case 16u: memcpy(dst,encoded,16u); break;
				default: memcpy(dst,encoded,texelByteSize); break;
			}
		}

		template<typename SwizzleT>
		static inline std::optional<ICPUImageView::SComponentMapping> getComponentMapping(const SwizzleT& swizzle)
		{
			if constexpr (std::is_same_v<SwizzleT,DefaultSwizzle>)
				return swizzle.swizzle;
			else if constexpr (std::is_same_v<SwizzleT,VoidSwizzle>)
				return ICPUImageView::SComponentMapping{};
			else if constexpr (requires { SwizzleT::componentMapping; })
				return SwizzleT::componentMapping;
			else
				return std::nullopt;
		}

		static inline bool getReshuffle(state_type* state, STexelReshuffle& reshuffle)
		{
			if constexpr (!std::is_same_v<Dither,IdentityDither> || !std::is_void_v<Normalization> || !std::is_base_of_v<Swizzle,state_type>)
				return false;
			else
			{
				const auto mapping = getComponentMapping(static_cast<const Swizzle&>(*state));
				return mapping && STexelReshuffle::create(state->inImage->getCreationParameters().format,state->outImage->getCreationParameters().format,*mapping,reshuffle);
			}
		}

		template<class ExecutionPolicy>
		static inline bool executeReshuffle(ExecutionPolicy&& policy, state_type* state, const STexelReshuffle& reshuffle)
		{
			auto perOutputRegion = [&policy,&reshuffle](const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				const TexelBlockInfo inBlockInfo(commonExecuteData.inFormat);
				const auto outStrides = commonExecuteData.oit->getBlockStrides(TexelBlockInfo(commonExecuteData.outFormat));
				for (auto region : commonExecuteData.inRegions)
				{
					if (!clip(region,&region))
						continue;
					const core::vectorSIMDu32 readTexelPos(region.imageOffset.x,region.imageOffset.y,region.imageOffset.z,region.imageSubresource.baseArrayLayer);
					const core::vectorSIMDu32 extent(region.imageExtent.width,region.imageExtent.height,region.imageExtent.depth,region.imageSubresource.layerCount);
					uint8_t* const dst = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(readTexelPos+commonExecuteData.offsetDifferenceInBlocks,commonExecuteData.outByteStrides);
					reshuffle.execute(policy,dst,outStrides,commonExecuteData.inData+region.bufferOffset,region.getBlockStrides(inBlockInfo),extent);
				}
				return true;
			};
			state->outImage->setContentHash(IPreHashed::INVALID_HASH);
			return CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
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

			if (STexelReshuffle reshuffle; base_t::getReshuffle(state,reshuffle))
				return base_t::executeReshuffle(policy,state,reshuffle);

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
						uint64_t encoded[4] = {};
						base_t::template onEncode<outFormat>(state, encoded, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
						base_t::storeEncoded(dstPix, encoded, commonExecuteData.outBlockByteSize);
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

			if (STexelReshuffle reshuffle; base_t::getReshuffle(state,reshuffle))
				return base_t::executeReshuffle(policy,state,reshuffle);

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
						uint64_t encoded[4] = {};
						base_t::template onEncode(outFormat, state, encoded, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
						base_t::storeEncoded(dstPix, encoded, commonExecuteData.outBlockByteSize);
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

			if (STexelReshuffle reshuffle; base_t::getReshuffle(state,reshuffle))
				return base_t::executeReshuffle(policy,state,reshuffle);

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
						uint64_t encoded[4] = {};
						base_t::template onEncode<outFormat>(state, encoded, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
						base_t::storeEncoded(dstPix, encoded, commonExecuteData.outBlockByteSize);
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

			if (STexelReshuffle reshuffle; base_t::getReshuffle(state,reshuffle))
				return base_t::executeReshuffle(policy,state,reshuffle);

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
							uint64_t encoded[4] = {};
							base_t::template onEncode(outFormat, state, encoded, decodeBuffer, localOutPos, blockX, blockY, outChannelsAmount);
							base_t::storeEncoded(dstPix, encoded, commonExecuteData.outBlockByteSize);
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
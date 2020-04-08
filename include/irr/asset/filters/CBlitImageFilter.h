// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_BLIT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_BLIT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CMatchedSizeInOutImageFilterCommon.h"

#include "irr/asset/filters/kernels/kernels.h"

namespace irr
{
namespace asset
{

// copy while filtering the input into the output
template<class Kernel=CBoxImageFilterKernel>
class CBlitImageFilter : public CImageFilter<CBlitImageFilter<Kernel> >, public CBasicImageFilterCommon
{
	public:
		virtual ~CBlitImageFilter() {}
		
		using texel_arithmetic_type = typename Kernel::value_type;

		class CState : public IImageFilter::IState
		{
			public:
				CState()
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}
				virtual ~CState() {}

				enum E_ALPHA_SEMANTIC : uint32_t
				{
					EAS_NONE_OR_PREMULTIPLIED=0u, // just filter the channels independently (also works for a texture for blending equation `dstCol*(1-srcAlpha)+srcCol`)
					EAS_REFERENCE_OR_COVERAGE, // try to preserve coverage (percentage of pixels above a threshold value) across mipmap levels
					EAS_SEPARATE_BLEND, // compute a new alpha value for a texture to be used with the blending equation `mix(dstCol,srcCol,srcAlpha)`
					EAS_COUNT
				};

				union
				{
					core::vectorSIMDu32				inOffsetBaseLayer;
					struct
					{
						VkOffset3D					inOffset;
						uint32_t					inBaseLayer;
					};
				};
				union
				{
					core::vectorSIMDu32 inExtentLayerCount;
					struct
					{
						VkExtent3D					inExtent;
						uint32_t					inLayerCount;
					};
				};
				union
				{
					core::vectorSIMDu32 outOffsetBaseLayer;
					struct
					{
						VkOffset3D					outOffset;
						uint32_t					outBaseLayer;
					};
				};
				union
				{
					core::vectorSIMDu32 outExtentLayerCount;
					struct
					{
						VkExtent3D					outExtent;
						uint32_t					outLayerCount;
					};
				};
				uint32_t							inMipLevel = 0u;
				uint32_t							outMipLevel = 0u;
				ICPUImage*							inImage = nullptr;
				ICPUImage*							outImage = nullptr;
				_IRR_STATIC_INLINE_CONSTEXPR auto	NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP			axisWraps[NumWrapAxes] = { ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT };
				ISampler::E_TEXTURE_BORDER_COLOR	borderColor = ISampler::ETBC_FLOAT_TRANSPARENT_BLACK;
				Kernel								kernel;
				E_ALPHA_SEMANTIC					alphaSemantic = EAS_NONE_OR_PREMULTIPLIED;
				// the next three are required if `alphaSemantic==EAS_REFERENCE_OR_COVERAGE`
				texel_arithmetic_type*				scratchMemory = nullptr;
				uint32_t							scratchMemoryByteSize = 0u;
				texel_arithmetic_type				alphaRefValue = 0.5;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state)
				return false;

			if (state->inLayerCount!=state->outLayerCount)
				return false;

			IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->inLayerCount };
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource, {state->inOffset,state->inExtent}, state->inImage))
				return false;
			subresource.mipLevel = state->outMipLevel;
			subresource.baseArrayLayer = state->outBaseLayer;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource, {state->outOffset,state->outExtent}, state->outImage))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
				return false;

			return state->kernel.validate(inImage,outImage);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			const auto* const inImg = state->inImage;
			auto* const outImg = state->outImage;
			const ICPUImage::SCreationParams& inParams = inImg->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = outImg->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;
			const auto inBlockDims = asset::getBlockDimensions(inFormat);
			const auto outBlockDims = asset::getBlockDimensions(outFormat);
			const auto* const inData = reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer());
			auto* const outData = reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer());

			constexpr auto MaxChannels = 4;
			const auto inMipLevel = state->inMipLevel;
			const auto outMipLevel = state->outMipLevel;
			const auto inBaseLayer = state->inBaseLayer;
			const auto outBaseLayer = state->outBaseLayer;
			const auto layerCount = state->inLayerCount;

			const auto inOffset = state->inOffset;
			const auto outOffset = state->outOffset;
			const auto inExtent = state->inExtent;
			const auto outExtent = state->outExtent;


			const bool nonPremultBlendSemantic = state->alphaSemantic==CState::EAS_SEPARATE_BLEND;
			// do the magical coverage adjustment trick
			const bool coverageSemantic = state->alphaSemantic==CState::EAS_REFERENCE_OR_COVERAGE;
			const auto alphaRefValue = state->alphaRefValue;
			core::rational inverseCoverage(0);
			if (coverageSemantic)
			{
				if (outExtent.width*outExtent.height*outExtent.depth*sizeof(texel_arithmetic_type) > scratchMemoryByteSize || !state->scratchMemory)
					return false;

				auto computeCoverage = [&inverseCoverage,inData,&inBlockDims,MaxChannels,alphaRefValue](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { inData+readBlockArrayOffset,nullptr,nullptr,nullptr };
				
					for (auto blockY=0u; blockY<inBlockDims.y; blockY++)
					for (auto blockX=0u; blockX<inBlockDims.x; blockX++)
					{
						texel_arithmetic_type decbuf[MaxChannels] = {0, 0, 0, 1};
						decodePixels<texel_arithmetic_type>(inFormat,srcPix,decbuf,blockX,blockY);
						if (decbuf[3]<=alphaRefValue)
							inverseCoverage.getNumerator()++;
						inverseCoverage.getDenominator()++;
					}
				};
				const core::SRange<const IImage::SBufferCopy> inRegions = outImg->getRegions(inMipLevel);
				CBasicImageFilterCommon::clip_region_functor_t clip({static_cast<IImage::E_ASPECT_FLAGS>(0u),inMipLevel,inBaseLayer,layerCount}, {inOffset,inExtent}, inFormat);
				CBasicImageFilterCommon::executePerRegion(inImg,computeCoverage,inRegions.begin(),inRegions.end(),clip);
			}

			auto* filteredAlphaArray = state->scratchMemory;
			auto* filteredAlphaArrayIt = filteredAlphaArray;
			const Kernel& kernel = state->kernel;
			auto blit = [MaxChannels,outData,kernel,nonPremultBlendSemantic,coverageSemantic,outFormat](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
			{
				constexpr auto MaxPlanes = 4;
				void* dstPix = outData+readBlockArrayOffset;
				texel_arithmetic_type valbuf[MaxTexelBlockDimensions.y*MaxTexelBlockDimensions.x][MaxChannels] = {0};
				for (auto blockY=0u; blockY<outBlockDims.y; blockY++)
				for (auto blockX=0u; blockX<outBlockDims.x; blockX++)
				{
					auto* value = valbuf[blockY*outBlockDims.x+blockX];
					texel_arithmetic_type avgColor = 0;

					// TODO: Sampling
					//for ()
					//for ()
					//for ()
					{
						texel_arithmetic_type sample[MaxChannels] = {0,0,0,1};

						for (auto i=0; i<MaxChannels; i++)
							value[i] += sample[i];

						if (nonPremultBlendSemantic)
						for (auto i=0; i<MaxChannels; i++)
							avgColor += sample[i]*sample[3];
					}

					if (nonPremultBlendSemantic && avgColor>FLT_MIN*1024.0*512.0)
						value[3] = avgColor/(value[0]+value[1]+value[2]);
					else if (coverageSemantic)
						*(filteredAlphaArrayIt++) = value[3];
				}
				asset::encodePixels<texel_arithmetic_type>(outFormat,dstPix,valbuf[0]);
			};
			const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(outMipLevel);
			CBasicImageFilterCommon::clip_region_functor_t clip({static_cast<IImage::E_ASPECT_FLAGS>(0u),outMipLevel,outBaseLayer,layerCount}, {outOffset,outExtent}, outFormat);
			CBasicImageFilterCommon::executePerRegion(outImg,blit,outRegions.begin(),outRegions.end(),clip);
			if (coverageSemantic)
			{
				// sort so that we can easily find the alpha value s.t. % of all texels is less than it
				std::sort(filteredAlphaArray,filteredAlphaArrayIt);
				auto outputTexelCount = std::distance(filteredAlphaArray,filteredAlphaArrayIt);
				// all values with index < rankIndex will be %==inverseCoverage of the overall array
				int32_t rankIndex = (inverseCoverage*core::rational<int32_t>(outputTexelCount)).getIntegerApprox();
				rankIndex--; // now all with index<=rankIndex
				// this is our new reference value
				auto newRefValue = filteredAlphaArray[core::max(rankIndex,0)];
				// scale all alpha texels to work with new reference value
				auto coverageScale = alphaRefValue/newRefValue;
				auto scaleCoverage = [outData,outBlockDims,MaxChannels,inFormat,coverageScale](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					constexpr auto MaxPlanes = 4;
					void* dstPix = outData+readBlockArrayOffset;
					const void* srcPix[MaxPlanes] = { dstPix,nullptr,nullptr,nullptr };

					texel_arithmetic_type valbuf[MaxTexelBlockDimensions.y*MaxTexelBlockDimensions.x][MaxChannels];
					for (auto blockY=0u; blockY<outBlockDims.y; blockY++)
					for (auto blockX=0u; blockX<outBlockDims.x; blockX++)
					{
						auto decbuf = valbuf[blockY*outBlockDims.x+blockX] = { 0,0,0,1 };
						decodePixels<texel_arithmetic_type>(inFormat,srcPix,decbuf,blockX,blockY);
						decbuf[3] *= coverageScale;
					}
					encodePixels<texel_arithmetic_type>(inFormat,dstPix,valbuf[0]);
				};
				CBasicImageFilterCommon::executePerRegion(outImg,scaleCoverage,outRegions.begin(),outRegions.end(),clip);
			}
			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif
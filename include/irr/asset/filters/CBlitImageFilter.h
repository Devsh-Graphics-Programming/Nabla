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


class CBlitImageFilterBase : public CBasicImageFilterCommon
{
	public:
		class CStateBase
		{
			public:
				enum E_ALPHA_SEMANTIC : uint32_t
				{
					EAS_NONE_OR_PREMULTIPLIED = 0u, // just filter the channels independently (also works for a texture for blending equation `dstCol*(1-srcAlpha)+srcCol`)
					EAS_REFERENCE_OR_COVERAGE, // try to preserve coverage (percentage of pixels above a threshold value) across mipmap levels
					EAS_SEPARATE_BLEND, // compute a new alpha value for a texture to be used with the blending equation `mix(dstCol,srcCol,srcAlpha)`
					EAS_COUNT
				};

				uint8_t*							scratchMemory = nullptr;
				uint32_t							scratchMemoryByteSize = 0u;
				_IRR_STATIC_INLINE_CONSTEXPR auto	NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP			axisWraps[NumWrapAxes] = { ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT };
				ISampler::E_TEXTURE_BORDER_COLOR	borderColor = ISampler::ETBC_FLOAT_TRANSPARENT_BLACK;
				E_ALPHA_SEMANTIC					alphaSemantic = EAS_NONE_OR_PREMULTIPLIED;
				double								alphaRefValue = 0.5; // only required to make sense if `alphaSemantic==EAS_REFERENCE_OR_COVERAGE`
		};

		template<class Kernel>
		static inline uint32_t getRequiredScratchByteSize(	const Kernel& k,
															typename CStateBase::E_ALPHA_SEMANTIC alphaSemantic=CStateBase::EAS_NONE_OR_PREMULTIPLIED,
															const core::vectorSIMDu32& outExtentLayerCount=core::vectorSIMDu32(0,0,0,0))
		{
			uint32_t retval = 0u;
			if (alphaSemantic==CStateBase::EAS_REFERENCE_OR_COVERAGE)
			{
				// no mul by channel count because we're only after alpha
				retval += outExtentLayerCount.x*outExtentLayerCount.y*outExtentLayerCount.z;
			}
			return retval*sizeof(Kernel::value_type);
		}

	protected:
		CBlitImageFilterBase() {}
		virtual ~CBlitImageFilterBase() {}

		static inline bool validate(CStateBase* state)
		{
			if (!state)
				return false;

			return true;
		}
};


// copy while filtering the input into the output
template<class Kernel=CBoxImageFilterKernel>
class CBlitImageFilter : public CImageFilter<CBlitImageFilter<Kernel> >, public CBlitImageFilterBase
{
	public:
		virtual ~CBlitImageFilter() {}

		class CProtoState : public IImageFilter::IState
		{
			public:
				CProtoState()
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}
				virtual ~CProtoState() {}

				union
				{
					core::vectorSIMDu32	inOffsetBaseLayer;
					struct
					{
						VkOffset3D		inOffset;
						uint32_t		inBaseLayer;
					};
				};
				union
				{
					core::vectorSIMDu32 inExtentLayerCount;
					struct
					{
						VkExtent3D		inExtent;
						uint32_t		inLayerCount;
					};
				};
				union
				{
					core::vectorSIMDu32 outOffsetBaseLayer;
					struct
					{
						VkOffset3D		outOffset;
						uint32_t		outBaseLayer;
					};
				};
				union
				{
					core::vectorSIMDu32 outExtentLayerCount;
					struct
					{
						VkExtent3D		outExtent;
						uint32_t		outLayerCount;
					};
				};
				uint32_t				inMipLevel = 0u;
				uint32_t				outMipLevel = 0u;
				ICPUImage*				inImage = nullptr;
				ICPUImage*				outImage = nullptr;
				Kernel					kernel;
		};
		class CState : public CProtoState, public CBlitImageFilterBase::CStateBase
		{
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!CBlitImageFilterBase::validate(state))
				return false;
			
			if (state->scratchMemoryByteSize<getRequiredScratchByteSize(state->kernel,state->alphaSemantic,state->inExtentLayerCount))
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

			const ICPUImage::SCreationParams& inParams = state->inImage->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = state->outImage->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;
			// cannot do alpha adjustment if we dont have alpha or will discard alpha
			if (state->alphaSemantic!=CState::EAS_NONE_OR_PREMULTIPLIED && (getFormatChannelCount(inFormat)!=4u||getFormatChannelCount(outFormat)!=4u))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(outFormat))
				return false;

			return state->kernel.validate(state->inImage,state->outImage);
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

			const auto inMipLevel = state->inMipLevel;
			const auto outMipLevel = state->outMipLevel;
			const auto inBaseLayer = state->inBaseLayer;
			const auto outBaseLayer = state->outBaseLayer;
			const auto layerCount = state->inLayerCount;
			assert(layerCount==state->outLayerCount); // validation bug?

			const auto inOffset = state->inOffset;
			const auto outOffset = state->outOffset;
			const auto inExtent = state->inExtent;
			const auto outExtent = state->outExtent;

			const bool nonPremultBlendSemantic = state->alphaSemantic==CState::EAS_SEPARATE_BLEND;
			const bool coverageSemantic = state->alphaSemantic==CState::EAS_REFERENCE_OR_COVERAGE;
			const auto alphaRefValue = state->alphaRefValue;
			for (uint32_t layer=0; layer!=layerCount; layer++)
			{
				auto load = [layer,inImg,inMipLevel,&state,inFormat](Kernel::value_type* windowSample, const core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord)
				{
					auto texelCoordAndLayer(globalTexelCoord);
					texelCoordAndLayer.w = layer;
					//
					core::vectorSIMDu32 inBlockCoord;
					const void* srcPix[] = {
						inImg->getTexelBlockData(inMipLevel,texelCoordAndLayer,inBlockCoord,state->axisWraps),
						nullptr,
						nullptr,
						nullptr
					};
					if (srcPix[0])
						decodePixels<Kernel::value_type>(inFormat,srcPix,windowSample,inBlockCoord.x,inBlockCoord.y);
				};

				// do the magical coverage adjustment trick suggested by developer of The Witness
				core::rational inverseCoverage(0);
				if (coverageSemantic)
				{
					auto computeCoverage = [&inverseCoverage,inData,&inBlockDims,inFormat,alphaRefValue](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						constexpr auto MaxPlanes = 4;
						const void* srcPix[MaxPlanes] = { inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

						for (auto blockY=0u; blockY<inBlockDims.y; blockY++)
						for (auto blockX=0u; blockX<inBlockDims.x; blockX++)
						{
							Kernel::value_type decbuf[Kernel::MaxChannels] = {0, 0, 0, 1};
							decodePixels<Kernel::value_type>(inFormat,srcPix,decbuf,blockX,blockY);
							if (decbuf[3]<=alphaRefValue)
								inverseCoverage.getNumerator()++;
							inverseCoverage.getDenominator()++;
						}
					};
					const core::SRange<const IImage::SBufferCopy> inRegions = inImg->getRegions(inMipLevel);
					CBasicImageFilterCommon::clip_region_functor_t clip({static_cast<IImage::E_ASPECT_FLAGS>(0u),inMipLevel,inBaseLayer+layer,1}, {inOffset,inExtent}, inFormat);
					CBasicImageFilterCommon::executePerRegion(inImg,computeCoverage,inRegions.begin(),inRegions.end(),clip);
				}
				// meat of the algorithm
				const auto outLayer = outBaseLayer+layer;
				const core::vectorSIMDf fInExtent(state->inExtentLayerCount);
				const core::vectorSIMDf fOutExtent(state->outExtentLayerCount);
				const auto kernel = CScaledImageFilterKernel<Kernel>(fInExtent.preciseDivision(fOutExtent),state->kernel);
				// sampling stuff
				const auto halfPixelOutOffset = core::vectorSIMDf(outBlockDims)*0.5f+core::vectorSIMDf(0.f,0.f,0.f,-float(outLayer)-0.5f);
				const auto outToInScale = core::vectorSIMDf(outBlockDims*state->inExtentLayerCount).preciseDivision(fOutExtent);
				// optionals
				auto* const filteredAlphaArray = reinterpret_cast<Kernel::value_type*>(state->scratchMemory+getRequiredScratchByteSize(kernel));
				auto* filteredAlphaArrayIt = filteredAlphaArray;
				auto blit = [outData,outBlockDims,&halfPixelOutOffset,&outToInScale,&load,&kernel,nonPremultBlendSemantic,coverageSemantic,&filteredAlphaArrayIt,outFormat](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 writeBlockPos) -> void
				{
					void* dstPix = outData+writeBlockArrayOffset;

					Kernel::value_type valbuf[MaxTexelBlockDimensions[1]*MaxTexelBlockDimensions[0]][Kernel::MaxChannels] = {0};
					for (auto blockY=0u; blockY<outBlockDims.y; blockY++)
					for (auto blockX=0u; blockX<outBlockDims.x; blockX++)
					{
						auto* value = valbuf[blockY*outBlockDims.x+blockX];
						Kernel::value_type avgColor = 0;
						auto evaluate = [value,nonPremultBlendSemantic,&avgColor](Kernel::value_type* windowSample, const core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord)
						{
							for (auto i=0; i<Kernel::MaxChannels; i++)
								value[i] += windowSample[i];

							if (!nonPremultBlendSemantic)
								return;

							for (auto i=0; i<Kernel::MaxChannels-1; i++)
								avgColor += windowSample[i]*windowSample[3];
						};
						auto inPos = (core::vectorSIMDf(writeBlockPos)+halfPixelOutOffset)*outToInScale;
						kernel.evaluate(inPos,load,evaluate); 
						// TODO: clamp value (some kernels will produce ringing)
						for (auto i=0; i<Kernel::MaxChannels; i++)
							value[i] = core::clamp<Kernel::value_type,Kernel::value_type>(value[i],0.0,1.0);
						// alpha handling
						if (coverageSemantic)
							*(filteredAlphaArrayIt++) = value[3];
						else if (nonPremultBlendSemantic && avgColor>FLT_MIN*1024.0*512.0)
							value[3] = avgColor/(value[0]+value[1]+value[2]);
					}
					// TODO IMPROVE: by adding random quantization noise (dithering) to break up any banding, could actually steal a sobol sampler for this
					asset::encodePixels<Kernel::value_type>(outFormat,dstPix,valbuf[0]);
				};
				const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(outMipLevel);
				CBasicImageFilterCommon::clip_region_functor_t clip({static_cast<IImage::E_ASPECT_FLAGS>(0u),outMipLevel,outLayer,1}, {outOffset,outExtent}, outFormat);
				CBasicImageFilterCommon::executePerRegion(outImg,blit,outRegions.begin(),outRegions.end(),clip);
				// second part of coverage adjustment
				if (coverageSemantic)
				{
					// sort so that we can easily find the alpha value s.t. % of all texels is less than it
					// TODO IMPROVE: bisection search instead of sort, then reuse the high precision values for alpha setting to reduce quantization error
					std::sort(filteredAlphaArray,filteredAlphaArrayIt);
					auto outputTexelCount = std::distance(filteredAlphaArray,filteredAlphaArrayIt);
					// all values with index < rankIndex will be %==inverseCoverage of the overall array
					int32_t rankIndex = (inverseCoverage*core::rational<int32_t>(outputTexelCount)).getIntegerApprox();
					rankIndex--; // now all with index<=rankIndex
					// this is our new reference value
					auto newRefValue = filteredAlphaArray[core::max(rankIndex,0)];
					// scale all alpha texels to work with new reference value
					auto coverageScale = alphaRefValue/newRefValue;
					auto scaleCoverage = [outData,outBlockDims,outFormat,coverageScale](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						constexpr auto MaxPlanes = 4;
						void* dstPix = outData+readBlockArrayOffset;
						const void* srcPix[MaxPlanes] = { dstPix,nullptr,nullptr,nullptr };

						Kernel::value_type valbuf[MaxTexelBlockDimensions[1]*MaxTexelBlockDimensions[0]][Kernel::MaxChannels] = {0};
						for (auto blockY=0u; blockY<outBlockDims.y; blockY++)
						for (auto blockX=0u; blockX<outBlockDims.x; blockX++)
						{
							auto decbuf = valbuf[blockY*outBlockDims.x+blockX];
							decodePixels<Kernel::value_type>(outFormat,srcPix,decbuf,blockX,blockY);
							decbuf[3] *= coverageScale;
						}
						encodePixels<Kernel::value_type>(outFormat,dstPix,valbuf[0]);
					};
					CBasicImageFilterCommon::executePerRegion(outImg,scaleCoverage,outRegions.begin(),outRegions.end(),clip);
				}
			}
			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif
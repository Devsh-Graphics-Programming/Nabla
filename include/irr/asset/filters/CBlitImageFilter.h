// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_BLIT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_BLIT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>
#include <algorithm>

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
				uint32_t							alphaChannel = 3u;
		};

	protected:
		CBlitImageFilterBase() {}
		virtual ~CBlitImageFilterBase() {}

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
		// we'll probably never remove this requirement
		static_assert(Kernel::is_separable,"Alpha Handling requires high precision and multipass filtering!");

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
				CProtoState(const CProtoState& other) : inMipLevel(other.inMipLevel),outMipLevel(other.outMipLevel),inImage(other.inImage),outImage(other.outImage),kernel(other.kernel)
				{
					inOffsetBaseLayer = other.inOffsetBaseLayer;
					inExtentLayerCount = other.inExtentLayerCount;
					outOffsetBaseLayer = other.outOffsetBaseLayer;
					outExtentLayerCount = other.outExtentLayerCount;
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
		

		static inline uint32_t getRequiredScratchByteSize(const state_type* state)
		{
			uint32_t retval = getScratchOffset(state,true);
			retval += CBlitImageFilterBase::getRequiredScratchByteSize<Kernel>(state->kernel,state->alphaSemantic,state->outExtentLayerCount);
			return retval;
		}

		static inline bool validate(state_type* state)
		{
			if (!CBlitImageFilterBase::validate(state))
				return false;
			
			if (state->scratchMemoryByteSize<getRequiredScratchByteSize(state))
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

			// load all the state
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

			const auto inOffsetBaseLayer = state->inOffsetBaseLayer;
			const auto outOffsetBaseLayer = state->outOffsetBaseLayer;
			const auto inExtentLayerCount = state->inExtentLayerCount;
			const auto outExtentLayerCount = state->outExtentLayerCount;
			const auto inLimit = inOffsetBaseLayer+inExtentLayerCount;
			const auto outLimit = outOffsetBaseLayer+outExtentLayerCount;

			const auto* const axisWraps = state->axisWraps;
			const bool nonPremultBlendSemantic = state->alphaSemantic==CState::EAS_SEPARATE_BLEND;
			const bool coverageSemantic = state->alphaSemantic==CState::EAS_REFERENCE_OR_COVERAGE;
			const auto alphaRefValue = state->alphaRefValue;
			const auto alphaChannel = state->alphaChannel;
			
			// prepare kernel
			const core::vectorSIMDf fInExtent(inExtentLayerCount);
			const core::vectorSIMDf fOutExtent(outExtentLayerCount);
			const auto fScale = fInExtent.preciseDivision(fOutExtent);
			const auto kernel = CScaledImageFilterKernel<Kernel>(fScale,state->kernel);

			// filtering and alpha handling happens separately for every layer, so save on scratch memory size
			const core::vectorSIMDu32 intermediateExtent[2] = {
				core::vectorSIMDu32(outExtent.width,inExtent.height,inExtent.depth),
				core::vectorSIMDu32(outExtent.width,outExtent.height,inExtent.depth)
			};
			const core::vectorSIMDu32 intermediateLastCoord[2] = {
				intermediateExtent[0]-core::vectorSIMDu32(1,1,1,0),
				intermediateExtent[1]-core::vectorSIMDu32(1,1,1,0)
			};
			Kernel::value_type* const intermediateStorage[2] = {
				reinterpret_cast<Kernel::value_type*>(state->scratchMemory),
				reinterpret_cast<Kernel::value_type*>(state->scratchMemory+getScratchOffset(state,false))
			};
			const core::vectorSIMDu32 intermediateStrides[2] = {
				core::vectorSIMDu32(Kernel::MaxChannels,Kernel::MaxChannels*intermediateExtent[0].x,Kernel::MaxChannels*intermediateExtent[0].x*intermediateExtent[0].y,0u),
				core::vectorSIMDu32(Kernel::MaxChannels,Kernel::MaxChannels*intermediateExtent[1].x,Kernel::MaxChannels*intermediateExtent[1].x*intermediateExtent[1].y,0u),
			};
			// load from source
			const auto halfScale = fScale*0.5f;
#if 0
			//const auto posHalfScale = core::vectorSIMDf(fScale.x*0.5f,fScale.y*0.5f,fScale.z*0.5f,FLT_MAX);
			//const auto negHalfScale = -posHalfScale;
			// intermediate store loads
			auto loadIntermediate = [axisWraps,intermediateExtent,intermediateLastCoord,intermediateStrides,intermediateStorage](const int storageID, Kernel::value_type* windowSample, const core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord) -> void
			{
				auto texelCoordAndLayer = ICPUSampler::wrapTextureCoordinate(globalTexelCoord,axisWraps,intermediateExtent[storageID],intermediateLastCoord[storageID]);
				const Kernel::value_type* srcPix = intermediateStorage[storageID]+core::dot(texelCoordAndLayer,intermediateStrides[storageID])[0];
				std::copy(srcPix,srcPix+Kernel::MaxChannels,windowSample);
			};
			auto loadIntermediate0 = [loadIntermediate](Kernel::value_type* windowSample, const core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord) -> void {loadIntermediate(0,windowSample,relativePosAndFactor,globalTexelCoord);};
			auto loadIntermediate1 = [loadIntermediate](Kernel::value_type* windowSample, const core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord) -> void {loadIntermediate(1,windowSample,relativePosAndFactor,globalTexelCoord);};
			// storage functions
			auto storeIntermediate = [](uint32_t writeBlockArrayOffset, const core::vectorSIMDu32& writeBlockPos) -> void
			{
				;
			};
#endif
			auto storeToTexel = [nonPremultBlendSemantic,alphaChannel,outFormat](Kernel::value_type* const sample, void* const dstPix) -> void
			{
				if (nonPremultBlendSemantic && sample[alphaChannel]>FLT_MIN*1024.0*512.0)
				{
					for (auto i=0; i<Kernel::MaxChannels; i++)
					if (i!=alphaChannel)
						sample[i] /= sample[alphaChannel];
				}
				// TODO IMPROVE: by adding random quantization noise (dithering) to break up any banding, could actually steal a sobol sampler for this
				for (auto i=0; i<Kernel::MaxChannels; i++)
					sample[i] = core::clamp<Kernel::value_type,Kernel::value_type>(sample[i],0.0,1.0);
				asset::encodePixels<Kernel::value_type>(outFormat,dstPix,sample);
			};
			const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(outMipLevel);
			auto storeToImage = [coverageSemantic,outExtent,intermediateStorage,alphaRefValue,outData,intermediateStrides,alphaChannel,storeToTexel,outMipLevel,outOffset,outFormat,outRegions,outImg](const core::rational<>& inverseCoverage, const int inputStorageID, const core::vectorSIMDu32& outOffsetLayer) -> void
			{
				// little thing for the coverage adjustment trick suggested by developer of The Witness
				assert(coverageSemantic);
				const auto outputTexelCount = outExtent.width*outExtent.height*outExtent.depth;
				// all values with index<=rankIndex will be %==inverseCoverage of the overall array
				const int32_t rankIndex = (inverseCoverage*core::rational<int32_t>(outputTexelCount)).getIntegerApprox()-1;
				auto* const begin = intermediateStorage[inputStorageID^0x1u];
				// this is our new reference value
				auto* const nth = begin+core::max(rankIndex,0);
				auto* const end = begin+outputTexelCount;
				for (auto i=0; i<outputTexelCount; i++)
				{
					begin[i] = intermediateStorage[inputStorageID][i*4];
					// TODO: add random quantization noise
				}
				std::nth_element(begin,nth,end);
				// scale all alpha texels to work with new reference value
				const auto coverageScale = alphaRefValue/(*nth);
				auto scaleCoverage = [outData,outOffsetLayer,intermediateStrides,inputStorageID,intermediateStorage,alphaChannel,coverageScale,storeToTexel](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 writeBlockPos) -> void
				{
					void* const dstPix = outData+writeBlockArrayOffset;

					Kernel::value_type sample[Kernel::MaxChannels];
					auto first = intermediateStorage[inputStorageID]+core::dot(writeBlockPos-outOffsetLayer,intermediateStrides[inputStorageID])[0];
					std::copy(first,first+Kernel::MaxChannels,sample);

					sample[alphaChannel] *= coverageScale;
					storeToTexel(sample,dstPix);
				};
				CBasicImageFilterCommon::clip_region_functor_t clip({static_cast<IImage::E_ASPECT_FLAGS>(0u),outMipLevel,outOffsetLayer.w,1}, {outOffset,outExtent}, outFormat);
				CBasicImageFilterCommon::executePerRegion(outImg,scaleCoverage,outRegions.begin(),outRegions.end(),clip);
			};
			// process
			const auto inImageType = inParams.type;
			core::vectorSIMDf windowStartFraction;
			const auto windowMinCoord = kernel.getWindowMinCoord(halfScale,windowStartFraction);
			for (uint32_t layer=0; layer!=layerCount; layer++)
			{
				auto outOffsetLayer = outOffsetBaseLayer;
				outOffsetLayer.w += layer;

				// reset coverage counter
				core::rational inverseCoverage(0);
				// filter lambda
				auto filterAxis = [&](IImage::E_TYPE axis) -> void
				{
					const bool lastPass = inImageType==axis;
					const core::vectorSIMDi32 unitIncrease(axis==IImage::ET_1D ? 1:0,axis==IImage::ET_2D ? 1:0,axis==IImage::ET_3D ? 1:0,0);

					const int inputStorageID = axis&0x1u;
					const int outputStorageID = inputStorageID^0x1u;

					const auto windowSize = kernel.getWindowSize()[axis];
					core::vectorSIMDu32 localTexCoord;
					// TODO: figure out the loops here
					for (auto& z=(localTexCoord.z=0); z<intermediateExtent[inputStorageID].z; z++)
					for (auto& y=(localTexCoord.y=0); y<intermediateExtent[inputStorageID].y; y++)
					{
						const int32_t windowStart = windowMinCoord[axis]+inOffsetBaseLayer[axis];
						const int32_t windowEnd = windowStart+inExtentLayerCount[axis]+windowSize;
						// preload whole line plus window borders
						Kernel::value_type* cacheBuffer = intermediateStorage[1];
						// TODO figure out the rest of indices
						localTexCoord.x = 0;
						for (auto i=windowStart; i<windowEnd; i++)
						{
							core::vectorSIMDi32 globalTexelCoord(localTexCoord+inOffsetBaseLayer);
							globalTexelCoord[0] += i;
							globalTexelCoord[3] += layer;

							core::vectorSIMDu32 inBlockCoord;
							const void* srcPix[] = { // multiple loads for texture boundaries aren't that bad
								inImg->getTexelBlockData(inMipLevel,inImg->wrapTextureCoordinate(inMipLevel,globalTexelCoord,axisWraps),inBlockCoord),
								nullptr,
								nullptr,
								nullptr
							};
							if (!srcPix[0])
								continue;

							auto sample = cacheBuffer+(i-windowStart)*Kernel::MaxChannels;
							decodePixels<Kernel::value_type>(inFormat,srcPix,sample,inBlockCoord.x,inBlockCoord.y);

							if (nonPremultBlendSemantic)
							{
								for (auto i=0; i<Kernel::MaxChannels; i++)
								if (i!=alphaChannel)
									sample[i] *= sample[alphaChannel];
							}
							else if (coverageSemantic && i>=inOffsetBaseLayer[axis] && i<inLimit[axis])
							{
								if (sample[alphaChannel]<=alphaRefValue)
									inverseCoverage.getNumerator()++;
								inverseCoverage.getDenominator()++;
							}
						}
						for (auto& x=localTexCoord[0]; x<outExtentLayerCount[axis]; x++) // TODO: x?
						{
							// get output pixel
							auto* const value = intermediateStorage[inputStorageID]+core::dot(intermediateStrides[inputStorageID],localTexCoord)[0];
							std::fill(value,value+Kernel::MaxChannels,Kernel::value_type(0));
							// kernel load functor
							auto load = [](Kernel::value_type* windowSample, const core::vectorSIMDf& unused0, const core::vectorSIMDi32& unused1) -> void
							{
								for (auto i=0; i<Kernel::MaxChannels; i++)
									windowSample[i] = 0.f;
							};
							// kernel evaluation functor
							auto evaluate = [value](const Kernel::value_type* windowSample, const core::vectorSIMDf& unused0, const core::vectorSIMDi32& unused1) -> void
							{
								for (auto i=0; i<Kernel::MaxChannels; i++)
									value[i] += windowSample[i];
							};
							core::vectorSIMDi32 windowCoord;
							windowCoord[axis] = windowStart+x; // TODO: x?
							core::vectorSIMDf relativePosAndFactor;
							relativePosAndFactor[axis] = windowStartFraction[axis]-float(windowCoord[axis]);
							for (auto i=0; i<windowSize; i++)
							{
								Kernel::value_type windowSample[Kernel::MaxChannels];
								kernel.evaluateImpl(load,evaluate,windowSample,relativePosAndFactor,windowCoord);
								relativePosAndFactor -= core::vectorSIMDf(unitIncrease);
								windowCoord += unitIncrease;
							}
							if (!coverageSemantic && lastPass) // store to image, we're done
							{
								core::vectorSIMDu32 dummy;
								storeToTexel(value,outImg->getTexelBlockData(outMipLevel,localTexCoord+outOffsetBaseLayer,dummy));
							}
						}
					}
					// we'll only get here if we have to do coverage adjustment
					if (coverageSemantic && lastPass)
						storeToImage(inverseCoverage,inputStorageID,outOffsetLayer);
				};
				// filter in X-axis (load from input, save to intermediate1)
				filterAxis(IImage::ET_1D);
				// filter in Y-axis (load from intermediate1, save to intermediate2)
				///filterAxis(IImage::ET_2D);
				// filter in Z-axis (load from intermediate2, save to output)
				assert(inImageType!=IImage::ET_3D); // I need to test this in the future
				///filterAxis(IImage::ET_3D);
			}
			return true;
		}

	private:
		static inline uint32_t getScratchOffset(const state_type* state, bool afterSecondPass)
		{
			const auto& inParams = state->inImage->getCreationParameters();
			const auto window_last = state->kernel.window_size-core::vectorSIMDi32(1,inParams.type!=IImage::ET_1D ? 1:state->kernel.window_size[1],inParams.type!=IImage::ET_2D ? 1:state->kernel.window_size[2],0);
			auto texelCount = state->outExtent.width*core::max((state->inExtent.height+window_last[1])*(state->inExtent.depth+window_last[2]),state->outExtent.height*state->outExtent.depth);
			if (afterSecondPass)
				texelCount += core::max(state->outExtent.width*state->outExtent.height*(state->inExtent.depth+window_last[2]),state->inExtent.height+window_last[0]);
			return texelCount*Kernel::MaxChannels*sizeof(Kernel::value_type);
		}
};

} // end namespace asset
} // end namespace irr

#endif
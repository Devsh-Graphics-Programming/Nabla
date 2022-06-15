// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BLIT_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_BLIT_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <type_traits>
#include <algorithm>

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/asset/filters/dithering/CWhiteNoiseDither.h"

#include "nbl/asset/filters/kernels/kernels.h"

#include "nbl/asset/format/decodePixels.h"

namespace nbl::asset
{

class IBlitUtilities
{
public:
	enum E_ALPHA_SEMANTIC : uint32_t
	{
		EAS_NONE_OR_PREMULTIPLIED = 0u, // just filter the channels independently (also works for a texture for blending equation `dstCol*(1-srcAlpha)+srcCol`)
		EAS_REFERENCE_OR_COVERAGE, // try to preserve coverage (percentage of pixels above a threshold value) across mipmap levels
		EAS_SEPARATE_BLEND, // compute a new alpha value for a texture to be used with the blending equation `mix(dstCol,srcCol,srcAlpha)`
		EAS_COUNT
	};

	static inline core::vectorSIMDu32 getPhaseCount(const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType)
	{
		core::vectorSIMDu32 result(0u);
		for (uint32_t i = 0u; i <= inImageType; ++i)
			result[i] = outExtent[i] / std::gcd(inExtent[i], outExtent[i]);
		return result;
	}

	// we'll need to rescale the kernel support to be relative to the output image but in the input image coordinate system
	// (if support is 3 pixels, it needs to be 3 output texels, but measured in input texels)
	template<class Kernel>
	static inline auto constructScaledKernel(const Kernel& kernel, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent)
	{
		const core::vectorSIMDf fInExtent(inExtent);
		const core::vectorSIMDf fOutExtent(outExtent);
		const auto fScale = fInExtent.preciseDivision(fOutExtent);
		return CScaledImageFilterKernel<Kernel>(fScale, kernel);
	}
};

template <class KernelX = CBoxImageFilterKernel, class KernelY = KernelX, class KernelZ = KernelX>
class CBlitUtilities : public IBlitUtilities
{
	static_assert(std::is_same<typename KernelX::value_type, typename KernelY::value_type>::value&& std::is_same<typename KernelZ::value_type, typename KernelY::value_type>::value, "Kernel value_type need to be identical");

public:
	_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = KernelX::MaxChannels > KernelY::MaxChannels ? (KernelX::MaxChannels > KernelZ::MaxChannels ? KernelX::MaxChannels : KernelZ::MaxChannels) : (KernelY::MaxChannels > KernelZ::MaxChannels ? KernelY::MaxChannels : KernelZ::MaxChannels);

	template <typename lut_value_type = KernelX::value_type>
	static inline size_t getScaledKernelPhasedLUTSize(const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ)
	{
		const auto scaledKernelX = constructScaledKernel(kernelX, inExtent, outExtent);
		const auto scaledKernelY = constructScaledKernel(kernelY, inExtent, outExtent);
		const auto scaledKernelZ = constructScaledKernel(kernelZ, inExtent, outExtent);

		const auto phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

		return ((phaseCount[0] * scaledKernelX.getWindowSize().x) + (phaseCount[1] * scaledKernelY.getWindowSize().y) + (phaseCount[2] * scaledKernelZ.getWindowSize().z)) * sizeof(lut_value_type) * MaxChannels;
	}

	template <typename lut_value_type = KernelX::value_type>
	static bool computeScaledKernelPhasedLUT(void* outKernelWeights, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ)
	{
		const core::vectorSIMDu32 phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

		for (auto i = 0; i <= inImageType; ++i)
		{
			if (phaseCount[i] == 0)
				return false;
		}

		const auto scaledKernelX = constructScaledKernel(kernelX, inExtent, outExtent);
		const auto scaledKernelY = constructScaledKernel(kernelY, inExtent, outExtent);
		const auto scaledKernelZ = constructScaledKernel(kernelZ, inExtent, outExtent);

		const auto windowDims = getRealWindowSize(inImageType, scaledKernelX, scaledKernelY, scaledKernelZ);
		const auto axisOffsets = getScaledKernelPhasedLUTAxisOffsets<lut_value_type>(phaseCount, windowDims);

		const core::vectorSIMDf fInExtent(inExtent);
		const core::vectorSIMDf fOutExtent(outExtent);
		const auto fScale = fInExtent.preciseDivision(fOutExtent);

		// a dummy load functor
		// does nothing but fills up the `windowSample` with 1s (identity) so we can preserve the value of kernel
		// weights when eventually `windowSample` gets multiplied by them later in
		// `CFloatingPointSeparableImageFilterKernelBase<CRTP>::sample_functor_t<PreFilter,PostFilter>::operator()`
		// this exists only because `evaluateImpl` expects a pre filtering step.
		auto dummyLoad = [](double* windowSample, const core::vectorSIMDf&, const core::vectorSIMDi32&, const IImageFilterKernel::UserData*) -> void
		{
			for (auto h = 0; h < MaxChannels; h++)
				windowSample[h] = 1.0;
		};

		double kernelWeight[MaxChannels];
		// actually used to put values in the LUT
		auto dummyEvaluate = [&kernelWeight](const double* windowSample, const core::vectorSIMDf&, const core::vectorSIMDi32&, const IImageFilterKernel::UserData*) -> void
		{
			for (auto h = 0; h < MaxChannels; h++)
				kernelWeight[h] = windowSample[h];
		};

		auto computeForAxis = [&](const asset::IImage::E_TYPE axis, const auto& scaledKernel)
		{
			if (axis > inImageType)
				return;

			const auto windowSize = scaledKernel.getWindowSize()[axis];

			IImageFilterKernel::ScaleFactorUserData scale(1.f / fScale[axis]);
			const IImageFilterKernel::ScaleFactorUserData* otherScale = nullptr;
			switch (axis)
			{
			case IImage::ET_1D:
				otherScale = IImageFilterKernel::ScaleFactorUserData::cast(kernelX.getUserData());
				break;
			case IImage::ET_2D:
				otherScale = IImageFilterKernel::ScaleFactorUserData::cast(kernelY.getUserData());
				break;
			case IImage::ET_3D:
				otherScale = IImageFilterKernel::ScaleFactorUserData::cast(kernelZ.getUserData());
				break;
			}
			if (otherScale)
			{
				for (auto k = 0; k < MaxChannels; k++)
					scale.factor[k] *= otherScale->factor[k];
			}

			lut_value_type* outKernelWeightsPixel = reinterpret_cast<lut_value_type*>(reinterpret_cast<uint8_t*>(outKernelWeights) + axisOffsets[axis]);
			for (uint32_t i = 0u; i < phaseCount[axis]; ++i)
			{
				core::vectorSIMDf tmp(0.f);
				tmp[axis] = float(i) + 0.5f;

				const int32_t windowCoord = scaledKernel.getWindowMinCoord(tmp * fScale, tmp)[axis];

				float relativePos = tmp[axis] - float(windowCoord); // relative position of the last pixel in window from current (ith) output pixel having a unique phase sequence of kernel evaluation points

				for (int32_t j = 0; j < windowSize; ++j)
				{
					core::vectorSIMDf tmp(relativePos, 0.f, 0.f);
					scaledKernel.evaluateImpl(dummyLoad, dummyEvaluate, kernelWeight, tmp, core::vectorSIMDi32(), &scale);
					for (uint32_t ch = 0; ch < MaxChannels; ++ch)
					{
						if constexpr (std::is_same_v<lut_value_type, uint16_t>)
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = core::Float16Compressor::compress(float(kernelWeight[ch]));
						else
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = lut_value_type(kernelWeight[ch]);

					}
					relativePos -= 1.f;
				}
			}
		};

		computeForAxis(asset::IImage::ET_1D, scaledKernelX);
		computeForAxis(asset::IImage::ET_2D, scaledKernelY);
		computeForAxis(asset::IImage::ET_3D, scaledKernelZ);

		return true;
	}

	static inline core::vectorSIMDi32 getRealWindowSize(const IImage::E_TYPE inImageType,
		const CScaledImageFilterKernel<KernelX>& kernelX,
		const CScaledImageFilterKernel<KernelY>& kernelY,
		const CScaledImageFilterKernel<KernelZ>& kernelZ)
	{
		core::vectorSIMDi32 last(kernelX.getWindowSize().x, 0, 0, 0);
		if (inImageType >= IImage::ET_2D)
			last.y = kernelY.getWindowSize().y;
		if (inImageType >= IImage::ET_3D)
			last.z = kernelZ.getWindowSize().z;
		return last;
	}

	template <typename lut_value_type = KernelX::value_type>
	static inline core::vectorSIMDu32 getScaledKernelPhasedLUTAxisOffsets(const core::vectorSIMDu32& phaseCount, const core::vectorSIMDi32& real_window_size)
	{
		core::vectorSIMDu32 result;
		result.x = 0u;
		result.y = (phaseCount[0] * real_window_size.x);
		result.z = ((phaseCount[0] * real_window_size.x) + (phaseCount[1] * real_window_size.y));
		return result * sizeof(lut_value_type) * MaxChannels;
	}
};

template<typename value_type, typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class NBL_API CBlitImageFilterBase : public impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>, public CBasicImageFilterCommon
{
	public:
		class CStateBase : public impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>::state_type
		{
			public:
				CStateBase() {}
				virtual ~CStateBase() {}

				// we need scratch memory because we'll decode the whole image into one contiguous chunk of memory for faster filtering amongst other things
				uint8_t*							scratchMemory = nullptr;
				uint32_t							scratchMemoryByteSize = 0u;
				_NBL_STATIC_INLINE_CONSTEXPR auto	NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP			axisWraps[NumWrapAxes] = { ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT };
				ISampler::E_TEXTURE_BORDER_COLOR	borderColor = ISampler::ETBC_FLOAT_TRANSPARENT_BLACK;
				IBlitUtilities::E_ALPHA_SEMANTIC	alphaSemantic = IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED;
				double								alphaRefValue = 0.5; // only required to make sense if `alphaSemantic==EAS_REFERENCE_OR_COVERAGE`
				uint32_t							alphaChannel = 3u; // index of the alpha channel (could be different cause of swizzles)
		};

	protected:
		CBlitImageFilterBase() {}
		virtual ~CBlitImageFilterBase() {}

		// this will be called by derived classes because it doesn't account for all scratch needed, just the stuff for coverage adjustment
		static inline uint32_t getRequiredScratchByteSize(IBlitUtilities::E_ALPHA_SEMANTIC alphaSemantic=IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED,
															const core::vectorSIMDu32& outExtentLayerCount=core::vectorSIMDu32(0,0,0,0))
		{
			uint32_t retval = 0u;
			// 
			if (alphaSemantic==IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
			{
				// no mul by channel count because we're only after alpha
				retval += outExtentLayerCount.x*outExtentLayerCount.y*outExtentLayerCount.z;
			}
			return retval*sizeof(value_type);
		}

		// nothing to validate here really
		static inline bool validate(CStateBase* state)
		{
			if (!state)
				return false;

			// only check that scratch exists, the derived class will check for actual size
			if (!state->scratchMemory)
				return false;

			for (auto i=0; i<CStateBase::NumWrapAxes; i++)
			if (state->axisWraps[i]>=ISampler::ETC_COUNT)
				return false;

			if (state->borderColor>=ISampler::ETBC_COUNT)
				return false;

			if (state->alphaSemantic>=IBlitUtilities::EAS_COUNT)
				return false;

			if (state->alphaChannel>=4)
				return false;

			if (!impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>::validate(state))
				return false;

			return true;
		}
};

// copy while filtering the input into the output, a rare filter where the input and output extents can be different, still works one mip level at a time
template<typename Swizzle=DefaultSwizzle, typename Dither=CWhiteNoiseDither, typename Normalization=void, bool Clamp=true, class KernelX=CBoxImageFilterKernel, class KernelY=KernelX, class KernelZ=KernelX, typename lut_value_type = typename KernelX::value_type>
class NBL_API CBlitImageFilter : public CImageFilter<CBlitImageFilter<Swizzle,Dither,Normalization,Clamp,KernelX,KernelX,KernelX>>, public CBlitImageFilterBase<typename KernelY::value_type,Swizzle,Dither,Normalization,Clamp>
{
	public:
		using utils_t = CBlitUtilities<KernelX, KernelY, KernelZ>;

	private:
		using value_type = typename KernelX::value_type;
		using base_t = CBlitImageFilterBase<value_type,Swizzle,Dither,Normalization,Clamp>;

		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = utils_t::MaxChannels;

	public:
		// we'll probably never remove this requirement
		static_assert(KernelX::is_separable&&KernelY::is_separable&&KernelZ::is_separable,"Alpha Handling requires high precision and multipass filtering!");

		virtual ~CBlitImageFilter() {}

		class CState : public IImageFilter::IState, public base_t::CStateBase
		{
			public:
				CState(KernelX&& kernel_x, KernelY&& kernel_y, KernelZ&& kernel_z) :
					kernelX(std::move(kernel_x)), kernelY(std::move(kernel_y)), kernelZ(std::move(kernel_z))
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}
				CState() : CState(KernelX(), KernelY(), KernelZ())
				{
				}
				CState(const CState& other) : IImageFilter::IState(), base_t::CStateBase{other},
					inMipLevel(other.inMipLevel),outMipLevel(other.outMipLevel),inImage(other.inImage),outImage(other.outImage),
					kernelX(other.kernelX), kernelY(other.kernelY), kernelZ(other.kernelZ)
				{
					inOffsetBaseLayer = other.inOffsetBaseLayer;
					inExtentLayerCount = other.inExtentLayerCount;
					outOffsetBaseLayer = other.outOffsetBaseLayer;
					outExtentLayerCount = other.outExtentLayerCount;
				}
				virtual ~CState() {}

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
				
				uint32_t							inMipLevel = 0u;
				uint32_t							outMipLevel = 0u;
				ICPUImage*							inImage = nullptr;
				ICPUImage*							outImage = nullptr;
				KernelX								kernelX;
				KernelY								kernelY;
				KernelZ								kernelZ;
		};
		using state_type = CState;
		
		static inline uint32_t getRequiredScratchByteSize(const state_type* state)
		{
			// need to add the memory for ping pong buffers
			uint32_t retval = getScaledKernelPhasedLUTByteOffset(state);
			
			// need to add the memory for phase support LUT
			retval += utils_t::template getScaledKernelPhasedLUTSize<lut_value_type>(state->inExtentLayerCount, state->outExtentLayerCount, state->inImage->getCreationParameters().type,
				state->kernelX, state->kernelY, state->kernelZ);

			return retval;
		}

		static inline uint32_t getScaledKernelPhasedLUTByteOffset(const state_type* state)
		{
			const auto scaledKernelX = utils_t::constructScaledKernel(state->kernelX, state->inExtentLayerCount, state->outExtentLayerCount);
			const auto scaledKernelY = utils_t::constructScaledKernel(state->kernelY, state->inExtentLayerCount, state->outExtentLayerCount);
			const auto scaledKernelZ = utils_t::constructScaledKernel(state->kernelZ, state->inExtentLayerCount, state->outExtentLayerCount);

			const uint32_t retval = getScratchOffset(state, true) + base_t::getRequiredScratchByteSize(state->alphaSemantic, state->outExtentLayerCount);
			return retval;
		}

		static inline bool validate(state_type* state)
		{
			if (!base_t::validate(state))
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
			if (state->alphaSemantic!=IBlitUtilities::EAS_NONE_OR_PREMULTIPLIED && (getFormatChannelCount(inFormat)!=4u||getFormatChannelCount(outFormat)!=4u))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(outFormat))
				return false;

			return state->kernelX.validate(state->inImage,state->outImage)&&state->kernelY.validate(state->inImage,state->outImage)&&state->kernelZ.validate(state->inImage,state->outImage);
		}

		// CBlitUtilities::computeScaledKernelPhasedLUT stores the kernel entries, in the LUT, in reverse, which are then forward iterated to compute the CONVOLUTION.
		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
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
			const bool nonPremultBlendSemantic = state->alphaSemantic==IBlitUtilities::EAS_SEPARATE_BLEND;
			// TODO: reformulate coverage adjustment as a normalization
			const bool coverageSemantic = state->alphaSemantic==IBlitUtilities::EAS_REFERENCE_OR_COVERAGE;
			const bool needsNormalization = !std::is_void_v<Normalization> || coverageSemantic;
			const auto alphaRefValue = state->alphaRefValue;
			const auto alphaChannel = state->alphaChannel;
			
			// prepare kernel
			const auto scaledKernelX = utils_t::constructScaledKernel(state->kernelX, inExtentLayerCount, outExtentLayerCount);
			const auto scaledKernelY = utils_t::constructScaledKernel(state->kernelY, inExtentLayerCount, outExtentLayerCount);
			const auto scaledKernelZ = utils_t::constructScaledKernel(state->kernelZ, inExtentLayerCount, outExtentLayerCount);

			// filtering and alpha handling happens separately for every layer, so save on scratch memory size
			const auto inImageType = inParams.type;
			const auto real_window_size = utils_t::getRealWindowSize(inImageType,scaledKernelX,scaledKernelY,scaledKernelZ);
			core::vectorSIMDi32 intermediateExtent[3];
			getIntermediateExtents(intermediateExtent, state, real_window_size);
			const core::vectorSIMDi32 intermediateLastCoord[3] = {
				intermediateExtent[0]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[1]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[2]-core::vectorSIMDi32(1,1,1,0)
			};
			value_type* const intermediateStorage[3] = {
				reinterpret_cast<value_type*>(state->scratchMemory),
				reinterpret_cast<value_type*>(state->scratchMemory+getScratchOffset(state, false)),
				reinterpret_cast<value_type*>(state->scratchMemory)
			};
			const core::vectorSIMDu32 intermediateStrides[3] = {
				core::vectorSIMDu32(MaxChannels*intermediateExtent[0].y,MaxChannels,MaxChannels*intermediateExtent[0].x*intermediateExtent[0].y,0u),
				core::vectorSIMDu32(MaxChannels*intermediateExtent[1].y*intermediateExtent[1].z,MaxChannels*intermediateExtent[1].z,MaxChannels,0u),
				core::vectorSIMDu32(MaxChannels,MaxChannels*intermediateExtent[2].x,MaxChannels*intermediateExtent[2].x*intermediateExtent[2].y,0u)
			};
			// storage
			core::RandomSampler sampler(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			auto storeToTexel = [state,nonPremultBlendSemantic,alphaChannel,outFormat](value_type* const sample, void* const dstPix, const core::vectorSIMDu32& localOutPos) -> void
			{
				if (nonPremultBlendSemantic && sample[alphaChannel]>FLT_MIN*1024.0*512.0)
				{
					for (auto i=0; i<MaxChannels; i++)
					if (i!=alphaChannel)
						sample[i] /= sample[alphaChannel];
				}

				base_t::onEncode(outFormat, state, dstPix, sample, localOutPos, 0, 0, MaxChannels);
			};
			const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(outMipLevel);
			auto storeToImage = [policy,coverageSemantic,needsNormalization,outExtent,intermediateStorage,&sampler,outFormat,alphaRefValue,outData,intermediateStrides,alphaChannel,storeToTexel,outMipLevel,outOffset,outRegions,outImg](
				const core::rational<int64_t>& inverseCoverage, const int axis, const core::vectorSIMDu32& outOffsetLayer
			) -> void
			{
				assert(needsNormalization);
				value_type coverageScale = 1.0;
				if (coverageSemantic) // little thing for the coverage adjustment trick suggested by developer of The Witness
				{
					const auto outputTexelCount = outExtent.width*outExtent.height*outExtent.depth;
					// all values with index<=rankIndex will be %==inverseCoverage of the overall array
					const int64_t rankIndex = (inverseCoverage*core::rational<int64_t>(outputTexelCount)).getIntegerApprox()-1;
					auto* const begin = intermediateStorage[(axis+1)%3];
					// this is our new reference value
					auto* const nth = begin+core::max<int64_t>(rankIndex,0);
					auto* const end = begin+outputTexelCount;
					std::for_each(policy,begin,end,[&intermediateStorage,axis,begin,alphaChannel,&sampler,outFormat](value_type& texelAlpha)
					{
						texelAlpha = intermediateStorage[axis][std::distance(begin,&texelAlpha)*4u+alphaChannel];
						texelAlpha -= double(sampler.nextSample())*(asset::getFormatPrecision<value_type>(outFormat,alphaChannel,texelAlpha)/double(~0u));
					});
					core::nth_element(policy,begin,nth,end);
					// scale all alpha texels to work with new reference value
					coverageScale = alphaRefValue/(*nth);
				}
				auto scaleCoverage = [outData,outOffsetLayer,intermediateStrides,axis,intermediateStorage,alphaChannel,coverageScale,storeToTexel](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 writeBlockPos) -> void
				{
					void* const dstPix = outData+writeBlockArrayOffset;
					const core::vectorSIMDu32 localOutPos = writeBlockPos - outOffsetLayer;

					value_type sample[MaxChannels];
					const size_t offset = IImage::SBufferCopy::getLocalByteOffset(localOutPos, intermediateStrides[axis]);
					const auto* first = intermediateStorage[axis]+offset;
					std::copy(first,first+MaxChannels,sample);

					sample[alphaChannel] *= coverageScale;
					storeToTexel(sample,dstPix,localOutPos);
				};
				const ICPUImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),outMipLevel,outOffsetLayer.w,1};
				const IImageFilter::IState::TexelRange range = {outOffset,outExtent};
				CBasicImageFilterCommon::clip_region_functor_t clip(subresource, range, outFormat);
				CBasicImageFilterCommon::executePerRegion(policy,outImg,scaleCoverage,outRegions.begin(),outRegions.end(),clip);
			};
			
			// process
			state->normalization.template initialize<double>();
			const core::vectorSIMDf fInExtent(inExtentLayerCount);
			const core::vectorSIMDf fOutExtent(outExtentLayerCount);
			const auto fScale = fInExtent.preciseDivision(fOutExtent);
			const auto halfTexelOffset = fScale*0.5f-core::vectorSIMDf(0.f,0.f,0.f,0.5f);
			const auto startCoord =  [&halfTexelOffset,&scaledKernelX,&scaledKernelY,&scaledKernelZ]() -> core::vectorSIMDi32
			{
				return core::vectorSIMDi32(scaledKernelX.getWindowMinCoord(halfTexelOffset).x,scaledKernelY.getWindowMinCoord(halfTexelOffset).y,scaledKernelZ.getWindowMinCoord(halfTexelOffset).z,0);
			}();
			const auto windowMinCoordBase = inOffsetBaseLayer+startCoord;

			core::vectorSIMDu32 phaseCount = IBlitUtilities::getPhaseCount(inExtentLayerCount, outExtentLayerCount, inImageType);
			phaseCount = core::max(phaseCount, core::vectorSIMDu32(1, 1, 1));
			const core::vectorSIMDu32 axisOffsets = utils_t::template getScaledKernelPhasedLUTAxisOffsets<lut_value_type>(phaseCount, real_window_size);
			constexpr auto MaxAxisCount = 3;
			lut_value_type* scaledKernelPhasedLUTPixel[MaxAxisCount];
			for (auto i = 0; i < MaxAxisCount; ++i)
				scaledKernelPhasedLUTPixel[i] = reinterpret_cast<lut_value_type*>(state->scratchMemory + getScaledKernelPhasedLUTByteOffset(state) + axisOffsets[i]);

			for (uint32_t layer=0; layer!=layerCount; layer++) // TODO: could be parallelized
			{
				const core::vectorSIMDi32 vLayer(0,0,0,layer);
				const auto windowMinCoord = windowMinCoordBase+vLayer;
				const auto outOffsetLayer = outOffsetBaseLayer+vLayer;
				// reset coverage counter
				constexpr bool is_seq_policy_v = std::is_same_v<std::remove_reference_t<ExecutionPolicy>,core::execution::sequenced_policy>;
				using cond_atomic_int32_t = std::conditional_t<is_seq_policy_v,int32_t,std::atomic_int32_t>;
				using cond_atomic_uint32_t = std::conditional_t<is_seq_policy_v,uint32_t,std::atomic_uint32_t>;
				cond_atomic_uint32_t inv_cvg_num(0u);
				cond_atomic_uint32_t inv_cvg_den(0u);
				// filter lambda
				auto filterAxis = [&](IImage::E_TYPE axis, auto& kernel) -> void
				{
					if (axis>inImageType)
						return;

					const bool lastPass = inImageType==axis;
					const auto windowSize = kernel.getWindowSize()[axis];

					IImageFilterKernel::ScaleFactorUserData scale(1.f/fScale[axis]);
					const IImageFilterKernel::ScaleFactorUserData* otherScale = nullptr;
					switch (axis)
					{
						case IImage::ET_1D:
							otherScale = IImageFilterKernel::ScaleFactorUserData::cast(state->kernelX.getUserData());
							break;
						case IImage::ET_2D:
							otherScale = IImageFilterKernel::ScaleFactorUserData::cast(state->kernelY.getUserData());
							break;
						case IImage::ET_3D:
							otherScale = IImageFilterKernel::ScaleFactorUserData::cast(state->kernelZ.getUserData());
							break;
					}
					if (otherScale)
					for (auto k=0; k<MaxChannels; k++)
						scale.factor[k] *= otherScale->factor[k];
	
					// z y x output along x
					// z x y output along y
					// x y z output along z
					const int loopCoordID[2] = {/*axis,*/axis!=IImage::ET_2D ? 1:0,axis!=IImage::ET_3D ? 2:0};
					//
					assert(is_seq_policy_v || std::thread::hardware_concurrency()<=64u);
					uint64_t decodeScratchAllocs[VectorizationBoundSTL];
					std::fill_n(decodeScratchAllocs,VectorizationBoundSTL,~0u);
					std::mutex scratchLock;
					auto alloc_decode_scratch = [is_seq_policy_v,&scratchLock,&decodeScratchAllocs]() -> int32_t
					{
						if /*constexpr*/ (is_seq_policy_v)
							return 0;
						else
						{
							std::unique_lock<std::mutex> lock(scratchLock);
							for (uint32_t j=0u; j<VectorizationBoundSTL; j++)
							{
								int32_t firstFree = core::findLSB(decodeScratchAllocs[j]);
								if (firstFree==-1)
									continue;
								decodeScratchAllocs[j] ^= 0x1u<<firstFree;
								return j*64u+firstFree;
							}
							assert(false);
							return 0xdeadbeef;
						}
					};
					auto free_decode_scratch = [is_seq_policy_v,&scratchLock,&decodeScratchAllocs](int32_t addr)
					{
						if /*constexpr*/ (!is_seq_policy_v)
						{
							std::unique_lock<std::mutex> lock(scratchLock);
							decodeScratchAllocs[addr/64u] ^= 0x1u<<(addr%64u);
						}
					};
					//
					constexpr uint32_t batch_dims = 2u;
					const uint32_t batchExtent[batch_dims] = {
						static_cast<uint32_t>(intermediateExtent[axis][loopCoordID[0]]),
						static_cast<uint32_t>(intermediateExtent[axis][loopCoordID[1]])
					};
					CBasicImageFilterCommon::BlockIterator<batch_dims> begin(batchExtent);
					const uint32_t spaceFillingEnd[batch_dims] = {0u,batchExtent[1]};
					CBasicImageFilterCommon::BlockIterator<batch_dims> end(begin.getExtentBatches(),spaceFillingEnd);
					std::for_each(policy,begin,end,[&](const std::array<uint32_t,batch_dims>& batchCoord) -> void
					{
						// we need some tmp memory for threads in the first pass so that they dont step on each other
						uint32_t decode_offset;
						// whole line plus window borders
						value_type* lineBuffer;
						core::vectorSIMDi32 localTexCoord(0);
						localTexCoord[loopCoordID[0]] = batchCoord[0];
						localTexCoord[loopCoordID[1]] = batchCoord[1];
						if (axis!=IImage::ET_1D)
							lineBuffer = intermediateStorage[axis-1]+core::dot(static_cast<const core::vectorSIMDi32&>(intermediateStrides[axis-1]),localTexCoord)[0];
						else
						{
							const auto inputEnd = inExtent.width+real_window_size.x;
							decode_offset = alloc_decode_scratch();
							lineBuffer = intermediateStorage[1]+decode_offset*MaxChannels*inputEnd;
							for (auto& i=localTexCoord.x; i<inputEnd; i++)
							{
								core::vectorSIMDi32 globalTexelCoord(localTexCoord+windowMinCoord);

								core::vectorSIMDu32 inBlockCoord(0u);
								const void* srcPix[] = { // multiple loads for texture boundaries aren't that bad
									inImg->getTexelBlockData(inMipLevel,inImg->wrapTextureCoordinate(inMipLevel,globalTexelCoord,axisWraps),inBlockCoord),
									nullptr,
									nullptr,
									nullptr
								};
								if (!srcPix[0])
									continue;

								auto sample = lineBuffer+i*MaxChannels;
								value_type swizzledSample[MaxChannels];

								// TODO: make sure there is no leak due to MaxChannels!
								base_t::template onDecode(inFormat, state, srcPix, sample, swizzledSample, inBlockCoord.x, inBlockCoord.y);

								if (nonPremultBlendSemantic)
								{
									for (auto i=0; i<MaxChannels; i++)
									if (i!=alphaChannel)
										sample[i] *= sample[alphaChannel];
								}
								else if (coverageSemantic && globalTexelCoord[axis]>=inOffsetBaseLayer[axis] && globalTexelCoord[axis]<inLimit[axis])
								{
									if (sample[alphaChannel]<=alphaRefValue)
										inv_cvg_num++;
									inv_cvg_den++;
								}
							}
						}

						auto getWeightedSample = [scaledKernelPhasedLUTPixel, windowSize, lineBuffer, &windowMinCoord, axis](const auto& windowCoord, const auto phaseIndex, const auto windowPixel, const auto channel) -> value_type
						{
							value_type kernelWeight;
							if constexpr (std::is_same_v<lut_value_type, uint16_t>)
								kernelWeight = value_type(core::Float16Compressor::decompress(scaledKernelPhasedLUTPixel[axis][(phaseIndex * windowSize + windowPixel) * MaxChannels + channel]));
							else
								kernelWeight = scaledKernelPhasedLUTPixel[axis][(phaseIndex * windowSize + windowPixel) * MaxChannels + channel];

							return kernelWeight * lineBuffer[(windowCoord[axis] - windowMinCoord[axis]) * MaxChannels + channel];
							// return lineBuffer[(windowCoord[axis] - windowMinCoord[axis]) * MaxChannels + channel];
						};

						uint32_t phaseIndex = 0;
						// TODO: this loop should probably get rewritten
						for (auto& i=(localTexCoord[axis]=0); i<outExtentLayerCount[axis]; i++)
						{
							// get output pixel
							auto* const value = intermediateStorage[axis]+core::dot(static_cast<const core::vectorSIMDi32&>(intermediateStrides[axis]),localTexCoord)[0];

							// do the filtering
							core::vectorSIMDf tmp;
							tmp[axis] = float(i)+0.5f;
							core::vectorSIMDi32 windowCoord(0);
							windowCoord[axis] = kernel.getWindowMinCoord(tmp*fScale,tmp)[axis];

							for (auto ch = 0; ch < MaxChannels; ++ch)
								value[ch] = getWeightedSample(windowCoord, phaseIndex, 0, ch);

							for (auto h=1; h<windowSize; h++)
							{
								windowCoord[axis]++;

								for (auto ch = 0; ch < MaxChannels; ch++)
									value[ch] += getWeightedSample(windowCoord, phaseIndex, h, ch);
							}
							if (lastPass)
							{
								const core::vectorSIMDu32 localOutPos = localTexCoord+outOffsetBaseLayer+vLayer;
								if (needsNormalization)
									state->normalization.prepass(value,localOutPos,0u,0u,MaxChannels);
								else // store to image, we're done
								{
									core::vectorSIMDu32 dummy(0u);
									storeToTexel(value,outImg->getTexelBlockData(outMipLevel,localOutPos,dummy),localOutPos);
								}
							}

							if (++phaseIndex == phaseCount[axis])
								phaseIndex = 0;
						}
						if (axis==IImage::ET_1D)
							free_decode_scratch(decode_offset);
					});
					// we'll only get here if we have to do coverage adjustment
					if (needsNormalization && lastPass)
						storeToImage(core::rational<int64_t>(inv_cvg_num,inv_cvg_den),axis,outOffsetLayer);
				};
				// filter in X-axis
				filterAxis(IImage::ET_1D,scaledKernelX);
				// filter in Y-axis
				filterAxis(IImage::ET_2D,scaledKernelY);
				// filter in Z-axis
				filterAxis(IImage::ET_3D,scaledKernelZ);
			}
			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}

	private:
		static inline constexpr uint32_t VectorizationBoundSTL = /*AVX2*/16u;
		//
		static inline void getIntermediateExtents(core::vectorSIMDi32* intermediateExtent, const state_type* state, const core::vectorSIMDi32& real_window_size)
		{
			assert(intermediateExtent);

			intermediateExtent[0] = core::vectorSIMDi32(state->outExtent.width, state->inExtent.height + real_window_size[1], state->inExtent.depth + real_window_size[2]);
			intermediateExtent[1] = core::vectorSIMDi32(state->outExtent.width, state->outExtent.height, state->inExtent.depth + real_window_size[2]);
			intermediateExtent[2] = core::vectorSIMDi32(state->outExtent.width, state->outExtent.height, state->outExtent.depth);
		}
		// the blit filter will filter one axis at a time, hence necessitating "ping ponging" between two scratch buffers
		static inline uint32_t getScratchOffset(const state_type* state, bool secondPong)
		{
			const auto inType = state->inImage->getCreationParameters().type;
			const auto scaledKernelX = utils_t::constructScaledKernel(state->kernelX, state->inExtentLayerCount, state->outExtentLayerCount);
			const auto scaledKernelY = utils_t::constructScaledKernel(state->kernelY, state->inExtentLayerCount, state->outExtentLayerCount);
			const auto scaledKernelZ = utils_t::constructScaledKernel(state->kernelZ, state->inExtentLayerCount, state->outExtentLayerCount);

			const auto real_window_size = utils_t::getRealWindowSize(inType,scaledKernelX,scaledKernelY,scaledKernelZ);
			// TODO: account for the size needed for coverage adjustment
			// the first pass will be along X, so new temporary image will have the width of the output extent, but the height and depth will need to be padded
			// but the last pass will be along Z and the new temporary image will have the exact dimensions of `outExtent` which is why there is a `core::max`

			core::vectorSIMDi32 intermediateExtent[3];
			getIntermediateExtents(intermediateExtent, state, real_window_size);

			assert(intermediateExtent[0].x == intermediateExtent[2].x);
			auto texelCount = intermediateExtent[0].x * core::max<uint32_t>(intermediateExtent[0].y*intermediateExtent[0].z, intermediateExtent[2].y * intermediateExtent[2].z);
			if (secondPong)
				texelCount += core::max<uint32_t>(intermediateExtent[1].x * intermediateExtent[1].y * intermediateExtent[1].z, (state->inExtent.width + real_window_size[0]) * std::thread::hardware_concurrency() * VectorizationBoundSTL);
			// obviously we have multiple channels and each channel has a certain type for arithmetic
			return texelCount*utils_t::MaxChannels*sizeof(value_type);
		}
};

} // end namespace nbl::asset

#endif

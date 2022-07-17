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
#include "nbl/asset/filters/CBlitUtilities.h"

#include "nbl/asset/filters/kernels/kernels.h"

#include "nbl/asset/format/decodePixels.h"

namespace nbl::asset
{

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
		using blit_utils_t = CBlitUtilities<KernelX, KernelY, KernelZ>;

	private:
		using value_type = typename KernelX::value_type;
		using base_t = CBlitImageFilterBase<value_type,Swizzle,Dither,Normalization,Clamp>;

		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = blit_utils_t::MaxChannels;

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
				uint32_t							alphaBinCount = blit_utils_t::DefaultAlphaBinCount;
		};
		using state_type = CState;

		enum E_SCRATCH_USAGE
		{
			ESU_SCALED_KERNEL_PHASED_LUT = 0,
			ESU_DECODE_WRITE = 1,
			ESU_BLIT_X_AXIS_READ=ESU_DECODE_WRITE,
			ESU_BLIT_X_AXIS_WRITE=2,
			ESU_BLIT_Y_AXIS_READ = ESU_BLIT_X_AXIS_WRITE,
			ESU_BLIT_Y_AXIS_WRITE = 3,
			ESU_BLIT_Z_AXIS_READ = ESU_BLIT_Y_AXIS_WRITE,
			ESU_BLIT_Z_AXIS_WRITE = 4,
			ESU_ALPHA_HISTOGRAM = 5,
			ESU_COUNT
		};

		//! Call `getScratchOffset(state, ESU_COUNT)` to get the total scratch size needed.
		static inline uint32_t getScratchOffset(const state_type* state, const E_SCRATCH_USAGE usage)
		{
			const auto inType = state->inImage->getCreationParameters().type;
			
			const auto scaledKernelX = asset::IBlitUtilities::constructScaledKernel(state->kernelX, state->inExtentLayerCount, state->outExtentLayerCount);
			const auto scaledKernelY = asset::IBlitUtilities::constructScaledKernel(state->kernelY, state->inExtentLayerCount, state->outExtentLayerCount);
			const auto scaledKernelZ = asset::IBlitUtilities::constructScaledKernel(state->kernelZ, state->inExtentLayerCount, state->outExtentLayerCount);

			const size_t scaledKernelPhasedLUTSize = blit_utils_t::template getScaledKernelPhasedLUTSize<lut_value_type>(state->inExtentLayerCount, state->outExtentLayerCount, inType,
				state->kernelX, state->kernelY, state->kernelZ);

			const auto real_window_size = blit_utils_t::getRealWindowSize(inType, scaledKernelX, scaledKernelY, scaledKernelZ);
			core::vectorSIMDi32 intermediateExtent[3];
			getIntermediateExtents(intermediateExtent, state, real_window_size);
			assert(intermediateExtent[0].x == intermediateExtent[2].x);

			const size_t MaxParallelism = std::thread::hardware_concurrency() * VectorizationBoundSTL;

			uint32_t pingBufferElementCount = (state->inExtent.width + real_window_size[0]) * MaxParallelism; // decode
			uint32_t pongBufferElementCount = intermediateExtent[0].x * intermediateExtent[0].y * intermediateExtent[0].z; // x-axis filter output

			const uint32_t yWriteElementCount = intermediateExtent[1].x * intermediateExtent[1].y * intermediateExtent[1].z;
			if (inType >= IImage::ET_2D && yWriteElementCount > pingBufferElementCount)
				pingBufferElementCount = yWriteElementCount;

			const uint32_t zWriteElementCount = intermediateExtent[2].x * intermediateExtent[2].y * intermediateExtent[2].z;
			if (inType >= IImage::ET_3D && zWriteElementCount > pongBufferElementCount)
				pongBufferElementCount = zWriteElementCount;

			const auto kAlphaHistogramSize = state->alphaBinCount * sizeof(uint32_t);

			uint32_t retval = 0;
			switch (usage)
			{
			case ESU_SCALED_KERNEL_PHASED_LUT:
				return 0;

			case ESU_DECODE_WRITE:
				[[fallthrough]];
			case ESU_BLIT_Y_AXIS_WRITE:
				return scaledKernelPhasedLUTSize;

			case ESU_BLIT_X_AXIS_WRITE:
				[[fallthrough]];
			case ESU_BLIT_Z_AXIS_WRITE:
				return scaledKernelPhasedLUTSize + pingBufferElementCount * MaxChannels * sizeof(value_type);

			case ESU_ALPHA_HISTOGRAM:
				return scaledKernelPhasedLUTSize + (pingBufferElementCount + pongBufferElementCount)*MaxChannels*sizeof(value_type);
				
			default: // ESU_COUNT
			{
				size_t totalScratchSize = scaledKernelPhasedLUTSize + (pingBufferElementCount + pongBufferElementCount) * MaxChannels * sizeof(value_type);
				if (state->alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
					totalScratchSize += kAlphaHistogramSize*MaxParallelism;
				return totalScratchSize;
			}
			}
		}
		
		static inline uint32_t getRequiredScratchByteSize(const state_type* state)
		{
			return getScratchOffset(state, ESU_COUNT);
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
			const auto scaledKernelX = blit_utils_t::constructScaledKernel(state->kernelX, inExtentLayerCount, outExtentLayerCount);
			const auto scaledKernelY = blit_utils_t::constructScaledKernel(state->kernelY, inExtentLayerCount, outExtentLayerCount);
			const auto scaledKernelZ = blit_utils_t::constructScaledKernel(state->kernelZ, inExtentLayerCount, outExtentLayerCount);

			// filtering and alpha handling happens separately for every layer, so save on scratch memory size
			const auto inImageType = inParams.type;
			const auto real_window_size = blit_utils_t::getRealWindowSize(inImageType,scaledKernelX,scaledKernelY,scaledKernelZ);
			core::vectorSIMDi32 intermediateExtent[3];
			getIntermediateExtents(intermediateExtent, state, real_window_size);
			const core::vectorSIMDi32 intermediateLastCoord[3] = {
				intermediateExtent[0]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[1]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[2]-core::vectorSIMDi32(1,1,1,0)
			};
			value_type* const intermediateStorage[3] = {
				reinterpret_cast<value_type*>(state->scratchMemory + getScratchOffset(state, ESU_BLIT_X_AXIS_WRITE)),
				reinterpret_cast<value_type*>(state->scratchMemory + getScratchOffset(state, ESU_BLIT_Y_AXIS_WRITE)),
				reinterpret_cast<value_type*>(state->scratchMemory + getScratchOffset(state, ESU_BLIT_Z_AXIS_WRITE))
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
			auto storeToImage = [policy,coverageSemantic,needsNormalization,outExtent,intermediateStorage,&sampler,outFormat,alphaRefValue,outData,intermediateStrides,alphaChannel,storeToTexel,outMipLevel,outOffset,outRegions,outImg,state](
				const core::rational<int64_t>& coverage, const int axis, const core::vectorSIMDu32& outOffsetLayer
			) -> void
			{
				assert(needsNormalization);
				value_type coverageScale = 1.0;
				if (coverageSemantic) // little thing for the coverage adjustment trick suggested by developer of The Witness
				{
					const auto outputTexelCount = outExtent.width*outExtent.height*outExtent.depth;
					const int64_t pixelsShouldPassCount = (coverage * core::rational<int64_t>(outputTexelCount)).getIntegerApprox();
					const int64_t pixelsShouldFailCount = outputTexelCount - pixelsShouldPassCount;
					// all values with index<=rankIndex will be %==inverseCoverage of the overall array
					const int64_t rankIndex = (coverage*core::rational<int64_t>(outputTexelCount)).getIntegerApprox()-1;

					// Todo(achal): I only use alphaBinCount*sizeof(uint32_t) here because I used atomic, switch to not using atomics.
					uint32_t* histogram = reinterpret_cast<uint32_t*>(state->scratchMemory + getScratchOffset(state, ESU_ALPHA_HISTOGRAM));
					memset(histogram, 0, state->alphaBinCount * sizeof(uint32_t));

					struct DummyTexelType
					{
						double texel[MaxChannels];
					};
					std::for_each(policy, reinterpret_cast<DummyTexelType*>(intermediateStorage[axis]), reinterpret_cast<DummyTexelType*>(intermediateStorage[axis] + outputTexelCount*MaxChannels), [&sampler, outFormat, &histogram, alphaChannel, state](const DummyTexelType& dummyTexel)
					{
						value_type texelAlpha = dummyTexel.texel[alphaChannel];
						texelAlpha -= double(sampler.nextSample()) * (asset::getFormatPrecision<value_type>(outFormat, alphaChannel, texelAlpha) / double(~0u));

						const uint32_t bucketIndex = uint32_t(core::round(core::clamp(texelAlpha, 0.0, 1.0) * double(state->alphaBinCount - 1)));
						assert(bucketIndex < state->alphaBinCount);
						std::atomic_ref(histogram[bucketIndex]).fetch_add(1);
					});

					std::inclusive_scan(histogram, histogram+state->alphaBinCount, histogram);
					const uint32_t bucketIndex = std::lower_bound(histogram, histogram+state->alphaBinCount, pixelsShouldFailCount) - histogram;
					const double newAlphaRefValue = core::min((bucketIndex - 0.5) / double(state->alphaBinCount - 1), 1.0);
					coverageScale = alphaRefValue / newAlphaRefValue;
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
			const core::vectorSIMDu32 axisOffsets = blit_utils_t::template getScaledKernelPhasedLUTAxisOffsets<lut_value_type>(phaseCount, real_window_size);
			constexpr auto MaxAxisCount = 3;
			lut_value_type* scaledKernelPhasedLUTPixel[MaxAxisCount];
			for (auto i = 0; i < MaxAxisCount; ++i)
				scaledKernelPhasedLUTPixel[i] = reinterpret_cast<lut_value_type*>(state->scratchMemory + getScratchOffset(state, ESU_SCALED_KERNEL_PHASED_LUT) + axisOffsets[i]);

			for (uint32_t layer=0; layer!=layerCount; layer++) // TODO: could be parallelized
			{
				const core::vectorSIMDi32 vLayer(0,0,0,layer);
				const auto windowMinCoord = windowMinCoordBase+vLayer;
				const auto outOffsetLayer = outOffsetBaseLayer+vLayer;
				// reset coverage counter
				constexpr bool is_seq_policy_v = std::is_same_v<std::remove_reference_t<ExecutionPolicy>,core::execution::sequenced_policy>;
				using cond_atomic_int32_t = std::conditional_t<is_seq_policy_v,int32_t,std::atomic_int32_t>;
				using cond_atomic_uint32_t = std::conditional_t<is_seq_policy_v,uint32_t,std::atomic_uint32_t>;
				cond_atomic_uint32_t cvg_num(0u);
				cond_atomic_uint32_t cvg_den(0u);
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
										cvg_num++;
									cvg_den++;
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
						storeToImage(core::rational<int64_t>(cvg_num,cvg_den),axis,outOffsetLayer);
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
};

} // end namespace nbl::asset

#endif

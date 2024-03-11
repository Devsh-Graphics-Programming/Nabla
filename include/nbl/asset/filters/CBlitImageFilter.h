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

#include "nbl/asset/format/decodePixels.h"

namespace nbl::asset
{

template<typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CBlitImageFilterBase : public impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>, public CBasicImageFilterCommon
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
template<
	typename Swizzle				= DefaultSwizzle,
	typename Dither					= CWhiteNoiseDither,
	typename Normalization			= void,
	bool Clamp						= true,
	typename BlitUtilities			= CBlitUtilities<>>
class CBlitImageFilter :
	public CImageFilter<CBlitImageFilter<Swizzle, Dither, Normalization, Clamp, BlitUtilities>>,
	public CBlitImageFilterBase<Swizzle, Dither, Normalization, Clamp>
{
	public:
		using blit_utils_t = BlitUtilities;
		static_assert(std::is_base_of_v<IBlitUtilities, blit_utils_t>, "Only template instantiations of CBlitUtilitiesare allowed as theBlitUtilities template argument!");
		using lut_value_t = blit_utils_t::lut_value_type;

	private:
		using value_t = blit_utils_t::value_type;
		using base_t = CBlitImageFilterBase<Swizzle, Dither, Normalization, Clamp>;

		static inline constexpr auto ChannelCount = blit_utils_t::ChannelCount;

	public:
		virtual ~CBlitImageFilter() {}

		enum E_SCRATCH_USAGE
		{
			ESU_SCALED_KERNEL_PHASED_LUT = 0,
			ESU_DECODE_WRITE = 1,
			ESU_BLIT_X_AXIS_READ = ESU_DECODE_WRITE,
			ESU_BLIT_X_AXIS_WRITE = 2,
			ESU_BLIT_Y_AXIS_READ = ESU_BLIT_X_AXIS_WRITE,
			ESU_BLIT_Y_AXIS_WRITE = 3,
			ESU_BLIT_Z_AXIS_READ = ESU_BLIT_Y_AXIS_WRITE,
			ESU_BLIT_Z_AXIS_WRITE = 4,
			ESU_ALPHA_HISTOGRAM = 5,
			ESU_COUNT
		};
		class CState : public IImageFilter::IState, public base_t::CStateBase
		{
			public:
				CState(blit_utils_t::convolution_kernels_t&& _kernels) : kernels(std::move(_kernels))
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}

				CState(const typename blit_utils_t::convolution_kernels_t& _kernels) : kernels(_kernels)
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}

				CState(const CState& other) : IImageFilter::IState(), base_t::CStateBase{other},
					inMipLevel(other.inMipLevel), outMipLevel(other.outMipLevel), inImage(other.inImage), outImage(other.outImage), kernels(other.kernels)
				{
					inOffsetBaseLayer = other.inOffsetBaseLayer;
					inExtentLayerCount = other.inExtentLayerCount;
					outOffsetBaseLayer = other.outOffsetBaseLayer;
					outExtentLayerCount = other.outExtentLayerCount;
				}

				virtual ~CState() {}

				inline bool recomputeScaledKernelPhasedLUT()
				{
					if (!base_t::CStateBase::scratchMemory || !inImage)
						return false;
					const size_t offset = getScratchOffset(this,ESU_SCALED_KERNEL_PHASED_LUT);
					const auto inType = inImage->getCreationParameters().type;
					const size_t size = blit_utils_t::getScaledKernelPhasedLUTSize(inExtentLayerCount,outExtentLayerCount,inType,kernels);
					auto* lut = base_t::CStateBase::scratchMemory+offset;
					return blit_utils_t::computeScaledKernelPhasedLUT(lut,inExtentLayerCount,outExtentLayerCount,inType, kernels);
				}

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
				
				uint32_t								inMipLevel = 0u;
				uint32_t								outMipLevel = 0u;
				ICPUImage*								inImage = nullptr;
				ICPUImage*								outImage = nullptr;
				blit_utils_t::convolution_kernels_t		kernels;
				uint32_t								alphaBinCount = blit_utils_t::DefaultAlphaBinCount;
		};
		using state_type = CState;

		//! Call `getScratchOffset(state, ESU_COUNT)` to get the total scratch size needed.
		static inline uint32_t getScratchOffset(const state_type* state, const E_SCRATCH_USAGE usage)
		{
			const auto inType = state->inImage->getCreationParameters().type;

			const auto windowSize = blit_utils_t::getWindowSize(inType, state->kernels);
			const size_t scaledKernelPhasedLUTSize = blit_utils_t::getScaledKernelPhasedLUTSize(state->inExtentLayerCount, state->outExtentLayerCount, inType, windowSize);

			core::vectorSIMDi32 intermediateExtent[3];
			getIntermediateExtents(intermediateExtent, state, windowSize);
			assert(intermediateExtent[0].x == intermediateExtent[2].x);

			uint32_t pingBufferElementCount = (state->inExtent.width + windowSize[0]) * m_maxParallelism; // decode
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
				return scaledKernelPhasedLUTSize + pingBufferElementCount * ChannelCount * sizeof(value_t);

			case ESU_ALPHA_HISTOGRAM:
				return scaledKernelPhasedLUTSize + (pingBufferElementCount + pongBufferElementCount)*ChannelCount*sizeof(value_t);
				
			default: // ESU_COUNT
			{
				size_t totalScratchSize = scaledKernelPhasedLUTSize + (pingBufferElementCount + pongBufferElementCount) * ChannelCount * sizeof(value_t);
				if (state->alphaSemantic == asset::IBlitUtilities::EAS_REFERENCE_OR_COVERAGE)
					totalScratchSize += kAlphaHistogramSize*m_maxParallelism;
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

			if (state->alphaChannel > ChannelCount)
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

			const auto& kernelX = std::get<0>(state->kernels);
			const auto& kernelY = std::get<1>(state->kernels);
			const auto& kernelZ = std::get<2>(state->kernels);

			return (kernelX.validate(state->inImage, state->outImage) && kernelY.validate(state->inImage, state->outImage) && kernelZ.validate(state->inImage, state->outImage));
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

			// filtering and alpha handling happens separately for every layer, so save on scratch memory size
			const auto inImageType = inParams.type;
			const auto real_window_size = blit_utils_t::getWindowSize(inImageType,state->kernels);
			core::vectorSIMDi32 intermediateExtent[3];
			getIntermediateExtents(intermediateExtent, state, real_window_size);
			const core::vectorSIMDi32 intermediateLastCoord[3] = {
				intermediateExtent[0]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[1]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[2]-core::vectorSIMDi32(1,1,1,0)
			};
			value_t* const intermediateStorage[3] = {
				reinterpret_cast<value_t*>(state->scratchMemory + getScratchOffset(state, ESU_BLIT_X_AXIS_WRITE)),
				reinterpret_cast<value_t*>(state->scratchMemory + getScratchOffset(state, ESU_BLIT_Y_AXIS_WRITE)),
				reinterpret_cast<value_t*>(state->scratchMemory + getScratchOffset(state, ESU_BLIT_Z_AXIS_WRITE))
			};
			const core::vectorSIMDu32 intermediateStrides[3] = {
				core::vectorSIMDu32(ChannelCount*intermediateExtent[0].y,ChannelCount,ChannelCount*intermediateExtent[0].x*intermediateExtent[0].y,0u),
				core::vectorSIMDu32(ChannelCount*intermediateExtent[1].y*intermediateExtent[1].z,ChannelCount*intermediateExtent[1].z,ChannelCount,0u),
				core::vectorSIMDu32(ChannelCount,ChannelCount*intermediateExtent[2].x,ChannelCount*intermediateExtent[2].x*intermediateExtent[2].y,0u)
			};
			// storage
			core::RandomSampler sampler(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			auto storeToTexel = [state,nonPremultBlendSemantic,alphaChannel,outFormat](value_t* const sample, void* const dstPix, const core::vectorSIMDu32& localOutPos) -> void
			{
				if (nonPremultBlendSemantic && sample[alphaChannel]>FLT_MIN*1024.0*512.0)
				{
					for (auto i=0; i<ChannelCount; i++)
					if (i!=alphaChannel)
						sample[i] /= sample[alphaChannel];
				}

				base_t::onEncode(outFormat, state, dstPix, sample, localOutPos, 0, 0, ChannelCount);
			};
			const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(outMipLevel);
			auto storeToImage = [policy,coverageSemantic,needsNormalization,outExtent,intermediateStorage,&sampler,outFormat,alphaRefValue,outData,intermediateStrides,alphaChannel,storeToTexel,outMipLevel,outOffset,outRegions,outImg,state](
				const core::rational<int64_t>& coverage, const int axis, const core::vectorSIMDu32& outOffsetLayer
			) -> void
			{
				assert(needsNormalization);
				value_t coverageScale = 1.0;
				if (coverageSemantic) // little thing for the coverage adjustment trick suggested by developer of The Witness
				{
					const auto outputTexelCount = outExtent.width*outExtent.height*outExtent.depth;
					const int64_t pixelsShouldPassCount = (coverage * core::rational<int64_t>(outputTexelCount)).getIntegerApprox();
					const int64_t pixelsShouldFailCount = outputTexelCount - pixelsShouldPassCount;

					uint32_t* histograms = reinterpret_cast<uint32_t*>(state->scratchMemory + getScratchOffset(state, ESU_ALPHA_HISTOGRAM));
					memset(histograms, 0, m_maxParallelism* state->alphaBinCount * sizeof(uint32_t));

					ParallelScratchHelper scratchHelper;

					constexpr bool is_seq_policy_v = std::is_same_v<std::remove_reference_t<ExecutionPolicy>, core::execution::sequenced_policy>;

					struct DummyTexelType
					{
						double texel[ChannelCount];
					};
					std::for_each(policy, reinterpret_cast<DummyTexelType*>(intermediateStorage[axis]), reinterpret_cast<DummyTexelType*>(intermediateStorage[axis] + outputTexelCount*ChannelCount), [&sampler, outFormat, &histograms, &scratchHelper, alphaChannel, state](const DummyTexelType& dummyTexel)
					{
						const uint32_t index = scratchHelper.template alloc<is_seq_policy_v>();

						value_t texelAlpha = dummyTexel.texel[alphaChannel];
						texelAlpha -= double(sampler.nextSample()) * (asset::getFormatPrecision<value_t>(outFormat, alphaChannel, texelAlpha) / double(~0u));

						const uint32_t binIndex = uint32_t(core::round(core::clamp(texelAlpha, 0.0, 1.0) * double(state->alphaBinCount - 1)));
						assert(binIndex < state->alphaBinCount);
						histograms[index*state->alphaBinCount+binIndex]++;

						scratchHelper.template free<is_seq_policy_v>(index);
					});

					uint32_t* mergedHistogram = histograms;
					for (auto hi = 1; hi < m_maxParallelism; ++hi)
					{
						for (auto bi = 0; bi < state->alphaBinCount; ++bi)
							histograms[bi] += histograms[hi * state->alphaBinCount + bi];
					}

					std::inclusive_scan(mergedHistogram, mergedHistogram +state->alphaBinCount, mergedHistogram);
					const uint32_t binIndex = std::lower_bound(mergedHistogram, mergedHistogram +state->alphaBinCount, pixelsShouldFailCount) - mergedHistogram;
					const double newAlphaRefValue = core::min((binIndex - 0.5) / double(state->alphaBinCount - 1), 1.0);
					coverageScale = alphaRefValue / newAlphaRefValue;
				}
				auto scaleCoverage = [outData,outOffsetLayer,intermediateStrides,axis,intermediateStorage,alphaChannel,coverageScale,storeToTexel](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 writeBlockPos) -> void
				{
					void* const dstPix = outData+writeBlockArrayOffset;
					const core::vectorSIMDu32 localOutPos = writeBlockPos - outOffsetLayer;

					value_t sample[ChannelCount];
					const size_t offset = IImage::SBufferCopy::getLocalByteOffset(localOutPos, intermediateStrides[axis]);
					const auto* first = intermediateStorage[axis]+offset;
					std::copy(first,first+ChannelCount,sample);

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
			const auto startCoord =  [&halfTexelOffset,state]() -> core::vectorSIMDi32
			{
				return core::vectorSIMDi32(
					std::get<0>(state->kernels).getWindowMinCoord(halfTexelOffset.x),
					std::get<1>(state->kernels).getWindowMinCoord(halfTexelOffset.y),
					std::get<2>(state->kernels).getWindowMinCoord(halfTexelOffset.z),0);
			}();
			const auto windowMinCoordBase = inOffsetBaseLayer+startCoord;

			core::vectorSIMDu32 phaseCount = IBlitUtilities::getPhaseCount(inExtentLayerCount, outExtentLayerCount, inImageType);
			phaseCount = core::max(phaseCount, core::vectorSIMDu32(1, 1, 1));
			const core::vectorSIMDu32 axisOffsets = blit_utils_t::template getScaledKernelPhasedLUTAxisOffsets(phaseCount, real_window_size);
			constexpr auto MaxAxisCount = 3;
			lut_value_t* scaledKernelPhasedLUTPixel[MaxAxisCount];
			for (auto i = 0; i < MaxAxisCount; ++i)
				scaledKernelPhasedLUTPixel[i] = reinterpret_cast<lut_value_t*>(state->scratchMemory + getScratchOffset(state, ESU_SCALED_KERNEL_PHASED_LUT) + axisOffsets[i]);

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
					const auto windowSize = kernel.getWindowSize();
	
					// z y x output along x
					// z x y output along y
					// x y z output along z
					const int loopCoordID[2] = {/*axis,*/axis!=IImage::ET_2D ? 1:0,axis!=IImage::ET_3D ? 2:0};
					//
					assert(is_seq_policy_v || std::thread::hardware_concurrency()<=64u);
					ParallelScratchHelper scratchHelper;

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
						constexpr bool is_seq_policy_v = std::is_same_v<std::remove_reference_t<ExecutionPolicy>, core::execution::sequenced_policy>;

						// we need some tmp memory for threads in the first pass so that they dont step on each other
						uint32_t decode_offset;
						// whole line plus window borders
						value_t* lineBuffer;
						core::vectorSIMDi32 localTexCoord(0);
						localTexCoord[loopCoordID[0]] = batchCoord[0];
						localTexCoord[loopCoordID[1]] = batchCoord[1];
						if (axis!=IImage::ET_1D)
							lineBuffer = intermediateStorage[axis-1]+core::dot(static_cast<const core::vectorSIMDi32&>(intermediateStrides[axis-1]),localTexCoord)[0];
						else
						{
							const auto inputEnd = inExtent.width+real_window_size.x;
							decode_offset = scratchHelper.template alloc<is_seq_policy_v>();
							lineBuffer = intermediateStorage[1]+decode_offset*ChannelCount*inputEnd;
							for (auto& i=localTexCoord.x; i<inputEnd; i++)
							{
								core::vectorSIMDi32 globalTexelCoord(localTexCoord+windowMinCoord);

								core::vectorSIMDu32 blockLocalTexelCoord(0u);
								const void* srcPix[] = { // multiple loads for texture boundaries aren't that bad
									inImg->getTexelBlockData(inMipLevel,inImg->wrapTextureCoordinate(inMipLevel,globalTexelCoord,axisWraps),blockLocalTexelCoord),
									nullptr,
									nullptr,
									nullptr
								};
								if (!srcPix[0])
									continue;

								auto sample = lineBuffer+i*ChannelCount;

								base_t::template onDecode(inFormat, state, srcPix, sample, blockLocalTexelCoord.x, blockLocalTexelCoord.y, ChannelCount);

								if (nonPremultBlendSemantic)
								{
									for (auto i=0; i<ChannelCount; i++)
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

						auto getWeightedSample = [scaledKernelPhasedLUTPixel, windowSize, lineBuffer, &windowMinCoord, axis](const auto& windowCoord, const auto phaseIndex, const auto windowPixel, const auto channel) -> value_t
						{
							value_t kernelWeight;
							if constexpr (std::is_same_v<lut_value_t, uint16_t>)
								kernelWeight = value_t(core::Float16Compressor::decompress(scaledKernelPhasedLUTPixel[axis][(phaseIndex * windowSize + windowPixel) * ChannelCount + channel]));
							else
								kernelWeight = scaledKernelPhasedLUTPixel[axis][(phaseIndex * windowSize + windowPixel) * ChannelCount + channel];

							return kernelWeight * lineBuffer[(windowCoord - windowMinCoord[axis]) * ChannelCount + channel];
						};

						uint32_t phaseIndex = 0;
						// TODO: this loop should probably get rewritten
						for (auto& i=(localTexCoord[axis]=0); i<outExtentLayerCount[axis]; i++)
						{
							// get output pixel
							auto* const value = intermediateStorage[axis]+core::dot(static_cast<const core::vectorSIMDi32&>(intermediateStrides[axis]),localTexCoord)[0];

							// do the filtering
							float tmp = float(i)+0.5f;
							int32_t windowCoord = kernel.getWindowMinCoord(tmp*fScale[axis], tmp);

							for (auto ch = 0; ch < ChannelCount; ++ch)
								value[ch] = getWeightedSample(windowCoord, phaseIndex, 0, ch);

							for (auto h=1; h<windowSize; h++)
							{
								windowCoord++;

								for (auto ch = 0; ch < ChannelCount; ch++)
									value[ch] += getWeightedSample(windowCoord, phaseIndex, h, ch);
							}
							if (lastPass)
							{
								const core::vectorSIMDu32 localOutPos = localTexCoord+outOffsetBaseLayer+vLayer;
								if (needsNormalization)
									state->normalization.prepass(value,localOutPos,0u,0u,ChannelCount);
								else // store to image, we're done
								{
									core::vectorSIMDu32 dummy(0u);
									storeToTexel(value,outImg->getTexelBlockData(outMipLevel,localOutPos,dummy),localOutPos);
								}
							}

							if (++phaseIndex == phaseCount[axis])
								phaseIndex = 0;
						}
						if (axis == IImage::ET_1D)
							scratchHelper.template free<is_seq_policy_v>(decode_offset);
					});
					// we'll only get here if we have to do coverage adjustment
					if (needsNormalization && lastPass)
					{
						state->normalization.finalize<value_t>();
						storeToImage(core::rational<int64_t>(cvg_num,cvg_den),axis,outOffsetLayer);
					}
				};
				
				filterAxis(IImage::ET_1D, std::get<0>(state->kernels));
				filterAxis(IImage::ET_2D, std::get<1>(state->kernels));
				filterAxis(IImage::ET_3D, std::get<2>(state->kernels));
			}
			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}

	private:
		static inline constexpr uint32_t VectorizationBoundSTL = /*AVX2*/16u;
		static inline const uint32_t m_maxParallelism = std::thread::hardware_concurrency() * VectorizationBoundSTL;

		class ParallelScratchHelper
		{
		public:
			ParallelScratchHelper()
			{
				std::fill_n(indices, VectorizationBoundSTL, ~0ull);
			}

			template<bool isSeqPolicy>
			inline uint32_t alloc()
			{
				if constexpr (isSeqPolicy)
					return 0;

				std::unique_lock<std::mutex> lock(mutex);
				for (uint32_t j = 0u; j < VectorizationBoundSTL; ++j)
				{
					int32_t firstFree = hlsl::findLSB(indices[j]);
					if (firstFree != -1)
					{
						indices[j] ^= (0x1u << firstFree); // mark using
						return j * MaxCores + firstFree;
					}
				}
				assert(false);
				return 0xdeadbeef;
			}

			template<bool isSeqPolicy>
			inline void free(const uint32_t index)
			{
				if constexpr (!isSeqPolicy)
				{
					std::unique_lock<std::mutex> lock(mutex);
					indices[index / MaxCores] ^= (0x1u << (index % MaxCores)); // mark free
				}
			}

		private:
			static inline constexpr auto MaxCores = 64;

			uint64_t indices[VectorizationBoundSTL];
			std::mutex mutex;
		};

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

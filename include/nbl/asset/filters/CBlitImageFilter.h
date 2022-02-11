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


template<typename value_type, typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CBlitImageFilterBase : public impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>, public CBasicImageFilterCommon
{
	public:
		class CStateBase : public impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>::state_type
		{
			public:
				CStateBase() {}
				virtual ~CStateBase() {}

				enum E_ALPHA_SEMANTIC : uint32_t
				{
					EAS_NONE_OR_PREMULTIPLIED = 0u, // just filter the channels independently (also works for a texture for blending equation `dstCol*(1-srcAlpha)+srcCol`)
					EAS_REFERENCE_OR_COVERAGE, // try to preserve coverage (percentage of pixels above a threshold value) across mipmap levels
					EAS_SEPARATE_BLEND, // compute a new alpha value for a texture to be used with the blending equation `mix(dstCol,srcCol,srcAlpha)`
					EAS_COUNT
				};

				// we need scratch memory because we'll decode the whole image into one contiguous chunk of memory for faster filtering amongst other things
				uint8_t*							scratchMemory = nullptr;
				uint32_t							scratchMemoryByteSize = 0u;
				_NBL_STATIC_INLINE_CONSTEXPR auto	NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP			axisWraps[NumWrapAxes] = { ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT };
				ISampler::E_TEXTURE_BORDER_COLOR	borderColor = ISampler::ETBC_FLOAT_TRANSPARENT_BLACK;
				E_ALPHA_SEMANTIC					alphaSemantic = EAS_NONE_OR_PREMULTIPLIED;
				double								alphaRefValue = 0.5; // only required to make sense if `alphaSemantic==EAS_REFERENCE_OR_COVERAGE`
				uint32_t							alphaChannel = 3u; // index of the alpha channel (could be different cause of swizzles)
		};

	protected:
		CBlitImageFilterBase() {}
		virtual ~CBlitImageFilterBase() {}

		// this will be called by derived classes because it doesn't account for all scratch needed, just the stuff for coverage adjustment
		static inline uint32_t getRequiredScratchByteSize(	typename CStateBase::E_ALPHA_SEMANTIC alphaSemantic=CStateBase::EAS_NONE_OR_PREMULTIPLIED,
															const core::vectorSIMDu32& outExtentLayerCount=core::vectorSIMDu32(0,0,0,0))
		{
			uint32_t retval = 0u;
			// 
			if (alphaSemantic==CStateBase::EAS_REFERENCE_OR_COVERAGE)
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

			if (state->alphaSemantic>=CStateBase::EAS_COUNT)
				return false;

			if (state->alphaChannel>=4)
				return false;

			if (!impl::CSwizzleableAndDitherableFilterBase<Swizzle,Dither,Normalization,Clamp>::validate(state))
				return false;

			return true;
		}
};

// copy while filtering the input into the output, a rare filter where the input and output extents can be different, still works one mip level at a time
template<typename Swizzle=DefaultSwizzle, typename Dither=CWhiteNoiseDither, typename Normalization=void, bool Clamp=true, class KernelX=CBoxImageFilterKernel, class KernelY=KernelX, class KernelZ=KernelX>
class CBlitImageFilter : public CImageFilter<CBlitImageFilter<Swizzle,Dither,Normalization,Clamp,KernelX,KernelX,KernelX>>, public CBlitImageFilterBase<typename KernelX::value_type,Swizzle,Dither,Normalization,Clamp>
{
		static_assert(std::is_same<typename KernelX::value_type,typename KernelY::value_type>::value&&std::is_same<typename KernelZ::value_type,typename KernelY::value_type>::value,"Kernel value_type need to be identical");
		using value_type = typename KernelX::value_type;
		
		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = KernelX::MaxChannels>KernelY::MaxChannels ? (KernelX::MaxChannels>KernelZ::MaxChannels ? KernelX::MaxChannels:KernelZ::MaxChannels):(KernelY::MaxChannels>KernelZ::MaxChannels ? KernelY::MaxChannels:KernelZ::MaxChannels);

		using base_t = CBlitImageFilterBase<value_type,Swizzle,Dither,Normalization,Clamp>;

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

				// we'll need to rescale the kernel support to be relative to the output image but in the input image coordinate system
				// (if support is 3 pixels, it needs to be 3 output texels, but measured in input texels)
				template<class Kernel>
				inline auto contructScaledKernel(const Kernel& kernel) const
				{
					const core::vectorSIMDf fInExtent(inExtentLayerCount);
					const core::vectorSIMDf fOutExtent(outExtentLayerCount);
					const auto fScale = fInExtent.preciseDivision(fOutExtent);
					return CScaledImageFilterKernel<Kernel>(fScale,kernel);
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
				
				uint32_t				inMipLevel = 0u;
				uint32_t				outMipLevel = 0u;
				ICPUImage*				inImage = nullptr;
				ICPUImage*				outImage = nullptr;
				KernelX					kernelX;
				KernelY					kernelY;
				KernelZ					kernelZ;
		};
		using state_type = CState;
		
		static inline uint32_t getRequiredScratchByteSize(const state_type* state)
		{
			// need to add the memory for ping pong buffers
			uint32_t retval = getScratchOffset(state,true);
			retval += base_t::getRequiredScratchByteSize(state->alphaSemantic,state->outExtentLayerCount);
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
			if (state->alphaSemantic!=CState::EAS_NONE_OR_PREMULTIPLIED && (getFormatChannelCount(inFormat)!=4u||getFormatChannelCount(outFormat)!=4u))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(outFormat))
				return false;

			return state->kernelX.validate(state->inImage,state->outImage)&&state->kernelY.validate(state->inImage,state->outImage)&&state->kernelZ.validate(state->inImage,state->outImage);
		}

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
			const bool nonPremultBlendSemantic = state->alphaSemantic==CState::EAS_SEPARATE_BLEND;
			// TODO: reformulate coverage adjustment as a normalization
			const bool coverageSemantic = state->alphaSemantic==CState::EAS_REFERENCE_OR_COVERAGE;
			const bool needsNormalization = !std::is_void_v<Normalization> || coverageSemantic;
			const auto alphaRefValue = state->alphaRefValue;
			const auto alphaChannel = state->alphaChannel;
			
			// prepare kernel
			const auto kernelX = state->contructScaledKernel(state->kernelX);
			const auto kernelY = state->contructScaledKernel(state->kernelY);
			const auto kernelZ = state->contructScaledKernel(state->kernelZ);

			// filtering and alpha handling happens separately for every layer, so save on scratch memory size
			const auto inImageType = inParams.type;
			const auto window_end = getWindowEnd(inImageType,kernelX,kernelY,kernelZ);
			const core::vectorSIMDi32 intermediateExtent[3] = {
				core::vectorSIMDi32(outExtent.width,inExtent.height+window_end[1],inExtent.depth+window_end[2]),
				core::vectorSIMDi32(outExtent.width,outExtent.height,inExtent.depth+window_end[2]),
				core::vectorSIMDi32(outExtent.width,outExtent.height,outExtent.depth)
			};
			const core::vectorSIMDi32 intermediateLastCoord[3] = {
				intermediateExtent[0]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[1]-core::vectorSIMDi32(1,1,1,0),
				intermediateExtent[2]-core::vectorSIMDi32(1,1,1,0)
			};
			value_type* const intermediateStorage[3] = {
				reinterpret_cast<value_type*>(state->scratchMemory),
				reinterpret_cast<value_type*>(state->scratchMemory+getScratchOffset(state,false)),
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
				const core::rational<>& inverseCoverage, const int axis, const core::vectorSIMDu32& outOffsetLayer
			) -> void
			{
				assert(needsNormalization);
				value_type coverageScale = 1.0;
				if (coverageSemantic) // little thing for the coverage adjustment trick suggested by developer of The Witness
				{
					const auto outputTexelCount = outExtent.width*outExtent.height*outExtent.depth;
					// all values with index<=rankIndex will be %==inverseCoverage of the overall array
					const int32_t rankIndex = (inverseCoverage*core::rational<int32_t>(outputTexelCount)).getIntegerApprox()-1;
					auto* const begin = intermediateStorage[(axis+1)%3];
					// this is our new reference value
					auto* const nth = begin+core::max<int32_t>(rankIndex,0);
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
			const auto startCoord =  [&halfTexelOffset,&kernelX,&kernelY,&kernelZ]() -> core::vectorSIMDi32
			{
				return core::vectorSIMDi32(kernelX.getWindowMinCoord(halfTexelOffset).x-1,kernelY.getWindowMinCoord(halfTexelOffset).y-1,kernelZ.getWindowMinCoord(halfTexelOffset).z-1,0);
			}();
			const auto windowMinCoordBase = inOffsetBaseLayer+startCoord;
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
					const uint32_t batchExtent[batch_dims] = {intermediateExtent[axis][loopCoordID[0]],intermediateExtent[axis][loopCoordID[1]]};
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
							const auto windowEnd = inExtent.width+window_end.x;
							decode_offset = alloc_decode_scratch();
							lineBuffer = intermediateStorage[1]+decode_offset*MaxChannels*windowEnd;
							for (auto& i=localTexCoord.x; i<windowEnd; i++)
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
						// TODO: this loop should probably get rewritten
						for (auto& i=(localTexCoord[axis]=0); i<outExtentLayerCount[axis]; i++)
						{
							// get output pixel
							auto* const value = intermediateStorage[axis]+core::dot(static_cast<const core::vectorSIMDi32&>(intermediateStrides[axis]),localTexCoord)[0];
							std::fill(value,value+MaxChannels,value_type(0));
							// kernel load functor
							auto load = [axis,&windowMinCoord,lineBuffer](value_type* windowSample, const core::vectorSIMDf& unused0, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData) -> void
							{
								for (auto h=0; h<MaxChannels; h++)
									windowSample[h] = lineBuffer[(globalTexelCoord[axis]-windowMinCoord[axis])*MaxChannels+h];
							};
							// kernel evaluation functor
							auto evaluate = [value](const value_type* windowSample, const core::vectorSIMDf& unused0, const core::vectorSIMDi32& unused1, const IImageFilterKernel::UserData* userData) -> void
							{
								for (auto h=0; h<MaxChannels; h++)
									value[h] += windowSample[h];
							};
							// do the filtering 
							core::vectorSIMDf tmp;
							tmp[axis] = float(i)+0.5f;
							core::vectorSIMDi32 windowCoord(0);
							windowCoord[axis] = kernel.getWindowMinCoord(tmp*fScale,tmp)[axis];
							auto relativePos = tmp[axis]-float(windowCoord[axis]);
							for (auto h=0; h<windowSize; h++)
							{
								value_type windowSample[MaxChannels];

								core::vectorSIMDf tmp(relativePos,0.f,0.f);
								kernel.evaluateImpl(load,evaluate,windowSample,tmp,windowCoord,&scale);
								relativePos -= 1.f;
								windowCoord[axis]++;
							}
							if (lastPass)
							{
								const core::vectorSIMDu32 localOutPos = localTexCoord+outOffsetBaseLayer;
								if (needsNormalization)
									state->normalization.prepass(value,localOutPos,0u,0u,MaxChannels);
								else // store to image, we're done
								{
									core::vectorSIMDu32 dummy(0u);
									storeToTexel(value,outImg->getTexelBlockData(outMipLevel,localOutPos,dummy),localOutPos);
								}
							}
						}
						if (axis==IImage::ET_1D)
							free_decode_scratch(decode_offset);
					});
					// we'll only get here if we have to do coverage adjustment
					if (needsNormalization && lastPass)
						storeToImage(core::rational<>(inv_cvg_num,inv_cvg_den),axis,outOffsetLayer);
				};
				// filter in X-axis
				filterAxis(IImage::ET_1D,kernelX);
				// filter in Y-axis
				filterAxis(IImage::ET_2D,kernelY);
				// filter in Z-axis
				assert(inImageType!=IImage::ET_3D); // TODO: Need to test this in the future
				filterAxis(IImage::ET_3D,kernelZ);
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
		static inline core::vectorSIMDi32 getWindowEnd(const IImage::E_TYPE inImageType,
			const CScaledImageFilterKernel<KernelX>& kernelX,
			const CScaledImageFilterKernel<KernelY>& kernelY,
			const CScaledImageFilterKernel<KernelZ>& kernelZ
		)
		{
			// TODO: investigate properly if its supposed be `size` or `size-1` (polyphase kinda shows need for `size`)
			core::vectorSIMDi32 last(kernelX.getWindowSize().x,0,0,0);
			if (inImageType>=IImage::ET_2D)
				last.y = kernelY.getWindowSize().x;
			if (inImageType>=IImage::ET_3D)
				last.z = kernelZ.getWindowSize().x;
			return last;
		}
		// the blit filter will filter one axis at a time, hence necessitating "ping ponging" between two scratch buffers
		static inline uint32_t getScratchOffset(const state_type* state, bool secondPong)
		{
			const auto inType = state->inImage->getCreationParameters().type;
			const auto kernelX = state->contructScaledKernel(state->kernelX);
			const auto kernelY = state->contructScaledKernel(state->kernelY);
			const auto kernelZ = state->contructScaledKernel(state->kernelZ);

			const auto window_end = getWindowEnd(state->inImage->getCreationParameters().type,kernelX,kernelY,kernelZ);
			// TODO: account for the size needed for coverage adjustment
			// the first pass will be along X, so new temporary image will have the width of the output extent, but the height and depth will need to be padded
			// but the last pass will be along Z and the new temporary image will have the exact dimensions of `outExtent` which is why there is a `core::max`
			auto texelCount = state->outExtent.width*core::max<uint32_t>((state->inExtent.height+window_end[1])*(state->inExtent.depth+window_end[2]),state->outExtent.height*state->outExtent.depth);
			// the second pass will result in an image that has the width and height equal to `outExtent`
			if (secondPong)
				texelCount += core::max<uint32_t>(state->outExtent.width*state->outExtent.height*(state->inExtent.depth+window_end[2]),(state->inExtent.width+window_end[0])*std::thread::hardware_concurrency()*VectorizationBoundSTL);
			// obviously we have multiple channels and each channel has a certain type for arithmetic
			return texelCount*MaxChannels*sizeof(value_type);
		}
};

} // end namespace nbl::asset

#endif

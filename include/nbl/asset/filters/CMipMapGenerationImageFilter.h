// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_MIP_MAP_GENERATION_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_MIP_MAP_GENERATION_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/filters/CBlitImageFilter.h"

namespace nbl
{
namespace asset
{

// Could be viewed as a specialized case of CBlitImageFilter, functionality is not derived, its composed.
// Each mip-map will be constructed from the convolution of the previous by the scaled kernel.
// For Sinc/Lanczos and Box Kernels this hierarchical computation gives the exact result.
// However if you desire to use other filters then you should carefully consider the implications
// For example in the gaussian case you want to filter with 2/originalResolution, 4/originalResolution, 8/originalResolution supports
// but iterative application of the filter will give you 2/originalResolution, 6/originalResolution, 14/originalResolution supports
// the correct usage is to compute the first mip map with a 100% support kernel, then subsequent iterations with 50% smaller pixel supports
// (actually in the case of using a Gaussian for both resampling and reconstruction, this is equivalent to using a single kernel of 3,3,5,9,..)

template<bool Normalize = false, bool Clamp = false, typename Swizzle = VoidSwizzle, typename Dither = IdentityDither, class ResamplingKernelX = CKaiserImageFilterKernel<>, class ReconstructionKernelX = CMitchellImageFilterKernel<>, class ResamplingKernelY = ResamplingKernelX, class ReconstructionKernelY = ReconstructionKernelX, class ResamplingKernelZ = ResamplingKernelY, class ReconstructionKernelZ = ReconstructionKernelY>
class CMipMapGenerationImageFilter : public CImageFilter<CMipMapGenerationImageFilter<Normalize, Clamp, Swizzle, Dither, ResamplingKernelX, ReconstructionKernelX, ResamplingKernelY, ReconstructionKernelY, ResamplingKernelZ, ReconstructionKernelZ> >, public CBasicImageFilterCommon
{
	public:
		virtual ~CMipMapGenerationImageFilter() {}

		// TODO: Improve and implement the cached convolution kernel
		using KernelX = ResamplingKernelX;//CKernelConvolution<ResamplingKernelX, ReconstructionKernelX>;
		using KernelY = ResamplingKernelY;//CKernelConvolution<ResamplingKernelY, ReconstructionKernelY>;
		using KernelZ = ResamplingKernelZ;//CKernelConvolution<ResamplingKernelZ, ReconstructionKernelZ>;

		class CState : public IImageFilter::IState, public CBlitImageFilterBase<typename KernelX::value_type,Normalize,Clamp,Swizzle,Dither>::CStateBase
		{
			public:
				virtual ~CState() {}

				uint32_t							baseLayer= 0u;
				uint32_t							layerCount = 0u;
				uint32_t							startMipLevel = 1u;
				uint32_t							endMipLevel = 0u;
				ICPUImage*							inOutImage = nullptr;
		};
		using state_type = CState;
		
		// since the only thing the mip map generator does is call the blit filter, the scratch memory amount is the same
		static inline uint32_t getRequiredScratchByteSize(const state_type* state)
		{
			auto blit = buildBlitState(state,state->startMipLevel);
			return CBlitImageFilter<Normalize,Clamp,Swizzle,Dither,KernelX>::getRequiredScratchByteSize(&blit);
		}

		static inline bool validate(state_type* state)
		{
			if (!state)
				return false;

			auto* const image = state->inOutImage;
			if (!image)
				return false;

			const auto& params = image->getCreationParameters();

			if (state->baseLayer+state->layerCount>params.arrayLayers)
				return false;
			if (state->startMipLevel>=state->endMipLevel || state->endMipLevel>params.mipLevels)
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(state->inOutImage->getCreationParameters().format))
				return false;
			
			for (auto inMipLevel=state->startMipLevel; inMipLevel!=state->endMipLevel; inMipLevel++)
			{
				auto blit = buildBlitState(state, inMipLevel);
				if (!CBlitImageFilter<Normalize,Clamp,Swizzle,Dither,KernelX>::validate(&blit))
					return false;
			}
			return true; // CBlit already checks kernel
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			for (auto inMipLevel=state->startMipLevel; inMipLevel!=state->endMipLevel; inMipLevel++)
			{
				auto blit = buildBlitState(state, inMipLevel);
				return false;
				if (!CBlitImageFilter<Normalize,Clamp,Swizzle,Dither,KernelX>::template execute<ExecutionPolicy>(std::forward<ExecutionPolicy>(policy),&blit))
					return false;
			}
			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}

	protected:
		static inline auto buildBlitState(const state_type* state, uint32_t inMipLevel)
		{
			const auto prevLevel = inMipLevel-1u;

			typename CBlitImageFilter<Normalize,Clamp,Swizzle,Dither,KernelX>::state_type blit;
			blit.inOffsetBaseLayer = blit.outOffsetBaseLayer = core::vectorSIMDu32(0, 0, 0, state->baseLayer);
			blit.inExtentLayerCount = state->inOutImage->getMipSize(prevLevel);
			blit.outExtentLayerCount = state->inOutImage->getMipSize(inMipLevel);
			blit.inLayerCount = blit.outLayerCount = state->layerCount;
			blit.inMipLevel = prevLevel;
			blit.outMipLevel = inMipLevel;
			blit.inImage = blit.outImage = state->inOutImage;
			//not all kernels are default-constructible, this is going to be a problem (i already added appropriate ctor for blit filter state class though)
			//blit.kernel = Kernel(); // gets default constructed, we should probably do a `static_assert` about this property
			using state_base_t = typename CBlitImageFilterBase<typename KernelX::value_type, Normalize,Clamp,Swizzle,Dither>::CStateBase;
			static_cast<state_base_t&>(blit) = *static_cast<const state_base_t*>(state);
			return blit;
		}
};


} // end namespace asset
} // end namespace nbl

#endif
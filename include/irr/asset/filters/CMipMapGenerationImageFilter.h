// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_MIP_MAP_GENERATION_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_MIP_MAP_GENERATION_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/CBlitImageFilter.h"

namespace irr
{
namespace asset
{

// Could be viewed as a specialized case of CBlitImageFilter, functionality is not derived, its composed.
template<class ResamplingKernel=CKaiserImageFilterKernel<>, class ReconstructionKernel=CMitchellImageFilterKernel<> >
class CMipMapGenerationImageFilter : public CImageFilter<CMipMapGenerationImageFilter<ResamplingKernel,ReconstructionKernel> >, public CBasicImageFilterCommon
{
	public:
		virtual ~CMipMapGenerationImageFilter() {}

		// TODO: Improve
		using Kernel = ResamplingKernel;//CKernelConvolution<ResamplingKernel, ReconstructionKernel>;

		class CProtoState : public IImageFilter::IState
		{
			public:
				virtual ~CProtoState() {}

				uint32_t							baseLayer= 0u;
				uint32_t							layerCount = 0u;
				uint32_t							startMipLevel = 1u;
				uint32_t							endMipLevel = 0u;
				ICPUImage*							inOutImage = nullptr;
		};
		class CState : public CProtoState, public CBlitImageFilterBase::CStateBase
		{
		};
		using state_type = CState;
		

		static inline uint32_t getRequiredScratchByteSize(const state_type* state)
		{
			auto blit = buildBlitState(state,state->startMipLevel);
			return CBlitImageFilter<Kernel>::getRequiredScratchByteSize(&blit);
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

			return Kernel::validate(image,image);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			for (auto inMipLevel=state->startMipLevel; inMipLevel!=state->endMipLevel; inMipLevel++)
			{
				auto blit = buildBlitState(state, inMipLevel);
				if (!CBlitImageFilter<Kernel>::execute(&blit))
					return false;
			}
			return true;
		}

	protected:
		static inline auto buildBlitState(const state_type* state, uint32_t inMipLevel)
		{
			const auto prevLevel = inMipLevel-1u;

			typename CBlitImageFilter<Kernel>::state_type blit;
			blit.inOffsetBaseLayer = blit.outOffsetBaseLayer = core::vectorSIMDu32(0, 0, 0, state->baseLayer);
			blit.inExtentLayerCount = state->inOutImage->getMipSize(prevLevel);
			blit.outExtentLayerCount = state->inOutImage->getMipSize(inMipLevel);
			blit.inLayerCount = blit.outLayerCount = state->layerCount;
			blit.inMipLevel = prevLevel;
			blit.outMipLevel = inMipLevel;
			blit.inImage = blit.outImage = state->inOutImage;
			//blit.kernel = Kernel();
			static_cast<typename CBlitImageFilterBase::CStateBase&>(blit) = *static_cast<const typename CBlitImageFilterBase::CStateBase*>(state);
			return blit;
		}
};


} // end namespace asset
} // end namespace irr

#endif
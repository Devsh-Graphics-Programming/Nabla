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

// specialized case of CBlitImageFilter
template<class ResamplingKernel=CTriangleImageFilterKernel, class ReconstructionKernel=CTriangleImageFilterKernel>
class CMipMapGenerationImageFilter : public CImageFilter<CMipMapGenerationImageFilter<ResamplingKernel,ReconstructionKernel> >
{
	public:
		virtual ~CMipMapGenerationImageFilter() {}
		
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				uint32_t	baseLayer= 0u;
				uint32_t	layerCount = 0u;
				uint32_t	startMipLevel = 1u;
				uint32_t	endMipLevel = 0u;
				ICPUImage*	inOutImage = nullptr;
		};
		using state_type = CState;

		using Kernel = ReconstructionKernel;//CKernelConvolution<ResamplingKernel, ReconstructionKernel>;

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

			return Kernel::validate(inImage,outImage);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			for (auto inMipLevel=state->startMipLevel; inMipLevel!=state->endMipLevel; inMipLevel++)
			{
				const auto prevLevel = inMipLevel-1u;

				CBlitImageFilter<Kernel>::state_type blit;
				blit.inOffsetBaseLayer = blit.outOffsetBaseLayer = core::vectorSIMDu32(0,0,0,state->baseLayer);
				blit.inExtentLayerCount = state->inOutImage->getMipSize(prevLevel);
				blit.outExtentLayerCount = state->inOutImage->getMipSize(inMipLevel);
				blit.inLayerCount = blit.outLayerCount = state->layerCount;
				blit.inMipLevel = prevLevel;
				blit.outMipLevel = inMipLevel;
				blit.inImage = blit.outImage = state->inOutImage;
				if (!CBlitImageFilter<Kernel>::execute(&blit))
					return false;
			}
			return true;
		}
};


} // end namespace asset
} // end namespace irr

#endif
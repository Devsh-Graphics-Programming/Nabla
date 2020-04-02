// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_MATCHED_SIZE_IN_OUT_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __IRR_C_MATCHED_SIZE_IN_OUT_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
{
namespace asset
{

class CMatchedSizeInOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				union
				{
					core::vectorSIMDu32 extentLayerCount;
					struct
					{
						VkExtent3D		extent;
						uint32_t		layerCount;
					};
				};
				union
				{
					core::vectorSIMDu32 inOffsetBaseLayer;
					struct
					{
						VkOffset3D		inOffset;
						uint32_t		inBaseLayer;
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
				uint32_t				inMipLevel = 0u;
				uint32_t				outMipLevel = 0u;
				ICPUImage*				inImage = nullptr;
				ICPUImage*				outImage = nullptr;
		};
		using state_type = CState;
		
		static inline bool validate(state_type* state)
		{
			if (!state)
				return nullptr;

			IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
			state_type::TexelRange range = {state->inOffset,state->extent};
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource,range,state->inImage))
				return false;
			subresource.mipLevel = state->outMipLevel;
			subresource.baseArrayLayer = state->outBaseLayer;
			range.offset = state->outOffset;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource,range,state->outImage))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
				return false;

			return true;
		}

	protected:
		virtual ~CMatchedSizeInOutImageFilterCommon() = 0;
};

} // end namespace asset
} // end namespace irr

#endif
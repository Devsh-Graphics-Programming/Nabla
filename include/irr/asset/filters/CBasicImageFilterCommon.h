// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __IRR_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/IImageFilter.h"

namespace irr
{
namespace asset
{

class CBasicImageFilterCommon
{
	public:
		template<typename F>
		static inline void executePerBlock(const ICPUImage* image, const irr::asset::IImage::SBufferCopy& region, F& f)
		{
			const auto& params = image->getCreationParameters();
			const auto& extent = params.extent;

			VkExtent3D trueExtent;
			trueExtent.width = region.bufferRowLength ? region.bufferRowLength:extent.width;
			trueExtent.height = region.bufferImageHeight ? region.bufferImageHeight:extent.height;
			trueExtent.depth = extent.depth;

			const auto blockBytesize = asset::getTexelOrBlockBytesize(params.format);
			for (uint32_t zPos=region.imageOffset.z; zPos<region.imageExtent.depth; ++zPos)
			for (uint32_t yPos=region.imageOffset.y; yPos<region.imageExtent.height; ++yPos)
			for (uint32_t xPos=region.imageOffset.x; xPos<region.imageExtent.width; ++xPos)
			{
				auto texelPtr = region.bufferOffset+((zPos*trueExtent.height+yPos)*trueExtent.width+xPos)*blockBytesize;
				f(texelPtr,xPos,yPos,zPos);
			}
		}

	protected:
		virtual ~CBasicImageFilterCommon() = 0;

		static inline bool validateSubresourceAndRange(	const ICPUImage::SSubresourceLayers& subresource,
														const IImageFilter::IState::TexelRange& range,
														const ICPUImage* image)
		{
			if (!image)
				return false;
			const auto& params = image->getCreationParameters();

			if (range.offset.x+range.extent.width>params.extent.width)
				return false;
			if (range.offset.y+range.extent.height>params.extent.height)
				return false;
			if (range.offset.z+range.extent.depth>params.extent.depth)
				return false;
			
			if (subresource.baseArrayLayer+subresource.layerCount>params.arrayLayers)
				return false;
			if (subresource.mipLevel>=params.mipLevels)
				return false;

			return true;
		}
};

class CBasicInImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				ICPUImage::SSubresourceLayers	subresource = {};
				TexelRange						inRange = {};
				const ICPUImage*				inImage = nullptr;
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->subresource,state->inRange,state->inImage))
				return false;

			return true;
		}
};
class CBasicOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				ICPUImage::SSubresourceLayers	subresource = {};
				TexelRange						outRange = {};
				ICPUImage*						outImage = nullptr;
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->subresource,state->outRange,state->outImage))
				return false;

			return true;
		}
};
class CBasicInOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				ICPUImage::SSubresourceLayers	subresource = {};
				TexelRange						inRange = {};
				ICPUImage*						inImage = nullptr;
				TexelRange						outRange = {};
				ICPUImage*						outImage = nullptr;
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->subresource,state->inRange,state->inImage))
				return false;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->subresource,state->outRange,state->outImage))
				return false;

			return true;
		}
};
// will probably need some per-pixel helper class/functions (that can run a templated functor per-pixel to reduce code clutter)

} // end namespace asset
} // end namespace irr

#endif
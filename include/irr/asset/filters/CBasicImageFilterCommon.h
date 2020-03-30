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
		struct TexelBlockInfo
		{
			TexelBlockInfo(E_FORMAT format) :
				dimension(getTexelOrBlockBytesize(format)),
				maxCoord(dimension-core::vector3du32_SIMD(1u,1u,1u))
			{}

			core::vector3du32_SIMD dimension;
			core::vector3du32_SIMD maxCoord;
		};
		static inline auto texelsToBlocks(const core::vector3du32_SIMD& coord, const TexelBlockInfo& info)
		{
			return (coord+info.maxCoord)/info.dimension;
		}

		template<typename F>
		static inline void executePerBlock(const ICPUImage* image, const IImage::SBufferCopy& region, F& f)
		{
			const auto& params = image->getCreationParameters();
			const auto& extent = params.extent;

			TexelBlockInfo blockInfo(params.format);

			core::vector3du32_SIMD trueOffset;
			trueOffset.x = region.imageOffset.x;
			trueOffset.y = region.imageOffset.y;
			trueOffset.z = region.imageOffset.z;
			trueOffset = texelsToBlocks(trueOffset,blockInfo);
			
			core::vector3du32_SIMD trueExtent;
			trueExtent.x = region.bufferRowLength ? region.bufferRowLength:region.imageExtent.width;
			trueExtent.y = region.bufferImageHeight ? region.bufferImageHeight:region.imageExtent.height;
			trueExtent.z = region.imageExtent.depth;
			trueExtent = texelsToBlocks(trueExtent,blockInfo);

			const auto blockBytesize = asset::getTexelOrBlockBytesize(params.format);
			for (uint32_t layer=0u; layer<region.imageSubresource.layerCount; layer++)
			for (uint32_t zBlock=trueOffset.z; zBlock<trueExtent.z; ++zBlock)
			for (uint32_t yBlock=trueOffset.y; yBlock<trueExtent.y; ++yBlock)
			for (uint32_t xBlock=trueOffset.x; xBlock<trueExtent.x; ++xBlock)
			{
				auto texelPtr = region.bufferOffset+((((region.imageSubresource.baseArrayLayer+layer)*trueExtent[2]+zBlock)*trueExtent[1]+yBlock)*trueExtent[0]+xBlock)*blockBytesize;
				f(texelPtr,xBlock,yBlock,zBlock,layer);
			}
		}

		struct default_region_functor_t
		{
			inline void operator()(IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion) {}
		};
		static default_region_functor_t default_region_functor;

		template<typename F, typename G=default_region_functor_t>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											const IImage::SBufferCopy* _begin=image->getRegions().begin(),
											const IImage::SBufferCopy* _end=image->getRegions().end(),
											G& g=default_region_functor)
		{
			for (auto it=_begin; it!=_end; it++)
			{
				IImage::SBufferCopy region = *it;
				g(region,it);
				executePerBlock<F>(image, region, f);
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
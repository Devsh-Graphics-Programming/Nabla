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
				dimension(getBlockDimensions(format)),
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
			
			core::vector3du32_SIMD trueExtent = texelsToBlocks(region.getTexelStrides(),blockInfo);

			trueExtent.w = asset::getTexelOrBlockBytesize(params.format);
			for (uint32_t layer=0u; layer<region.imageSubresource.layerCount; layer++)
			for (uint32_t zBlock=trueOffset.z; zBlock<trueExtent.z; ++zBlock)
			for (uint32_t yBlock=trueOffset.y; yBlock<trueExtent.y; ++yBlock)
			for (uint32_t xBlock=trueOffset.x; xBlock<trueExtent.x; ++xBlock)
			{
				auto texelPtr = region.bufferOffset+((((region.imageSubresource.baseArrayLayer+layer)*trueExtent[2]+zBlock)*trueExtent[1]+yBlock)*trueExtent[0]+xBlock)*trueExtent[3];
				f(texelPtr,xBlock,yBlock,zBlock,layer);
			}
		}

		struct default_region_functor_t
		{
			inline bool operator()(IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion) { return true; }
		};
		static default_region_functor_t default_region_functor;
		
		struct clip_region_functor_t
		{
			clip_region_functor_t(const ICPUImage::SSubresourceLayers& _subresrouce, const IImageFilter::IState::TexelRange& _range, E_FORMAT format) : 
				subresource(_subresrouce), range(_range), blockByteSize(getTexelOrBlockBytesize(format)) {}

			const ICPUImage::SSubresourceLayers&	subresource;
			const IImageFilter::IState::TexelRange&	range;
			const uint32_t							blockByteSize;

			inline bool operator()(IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion)
			{
				if (subresource.mipLevel!=referenceRegion->imageSubresource.mipLevel)
					return false;
				newRegion.imageSubresource.baseArrayLayer = core::max(subresource.baseArrayLayer,referenceRegion->imageSubresource.baseArrayLayer);
				newRegion.imageSubresource.layerCount = core::min(	subresource.baseArrayLayer+subresource.layerCount,
																	referenceRegion->imageSubresource.baseArrayLayer+referenceRegion->imageSubresource.layerCount);
				if (newRegion.imageSubresource.layerCount <= newRegion.imageSubresource.baseArrayLayer)
					return false;
				newRegion.imageSubresource.layerCount -= newRegion.imageSubresource.baseArrayLayer;

				// handle the clipping
				uint32_t offsetInOffset[3u] = {0u,0u,0u};
				for (uint32_t i=0u; i<3u; i++)
				{
					const auto& ref = (&referenceRegion->imageOffset.x)[i];
					const auto& _new = (&range.offset.x)[i];
					bool clip = _new>ref;
					(&newRegion.imageOffset.x)[i] = clip ? _new:ref;
					if (clip)
						offsetInOffset[i] = _new-ref;
					(&newRegion.imageExtent.width)[i] = core::min((&referenceRegion->imageExtent.width)[i],(&range.extent.width)[i]);
				}

				// compute new offset
				newRegion.bufferOffset += ((offsetInOffset[2]*referenceRegion->bufferImageHeight+offsetInOffset[1])*referenceRegion->bufferRowLength+offsetInOffset[0])*blockByteSize;

				return true;
			}
		};

		template<typename F, typename G=default_region_functor_t>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											const IImage::SBufferCopy* _begin=image->getRegions().begin(),
											const IImage::SBufferCopy* _end=image->getRegions().end(),
											G& g=default_region_functor)
		{
			for (auto it=_begin; it!=_end; it++)
			{
				IImage::SBufferCopy region = *it;
				if (g(region,it))
					executePerBlock<F>(image, region, f);
			}
		}

	protected:
		virtual ~CBasicImageFilterCommon() =0;

		static inline bool validateSubresourceAndRange(	const ICPUImage::SSubresourceLayers& subresource,
														const IImageFilter::IState::TexelRange& range,
														const ICPUImage* image)
		{
			if (!image)
				return false;
			const auto& params = image->getCreationParameters();

			if (!(range.extent.width&&range.extent.height&&range.extent.depth))
				return false;

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

	protected:
		virtual ~CBasicInImageFilterCommon() = 0;
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

	protected:
		virtual ~CBasicOutImageFilterCommon() = 0;
};
class CBasicInOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				ICPUImage::SSubresourceLayers	inSubresource = {};
				TexelRange						inRange = {};
				ICPUImage*						inImage = nullptr;
				ICPUImage::SSubresourceLayers	outSubresource = {};
				TexelRange						outRange = {};
				ICPUImage*						outImage = nullptr;
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->inSubresource,state->inRange,state->inImage))
				return false;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->outSubresource,state->outRange,state->outImage))
				return false;

			return true;
		}

	protected:
		virtual ~CBasicInOutImageFilterCommon() = 0;
};
// will probably need some per-pixel helper class/functions (that can run a templated functor per-pixel to reduce code clutter)

} // end namespace asset
} // end namespace irr

#endif
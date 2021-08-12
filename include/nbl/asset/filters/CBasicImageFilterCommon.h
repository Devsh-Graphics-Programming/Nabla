// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __NBL_ASSET_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/filters/IImageFilter.h"

namespace nbl
{
namespace asset
{

class CBasicImageFilterCommon
{
	public:
		template<typename F>
		static inline void executePerBlock(const ICPUImage* image, const IImage::SBufferCopy& region, F& f)
		{
			const auto& subresource = region.imageSubresource;

			const auto& params = image->getCreationParameters();
			TexelBlockInfo blockInfo(params.format);

			core::vector3du32_SIMD trueOffset;
			trueOffset.x = region.imageOffset.x;
			trueOffset.y = region.imageOffset.y;
			trueOffset.z = region.imageOffset.z;
			trueOffset = blockInfo.convertTexelsToBlocks(trueOffset);
			trueOffset.w = subresource.baseArrayLayer;
			
			core::vector3du32_SIMD trueExtent;
			trueExtent.x = region.imageExtent.width;
			trueExtent.y = region.imageExtent.height;
			trueExtent.z = region.imageExtent.depth;
			trueExtent  = blockInfo.convertTexelsToBlocks(trueExtent);
			trueExtent.w = subresource.layerCount;

			const auto strides = region.getByteStrides(blockInfo);

			core::vector3du32_SIMD localCoord;
			for (auto& layer =localCoord[3]=0u; layer<trueExtent.w; ++layer)
			for (auto& zBlock=localCoord[2]=0u; zBlock<trueExtent.z; ++zBlock)
			for (auto& yBlock=localCoord[1]=0u; yBlock<trueExtent.y; ++yBlock)
			for (auto& xBlock=localCoord[0]=0u; xBlock<trueExtent.x; ++xBlock)
				f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
		}

		struct default_region_functor_t
		{
			constexpr default_region_functor_t() = default;
			inline constexpr bool operator()(IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion) const { return true; }
		};
		
		struct clip_region_functor_t
		{
			clip_region_functor_t(const ICPUImage::SSubresourceLayers& _subresrouce, const IImageFilter::IState::TexelRange& _range, E_FORMAT format) : 
				subresource(_subresrouce), range(_range), blockInfo(format) {}
			clip_region_functor_t(const ICPUImage::SSubresourceLayers& _subresrouce, const IImageFilter::IState::TexelRange& _range, const TexelBlockInfo& _blockInfo, uint32_t _blockByteSize) :
				subresource(_subresrouce), range(_range), blockInfo(_blockInfo)  {}

			const ICPUImage::SSubresourceLayers&	subresource;
			const IImageFilter::IState::TexelRange&	range;
			const TexelBlockInfo					blockInfo;

			inline bool operator()(IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion) const
			{
				if (subresource.mipLevel!=referenceRegion->imageSubresource.mipLevel)
					return false;

				core::vector3du32_SIMD targetOffset(range.offset.x,range.offset.y,range.offset.z,subresource.baseArrayLayer);
				core::vector3du32_SIMD targetExtent(range.extent.width,range.extent.height,range.extent.depth,subresource.layerCount);
				auto targetLimit = targetOffset+targetExtent;

				const core::vector3du32_SIMD resultOffset(referenceRegion->imageOffset.x,referenceRegion->imageOffset.y,referenceRegion->imageOffset.z,referenceRegion->imageSubresource.baseArrayLayer);
				const core::vector3du32_SIMD resultExtent(referenceRegion->imageExtent.width,referenceRegion->imageExtent.height,referenceRegion->imageExtent.depth,referenceRegion->imageSubresource.layerCount);
				const auto resultLimit = resultOffset+resultExtent;

				auto offset = core::max<core::vector3du32_SIMD>(targetOffset,resultOffset);
				auto limit = core::min<core::vector3du32_SIMD>(targetLimit,resultLimit);
				if ((offset>=limit).any())
					return false;

				// compute new offset
				{
					const auto strides = referenceRegion->getByteStrides(blockInfo);
					const core::vector3du32_SIMD offsetInOffset = offset-resultOffset;
					newRegion.bufferOffset += referenceRegion->getLocalByteOffset(offsetInOffset,strides);
				}

				if (!referenceRegion->bufferRowLength)
					newRegion.bufferRowLength = referenceRegion->imageExtent.width;
				if (!referenceRegion->bufferImageHeight)
					newRegion.bufferImageHeight = referenceRegion->imageExtent.height;

				newRegion.imageOffset.x = offset.x;
				newRegion.imageOffset.y = offset.y;
				newRegion.imageOffset.z = offset.z;
				newRegion.imageSubresource.baseArrayLayer = offset.w;
				auto extent = limit - offset;
				newRegion.imageExtent.width = extent.x;
				newRegion.imageExtent.height = extent.y;
				newRegion.imageExtent.depth = extent.z;
				newRegion.imageSubresource.layerCount = extent.w;
				return true;
			}
		};
		
		template<typename F, typename G>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											const IImage::SBufferCopy* _begin,
											const IImage::SBufferCopy* _end,
											G& g)
		{
			for (auto it=_begin; it!=_end; it++)
			{
				IImage::SBufferCopy region = *it;
				if (g(region,it))
					executePerBlock<F>(image, region, f);
			}
		}
		template<typename F>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											const IImage::SBufferCopy* _begin,
											const IImage::SBufferCopy* _end)
		{
			default_region_functor_t voidFunctor;
			return executePerRegion<F,default_region_functor_t>(image,f,_begin,_end,voidFunctor);
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
				return false;

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
				return false;

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
				return false;

			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->inSubresource,state->inRange,state->inImage))
				return false;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(state->outSubresource,state->outRange,state->outImage))
				return false;

			return true;
		}

	protected:
		virtual ~CBasicInOutImageFilterCommon() = 0;
};

} // end namespace asset
} // end namespace nbl

#endif

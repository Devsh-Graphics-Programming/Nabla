// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __NBL_ASSET_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "nbl/core/core.h"

#include <algorithm>
#include <execution>

#include "nbl/asset/filters/IImageFilter.h"

namespace nbl
{
namespace asset
{

class CBasicImageFilterCommon
{
	public:
		template<uint32_t batch_dims>
		struct BlockIterator
		{
			public:
				using iterator_category = std::random_access_iterator_tag;
				using difference_type = int64_t;
				using value_type = core::vectorSIMDu32;
				using pointer = value_type;
				using reference = value_type;

				static inline constexpr uint32_t iterator_dims = 4u-batch_dims;
				static inline constexpr uint32_t last_iterator_dim = iterator_dims - 1u;

				BlockIterator()
				{
					std::fill_n(remainingExtents,iterator_dims,0u);
					std::fill_n(localCoord,iterator_dims,0u);
				}
				explicit inline BlockIterator(const uint32_t* _remainingExtents)
				{
					std::copy_n(_remainingExtents,iterator_dims,remainingExtents);
					std::fill_n(localCoord,iterator_dims,0u);
				}
				explicit inline BlockIterator(const uint32_t* _remainingExtents, const uint32_t* _localCoord)
				{
					std::copy_n(_remainingExtents,iterator_dims,remainingExtents);
					std::copy_n(_localCoord,iterator_dims,localCoord);
				}
				BlockIterator(const BlockIterator<batch_dims>& other) = default;
				BlockIterator(BlockIterator<batch_dims>&& other) = default;

				BlockIterator<batch_dims>& operator=(const BlockIterator<batch_dims>& other) = default;
				BlockIterator<batch_dims>& operator=(BlockIterator<batch_dims>&& other) = default;

				inline auto operator*() const
				{
					core::vectorSIMDu32 retval;
					std::copy_n(localCoord,iterator_dims,retval.pointer+batch_dims);
					return retval;
				}
				inline auto operator->() const { return operator*(); } // total abuse
				/*
				inline const BlockIterator<batch_dims>& operator--()
				{
					for (uint32_t i=0u; i<last_iterator_dim; i++)
					{
						if (localCoord[i]--!=0u)
							return *this;
						localCoord[i] = 0u;
					}
					--localCoord[last_iterator_dim];
					return *this;
				}
				inline BlockIterator<batch_dims> operator--(int)
				{
					BlockIterator<batch_dims> copy(*this);
					this->operator--();
					return copy;
				}
				*/
				inline BlockIterator<batch_dims>& operator++()
				{
					for (uint32_t i=0u; i<last_iterator_dim; i++)
					{
						if (++localCoord[i]!=remainingExtents[i])
							return *this;
						localCoord[i] = 0u;
					}
					++localCoord[last_iterator_dim];
					return *this;
				}
				inline BlockIterator<batch_dims> operator++(int)
				{
					BlockIterator<batch_dims> copy(*this);
					this->operator++();
					return copy;
				}

				inline bool operator==(const BlockIterator<batch_dims>& other) const {return std::equal(localCoord,localCoord+iterator_dims,other.localCoord);}
				inline bool operator!=(const BlockIterator<batch_dims>& other) const {return !operator==(other);}

				inline BlockIterator<batch_dims>& operator+=(const difference_type advance)
				{
					return operator=(BlockIterator<batch_dims>(remainingExtents,toLinearAddress()+advance));
				}
				inline BlockIterator<batch_dims> operator+(const difference_type advance) const
				{
					BlockIterator<batch_dims> copy(*this);
					copy += advance;
					return copy;
				}
				/*
				inline BlockIterator<batch_dims> operator+(const BlockIterator<batch_dims>& other) const
				{
					return BlockIterator<batch_dims>();
				}*/
				inline difference_type operator-(const BlockIterator<batch_dims>& other) const
				{
					return toLinearAddress()-other.toLinearAddress();
				}
				
				inline const uint32_t* getRemainingExtents() const {return remainingExtents;}
			private:
				uint32_t remainingExtents[iterator_dims];
				uint32_t localCoord[iterator_dims];
				
				
				explicit inline BlockIterator(const uint32_t* _remainingExtents, difference_type linearAddress)
				{
					std::copy_n(_remainingExtents,iterator_dims,remainingExtents);
					for (uint32_t i=0u; i<last_iterator_dim; i++)
					{
						difference_type d = linearAddress/remainingExtents[i];
						localCoord[i] = linearAddress-d*remainingExtents[i];
						linearAddress = d;
					}
					localCoord[last_iterator_dim] = linearAddress;
				}
				inline difference_type toLinearAddress() const
				{
					difference_type retval = localCoord[last_iterator_dim];
					for (auto i=last_iterator_dim; i!=0u; )
					{
						i--;
						retval = retval*difference_type(remainingExtents[i])+difference_type(localCoord[i]);
					}
					return retval;
				}
		};

		template<class ExecutionPolicy, typename F>
		static inline void executePerBlock(ExecutionPolicy&& policy, const ICPUImage* image, const IImage::SBufferCopy& region, F& f)
		{
			const auto& subresource = region.imageSubresource;

			const auto& params = image->getCreationParameters();
			TexelBlockInfo blockInfo(params.format);

			core::vectorSIMDu32 trueOffset;
			trueOffset.x = region.imageOffset.x;
			trueOffset.y = region.imageOffset.y;
			trueOffset.z = region.imageOffset.z;
			trueOffset = blockInfo.convertTexelsToBlocks(trueOffset);
			trueOffset.w = subresource.baseArrayLayer;
			
			core::vectorSIMDu32 trueExtent;
			trueExtent.x = region.imageExtent.width;
			trueExtent.y = region.imageExtent.height;
			trueExtent.z = region.imageExtent.depth;
			trueExtent  = blockInfo.convertTexelsToBlocks(trueExtent);
			trueExtent.w = subresource.layerCount;

			const auto strides = region.getByteStrides(blockInfo);
			
			auto batch1D = [&f,&region,trueExtent,strides,trueOffset](core::vectorSIMDu32 localCoord)
			{
				for (auto& xBlock=localCoord[0]=0u; xBlock<trueExtent.x; ++xBlock)
					f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
			};
			auto batch2D = [&f,&region,trueExtent,strides,trueOffset](core::vectorSIMDu32 localCoord)
			{
				for (auto& yBlock=localCoord[1]=0u; yBlock<trueExtent.y; ++yBlock)
				for (auto& xBlock=localCoord[0]=0u; xBlock<trueExtent.x; ++xBlock)
					f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
			};
			auto batch3D = [&f,&region,trueExtent,strides,trueOffset](core::vectorSIMDu32 localCoord)
			{
				for (auto& zBlock=localCoord[2]=0u; zBlock<trueExtent.z; ++zBlock)
				for (auto& yBlock=localCoord[1]=0u; yBlock<trueExtent.y; ++yBlock)
				for (auto& xBlock=localCoord[0]=0u; xBlock<trueExtent.x; ++xBlock)
					f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
			};

			constexpr uint32_t batchSizeThreshold = 0x80u;
			const core::vectorSIMDu32 spaceFillingEnd(0u,0u,0u,trueExtent.w);
			if (std::is_same_v<ExecutionPolicy,std::execution::sequenced_policy> || trueExtent.x*trueExtent.y<batchSizeThreshold)
			{
				constexpr uint32_t batch_dims = 3u;
				BlockIterator<batch_dims> begin(trueExtent.pointer+batch_dims);
				BlockIterator<batch_dims> end(trueExtent.pointer+batch_dims,spaceFillingEnd.pointer+batch_dims);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,batch3D);
			}
			else if (trueExtent.x<batchSizeThreshold)
			{
				constexpr uint32_t batch_dims = 2u;
				BlockIterator<batch_dims> begin(trueExtent.pointer+batch_dims);
				BlockIterator<batch_dims> end(trueExtent.pointer+batch_dims,spaceFillingEnd.pointer+batch_dims);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,batch2D);
			}
			else
			{
				constexpr uint32_t batch_dims = 1u;
				BlockIterator<batch_dims> begin(trueExtent.pointer+batch_dims);
				BlockIterator<batch_dims> end(trueExtent.pointer+batch_dims,spaceFillingEnd.pointer+batch_dims);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,batch1D);
			}
		}
		template<typename F>
		static inline void executePerBlock(const ICPUImage* image, const IImage::SBufferCopy& region, F& f)
		{
			executePerBlock(std::execution::seq,image,region,f);
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

				core::vectorSIMDu32 targetOffset(range.offset.x,range.offset.y,range.offset.z,subresource.baseArrayLayer);
				core::vectorSIMDu32 targetExtent(range.extent.width,range.extent.height,range.extent.depth,subresource.layerCount);
				auto targetLimit = targetOffset+targetExtent;

				const core::vectorSIMDu32 resultOffset(referenceRegion->imageOffset.x,referenceRegion->imageOffset.y,referenceRegion->imageOffset.z,referenceRegion->imageSubresource.baseArrayLayer);
				const core::vectorSIMDu32 resultExtent(referenceRegion->imageExtent.width,referenceRegion->imageExtent.height,referenceRegion->imageExtent.depth,referenceRegion->imageSubresource.layerCount);
				const auto resultLimit = resultOffset+resultExtent;

				auto offset = core::max<core::vectorSIMDu32>(targetOffset,resultOffset);
				auto limit = core::min<core::vectorSIMDu32>(targetLimit,resultLimit);
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
		
		template<class ExecutionPolicy, typename F, typename G>
		static inline void executePerRegion(ExecutionPolicy&& policy,
											const ICPUImage* image, F& f,
											const IImage::SBufferCopy* _begin,
											const IImage::SBufferCopy* _end,
											G& g)
		{
			for (auto it=_begin; it!=_end; it++)
			{
				IImage::SBufferCopy region = *it;
				if (g(region,it))
					executePerBlock<ExecutionPolicy,F>(std::forward<ExecutionPolicy>(policy),image,region,f);
			}
		}
		template<typename F, typename G>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											const IImage::SBufferCopy* _begin,
											const IImage::SBufferCopy* _end,
											G& g)
		{
			return executePerRegion<const std::execution::sequenced_policy&,F,G>(std::execution::seq,image,f,_begin,_end,g);
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

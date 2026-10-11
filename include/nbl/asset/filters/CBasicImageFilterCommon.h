// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __NBL_ASSET_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/core/execution.h"

#include <algorithm>

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
				using value_type = std::array<uint32_t,batch_dims>;
				using pointer = const value_type*;
				using reference = value_type;

				static inline constexpr uint32_t last_dim = batch_dims-1u;

				BlockIterator()
				{
					std::fill_n(extentBatches,batch_dims,0u);
					std::fill_n(batchCoord.begin(),batch_dims,0u);
				}
				explicit inline BlockIterator(const uint32_t* _extentBatches)
				{
					std::copy_n(_extentBatches,batch_dims,extentBatches);
					std::fill_n(batchCoord.begin(),batch_dims,0u);
				}
				explicit inline BlockIterator(const uint32_t* _extentBatches, const uint32_t* _batchCoord)
				{
					std::copy_n(_extentBatches,batch_dims,extentBatches);
					std::copy_n(_batchCoord,batch_dims,batchCoord.begin());
				}
				BlockIterator(const BlockIterator<batch_dims>& other) = default;
				BlockIterator(BlockIterator<batch_dims>&& other) = default;

				BlockIterator<batch_dims>& operator=(const BlockIterator<batch_dims>& other) = default;
				BlockIterator<batch_dims>& operator=(BlockIterator<batch_dims>&& other) = default;

				inline reference operator*() const
				{
					return batchCoord;
				}
				inline pointer operator->() const {return &batchCoord;}

				inline value_type operator[] (int index) const
				{
					return *(operator+(index));
				}
				
				inline bool operator< (const BlockIterator<batch_dims>& _rhs) const 
				{ 
					return toLinearAddress() < _rhs.toLinearAddress();
				}

				/*
				inline const BlockIterator<batch_dims>& operator--()
				{
					for (uint32_t i=0u; i<last_dim; i++)
					{
						if (batchCoord[i]--!=0u)
							return *this;
						batchCoord[i] = 0u;
					}
					--batchCoord[last_dim];
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
					for (uint32_t i=0u; i<last_dim; i++)
					{
						if (++batchCoord[i]!=extentBatches[i])
							return *this;
						batchCoord[i] = 0u;
					}
					++batchCoord[last_dim];
					return *this;
				}
				inline BlockIterator<batch_dims> operator++(int)
				{
					BlockIterator<batch_dims> copy(*this);
					this->operator++();
					return copy;
				}

				inline bool operator==(const BlockIterator<batch_dims>& other) const {return std::equal(batchCoord.begin(),batchCoord.end(),other.batchCoord.begin());}
				inline bool operator!=(const BlockIterator<batch_dims>& other) const {return !operator==(other);}

				inline BlockIterator<batch_dims>& operator+=(const difference_type advance)
				{
					return operator=(BlockIterator<batch_dims>(extentBatches,toLinearAddress()+advance));
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
				
				inline const uint32_t* getExtentBatches() const {return extentBatches;}
			private:
				uint32_t extentBatches[batch_dims];
				value_type batchCoord;
				
				
				explicit inline BlockIterator(const uint32_t* _extentBatches, difference_type linearAddress)
				{
					std::copy_n(_extentBatches,batch_dims,extentBatches);
					for (uint32_t i=0u; i<last_dim; i++)
					{
						difference_type d = linearAddress/extentBatches[i];
						batchCoord[i] = linearAddress-d*extentBatches[i];
						linearAddress = d;
					}
					batchCoord[last_dim] = linearAddress;
				}
				inline difference_type toLinearAddress() const
				{
					difference_type retval = batchCoord[last_dim];
					for (auto i=last_dim; i!=0u; )
					{
						i--;
						retval = retval*difference_type(extentBatches[i])+difference_type(batchCoord[i]);
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
			
			auto batch1D = [&f,&region,trueExtent,strides,trueOffset](const std::array<uint32_t,3u>& batchCoord)
			{
				for (auto xBlock=0u; xBlock<trueExtent.x; ++xBlock)
				{
					const core::vectorSIMDu32 localCoord(xBlock,batchCoord[0],batchCoord[1],batchCoord[2]);
					f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
				}
			};
			auto batch2D = [&f,&region,trueExtent,strides,trueOffset](const std::array<uint32_t,2u>& batchCoord)
			{
				for (auto yBlock=0u; yBlock<trueExtent.y; ++yBlock)
				for (auto xBlock=0u; xBlock<trueExtent.x; ++xBlock)
				{
					const core::vectorSIMDu32 localCoord(xBlock,yBlock,batchCoord[0],batchCoord[1]);
					f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
				}
			};
			auto batch3D = [&f,&region,trueExtent,strides,trueOffset](const std::array<uint32_t,1u>& batchCoord)
			{
				for (auto zBlock=0u; zBlock<trueExtent.z; ++zBlock)
				for (auto yBlock=0u; yBlock<trueExtent.y; ++yBlock)
				for (auto xBlock=0u; xBlock<trueExtent.x; ++xBlock)
				{
					const core::vectorSIMDu32 localCoord(xBlock,yBlock,zBlock,batchCoord[0]);
					f(region.getByteOffset(localCoord,strides),localCoord+trueOffset);
				}
			};

			constexpr uint32_t batchSizeThreshold = 0x80u;
			const core::vectorSIMDu32 spaceFillingEnd(0u,0u,0u,trueExtent.w);
			if (std::is_same_v<ExecutionPolicy,core::execution::sequenced_policy> || trueExtent.x*trueExtent.y<batchSizeThreshold)
			{
				constexpr uint32_t batch_dims = 1u;
				BlockIterator<batch_dims> begin(trueExtent.pointer+4u-batch_dims);
				BlockIterator<batch_dims> end(begin.getExtentBatches(),spaceFillingEnd.pointer+4u-batch_dims);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,batch3D);
			}
			else if (trueExtent.x<batchSizeThreshold)
			{
				constexpr uint32_t batch_dims = 2u;
				BlockIterator<batch_dims> begin(trueExtent.pointer+4u-batch_dims);
				BlockIterator<batch_dims> end(begin.getExtentBatches(),spaceFillingEnd.pointer+4u-batch_dims);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,batch2D);
			}
			else
			{
				constexpr uint32_t batch_dims = 3u;
				BlockIterator<batch_dims> begin(trueExtent.pointer+4u-batch_dims);
				BlockIterator<batch_dims> end(begin.getExtentBatches(),spaceFillingEnd.pointer+4u-batch_dims);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,batch1D);
			}
		}
		template<typename F>
		static inline void executePerBlock(const ICPUImage* image, const IImage::SBufferCopy& region, F& f)
		{
			executePerBlock(core::execution::seq,image,region,f);
		}

		//! Calls `f(inByteOffset,outByteOffset,blockCount)` for every run of blocks contiguous in both the input and the output of a box of `extentInBlocks` (`w` is the layer count).
		//! The strides are `IImage::SBufferCopy::getBlockStrides`, rows, slices and layers merge into one run wherever neither side has padding between them, parallel policies split runs into 256 KiB pieces.
		template<class ExecutionPolicy, typename F>
		static inline void executePerRun(ExecutionPolicy&& policy, const core::vectorSIMDu32& extentInBlocks, const core::vectorSIMDu32& inBlockStrides, const size_t inBlockByteSize, const core::vectorSIMDu32& outBlockStrides, const size_t outBlockByteSize, F&& f)
		{
			if (!extentInBlocks.x || !extentInBlocks.y || !extentInBlocks.z || !extentInBlocks.w)
				return;

			const size_t inRowPitch = inBlockStrides.x*inBlockByteSize;
			const size_t outRowPitch = outBlockStrides.x*outBlockByteSize;
			const size_t inPitches[3] = {inRowPitch,inRowPitch*inBlockStrides.y,inRowPitch*inBlockStrides.y*inBlockStrides.z};
			const size_t outPitches[3] = {outRowPitch,outRowPitch*outBlockStrides.y,outRowPitch*outBlockStrides.y*outBlockStrides.z};
			uint32_t counts[3] = {extentInBlocks.y,extentInBlocks.z,extentInBlocks.w};
			size_t runBlocks = extentInBlocks.x;
			for (uint32_t i=0u; i<3u; i++)
			{
				if (counts[i]!=1u && (inPitches[i]!=runBlocks*inBlockByteSize || outPitches[i]!=runBlocks*outBlockByteSize))
					break;
				runBlocks *= counts[i];
				counts[i] = 1u;
			}

			constexpr size_t PieceByteSize = 0x40000u;
			const size_t blockByteSize = core::max<size_t>(core::max(inBlockByteSize,outBlockByteSize),1u);
			constexpr bool Sequential = std::is_same_v<std::decay_t<ExecutionPolicy>,core::execution::sequenced_policy>;
			if (Sequential || runBlocks*blockByteSize*counts[0]*counts[1]*counts[2]<=PieceByteSize)
			{
				for (uint32_t layer=0u; layer<counts[2]; layer++)
				for (uint32_t slice=0u; slice<counts[1]; slice++)
				for (uint32_t row=0u; row<counts[0]; row++)
					f(row*inPitches[0]+slice*inPitches[1]+layer*inPitches[2],row*outPitches[0]+slice*outPitches[1]+layer*outPitches[2],runBlocks);
			}
			else
			{
				const size_t pieceBlocks = core::max<size_t>(PieceByteSize/blockByteSize,1u);
				const uint32_t batches[4] = {uint32_t((runBlocks+pieceBlocks-1u)/pieceBlocks),counts[0],counts[1],counts[2]};
				const uint32_t endCoord[4] = {0u,0u,0u,counts[2]};
				BlockIterator<4u> begin(batches);
				BlockIterator<4u> end(begin.getExtentBatches(),endCoord);
				std::for_each(std::forward<ExecutionPolicy>(policy),begin,end,[&](const std::array<uint32_t,4u>& coord) -> void
				{
					const size_t firstBlock = coord[0]*pieceBlocks;
					f(coord[1]*inPitches[0]+coord[2]*inPitches[1]+coord[3]*inPitches[2]+firstBlock*inBlockByteSize,
						coord[1]*outPitches[0]+coord[2]*outPitches[1]+coord[3]*outPitches[2]+firstBlock*outBlockByteSize,
						core::min(pieceBlocks,runBlocks-firstBlock));
				});
			}
		}

		//! Copies a box of blocks between two layouts with one memcpy per run, @see `executePerRun`
		template<class ExecutionPolicy>
		static inline void copyBlocks(ExecutionPolicy&& policy, uint8_t* const dst, const core::vectorSIMDu32& dstBlockStrides, const uint8_t* const src, const core::vectorSIMDu32& srcBlockStrides, const size_t blockByteSize, const core::vectorSIMDu32& extentInBlocks)
		{
			executePerRun(std::forward<ExecutionPolicy>(policy),extentInBlocks,srcBlockStrides,blockByteSize,dstBlockStrides,blockByteSize,[&](const size_t srcByteOffset, const size_t dstByteOffset, const size_t blockCount) -> void
			{
				memcpy(dst+dstByteOffset,src+srcByteOffset,blockCount*blockByteSize);
			});
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
					const core::vector3du32_SIMD offsetInOffset = blockInfo.convertTexelsToBlocks(offset-resultOffset);
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
											std::span<const IImage::SBufferCopy> regions,
											G& g)
		{
			for(auto region : regions)
			{
				if (g(region,&region))
					executePerBlock<ExecutionPolicy,F>(std::forward<ExecutionPolicy>(policy),image,region,f);
			}
		}
		template<typename F, typename G>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											std::span<const IImage::SBufferCopy> regions,
											G& g)
		{
			return executePerRegion<const core::execution::sequenced_policy&,F,G>(core::execution::seq,image,f,regions,g);
		}
		template<typename F>
		static inline void executePerRegion(const ICPUImage* image, F& f,
											std::span<const IImage::SBufferCopy> regions)
		{
			default_region_functor_t voidFunctor;
			return executePerRegion<F,default_region_functor_t>(image,f,regions,voidFunctor);
		}

	protected:
		virtual NBL_API2 ~CBasicImageFilterCommon() =0;

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

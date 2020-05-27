// Copyright (C) 2020 - AnastaZIuk
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>
#include <functional>

#include "irr/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "CConvertFormatImageFilter.h"

namespace irr
{
namespace asset
{

namespace impl
{
	struct plusTexels
	{
		core::vector4df_SIMD operator()(const core::vector4df_SIMD& lhs, const core::vector4df_SIMD& rhs) const	// it will fuck up 100% XD, TODO
		{
			return lhs + rhs;
		}
	};
}

template<bool ExclusiveMode>
class CSummedAreaTableImageFilterBase : public CBasicImageFilterCommon
{
	public:
		class CStateBase
		{
			public:
				enum E_SUM_MODE
				{
					ESM_INCLUSIVE,	//!< all the values are summed withing the pixel summing begin from, so (x,y,z) values <= than itself
					ESM_EXCLUSIVE,	//!< all the values are summed without the pixel summing begin from, so (x,y,z) values < than itself
					EAS_COUNT
				};

				uint8_t*	scratchMemory = nullptr;										//!< memory used for temporary filling within computation of sum values
				uint32_t	scratchMemoryByteSize = 0u;
				core::smart_refctd_ptr<asset::ICPUBuffer>	rowSumCache;					//!< pointer of row extent size for y values used for holding max values going down till 0 is achievied
				core::smart_refctd_ptr<asset::ICPUBuffer>	columnSumCache;					//!< pointer of column extent size for x values used for holding max values going down till 0 is achievied
				constexpr E_SUM_MODE mode = ExclusiveMode ? ESM_EXCLUSIVE : ESM_INCLUSIVE;
		};

	protected:
		CSummedAreaTableImageFilterBase() {}
		virtual ~CSummedAreaTableImageFilterBase() {}

		static inline uint32_t getRequiredScratchByteSize(E_FORMAT outFormat, const core::vectorSIMDu32& outExtent = core::vectorSIMDu32(0,0,0,0))
		{
			return asset::getTexelOrBlockBytesize(outFormat) * outExtent.X * outExtent.Y * outExtent.Z;
		}

		static inline bool validate(CStateBase* state)
		{
			if (!state)
				return false;

			return true;
		}
};

//! Fill texel buffer with computed sum of left and down texels placed in input image
/*
	When the summing is in exclusive mode - it computes the sum of all the pixels placed
	on the left and down for a new single texel but it doesn't take sum the main texel itself.
	In inclusive mode, the texel we start from is taken as well and added to the sum.
*/

template<bool ExclusiveMode = false>
class CSummedAreaTableImageFilter : public CImageFilter<CSummedAreaTableImageFilter>, public CSummedAreaTableImageFilterBase<ExclusiveMode>
{
	public:
		virtual ~CSummedAreaTableImageFilter() {}

		class CPrototypeState : public IImageFilter::IState
		{
			public:
				CPrototypeState()
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}
				CPrototypeState(const CPrototypeState& other) : inMipLevel(other.inMipLevel), outMipLevel(other.outMipLevel), inImage(other.inImage), outImage(other.outImage)
				{
					inOffsetBaseLayer = other.inOffsetBaseLayer;
					inExtentLayerCount = other.inExtentLayerCount;
					outOffsetBaseLayer = other.outOffsetBaseLayer;
					outExtentLayerCount = other.outExtentLayerCount;
				}
				virtual ~CPrototypeState() {}

				union
				{
					core::vectorSIMDu32	inOffsetBaseLayer;
					struct
					{
						VkOffset3D		inOffset;
						uint32_t		inBaseLayer;
					};
				};
				union
				{
					core::vectorSIMDu32 inExtentLayerCount;
					struct
					{
						VkExtent3D		inExtent;
						uint32_t		inLayerCount;
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
				union
				{
					core::vectorSIMDu32 outExtentLayerCount;
					struct
					{
						VkExtent3D		outExtent;
						uint32_t		outLayerCount;
					};
				};

				uint32_t				inMipLevel = 0u;
				uint32_t				outMipLevel = 0u;
				ICPUImage* inImage = nullptr;
				ICPUImage* outImage = nullptr;
		};

		class CState : public CPrototypeState, public CSummedAreaTableImageFilterBase<ExclusiveMode> {};
		using state_type = CState; //!< full combined state

		static inline bool validate(state_type* state)
		{
			if (!CSummedAreaTableImageFilterBase<ExclusiveMode>::validate(state))
				return false;
			
			const ICPUImage::SCreationParams& inParams = state->inImage->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = state->outImage->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;

			if (state->scratchMemoryByteSize < CSummedAreaTableImageFilterBase<ExclusiveMode>::getRequiredScratchByteSize(outFormat, {outParams.extent.width, outParams.extent.height, outParams.extent.depth}))
				return false;

			if (state->inLayerCount != state->outLayerCount)
				return false;

			IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->inLayerCount };
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource, { state->inOffset,state->inExtent }, state->inImage))
				return false;

			subresource.mipLevel = state->outMipLevel;
			subresource.baseArrayLayer = state->outBaseLayer;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource, { state->outOffset,state->outExtent }, state->outImage))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (asset::isBlockCompressionFormat(outFormat))
				return false;

			if (asset::getFormatChannelCount(outFormat) < asset::getFormatChannelCount(inFormat))
				return false;

			if (getFormatClass(state->inImage->getCreationParameters().format) <= getFormatClass(outFormat))
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			// load all the state
			const auto* const inImg = state->inImage;
			auto* const outImg = state->outImage;
			const ICPUImage::SCreationParams& inParams = inImg->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = outImg->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;
			const auto inBlockDims = asset::getBlockDimensions(inFormat);
			const auto outBlockDims = asset::getBlockDimensions(outFormat);
			const auto texelByteSize = asset::getTexelOrBlockBytesize(outFormat);
			const auto* const inData = reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer());
			auto* const outData = reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer());
			const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(state->outMipLevel);

			const auto inMipLevel = state->inMipLevel;
			const auto outMipLevel = state->outMipLevel;
			const auto inBaseLayer = state->inBaseLayer;
			const auto outBaseLayer = state->outBaseLayer;
			const auto layerCount = state->inLayerCount;
			assert(layerCount == state->outLayerCount); // validation bug?

			const auto inOffset = state->inOffset;
			const auto outOffset = state->outOffset;
			const auto inExtent = state->inExtent;
			const auto outExtent = state->outExtent;

			const auto inOffsetBaseLayer = state->inOffsetBaseLayer;
			const auto outOffsetBaseLayer = state->outOffsetBaseLayer;
			const auto inExtentLayerCount = state->inExtentLayerCount;
			const auto outExtentLayerCount = state->outExtentLayerCount;
			const auto inLimit = inOffsetBaseLayer + inExtentLayerCount;
			const auto outLimit = outOffsetBaseLayer + outExtentLayerCount;

			CMatchedSizeInOutImageFilterCommon::CommonExecuteData commonExecuteData =
			{
				inImg,
				outImg,
				inParams,
				outParams,
				inParams.format,
				outParams.format,
				getTexelOrBlockBytesize(inParams.format),
				getTexelOrBlockBytesize(outParams.format),
				reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer()),
				reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer()),
				inImg->getRegions(state->inMipLevel),
				outRegions,
				outRegions.begin(), {}, {}
			};

			auto allocateCache = [](core::smart_refctd_ptr<asset::ICPUBuffer> cache, size_t size)
			{
				auto newPointer = reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(size, _IRR_SIMD_ALIGNMENT));
				cache = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(size, newPointer, core::adopt_memory);
			};

			allocateCache(state->rowSumCache, outExtent.X * texelByteSize);
			allocateCache(state->columnSumCache, outExtent.Y * texelByteSize);

			auto perOutputRegion = [state]( const CMatchedSizeInOutImageFilterCommon::CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				const auto blockDims = asset::getBlockDimensions(inFormat);

				auto sum = [&commonExecuteData, &blockDims](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto localOutPos = readBlockPos * blockDims + commonExecuteData.offsetDifference; 
					auto outPtrTexelPosition = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos, commonExecuteData.outByteStrides);
					auto inPtrTexelPosition = commonExecuteData.inData + readBlockArrayOffset;

					constexpr auto MAX_PLANES = 4;
					const void* srcInPix[MAX_PLANES] = { inPtrTexelPosition, nullptr, nullptr, nullptr };
					uint8_t* dstPix = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos, commonExecuteData.outByteStrides);

					// with caching we can just substract one value (looking at left or down) to get appropriate sum without summing everything many times per texel 

					// fill cache with max values
					auto fillCacheIfEnteredMaxValues = [&]()
					{
						const auto width = commonExecuteData.oit->imageExtent.width;
						const auto height = commonExecuteData.oit->imageExtent.height;
						core::vector4df_SIMD channels; // tmp, TODO

						//convertColor<inFormat, outFormat>(srcPix, dstPix, blockX, blockY); nah		

						if (localOutPos.X == width - 1 && localOutPos.Y == height - 1)
						{
							double finalMaxSumForTexel = {};

							// encode and decode it properly
							for (size_t yPos = 0; yPos < height; ++yPos)	// compute max sum values in rows for y texel sum values
							{
								// use finalMaxSumForTexel and pointer arythmetic to calculate final sum for a texel, towards LEFT
								// gain max sum for X values per each y
							}

							for (size_t xPos = 0; xPos < width; ++xPos)
							{
								// use finalMaxSumForTexel and pointer arythmetic to calculate final sum for a texel, towards DOWN
								// gain max sum for Y values per each X
							}

							if (!ExclusiveMode)
							{
								// add to finalMaxSumForTexel the reference texel as well
							}
						}
					};

					//uint8_t* imageSumOutData = nullptr; // TODO
					//memcpy(commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides),commonExecuteData.inData+readBlockArrayOffset,commonExecuteData.outBlockByteSize);
				};

				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, sum, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(),clip);

				return true;
			};

			// TODO return commonExecute(state, perOutputRegion);
		}

	private:

		/*
			Since it is desired to change the way local position are obtained, it can't use default common filter functions.
		*/

		template<typename F>
		static inline void executePerRegion(const ICPUImage* image, F& f, const IImage::SBufferCopy* _begin, const IImage::SBufferCopy* _end)
		{
			CBasicImageFilterCommon::default_region_functor_t voidFunctor;
			return executePerRegion<F, CBasicImageFilterCommon::default_region_functor_t>(image, f, _begin, _end, voidFunctor);
		}

		/*
			The function will use default functor to determine which region it should take to execute operations on,
			and that why the block will be specified - area of operations going on.
		*/

		template<typename F, typename G>
		static inline void executePerRegion(const ICPUImage* image, F& f, const IImage::SBufferCopy* _begin, const IImage::SBufferCopy* _end, G& g)
		{
			for (auto it = _begin; it != _end; it++)
			{
				IImage::SBufferCopy region = *it;
				if (g(region, it))
					executePerBlock<F>(image, region, f);
			}
		}

		/* 
			Finally the function will be iterating through image passing it's local coordinates the way
			a lambde being used can use it to fill preallocated table with data to compute the sum 
		 */

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
			trueExtent = blockInfo.convertTexelsToBlocks(trueExtent);
			trueExtent.w = subresource.layerCount;

			const auto strides = region.getByteStrides(blockInfo);

			core::vector3du32_SIMD localCoord;
			for (auto& layer = localCoord[3] = 0u; layer < trueExtent.w; ++layer)			
				for (auto& zBlock = localCoord[2] = 0u; zBlock < trueExtent.z; ++zBlock)				
					for (auto& yBlock = localCoord[1] = 0u; yBlock < trueExtent.y; ++yBlock)			
						for (auto& xBlock = localCoord[0] = trueExtent.x - 1; xBlock >= 0; --xBlock)	// take a look at the inverse iteration, it's for stride caching for touching texels as much little as possible
							f(region.getByteOffset(localCoord, strides), localCoord + trueOffset);
		}
};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
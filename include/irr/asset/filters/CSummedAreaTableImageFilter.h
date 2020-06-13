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

template<bool ExclusiveMode>
class CSummedAreaTableImageFilterBase
{
	public:
		class CSummStateBase
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
				const E_SUM_MODE mode = ExclusiveMode ? ESM_EXCLUSIVE : ESM_INCLUSIVE;

				static inline uint32_t getRequiredScratchByteSize(E_FORMAT format, const asset::VkExtent3D& extent = asset::VkExtent3D())
				{
					constexpr auto decodeByteSize = 8u;
					auto channels = asset::getFormatChannelCount(format);
					return channels * decodeByteSize * extent.width * extent.height * extent.depth;
				}
		};

	protected:
		CSummedAreaTableImageFilterBase() {}
		virtual ~CSummedAreaTableImageFilterBase() {}

		static inline bool validate(CSummStateBase* state)
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
class CSummedAreaTableImageFilter : public CMatchedSizeInOutImageFilterCommon, public CSummedAreaTableImageFilterBase<ExclusiveMode>
{
	public:
		virtual ~CSummedAreaTableImageFilter() {}

		class CStateBase : public CMatchedSizeInOutImageFilterCommon::state_type, public CSummedAreaTableImageFilterBase<ExclusiveMode>::CSummStateBase {};
		using state_type = CStateBase; //!< full combined state

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			if (!CSummedAreaTableImageFilterBase<ExclusiveMode>::validate(state))
				return false;
			
			const ICPUImage::SCreationParams& inParams = state->inImage->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = state->outImage->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;

			if (state->scratchMemoryByteSize < state_type::getRequiredScratchByteSize(outFormat, {outParams.extent.width, outParams.extent.height, outParams.extent.depth}))
				return false;

			if (asset::getFormatChannelCount(outFormat) < asset::getFormatChannelCount(inFormat))
				return false;

			if (asset::getFormatClass(state->inImage->getCreationParameters().format) < asset::getFormatClass(outFormat))
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			if (isIntegerFormat(state->inImage->getCreationParameters().format))
				return executeInterprated(state, reinterpret_cast<uint64_t*>(state->scratchMemory));
			else
				return executeInterprated(state, reinterpret_cast<double*>(state->scratchMemory));
		}

	private:

		template<typename decodeTypePointer> //!< double or uint64_t
		static inline bool executeInterprated(state_type* state, decodeTypePointer scratchMemory)
		{
			using decodeType = typename std::decay<decltype(*scratchMemory)>::type;

			auto perOutputRegion = [&](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat) == getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something
				core::vector3du32_SIMD trueExtent(commonExecuteData.oit->imageExtent.width, commonExecuteData.oit->imageExtent.height, commonExecuteData.oit->imageExtent.depth);
				const auto texelByteSize = asset::getTexelOrBlockBytesize(commonExecuteData.inFormat);
				const auto currentChannelCount = asset::getFormatChannelCount(commonExecuteData.inFormat);
				static constexpr auto maxChannels = 4;

				memset(scratchMemory, 0, state->scratchMemoryByteSize);

				auto decodeEntireImageToTemporaryScratchImage = [&]() 
				{
					auto* temporaryMemory = scratchMemory;
					decodeType decodeBuffer[maxChannels] = {};

					size_t regionLayerByteOffset = {}; // TODO need to use filters API to stay at layer-mipmap memory 
					const auto* entryData = commonExecuteData.inData + regionLayerByteOffset;

					core::vector3du32_SIMD localCoord;
					for (auto& z = localCoord[2] = 0u; z < trueExtent.z; ++z)
						for (auto& y = localCoord[1] = 0u; y < trueExtent.y; ++y)
							for (auto& x = localCoord[0] = 0u; x < trueExtent.x; ++x)
							{
								const size_t independentPtrOffset = ((z * trueExtent.y + y) * trueExtent.x + x);
								auto* inDataAdress = entryData + independentPtrOffset * texelByteSize;
								auto* outDataAdress = scratchMemory + independentPtrOffset * currentChannelCount;

								const void* sourcePixels[maxChannels] = { inDataAdress, nullptr, nullptr, nullptr };
								asset::decodePixelsRuntime(commonExecuteData.inImg->getCreationParameters().format, sourcePixels, decodeBuffer, 1, 1);
								memcpy(outDataAdress, decodeBuffer, sizeof(decodeType) * currentChannelCount);
							}
				};

				decodeEntireImageToTemporaryScratchImage();

				auto mainRowCacheBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(decodeType) * currentChannelCount);
				auto mainRowCache = reinterpret_cast<decodeType*>(mainRowCacheBuffer->getPointer()); // row cache is independent, we always put to it data per each row summing everything and use it to fill column cache as well
				memset(mainRowCache, 0, mainRowCacheBuffer->getSize()); 

				constexpr auto decodeByteSize = 8;
				auto sum = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					decltype(readBlockPos) newReadBlockPos = decltype(newReadBlockPos)(readBlockPos.x, trueExtent.y - 1 - readBlockPos.y, readBlockPos.z);
					const size_t columnCachePtrOffset = ((newReadBlockPos.z * trueExtent.y + 0) * trueExtent.x + 0);
					decodeType* mainColumnCache = scratchMemory + columnCachePtrOffset * currentChannelCount; // column cache is embedded into scratch memory for not wasting memory space, beginning with (0, 0, z)

					const auto globalIndependentOffset = ((newReadBlockPos.z * trueExtent.y + newReadBlockPos.y) * trueExtent.x + newReadBlockPos.x);
					const auto outDataAdressOffsetInput = globalIndependentOffset * texelByteSize;
					const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

					auto finalPixelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(decodeType) * currentChannelCount);
					auto finalPixel = reinterpret_cast<decodeType*>(finalPixelBuffer->getPointer());
					memset(finalPixel, 0, finalPixelBuffer->getSize());
					auto fetchedPixelColorItself = commonExecuteData.inData + outDataAdressOffsetInput;

					decodeType decodeBuffer[maxChannels] = {};
					const void* sourcePixels[maxChannels] = { fetchedPixelColorItself, nullptr, nullptr, nullptr };
					asset::decodePixelsRuntime(commonExecuteData.inFormat, sourcePixels, decodeBuffer, 1, 1);

					for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
					{ 
						*(mainRowCache + channel) += decodeBuffer[channel];
						*(mainColumnCache + channel) += decodeBuffer[channel];

						*(finalPixel + channel)
						= (newReadBlockPos.x > 0 ? mainRowCache[channel] : 0)
						+ (newReadBlockPos.y < trueExtent.y - 1 ? *(mainColumnCache + channel) : 0)
						+ (!ExclusiveMode ? decodeBuffer[channel] : 0);
					}

					memcpy(scratchMemory + outDataAdressOffsetScratch, finalPixel, sizeof(decodeType) * currentChannelCount);

					if (newReadBlockPos.x == commonExecuteData.oit->imageExtent.width - 1) // reset row cache when changing y
						memset(mainRowCache, 0, sizeof(decodeType) * currentChannelCount);
				};

				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, sum, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);

				auto encodeTo = [&](ICPUImage* outImage)
				{
					core::vector3du32_SIMD localCoord;
					for (auto& z = localCoord[2] = 0u; z < trueExtent.z; ++z)
						for (auto& y = localCoord[1] = 0u; y < trueExtent.y; ++y)
							for (auto& x = localCoord[0] = 0u; x < trueExtent.x; ++x)
							{
								const auto globalIndependentOffset = ((localCoord.z * trueExtent.y + localCoord.y) * trueExtent.x + localCoord.x);
								const auto outDataAdressOffsetInput = globalIndependentOffset * texelByteSize;
								const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

								auto* outScratchAdress = scratchMemory + outDataAdressOffsetScratch;
								auto* entryData = commonExecuteData.inData + outDataAdressOffsetInput; // TODO need to use filters API to stay at layer-mipmap memory
								asset::encodePixelsRuntime(commonExecuteData.inFormat, outScratchAdress, entryData);
							}
				};

				encodeTo(commonExecuteData.outImg);
				return true;
			};

			return commonExecute(state, perOutputRegion);
		}

};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
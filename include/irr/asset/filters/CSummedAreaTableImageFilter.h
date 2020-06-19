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
				
				uint8_t*	scratchMemory = nullptr;										//!< memory covering all regions used for temporary filling within computation of sum values
				size_t	scratchMemoryByteSize = {};											//!< required byte size for entire scratch memory

				static inline size_t getRequiredScratchByteSize(const ICPUImage* inputImage)
				{
					constexpr auto decodeByteSize = 8u;
					auto channels = asset::getFormatChannelCount(inputImage->getCreationParameters().format);
					const auto regions = inputImage->getRegions();

					size_t retval = {};
					for (auto& region = regions.begin(); region < regions.end(); ++region)
						retval += region->imageExtent.width * region->imageExtent.height * region->imageExtent.depth;

					retval *= channels * decodeByteSize;

					return retval;
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

			if (state->scratchMemoryByteSize < state_type::getRequiredScratchByteSize(state->inImage))
				return false;

			if (asset::getFormatChannelCount(outFormat) != asset::getFormatChannelCount(inFormat))
				return false;

			if (asset::getFormatClass(state->inImage->getCreationParameters().format) < asset::getFormatClass(outFormat))
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			auto checkFormat = state->inImage->getCreationParameters().format;
			if (isIntegerFormat(checkFormat) || isSRGBFormat(checkFormat))
				return executeInterprated(state, reinterpret_cast<uint64_t*>(state->scratchMemory));
			else
				return executeInterprated(state, reinterpret_cast<double*>(state->scratchMemory));
		}

	private:

		template<typename decodeType> //!< double or uint64_t
		static inline bool executeInterprated(state_type* state, decodeType* scratchMemory)
		{
			const asset::E_FORMAT inFormat = state->inImage->getCreationParameters().inFormat;
			const auto texelByteSize = asset::getTexelOrBlockBytesize(inFormat);
			const auto currentChannelCount = asset::getFormatChannelCount(inFormat);
			static constexpr auto maxChannels = 4u;

			auto decodeEntireImageToTemporaryScratchImage = [&]()
			{
				decodeType decodeBuffer[maxChannels] = {};
				const auto regions = state->inImage->getRegions();
				const ICPUImage::SCreationParams& imageInfo = state->inImage->getCreationParameters();

				size_t outRegionBufferOffset = {};
				for (const IImage::SBufferCopy*& region = regions.begin(); region < regions.end(); ++region)
				{
					const auto& trueExtent = region->getExtent(); // TODO row length
					const auto inLayerByteSize = trueExtent.width * trueExtent.height * trueExtent.depth * texelByteSize;
					const auto outLayerByteSize = trueExtent.width * trueExtent.height * trueExtent.depth * sizeof(decodeType) * currentChannelCount;

					for (uint16_t layer = 0; layer < imageInfo.arrayLayers; ++layer)
					{
						const uint8_t* inImageData = reinterpret_cast<uint8_t*>(state->inImage->getBuffer()->getPointer()) + region->bufferOffset + layer * inLayerByteSize;
						const decodeType& outImageData = scratchMemory + outRegionBufferOffset + layer * outLayerByteSize;

						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < trueExtent.z; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueExtent.y; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueExtent.x; ++x)
								{
									const size_t independentPtrOffset = ((z * trueExtent.y + y) * trueExtent.x + x);
									auto* inDataAdress = inImageData + independentPtrOffset * texelByteSize;
									decodeType* outDataAdress = outImageData + independentPtrOffset * currentChannelCount;

									const void* sourcePixels[maxChannels] = { inDataAdress, nullptr, nullptr, nullptr };
									asset::decodePixelsRuntime(inFormat, sourcePixels, decodeBuffer, 1, 1);
									memcpy(outDataAdress, decodeBuffer, sizeof(decodeType) * currentChannelCount);
								}
					}

					outRegionBufferOffset += outLayerByteSize * imageInfo.arrayLayers;
				}
			};

			decodeEntireImageToTemporaryScratchImage();

			// TODO, it's wrong now
			auto perOutputRegion = [&](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat) == getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something
				core::vector3du32_SIMD trueExtent(commonExecuteData.oit->imageExtent.width, commonExecuteData.oit->imageExtent.height, commonExecuteData.oit->imageExtent.depth);

				memset(scratchMemory, 0, state->scratchMemoryByteSize);

				auto imageTotalCacheBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(decodeType) * currentChannelCount * commonExecuteData.inParams.extent.depth);
				auto imageTotalCache = reinterpret_cast<decodeType*>(imageTotalCacheBuffer->getPointer());
				memset(imageTotalCache, 0, imageTotalCacheBuffer->getSize());

				auto mainRowCacheBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(decodeType) * currentChannelCount);
				auto mainRowCache = reinterpret_cast<decodeType*>(mainRowCacheBuffer->getPointer()); // row cache is independent, we always put to it data per each row summing everything and use it to fill column cache as well
				memset(mainRowCache, 0, mainRowCacheBuffer->getSize()); 

				constexpr auto decodeByteSize = 8;
				auto sum = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					decltype(readBlockPos) newReadBlockPos = decltype(newReadBlockPos)(readBlockPos.x, trueExtent.y - 1 - readBlockPos.y, readBlockPos.z); // todo take readBlockArrayOffset
					const size_t columnCachePtrOffset = ((newReadBlockPos.z * trueExtent.y + 0) * trueExtent.x + newReadBlockPos.x);
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
						*(mainRowCache + channel) += newReadBlockPos.x > 0 ? decodeBuffer[channel] : 0;
						*(mainColumnCache + channel) += newReadBlockPos.y < trueExtent.y - 1 ? decodeBuffer[channel] : 0;

						*(finalPixel + channel)
						= mainRowCache[channel]
						+ (newReadBlockPos.y < trueExtent.y - 1 ? *(mainColumnCache + channel) : 0) // gotta check it as well as above to work correctly
						+ (!ExclusiveMode ? decodeBuffer[channel] : 0);
					}

					memcpy(scratchMemory + outDataAdressOffsetScratch, finalPixel, sizeof(decodeType) * currentChannelCount);

					if (newReadBlockPos.x == commonExecuteData.oit->imageExtent.width - 1) // reset row cache when changing y
					{
						if (newReadBlockPos.y == 0) // sum values in (maxX, maxY, z)
							memcpy(imageTotalCache + newReadBlockPos.z * currentChannelCount, finalPixel, sizeof(decodeType) * currentChannelCount);
							
						memset(mainRowCache, 0, sizeof(decodeType) * currentChannelCount);
					}
				};

				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, sum, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);

				auto normalizeScratch = [&]()
				{
					core::vector3du32_SIMD localCoord;
					for (auto& z = localCoord[2] = 0u; z < trueExtent.z; ++z)
						for (auto& y = localCoord[1] = 0u; y < trueExtent.y; ++y)
							for (auto& x = localCoord[0] = 0u; x < trueExtent.x; ++x)
							{
								const auto globalIndependentOffset = ((localCoord.z * trueExtent.y + localCoord.y) * trueExtent.x + localCoord.x);
								const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

								auto* entryScratchAdress = scratchMemory + outDataAdressOffsetScratch;
								
								for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
									*(entryScratchAdress + channel) /= *(imageTotalCache + localCoord.z * currentChannelCount + channel);
							}
				};

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

								auto* entryScratchAdress = scratchMemory + outDataAdressOffsetScratch;
								auto* outData = commonExecuteData.outData + outDataAdressOffsetInput; // TODO need to use filters API to stay at layer-mipmap memory
								asset::encodePixelsRuntime(commonExecuteData.inFormat, outData, entryScratchAdress);
							}
				};

				if (!asset::isNormalizedFormat(commonExecuteData.inFormat))
					normalizeScratch();

				encodeTo(commonExecuteData.outImg);
				return true;
			};

			return commonExecute(state, perOutputRegion);
		}

};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
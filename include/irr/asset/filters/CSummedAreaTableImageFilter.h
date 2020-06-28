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
				
				static inline constexpr size_t decodeTypeByteSize = sizeof(double);
				uint8_t*	scratchMemory = nullptr;										//!< memory covering all regions used for temporary filling within computation of sum values
				size_t	scratchMemoryByteSize = {};											//!< required byte size for entire scratch memory
				bool normalizeImageByTotalSATValues = false;								//!< after sum performation, the option decide whether to divide the entire image by the max sum values in (maxX, 0, z) 
				
				static inline size_t getRequiredScratchByteSize(const ICPUImage* inputImage, const ICPUImage* outputImage)
				{
					auto channels = asset::getFormatChannelCount(inputImage->getCreationParameters().format);
					const auto regions = inputImage->getRegions();

					size_t retval = {};
					for (const auto* region = regions.begin(); region != regions.end(); ++region)
						retval += (region->bufferRowLength ? region->bufferRowLength : region->imageExtent.width) * region->imageExtent.height * region->imageExtent.depth;

					retval *= inputImage->getCreationParameters().arrayLayers * channels * decodeTypeByteSize;
					cachesOffset = retval;

					retval += (imageTotalScratchCacheByteSize = asset::getFormatChannelCount(inputImage->getCreationParameters().format) * inputImage->getCreationParameters().extent.depth);
					retval += (imageTotalOutputCacheByteSize = asset::getTexelOrBlockBytesize(outputImage->getCreationParameters().format) * outputImage->getCreationParameters().extent.depth); 

					return retval;
				}

				/*
					It returns a pointer to max sum values in the output image.
					A user is responsible to reinterprate the memory correctly. 
				*/

				inline auto getImageTotalOutputCacheBuffersPointer()
				{
					return imageTotalCacheOutput;
				}

			protected:

				static inline size_t cachesOffset = {};
				static inline size_t imageTotalScratchCacheByteSize = {};
				static inline size_t imageTotalOutputCacheByteSize = {};
				static inline uint8_t* imageTotalCacheScratch = nullptr;		//!< A pointer to buffers holding total image SAT values in decoding mode.
				static inline uint8_t* imageTotalCacheOutput = nullptr;			//!< A pointer to buffers holding total image SAT values after encoding to image. A user is responsible to reinterprate the memory correctly.
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

		class CStateBase : public CMatchedSizeInOutImageFilterCommon::state_type, public CSummedAreaTableImageFilterBase<ExclusiveMode>::CSummStateBase { friend class CSummedAreaTableImageFilter<ExclusiveMode>; };
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

			if (state->scratchMemoryByteSize < state_type::getRequiredScratchByteSize(state->inImage, state->outImage))
				return false;

			if (state->inMipLevel != state->outMipLevel)
				return false;

			if (asset::getFormatChannelCount(outFormat) != asset::getFormatChannelCount(inFormat))
				return false;

			if (asset::getFormatClass(inFormat) >= asset::getFormatClass(outFormat))
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
			{
				state->imageTotalCacheScratch = nullptr;
				state->imageTotalCacheOutput = nullptr;
				return false;
			}

			auto checkFormat = state->inImage->getCreationParameters().format;
			if (isIntegerFormat(checkFormat))
				return executeInterprated(state, reinterpret_cast<uint64_t*>(state->scratchMemory));
			else
				return executeInterprated(state, reinterpret_cast<double*>(state->scratchMemory));
		}	

	private:

		template<typename decodeType> //!< double or uint64_t
		static inline bool executeInterprated(state_type* state, decodeType* scratchMemory)
		{
			const asset::E_FORMAT inFormat = state->inImage->getCreationParameters().format;
			const asset::E_FORMAT outFormat = state->outImage->getCreationParameters().format;
			const auto inTexelByteSize = asset::getTexelOrBlockBytesize(inFormat);
			const auto outTexelByteSize = asset::getTexelOrBlockBytesize(outFormat);
			const auto currentChannelCount = asset::getFormatChannelCount(inFormat);
			static constexpr auto maxChannels = 4u;

			uint8_t* const totalImageCacheScratch = state->imageTotalCacheScratch = state->scratchMemory + state->cachesOffset;
			uint8_t* const totalImageCacheOutput = state->imageTotalCacheOutput = state->imageTotalCacheScratch + state->imageTotalScratchCacheByteSize;

			auto decodeEntireImageToTemporaryScratchImage = [&]()
			{
				decodeType decodeBuffer[maxChannels] = {};
				const auto inRegions = state->inImage->getRegions();
				const ICPUImage::SCreationParams& imageInfo = state->inImage->getCreationParameters();

				memset(scratchMemory, 0, state->scratchMemoryByteSize);

				size_t outRegionBufferOffset = {};
				for (const IImage::SBufferCopy* region = inRegions.begin(); region != inRegions.end(); ++region)
				{
					const auto trueExtent = asset::VkExtent3D({ region->bufferRowLength ? region->bufferRowLength : region->imageExtent.width, region->imageExtent.height, region->imageExtent.depth });
					const auto inLayerByteSize = trueExtent.width * trueExtent.height * trueExtent.depth * inTexelByteSize;
					const auto outLayerOffset = trueExtent.width * trueExtent.height * trueExtent.depth * currentChannelCount;

					for (uint16_t layer = 0; layer < imageInfo.arrayLayers; ++layer)
					{
						const uint8_t* inImageData = reinterpret_cast<const uint8_t*>(state->inImage->getBuffer()->getPointer()) + region->bufferOffset + layer * inLayerByteSize;
						decodeType* outImageData = scratchMemory + outRegionBufferOffset + layer * outLayerOffset;

						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < trueExtent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueExtent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueExtent.width; ++x)
								{
									const size_t independentPtrOffset = ((z * trueExtent.height + y) * trueExtent.width + x);
									auto* inDataAdress = inImageData + independentPtrOffset * inTexelByteSize;
									decodeType* outDataAdress = outImageData + independentPtrOffset * currentChannelCount;

									const void* sourcePixels[maxChannels] = { inDataAdress, nullptr, nullptr, nullptr };
									asset::decodePixelsRuntime(inFormat, sourcePixels, decodeBuffer, 1, 1);
									memcpy(outDataAdress, decodeBuffer, sizeof(decodeType) * currentChannelCount);
								}
					}

					outRegionBufferOffset += outLayerOffset * imageInfo.arrayLayers;
				}
			};

			decodeEntireImageToTemporaryScratchImage();

			auto perOutputRegion = [&](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				core::vector3du32_SIMD trueExtent(commonExecuteData.oit->imageExtent.width, commonExecuteData.oit->imageExtent.height, commonExecuteData.oit->imageExtent.depth);
				const size_t singleScratchLayerOffset = (trueExtent.x * trueExtent.y * trueExtent.z * trueExtent.w * currentChannelCount);

				auto getTranslatedInRegionOffset = [&]() -> std::intptr_t
				{
					std::intptr_t retval = {}, handledRegionAdress = reinterpret_cast<std::intptr_t>(commonExecuteData.inRegions.begin()); 
					for (auto inRegion = commonExecuteData.inRegions.begin(); inRegion != commonExecuteData.inRegions.end(); ++inRegion)
					{
						std::intptr_t currentAdress = reinterpret_cast<std::intptr_t>(inRegion);
						
						if (currentAdress == handledRegionAdress)
							return retval;

						retval += inRegion->imageExtent.width * inRegion->imageExtent.height * inRegion->imageExtent.depth * currentChannelCount * state->inImage->getCreationParameters().arrayLayers;
					}

					return {};
				};

				auto mainRowCacheBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(decodeType) * currentChannelCount);
				auto mainRowCache = reinterpret_cast<decodeType*>(mainRowCacheBuffer->getPointer()); // row cache is independent, we always put to it data per each row summing everything and use it to fill column cache as well
				memset(mainRowCache, 0, mainRowCacheBuffer->getSize()); 

				auto mainColumnCacheBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(mainRowCacheBuffer->getSize() * commonExecuteData.oit->imageExtent.width);
				decodeType * const mainColumnCache = reinterpret_cast<decodeType*>(mainColumnCacheBuffer->getPointer());
				memset(mainColumnCache, 0, mainColumnCacheBuffer->getSize());

				const size_t globalTranslatedDecodeTypeRegionOffset = getTranslatedInRegionOffset();

				auto sum = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					decltype(readBlockPos) newReadBlockPos = decltype(newReadBlockPos)(readBlockPos.x, trueExtent.y - 1 - readBlockPos.y, readBlockPos.z, readBlockPos.w);
					const size_t columnCachePtrOffset = ((newReadBlockPos.z * trueExtent.y + 0) * trueExtent.x + newReadBlockPos.x);
					decodeType* const currentColumnCache = mainColumnCache + newReadBlockPos.x * currentChannelCount;
					
					const auto globalIndependentOffset = ((newReadBlockPos.z * trueExtent.y + newReadBlockPos.y) * trueExtent.x + newReadBlockPos.x);
					const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;
					const auto layersOffset = singleScratchLayerOffset * newReadBlockPos.w;

					auto finalPixelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(mainRowCacheBuffer->getSize());
					auto finalPixel = reinterpret_cast<decodeType*>(finalPixelBuffer->getPointer());
					memset(finalPixel, 0, finalPixelBuffer->getSize());
	
					decodeType* decodeBuffer = scratchMemory + globalTranslatedDecodeTypeRegionOffset + outDataAdressOffsetScratch + layersOffset;

					for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
					{ 
						*(currentColumnCache + channel) += *(mainRowCache + channel) += decodeBuffer[channel];
						*(finalPixel + channel) = *(currentColumnCache + channel) - (ExclusiveMode ? decodeBuffer[channel] : 0);
					}

					memcpy(scratchMemory + globalTranslatedDecodeTypeRegionOffset + outDataAdressOffsetScratch + layersOffset, finalPixel, mainRowCacheBuffer->getSize());

					if (newReadBlockPos.x == commonExecuteData.oit->imageExtent.width - 1) // reset row cache when changing y
					{
						if (newReadBlockPos.y == 0) // sum values in (maxX, 0, z) and remove values from column cache
						{
							memcpy(totalImageCacheScratch + newReadBlockPos.z * mainRowCacheBuffer->getSize(), finalPixel, mainRowCacheBuffer->getSize());
							memset(mainColumnCache, 0, mainColumnCacheBuffer->getSize());
						}
							
						memset(mainRowCache, 0, mainRowCacheBuffer->getSize());
					}
				};

				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, sum, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);

				auto normalizeScratch = [&]()
				{
					core::vector3du32_SIMD localCoord;
					for (auto& w = localCoord[3] = 0u; w < trueExtent.w; ++w)
						for (auto& z = localCoord[2] = 0u; z < trueExtent.z; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueExtent.y; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueExtent.x; ++x)
								{
									const auto globalIndependentOffset = ((localCoord.z * trueExtent.y + localCoord.y) * trueExtent.x + localCoord.x);
									const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

									auto* entryScratchAdress = scratchMemory + globalTranslatedDecodeTypeRegionOffset + outDataAdressOffsetScratch + singleScratchLayerOffset * w;

									const auto totalImageBufferOffset = localCoord.z * currentChannelCount;
									for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
										*(entryScratchAdress + channel) /= *(reinterpret_cast<decodeType*>(totalImageCacheScratch) + totalImageBufferOffset + channel);
								}
				};

				auto encodeTo = [&](ICPUImage* outImage)
				{
					core::vector3du32_SIMD localCoord;
					for (auto& w = localCoord[3] = 0u; w < trueExtent.w; ++w)
						for (auto& z = localCoord[2] = 0u; z < trueExtent.z; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueExtent.y; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueExtent.x; ++x)
								{
									const auto globalIndependentOffset = ((localCoord.z * trueExtent.y + localCoord.y) * trueExtent.x + localCoord.x);
									const auto outDataAdressOffsetInput = commonExecuteData.oit->bufferOffset + outTexelByteSize * (state->outBaseLayer * trueExtent.x * trueExtent.y * trueExtent.z + globalIndependentOffset);
									const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

									auto* entryScratchAdress = scratchMemory + globalTranslatedDecodeTypeRegionOffset + outDataAdressOffsetScratch + singleScratchLayerOffset * w;
									auto* outData = commonExecuteData.outData + outDataAdressOffsetInput;
									asset::encodePixelsRuntime(commonExecuteData.outFormat, outData, entryScratchAdress);

									if (x == trueExtent.x - 1 && y == 0)
										memcpy(totalImageCacheOutput + z * outTexelByteSize, outData, outTexelByteSize);
								}
				};

				if(state->normalizeImageByTotalSATValues)
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
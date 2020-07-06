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
					const auto& inputCreationParams = inputImage->getCreationParameters();
					const auto channels = asset::getFormatChannelCount(inputCreationParams.format);
					const auto regions = inputImage->getRegions();
					const auto& trueExtent = inputCreationParams.extent;

					size_t retval = cachesOffset = trueExtent.width * trueExtent.height * trueExtent.depth * inputCreationParams.arrayLayers * channels * decodeTypeByteSize;

					retval += (imageTotalScratchCacheByteSize = channels * inputImage->getCreationParameters().extent.depth);
					retval += (imageTotalOutputCacheByteSize = asset::getTexelOrBlockBytesize(outputImage->getCreationParameters().format) * outputImage->getCreationParameters().extent.depth); 

					retval += (mainRowScratchCacheByteSize = decodeTypeByteSize * channels);
					retval += (mainColumnScratchCacheByteSize = mainRowScratchCacheByteSize * trueExtent.width);

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
				static inline size_t mainRowScratchCacheByteSize = {};
				static inline size_t mainColumnScratchCacheByteSize = {};
				static inline uint8_t* imageTotalCacheScratch = nullptr;		//!< A pointer to buffers holding total image SAT values in decoding mode.
				static inline uint8_t* imageTotalCacheOutput = nullptr;			//!< A pointer to buffers holding total image SAT values after encoding to image. A user is responsible to reinterprate the memory correctly.
				static inline uint8_t* mainRowCacheScratch = nullptr;			//!< A pointer to a buffer holding current single texel values that are summed within row iteration.
				static inline uint8_t* mainColumnCacheScratch = nullptr;		//!< A pointer to a buffer holding current summed values per texel for choosen x coordinate in a column orientation.
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
			const auto outScratchLayerOffset = state->extent.width * state->extent.height * state->extent.depth * currentChannelCount;

			uint8_t* const totalImageCacheScratch = state->imageTotalCacheScratch = state->scratchMemory + state->cachesOffset;
			uint8_t* const totalImageCacheOutput = state->imageTotalCacheOutput = state->imageTotalCacheScratch + state->imageTotalScratchCacheByteSize;
			decodeType* const mainRowCacheScratch = reinterpret_cast<decodeType*>(state->mainRowCacheScratch = state->imageTotalCacheOutput + state->imageTotalOutputCacheByteSize);
			decodeType* const mainColumnCacheScratch = reinterpret_cast<decodeType*>(state->mainColumnCacheScratch = state->mainRowCacheScratch + state->mainRowScratchCacheByteSize);

			memset(scratchMemory, 0, state->scratchMemoryByteSize);

			auto perOutputRegionDecode = [&](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				/*
					Since regions may specify areas in a layer on a certain mipmap - it is desired to
					decode an entire image itereting through all the regions overlapping as first
				*/

				auto decode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					decodeType decodeBuffer[maxChannels] = {};
					decodeType* outImageData = scratchMemory + readBlockPos.w * outScratchLayerOffset;
					
					auto* inDataAdress = commonExecuteData.inData + readBlockArrayOffset;
					decodeType* outDataAdress = outImageData + ((readBlockPos.z * state->extent.height + readBlockPos.y) * state->extent.width + readBlockPos.x) * currentChannelCount;

					const void* sourcePixels[maxChannels] = { inDataAdress, nullptr, nullptr, nullptr };
					asset::decodePixelsRuntime(inFormat, sourcePixels, decodeBuffer, 1, 1);
					memcpy(outDataAdress, decodeBuffer, sizeof(decodeType) * currentChannelCount);
				};

				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, decode, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);

				return true;
			};

			bool decodeStatus = commonExecute(state, perOutputRegionDecode);

			if (decodeStatus)
			{
				const auto arrayLayers = state->inImage->getCreationParameters().arrayLayers;
				const auto outRegions = state->outImage->getRegions(state->outMipLevel);
				const auto trueOutExtent = asset::VkExtent3D({ outRegions.begin()->bufferRowLength ? outRegions.begin()->bufferRowLength : outRegions.begin()->imageExtent.width, outRegions.begin()->imageExtent.height, outRegions.begin()->imageExtent.depth });
				const auto regionBufferOffset = outRegions.begin()->bufferOffset;
				auto* imageEntireOutData = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());
				const size_t singleScratchLayerOffset = (state->extent.width * state->extent.height * state->extent.depth * currentChannelCount); 

				auto sum = [&](core::vectorSIMDu32 readBlockPos) -> void
				{
					decltype(readBlockPos) newReadBlockPos = decltype(newReadBlockPos)(readBlockPos.x, trueOutExtent.height - 1 - readBlockPos.y, readBlockPos.z, readBlockPos.w);
					decodeType* const currentColumnCache = mainColumnCacheScratch + newReadBlockPos.x * currentChannelCount;

					const auto globalIndependentOffset = ((newReadBlockPos.z * state->extent.height + newReadBlockPos.y) * state->extent.width + newReadBlockPos.x);
					const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;
					const auto layersOffset = singleScratchLayerOffset * newReadBlockPos.w;

					auto finalPixelBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(state->mainRowScratchCacheByteSize);
					auto finalPixel = reinterpret_cast<decodeType*>(finalPixelBuffer->getPointer());
					memset(finalPixel, 0, finalPixelBuffer->getSize());

					decodeType* decodeBuffer = scratchMemory + outDataAdressOffsetScratch + layersOffset;

					for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
					{
						*(currentColumnCache + channel) += *(mainRowCacheScratch + channel) += decodeBuffer[channel];
						*(finalPixel + channel) = *(currentColumnCache + channel) - (ExclusiveMode ? decodeBuffer[channel] : 0);
					}

					memcpy(scratchMemory + outDataAdressOffsetScratch + layersOffset, finalPixel, state->mainRowScratchCacheByteSize);

					if (newReadBlockPos.x == trueOutExtent.width - 1) // reset row cache when changing y
					{
						if (newReadBlockPos.y == 0) // sum values in (maxX, 0, z) and remove values from column cache
						{
							memcpy(totalImageCacheScratch + newReadBlockPos.z * state->mainRowScratchCacheByteSize, finalPixel, state->mainRowScratchCacheByteSize);
							memset(mainColumnCacheScratch, 0, state->mainColumnScratchCacheByteSize);
						}

						memset(mainRowCacheScratch, 0, state->mainRowScratchCacheByteSize);
					}
				};

				{
					core::vector3du32_SIMD localCoord;
					for (auto& w = localCoord[3] = 0u; w < arrayLayers; ++w)
						for (auto& z = localCoord[2] = 0u; z < trueOutExtent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueOutExtent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueOutExtent.width; ++x)
									sum(core::vectorSIMDu32( x, y, z, w ));
				}

				auto normalizeScratch = [&]()
				{
					core::vector3du32_SIMD localCoord;
					for (auto& w = localCoord[3] = 0u; w < arrayLayers; ++w)
						for (auto& z = localCoord[2] = 0u; z < trueOutExtent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueOutExtent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueOutExtent.width; ++x)
								{
									const auto globalIndependentOffset = ((localCoord.z * state->extent.height + localCoord.y) * state->extent.width + localCoord.x);
									const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

									auto* entryScratchAdress = scratchMemory + outDataAdressOffsetScratch + singleScratchLayerOffset * w;

									const auto totalImageBufferOffset = localCoord.z * currentChannelCount;
									for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
										*(entryScratchAdress + channel) /= *(reinterpret_cast<decodeType*>(totalImageCacheScratch) + totalImageBufferOffset + channel);
								}
				};

				auto encodeTo = [&](ICPUImage* outImage)
				{
					core::vector3du32_SIMD localCoord;
					for (auto& w = localCoord[3] = 0u; w < arrayLayers; ++w)
						for (auto& z = localCoord[2] = 0u; z < trueOutExtent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < trueOutExtent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < trueOutExtent.width; ++x)
								{
									const auto globalIndependentOffset = ((localCoord.z * state->extent.height + localCoord.y) * state->extent.width + localCoord.x);
									const auto outDataAdressOffsetInput = regionBufferOffset + outTexelByteSize * (w * state->extent.width * state->extent.height * state->extent.depth + globalIndependentOffset);
									const auto outDataAdressOffsetScratch = globalIndependentOffset * currentChannelCount;

									auto* entryScratchAdress = scratchMemory  + outDataAdressOffsetScratch + singleScratchLayerOffset * w;
									auto* outData = imageEntireOutData + outDataAdressOffsetInput;
									asset::encodePixelsRuntime(outFormat, outData, entryScratchAdress);

									if (x == trueOutExtent.width - 1 && y == 0)
										memcpy(totalImageCacheOutput + z * outTexelByteSize, outData, outTexelByteSize);
								}
				};

				if (state->normalizeImageByTotalSATValues)
					if (!asset::isNormalizedFormat(inFormat))
						normalizeScratch();

				encodeTo(state->outImage);
				return true;
			}
			else
				return false;
		}

};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
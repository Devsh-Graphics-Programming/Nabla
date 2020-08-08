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
				bool normalizeImageByTotalSATValues = false;								//!< after sum performation division will be performed for the entire image by the max sum values in (maxX, 0, z) depending on input image - needed for UNORM and SNORM
				
				static inline size_t getRequiredScratchByteSize(const ICPUImage* inputImage, asset::VkExtent3D extent)
				{
					const auto& inputCreationParams = inputImage->getCreationParameters();
					const auto channels = asset::getFormatChannelCount(inputCreationParams.format);

					size_t retval = cachesOffset = extent.width * extent.height * extent.depth * channels * decodeTypeByteSize;
					
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

		class CStateBase : public CMatchedSizeInOutImageFilterCommon::state_type, public CSummedAreaTableImageFilterBase<ExclusiveMode>::CSummStateBase 
		{ 
			public:
				CStateBase() = default;
				virtual ~CStateBase() = default;

			private:

				friend class CSummedAreaTableImageFilter<ExclusiveMode>;
		};
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

			if (asset::getFormatChannelCount(outFormat) != asset::getFormatChannelCount(inFormat))
				return false;

			if (asset::getFormatClass(inFormat) >= asset::getFormatClass(outFormat)) // TODO in future! Change to a function checking a bit-depth for an each single channel
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

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
			const auto arrayLayers = state->inImage->getCreationParameters().arrayLayers;
			static constexpr auto maxChannels = 4u;

			#ifdef _IRR_DEBUG
			memset(scratchMemory, 0, state->scratchMemoryByteSize);
			#endif // _IRR_DEBUG

			const auto&& [copyInBaseLayer, copyOutBaseLayer, copyLayerCount] = std::make_tuple(state->inBaseLayer, state->outBaseLayer, state->layerCount);

			auto resetState = [&]()
			{
				state->inBaseLayer = copyInBaseLayer;
				state->outBaseLayer = copyOutBaseLayer;
				state->layerCount = copyLayerCount;
			};

			core::vector3du32_SIMD localCoord;
			for (auto& w = localCoord[3] = 0u; w < copyLayerCount - copyInBaseLayer; ++w) 
			{
				std::array<decodeType, maxChannels> minDecodeValues = {};
				std::array<decodeType, maxChannels> maxDecodeValues = {};

				++state->inBaseLayer;
				++state->outBaseLayer;
				state->layerCount = 1u;

				{
					auto inData = reinterpret_cast<uint8_t*>(state->inImage->getBuffer()->getPointer());
					const auto blockDims = asset::getBlockDimensions(state->inImage->getCreationParameters().format);
					const auto texelByteSizeScratch = sizeof(decodeType) * currentChannelCount;

					auto decode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						auto newReadBlockPos = (readBlockPos - decltype(readBlockPos)(state.inOffset.x, state.inOffset.y, state.inOffset.z));

						auto localOutPos = newReadBlockPos * blockDims;
						auto* inDataAdress = inData + readBlockArrayOffset;
						decodeType* outDataAdress = scratchMemory + newReadBlockPos * blockDims * currentChannelCount; // not sure

						constexpr uint8_t maxPlanes = 4;
						const void* inSourcePixels[maxPlanes] = { inDataAdress, nullptr, nullptr, nullptr };

						for (auto blockY = 0u; blockY < blockDims.y; blockY++)
							for (auto blockX = 0u; blockX < blockDims.x; blockX++)
							{
								decodeType decodeBuffer[maxChannels] = {};

								asset::decodePixelsRuntime(inFormat, inSourcePixels, decodeBuffer, blockX, blockY);
								memcpy(outDataAdress, decodeBuffer, texelByteSizeScratch);
								outDataAdress += texelByteSizeScratch;
							}
					};

					auto& inRegions = state->inImage->getRegions(state->inMipLevel);
					CBasicImageFilterCommon::executePerRegion(state->inImage, decode, inRegions.begin(), inRegions.end());
				}

				auto perOutputRegionEncode = [&](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
				{
					const auto blockDims = asset::getBlockDimensions(commonExecuteData.outFormat);
					const auto texelOrBlockByteSize = asset::getTexelOrBlockBytesize(commonExecuteData.outFormat);

					auto encode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						auto newReadBlockPos = (readBlockPos - decltype(readBlockPos)(state.outOffset.x, state.outOffset.y, state.outOffset.z));

						auto localOutPos = readBlockPos * blockDims + commonExecuteData.offsetDifference; 
						auto* outDataAdress = commonExecuteData.outData + commonExecuteData.oit->getByteOffset(localOutPos, commonExecuteData.outByteStrides);
						const auto outDataAdressOffsetScratch = newReadBlockPos * blockDims * currentChannelCount; // I dont think so actually
						auto* entryScratchAdress = scratchMemory + outDataAdressOffsetScratch;

						for (auto blockY = 0u; blockY < blockDims.y; blockY++)
							for (auto blockX = 0u; blockX < blockDims.x; blockX++)
							{
								asset::encodePixelsRuntime(outFormat, outDataAdress, entryScratchAdress); // overrrides texels, so region-overlapping case is fine
								//outDataAdress +=				TODO: sleepy, I will look at it tomorrow, seems I screwd up
								// okay Matt said we don't support encoding to BC so that clears a little
								// correct it
							}	
					};

					CBasicImageFilterCommon::executePerRegion(commonExecuteData.outImg, encode, commonExecuteData.outRegions.begin(), commonExecuteData.outRegions.end(), clip);

					return true;
				};

				{
					auto extent = state->outImage->getMipSize(state->outMipLevel);
					
					auto getScratchPixel = [&](core::vector4di32_SIMD readBlockPos) -> decodeType*
					{
						const auto offset = ((readBlockPos.z * state->extent.height + readBlockPos.y) * state->extent.width + readBlockPos.x) * currentChannelCount;
						return scratchMemory + offset;
					};

					auto sum = [&](core::vectorSIMDi32 readBlockPos) -> void
					{
						auto current = getScratchPixel(readBlockPos);

						auto areAxisSafe = [&]()
						{
							const auto position = core::vectorSIMDi32(0u, 0u, 0u, 0u);
	
							return core::vector4du32_SIMD
							(
								readBlockPos.x > position.x,
								readBlockPos.y > position.y,
								readBlockPos.z > position.z,
								readBlockPos.w > position.w
							);
						};

						auto addScratchPixelToCurrentOne = [&](decodeType* values)
						{
							for (auto i = 0; i < currentChannelCount; ++i)
								current[i] += values[i];
						};
						
						auto substractScratchPixelFromCurrentOne = [&](decodeType* values)
						{
							for (auto i = 0; i < currentChannelCount; ++i)
								current[i] -= values[i];
						};

						const auto axisSafe = areAxisSafe();
						
						if (axisSafe.z)
							addScratchPixelToCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(0, 0, 1, 0)));					// add box x<=current_x && y<=current_y && z<current_z
						if (axisSafe.y)
						{
							addScratchPixelToCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(0, 1, 0, 0)));					// add box x<=current_x && y<current_y && z<=current_z
							if (axisSafe.z)
								substractScratchPixelFromCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(0, 1, 1, 0)));		// remove overlap box x<=current_x && y<current_y && z<current_z
						}

						/*
							Now I have the sum of all layers below me in the bound x<=current_x && y<=current_y && z<current_z and the current pixel.
							Time to add the missing top layer pixels.
						*/

						if (axisSafe.x)
						{
							addScratchPixelToCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(1, 0, 0, 0)));					 // add box x<current_x && y<=current_y && z<=current_z
							if (axisSafe.z)
								substractScratchPixelFromCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(1, 0, 1, 0)));		 // remove overlap box x<current_x && y<=current_y && z<current_z
							if (axisSafe.y)
							{
								substractScratchPixelFromCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(1, 1, 0, 0)));		 // remove future overlap box x<current_x && y<current_y && z<=current_z
								if (axisSafe.z)
									addScratchPixelToCurrentOne(getScratchPixel(readBlockPos - core::vectorSIMDi32(1, 1, 1, 0)));			 // add box x<current_x && y<current_y && z<current_z
							}
						}

						std::for_each(current, current + currentChannelCount,
							[&](const decodeType& itrValue)
							{
								uint8_t offset = &itrValue - current;

								if (maxDecodeValues[offset] > itrValue)
									maxDecodeValues[offset] = itrValue;

								if (minDecodeValues[offset] < itrValue)
									minDecodeValues[offset] = itrValue;
							}
						);
					};

					{
						for (auto& z = localCoord[2] = 0u; z < extent.z; ++z)
							for (auto& y = localCoord[1] = 0u; y < extent.y; ++y)
								for (auto& x = localCoord[0] = 0u; x < extent.x; ++x)
									sum(core::vectorSIMDi32(x, y, z, w));
					}

					auto normalizeScratch = [&](bool isNormalizedFormat, bool isSignedFormat)
					{
						auto normalizeAsUsual = [&](decodeType* entryScratchAdress)
						{
							for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
								entryScratchAdress[channel] /= maxDecodeValues[channel];
						};

						auto normalizeAsSigned = [&](decodeType* entryScratchAdress)
						{
							for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
								entryScratchAdress[channel] = (2.0 * entryScratchAdress[channel] - maxDecodeValues[channel] - minDecodeValues[channel]) / (maxDecodeValues[channel] - minDecodeValues[channel]);
						};

						auto normalizeAsUnsigned = [&](decodeType* entryScratchAdress)
						{
							for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
								entryScratchAdress[channel] = (entryScratchAdress[channel] - minDecodeValues[channel]) / (maxDecodeValues[channel] - minDecodeValues[channel]);
						};

						std::function<void(decodeType*)> normalize;
						{
							if (isNormalizedFormat)
								if (isSignedFormat)
									normalize = normalizeAsSigned;
								else
									normalize = normalizeAsUnsigned;
							else
								normalize = normalizeAsUsual;
						}

						core::vector3du32_SIMD localCoord;
							for (auto& z = localCoord[2] = 0u; z < extent.z; ++z)
								for (auto& y = localCoord[1] = 0u; y < extent.y; ++y)
									for (auto& x = localCoord[0] = 0u; x < extent.x; ++x)
									{
										const auto outDataAdressOffsetScratch = ((localCoord.z * state->extent.height + localCoord.y) * state->extent.width + localCoord.x) * currentChannelCount;
										auto* entryScratchAdress = scratchMemory + outDataAdressOffsetScratch;

										normalize(entryScratchAdress);
									}
					};
					
					bool normalized = asset::isNormalizedFormat(inFormat);
					if (state->normalizeImageByTotalSATValues || normalized)
						normalizeScratch(normalized, asset::isSignedFormat(inFormat));

					return commonExecute(state, perOutputRegionEncode);
				}
			}

			resetState();
		}

};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
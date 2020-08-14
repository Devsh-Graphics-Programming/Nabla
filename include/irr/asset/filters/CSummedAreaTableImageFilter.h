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

			if (state->scratchMemoryByteSize < state_type::getRequiredScratchByteSize(state->inImage, state->extent))
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

			const core::vector3du32_SIMD scratchByteStrides = [&]()
			{
				const core::vectorSIMDu32 trueExtent = state->extentLayerCount;

				switch (currentChannelCount)
				{
					case 1:
					{
						return TexelBlockInfo(asset::E_FORMAT::EF_R64_SFLOAT).convert3DTexelStridesTo1DByteStrides(trueExtent);
					}

					case 2:
					{
						return TexelBlockInfo(asset::E_FORMAT::EF_R64G64_SFLOAT).convert3DTexelStridesTo1DByteStrides(trueExtent);
					}

					case 3:
					{
						return TexelBlockInfo(asset::E_FORMAT::EF_R64G64B64_SFLOAT).convert3DTexelStridesTo1DByteStrides(trueExtent);
					}
					case 4:
					{
						return TexelBlockInfo(asset::E_FORMAT::EF_R64G64B64A64_SFLOAT).convert3DTexelStridesTo1DByteStrides(trueExtent);
					}
				}
			}();
			const auto scratchTexelByteSize = scratchByteStrides[0];

			const auto&& [copyInBaseLayer, copyOutBaseLayer, copyLayerCount] = std::make_tuple(state->inBaseLayer, state->outBaseLayer, state->layerCount);
			state->layerCount = 1u;

			auto resetState = [&]()
			{
				state->inBaseLayer = copyInBaseLayer;
				state->outBaseLayer = copyOutBaseLayer;
				state->layerCount = copyLayerCount;
			};

			for (uint16_t w = 0u; w < copyLayerCount - copyInBaseLayer; ++w) 
			{
				std::array<decodeType, maxChannels> minDecodeValues = {};
				std::array<decodeType, maxChannels> maxDecodeValues = {};

				{
					const uint8_t* inData = reinterpret_cast<const uint8_t*>(state->inImage->getBuffer()->getPointer());
					const auto blockDims = asset::getBlockDimensions(state->inImage->getCreationParameters().format);
					constexpr uint8_t maxPlanes = 4;
					
					auto decode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						auto newReadBlockPos = (readBlockPos - decltype(readBlockPos)(state->inOffset.x, state->inOffset.y, state->inOffset.z)); 
						core::vectorSIMDu32 localOutPos = newReadBlockPos * blockDims;

						auto* inDataAdress = inData + readBlockArrayOffset;
						const void* inSourcePixels[maxPlanes] = { inDataAdress, nullptr, nullptr, nullptr };

						if constexpr (ExclusiveMode)
						{
							auto movedLocalOutPos = localOutPos + core::vectorSIMDu32(1, 1, 1);
							const auto isSatMemorySafe = (localOutPos < core::vectorSIMDu32(state->extent.width, state->extent.height, state->extent.depth));
							const auto shouldItResetSatMemoryToZero = (localOutPos < core::vectorSIMDu32(1, 1, 1));

							if (isSatMemorySafe.all())
							{
								for (auto blockY = 0u; blockY < blockDims.y; blockY++)
									for (auto blockX = 0u; blockX < blockDims.x; blockX++)
									{
										decodeType decodeBuffer[maxChannels] = {};

										if (shouldItResetSatMemoryToZero)
										{
											const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x + blockX, localOutPos.y + blockY, localOutPos.z), scratchByteStrides);
											memcpy(reinterpret_cast<uint8_t*>(scratchMemory) + offset, decodeBuffer, scratchTexelByteSize);
										}
										
										asset::decodePixelsRuntime(inFormat, inSourcePixels, decodeBuffer, blockX, blockY);
										const size_t movedOffset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(movedLocalOutPos.x + blockX, movedLocalOutPos.y + blockY, movedLocalOutPos.z), scratchByteStrides);
										memcpy(reinterpret_cast<uint8_t*>(scratchMemory) + movedOffset, decodeBuffer, scratchTexelByteSize);
									}
							}
						}
						else
						{
							for (auto blockY = 0u; blockY < blockDims.y; blockY++)
								for (auto blockX = 0u; blockX < blockDims.x; blockX++)
								{
									decodeType decodeBuffer[maxChannels] = {};

									asset::decodePixelsRuntime(inFormat, inSourcePixels, decodeBuffer, blockX, blockY);
									const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x + blockX, localOutPos.y + blockY, localOutPos.z), scratchByteStrides);
									memcpy(reinterpret_cast<uint8_t*>(scratchMemory) + offset, decodeBuffer, scratchTexelByteSize);
								}
						}
					};

					auto& inRegions = state->inImage->getRegions(state->inMipLevel);
					CBasicImageFilterCommon::executePerRegion(state->inImage, decode, inRegions.begin(), inRegions.end());
				}

				{
					auto getScratchPixel = [&](core::vector4di32_SIMD readBlockPos) -> decodeType*
					{
						const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(readBlockPos.x, readBlockPos.y, readBlockPos.z, 0), scratchByteStrides);
						return reinterpret_cast<decodeType*>(reinterpret_cast<uint8_t*>(scratchMemory) + scratchOffset);
					};

					auto sum = [&](core::vectorSIMDi32 readBlockPos) -> void
					{
						decodeType* current = getScratchPixel(readBlockPos);

						auto addScratchPixelToCurrentOne = [&](const decodeType* values)
						{
							for (auto i = 0; i < currentChannelCount; ++i)
								current[i] += values[i];
						};
						
						auto substractScratchPixelFromCurrentOne = [&](const decodeType* values)
						{
							for (auto i = 0; i < currentChannelCount; ++i)
								current[i] -= values[i];
						};

						const auto axisSafe = readBlockPos > core::vectorSIMDi32(0, 0, 0, 0);

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
						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
									sum(core::vectorSIMDu32(x, y, z));
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
							for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z)
								for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
									for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
									{
										const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(localCoord, scratchByteStrides);
										decodeType* entryScratchAdress = reinterpret_cast<decodeType*>(reinterpret_cast<uint8_t*>(scratchMemory) + scratchOffset);

										normalize(entryScratchAdress);
									}
					};
					
					bool normalized = asset::isNormalizedFormat(inFormat);
					if (state->normalizeImageByTotalSATValues || normalized)
						normalizeScratch(normalized, asset::isSignedFormat(inFormat));

					{
						uint8_t* outData = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());
						const auto blockDims = asset::getBlockDimensions(outFormat);

						auto encode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
						{
							auto newReadBlockPos = (readBlockPos - decltype(readBlockPos)(state->outOffset.x, state->outOffset.y, state->outOffset.z));

							auto localOutPos = newReadBlockPos * blockDims;
							uint8_t* outDataAdress = outData + readBlockArrayOffset;

							const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x, localOutPos.y, localOutPos.z, 0), scratchByteStrides);
							decodeType* entryScratchAdress = reinterpret_cast<decodeType*>(reinterpret_cast<uint8_t*>(scratchMemory) + scratchOffset);

							for (auto blockY = 0u; blockY < blockDims.y; blockY++)
								for (auto blockX = 0u; blockX < blockDims.x; blockX++)
								{
									/*
										We haven't supported BC encoding yet,
										the loop will perform one time
									*/

									const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x + blockX, localOutPos.y + blockY, localOutPos.z), scratchByteStrides);
									asset::encodePixelsRuntime(outFormat, outDataAdress, reinterpret_cast<uint8_t*>(entryScratchAdress) + offset); // overrrides texels, so region-overlapping case is fine
								}
						};

						auto& outRegions = state->outImage->getRegions(state->outMipLevel);
						CBasicImageFilterCommon::executePerRegion(state->outImage, encode, outRegions.begin(), outRegions.end());
					}
				}

				++state->inBaseLayer;
				++state->outBaseLayer;
			}

			resetState();
			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <type_traits>
#include <functional>

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "CConvertFormatImageFilter.h"

namespace nbl
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
				uint8_t axesToSum = 0u;														//!< which axes you want to sum; X: bit0, Y: bit1, Z: bit2

				static inline size_t getRequiredScratchByteSize(const ICPUImage* inputImage, asset::VkExtent3D extent)
				{
					const auto& inputCreationParams = inputImage->getCreationParameters();
					const auto channels = asset::getFormatChannelCount(inputCreationParams.format);

					size_t retval = extent.width * extent.height * extent.depth * channels * decodeTypeByteSize;
					
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

			if (asset::getFormatClass(inFormat) > asset::getFormatClass(outFormat)) // TODO in future! Change to a function checking a bit-depth for an each single channel
				return false;

			return true;
		}

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			auto checkFormat = state->inImage->getCreationParameters().format;
			if (isIntegerFormat(checkFormat))
				return executeInterprated(std::forward<ExecutionPolicy>(policy), state, reinterpret_cast<uint64_t*>(state->scratchMemory));
			else
				return executeInterprated(std::forward<ExecutionPolicy>(policy), state, reinterpret_cast<double*>(state->scratchMemory));
		}	
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}

	private:

		template<class ExecutionPolicy, typename decodeType> //!< double or uint64_t
		static inline bool executeInterprated(ExecutionPolicy&& policy, state_type* state, decodeType* scratchMemory)
		{
			const asset::E_FORMAT inFormat = state->inImage->getCreationParameters().format;
			const asset::E_FORMAT outFormat = state->outImage->getCreationParameters().format;
			const auto inTexelByteSize = asset::getTexelOrBlockBytesize(inFormat);
			const auto outTexelByteSize = asset::getTexelOrBlockBytesize(outFormat);
			const auto currentChannelCount = asset::getFormatChannelCount(inFormat);
			const auto arrayLayers = state->inImage->getCreationParameters().arrayLayers;
			static constexpr auto maxChannels = 4u;

			#ifdef _NBL_DEBUG
			memset(scratchMemory, 0, state->scratchMemoryByteSize);
			#endif // _NBL_DEBUG

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

			auto resetState = [&, copyInBaseLayer=copyInBaseLayer, copyOutBaseLayer=copyOutBaseLayer, copyLayerCount=copyLayerCount]()
			{
				state->inBaseLayer = copyInBaseLayer;
				state->outBaseLayer = copyOutBaseLayer;
				state->layerCount = copyLayerCount;
			};

			for (uint16_t w = 0u; w < copyLayerCount; ++w) // this could be parallelized
			{
				std::array<decodeType, maxChannels> minDecodeValues = {};
				std::array<decodeType, maxChannels> maxDecodeValues = {};

				{
					const uint8_t* inData = reinterpret_cast<const uint8_t*>(state->inImage->getBuffer()->getPointer());
					const auto blockDims = asset::getBlockDimensions(state->inImage->getCreationParameters().format);
					static constexpr uint8_t maxPlanes = 4;

					/*
						Make sure we are able to move as (+ 1) in a certain plane,
						otherwise memory leaks may occur
					*/

					bool is2DAndBelow = state->inImage->getCreationParameters().type == IImage::ET_2D;
					bool is3DAndBelow = state->inImage->getCreationParameters().type == IImage::ET_3D;
					const core::vectorSIMDu32 limit(1, is2DAndBelow, is3DAndBelow);
					const core::vectorSIMDu32 movingExclusiveVector = limit, movingOnYZorXZorXYCheckingVector = limit;

					auto decode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						core::vectorSIMDu32 localOutPos = readBlockPos * blockDims - core::vectorSIMDu32(state->inOffset.x, state->inOffset.y, state->inOffset.z);

						auto* inDataAdress = inData + readBlockArrayOffset;
						const void* inSourcePixels[maxPlanes] = { inDataAdress, nullptr, nullptr, nullptr };

						if constexpr (ExclusiveMode)
						{
							auto movedLocalOutPos = localOutPos + movingExclusiveVector;
							const auto isSatMemorySafe = (movedLocalOutPos < core::vectorSIMDu32(state->extent.width, state->extent.height, state->extent.depth, movedLocalOutPos.w + 1)); // force true on .w

							if (isSatMemorySafe.all())
							{
								decodeType decodeBuffer[maxChannels] = {};

								for (auto blockY = 0u; blockY < blockDims.y; blockY++)
									for (auto blockX = 0u; blockX < blockDims.x; blockX++)
									{
										asset::decodePixelsRuntime(inFormat, inSourcePixels, decodeBuffer, blockX, blockY);
										const size_t movedOffset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(movedLocalOutPos.x + blockX, movedLocalOutPos.y + blockY, movedLocalOutPos.z), scratchByteStrides);
										memcpy(reinterpret_cast<uint8_t*>(scratchMemory) + movedOffset, decodeBuffer, scratchTexelByteSize);
									}
							}
						}
						else
						{
							decodeType decodeBuffer[maxChannels] = {};
							for (auto blockY = 0u; blockY < blockDims.y; blockY++)
								for (auto blockX = 0u; blockX < blockDims.x; blockX++)
								{
									asset::decodePixelsRuntime(inFormat, inSourcePixels, decodeBuffer, blockX, blockY);
									const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x + blockX, localOutPos.y + blockY, localOutPos.z), scratchByteStrides);
									memcpy(reinterpret_cast<uint8_t*>(scratchMemory) + offset, decodeBuffer, scratchTexelByteSize);
								}
						}
					};

					IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u), state->inMipLevel, state->inBaseLayer, 1 };
					CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { state->inOffset,state->extent };
					CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, inFormat);

					auto& inRegions = state->inImage->getRegions(state->inMipLevel);
					CBasicImageFilterCommon::executePerRegion(policy, state->inImage, decode, inRegions.begin(), inRegions.end(), clipFunctor);

					if constexpr (ExclusiveMode)
					{
						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z) // TODO: parallelize
							for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
								{
									auto resetSATMemory = [&](const core::vector3du32_SIMD& position)
									{
										const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(position, scratchByteStrides);
										memset(reinterpret_cast<uint8_t*>(scratchMemory) + offset, 0, scratchTexelByteSize);
									};

									const auto doesItMoveOnYZorXZorXY = (localCoord < movingOnYZorXZorXYCheckingVector);
									if (doesItMoveOnYZorXZorXY.any())
										resetSATMemory(localCoord);
								}
					}
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

						auto areAxisSafe = [&]()
						{
							const auto position = core::vectorSIMDi32(0u, 0u, 0u, 0u);

							const bool shouldSumX = (state->axesToSum >> 0) & 0x1u;
							const bool shouldSumY = (state->axesToSum >> 1) & 0x1u;
							const bool shouldSumZ = (state->axesToSum >> 2) & 0x1u;

							return core::vector4du32_SIMD
							(
								(readBlockPos.x > position.x) && (shouldSumX),
								(readBlockPos.y > position.y) && (shouldSumY),
								(readBlockPos.z > position.z) && (shouldSumZ),
								(readBlockPos.w > position.w)
							);
						};

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

								if (maxDecodeValues[offset] < itrValue)
									maxDecodeValues[offset] = itrValue;

								if (minDecodeValues[offset] > itrValue)
									minDecodeValues[offset] = itrValue;
							}
						);
					};

					{
						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z)  // TODO: parallelize (will be tough!)
							for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
									sum(core::vectorSIMDu32(x, y, z));
					}

					auto normalizeScratch = [&](bool isSignedFormat)
					{
						core::vector3du32_SIMD localCoord;
							for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z) // TODO: parallelize
								for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
									for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
									{
										const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(localCoord, scratchByteStrides);
										decodeType* entryScratchAdress = reinterpret_cast<decodeType*>(reinterpret_cast<uint8_t*>(scratchMemory) + scratchOffset);

										if(isSignedFormat)
											for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
												entryScratchAdress[channel] = (2.0 * entryScratchAdress[channel] - maxDecodeValues[channel] - minDecodeValues[channel]) / (maxDecodeValues[channel] - minDecodeValues[channel]);
										else
											for (uint8_t channel = 0; channel < currentChannelCount; ++channel)
												entryScratchAdress[channel] = (entryScratchAdress[channel] - minDecodeValues[channel]) / (maxDecodeValues[channel] - minDecodeValues[channel]);
									}
					};

					bool normalized = asset::isNormalizedFormat(inFormat);
					if (state->normalizeImageByTotalSATValues || normalized)
						normalizeScratch(asset::isSignedFormat(inFormat));

					{
						uint8_t* outData = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());

						auto encode = [&](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
						{
							// encoding format cannot be block compressed so in this case block==texel
							auto localOutPos = readBlockPos - core::vectorSIMDu32(state->outOffset.x, state->outOffset.y, state->outOffset.z, readBlockPos.w); // force 0 on .w compoment to obtain valid offset
							uint8_t* outDataAdress = outData + writeBlockArrayOffset;

							const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(localOutPos, scratchByteStrides);
							asset::encodePixelsRuntime(outFormat, outDataAdress, reinterpret_cast<uint8_t*>(scratchMemory) + offset); // overrrides texels, so region-overlapping case is fine
						};

						IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u), state->outMipLevel, state->outBaseLayer, 1 };
						CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { state->outOffset,state->extent };
						CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, outFormat);

						auto& outRegions = state->outImage->getRegions(state->outMipLevel);
						CBasicImageFilterCommon::executePerRegion(policy,state->outImage, encode, outRegions.begin(), outRegions.end(), clipFunctor);
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
} // end namespace nbl

#endif
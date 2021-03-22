// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED__

#include "nbl/core/core.h"

#include <type_traits>
#include <functional>
 
#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "CConvertFormatImageFilter.h"

namespace nbl
{
namespace asset
{

class CNormalMapToDerivativeFilterBase
{
	public:
		class CNormalMapToDerivativeStateBase
		{
			public:

				static inline constexpr size_t decodeTypeByteSize = sizeof(double);
				static inline constexpr size_t forcedScratchChannelAmount = 4;
				uint8_t*	scratchMemory = nullptr;										//!< memory covering all regions used for temporary filling within computation of sum values
				size_t	scratchMemoryByteSize = {};											//!< required byte size for entire scratch memory

				static inline size_t getRequiredScratchByteSize(asset::VkExtent3D extent)
				{
					size_t retval = extent.width * extent.height * extent.depth * decodeTypeByteSize * forcedScratchChannelAmount;
					
					return retval;
				}

				/*
					Layer ID is relative to outBaseLayer in state
				*/

				const std::array<double, forcedScratchChannelAmount>& getAbsoluteLayerScaleValue(size_t layer)
				{
					if (!maxAbsLayerScaleValues.empty())
						return maxAbsLayerScaleValues[layer];
					else
						return {};
				}

			protected:
				std::vector<std::array<double, forcedScratchChannelAmount>> maxAbsLayerScaleValues; 								//!< scales gained by the filter (each layer handled) for derivative map shader usage
		};

	protected:
		CNormalMapToDerivativeFilterBase() {}
		virtual ~CNormalMapToDerivativeFilterBase() {}

		static inline bool validate(CNormalMapToDerivativeStateBase* state)
		{
			if (!state)
				return false;

			return true;
		}
};

//! Convert Normal Map to Derivative Normal Map
/*
	
*/

class CNormalMapToDerivativeFilter : public CMatchedSizeInOutImageFilterCommon, public CNormalMapToDerivativeFilterBase
{
	public:
		virtual ~CNormalMapToDerivativeFilter() {}

		class CStateBase : public CMatchedSizeInOutImageFilterCommon::state_type, public CNormalMapToDerivativeFilterBase::CNormalMapToDerivativeStateBase
		{ 
			public:
				CStateBase() = default;
				virtual ~CStateBase() = default;

			private:

				void resetLayerScaleValues()
				{
					maxAbsLayerScaleValues.clear();
				}

				friend class CNormalMapToDerivativeFilter;
		};
		using state_type = CStateBase; //!< full combined state

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			if (!CNormalMapToDerivativeFilterBase::validate(state))
				return false;
			
			const ICPUImage::SCreationParams& inParams = state->inImage->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = state->outImage->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;

			if (outFormat != asset::EF_R8G8_SNORM)
				return false;

			if (state->scratchMemoryByteSize < state_type::getRequiredScratchByteSize(state->extent))
				return false;

			if (asset::getFormatChannelCount(inFormat) < 3 )
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			state->resetLayerScaleValues();

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
			const auto inTexelByteSize = asset::getTexelOrBlockBytesize(inFormat);
			const auto currentChannelCount = asset::getFormatChannelCount(inFormat);
			const auto arrayLayers = state->inImage->getCreationParameters().arrayLayers;
			static constexpr auto maxChannels = 4u;

			#ifdef _NBL_DEBUG
			memset(scratchMemory, 0, state->scratchMemoryByteSize);
			#endif // _NBL_DEBUG

			const core::vector3du32_SIMD scratchByteStrides = TexelBlockInfo(asset::E_FORMAT::EF_R64G64B64A64_SFLOAT).convert3DTexelStridesTo1DByteStrides(state->extentLayerCount);
			const auto scratchTexelByteSize = scratchByteStrides[0];

			// I wonder if we should let somebody pass through more than 1 layer, though I find it cool

			const auto&& [copyInBaseLayer, copyOutBaseLayer, copyLayerCount] = std::make_tuple(state->inBaseLayer, state->outBaseLayer, state->layerCount);
			state->layerCount = 1u;
			 
			auto resetState = [&]()
			{
				state->inBaseLayer = copyInBaseLayer;
				state->outBaseLayer = copyOutBaseLayer;
				state->layerCount = copyLayerCount;
			};

			for (uint16_t w = 0u; w < copyLayerCount; ++w)
			{
				std::array<decodeType, maxChannels> maxAbsoluteDecodeValues = {};

				{
					const uint8_t* inData = reinterpret_cast<const uint8_t*>(state->inImage->getBuffer()->getPointer());
					const auto blockDims = asset::getBlockDimensions(state->inImage->getCreationParameters().format);
					static constexpr uint8_t maxPlanes = 4;

					auto decode = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						core::vectorSIMDu32 localOutPos = readBlockPos * blockDims - core::vectorSIMDu32(state->inOffset.x, state->inOffset.y, state->inOffset.z);

						auto* inDataAdress = inData + readBlockArrayOffset;
						const void* inSourcePixels[maxPlanes] = { inDataAdress, nullptr, nullptr, nullptr };

						decodeType decodeBuffer[maxChannels] = {};
						for (auto blockY = 0u; blockY < blockDims.y; blockY++)
							for (auto blockX = 0u; blockX < blockDims.x; blockX++)
							{
								asset::decodePixelsRuntime(inFormat, inSourcePixels, decodeBuffer, blockX, blockY);
								const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x + blockX, localOutPos.y + blockY, localOutPos.z), scratchByteStrides);
								memcpy(reinterpret_cast<uint8_t*>(scratchMemory) + offset, decodeBuffer, scratchTexelByteSize);
							}
					};

					IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u), state->inMipLevel, state->inBaseLayer, 1 };
					CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { state->inOffset,state->extent };
					CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, inFormat);

					auto& inRegions = state->inImage->getRegions(state->inMipLevel);
					CBasicImageFilterCommon::executePerRegion(state->inImage, decode, inRegions.begin(), inRegions.end(), clipFunctor);
				}

				{
					auto getScratchPixel = [&](core::vector4di32_SIMD readBlockPos) -> decodeType*
					{
						const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(readBlockPos.x, readBlockPos.y, readBlockPos.z, 0), scratchByteStrides);
						return reinterpret_cast<decodeType*>(reinterpret_cast<uint8_t*>(scratchMemory) + scratchOffset);
					};

					auto computeDerivativeTexel = [&](core::vectorSIMDi32 readBlockPos) -> void
					{
						decodeType* current = getScratchPixel(readBlockPos);
						auto& [x, y, z, a] = std::make_tuple(*current, *(current + 1), *(current + 2), *(current + 3));

						std::for_each(current, current + currentChannelCount,
							[&](const decodeType& itrValue)
							{
								uint8_t offset = &itrValue - current;
								const decodeType absoluteValue = core::abs(itrValue);
							
								if (maxAbsoluteDecodeValues[offset] < absoluteValue)
									maxAbsoluteDecodeValues[offset] = absoluteValue;
							}
						);

						x = -x / z;
						y = -y / z;
					};

					{
						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
									computeDerivativeTexel(core::vectorSIMDu32(x, y, z));
					}

					auto& maxAbsLayerScaleValues = state->maxAbsLayerScaleValues.emplace_back();
					for (auto& absLayerScaleValue : maxAbsLayerScaleValues)
						absLayerScaleValue = maxAbsoluteDecodeValues[&absLayerScaleValue - &maxAbsLayerScaleValues[0]];

					// what about normalize, should it be done like SAT ?

					{
						uint8_t* outData = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());

						auto encode = [&](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
						{
							// encoding format cannot be block compressed so in this case block==texel
							auto localOutPos = readBlockPos - core::vectorSIMDu32(state->outOffset.x, state->outOffset.y, state->outOffset.z, readBlockPos.w); // force 0 on .w compoment to obtain valid offset
							uint8_t* outDataAdress = outData + writeBlockArrayOffset;

							const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(localOutPos, scratchByteStrides);
							auto* data = reinterpret_cast<uint8_t*>(scratchMemory) + offset;
							asset::encodePixels<asset::EF_R8G8_SNORM, double>(outDataAdress, reinterpret_cast<double*>(data)); // overrrides texels, so region-overlapping case is fine
						};

						IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u), state->outMipLevel, state->outBaseLayer, 1 };
						CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { state->outOffset,state->extent };
						CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, asset::EF_R8G8_SNORM);

						auto& outRegions = state->outImage->getRegions(state->outMipLevel);
						CBasicImageFilterCommon::executePerRegion(state->outImage, encode, outRegions.begin(), outRegions.end(), clipFunctor);
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

#endif // __NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED__
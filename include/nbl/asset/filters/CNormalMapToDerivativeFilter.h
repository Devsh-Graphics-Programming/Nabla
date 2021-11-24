// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_NORMAL_MAP_TO_DERIVATIVE_FILTER_H_INCLUDED__

#include <type_traits>
#include <functional>
 
#include "nbl/core/pch_core.h"
#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/asset/filters/CConvertFormatImageFilter.h"

namespace nbl
{
namespace asset
{

template<typename Swizzle, typename Dither>
class CNormalMapToDerivativeFilterBase : public impl::CSwizzleableAndDitherableFilterBase<false, false, Swizzle, Dither>
{
	public:
		class Cthis : public impl::CSwizzleableAndDitherableFilterBase<false, false, Swizzle, Dither>::state_type
		{
			public:
				using decodeType = float;
				static inline constexpr size_t decodeTypeByteSize = sizeof(float);
				static inline constexpr size_t forcedScratchChannelAmount = 2;
				uint8_t*	scratchMemory = nullptr;										//!< memory covering all regions used for temporary filling within computation of sum values
				size_t	scratchMemoryByteSize = {};											//!< required byte size for entire scratch memory
				bool normalizeImageByTotalABSValues = true;									//!< force normalizing by maximum absolute values

				/*
					layerCount - layer count used to execute the filter, not global layer count!
					extent - extent of input image at chosen mip map level
				*/

				static inline size_t getRequiredScratchByteSize(size_t layerCount, asset::VkExtent3D extent)
				{
					size_t retval = extent.width * extent.height * extent.depth * decodeTypeByteSize * forcedScratchChannelAmount + (layerCount * decodeTypeByteSize * forcedScratchChannelAmount);
					
					return retval;
				}

				/*
					Layer ID is relative to outBaseLayer in state
				*/

				const core::vectorSIMDf getAbsoluteLayerScaleValues(size_t layer)
				{
					if (!scaleValuesPointer)
					{
						auto offset = layer * forcedScratchChannelAmount;
						auto* data = scaleValuesPointer + offset;
						return core::vectorSIMDf (*data, *(data + 1));
					}
					else
						return 0; // or maybe assert?
				}

			protected:

				float* scaleValuesPointer = nullptr;

				std::vector<std::array<double, forcedScratchChannelAmount>> maxAbsLayerScaleValues; 								//!< scales gained by the filter (each layer handled) for derivative map shader usage
		};

	protected:
		CNormalMapToDerivativeFilterBase() {}
		virtual ~CNormalMapToDerivativeFilterBase() {}

		static inline bool validate(Cthis* state)
		{
			if (!state)
				return false;
			
			if (!state->scratchMemory)
				return false;

			if (state->scratchMemoryByteSize == 0)
				return false;

			if (!impl::CSwizzleableAndDitherableFilterBase<false, false, Swizzle, Dither>::validate(state))
				return false;

			return true;
		}
};

//! Convert Normal Map to Derivative Normal Map
/*
	
*/

template<typename Swizzle = DefaultSwizzle, typename Dither = IdentityDither>
class CNormalMapToDerivativeFilter : public CMatchedSizeInOutImageFilterCommon, public CNormalMapToDerivativeFilterBase<Swizzle, Dither>
{
	public:
		virtual ~CNormalMapToDerivativeFilter() {}

		class CStateBase : public CMatchedSizeInOutImageFilterCommon::state_type, public CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis
		{ 
			public:
				CStateBase() = default;
				virtual ~CStateBase() = default;

			private:

				void setLayerScaleValuesOffset()
				{
					this->scaleValuesPointer = reinterpret_cast<float*>(this->scratchMemory) + (extent.width * extent.height * extent.depth * this->forcedScratchChannelAmount);
				}

				void resetLayerScaleValues()
				{
					memset(const_cast<float*>(this->scaleValuesPointer), 0, layerCount * this->forcedScratchChannelAmount * this->decodeTypeByteSize);
				}

				friend class CNormalMapToDerivativeFilter;
		};
		using state_type = CStateBase; //!< full combined state

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			if (!CNormalMapToDerivativeFilterBase<Swizzle, Dither>::validate(state))
				return false;
			
			const ICPUImage::SCreationParams& inParams = state->inImage->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = state->outImage->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;

			if (state->scratchMemoryByteSize < state_type::getRequiredScratchByteSize(state->layerCount, state->extent))
				return false;

			if (asset::getFormatChannelCount(inFormat) < 3 && asset::getFormatChannelCount(outFormat) != 2)
				return false;

			if (asset::isIntegerFormat(inFormat) || asset::isIntegerFormat(outFormat))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (asset::isBlockCompressionFormat(outFormat))
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			state->setLayerScaleValuesOffset();
			state->resetLayerScaleValues();
			
			const asset::E_FORMAT inFormat = state->inImage->getCreationParameters().format;
			const asset::E_FORMAT outFormat = state->outImage->getCreationParameters().format;
			const auto inTexelByteSize = asset::getTexelOrBlockBytesize(inFormat);
			const auto outTexelByteSize = asset::getTexelOrBlockBytesize(outFormat);
			const auto currentChannelCount = asset::getFormatChannelCount(inFormat);
			const auto arrayLayers = state->inImage->getCreationParameters().arrayLayers;
			static constexpr auto maxChannels = 4u;

			#ifdef _NBL_DEBUG
			memset(state->scratchMemory, 0, state->scratchMemoryByteSize);
			#endif // _NBL_DEBUG

			const core::vector3du32_SIMD scratchByteStrides = TexelBlockInfo(asset::E_FORMAT::EF_R32G32_SFLOAT).convert3DTexelStridesTo1DByteStrides(state->extentLayerCount);
			const auto scratchTexelByteSize = scratchByteStrides[0];

			auto copyInBaseLayer = state->inBaseLayer, copyOutBaseLayer = state->outBaseLayer, copyLayerCount = state->layerCount;
			state->layerCount = 1u;

			auto resetState = [&]()
			{
				state->inBaseLayer = copyInBaseLayer;
				state->outBaseLayer = copyOutBaseLayer;
				state->layerCount = copyLayerCount;
			};

			for (uint16_t w = 0u; w < copyLayerCount; ++w)
			{
				float* decodeAbsValuesOffset = state->scaleValuesPointer + (w * CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::forcedScratchChannelAmount);

				auto& xMaxDecodeAbsValue = *decodeAbsValuesOffset;
				auto& yMaxDecodeAbsValue = *(decodeAbsValuesOffset + 1);
				{
					const uint8_t* inData = reinterpret_cast<const uint8_t*>(state->inImage->getBuffer()->getPointer());
					const auto blockDims = asset::getBlockDimensions(state->inImage->getCreationParameters().format);
					static constexpr uint8_t maxPlanes = 4;

					auto decodeAndDivide = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
					{
						core::vectorSIMDu32 localOutPos = readBlockPos * blockDims - core::vectorSIMDu32(state->inOffset.x, state->inOffset.y, state->inOffset.z);

						auto* inDataAdress = inData + readBlockArrayOffset;
						const void* inSourcePixels[maxPlanes] = { inDataAdress, nullptr, nullptr, nullptr };

						double decodeBuffer[maxChannels] = {}; // ASCT TODO?
						double swizzledBuffer[maxChannels] = {}; // ASCT TODO?

						for (auto blockY = 0u; blockY < blockDims.y; blockY++)
							for (auto blockX = 0u; blockX < blockDims.x; blockX++)
							{
								impl::CSwizzleableAndDitherableFilterBase<false, false, Swizzle, IdentityDither>::onDecode(inFormat, state, inSourcePixels, decodeBuffer, swizzledBuffer, blockX, blockY);

								const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(localOutPos.x + blockX, localOutPos.y + blockY, localOutPos.z), scratchByteStrides);
								float* data = reinterpret_cast<float*>(state->scratchMemory + offset);

								const auto [xDecode, yDecode, zDecode] = std::make_tuple(*swizzledBuffer, *(swizzledBuffer + 1), *(swizzledBuffer + 2));

								const auto& newComputedX = *data = -xDecode / zDecode;
								const auto& newComputedY = *(data + 1) = -yDecode / zDecode;

								auto absoluteX = core::abs(newComputedX);
								auto absoluteY = core::abs(newComputedY);

								if (xMaxDecodeAbsValue < absoluteX)
									xMaxDecodeAbsValue = absoluteX;

								if (yMaxDecodeAbsValue < absoluteY)
									yMaxDecodeAbsValue = absoluteY;
							}
					};

					IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u), state->inMipLevel, state->inBaseLayer, 1 };
					CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { state->inOffset,state->extent };
					CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, inFormat);

					const auto& inRegions = state->inImage->getRegions(state->inMipLevel);
					CBasicImageFilterCommon::executePerRegion(state->inImage, decodeAndDivide, inRegions.begin(), inRegions.end(), clipFunctor);
				}

				{
					auto getScratchPixel = [&](core::vector4di32_SIMD readBlockPos) -> typename CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::decodeType*
					{
						const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(core::vector3du32_SIMD(readBlockPos.x, readBlockPos.y, readBlockPos.z, 0), scratchByteStrides); // TODO
						return reinterpret_cast<typename CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::decodeType*>(reinterpret_cast<uint8_t*>(state->scratchMemory) + scratchOffset);
					};

					auto normalizeScratch = [&](bool isSigned)
					{
						core::vector3du32_SIMD localCoord;
						for (auto& z = localCoord[2] = 0u; z < state->extent.depth; ++z)
							for (auto& y = localCoord[1] = 0u; y < state->extent.height; ++y)
								for (auto& x = localCoord[0] = 0u; x < state->extent.width; ++x)
								{
									const size_t scratchOffset = asset::IImage::SBufferCopy::getLocalByteOffset(localCoord, scratchByteStrides);
									auto* entryScratchAdress = reinterpret_cast<typename CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::decodeType*>(reinterpret_cast<uint8_t*>(state->scratchMemory) + scratchOffset);

									if (isSigned)
										for (uint8_t channel = 0; channel < CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::forcedScratchChannelAmount; ++channel)
											entryScratchAdress[channel] = entryScratchAdress[channel] / decodeAbsValuesOffset[channel];
									else
										for (uint8_t channel = 0; channel < CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::forcedScratchChannelAmount; ++channel)
											entryScratchAdress[channel] = entryScratchAdress[channel] * 0.5f / decodeAbsValuesOffset[channel] + 0.5f;
								}
					};

					bool normalized = asset::isNormalizedFormat(outFormat);
					if (state->normalizeImageByTotalABSValues || normalized)
						normalizeScratch(asset::isSignedFormat(outFormat));

					{
						uint8_t* outData = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());

						auto encode = [&](uint32_t writeBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
						{
							// encoding format cannot be block compressed so in this case block==texel
							auto localOutPos = readBlockPos - core::vectorSIMDu32(state->outOffset.x, state->outOffset.y, state->outOffset.z, readBlockPos.w); // force 0 on .w compoment to obtain valid offset
							uint8_t* outDataAdress = outData + writeBlockArrayOffset;

							const size_t offset = asset::IImage::SBufferCopy::getLocalByteOffset(localOutPos, scratchByteStrides);
							auto* data = reinterpret_cast<uint8_t*>(state->scratchMemory) + offset;

							impl::CSwizzleAndConvertImageFilterBase<false, false, Swizzle, IdentityDither>::onEncode(outFormat, state, outDataAdress, data, localOutPos, 0, 0, CNormalMapToDerivativeFilterBase<Swizzle, Dither>::Cthis::forcedScratchChannelAmount); // overrrides texels, so region-overlapping case is fine
						};

						IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u), state->outMipLevel, state->outBaseLayer, 1 };
						CMatchedSizeInOutImageFilterCommon::state_type::TexelRange range = { state->outOffset,state->extent };
						CBasicImageFilterCommon::clip_region_functor_t clipFunctor(subresource, range, outFormat);

						const auto& outRegions = state->outImage->getRegions(state->outMipLevel);
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
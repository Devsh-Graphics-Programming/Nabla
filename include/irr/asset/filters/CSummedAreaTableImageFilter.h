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
			if (!CMatchedSizeInOutImageFilterCommon::validate(state));
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

			if (asset::getFormatClass(state->inImage->getCreationParameters().format) <= asset::getFormatClass(outFormat))
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			auto perOutputRegion = [&](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat) == getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something
				const auto blockDims = asset::getBlockDimensions(commonExecuteData.inFormat);
				const auto currentChannelCount = asset::getFormatChannelCount(commonExecuteData.inFormat);
				static constexpr auto maxChannels = 4;

				memset(state->scratchMemory, 0, state->scratchMemoryByteSize);

				auto decodeEntireImageToTemporaryScratchImage = [&](auto* scratchMemory) //!< double or uint64_t
				{
					auto* temporaryMemory = scratchMemory;

					using decodeType = typename std::decay<decltype(*scratchMemory)>::type;
					const auto* entryData = reinterpret_cast<const uint8_t*>(commonExecuteData.inImg->getBuffer()->getPointer()); // TODO need to use filters API to stay at layer-mipmap memory
					decodeType decodeBuffer[maxChannels] = {};

					core::vector3du32_SIMD localCoord;
						for (auto& zBlock = localCoord[2] = 0u; zBlock < blockDims.z; ++zBlock)
							for (auto& yBlock = localCoord[1] = 0u; yBlock < blockDims.y; ++yBlock)
								for (auto& xBlock = localCoord[0] = 0u; xBlock < blockDims.x; ++xBlock)
								{
									const size_t independentPtrOffset = ((zBlock * blockDims.y + yBlock) * blockDims.x + xBlock);
									auto* inDataAdress = entryData + independentPtrOffset * asset::getTexelOrBlockBytesize(commonExecuteData.inFormat);
									auto* outDataAdress = scratchMemory + independentPtrOffset * currentChannelCount;

									const void* sourcePixels[maxChannels] = { inDataAdress, nullptr, nullptr, nullptr };
									asset::decodePixelsRuntime(commonExecuteData.inImg->getCreationParameters().format, sourcePixels, decodeBuffer, 1, 1);
									memcpy(decodeBuffer, outDataAdress, sizeof(decodeType) * currentChannelCount);
								}
				};

				bool decodeAsDouble = false;
				if (isIntegerFormat(commonExecuteData.inFormat))
					decodeEntireImageToTemporaryScratchImage(reinterpret_cast<uint64_t*>(state->scratchMemory));
				else
				{
					decodeAsDouble = true;
					decodeEntireImageToTemporaryScratchImage(reinterpret_cast<double*>(state->scratchMemory));
				}

				constexpr auto decodeByteSize = 8;
				auto sum = [&](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto performTheProcess = [&](auto* scratchMemory) // double or uint64_t
					{
						using decodeType = typename std::decay<decltype(*scratchMemory)>::type;

						for (uint32_t z = 0; z < blockDims.z; z++)
						{
							decodeType* columnTotalCache = scratchMemory; // cache using first entire row for sum values in columns
				
							for (uint32_t y = blockDims.y - 1; y >= 0; y--)
							{
								decodeType rowtotal[4] = {};
								for (uint32_t x = 0; x < blockDims.x; x++)
									for (int c = 0; c < currentChannelCount; c++)
									{
										auto outDataAdressItself = ((z * blockDims.y + y) * blockDims.x + x) * currentChannelCount + c;
										auto outDataAdressX = ((z * blockDims.y + y) * blockDims.x + x - 1) * currentChannelCount + c;
										auto pixelColorX = *(scratchMemory + outDataAdressX);
										auto pixelColorItself = *(scratchMemory + outDataAdressItself);
										decltype(pixelColorX) pixelColorY;
										auto* columnTotal = columnTotalCache + ((z * blockDims.y) * blockDims.x + x) * currentChannelCount + c;

										if(x > 0)
											rowtotal[c] += pixelColorX + c;

										if (y > 0)
										{
											auto outDataAdressY = ((z * blockDims.y + y - 1) * blockDims.x + x) * currentChannelCount + c;
											pixelColorY = *(scratchMemory + outDataAdressY);
											*(columnTotal + c) += pixelColorY + c;
										}
					
										*(scratchMemory + outDataAdressItself) = columnTotal[c] + rowtotal[c] + (!ExclusiveMode ? pixelColorItself + c : 0);
									}
							}
						}
						
					};

					if (decodeAsDouble)
						performTheProcess(reinterpret_cast<double*>(state->scratchMemory));
					else
						performTheProcess(reinterpret_cast<uint64_t*>(state->scratchMemory));
					};

				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, sum, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);

				auto encodeTo = [&](auto* scratchMemory, ICPUImage* outImage)
				{
					core::vector3du32_SIMD localCoord;
					for (auto& zBlock = localCoord[2] = 0u; zBlock < blockDims.z; ++zBlock)
						for (auto& yBlock = localCoord[1] = 0u; yBlock < blockDims.y; ++yBlock)
							for (auto& xBlock = localCoord[0] = 0u; xBlock < blockDims.x; ++xBlock)
							{
								const size_t independentPtrOffset = ((zBlock * blockDims.y + yBlock) * blockDims.x + xBlock);
								auto* outScratchAdress = scratchMemory + ((zBlock * blockDims.y + yBlock) * blockDims.x + xBlock) * currentChannelCount;
								auto* inData = reinterpret_cast<uint8_t*>(outImage->getBuffer()->getPointer()) + independentPtrOffset * asset::getTexelOrBlockBytesize(commonExecuteData.outFormat); // TODO need to use filters API to stay at layer-mipmap memory
								asset::encodePixelsRuntime(commonExecuteData.outFormat, inData, outScratchAdress);
							}
				};
		
				if (decodeAsDouble)
					encodeTo(reinterpret_cast<double*>(state->scratchMemory), commonExecuteData.outImg);
				else
					encodeTo(reinterpret_cast<uint64_t*>(state->scratchMemory), commonExecuteData.outImg);

				return true;
			};

			return commonExecute(state, perOutputRegion);
		}

	private:

};

} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SUMMED_AREA_TABLE_IMAGE_FILTER_H_INCLUDED__
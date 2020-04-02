// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_MATCHED_SIZE_IN_OUT_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __IRR_C_MATCHED_SIZE_IN_OUT_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
{
namespace asset
{

class CMatchedSizeInOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				union
				{
					core::vectorSIMDu32 extentLayerCount;
					struct
					{
						VkExtent3D		extent;
						uint32_t		layerCount;
					};
				};
				union
				{
					core::vectorSIMDu32 inOffsetBaseLayer;
					struct
					{
						VkOffset3D		inOffset;
						uint32_t		inBaseLayer;
					};
				};
				union
				{
					core::vectorSIMDu32 outOffsetBaseLayer;
					struct
					{
						VkOffset3D		outOffset;
						uint32_t		outBaseLayer;
					};
				};
				uint32_t				inMipLevel = 0u;
				uint32_t				outMipLevel = 0u;
				ICPUImage*				inImage = nullptr;
				ICPUImage*				outImage = nullptr;
		};
		using state_type = CState;
		
		static inline bool validate(state_type* state)
		{
			if (!state)
				return nullptr;

			IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
			state_type::TexelRange range = {state->inOffset,state->extent};
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource,range,state->inImage))
				return false;
			subresource.mipLevel = state->outMipLevel;
			subresource.baseArrayLayer = state->outBaseLayer;
			range.offset = state->outOffset;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource,range,state->outImage))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
				return false;

			return true;
		}

	protected:
		virtual ~CMatchedSizeInOutImageFilterCommon() = 0;

		template<typename PerOutputFunctor>
		static inline bool commonExecute(state_type* state, PerOutputFunctor& perOutput)
		{
			if (!validate(state))
				return false;

			// I'm a lazy fuck and requiring that `PerOutputFunctor` be a generic lambda
			struct CommonExecuteData
			{
				auto* const outImg = state->outImage;
				auto* const inImg = state->inImage;
				const auto& inParams = inImg->getCreationParameters();
				const auto& outParams = outImg->getCreationParameters();
				const auto inFormat = inParams.format;
				const auto outFormat = outParams.format;
				const auto* const inData = reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer());
				auto* const outData = reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer());
				const auto inRegions = inImg->getRegions(state->inMipLevel);
				const auto outRegions = outImg->getRegions(state->outMipLevel);
				auto oit = outRegions.begin();
				core::vectorSIMDu32 offsetDifference, outByteStrides;
			} commonExecuteData;

			// iterate over output regions, then input cause read cache miss is faster
			for (; commonExecuteData.oit!=commonExecuteData.outRegions.end(); commonExecuteData.oit++)
			{
				IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
				state_type::TexelRange range = {state->inOffset,state->extent};
				CBasicImageFilterCommon::clip_region_functor_t clip(subresource,range,outFormat);
				// setup convert state
				// I know my two's complement wraparound well enough to make this work
				commonExecuteData.offsetDifference = state->outOffsetBaseLayer-(core::vectorSIMDu32(oit->imageOffset.x,oit->imageOffset.y,oit->imageOffset.z,oit->imageSubresource.baseArrayLayer)+state->inOffsetBaseLayer);
				commonExecuteData.outByteStrides = commonExecuteData.oit->getByteStrides(IImage::SBufferCopy::TexelBlockInfo(commonExecuteData.outFormat),getTexelOrBlockBytesize(commonExecuteData.outFormat));
				if (!perOutput(commonExecuteData,clip))
					return false;
			}

			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif
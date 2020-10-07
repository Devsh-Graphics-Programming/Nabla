// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

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
				CState()
				{
					extentLayerCount = core::vectorSIMDu32();
					inOffsetBaseLayer = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
				}
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
				const ICPUImage*		inImage = nullptr;
				ICPUImage*				outImage = nullptr;
		};
		using state_type = CState;
		
		static inline bool validate(state_type* state)
		{
			if (!state)
				return false;

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
		struct CommonExecuteData
		{
			const ICPUImage* const inImg;
			ICPUImage* const outImg;
			const ICPUImage::SCreationParams& inParams;
			const ICPUImage::SCreationParams& outParams;
			const E_FORMAT inFormat;
			const E_FORMAT outFormat;
			const uint32_t inBlockByteSize;
			const uint32_t outBlockByteSize;
			const uint8_t* const inData;
			uint8_t* const outData;
			const core::SRange<const IImage::SBufferCopy> inRegions;
			const core::SRange<const IImage::SBufferCopy> outRegions;
			const IImage::SBufferCopy* oit;									//!< oit is a current output handled region by commonExecute lambda. Notice that the lambda may execute executePerRegion a few times with different oits data since regions may overlap in a certain mipmap in an image!
			core::vectorSIMDu32 offsetDifference, outByteStrides; 
		};
		template<typename PerOutputFunctor>
		static inline bool commonExecute(state_type* state, PerOutputFunctor& perOutput)
		{
			if (!validate(state))
				return false;

			const auto* const inImg = state->inImage;
			auto* const outImg = state->outImage;
			const ICPUImage::SCreationParams& inParams = inImg->getCreationParameters();
			const ICPUImage::SCreationParams& outParams = outImg->getCreationParameters();
			const core::SRange<const IImage::SBufferCopy> outRegions = outImg->getRegions(state->outMipLevel);
			CommonExecuteData commonExecuteData =
			{
				inImg,
				outImg,
				inParams,
				outParams,
				inParams.format,
				outParams.format,
				getTexelOrBlockBytesize(inParams.format),
				getTexelOrBlockBytesize(outParams.format),
				reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer()),
				reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer()),
				inImg->getRegions(state->inMipLevel),
				outRegions,
				outRegions.begin(), {}, {}
			};

			// iterate over output regions, then input cause read cache miss is faster
			for (; commonExecuteData.oit!=commonExecuteData.outRegions.end(); commonExecuteData.oit++)
			{
				IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
				state_type::TexelRange range = {state->inOffset,state->extent};
				CBasicImageFilterCommon::clip_region_functor_t clip(subresource,range,commonExecuteData.outFormat);
				// setup convert state
				// I know my two's complement wraparound well enough to make this work
				const auto& outRegionOffset = commonExecuteData.oit->imageOffset;
				commonExecuteData.offsetDifference = state->outOffsetBaseLayer - (core::vectorSIMDu32(outRegionOffset.x, outRegionOffset.y, outRegionOffset.z, commonExecuteData.oit->imageSubresource.baseArrayLayer) + state->inOffsetBaseLayer);
				commonExecuteData.outByteStrides = commonExecuteData.oit->getByteStrides(TexelBlockInfo(commonExecuteData.outFormat));
				if (!perOutput(commonExecuteData,clip))
					return false;
			}

			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif
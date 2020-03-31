// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
{
namespace asset
{

// `FlatImageInput` means that there is at most one region per mip-map level
template<bool FlatImageInput=false>
class CConvertFormatImageFilter : public CImageFilter<CConvertFormatImageFilter<FlatImageInput>>
{
	public:
		virtual ~CConvertFormatImageFilter() {}
		
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				struct SubresourceOffset
				{
					uint32_t		mipLevel = 0u;
					uint32_t		baseArrayLayer = 0u;
				};

				VkExtent3D			extent = { 0u,0u,0u };
				SubresourceOffset	inSubresource = {};
				VkOffset3D			inOffset = { 0u,0u,0u };
				ICPUImage*			inImage = nullptr;
				SubresourceOffset	outSubresource = {};
				VkOffset3D			inOffset = { 0u,0u,0u };
				ICPUImage*			outImage = nullptr;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state)
				return nullptr;

			IImage::SSubresourceOffset subresource = {};
			TexelRange range = {state->inOffset,state->extent};
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource,range,state->inImage))
				return false;
			subresource.mipLevel = state->inSubresource.mipLevel;
			subresource.baseArrayLayer = state->inSubresource.baseArrayLayer;
			range.offset = state->outOffset;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource,range,state->outImage))
				return false;
			
			if (state->inSubresource.layerCount!=state.outSubresource.layerCount)
				return false;

			if (state->inRange.extent!=state->outRange.extent)
				return false;

			return true;
		}

#if 0
		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			auto* outImg = state->outImage;
			auto* inImg = state->inImage;
			const auto& inParams = inImg->getCreationParameters();
			const auto& outParams = outImg->getCreationParameters();
			const auto inFormat = inParams.format;
			const auto outFormat = outParams.format;
			const auto* inData = reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer());
			auto* outData = reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer());

			auto layerDifference = state->inSubresource.baseArrayLayer-state->outSubresource.baseArrayLayer;
			auto offsetDifference = state->inOffset-state->outOffset;
			auto convert = [state,inFormat,outFormat,inImg,offsetDifference](uint32_t blockArrayOffset, uint32_t x, uint32_t y, uint32_t z, uint32_t layer) -> bool
			{
				VkOffset3D pos = {x,y,z};
				pos += offsetDifference;
				const auto& rit = inImg->findPixelRegion(pos,layer+layerDifference);
				auto offset = rit->offset;

				const void* const sourcePixels[4] = {inData+offset,nullptr,nullptr,nullptr};
				convertColor(inFormat,outFormat,sourcePixels,outData+blockArrayOffset,1u,core::vector3d<uint32_t>(1u,1u,1u));
				return true;
			};
			IImage::SSubresourceOffset subresource = {};
			TexelRange range = {state->outOffset,state->extent};
			CBasicImageFilterCommon::clip_region_functor_t clip(subresource,range,outFormat);
			const auto& regions = img->getRegions();
			CBasicImageFilterCommon::executePerRegion(img, convert, regions.begin(), regions.end(), clip);

			return true;
		}
#endif
};

} // end namespace asset
} // end namespace irr

#endif
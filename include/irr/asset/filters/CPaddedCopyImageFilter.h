// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_PADDED_COPY_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_PADDED_COPY_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/ISampler.h"
#include "irr/asset/filters/CCopyImageFilter.h"

namespace irr
{
namespace asset
{

// copy while pasting a configurable border
class CPaddedCopyImageFilter : public CImageFilter<CPaddedCopyImageFilter>, public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CPaddedCopyImageFilter() {}
		
		class CState : public CMatchedSizeInOutImageFilterCommon::state_type
		{
			public:
				virtual ~CState() {}
				
				VkExtent3D borderPadding = { 0u,0u,0u };
				_IRR_STATIC_INLINE_CONSTEXPR auto NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP axisWraps[NumWrapAxes] = {ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT};
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			core::vectorSIMDu32 borderPadding(&state->borderPadding.width); borderPadding.w = 0u;
			if ((state->outOffsetBaseLayer<borderPadding).any())
				return false;
			const auto& outParams = state->outImage->getCreationParameters();
			core::vectorSIMDu32 extent(&outParams.extent.width); extent.w = outParams.arrayLayers;
			if ((state->outOffsetBaseLayer+state->extentLayerCount+borderPadding>extent).any())
				return false;

			auto const inFormat = state->inImage->getCreationParameters().format;
			auto const outFormat = outParams.format;
			// TODO: eventually remove when we can encode blocks
			for (auto i=0; i<CState::NumWrapAxes; i++)
			{
				if ((isBlockCompressionFormat(inFormat)||isBlockCompressionFormat(outFormat))&&state->axisWraps[i]!=ISampler::ETC_REPEAT)
					return false;
			}

			return getFormatClass(inFormat)==getFormatClass(outFormat);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			if (!CCopyImageFilter::execute(state))
				return false;

			core::vectorSIMDu32 padding(&state->borderPadding.width);
			padding = padding&core::vectorSIMDu32(~0u,~0u,~0u,0u);

			constexpr uint32_t borderRegionsCnt = 12u;
			IImageFilter::IState::TexelRange borderRegions[borderRegionsCnt];
			{
				core::vectorSIMDu32 ones(1u);
				struct STexelRange
				{
					core::vectorSIMDu32 offset;
					core::vectorSIMDu32 extent;
				};
				STexelRange regs[borderRegionsCnt];
				regs[0].offset = state->outOffsetBaseLayer - padding;
				regs[0].extent = padding;
				regs[0].extent.z = state->extentLayerCount.z + 2u * padding.z;

				regs[1].offset = state->outOffsetBaseLayer - padding.wyzw();
				regs[1].extent = core::max(padding,ones);
				regs[1].extent.x = state->extentLayerCount.x;

				regs[2].offset = state->outOffsetBaseLayer - padding.wyww();
				regs[2].offset.z += state->extentLayerCount.z;
				regs[2].extent = regs[1].extent;

				regs[3].offset = state->outOffsetBaseLayer - padding.wyzw();
				regs[3].offset.x += state->extentLayerCount.x;
				regs[3].extent = regs[0].extent;

				for (uint32_t i = 0u; i < 4u; ++i)
				{
					regs[4u + i] = regs[i];
					regs[4u + i].offset.y += padding.y + state->extentLayerCount.y;
				}

				regs[8].offset = state->outOffsetBaseLayer - padding.wwzw();
				regs[8].offset.x += state->extentLayerCount.x;
				regs[8].extent = core::max(padding,ones);
				regs[8].extent.y = state->extentLayerCount.y;

				regs[9] = regs[8];
				regs[9].offset.z += state->extentLayerCount.z + padding.z;

				regs[10].offset = state->outOffsetBaseLayer - padding.xwzw();
				regs[10].extent = regs[8].extent;

				regs[11] = regs[10];
				regs[11].offset.z += state->extentLayerCount.z + padding.z;

				for (uint32_t i = 0u; i < borderRegionsCnt; ++i)
				{
					memcpy(&borderRegions[i].offset.x, regs[i].offset.pointer, 3u*sizeof(uint32_t));
					memcpy(&borderRegions[i].extent.width, regs[i].extent.pointer, 3u*sizeof(uint32_t));
				}
			}

			auto perBlock = [&state](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
			{
				auto wrapped = wrapCoords(state, readBlockPos-state->outOffsetBaseLayer, state->extentLayerCount) + state->outOffsetBaseLayer;
				for (const auto& outreg : state->outImage->getRegions())
				{
					core::vectorSIMDu32 min(&outreg.imageOffset.x);
					min.w = outreg.imageSubresource.baseArrayLayer;
					core::vectorSIMDu32 max(&outreg.imageExtent.width);
					max.w = outreg.imageSubresource.layerCount;
					max += min;

					if ((wrapped>=min).all() && (wrapped<max).all())
					{
						const IImage::SBufferCopy::TexelBlockInfo blockInfo(state->outImage->getCreationParameters().format);
						const uint32_t texelSz = asset::getTexelOrBlockBytesize(state->outImage->getCreationParameters().format);
						const auto strides = outreg.getByteStrides(blockInfo, texelSz);//TODO precompute strides
						const uint64_t srcOffset = outreg.getByteOffset(wrapped-min, strides);

						uint8_t* const bufptr = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());
						memcpy(bufptr + readBlockArrayOffset, bufptr + srcOffset, texelSz);
						break;
					}
				}
			};
			for (const auto& outreg : state->outImage->getRegions())
			{
				for (uint32_t i = 0u; i < borderRegionsCnt; ++i)
				{
					IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
					clip_region_functor_t clip(subresource, borderRegions[i], state->outImage->getCreationParameters().format);
					IImage::SBufferCopy clipped_reg = outreg;
					if (clip(clipped_reg, &outreg))
						executePerBlock(state->outImage, clipped_reg, perBlock);
				}
			}

			return true;
		}

	private:
		static core::vectorSIMDu32 wrapCoords(const state_type* _state, const core::vectorSIMDu32& _coords, const core::vectorSIMDu32& _extent)
		{
			auto wrap_clamp_to_edge = [](int32_t a, int32_t sz) {
				return core::clamp(a, 0, sz-1);
			};
			auto wrap_clamp_to_border = [](int32_t a, int32_t sz) {
				return core::clamp(a, -1, sz);
			};
			auto wrap_repeat = [](int32_t a, int32_t sz) {
				return std::abs(a % sz);
			};
			auto wrap_mirror = [](int32_t a, int32_t sz) {
				const int32_t b = a % (2*sz) - sz;
				return std::abs( (sz-1) - (b>=0 ? b : -(b+1)) );
			};
			auto wrap_mirror_clamp_edge = [](int32_t a, int32_t sz) {
				return core::clamp(a>=0 ? a : -(1+a), 0, sz-1);
			};
			using wrap_fn_t = int32_t(*)(int32_t,int32_t);
			wrap_fn_t wrapfn[6]{
				wrap_repeat,
				wrap_clamp_to_edge,
				wrap_clamp_to_border,
				wrap_mirror,
				wrap_mirror_clamp_edge,
				nullptr
			};

			core::vectorSIMDu32 wrapped;
			wrapped.w = _coords.w;
			wrapped.x = wrapfn[_state->axisWraps[0]](_coords.x, _extent.x);
			wrapped.y = wrapfn[_state->axisWraps[1]](_coords.y, _extent.y);
			wrapped.z = wrapfn[_state->axisWraps[2]](_coords.z, _extent.y);

			return wrapped;
		}
};

} // end namespace asset
} // end namespace irr

#endif
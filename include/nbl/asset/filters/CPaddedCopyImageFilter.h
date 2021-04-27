// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_PADDED_COPY_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_PADDED_COPY_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/core.h"

#include <type_traits>

#include "nbl/asset/ISampler.h"
#include "nbl/asset/filters/CCopyImageFilter.h"
#include "nbl/asset/format/encodePixels.h"

namespace nbl
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
				
				_NBL_STATIC_INLINE_CONSTEXPR auto NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP axisWraps[NumWrapAxes] = {ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT};
				ISampler::E_TEXTURE_BORDER_COLOR borderColor;
				VkOffset3D relativeOffset;
				VkExtent3D paddedExtent;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			const auto& outParams = state->outImage->getCreationParameters();

			core::vector3du32_SIMD paddedExtent(&state->paddedExtent.width); paddedExtent = paddedExtent&core::vectorSIMDu32(~0u,~0u,~0u,0u);
			core::vector3du32_SIMD reloffset(&state->relativeOffset.x); reloffset = reloffset&core::vectorSIMDu32(~0u,~0u,~0u,0u);
			core::vectorSIMDu32 outImgExtent(&outParams.extent.width); outImgExtent.w = outParams.arrayLayers;
			if (((reloffset+state->extentLayerCount)>paddedExtent).xyzz().any())
				return false;
			if (((state->outOffsetBaseLayer+paddedExtent)>outImgExtent).xyzz().any())
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

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			core::vector3du32_SIMD paddedExtent(&state->paddedExtent.width); paddedExtent = paddedExtent&core::vectorSIMDu32(~0u, ~0u, ~0u, 0u);
			core::vector3du32_SIMD reloffset(&state->relativeOffset.x); reloffset = reloffset&core::vectorSIMDu32(~0u,~0u,~0u,0u);
			state->outOffsetBaseLayer += reloffset;//abuse state for a moment
			if (!CCopyImageFilter::execute<ExecutionPolicy>(policy,state))
				return false;
			state->outOffsetBaseLayer -= reloffset;

			constexpr uint32_t maxBorderRegions = 6u;
			uint32_t borderRegionCount = 0u;
			IImageFilter::IState::TexelRange borderRegions[maxBorderRegions];
			{
				uint32_t i = 0u;
				//x-
				core::vector3du32_SIMD extent;
				core::vector3du32_SIMD offset;
				if (reloffset.x)
				{
					extent = paddedExtent;
					extent.x = reloffset.x;
					memcpy(&borderRegions[i].extent.width, &extent.x, 3u*sizeof(uint32_t));
					offset = state->outOffsetBaseLayer;
					memcpy(&borderRegions[i].offset.x, &offset.x, 3u*sizeof(uint32_t));
					++i;
				}
				//x+
				extent = paddedExtent;
				extent.x -= state->extentLayerCount.x + reloffset.x;
				if (extent.x)
				{
					offset = core::vector3du32_SIMD(0u);
					offset.x = reloffset.x + state->extent.width;
					memcpy(&borderRegions[i].extent.width, &extent.x, 3u*sizeof(uint32_t));
					if (offset.x < paddedExtent.x)
					{
						offset += state->outOffsetBaseLayer;
						memcpy(&borderRegions[i].offset.x, &offset.x, 3u*sizeof(uint32_t));
						++i;
					}
				}
				//y-
				if (reloffset.y)
				{
					extent = paddedExtent;
					extent.y = reloffset.y;
					memcpy(&borderRegions[i].extent.width, &extent.x, 3u*sizeof(uint32_t));
					offset = state->outOffsetBaseLayer;
					memcpy(&borderRegions[i].offset.x, &offset.x, 3u*sizeof(uint32_t));
					++i;
				}
				//y+
				extent = paddedExtent;
				extent.y -= state->extentLayerCount.y + reloffset.y;
				if (extent.y)
				{
					offset = core::vector3du32_SIMD(0u);
					offset.y = reloffset.y + state->extent.height;
					memcpy(&borderRegions[i].extent.width, &extent.x, 3u*sizeof(uint32_t));
					if (offset.y < paddedExtent.y)
					{
						offset += state->outOffsetBaseLayer;
						memcpy(&borderRegions[i].offset.x, &offset.x, 3u*sizeof(uint32_t));
						++i;
					}
				}
				//z-
				if (reloffset.z)
				{
					extent = paddedExtent;
					extent.z = reloffset.z;
					memcpy(&borderRegions[i].extent.width, &extent.x, 3u*sizeof(uint32_t));
					borderRegions[i].offset = {0u,0u,0u};
					++i;
				}
				//z+
				extent = paddedExtent;
				extent.z -= state->extentLayerCount.z + reloffset.z;
				if (extent.z)
				{
					offset = core::vector3du32_SIMD(0u);
					offset.z = reloffset.z + state->extent.depth;
					memcpy(&borderRegions[i].extent.width, &extent.x, 3u*sizeof(uint32_t));
					if (offset.z < paddedExtent.z)
					{
						offset += state->outOffsetBaseLayer;
						memcpy(&borderRegions[i].offset.x, &offset.x, 3u*sizeof(uint32_t));
						++i;
					}
				}
				borderRegionCount = i;
			}

			uint8_t* const bufptr = reinterpret_cast<uint8_t*>(state->outImage->getBuffer()->getPointer());
			IImageFilter::IState::ColorValue borderColor;
			encodeBorderColor(state->borderColor, state->outImage->getCreationParameters().format, borderColor.asByte);
			IImageFilter::IState::ColorValue::WriteMemoryInfo borderColorWrite(state->outImage->getCreationParameters().format, bufptr);

			auto perBlock = [&state,&borderColor,&borderColorWrite,&bufptr,&reloffset](uint32_t blockArrayOffset, core::vectorSIMDu32 readBlockPos)
			{
				const TexelBlockInfo blockInfo(state->outImage->getCreationParameters().format);
				const uint32_t texelSz = asset::getTexelOrBlockBytesize(state->outImage->getCreationParameters().format);

				auto wrapped = wrapCoords(state, readBlockPos-state->outOffsetBaseLayer-reloffset, state->extentLayerCount);
				//wrapped coords exceeding image on any axis implies usage of border color for this border-texel
				//this also covers check for -1 (-1 is max unsigned val)
				if ((wrapped>=state->extentLayerCount).xyzz().any())
				{
					borderColor.writeMemory(borderColorWrite, blockArrayOffset);
					return;
				}

				wrapped += state->outOffsetBaseLayer+reloffset;
				for (const auto& outreg : state->outImage->getRegions(state->outMipLevel))
				{
					core::vectorSIMDu32 _min(&outreg.imageOffset.x);
					_min.w = outreg.imageSubresource.baseArrayLayer;
					core::vectorSIMDu32 _max(&outreg.imageExtent.width);
					_max.w = outreg.imageSubresource.layerCount;
					_max += _min;

					if ((wrapped>= _min).all() && (wrapped<_max).all())
					{
						const auto strides = outreg.getByteStrides(blockInfo);//TODO precompute strides
						const uint64_t srcOffset = outreg.getByteOffset(wrapped-_min, strides);

						memcpy(bufptr+blockArrayOffset, bufptr+srcOffset, texelSz);
						break;
					}
				}
			};
			for (const auto& outreg : state->outImage->getRegions(state->outMipLevel))
			{
				for (uint32_t i = 0u; i < borderRegionCount; ++i)
				{
					IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->outMipLevel,state->outBaseLayer,state->layerCount};
					clip_region_functor_t clip(subresource, borderRegions[i], state->outImage->getCreationParameters().format);
					IImage::SBufferCopy clipped_reg = outreg;
					if (clip(clipped_reg, &outreg))
						executePerBlock<ExecutionPolicy>(policy,state->outImage,clipped_reg,perBlock);
				}
			}

			return true;
		}
		static inline bool execute(state_type* state)
		{
			return execute(std::execution::seq,state);
		}

	private:
		static core::vectorSIMDu32 wrapCoords(const state_type* _state, const core::vectorSIMDu32& _coords, const core::vectorSIMDu32& _extent)
		{
			//i am totally confused about wrapping equations given in vulkan/opengl spec... i'm misunderstanding something or equations given there are not complete
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
				return std::abs( (sz-1) - (b>=0 ? b : -(b+1)) ) % sz;
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
			wrapped.z = wrapfn[_state->axisWraps[2]](_coords.z, _extent.z);

			return wrapped;
		}
		static void encodeBorderColor(ISampler::E_TEXTURE_BORDER_COLOR _color, E_FORMAT _fmt, void* _encbuf)
		{
			constexpr double fpColors[3][4]
			{
				{0.0,0.0,0.0,0.0},
				{0.0,0.0,0.0,1.0},
				{1.0,1.0,1.0,1.0}
			};
			constexpr uint64_t intColors[3][4]
			{
				{0u,0u,0u,0u},
				{0u,0u,0u,~0ull},
				{~0ull,~0ull,~0ull,~0ull}
			};

			const void* src = nullptr;
			if (_color&1u)
				src = intColors[_color/2];
			else
				src = fpColors[_color/2];
			encodePixelsRuntime(_fmt, _encbuf, src);
		}
};

} // end namespace asset
} // end namespace nbl

#endif

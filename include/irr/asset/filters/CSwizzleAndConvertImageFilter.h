// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/ICPUImageView.h"
#include "irr/asset/filters/CConvertFormatImageFilter.h"

namespace irr
{
namespace asset
{

// do a per-pixel recombination of image channels while converting
class CSwizzleAndConvertImageFilter : public CConvertFormatImageFilter
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = CMatchedSizeInOutImageFilterCommon::state_type;

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

			enum FORMAT_META_TYPE
			{
				FMT_FLOAT,
				FMT_INT,
				FMT_UINT
			};
			auto getMetaType = [](E_FORMAT format) -> FORMAT_META_TYPE
			{
				if (!isIntegerFormat(format))
					return FMT_FLOAT;
				else if (isSignedFormat(format))
					return FMT_INT;
				else
					return FMT_UINT;
			};
			const FORMAT_META_TYPE inMetaType = getMetaType(inFormat);
			const FORMAT_META_TYPE outMetaType = getMetaType(outFormat);
			
			const auto* inData = reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer());
			auto* outData = reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer());

			const auto outRegions = outImg->getRegions(state->outMipLevel);
			auto oit = outRegions.begin();
			core::vectorSIMDu32 offsetDifference,outByteStrides;

			auto inRegions = inImg->getRegions(state->inMipLevel);
			// iterate over output regions, then input cause read cache miss is faster
			for (; oit!=outRegions.end(); oit++)
			{
				IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
				state_type::TexelRange range = {state->inOffset,state->extent};
				CBasicImageFilterCommon::clip_region_functor_t clip(subresource,range,outFormat);
				// setup convert state
				// I know my two's complement wraparound well enough to make this work
				offsetDifference = state->outOffsetBaseLayer-(core::vectorSIMDu32(oit->imageOffset.x,oit->imageOffset.y,oit->imageOffset.z,oit->imageSubresource.baseArrayLayer)+state->inOffsetBaseLayer);
				outByteStrides = oit->getByteStrides(IImage::SBufferCopy::TexelBlockInfo(outFormat), getTexelOrBlockBytesize(outFormat));
				switch (outMetaType)
				{
					case FMT_FLOAT:
					{
						switch (outMetaType)
						{
							case FMT_FLOAT:
							{
								Swizzle<double,double> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
							case FMT_INT:
							{
								Swizzle<double,int64_t> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
							default:
							{
								Swizzle<double,uint64_t> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
						}
						break;
					}
					case FMT_INT:
					{
						switch (outMetaType)
						{
							case FMT_FLOAT:
							{
								Swizzle<int64_t,double> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
							case FMT_INT:
							{
								Swizzle<int64_t,int64_t> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
							default:
							{
								Swizzle<int64_t,uint64_t> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
						}
						break;
						break;
					}
					default:
					{
						switch (outMetaType)
						{
							case FMT_FLOAT:
							{
								Swizzle<uint64_t,double> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
							case FMT_INT:
							{
								Swizzle<uint64_t,int64_t> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
							default:
							{
								Swizzle<uint64_t,uint64_t> swizzle();
								CBasicImageFilterCommon::executePerRegion(inImg, swizzle, inRegions.begin(), inRegions.end(), clip);
								break;
							}
						}
						break;
					}
				}
			}

			return true;
		}

	private:
		template<typename InType, typename OutType>
		struct Swizzle
		{
				inline void operator()(uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					constexpr auto MaxChannels = 4;

					const void* pixels[MaxPlanes] = { inData+readBlockArrayOffset,nullptr,nullptr,nullptr };
					InType inTmp[MaxChannels];
#if 0
					decodePixels(pixels,inTmp,,);

					OutType outTmp[MaxChannels];
					for (auto i=0; i<MaxChannels; i++)
						doSwizzle(outTmp[i],(&swizzle.r)[i]);
					auto localOutPos = readBlockPos+offsetDifference;
					encodePixels(outData+oit->getByteOffset(localOutPos, outByteStrides),outTmp);
#endif
					assert(false);
				}
			private:
				static inline void doSwizzle(OutType& out, void* swizzle)
				{
					assert(false);
				}
		};
		/*
				void* pixels[4] = {,nullptr,nullptr,nullptr};
				auto doSwizzle = [&pixels](auto tmp[4]) -> void
				{
					decodePixels(format,pixels,tmp,0u,0u);
					std::decay<decltype(tmp[0])>::type tmp2[4];
					tmp2[0] = tmp[swizzle.r];
					tmp2[1] = tmp[swizzle.g];
					tmp2[2] = tmp[swizzle.b];
					tmp2[3] = tmp[swizzle.a];
					encodePixels(format,pixels[0],tmp2);
				};
		*/
};

} // end namespace asset
} // end namespace irr

#endif
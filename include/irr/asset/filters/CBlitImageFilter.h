// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_BLIT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_BLIT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CMatchedSizeInOutImageFilterCommon.h"

namespace irr
{
namespace asset
{


class IImageFilterKernel
{
	public
		virtual bool isSeparable() const = 0;
		virtual bool validate(ICPUImage* inImage, ICPUImage* outImage) const = 0;
};

template<class CRTP>
class CImageFilterKernel
{
	public:
		inline virtual bool isSeparable() const override
		{
			return CRTP::is_separable;
		}
		inline virtual bool validate(ICPUImage* inImage, ICPUImage* outImage) const override
		{
			return CRTP::validate(inImage,outImage);
		}
};


class CBoxImageFilterKernel : public CImageFilterKernel<CBoxImageFilterKernel>
{
	public:
		IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			return !isIntegerFormat(inParams);
		}
};

class CTriangleImageFilterKernel : public CImageFilterKernel<CTriangleImageFilterKernel>
{
	public:
		IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			return !isIntegerFormat(inParams);
		}
};

class CGaussianImageFilterKernel : public CImageFilterKernel<CGaussianImageFilterKernel>
{
	public:
		IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			assert(false); // TBD
			const auto& inParams = inImage->getCreationParameters();
			return !isIntegerFormat(inParams);
		}
};

class CKaiserImageFilterKernel : public CImageFilterKernel<CKaiserImageFilterKernel>
{
	public:
		IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			return !isIntegerFormat(inParams);
		}
};

template<typename Functor>
class CCompareImageFilterKernel : public CImageFilterKernel<CCompareImageFilterKernel>
{
	public:
		IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			assert(false); // TBD
			return true;
		}
};
using CMinImageFilterKernel = CCompareImageFilterKernel<core::min_t<double,double> >;
using CMaxImageFilterKernel = CCompareImageFilterKernel<core::max_t<double,double> >;


// copy while filtering the input into the output
template<class Kernel=CBoxImageFilterKernel>
class CBlitImageFilter : public CImageFilter<CBlitImageFilter<Kernel> >, public CBasicImageFilterCommon
{
	public:
		virtual ~CBlitImageFilter() {}
		
		class CState : public IImageFilter::IState
		{
			public:
				CState()
				{
					inOffsetBaseLayer = core::vectorSIMDu32();
					inExtentLayerCount = core::vectorSIMDu32();
					outOffsetBaseLayer = core::vectorSIMDu32();
					outExtentLayerCount = core::vectorSIMDu32();
				}
				virtual ~CState() {}

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
					core::vectorSIMDu32 inExtentLayerCount;
					struct
					{
						VkExtent3D		inExtent;
						uint32_t		inLayerCount;
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
				union
				{
					core::vectorSIMDu32 outExtentLayerCount;
					struct
					{
						VkExtent3D		outExtent;
						uint32_t		outLayerCount;
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
				return false;

			if (state->inLayerCount!=state->outLayerCount)
				return false;

			IImage::SSubresourceLayers subresource = { static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->inLayerCount };
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource, {state->inOffset,state->inExtent}, state->inImage))
				return false;
			subresource.mipLevel = state->outMipLevel;
			subresource.baseArrayLayer = state->outBaseLayer;
			if (!CBasicImageFilterCommon::validateSubresourceAndRange(subresource, {state->outOffset,state->outExtent}, state->outImage))
				return false;

			// TODO: remove this later when we can actually write/encode to block formats
			if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
				return false;

			return Kernel::validate(inImage,outImage);
		}

		static inline bool execute(state_type* state)
		{
			// go over output regions
				// for every output pixel gather inputs
#if 0
			auto perOutputRegion = [](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat)==getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something

				const auto blockDims = asset::getBlockDimensions(commonExecuteData.inFormat);
				auto copy = [&commonExecuteData,&blockDims](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
					memcpy(commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides),commonExecuteData.inData+readBlockArrayOffset,commonExecuteData.outBlockByteSize);
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg,copy,commonExecuteData.inRegions.begin(),commonExecuteData.inRegions.end(),clip);

				return true;
			};
			return commonExecute(state,perOutputRegion);
#else
			assert(false);
			return false;
#endif
		}
};

} // end namespace asset
} // end namespace irr

#endif
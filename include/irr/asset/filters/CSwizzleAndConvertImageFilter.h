// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_SWIZZLE_AND_CONVERT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"


#include <type_traits>


#include "irr/asset/filters/CMatchedSizeInOutImageFilterCommon.h"

#include "irr/asset/ICPUImageView.h"

#include "irr/asset/format/convertColor.h"


namespace irr
{
namespace asset
{


namespace impl
{


template<typename Swizzle>
class CSwizzleAndConvertImageFilterBase : public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CSwizzleAndConvertImageFilterBase() {}

		using state_type = CMatchedSizeInOutImageFilterCommon::state_type;

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			// TODO: need to triple check it works when we finally enable this feature
			if (isBlockCompressionFormat(state->outImage->getCreationParameters().format))
				return false;

			return true;
		}
};

template<>
class CSwizzleAndConvertImageFilterBase<void> : public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CSwizzleAndConvertImageFilterBase() {}

		class CState : public CMatchedSizeInOutImageFilterCommon::state_type
		{
			public:
				PolymorphicSwizzle* swizzle;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!CSwizzleAndConvertImageFilterBase<VoidSwizzle>::validate(state))
				return false;

			if (!state->swizzle)
				return false;

			return true;
		}
};


}

struct DefaultSwizzle
{
	ICPUImageView::SComponentMapping swizzle;

	template<typename InT, typename OutT>
	inline void operator()(const InT in[SwizzleBase::MaxChannels], OutT out[SwizzleBase::MaxChannels]) const
	{
		auto getComponent = [&in](E_SWIZZLE s, auto id) -> type
		{
			if (s < ICPUImageView::SComponentMapping::ES_IDENTITY)
				return in[id];
			else if (s < ICPUImageView::SComponentMapping::ES_ZERO)
				return type(0);
			else if (s == ICPUImageView::SComponentMapping::ES_ONE)
				return type(1);
			else
				return in[s-ICPUImageView::SComponentMapping::ES_R];
		};
		for (auto i=0; i<SwizzleBase::MaxChannels; i++)
			out[i] = getComponent((&swizzle.r)[i],i);
	}
};


// do a per-pixel recombination of image channels while converting
template<E_FORMAT inFormat=EF_UNKNOWN, E_FORMAT outFormat=EF_UNKNOWN, typename Swizzle=DefaultSwizzle>
class CSwizzleAndConvertImageFilter : public CImageFilter<CSwizzleAndConvertImageFilter<inFormat,outFormat,Swizzle>>, public impl::CSwizzleAndConvertImageFilterBase<Swizzle>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Swizzle>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Swizzle>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=inFormat)
				return false;

			if (state->inImage->getCreationParameters().format!=outFormat)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			const auto blockDims = asset::getBlockDimensions(inFormat);
			#ifdef _IRR_DEBUG
				assert(blockDims.z==1u);
				assert(blockDims.w==1u);
			#endif
			auto perOutputRegion = [&blockDims](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				auto swizzle = [&commonExecuteData,&blockDims](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY=0u; blockY<blockDims; blockY++)
					for (auto blockX=0u; blockX<blockDims; blockX++)
					{
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides); // TODO!
						if constexpr(!std::is_void<Swizzle>::value)
							convertColor<inFormat,outFormat,Swizzle>(srcPix,dstPix,blockX,blockY,state);
						else
							convertColor<inFormat,outFormat,Swizzle>(srcPix,dstPix,blockX,blockY,state->swizzle);
					}
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
			};
			CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
};

template<typename Swizzle>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle> : public CImageFilter<CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>>, public impl::CSwizzleAndConvertImageFilterBase<Swizzle>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename impl::CSwizzleAndConvertImageFilterBase<Swizzle>::state_type;

		static inline bool validate(state_type* state)
		{
			return impl::CSwizzleAndConvertImageFilterBase<Swizzle>::validate(state);
		}

		static inline bool execute(state_type* state)
		{
			const auto blockDims = asset::getBlockDimensions(inFormat);
			#ifdef _IRR_DEBUG
				assert(blockDims.z==1u);
				assert(blockDims.w==1u);
			#endif
			auto perOutputRegion = [&blockDims](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				auto swizzle = [&commonExecuteData,&blockDims](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
				{
					constexpr auto MaxPlanes = 4;
					const void* srcPix[MaxPlanes] = { commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr };

					for (auto blockY=0u; blockY<blockDims; blockY++)
					for (auto blockX=0u; blockX<blockDims; blockX++)
					{
						auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
						uint8_t* dstPix = commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides); // TODO!
						if constexpr(!std::is_void<Swizzle>::value)
							convertColor<Swizzle>(inFormat,outFormat,srcPix,dstPix,blockX,blockY,state);
						else
							convertColor<Swizzle>(inFormat,outFormat,srcPix,dstPix,blockX,blockY,state->swizzle);
					}
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg, swizzle, commonExecuteData.inRegions.begin(), commonExecuteData.inRegions.end(), clip);
			};
			CMatchedSizeInOutImageFilterCommon::commonExecute(state,perOutputRegion);
		}
};

template<E_FORMAT outFormat, typename Swizzle>
class CSwizzleAndConvertImageFilter<EF_UNKNOWN,outFormat,Swizzle> : public CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Swizzle>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=outFormat)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			// TODO: improve later
			return CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>::execute(state);
		}
};

template<E_FORMAT inFormat, typename Swizzle>
class CSwizzleAndConvertImageFilter<inFormat,EF_UNKNOWN,Swizzle> : public CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>
{
	public:
		virtual ~CSwizzleAndConvertImageFilter() {}

		using state_type = typename CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>::state_type;

		static inline bool validate(state_type* state)
		{
			if (!impl::CSwizzleAndConvertImageFilterBase<Swizzle>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=inFormat)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			// TODO: improve later
			return CSwizzleAndConvertImageFilter<EF_UNKNOWN,EF_UNKNOWN,Swizzle>::execute(state);
		}
};


} // end namespace asset
} // end namespace irr

#endif
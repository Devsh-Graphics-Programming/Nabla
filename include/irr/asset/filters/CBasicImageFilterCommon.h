// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__
#define __IRR_C_BASIC_IMAGE_FILTER_COMMON_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/IImageFilter.h"

namespace irr
{
namespace asset
{

class CBasicImageFilterCommon
{
	public:
		template<typename F>
		void executePerBlock(ICPUImage* image, const irr::asset::IImage::SBufferCopy& region, F& f)
		{
			const auto& extent = image->getCreationParameters().extent;
			VkExtent trueExtent;
			trueExtent.width = region.bufferRowLength ? region.bufferRowLength:region.imageExtent.width;
			trueExtent.height = region.bufferImageHeight ? region.bufferImageHeight:region.imageExtent.height;
			trueExtent.depth = region.imageExtent.depth;

			for (uint32_t zPos = 0; zPos<extent.depth; ++zPos)
			for (uint32_t yPos = 0; yPos<extent.height; ++yPos)
			for (uint32_t xPos = 0; xPos<extent.width; ++xPos)
			{
				auto texelPtr = (zPos*trueExtent.Y+yPos)*trueExtent.X+xPos;
				f(texelPtr,xPos,yPos,zPos);
			}
		}
};

class CBasicInImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				Subsection			subsection = {};
				const ICPUImage*	inImage = nullptr;
				Offsets				inOffsets = {};
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			const auto& inCreationParams = state->inImage->getCreationParameters();
			// TODO: Extra epic validation

			return true;
		}

	protected:
		virtual ~CBasicInImageFilterCommon() = 0;
};
class CBasicOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				Subsection	subsection = {};
				ICPUImage*	outImage = nullptr;
				Offsets		outOffsets = {};
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			const auto& outCreationParams = state->outImage->getCreationParameters();
			// TODO: Extra epic validation

			return true;
		}

	protected:
		virtual ~CBasicOutImageFilterCommon() = 0;
};
class CBasicInOutImageFilterCommon : public CBasicImageFilterCommon
{
	public:
		class CState : public IImageFilter::IState
		{
			public:
				virtual ~CState() {}

				Subsection			subsection = {};
				const ICPUImage*	inImage = nullptr;
				ICPUImage*			outImage = nullptr;
				Offsets				inOffsets = {};
				Offsets				outOffsets = {};
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			if (!state)
				return nullptr;

			const auto& inCreationParams = state->inImage->getCreationParameters();
			const auto& outCreationParams = state->outImage->getCreationParameters();
			// TODO: Extra epic validation

			return true;
		}
		

	protected:
		virtual ~CBasicInOutImageFilterCommon() = 0;
};
// will probably need some per-pixel helper class/functions (that can run a templated functor per-pixel to reduce code clutter)

} // end namespace asset
} // end namespace irr

#endif
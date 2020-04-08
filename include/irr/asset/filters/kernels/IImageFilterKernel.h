// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_I_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/ICPUImage.h"

namespace irr
{
namespace asset
{


class IImageFilterKernel
{
	public:
		virtual bool pIsSeparable() const = 0;
		virtual bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const = 0;
};

template<class CRTP>
class CImageFilterKernel : public IImageFilterKernel
{
	public:
		inline virtual bool pIsSeparable() const override
		{
			return CRTP::is_separable;
		}
		inline virtual bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const override
		{
			return CRTP::validate(inImage,outImage);
		}
};

} // end namespace asset
} // end namespace irr

#endif
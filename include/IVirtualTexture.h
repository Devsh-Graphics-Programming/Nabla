// Copyright (C) 2017 Mateusz 'DevSH' Kielan
// This file is part of "IrrlichtBAw".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_VIRTUAL_TEXTURE_H_INCLUDED__
#define __I_VIRTUAL_TEXTURE_H_INCLUDED__

#include "irr/asset/CImageData.h"
#include "IFrameBuffer.h"

namespace irr
{
namespace video
{

class IVirtualTexture : public virtual core::IReferenceCounted
{
        _IRR_INTERFACE_CHILD(IVirtualTexture) {}
    public:
        enum E_DIMENSION_COUNT
        {
            EDC_ZERO=0,
            EDC_ONE,
            EDC_TWO,
            EDC_THREE,
            EDC_COUNT,
            EDC_FORCE32BIT=0xffffffffu
        };
        enum E_VIRTUAL_TEXTURE_TYPE
        {
            EVTT_OPAQUE_FILTERABLE,
            EVTT_2D_MULTISAMPLE,
            EVTT_BUFFER_OBJECT,
            EVTT_VIEW,
            EVTT_COUNT
        };

        virtual E_DIMENSION_COUNT getDimensionality() const = 0;

        //! Get dimension (=size) of the texture.
        /** \return The size of the texture. */
        virtual const uint32_t* getSize() const = 0;

        //!
        virtual E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const = 0;

        //! Get driver type of texture.
        /** This is the driver, which created the texture. This method is used
        internally by the video devices, to check, if they may use a texture
        because textures may be incompatible between different devices.
        \return Driver type of texture. */
        virtual E_DRIVER_TYPE getDriverType() const = 0;

        //! Get the color format of texture.
        /** \return The color format of texture. */
        virtual asset::E_FORMAT getColorFormat() const = 0;

        //! Returns if the texture has an alpha channel
        inline bool hasAlpha() const {
            return asset::getFormatChannelCount(getColorFormat()) == 4u;
        }
};

class IRenderableVirtualTexture : public IVirtualTexture
{
        _IRR_INTERFACE_CHILD(IRenderableVirtualTexture) {}
    public:
		//! Returns the two dimensional size of an IFrameBuffer attachment
		/**
		@returns The two dimensional size of the max rendering viewport which could be configured on an IFrameBuffer with this object attached.
		*/
        virtual core::dimension2du getRenderableSize() const = 0;
};

}
}

#endif // __I_VIRTUAL_TEXTURE_H_INCLUDED__

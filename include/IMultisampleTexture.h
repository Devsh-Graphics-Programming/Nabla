// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MULTISAMPLE_TEXTURE_H_INCLUDED__
#define __I_MULTISAMPLE_TEXTURE_H_INCLUDED__

#include "path.h"
#include "dimension2d.h"
#include "irr/asset/CImageData.h"
#include "IVirtualTexture.h"
#include "IFrameBuffer.h"

namespace irr
{
namespace video
{

class IMultisampleTexture : public IRenderableVirtualTexture, public IDriverMemoryBacked
{
    protected:
        IMultisampleTexture(const SDriverMemoryRequirements& reqs) : IDriverMemoryBacked(reqs) {}

            _IRR_INTERFACE_CHILD(IMultisampleTexture) {}
    public:
        enum E_MULTISAMPLE_TEXTURE_TYPE
        {
            EMTT_2D=0,
            EMTT_2D_ARRAY,
            EMTT_COUNT
        };


        //!
        virtual E_MULTISAMPLE_TEXTURE_TYPE getTextureType() const = 0;

        //! sampleCount of 0 indicates not 0 samples but the same amount as old texture
        virtual bool resize(const uint32_t* size, const uint32_t& sampleCount=0) = 0;

        //!
        virtual bool resize(const uint32_t* size, const uint32_t& sampleCount, const bool& fixedSampleLocations) = 0;

        //!
        virtual uint32_t getSampleCount() const = 0;

        //!
        virtual bool usesFixedSampleLocations() const = 0;
};


} // end namespace video
} // end namespace irr

#endif



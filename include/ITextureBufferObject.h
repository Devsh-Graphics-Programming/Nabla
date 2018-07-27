// Copyright (C) 2017 Mateusz 'DevSH' Kielan
// This file is part of "IrrlichtBAw".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_TEXTURE_BUFFER_OBJECT_H_INCLUDED__
#define __I_TEXTURE_BUFFER_OBJECT_H_INCLUDED__

#include "IVirtualTexture.h"

namespace irr
{
namespace video
{

class ITextureBufferObject : public IVirtualTexture
{
    public:
        enum E_TEXURE_BUFFER_OBJECT_FORMAT
        {
            ///1
            ETBOF_R8=0,
            ETBOF_R16,
            ETBOF_R16F,
            ETBOF_R32F,
            ETBOF_R8I,
            ETBOF_R16I,
            ETBOF_R32I,
            ETBOF_R8UI,
            ETBOF_R16UI,
            ETBOF_R32UI,
            ///2
            ETBOF_RG8,
            ETBOF_RG16,
            ETBOF_RG16F,
            ETBOF_RG32F,
            ETBOF_RG8I,
            ETBOF_RG16I,
            ETBOF_RG32I,
            ETBOF_RG8UI,
            ETBOF_RG16UI,
            ETBOF_RG32UI,
            ///3
            ETBOF_RGB32F,
            ETBOF_RGB32I,
            ETBOF_RGB32UI,
            ///4
            ETBOF_RGBA8,
            ETBOF_RGBA16,
            ETBOF_RGBA16F,
            ETBOF_RGBA32F,
            ETBOF_RGBA8I,
            ETBOF_RGBA16I,
            ETBOF_RGBA32I,
            ETBOF_RGBA8UI,
            ETBOF_RGBA16UI,
            ETBOF_RGBA32UI,
            ETBOF_COUNT
        };

        virtual E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const {return EVTT_BUFFER_OBJECT;}

        // Use ETBOF_COUNT to keep current format
        virtual bool bind(IGPUBuffer* buffer, E_TEXURE_BUFFER_OBJECT_FORMAT format=ETBOF_COUNT, const size_t& offset=0, const size_t& length=0) = 0;

        virtual bool rebindRevalidate() = 0;

        virtual uint64_t getByteSize() const = 0;
};

}
}

#endif // __I_TEXTURE_BUFFER_OBJECT_H_INCLUDED__


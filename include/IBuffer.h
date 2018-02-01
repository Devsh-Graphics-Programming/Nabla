
// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_BUFFER_H_INCLUDED__
#define __I_BUFFER_H_INCLUDED__

#include "IReferenceCounted.h"
#include "stdint.h"

namespace irr
{
namespace core
{

enum E_BUFFER_TYPE
{
    EBT_VERTEX_ATTRIBUTE = 0,
    EBT_INDEX,
    EBT_COPY_READ,
    EBT_COPY_WRITE,
    EBT_PIXEL_UPLOAD,
    EBT_PIXEL_DOWNLOAD,
    EBT_QUERY,
    EBT_TEXTURE,
    EBT_TRANSFORM_FEEDBACK,
    EBT_SHADER_UNIFORM,
    EBT_DRAW_INDIRECT,
    EBT_INDIRECT_PARAMETER,
    EBT_ATOMIC_COUNTER,
    EBT_COMPUTE_INDIRECT_DISPATCH,
    EBT_SHADER_STORAGE,
    EBT_UNSPECIFIED_BUFFER
};

class IBuffer : public virtual IReferenceCounted
{
    public:
        virtual E_BUFFER_TYPE getBufferType() const = 0;
        //virtual E_BUFFER_CLASS getBufferClass() const = 0;
        //! size in BYTES
        virtual const uint64_t& getSize() const = 0;
        //! This function will invalidate any sizes, pointers etc. returned before!
        /** @returns True on success (some types always return false since they dont support resize) */
        virtual bool reallocate(const size_t &newSize, const bool& forceRetentionOfData=false, const bool &reallocateIfShrink=false) = 0;

        virtual const uint64_t& getLastTimeReallocated() const {return lastTimeReallocated;}
    protected:
        uint64_t lastTimeReallocated;
    private:
        //
};

} // end namespace scene
} // end namespace irr

#endif


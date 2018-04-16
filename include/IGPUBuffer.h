// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_GPU_BUFFER_H_INCLUDED__
#define __I_GPU_BUFFER_H_INCLUDED__

#include "IBuffer.h"

namespace irr
{
namespace video
{

enum E_GPU_BUFFER_ACCESS
{
    EGBA_READ=0,
    EGBA_WRITE,
    EGBA_READ_WRITE,
    EGBA_NONE
};

class IGPUBuffer : public core::IBuffer
{
    public:
        virtual void clandestineRecreate(const size_t& size, const void* data) = 0;

        virtual void updateSubRange(const size_t& offset, const size_t& size, const void* data) = 0;

        virtual bool canUpdateSubRange() const = 0;

	//! @returns True if the buffer is actually a mapped buffer.
        virtual const bool isMappedBuffer() const {return false;}
    private:
        //
};

} // end namespace scene
} // end namespace irr

#endif


// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_GPU_MAPPED_BUFFER_H_INCLUDED__
#define __I_GPU_MAPPED_BUFFER_H_INCLUDED__

#include "IGPUBuffer.h"

namespace irr
{
namespace video
{

//! Persistently Mapped buffer
class IGPUMappedBuffer : public virtual video::IGPUBuffer
{
    public:
        //! WARNING: RESIZE will invalidate pointer
        //! WARNING: NEED TO FENCE BEFORE USE!!!!!!!!!!!!!
        virtual void* getPointer() = 0;
    private:
        //
};

} // end namespace scene
} // end namespace irr

#endif

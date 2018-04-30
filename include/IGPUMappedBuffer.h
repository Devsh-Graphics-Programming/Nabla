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
class IGPUMappedBuffer : public video::IGPUBuffer
{
    public:
		//! Gets internal pointer.
        /**
		It's best you use a GPU Fence to ensure any operations which you've queued up that are writing to this buffer or reading from it have completed before you start using this pointer.
		Otherwise you will have a race condition.
		WARNING: RESIZE will invalidate pointer!
        WARNING: NEED TO FENCE BEFORE USE!
		@returns Internal pointer. */
        virtual void* getPointer() = 0;

		//! @returns Always true.
		/**
		We only support persistently mapped buffers with ARB_buffer_storage.
		It's almost the fastest across all cards, and is more in line with what Vulkan has.
		Please don't ask us to support Buffer Orphaning, or Map/Unmap.
		*/
        virtual const bool isMappedBuffer() const = 0;
    private:
        //
};

} // end namespace scene
} // end namespace irr

#endif
// documented by Krzysztof Szenk on 12-02-2018

#ifndef __NBL_I_GPU_SEMAPHORE_H_INCLUDED__
#define __NBL_I_GPU_SEMAPHORE_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class NBL_API IGPUSemaphore : public core::IReferenceCounted, public IBackendObject
{
    protected:
        IGPUSemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)) {}

        virtual ~IGPUSemaphore() = default;

        // OpenGL: core::smart_refctd_ptr<COpenGLSync>*
        // Vulkan: const VkSemaphore*
        virtual void* getNativeHandle() = 0;
};

}

#endif
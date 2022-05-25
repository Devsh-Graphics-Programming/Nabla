#ifndef __NBL_I_OPENGL_MEMORY_ALLOCATION_H_INCLUDED__

#include "nbl/video/IDeviceMemoryAllocation.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class IOpenGLMemoryAllocation : public IDeviceMemoryAllocation
{
    using base_t = IDeviceMemoryAllocation;
public:
    IOpenGLMemoryAllocation(const ILogicalDevice* dev)
        : IDeviceMemoryAllocation(dev, IDeviceMemoryAllocation::EMAF_NONE, IDeviceMemoryAllocation::EMPF_NONE), initialized(false)
    {}

    ~IOpenGLMemoryAllocation() = default;
    
    //! Whether the allocation was made for a specific resource and is supposed to only be bound to that resource.
    bool isDedicated() const override { return true; }
    
    //! Strictly prohibited to use buffers and images in openGl when the memory is not initialized (no descriptorset writing, imageview creation, transfer, ...) 
    //! Important to check
    bool isInitialized() const { return initialized; }

    virtual bool initMemory(
        IOpenGL_FunctionTable* gl,
        core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
        core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags)
    {
        setCachedAllocateInfo(allocateFlags, memoryPropertyFlags);
        initialized = true;
        return true;
    }

    //! Allocations happen after buffer/image creation requests based on memory requirements and since COpenGLImage/Buffer are memory themselves, these parameters can't be passed during IOpenGLMemoryAllocation construction
    void setCachedAllocateInfo(
        core::bitflag<E_MEMORY_ALLOCATE_FLAGS> allocateFlags,
        core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags)
    {
        base_t::allocateFlags = allocateFlags;
        base_t::memoryPropertyFlags = memoryPropertyFlags;
    }

protected:
    bool initialized = false;
};

}

#define __NBL_I_OPENGL_MEMORY_ALLOCATION_H_INCLUDED__
#endif

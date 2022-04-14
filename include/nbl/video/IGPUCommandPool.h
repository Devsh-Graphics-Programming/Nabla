#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUCommandPool : public core::IReferenceCounted, public IBackendObject
{
    public:
        class ICommand;
        class CBeginRenderPassCmd;
        class CEndRenderPassCmd;

        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_NONE = 0x00,
            ECF_TRANSIENT_BIT = 0x01,
            ECF_RESET_COMMAND_BUFFER_BIT = 0x02,
            ECF_PROTECTED_BIT = 0x04
        };

        IGPUCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<E_CREATE_FLAGS> _flags, uint32_t _familyIx) : IBackendObject(std::move(dev)), m_flags(_flags), m_familyIx(_familyIx) {}

        core::bitflag<E_CREATE_FLAGS> getCreationFlags() const { return m_flags; }
        uint32_t getQueueFamilyIndex() const { return m_familyIx; }

        // OpenGL: nullptr, because commandpool doesn't exist in GL (we might expose the internal allocator in the future)
        // Vulkan: const VkCommandPool*
        virtual const void* getNativeHandle() const = 0;

    protected:
        virtual ~IGPUCommandPool() = default;

        core::bitflag<E_CREATE_FLAGS> m_flags;
        uint32_t m_familyIx;
};

class alignas(64u) IGPUCommandPool::ICommand
{
public:
    virtual ~ICommand() {}

    // static void* operator new(std::size_t size) = delete;
    static void* operator new[](std::size_t size) = delete;
    // static void* operator new(std::size_t size, std::align_val_t al) = delete;
    static void* operator new[](std::size_t size, std::align_val_t al) = delete;

    // static void operator delete  (void* ptr) = delete;
    static void operator delete[](void* ptr) = delete;
    static void operator delete  (void* ptr, std::align_val_t al) = delete;
    static void operator delete[](void* ptr, std::align_val_t al) = delete;
    static void operator delete  (void* ptr, std::size_t sz) = delete;
    static void operator delete[](void* ptr, std::size_t sz) = delete;
    static void operator delete  (void* ptr, std::size_t sz, std::align_val_t al) = delete;
    static void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) = delete;

    uint32_t m_size;
protected:
    ICommand(uint32_t size) : m_size(size) {}
};

class IGPUCommandPool::CBeginRenderPassCmd : public IGPUCommandPool::ICommand
{
public:
    CBeginRenderPassCmd(const core::smart_refctd_ptr<const video::IGPURenderpass>&& renderpass, const core::smart_refctd_ptr<const video::IGPUFramebuffer>&& framebuffer)
        : ICommand(calc_size(core::smart_refctd_ptr(renderpass), core::smart_refctd_ptr(framebuffer))), m_renderpass(std::move(renderpass)), m_framebuffer(std::move(framebuffer))
    {}

    static uint32_t calc_size(const core::smart_refctd_ptr<const video::IGPURenderpass>& renderpass, const core::smart_refctd_ptr<const video::IGPUFramebuffer>& framebuffer)
    {
        return core::alignUp(2ull*sizeof(void*), alignof(CBeginRenderPassCmd));
    }

private:
    core::smart_refctd_ptr<const video::IGPURenderpass> m_renderpass;
    core::smart_refctd_ptr<const video::IGPUFramebuffer> m_framebuffer;
};

class IGPUCommandPool::CEndRenderPassCmd : public IGPUCommandPool::ICommand
{
    // no params
};

}


#endif
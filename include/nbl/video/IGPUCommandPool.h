#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"
#include "nbl/core/containers/CMemoryPool.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class IGPUCommandPool : public core::IReferenceCounted, public IBackendObject
{
public:
        static inline constexpr uint32_t COMMAND_SEGMENT_SIZE = 128u << 10u;

private:
        static inline constexpr uint32_t COMMAND_ALIGNMENT = 64u;

        static inline constexpr uint32_t COMMAND_SEGMENT_ALIGNMENT = 64u;

        static inline constexpr uint32_t MAX_COMMAND_SEGMENT_BLOCK_COUNT = 3u;// 16u;
        static inline constexpr uint32_t COMMAND_SEGMENTS_PER_BLOCK = 2u; // 256u;
        static inline constexpr uint32_t MIN_POOL_ALLOC_SIZE = COMMAND_SEGMENT_SIZE;

    public:
        class ICommand;
        class CommandSegment;
        class CBeginRenderPassCmd;
        class CEndRenderPassCmd;

        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_NONE = 0x00,
            ECF_TRANSIENT_BIT = 0x01,
            ECF_RESET_COMMAND_BUFFER_BIT = 0x02,
            ECF_PROTECTED_BIT = 0x04
        };

        IGPUCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<E_CREATE_FLAGS> _flags, uint32_t _familyIx)
            : IBackendObject(std::move(dev)), m_commandSegmentPool(COMMAND_SEGMENTS_PER_BLOCK* COMMAND_SEGMENT_SIZE, 0u, MAX_COMMAND_SEGMENT_BLOCK_COUNT, MIN_POOL_ALLOC_SIZE),
            m_flags(_flags), m_familyIx(_familyIx)
        {}

        core::bitflag<E_CREATE_FLAGS> getCreationFlags() const { return m_flags; }
        uint32_t getQueueFamilyIndex() const { return m_familyIx; }

        // OpenGL: nullptr, because commandpool doesn't exist in GL (we might expose the internal allocator in the future)
        // Vulkan: const VkCommandPool*
        virtual const void* getNativeHandle() const = 0;

        core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>, core::default_aligned_allocator, false, uint32_t> m_commandSegmentPool;

    protected:
        virtual ~IGPUCommandPool() = default;

        core::bitflag<E_CREATE_FLAGS> m_flags;
        uint32_t m_familyIx;
};

class alignas(COMMAND_ALIGNMENT) IGPUCommandPool::ICommand
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

class alignas(COMMAND_SEGMENT_ALIGNMENT) IGPUCommandPool::CommandSegment
{
public:
    struct Iterator;

    struct params_t
    {
        core::LinearAddressAllocator<uint32_t> m_commandAllocator;
        CommandSegment* m_next;
    } params;

    static inline constexpr uint32_t STORAGE_SIZE = COMMAND_SEGMENT_SIZE - sizeof(params_t);
    alignas(ICommand) uint8_t m_data[STORAGE_SIZE];

    CommandSegment()
    {
        params.m_commandAllocator = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, COMMAND_SEGMENT_ALIGNMENT, COMMAND_SEGMENT_SIZE);
        params.m_next = nullptr;

        wipeNextCommandSize();
    }

    template <typename Cmd, typename... Args>
    Cmd* allocate(const Args&... args)
    {
        const uint32_t cmdSize = Cmd::calc_size(args...);
        const auto address = params.m_commandAllocator.alloc_addr(cmdSize, alignof(Cmd));
        if (address == decltype(params.m_commandAllocator)::invalid_address)
            return nullptr;

        wipeNextCommandSize();

        void* cmdMem = m_data + address;
        return new (cmdMem) Cmd(std::forward<Args>(args)...);
    }

private:
    void wipeNextCommandSize()
    {
        const auto cursor = params.m_commandAllocator.get_allocated_size();
        const uint32_t wipeSize = offsetof(IGPUCommandPool::ICommand, m_size) + sizeof(IGPUCommandPool::ICommand::m_size);
        if (cursor + wipeSize < params.m_commandAllocator.get_total_size())
            memset(m_data + cursor, 0, wipeSize);
    }
};

struct IGPUCommandPool::CommandSegment::Iterator
{
    CommandSegment* m_segment = nullptr;
    ICommand* m_cmd = nullptr;
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

protected:
    core::smart_refctd_ptr<const video::IGPURenderpass> m_renderpass;
    core::smart_refctd_ptr<const video::IGPUFramebuffer> m_framebuffer;
};

class IGPUCommandPool::CEndRenderPassCmd : public IGPUCommandPool::ICommand
{
public:
    CEndRenderPassCmd() : ICommand(calc_size()) {}

    static uint32_t calc_size()
    {
        return core::alignUp(1u, alignof(CEndRenderPassCmd));
    }
};

}


#endif
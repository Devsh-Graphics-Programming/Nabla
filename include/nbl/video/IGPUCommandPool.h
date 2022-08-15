#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"
#include "nbl/core/containers/CMemoryPool.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class NBL_API IGPUCommandPool : public core::IReferenceCounted, public IBackendObject
{
public:
    static inline constexpr uint32_t COMMAND_SEGMENT_SIZE = 128u << 10u;

    enum E_CREATE_FLAGS : uint32_t
    {
        ECF_NONE = 0x00,
        ECF_TRANSIENT_BIT = 0x01,
        ECF_RESET_COMMAND_BUFFER_BIT = 0x02,
        ECF_PROTECTED_BIT = 0x04
    };

private:
    static inline constexpr uint32_t COMMAND_ALIGNMENT = 64u;

    static inline constexpr uint32_t COMMAND_SEGMENT_ALIGNMENT = 64u;

    static inline constexpr uint32_t MAX_COMMAND_SEGMENT_BLOCK_COUNT = 3u;// 16u;
    static inline constexpr uint32_t COMMAND_SEGMENTS_PER_BLOCK = 2u; // 256u;
    static inline constexpr uint32_t MIN_POOL_ALLOC_SIZE = COMMAND_SEGMENT_SIZE;

public:
    class alignas(COMMAND_ALIGNMENT) ICommand
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

    template <class CRTP>
    class NBL_FORCE_EBO IFixedSizeCommand : public IGPUCommandPool::ICommand
    {
    public:
        template <typename... Args>
        static uint32_t calc_size(const Args&...)
        {
            return core::alignUp(sizeof(CRTP), alignof(CRTP));
        }

    protected:
        IFixedSizeCommand() : ICommand(calc_size()) {}
    };

    class alignas(COMMAND_SEGMENT_ALIGNMENT) CommandSegment
    {
    public:
        struct Iterator
        {
            CommandSegment* m_segment = nullptr;
            ICommand* m_cmd = nullptr;
        };

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
        // Cmd* allocate(const Args&... args)
        Cmd* allocate(Args&&... args)
        {
            const uint32_t cmdSize = Cmd::calc_size(args...);
            const auto address = params.m_commandAllocator.alloc_addr(cmdSize, alignof(Cmd));
            if (address == decltype(params.m_commandAllocator)::invalid_address)
                return nullptr;

            wipeNextCommandSize();

            void* cmdMem = m_data + address;
            return new (cmdMem) Cmd(args...);
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

    class CBindIndexBufferCmd;

    class CDrawIndirectCmd;
    class CDrawIndexedIndirectCmd;
    class CDrawIndirectCountCmd;
    class CDrawIndexedIndirectCountCmd;

    class CBeginRenderPassCmd;

    IGPUCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<E_CREATE_FLAGS> _flags, uint32_t _familyIx)
        : IBackendObject(std::move(dev)), m_commandSegmentPool(COMMAND_SEGMENTS_PER_BLOCK* COMMAND_SEGMENT_SIZE, 0u, MAX_COMMAND_SEGMENT_BLOCK_COUNT, MIN_POOL_ALLOC_SIZE),
        m_flags(_flags), m_familyIx(_familyIx)
    {}

    inline core::bitflag<E_CREATE_FLAGS> getCreationFlags() const { return m_flags; }
    inline uint32_t getQueueFamilyIndex() const { return m_familyIx; }

    // OpenGL: nullptr, because commandpool doesn't exist in GL (we might expose the internal allocator in the future)
    // Vulkan: const VkCommandPool*
    virtual const void* getNativeHandle() const = 0;

    template <typename Cmd, typename... Args>
    Cmd* emplace(CommandSegment::Iterator& segmentListHeadItr, CommandSegment*& segmentListTail, Args&&... args)
    {
        if (segmentListTail == nullptr)
        {
            void* cmdSegmentMem = m_commandSegmentPool.allocate(COMMAND_SEGMENT_SIZE, alignof(CommandSegment));
            if (!cmdSegmentMem)
                return nullptr;

            segmentListTail = new (cmdSegmentMem) CommandSegment;
            segmentListHeadItr.m_segment = segmentListTail;
        }

        // Cmd* cmd = m_segmentListTail->allocate<Cmd, Args...>(args...);
        Cmd* cmd = segmentListTail->allocate<Cmd>(std::forward<Args>(args)...);
        if (!cmd)
        {
            void* nextSegmentMem = m_commandSegmentPool.allocate(COMMAND_SEGMENT_SIZE, alignof(CommandSegment));
            if (nextSegmentMem == nullptr)
                return nullptr;

            CommandSegment* nextSegment = new (nextSegmentMem) CommandSegment;

            // cmd = m_segmentListTail->allocate<Cmd, Args...>(args...);
            cmd = segmentListTail->allocate<Cmd>(std::forward<Args>(args)...);
            if (!cmd)
                return nullptr;

            segmentListTail->params.m_next = nextSegment;
            segmentListTail = segmentListTail->params.m_next;
        }

        if (segmentListHeadItr.m_cmd == nullptr)
            segmentListHeadItr.m_cmd = cmd;

        return cmd;
    }

    void deleteCommandSegmentList(CommandSegment::Iterator& segmentListHeadItr, CommandSegment*& segmentListTail)
    {
        auto& itr = segmentListHeadItr;

        if (itr.m_segment && itr.m_cmd)
        {
            bool lastCmd = itr.m_cmd->m_size == 0u;
            while (!lastCmd)
            {
                ICommand* currCmd = itr.m_cmd;
                CommandSegment* currSegment = itr.m_segment;

                itr.m_cmd = reinterpret_cast<ICommand*>(reinterpret_cast<uint8_t*>(itr.m_cmd) + currCmd->m_size);
                currCmd->~ICommand();
                // No need to deallocate currCmd because it has been allocated from the LinearAddressAllocator where deallocate is a No-OP and the memory will
                // get reclaimed in ~LinearAddressAllocator

                if ((reinterpret_cast<uint8_t*>(itr.m_cmd) - reinterpret_cast<uint8_t*>(itr.m_segment)) > CommandSegment::STORAGE_SIZE)
                {
                    CommandSegment* nextSegment = currSegment->params.m_next;
                    if (!nextSegment)
                        break;

                    currSegment->~CommandSegment();
                    m_commandSegmentPool.deallocate(currSegment, COMMAND_SEGMENT_SIZE);

                    itr.m_segment = nextSegment;
                    itr.m_cmd = reinterpret_cast<ICommand*>(itr.m_segment->m_data);
                }

                lastCmd = itr.m_cmd->m_size == 0u;
                if (lastCmd)
                {
                    currSegment->~CommandSegment();
                    m_commandSegmentPool.deallocate(currSegment, COMMAND_SEGMENT_SIZE);
                }
            }

            segmentListHeadItr.m_cmd = nullptr;
            segmentListHeadItr.m_segment = nullptr;
            segmentListTail = nullptr;
        }
    }

    bool reset()
    {
        m_resetCount.fetch_add(1);
        // TODO(achal): Reset the entire m_commandSegmentPool without keeping track of the command buffer's heads.
        return reset_impl();
    }

    inline uint32_t getResetCounter() { return m_resetCount.load(); }

protected:
    virtual ~IGPUCommandPool() = default;

    virtual bool reset_impl() { return true; };

    core::bitflag<E_CREATE_FLAGS> m_flags;
    uint32_t m_familyIx;

private:
    std::atomic_uint32_t m_resetCount = 0;
    core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>, core::default_aligned_allocator, false, uint32_t> m_commandSegmentPool;
};

class IGPUCommandPool::CBindIndexBufferCmd : public IGPUCommandPool::IFixedSizeCommand<CBindIndexBufferCmd>
{
public:
    CBindIndexBufferCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& indexBuffer) : m_indexBuffer(indexBuffer) {}

protected:
    core::smart_refctd_ptr<const video::IGPUBuffer> m_indexBuffer;
};

class IGPUCommandPool::CDrawIndirectCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndirectCmd>
{
public:
    CDrawIndirectCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer) : m_buffer(buffer) {}

protected:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CDrawIndexedIndirectCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndexedIndirectCmd>
{
public:
    CDrawIndexedIndirectCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer) : m_buffer(buffer) {}

protected:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CDrawIndirectCountCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndirectCountCmd>
{
public:
    CDrawIndirectCountCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const core::smart_refctd_ptr<const video::IGPUBuffer>& countBuffer)
        : m_buffer(buffer) , m_countBuffer(countBuffer)
    {}

protected:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
    core::smart_refctd_ptr<const IGPUBuffer> m_countBuffer;
};

class IGPUCommandPool::CDrawIndexedIndirectCountCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndexedIndirectCountCmd>
{
public:
    CDrawIndexedIndirectCountCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const core::smart_refctd_ptr<const video::IGPUBuffer>& countBuffer)
        : m_buffer(buffer), m_countBuffer(countBuffer)
    {}

protected:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
    core::smart_refctd_ptr<const IGPUBuffer> m_countBuffer;
};

class IGPUCommandPool::CBeginRenderPassCmd : public IGPUCommandPool::IFixedSizeCommand<CBeginRenderPassCmd>
{
public:
    CBeginRenderPassCmd(const core::smart_refctd_ptr<const video::IGPURenderpass>& renderpass, const core::smart_refctd_ptr<const video::IGPUFramebuffer>& framebuffer)
        : m_renderpass(renderpass), m_framebuffer(framebuffer)
    {}

protected:
    core::smart_refctd_ptr<const video::IGPURenderpass> m_renderpass;
    core::smart_refctd_ptr<const video::IGPUFramebuffer> m_framebuffer;
};

}


#endif
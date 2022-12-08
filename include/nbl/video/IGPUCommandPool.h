#ifndef __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__
#define __NBL_I_GPU_COMMAND_POOL_H_INCLUDED__


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"
#include "nbl/core/containers/CMemoryPool.h"

#include "nbl/video/decl/IBackendObject.h"
#include "nbl/video/IGPUPipelineLayout.h"

namespace nbl::video
{
class IGPUCommandBuffer;

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

    static inline constexpr uint32_t MAX_COMMAND_SEGMENT_BLOCK_COUNT = 16u;
    static inline constexpr uint32_t COMMAND_SEGMENTS_PER_BLOCK = 256u;
    static inline constexpr uint32_t MIN_POOL_ALLOC_SIZE = COMMAND_SEGMENT_SIZE;

public:
    class CCommandSegment;

    class alignas(COMMAND_ALIGNMENT) ICommand
    {
        friend class CCommandSegment;

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

        inline uint32_t getSize() const { return m_size; }

    protected:
        ICommand(uint32_t size) : m_size(size) {}

    private:
        friend CCommandSegment;

        uint32_t m_size;
    };

    template <class CRTP>
    class NBL_FORCE_EBO IFixedSizeCommand : public IGPUCommandPool::ICommand
    {
    public:
        template <typename... Args>
        static uint32_t calc_size(const Args&...)
        {
            return sizeof(CRTP);
        }

    protected:
        IFixedSizeCommand() : ICommand(calc_size()) {}
    };

    class alignas(COMMAND_SEGMENT_ALIGNMENT) CCommandSegment
    {
        struct header_t
        {
            core::LinearAddressAllocator<uint32_t> commandAllocator;
            CCommandSegment* next;
        } m_header;

    public:
        static inline constexpr uint32_t STORAGE_SIZE = COMMAND_SEGMENT_SIZE - core::roundUp(sizeof(header_t), alignof(ICommand));

        struct Iterator
        {
            ICommand* cmd = nullptr;
            CCommandSegment* segment = nullptr;

            inline bool operator!=(const Iterator& other) const { return cmd != other.cmd; }

            inline Iterator operator++(int)
            {
                Iterator old = *this;
                if (cmd)
                {
                    assert(segment);

                    // If `cmd` was "one past the end" then the size was wiped to 0 and it won't move.
                    cmd = reinterpret_cast<ICommand*>(reinterpret_cast<uint8_t*>(cmd) + cmd->getSize());

                    if (cmd == segment->getLastCommand())
                        *this = segment->end();
                }
                return old;
            }

            inline ICommand* operator*() { return cmd; }
        };

        CCommandSegment()
        {
            m_header.commandAllocator = core::LinearAddressAllocator<uint32_t>(nullptr, 0u, 0u, alignof(ICommand), STORAGE_SIZE);
            m_header.next = nullptr;

            wipeNextCommandSize();
        }

        template <typename Cmd, typename... Args>
        void* allocate(const Args&... args)
        {
            const uint32_t cmdSize = Cmd::calc_size(args...);
            const auto address = m_header.commandAllocator.alloc_addr(cmdSize, alignof(Cmd));
            if (address == decltype(m_header.commandAllocator)::invalid_address)
                return nullptr;

            wipeNextCommandSize();

            void* cmdMem = m_data + address;
            return cmdMem;
        }

        inline void setNext(CCommandSegment* segment) { m_header.next = segment; }
        inline CCommandSegment* getNext() const { return m_header.next; }
        inline ICommand* getFirstCommand() { return reinterpret_cast<ICommand*>(m_data); } // TODO(achal): Probably remove?
        inline uint8_t* getData() { return m_data; }

        inline CCommandSegment::Iterator begin()
        {
            ICommand* cmd = reinterpret_cast<ICommand*>(m_data);
            return { cmd, this };
        }

        inline CCommandSegment::Iterator end()
        {
            if (m_header.next)
            {
                return m_header.next->begin();
            }
            else
            {
                return { getLastCommand(), this};
            }
        }

        inline ICommand* getLastCommand()
        {
            return reinterpret_cast<ICommand*>(m_data + m_header.commandAllocator.get_allocated_size());
        }

    private:
        alignas(ICommand) uint8_t m_data[STORAGE_SIZE];

        void wipeNextCommandSize()
        {
            const auto nextCmdOffset = m_header.commandAllocator.get_allocated_size();
            const auto wipeEnd = nextCmdOffset + offsetof(IGPUCommandPool::ICommand, m_size) + sizeof(IGPUCommandPool::ICommand::m_size);
            if (wipeEnd < m_header.commandAllocator.get_total_size())
                reinterpret_cast<ICommand*>(m_data + nextCmdOffset)->m_size = 0;
        }
    };
    static_assert(sizeof(CCommandSegment) == COMMAND_SEGMENT_SIZE);

    class CBeginCmd;
    class CBindIndexBufferCmd;
    class CDrawIndirectCmd;
    class CDrawIndexedIndirectCmd;
    class CDrawIndirectCountCmd;
    class CDrawIndexedIndirectCountCmd;
    class CBeginRenderPassCmd;
    class CPipelineBarrierCmd;
    class CBindDescriptorSetsCmd;
    class CBindComputePipelineCmd;
    class CUpdateBufferCmd;
    class CResetQueryPoolCmd;
    class CWriteTimestampCmd;
    class CBeginQueryCmd;
    class CEndQueryCmd;
    class CCopyQueryPoolResultsCmd;
    class CBindGraphicsPipelineCmd;
    class CPushConstantsCmd;
    class CBindVertexBuffersCmd;
    class CCopyBufferCmd;
    class CCopyBufferToImageCmd;
    class CBlitImageCmd;
    class CCopyImageToBufferCmd;
    class CExecuteCommandsCmd;
    class CDispatchIndirectCmd;
    class CWaitEventsCmd;
    class CCopyImageCmd;
    class CResolveImageCmd;
    class CClearColorImageCmd;
    class CClearDepthStencilImageCmd;
    class CFillBufferCmd;
    class CSetEventCmd;
    class CResetEventCmd;
    class CWriteAccelerationStructurePropertiesCmd;
    class CBuildAccelerationStructuresCmd; // for both vkCmdBuildAccelerationStructuresKHR and vkCmdBuildAccelerationStructuresIndirectKHR
    class CCopyAccelerationStructureCmd;
    class CCopyAccelerationStructureToOrFromMemoryCmd; // for both vkCmdCopyAccelerationStructureToMemoryKHR and vkCmdCopyMemoryToAccelerationStructureKHR

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
    Cmd* emplace(CCommandSegment::Iterator& segmentListHeadItr, CCommandSegment*& segmentListTail, Args&&... args)
    {
        if (segmentListTail == nullptr)
        {
            void* cmdSegmentMem = m_commandSegmentPool.allocate(COMMAND_SEGMENT_SIZE, alignof(CCommandSegment));
            if (!cmdSegmentMem)
            {
                assert(false);
                return nullptr;
            }

            segmentListTail = new (cmdSegmentMem) CCommandSegment;
            segmentListHeadItr.segment = segmentListTail;
        }

        void* cmdMem = segmentListTail->allocate<Cmd>(args...);
        if (!cmdMem)
        {
            void* nextSegmentMem = m_commandSegmentPool.allocate(COMMAND_SEGMENT_SIZE, alignof(CCommandSegment));
            if (nextSegmentMem == nullptr)
            {
                assert(false);
                return nullptr;
            }

            CCommandSegment* nextSegment = new (nextSegmentMem) CCommandSegment;

            cmdMem = nextSegment->allocate<Cmd>(args...);
            if (!cmdMem)
            {
                assert(false);
                return nullptr;
            }

            segmentListTail->setNext(nextSegment);
            segmentListTail = segmentListTail->getNext();
        }
        Cmd* cmd = new (cmdMem) Cmd(std::forward<Args>(args)...);

        if (segmentListHeadItr.cmd == nullptr)
            segmentListHeadItr.cmd = cmd;

        return cmd;
    }

    void deleteCommandSegmentList(CCommandSegment::Iterator& segmentListHeadItr, CCommandSegment*& segmentListTail)
    {
        if (segmentListTail)
        {
            assert(segmentListHeadItr.cmd && segmentListHeadItr.segment);

            auto freeSegment = [this](CCommandSegment* segment)
            {
                segment->~CCommandSegment();
                m_commandSegmentPool.deallocate(segment, COMMAND_SEGMENT_SIZE);
            };

            CCommandSegment* prevSegment = nullptr;
            for (auto& itr = segmentListHeadItr; itr != segmentListTail->end();)
            {
                prevSegment = itr.segment;
                auto prevCmd = *(itr++);

                prevCmd->~ICommand();

                // No need to deallocate prevCmd because:
                // - it has been allocated from the LinearAddressAllocator where deallocate is a No-OP
                // - the memory will get reclaimed in `~LinearAddressAllocator` when whole segment is destroyed

                if (itr.segment == prevSegment)
                    continue;

                // Segment has changed, we need to free the previous one.
                freeSegment(prevSegment);
            }

            // The last segment is always freed outside the loop.
            assert(segmentListTail==prevSegment);
            freeSegment(segmentListTail);

            segmentListHeadItr.cmd = nullptr;
            segmentListHeadItr.segment = nullptr;
            segmentListTail = nullptr;
        }
        else
        {
            assert(!segmentListHeadItr.segment && !segmentListHeadItr.cmd);
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
    core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, core::default_aligned_allocator, false, uint32_t> m_commandSegmentPool;
};

class IGPUCommandPool::CBeginCmd : public IGPUCommandPool::IFixedSizeCommand<CBeginCmd>
{
public:
    CBeginCmd(core::smart_refctd_ptr<const IGPURenderpass>&& renderpass, core::smart_refctd_ptr<const IGPUFramebuffer>&& framebuffer) : m_renderpass(std::move(renderpass)), m_framebuffer(std::move(framebuffer)) {}

private:
    core::smart_refctd_ptr<const IGPURenderpass> m_renderpass;
    core::smart_refctd_ptr<const IGPUFramebuffer> m_framebuffer;
};

class IGPUCommandPool::CBindIndexBufferCmd : public IGPUCommandPool::IFixedSizeCommand<CBindIndexBufferCmd>
{
public:
    CBindIndexBufferCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& indexBuffer) : m_indexBuffer(std::move(indexBuffer)) {}

private:
    core::smart_refctd_ptr<const video::IGPUBuffer> m_indexBuffer;
};

class IGPUCommandPool::CDrawIndirectCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndirectCmd>
{
public:
    CDrawIndirectCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CDrawIndexedIndirectCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndexedIndirectCmd>
{
public:
    CDrawIndexedIndirectCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CDrawIndirectCountCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndirectCountCmd>
{
public:
    CDrawIndirectCountCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer, core::smart_refctd_ptr<const video::IGPUBuffer>&& countBuffer)
        : m_buffer(std::move(buffer)) , m_countBuffer(std::move(countBuffer))
    {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
    core::smart_refctd_ptr<const IGPUBuffer> m_countBuffer;
};

class IGPUCommandPool::CDrawIndexedIndirectCountCmd : public IGPUCommandPool::IFixedSizeCommand<CDrawIndexedIndirectCountCmd>
{
public:
    CDrawIndexedIndirectCountCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer, core::smart_refctd_ptr<const video::IGPUBuffer>&& countBuffer)
        : m_buffer(std::move(buffer)), m_countBuffer(std::move(countBuffer))
    {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
    core::smart_refctd_ptr<const IGPUBuffer> m_countBuffer;
};

class IGPUCommandPool::CBeginRenderPassCmd : public IGPUCommandPool::IFixedSizeCommand<CBeginRenderPassCmd>
{
public:
    CBeginRenderPassCmd(core::smart_refctd_ptr<const video::IGPURenderpass>&& renderpass, core::smart_refctd_ptr<const video::IGPUFramebuffer>&& framebuffer)
        : m_renderpass(std::move(renderpass)), m_framebuffer(std::move(framebuffer))
    {}

private:
    core::smart_refctd_ptr<const video::IGPURenderpass> m_renderpass;
    core::smart_refctd_ptr<const video::IGPUFramebuffer> m_framebuffer;
};

class IGPUCommandPool::CPipelineBarrierCmd : public IGPUCommandPool::ICommand
{
public:
    CPipelineBarrierCmd(const uint32_t bufferCount, const core::smart_refctd_ptr<const IGPUBuffer>* buffers, const uint32_t imageCount, const core::smart_refctd_ptr<const IGPUImage>* images)
        : ICommand(calc_size(bufferCount, buffers, imageCount, images)), m_resourceCount(bufferCount+imageCount)
    {
        m_barrierResources = reinterpret_cast<core::smart_refctd_ptr<const core::IReferenceCounted>*>(reinterpret_cast<uint8_t*>(this) + sizeof(CPipelineBarrierCmd));
        std::uninitialized_default_construct_n(m_barrierResources, m_resourceCount);

        uint32_t k = 0;

        for (auto i = 0; i < bufferCount; ++i)
            m_barrierResources[k++] = buffers[i];

        for (auto i = 0; i < imageCount; ++i)
            m_barrierResources[k++] = images[i];
    }

    ~CPipelineBarrierCmd()
    {
        for (auto i = 0; i < m_resourceCount; ++i)
            m_barrierResources[i].~smart_refctd_ptr();
    }

    static uint32_t calc_size(const uint32_t bufferCount, const core::smart_refctd_ptr<const IGPUBuffer>* buffers, const uint32_t imageCount, const core::smart_refctd_ptr<const IGPUImage>* images)
    {
        return core::alignUp(sizeof(CPipelineBarrierCmd) + (bufferCount+imageCount)*sizeof(core::smart_refctd_ptr<const core::IReferenceCounted>), alignof(CPipelineBarrierCmd));
    }

private:
    const uint32_t m_resourceCount;
    core::smart_refctd_ptr<const core::IReferenceCounted>* m_barrierResources;
};

class IGPUCommandPool::CBindDescriptorSetsCmd : public IGPUCommandPool::IFixedSizeCommand<CBindDescriptorSetsCmd>
{
public:
    CBindDescriptorSetsCmd(core::smart_refctd_ptr<const IGPUPipelineLayout>&& pipelineLayout, const uint32_t setCount, const core::smart_refctd_ptr<const IGPUDescriptorSet>* sets)
        : m_layout(std::move(pipelineLayout))
    {
        for (auto i = 0; i < setCount; ++i)
            m_sets[i] = sets[i];
    }

private:
    core::smart_refctd_ptr<const IGPUPipelineLayout> m_layout;
    core::smart_refctd_ptr<const IGPUDescriptorSet> m_sets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
};

class IGPUCommandPool::CBindComputePipelineCmd : public IGPUCommandPool::IFixedSizeCommand<CBindComputePipelineCmd>
{
public:
    CBindComputePipelineCmd(core::smart_refctd_ptr<const IGPUComputePipeline>&& pipeline) : m_pipeline(std::move(pipeline)) {}

private:
    core::smart_refctd_ptr<const IGPUComputePipeline> m_pipeline;
};

class IGPUCommandPool::CUpdateBufferCmd : public IGPUCommandPool::IFixedSizeCommand<CUpdateBufferCmd>
{
public:
    CUpdateBufferCmd(core::smart_refctd_ptr<const IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CResetQueryPoolCmd : public IGPUCommandPool::IFixedSizeCommand<CResetQueryPoolCmd>
{
public:
    CResetQueryPoolCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

private:
    core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CWriteTimestampCmd : public IGPUCommandPool::IFixedSizeCommand<CWriteTimestampCmd>
{
public:
    CWriteTimestampCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

private:
    core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CBeginQueryCmd : public IGPUCommandPool::IFixedSizeCommand<CBeginQueryCmd>
{
public:
    CBeginQueryCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

private:
    core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CEndQueryCmd : public IGPUCommandPool::IFixedSizeCommand<CEndQueryCmd>
{
public:
    CEndQueryCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

private:
    core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CCopyQueryPoolResultsCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyQueryPoolResultsCmd>
{
public:
    CCopyQueryPoolResultsCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool, core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer)
        : m_queryPool(std::move(queryPool)), m_dstBuffer(std::move(dstBuffer))
    {}

private:
    core::smart_refctd_ptr<const IQueryPool> m_queryPool;
    core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CBindGraphicsPipelineCmd : public IGPUCommandPool::IFixedSizeCommand<CBindGraphicsPipelineCmd>
{
public:
    CBindGraphicsPipelineCmd(core::smart_refctd_ptr<const IGPUGraphicsPipeline>&& pipeline) : m_pipeline(std::move(pipeline)) {}

private:
    core::smart_refctd_ptr<const IGPUGraphicsPipeline> m_pipeline;
};

class IGPUCommandPool::CPushConstantsCmd : public IGPUCommandPool::IFixedSizeCommand<CPushConstantsCmd>
{
public:
    CPushConstantsCmd(core::smart_refctd_ptr<const IGPUPipelineLayout>&& layout) : m_layout(std::move(layout)) {}

private:
    core::smart_refctd_ptr<const IGPUPipelineLayout> m_layout;
};

class IGPUCommandPool::CBindVertexBuffersCmd : public IGPUCommandPool::IFixedSizeCommand<CBindVertexBuffersCmd>
{
    static inline constexpr auto MaxBufferCount = asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT;

public:
    CBindVertexBuffersCmd(const uint32_t first, const uint32_t count, const IGPUBuffer *const *const buffers)
    {
        for (auto i = first; i < count; ++i)
        {
            assert(i < MaxBufferCount);
            m_buffers[i] = core::smart_refctd_ptr<const IGPUBuffer>(buffers[i]);
        }
    }

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffers[MaxBufferCount];
};

class IGPUCommandPool::CCopyBufferCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyBufferCmd>
{
public:
    CCopyBufferCmd(core::smart_refctd_ptr<const IGPUBuffer>&& srcBuffer, core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer)
        : m_srcBuffer(std::move(srcBuffer)), m_dstBuffer(std::move(dstBuffer))
    {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_srcBuffer;
    core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CCopyBufferToImageCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyBufferToImageCmd>
{
public:
    CCopyBufferToImageCmd(core::smart_refctd_ptr<const IGPUBuffer>&& srcBuffer, core::smart_refctd_ptr<const IGPUImage>&& dstImage)
        : m_srcBuffer(std::move(srcBuffer)), m_dstImage(std::move(dstImage))
    {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_srcBuffer;
    core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CBlitImageCmd : public IGPUCommandPool::IFixedSizeCommand<CBlitImageCmd>
{
public:
    CBlitImageCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUImage>&& dstImage)
        : m_srcImage(std::move(srcImage)), m_dstImage(std::move(dstImage))
    {}

private:
    core::smart_refctd_ptr<const IGPUImage> m_srcImage;
    core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CCopyImageToBufferCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyImageToBufferCmd>
{
public:
    CCopyImageToBufferCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer)
        : m_srcImage(std::move(srcImage)), m_dstBuffer(std::move(dstBuffer))
    {}

private:
    core::smart_refctd_ptr<const IGPUImage> m_srcImage;
    core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CExecuteCommandsCmd : public IGPUCommandPool::ICommand
{
public:
    CExecuteCommandsCmd(const uint32_t count, IGPUCommandBuffer* const* const commandBuffers) : ICommand(calc_size(count, commandBuffers)), m_count(count)
    {
        auto dataPtr = reinterpret_cast<core::smart_refctd_ptr<const IGPUCommandBuffer>*>(reinterpret_cast<uint8_t*>(this) + sizeof(CExecuteCommandsCmd));
        m_commandBuffers = new (dataPtr) core::smart_refctd_ptr<const IGPUCommandBuffer>[count];
        for (auto i = 0; i < m_count; ++i)
            m_commandBuffers[i] = core::smart_refctd_ptr<const IGPUCommandBuffer>(commandBuffers[i]);
    }

    ~CExecuteCommandsCmd()
    {
        for (auto i = 0; i < m_count; ++i)
            m_commandBuffers->~smart_refctd_ptr();
    }

    static uint32_t calc_size(const uint32_t count, IGPUCommandBuffer* const* const commandBuffers)
    {
        return core::alignUp(sizeof(CExecuteCommandsCmd) + count*sizeof(core::smart_refctd_ptr<const IGPUCommandBuffer>), alignof(CExecuteCommandsCmd));
    }

private:
    const uint32_t m_count;
    core::smart_refctd_ptr<const IGPUCommandBuffer>* m_commandBuffers;
};

class IGPUCommandPool::CDispatchIndirectCmd : public IGPUCommandPool::IFixedSizeCommand<CDispatchIndirectCmd>
{
public:
    CDispatchIndirectCmd(core::smart_refctd_ptr<const IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CWaitEventsCmd : public IGPUCommandPool::ICommand
{
public:
    CWaitEventsCmd(const uint32_t bufferCount, const IGPUBuffer *const *const buffers, const uint32_t imageCount, const IGPUImage *const *const images, const uint32_t eventCount, IGPUEvent *const *const events)
        : ICommand(calc_size(bufferCount, buffers, imageCount, images, eventCount, events)), m_resourceCount(bufferCount + imageCount + eventCount)
    {
        auto dataPtr = reinterpret_cast<core::smart_refctd_ptr<const IReferenceCounted>*>(reinterpret_cast<uint8_t*>(this) + sizeof(CWaitEventsCmd));
        m_resources = new (dataPtr) core::smart_refctd_ptr<const IReferenceCounted>[m_resourceCount];

        uint32_t k = 0u;
        for (auto i = 0; i < bufferCount; ++i)
            m_resources[k++] = core::smart_refctd_ptr<const IReferenceCounted>(buffers[i]);

        for (auto i = 0; i < imageCount; ++i)
            m_resources[k++] = core::smart_refctd_ptr<const IReferenceCounted>(images[i]);

        for (auto i = 0; i < eventCount; ++i)
            m_resources[k++] = core::smart_refctd_ptr<const IReferenceCounted>(events[i]);
    }

    ~CWaitEventsCmd()
    {
        for (auto i = 0; i < m_resourceCount; ++i)
            m_resources[i].~smart_refctd_ptr();
    }

    static uint32_t calc_size(const uint32_t bufferCount, const IGPUBuffer *const *const, const uint32_t imageCount, const IGPUImage *const *const, const uint32_t eventCount, IGPUEvent *const *const)
    {
        const uint32_t resourceCount = bufferCount + imageCount + eventCount;
        return core::alignUp(sizeof(CWaitEventsCmd) + resourceCount * sizeof(core::smart_refctd_ptr<const IReferenceCounted>), alignof(CWaitEventsCmd));
    }

private:
    const uint32_t m_resourceCount;
    core::smart_refctd_ptr<const IReferenceCounted>* m_resources;
};

class IGPUCommandPool::CCopyImageCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyImageCmd>
{
public:
    CCopyImageCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUImage>&& dstImage) : m_srcImage(std::move(srcImage)), m_dstImage(std::move(dstImage)) {}

private:
    core::smart_refctd_ptr<const IGPUImage> m_srcImage;
    core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CResolveImageCmd : public IGPUCommandPool::IFixedSizeCommand<CResolveImageCmd>
{
public:
    CResolveImageCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUImage>&& dstImage) : m_srcImage(std::move(srcImage)), m_dstImage(std::move(dstImage)) {}

private:
    core::smart_refctd_ptr<const IGPUImage> m_srcImage;
    core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CClearColorImageCmd : public IGPUCommandPool::IFixedSizeCommand<CClearColorImageCmd>
{
public:
    CClearColorImageCmd(core::smart_refctd_ptr<const IGPUImage>&& image) : m_image(std::move(image)) {}

private:
    core::smart_refctd_ptr<const IGPUImage> m_image;
};

class IGPUCommandPool::CClearDepthStencilImageCmd : public IGPUCommandPool::IFixedSizeCommand<CClearDepthStencilImageCmd>
{
public:
    CClearDepthStencilImageCmd(core::smart_refctd_ptr<const IGPUImage>&& image) : m_image(std::move(image)) {}

private:
    core::smart_refctd_ptr<const IGPUImage> m_image;
};

class IGPUCommandPool::CFillBufferCmd : public IGPUCommandPool::IFixedSizeCommand<CFillBufferCmd>
{
public:
    CFillBufferCmd(core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer) : m_dstBuffer(std::move(dstBuffer)) {}

private:
    core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CSetEventCmd : public IGPUCommandPool::IFixedSizeCommand<CSetEventCmd>
{
public:
    CSetEventCmd(core::smart_refctd_ptr<const IGPUEvent>&& _event) : m_event(std::move(_event)) {}

private:
    core::smart_refctd_ptr<const IGPUEvent> m_event;
};

class IGPUCommandPool::CResetEventCmd : public IGPUCommandPool::IFixedSizeCommand<CResetEventCmd>
{
public:
    CResetEventCmd(core::smart_refctd_ptr<const IGPUEvent>&& _event) : m_event(std::move(_event)) {}

private:
    core::smart_refctd_ptr<const IGPUEvent> m_event;
};

class IGPUCommandPool::CWriteAccelerationStructurePropertiesCmd : public IGPUCommandPool::ICommand
{
public:
    // If we take queryPool as rvalue ref here (core::smart_refctd_ptr<const IQueryPool>&&), in calc_size it will become const core::smart_refctd_ptr<const IQueryPool>
    // because calc_size takes its arguments by const ref (https://github.com/Devsh-Graphics-Programming/Nabla/blob/04fcae3029772cbc739ccf6ba80f72e6e12f54e8/include/nbl/video/IGPUCommandPool.h#L76)
    // , that means we will not be able to pass a core::smart_refctd_ptr<const IQueryPool> when emplacing the command. So instead, we take a raw pointer and create refctd pointers here.
    CWriteAccelerationStructurePropertiesCmd(const IQueryPool* queryPool, const uint32_t accelerationStructureCount, IGPUAccelerationStructure const *const *const accelerationStructures)
        : ICommand(calc_size(queryPool, accelerationStructureCount, accelerationStructures)), m_queryPool(core::smart_refctd_ptr<const IQueryPool>(queryPool)), m_accelerationStructureCount(accelerationStructureCount)
    {
        auto dataPtr = reinterpret_cast<core::smart_refctd_ptr<const IGPUAccelerationStructure>*>(reinterpret_cast<uint8_t*>(this) + sizeof(CWriteAccelerationStructurePropertiesCmd));
        m_accelerationStructures = new (dataPtr) core::smart_refctd_ptr<const IGPUAccelerationStructure>[m_accelerationStructureCount];

        for (auto i = 0; i < m_accelerationStructureCount; ++i)
            m_accelerationStructures[i] = core::smart_refctd_ptr<const IGPUAccelerationStructure>(accelerationStructures[i]);
    }

    ~CWriteAccelerationStructurePropertiesCmd()
    {
        for (auto i = 0; i < m_accelerationStructureCount; ++i)
            m_accelerationStructures[i].~smart_refctd_ptr();
    }

    static uint32_t calc_size(const IQueryPool* queryPool, const uint32_t accelerationStructureCount, IGPUAccelerationStructure const *const *const accelerationStructures)
    {
        return core::alignUp(sizeof(CWriteAccelerationStructurePropertiesCmd) + (accelerationStructureCount + 1)* sizeof(core::smart_refctd_ptr<const IReferenceCounted>), alignof(CWriteAccelerationStructurePropertiesCmd));
    }

private:
    core::smart_refctd_ptr<const IQueryPool> m_queryPool;
    const uint32_t m_accelerationStructureCount;
    core::smart_refctd_ptr<const IGPUAccelerationStructure>* m_accelerationStructures;
};

class IGPUCommandPool::CBuildAccelerationStructuresCmd : public IGPUCommandPool::ICommand
{
public:
    CBuildAccelerationStructuresCmd(const uint32_t accelerationStructureCount, core::smart_refctd_ptr<const IGPUAccelerationStructure>* accelerationStructures, const uint32_t bufferCount, core::smart_refctd_ptr<const IGPUBuffer>* buffers)
        : ICommand(calc_size(accelerationStructureCount, accelerationStructures, bufferCount, buffers)), m_resourceCount(accelerationStructureCount + bufferCount)
    {
        auto dataPtr = reinterpret_cast<core::smart_refctd_ptr<const IReferenceCounted>*>(reinterpret_cast<uint8_t*>(this) + sizeof(CBuildAccelerationStructuresCmd));
        m_resources = new (dataPtr) core::smart_refctd_ptr<const IReferenceCounted>[m_resourceCount];

        uint32_t k = 0u;
        for (auto i = 0; i < accelerationStructureCount; ++i)
            m_resources[k++] = core::smart_refctd_ptr<const IReferenceCounted>(accelerationStructures[i]);

        for (auto i = 0; i < bufferCount; ++i)
            m_resources[k++] = core::smart_refctd_ptr<const IReferenceCounted>(buffers[i]);
    }

    ~CBuildAccelerationStructuresCmd()
    {
        for (auto i = 0; i < m_resourceCount; ++i)
            m_resources[i].~smart_refctd_ptr();
    }

    static uint32_t calc_size(const uint32_t accelerationStructureCount, core::smart_refctd_ptr<const IGPUAccelerationStructure>* accelerationStructures, const uint32_t bufferCount, core::smart_refctd_ptr<const IGPUBuffer>* buffers)
    {
        const auto resourceCount = accelerationStructureCount + bufferCount;
        return core::alignUp(sizeof(CBuildAccelerationStructuresCmd) + resourceCount * sizeof(core::smart_refctd_ptr<const IReferenceCounted>), alignof(CBuildAccelerationStructuresCmd));
    }

private:
    const uint32_t m_resourceCount;
    core::smart_refctd_ptr<const IReferenceCounted>* m_resources;
};

class IGPUCommandPool::CCopyAccelerationStructureCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyAccelerationStructureCmd>
{
public:
    CCopyAccelerationStructureCmd(core::smart_refctd_ptr<const IGPUAccelerationStructure>&& src, core::smart_refctd_ptr<const IGPUAccelerationStructure>&& dst)
        : m_src(std::move(src)), m_dst(std::move(dst))
    {}

private:
    core::smart_refctd_ptr<const IGPUAccelerationStructure> m_src;
    core::smart_refctd_ptr<const IGPUAccelerationStructure> m_dst;
};

class IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd : public IGPUCommandPool::IFixedSizeCommand<CCopyAccelerationStructureToOrFromMemoryCmd>
{
public:
    CCopyAccelerationStructureToOrFromMemoryCmd(core::smart_refctd_ptr<const IGPUAccelerationStructure>&& accelStructure, core::smart_refctd_ptr<const IGPUBuffer>&& buffer)
        : m_accelStructure(std::move(accelStructure)), m_buffer(std::move(buffer))
    {}

private:
    core::smart_refctd_ptr<const IGPUAccelerationStructure> m_accelStructure;
    core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

}


#endif
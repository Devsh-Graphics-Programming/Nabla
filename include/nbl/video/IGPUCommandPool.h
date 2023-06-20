#ifndef _NBL_VIDEO_I_GPU_COMMAND_POOL_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_COMMAND_POOL_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/util/bitflag.h"
#include "nbl/core/containers/CMemoryPool.h"

#include "nbl/video/IGPUEvent.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUComputePipeline.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IQueryPool.h"


namespace nbl::video
{

class IGPUCommandBuffer;

class IGPUCommandPool : public core::IReferenceCounted, public IBackendObject
{
        static inline constexpr uint32_t COMMAND_ALIGNMENT = 64u;

        static inline constexpr uint32_t COMMAND_SEGMENT_ALIGNMENT = 64u;
        static inline constexpr uint32_t COMMAND_SEGMENT_SIZE = 128u << 10u;

        static inline constexpr uint32_t MAX_COMMAND_SEGMENT_BLOCK_COUNT = 16u;
        static inline constexpr uint32_t COMMAND_SEGMENTS_PER_BLOCK = 256u;
        static inline constexpr uint32_t MIN_POOL_ALLOC_SIZE = COMMAND_SEGMENT_SIZE;

    public:
        enum class CREATE_FLAGS : uint8_t
        {
            NONE = 0x00,
            TRANSIENT_BIT = 0x01,
            RESET_COMMAND_BUFFER_BIT = 0x02,
            PROTECTED_BIT = 0x04
        };

        class CCommandSegment;
        class alignas(COMMAND_ALIGNMENT) ICommand
        {
                friend class CCommandSegment;

            public:
                virtual ~ICommand() = default;

                // static void* operator new(std::size_t size) = delete;
                static void* operator new[](std::size_t size) = delete;
                // static void* operator new(std::size_t size, std::align_val_t al) = delete;
                static void* operator new[](std::size_t size, std::align_val_t al) = delete;

                static void operator delete[](void* ptr) = delete;
                static void operator delete[](void* ptr, std::align_val_t al) = delete;
                static void operator delete[](void* ptr, std::size_t sz) = delete;
                static void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) = delete;

                inline uint32_t getSize() const { return m_size; }

            protected:
                inline ICommand(uint32_t size) : m_size(size)
                {
                    assert(ptrdiff_t(this) % alignof(ICommand) == 0);
                    assert(m_size % alignof(ICommand) == 0);
                }

                void operator delete(ICommand* ptr, std::destroying_delete_t) { ptr->~ICommand(); }
                void operator delete( ICommand* ptr, std::destroying_delete_t,
                                        std::align_val_t al ) { ptr->~ICommand(); }
                void operator delete( ICommand* ptr, std::destroying_delete_t, std::size_t sz )  { ptr->~ICommand(); }
                void operator delete( ICommand* ptr, std::destroying_delete_t,
                                        std::size_t sz, std::align_val_t al ) { ptr->~ICommand(); }


            private:

                friend CCommandSegment;

                const uint32_t m_size;
        };

        template<class CRTP>
        class NBL_FORCE_EBO IFixedSizeCommand : public ICommand
        {
            public:
                template <typename... Args>
                static uint32_t calc_size(const Args&...)
                {
                    static_assert(std::is_final_v<CRTP>);
                    return sizeof(CRTP);
                }

                virtual ~IFixedSizeCommand() = default;

            protected:
                inline IFixedSizeCommand() : ICommand(calc_size()) {}
        };
        template<class CRTP>
        class NBL_FORCE_EBO IVariableSizeCommand : public ICommand
        {
            public:
                template <typename... Args>
                static uint32_t calc_size(const Args&... args)
                {
                    static_assert(std::is_final_v<CRTP>);
                    return core::alignUp(sizeof(CRTP)+CRTP::calc_resources(args...)*sizeof(core::smart_refctd_ptr<const IReferenceCounted>),alignof(CRTP));
                }

                virtual ~IVariableSizeCommand()
                {
                    std::destroy_n(getVariableCountResources(),m_resourceCount);
                }
                inline core::smart_refctd_ptr<const IReferenceCounted>* getVariableCountResources() { return reinterpret_cast<core::smart_refctd_ptr<const core::IReferenceCounted>*>(static_cast<CRTP*>(this)+1); }

            protected:
                template <typename... Args>
                inline IVariableSizeCommand(const Args&... args) : ICommand(calc_size(args...)), m_resourceCount(CRTP::calc_resources(args...))
                {
                    std::uninitialized_default_construct_n(getVariableCountResources(),m_resourceCount);
                }

                const uint32_t m_resourceCount;
        };

        class alignas(COMMAND_SEGMENT_ALIGNMENT) CCommandSegment
        {
                struct header_t
                {
                    template<typename... Args>
                    inline header_t(Args&&... args) : commandAllocator(std::forward<Args>(args)...) {}

                    core::LinearAddressAllocator<uint32_t> commandAllocator;
                    CCommandSegment* next = nullptr;

                    CCommandSegment* nextHead = nullptr;
                    CCommandSegment* prevHead = nullptr;
                } m_header;

            public:
                static inline constexpr uint32_t STORAGE_SIZE = COMMAND_SEGMENT_SIZE - core::roundUp(sizeof(header_t), alignof(ICommand));

                CCommandSegment(CCommandSegment* prev):
                    m_header(nullptr, 0u, 0u, alignof(ICommand), STORAGE_SIZE)
                {
                    static_assert(alignof(ICommand) == COMMAND_SEGMENT_ALIGNMENT);
                    wipeNextCommandSize();

                    if (prev)
                        prev->m_header.next = this;
                }

                ~CCommandSegment()
                {
                    for (ICommand* cmd = begin(); cmd != end();)
                    {
                        if (cmd->getSize() == 0)
                            break;

                        auto* nextCmd = reinterpret_cast<ICommand*>(reinterpret_cast<uint8_t*>(cmd) + cmd->getSize());
                        cmd->~ICommand();
                        cmd = nextCmd;
                    }
                }

                template <typename Cmd, typename... Args>
                Cmd* allocate(const Args&... args)
                {
                    const uint32_t cmdSize = Cmd::calc_size(args...);
                    const auto address = m_header.commandAllocator.alloc_addr(cmdSize, alignof(Cmd));
                    if (address == decltype(m_header.commandAllocator)::invalid_address)
                        return nullptr;

                    wipeNextCommandSize();

                    auto cmdMem = reinterpret_cast<Cmd*>(m_data + address);
                    return cmdMem;
                }

                inline CCommandSegment* getNext() const { return m_header.next; }
                inline CCommandSegment* getNextHead() const { return m_header.nextHead; }
                inline CCommandSegment* getPrevHead() const { return m_header.prevHead; }

                inline ICommand* begin()
                {
                    return reinterpret_cast<ICommand*>(m_data);
                }

                inline ICommand* end()
                {
                    return reinterpret_cast<ICommand*>(m_data + m_header.commandAllocator.get_allocated_size());
                }

                static void linkHeads(CCommandSegment* prev, CCommandSegment* next)
                {
                    if (prev)
                        prev->m_header.nextHead = next;

                    if (next)
                        next->m_header.prevHead = prev;
                }

            private:
                alignas(ICommand) uint8_t m_data[STORAGE_SIZE];

                void wipeNextCommandSize()
                {
                    const auto nextCmdOffset = m_header.commandAllocator.get_allocated_size();
                    const auto wipeEnd = nextCmdOffset + offsetof(IGPUCommandPool::ICommand, m_size) + sizeof(IGPUCommandPool::ICommand::m_size);
                    if (wipeEnd < m_header.commandAllocator.get_total_size())
                        *(const_cast<uint32_t*>(&(reinterpret_cast<ICommand*>(m_data + nextCmdOffset)->m_size))) = 0;
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

        inline core::bitflag<CREATE_FLAGS> getCreationFlags() const { return m_flags; }
        inline uint32_t getQueueFamilyIndex() const { return m_familyIx; }

        // Vulkan: const VkCommandPool*
        virtual const void* getNativeHandle() const = 0;

        bool reset()
        {
            m_resetCount.fetch_add(1);
            m_commandListPool.clear();
            return reset_impl();
        }

        inline uint32_t getResetCounter() { return m_resetCount.load(); }

    protected:
        IGPUCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<CREATE_FLAGS> _flags, uint32_t _familyIx)
            : IBackendObject(std::move(dev)), m_flags(_flags), m_familyIx(_familyIx)
        {}

        virtual ~IGPUCommandPool() = default;

        virtual bool reset_impl() { return true; };

        core::bitflag<CREATE_FLAGS> m_flags;
        uint32_t m_familyIx;

private:
    std::atomic_uint64_t m_resetCount = 0;

    class CCommandSegmentListPool
    {
    public:
        struct SCommandSegmentList
        {
            CCommandSegment* head = nullptr;
            CCommandSegment* tail = nullptr;
        };

        CCommandSegmentListPool() : m_pool(COMMAND_SEGMENTS_PER_BLOCK*COMMAND_SEGMENT_SIZE, 0u, MAX_COMMAND_SEGMENT_BLOCK_COUNT, MIN_POOL_ALLOC_SIZE) {}

        template <typename Cmd, typename... Args>
        Cmd* emplace(SCommandSegmentList& list, Args&&... args)
        {
            if (!list.tail && !appendToList(list))
                return nullptr;

            // not forwarding twice because newCmd() will never be called the second time
            auto newCmd = [&]() -> Cmd*
            {
                auto cmdMem = list.tail->allocate<Cmd>(args...);
                if (!cmdMem)
                    return nullptr;

                return new (cmdMem) Cmd(std::forward<Args>(args)...);
            };

            auto cmd = newCmd();
            if (!cmd)
            {
                if (!appendToList(list))
                    return nullptr;

                cmd = newCmd();
                if (!cmd)
                {
                    assert(false);
                    return nullptr;
                }
            }

            return cmd;
        }

        // Nullifying the head of the passed segment list is NOT the responsibility of deleteList.
        inline void deleteList(CCommandSegment* head)
        {
            if (!head)
                return;

            if (head == m_head)
                m_head = head->getNextHead();

            CCommandSegment::linkHeads(head->getPrevHead(), head->getNextHead());

            for (auto& segment = head; segment;)
            {
                auto nextSegment = segment->getNext();
                segment->~CCommandSegment();
                m_pool.deallocate(segment, COMMAND_SEGMENT_SIZE);
                segment = nextSegment;
            }
        }

        inline void clear()
        {
            for (auto* currHead = m_head; currHead;)
            {
                auto* nextHead = currHead->getNextHead();
                // We don't (and also can't) nullify the tail here because when the command buffer detects that its parent pool has been resetted
                // it nullifies both head and tail itself.
                deleteList(currHead);
                currHead = nextHead;
            }

            m_head = nullptr;
        }

    private:
        inline bool appendToList(SCommandSegmentList& list)
        {
            auto segment = m_pool.emplace<CCommandSegment>(list.tail);
            if (!segment)
            {
                assert(false);
                return false;
            }

            if (!list.tail)
            {
                assert(!list.head && "List should've been empty.");

                list.head = segment;

                CCommandSegment::linkHeads(segment, m_head);
                m_head = segment;
            }
            list.tail = segment;
            return true;
        }

        CCommandSegment* m_head = nullptr;
        
        template <typename T>
        using pool_alignment = core::aligned_allocator<T,COMMAND_SEGMENT_ALIGNMENT>;
        core::CMemoryPool<core::PoolAddressAllocator<uint32_t>, pool_alignment, false, uint32_t> m_pool;
    };

    friend class IGPUCommandBuffer;
    CCommandSegmentListPool m_commandListPool;
};

class IGPUCommandPool::CBindIndexBufferCmd final : public IFixedSizeCommand<CBindIndexBufferCmd>
{
    public:
        CBindIndexBufferCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& indexBuffer) : m_indexBuffer(std::move(indexBuffer)) {}

    private:
        core::smart_refctd_ptr<const video::IGPUBuffer> m_indexBuffer;
};

class IGPUCommandPool::CDrawIndirectCmd final : public IFixedSizeCommand<CDrawIndirectCmd>
{
    public:
        CDrawIndirectCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CDrawIndexedIndirectCmd final : public IFixedSizeCommand<CDrawIndexedIndirectCmd>
{
    public:
        CDrawIndexedIndirectCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CDrawIndirectCountCmd final : public IFixedSizeCommand<CDrawIndirectCountCmd>
{
    public:
        CDrawIndirectCountCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer, core::smart_refctd_ptr<const video::IGPUBuffer>&& countBuffer)
            : m_buffer(std::move(buffer)) , m_countBuffer(std::move(countBuffer))
        {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
        core::smart_refctd_ptr<const IGPUBuffer> m_countBuffer;
};

class IGPUCommandPool::CDrawIndexedIndirectCountCmd final : public IFixedSizeCommand<CDrawIndexedIndirectCountCmd>
{
    public:
        CDrawIndexedIndirectCountCmd(core::smart_refctd_ptr<const video::IGPUBuffer>&& buffer, core::smart_refctd_ptr<const video::IGPUBuffer>&& countBuffer)
            : m_buffer(std::move(buffer)), m_countBuffer(std::move(countBuffer))
        {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
        core::smart_refctd_ptr<const IGPUBuffer> m_countBuffer;
};

class IGPUCommandPool::CBeginRenderPassCmd final : public IFixedSizeCommand<CBeginRenderPassCmd>
{
    public:
        CBeginRenderPassCmd(core::smart_refctd_ptr<const video::IGPURenderpass>&& renderpass, core::smart_refctd_ptr<const video::IGPUFramebuffer>&& framebuffer)
            : m_renderpass(std::move(renderpass)), m_framebuffer(std::move(framebuffer))
        {}

    private:
        core::smart_refctd_ptr<const video::IGPURenderpass> m_renderpass;
        core::smart_refctd_ptr<const video::IGPUFramebuffer> m_framebuffer;
};

class IGPUCommandPool::CPipelineBarrierCmd final : public IVariableSizeCommand<CPipelineBarrierCmd>
{
    public:
        static uint32_t calc_resources(const uint32_t bufferCount, const uint32_t imageCount)
        {
            return bufferCount+imageCount;
        }
};

class IGPUCommandPool::CBindDescriptorSetsCmd final : public IFixedSizeCommand<CBindDescriptorSetsCmd>
{
    public:
        CBindDescriptorSetsCmd(core::smart_refctd_ptr<const IGPUPipelineLayout>&& pipelineLayout, const uint32_t setCount, const IGPUDescriptorSet* const* const sets)
            : m_layout(std::move(pipelineLayout))
        {
            for (auto i = 0; i < setCount; ++i)
            {
                assert(i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT);
                m_sets[i] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(sets[i]);
            }
        }

    private:
        core::smart_refctd_ptr<const IGPUPipelineLayout> m_layout;
        core::smart_refctd_ptr<const IGPUDescriptorSet> m_sets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
};

class IGPUCommandPool::CBindComputePipelineCmd final : public IFixedSizeCommand<CBindComputePipelineCmd>
{
    public:
        CBindComputePipelineCmd(core::smart_refctd_ptr<const IGPUComputePipeline>&& pipeline) : m_pipeline(std::move(pipeline)) {}

    private:
        core::smart_refctd_ptr<const IGPUComputePipeline> m_pipeline;
};

class IGPUCommandPool::CUpdateBufferCmd final : public IFixedSizeCommand<CUpdateBufferCmd>
{
    public:
        CUpdateBufferCmd(core::smart_refctd_ptr<const IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CResetQueryPoolCmd final : public IFixedSizeCommand<CResetQueryPoolCmd>
{
    public:
        CResetQueryPoolCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

    private:
        core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CWriteTimestampCmd final : public IFixedSizeCommand<CWriteTimestampCmd>
{
    public:
        CWriteTimestampCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

    private:
        core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CBeginQueryCmd final : public IFixedSizeCommand<CBeginQueryCmd>
{
    public:
        CBeginQueryCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

    private:
        core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CEndQueryCmd final : public IFixedSizeCommand<CEndQueryCmd>
{
    public:
        CEndQueryCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool) : m_queryPool(std::move(queryPool)) {}

    private:
        core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CCopyQueryPoolResultsCmd final : public IFixedSizeCommand<CCopyQueryPoolResultsCmd>
{
    public:
        CCopyQueryPoolResultsCmd(core::smart_refctd_ptr<const IQueryPool>&& queryPool, core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer)
            : m_queryPool(std::move(queryPool)), m_dstBuffer(std::move(dstBuffer))
        {}

    private:
        core::smart_refctd_ptr<const IQueryPool> m_queryPool;
        core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CBindGraphicsPipelineCmd final : public IFixedSizeCommand<CBindGraphicsPipelineCmd>
{
    public:
        CBindGraphicsPipelineCmd(core::smart_refctd_ptr<const IGPUGraphicsPipeline>&& pipeline) : m_pipeline(std::move(pipeline)) {}

    private:
        core::smart_refctd_ptr<const IGPUGraphicsPipeline> m_pipeline;
};

class IGPUCommandPool::CPushConstantsCmd final : public IFixedSizeCommand<CPushConstantsCmd>
{
    public:
        CPushConstantsCmd(core::smart_refctd_ptr<const IGPUPipelineLayout>&& layout) : m_layout(std::move(layout)) {}

    private:
        core::smart_refctd_ptr<const IGPUPipelineLayout> m_layout;
};

class IGPUCommandPool::CBindVertexBuffersCmd final : public IFixedSizeCommand<CBindVertexBuffersCmd>
{
        static inline constexpr auto MaxBufferCount = asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT;

    public:
        CBindVertexBuffersCmd(const uint32_t count, const asset::SBufferBinding<const IGPUBuffer>* const pBindings)
        {
            for (auto i=0; i<count; ++i)
                m_buffers[i] = core::smart_refctd_ptr<const IGPUBuffer>(pBindings[i].buffer);
        }

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffers[MaxBufferCount];
};

class IGPUCommandPool::CCopyBufferCmd final : public IFixedSizeCommand<CCopyBufferCmd>
{
    public:
        CCopyBufferCmd(core::smart_refctd_ptr<const IGPUBuffer>&& srcBuffer, core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer)
            : m_srcBuffer(std::move(srcBuffer)), m_dstBuffer(std::move(dstBuffer))
        {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_srcBuffer;
        core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CCopyBufferToImageCmd final : public IFixedSizeCommand<CCopyBufferToImageCmd>
{
    public:
        CCopyBufferToImageCmd(core::smart_refctd_ptr<const IGPUBuffer>&& srcBuffer, core::smart_refctd_ptr<const IGPUImage>&& dstImage)
            : m_srcBuffer(std::move(srcBuffer)), m_dstImage(std::move(dstImage))
        {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_srcBuffer;
        core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CBlitImageCmd final : public IFixedSizeCommand<CBlitImageCmd>
{
    public:
        CBlitImageCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUImage>&& dstImage)
            : m_srcImage(std::move(srcImage)), m_dstImage(std::move(dstImage))
        {}

    private:
        core::smart_refctd_ptr<const IGPUImage> m_srcImage;
        core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CCopyImageToBufferCmd final : public IFixedSizeCommand<CCopyImageToBufferCmd>
{
    public:
        CCopyImageToBufferCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer)
            : m_srcImage(std::move(srcImage)), m_dstBuffer(std::move(dstBuffer))
        {}

    private:
        core::smart_refctd_ptr<const IGPUImage> m_srcImage;
        core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CExecuteCommandsCmd final : public IVariableSizeCommand<CExecuteCommandsCmd>
{
    public:
        static uint32_t calc_resources(const uint32_t count)
        {
            return count;
        }
};

class IGPUCommandPool::CDispatchIndirectCmd final : public IFixedSizeCommand<CDispatchIndirectCmd>
{
    public:
        CDispatchIndirectCmd(core::smart_refctd_ptr<const IGPUBuffer>&& buffer) : m_buffer(std::move(buffer)) {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_buffer;
};

class IGPUCommandPool::CWaitEventsCmd final : public IVariableSizeCommand<CWaitEventsCmd>
{
    public:
        CWaitEventsCmd(const uint32_t eventCount, IGPUEvent *const *const events, const uint32_t totalBufferCount, const uint32_t totalImageCount)
            : IVariableSizeCommand<CWaitEventsCmd>(eventCount,events,totalBufferCount,totalImageCount), m_eventCount(eventCount)
        {
            for (auto i=0u; i<eventCount; ++i)
                getVariableCountResources()[i] = core::smart_refctd_ptr<const IGPUEvent>(events[i]);
        }

        inline auto* getDeviceMemoryBacked() { return reinterpret_cast<core::smart_refctd_ptr<const IDeviceMemoryBacked>*>(getVariableCountResources()+m_eventCount); }

        static uint32_t calc_resources(const uint32_t eventCount, const IGPUEvent *const *const, const uint32_t totalBufferCount, const uint32_t totalImageCount)
        {
            return eventCount+totalBufferCount+totalImageCount;
        }

    private:
        const uint32_t m_eventCount;
};

class IGPUCommandPool::CCopyImageCmd final : public IFixedSizeCommand<CCopyImageCmd>
{
    public:
        CCopyImageCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUImage>&& dstImage) : m_srcImage(std::move(srcImage)), m_dstImage(std::move(dstImage)) {}

    private:
        core::smart_refctd_ptr<const IGPUImage> m_srcImage;
        core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CResolveImageCmd final : public IFixedSizeCommand<CResolveImageCmd>
{
    public:
        CResolveImageCmd(core::smart_refctd_ptr<const IGPUImage>&& srcImage, core::smart_refctd_ptr<const IGPUImage>&& dstImage) : m_srcImage(std::move(srcImage)), m_dstImage(std::move(dstImage)) {}

    private:
        core::smart_refctd_ptr<const IGPUImage> m_srcImage;
        core::smart_refctd_ptr<const IGPUImage> m_dstImage;
};

class IGPUCommandPool::CClearColorImageCmd final : public IFixedSizeCommand<CClearColorImageCmd>
{
    public:
        CClearColorImageCmd(core::smart_refctd_ptr<const IGPUImage>&& image) : m_image(std::move(image)) {}

    private:
        core::smart_refctd_ptr<const IGPUImage> m_image;
};

class IGPUCommandPool::CClearDepthStencilImageCmd final : public IFixedSizeCommand<CClearDepthStencilImageCmd>
{
    public:
        CClearDepthStencilImageCmd(core::smart_refctd_ptr<const IGPUImage>&& image) : m_image(std::move(image)) {}

    private:
        core::smart_refctd_ptr<const IGPUImage> m_image;
};

class IGPUCommandPool::CFillBufferCmd final : public IFixedSizeCommand<CFillBufferCmd>
{
    public:
        CFillBufferCmd(core::smart_refctd_ptr<const IGPUBuffer>&& dstBuffer) : m_dstBuffer(std::move(dstBuffer)) {}

    private:
        core::smart_refctd_ptr<const IGPUBuffer> m_dstBuffer;
};

class IGPUCommandPool::CSetEventCmd final : public IFixedSizeCommand<CSetEventCmd>
{
    public:
        CSetEventCmd(core::smart_refctd_ptr<const IGPUEvent>&& _event) : m_event(std::move(_event)) {}

    private:
        core::smart_refctd_ptr<const IGPUEvent> m_event;
};

class IGPUCommandPool::CResetEventCmd final : public IFixedSizeCommand<CResetEventCmd>
{
    public:
        CResetEventCmd(core::smart_refctd_ptr<const IGPUEvent>&& _event) : m_event(std::move(_event)) {}

    private:
        core::smart_refctd_ptr<const IGPUEvent> m_event;
};

class IGPUCommandPool::CWriteAccelerationStructurePropertiesCmd final : public IVariableSizeCommand<CWriteAccelerationStructurePropertiesCmd>
{
    public:
        // If we take queryPool as rvalue ref here (core::smart_refctd_ptr<const IQueryPool>&&), in calc_size it will become const core::smart_refctd_ptr<const IQueryPool>
        // because calc_size takes its arguments by const ref (https://github.com/Devsh-Graphics-Programming/Nabla/blob/04fcae3029772cbc739ccf6ba80f72e6e12f54e8/include/nbl/video/IGPUCommandPool.h#L76)
        // , that means we will not be able to pass a core::smart_refctd_ptr<const IQueryPool> when emplacing the command. So instead, we take a raw pointer and create refctd pointers here.
        CWriteAccelerationStructurePropertiesCmd(const IQueryPool* queryPool, const uint32_t accelerationStructureCount)
            : IVariableSizeCommand<CWriteAccelerationStructurePropertiesCmd>(queryPool,accelerationStructureCount), m_queryPool(core::smart_refctd_ptr<const IQueryPool>(queryPool))
        {}

        static uint32_t calc_resources(const IQueryPool* queryPool, const uint32_t accelerationStructureCount)
        {
            return accelerationStructureCount;
        }

    private:
        core::smart_refctd_ptr<const IQueryPool> m_queryPool;
};

class IGPUCommandPool::CBuildAccelerationStructuresCmd final : public IVariableSizeCommand<CBuildAccelerationStructuresCmd>
{
    public:
        static inline constexpr uint32_t MaxGeometryPerBuildInfoCount = 64u;

        static inline uint32_t calc_resources(const uint32_t resourceCount)
        {
            return resourceCount;
        }

        inline void fill(const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos)
        {
            auto oit = getVariableCountResources();
            for (auto& info : pInfos)
            {
                *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(info.srcAS);
                *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(info.dstAS);
                *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(info.scratchAddr.buffer);
                for (auto& geometry : info.geometries)
                switch (geometry.type)
                {
                    case IGPUAccelerationStructure::E_GEOM_TYPE::EGT_TRIANGLES:
                        *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(geometry.data.triangles.vertexData.buffer);
                        *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(geometry.data.triangles.indexData.buffer);
                        if (geometry.data.triangles.transformData.isValid())
                            *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(geometry.data.triangles.transformData.buffer);
                        break;
                    case IGPUAccelerationStructure::E_GEOM_TYPE::EGT_AABBS:
                        *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(geometry.data.aabbs.data.buffer);
                        break;
                    case IGPUAccelerationStructure::E_GEOM_TYPE::EGT_INSTANCES:
                        *(oit++) = core::smart_refctd_ptr<const IReferenceCounted>(geometry.data.instances.data.buffer);
                        break;
                }
            }
        }
};

class IGPUCommandPool::CCopyAccelerationStructureCmd final : public IFixedSizeCommand<CCopyAccelerationStructureCmd>
{
    public:
        CCopyAccelerationStructureCmd(core::smart_refctd_ptr<const IGPUAccelerationStructure>&& src, core::smart_refctd_ptr<const IGPUAccelerationStructure>&& dst)
            : m_src(std::move(src)), m_dst(std::move(dst))
        {}

    private:
        core::smart_refctd_ptr<const IGPUAccelerationStructure> m_src;
        core::smart_refctd_ptr<const IGPUAccelerationStructure> m_dst;
};

class IGPUCommandPool::CCopyAccelerationStructureToOrFromMemoryCmd final : public IFixedSizeCommand<CCopyAccelerationStructureToOrFromMemoryCmd>
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
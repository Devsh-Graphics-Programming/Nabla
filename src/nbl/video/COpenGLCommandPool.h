#ifndef __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"
#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"
#include <mutex>

namespace nbl::video
{
class COpenGLCommandBuffer;

class COpenGLCommandPool final : public IGPUCommandPool
{
        constexpr static inline size_t MIN_ALLOC_SZ = 64ull;
        // TODO: there's an optimization possible if we set the block size to 64kb (max cmd upload size) and then:
        // - use Pool Address Allocator within a block
        // - allocate new block whenever we want to update buffer via cmdbuf
        // - set the pool's brick size to the maximum data store by any command possible
        // - when a command needs an unbounded variable length list of arguments, store it via linked list (chunk it)
        //constexpr static inline size_t MAX_COMMAND_STORAGE_SZ = 32ull;
        constexpr static inline size_t BLOCK_SIZE = 1ull<<21u;
        constexpr static inline size_t MAX_BLOCK_COUNT = 256u;

    public:
        class ICommand;

        class CBindIndexBufferCmd;

        class CDrawCmd;
        class CDrawIndexedCmd;

        class CDrawIndirectCommonBase;
        class CDrawIndirectCmd;
        class CDrawIndexedIndirectCmd;
        class CDrawIndirectCountCmd;
        class CDrawIndexedIndirectCountCmd;

        class CBeginRenderPassCmd;
        class CEndRenderPassCmd;

        using IGPUCommandPool::IGPUCommandPool;
        COpenGLCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, core::bitflag<E_CREATE_FLAGS> _flags, uint32_t _familyIx) : IGPUCommandPool(std::move(dev), _flags.value, _familyIx), mempool(BLOCK_SIZE,0u,MAX_BLOCK_COUNT,MIN_ALLOC_SZ) {}

        template <typename T, typename... Args>
        T* emplace_n(uint32_t n, Args&&... args)
        {
            //static_assert(n*sizeof(T)<=MAX_COMMAND_STORAGE_SZ,"Command Data Store Type larger than preset limit!");
            std::unique_lock<std::mutex> lk(mutex);
            return mempool.emplace_n<T>(n, std::forward<Args>(args)...);
        }
        template <typename T>
        void free_n(const T* ptr, uint32_t n)
        {
            //static_assert(n*sizeof(T)<=MAX_COMMAND_STORAGE_SZ,"Command Data Store Type larger than preset limit!");
            std::unique_lock<std::mutex> lk(mutex);
            mempool.free_n<T>(const_cast<T*>(ptr), n);
        }
        
		inline const void* getNativeHandle() const override {return nullptr;}

    private:
        std::mutex mutex;
        core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>,core::default_aligned_allocator,false,uint32_t> mempool;
};

class NBL_FORCE_EBO COpenGLCommandPool::ICommand
{
public:
    virtual void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxLocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) = 0;

    using base_cmd_t = void;

protected:
    template <typename CRTP, typename... Args>
    static uint32_t end_offset(const Args&... args)
    {
        using other_base_t = typename CRTP::base_cmd_t;
        if constexpr (std::is_void_v<other_base_t>)
            return sizeof(CRTP);
        else
            return other_base_t::calc_size(args...) - sizeof(other_base_t) + sizeof(CRTP);
    }
};

class COpenGLCommandPool::CBindIndexBufferCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CBindIndexBufferCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CBindIndexBufferCmd;

    CBindIndexBufferCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& indexBuffer, const size_t offset, const asset::E_INDEX_TYPE indexType)
        : base_cmd_t(indexBuffer), m_offset(offset), m_indexType(indexType)
    {
        m_size = calc_size(indexBuffer, offset, indexType);
    }

    static uint32_t calc_size(const core::smart_refctd_ptr<const video::IGPUBuffer>& indexBuffer, const size_t offset, const asset::E_INDEX_TYPE indexType)
    {
        return core::alignUp(end_offset<CBindIndexBufferCmd>(indexBuffer), alignof(CBindIndexBufferCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    size_t m_offset;
    asset::E_INDEX_TYPE m_indexType;
};

class COpenGLCommandPool::CDrawCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CDrawCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CDrawCmd;

    CDrawCmd(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance)
        : m_vertexCount(vertexCount), m_instanceCount(instanceCount), m_firstVertex(firstVertex), m_firstInstance(firstInstance)
    {
        m_size = calc_size(vertexCount, instanceCount, firstVertex, firstInstance);
    }

    static uint32_t calc_size(const uint32_t vertexCount, const uint32_t instanceCount, const uint32_t firstVertex, const uint32_t firstInstance)
    {
        return core::alignUp(end_offset<CDrawCmd>(), alignof(CDrawCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    uint32_t m_vertexCount;
    uint32_t m_instanceCount;
    uint32_t m_firstVertex;
    uint32_t m_firstInstance;
};

class COpenGLCommandPool::CDrawIndexedCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CDrawIndexedCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CDrawIndexedCmd;

    CDrawIndexedCmd(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance)
        : m_indexCount(indexCount), m_instanceCount(instanceCount), m_firstIndex(firstIndex), m_vertexOffset(vertexOffset), m_firstInstance(firstInstance)
    {
        m_size = calc_size(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }

    static uint32_t calc_size(const uint32_t indexCount, const uint32_t instanceCount, const uint32_t firstIndex, const int32_t vertexOffset, const uint32_t firstInstance)
    {
        return core::alignUp(end_offset<CDrawIndexedCmd>(), alignof(CDrawIndexedCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    uint32_t m_indexCount;
    uint32_t m_instanceCount;
    uint32_t m_firstIndex;
    uint32_t m_vertexOffset;
    uint32_t m_firstInstance;
};

class COpenGLCommandPool::CDrawIndirectCommonBase : public COpenGLCommandPool::ICommand
{
public:
    CDrawIndirectCommonBase(const size_t offset, const uint32_t maxDrawCount, const uint32_t stride) : m_offset(offset), m_maxDrawCount(maxDrawCount), m_stride(stride) {}

protected:
    size_t m_offset;
    uint32_t m_maxDrawCount;
    uint32_t m_stride;
};

class COpenGLCommandPool::CDrawIndirectCmd : public COpenGLCommandPool::CDrawIndirectCommonBase, public IGPUCommandPool::CDrawIndirectCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CDrawIndirectCmd;

    CDrawIndirectCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const uint32_t maxDrawCount, const uint32_t stride)
        : CDrawIndirectCommonBase(offset, maxDrawCount, stride), base_cmd_t(buffer)
    {
        m_size = calc_size(buffer, offset, maxDrawCount, stride);
    }

    static uint32_t calc_size(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const uint32_t maxDrawCount, const uint32_t stride)
    {
        return core::alignUp(end_offset<CDrawIndirectCmd>(buffer), alignof(CDrawIndirectCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;
};

class COpenGLCommandPool::CDrawIndirectCountCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CDrawIndirectCountCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CDrawIndirectCountCmd;

    CDrawIndirectCountCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const core::smart_refctd_ptr<const video::IGPUBuffer>& countBuffer,
        const size_t countBufferOffset, const uint32_t maxDrawCount, const uint32_t stride)
        : base_cmd_t(buffer, countBuffer), m_offset(offset), m_countBufferOffset(countBufferOffset), m_maxDrawCount(maxDrawCount), m_stride(stride)
    {
        m_size = calc_size(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
    }

    static uint32_t calc_size(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const core::smart_refctd_ptr<const video::IGPUBuffer>& countBuffer,
        const size_t countBufferOffset, const  uint32_t maxDrawCount, const uint32_t stride)
    {
        return core::alignUp(end_offset<CDrawIndirectCountCmd>(buffer, countBuffer), alignof(CDrawIndirectCountCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    size_t m_offset;
    size_t m_countBufferOffset;
    uint32_t m_maxDrawCount;
    uint32_t m_stride;
};

class COpenGLCommandPool::CDrawIndexedIndirectCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CDrawIndexedIndirectCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CDrawIndexedIndirectCmd;

    CDrawIndexedIndirectCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const uint32_t maxDrawCount, const uint32_t stride)
        : base_cmd_t(buffer), m_offset(offset), m_maxDrawCount(maxDrawCount), m_stride(stride)
    {
        m_size = calc_size(buffer, offset, maxDrawCount, stride);
    }

    static uint32_t calc_size(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const  uint32_t maxDrawCount, const uint32_t stride)
    {
        return core::alignUp(end_offset<CDrawIndexedIndirectCmd>(buffer), alignof(CDrawIndexedIndirectCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    size_t m_offset;
    size_t m_countBufferOffset;
    uint32_t m_maxDrawCount;
    uint32_t m_stride;
};

class COpenGLCommandPool::CDrawIndexedIndirectCountCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CDrawIndexedIndirectCountCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CDrawIndexedIndirectCountCmd;

    CDrawIndexedIndirectCountCmd(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const core::smart_refctd_ptr<const video::IGPUBuffer>& countBuffer,
        const size_t countBufferOffset, const uint32_t maxDrawCount, const uint32_t stride)
        : base_cmd_t(buffer, countBuffer), m_offset(offset), m_countBufferOffset(countBufferOffset), m_maxDrawCount(maxDrawCount), m_stride(stride)
    {
        m_size = calc_size(buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
    }

    static uint32_t calc_size(const core::smart_refctd_ptr<const video::IGPUBuffer>& buffer, const size_t offset, const core::smart_refctd_ptr<const video::IGPUBuffer>& countBuffer,
        const size_t countBufferOffset, const  uint32_t maxDrawCount, const uint32_t stride)
    {
        return core::alignUp(end_offset<CDrawIndexedIndirectCountCmd>(buffer, countBuffer), alignof(CDrawIndexedIndirectCountCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    size_t m_offset;
    size_t m_countBufferOffset;
    uint32_t m_maxDrawCount;
    uint32_t m_stride;
};

class COpenGLCommandPool::CBeginRenderPassCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CBeginRenderPassCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CBeginRenderPassCmd;

    CBeginRenderPassCmd(const IGPUCommandBuffer::SRenderpassBeginInfo& beginInfo, const asset::E_SUBPASS_CONTENTS subpassContents)
        : base_cmd_t(beginInfo.renderpass, beginInfo.framebuffer), m_renderArea(beginInfo.renderArea), m_clearValueCount(beginInfo.clearValueCount), m_subpassContents(subpassContents)
    {
        m_size = calc_size(beginInfo, subpassContents);
        if (m_clearValueCount > 0u)
            memcpy(m_clearValues, beginInfo.clearValues, m_clearValueCount * sizeof(asset::SClearValue));
    }

    static uint32_t calc_size(const IGPUCommandBuffer::SRenderpassBeginInfo& beginInfo, const asset::E_SUBPASS_CONTENTS subpassContents)
    {
        return core::alignUp(end_offset<CBeginRenderPassCmd>(beginInfo.renderpass, beginInfo.framebuffer), alignof(CBeginRenderPassCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;

private:
    const VkRect2D m_renderArea;
    const uint32_t m_clearValueCount;
    asset::SClearValue m_clearValues[asset::IRenderpass::SCreationParams::MaxColorAttachments];
    const asset::E_SUBPASS_CONTENTS m_subpassContents;
};

class COpenGLCommandPool::CEndRenderPassCmd : public COpenGLCommandPool::ICommand, public IGPUCommandPool::CEndRenderPassCmd
{
public:
    using base_cmd_t = IGPUCommandPool::CEndRenderPassCmd;

    static uint32_t calc_size()
    {
        return core::alignUp(end_offset<CEndRenderPassCmd>(), alignof(CBeginRenderPassCmd));
    }

    void operator() (IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxLocal, uint32_t ctxid, const system::logger_opt_ptr logger, const COpenGLCommandBuffer* executingCmdbuf) override;
};

}

#endif

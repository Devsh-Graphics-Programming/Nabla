#ifndef __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <variant>

#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/video/IGPUMeshBuffer.h"

#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/SOpenGLContextLocalCache.h"
#include "nbl/video/IQueryPool.h"
#include "nbl/video/COpenGLCommandPool.h"

#define NEW_WAY

#ifdef NEW_WAY
#define TODO_CMD __debugbreak()
#else
#define TODO_CMD
#endif

namespace nbl::video
{

namespace impl
{
#define _NBL_COMMAND_TYPES_LIST \
    ECT_BIND_INDEX_BUFFER,\
    ECT_DRAW,\
    ECT_DRAW_INDEXED,\
    ECT_DRAW_INDIRECT,\
    ECT_DRAW_INDEXED_INDIRECT,\
\
    ECT_SET_VIEWPORT,\
\
    ECT_SET_LINE_WIDTH,\
    ECT_SET_DEPTH_BIAS,\
\
    ECT_SET_BLEND_CONSTANTS,\
\
    ECT_COPY_BUFFER,\
    ECT_COPY_IMAGE,\
    ECT_COPY_BUFFER_TO_IMAGE,\
    ECT_COPY_IMAGE_TO_BUFFER,\
    ECT_BLIT_IMAGE,\
    ECT_RESOLVE_IMAGE,\
\
    ECT_BIND_VERTEX_BUFFERS,\
\
    ECT_SET_SCISSORS,\
    ECT_SET_DEPTH_BOUNDS,\
    ECT_SET_STENCIL_COMPARE_MASK,\
    ECT_SET_STENCIL_WRITE_MASK,\
    ECT_SET_STENCIL_REFERENCE,\
\
    ECT_DISPATCH,\
    ECT_DISPATCH_INDIRECT,\
    ECT_DISPATCH_BASE,\
\
    ECT_SET_EVENT,\
    ECT_RESET_EVENT,\
    ECT_WAIT_EVENTS,\
\
    ECT_PIPELINE_BARRIER,\
\
    ECT_BEGIN_RENDERPASS,\
    ECT_NEXT_SUBPASS,\
    ECT_END_RENDERPASS,\
\
    ECT_SET_DEVICE_MASK,\
\
    ECT_BIND_GRAPHICS_PIPELINE,\
    ECT_BIND_COMPUTE_PIPELINE,\
\
    ECT_RESET_QUERY_POOL,\
    ECT_BEGIN_QUERY,\
    ECT_END_QUERY,\
    ECT_COPY_QUERY_POOL_RESULTS,\
    ECT_WRITE_TIMESTAMP,\
\
    ECT_BIND_DESCRIPTOR_SETS,\
\
    ECT_PUSH_CONSTANTS,\
\
    ECT_CLEAR_COLOR_IMAGE,\
    ECT_CLEAR_DEPTH_STENCIL_IMAGE,\
    ECT_CLEAR_ATTACHMENTS,\
    ECT_FILL_BUFFER,\
    ECT_UPDATE_BUFFER,\
    ECT_EXECUTE_COMMANDS,\
    ECT_REGENERATE_MIPMAPS

    enum E_COMMAND_TYPE
    {
        _NBL_COMMAND_TYPES_LIST
    };

    template <E_COMMAND_TYPE ECT>
    struct SCmd_base
    {
        static inline constexpr E_COMMAND_TYPE type = ECT;
    };
    template <E_COMMAND_TYPE ECT>
    struct SCmd : SCmd_base<ECT>
    {

    };
#define _NBL_DEFINE_SCMD_SPEC(ECT) template<> struct SCmd<ECT> : SCmd_base<ECT>
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_INDEX_BUFFER)
    {
        core::smart_refctd_ptr<const IGPUBuffer> buffer;
        size_t offset;
        asset::E_INDEX_TYPE indexType;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DRAW)
    {
        uint32_t vertexCount;
        uint32_t instanceCount;
        uint32_t firstVertex;
        uint32_t firstInstance;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DRAW_INDEXED)
    {
        uint32_t indexCount;
        uint32_t instanceCount;
        uint32_t firstIndex;
        uint32_t vertexOffset;
        uint32_t firstInstance;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DRAW_INDIRECT)
    {
        core::smart_refctd_ptr<const IGPUBuffer> buffer;
        size_t offset;
        core::smart_refctd_ptr<const IGPUBuffer> countBuffer;
        size_t countBufferOffset;
        uint32_t maxDrawCount;
        uint32_t stride;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DRAW_INDEXED_INDIRECT)
    {
        core::smart_refctd_ptr<const IGPUBuffer> buffer;
        size_t offset;
        core::smart_refctd_ptr<const IGPUBuffer> countBuffer;
        size_t countBufferOffset;
        uint32_t maxDrawCount;
        uint32_t stride;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_VIEWPORT)
    {
        uint32_t firstViewport;
        uint32_t viewportCount;
        const asset::SViewport* viewports;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_LINE_WIDTH)
    {
        float lineWidth;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_DEPTH_BIAS)
    {
        float depthBiasConstantFactor;
        float depthBiasClamp;
        float depthBiasSlopeFactor;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_BLEND_CONSTANTS)
    {
        float blendConstants[4];
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_BUFFER)
    {
        core::smart_refctd_ptr<const IGPUBuffer> srcBuffer;
        core::smart_refctd_ptr<IGPUBuffer> dstBuffer;
        uint32_t regionCount;
        const asset::SBufferCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_IMAGE)
    {
        core::smart_refctd_ptr<const IGPUImage> srcImage;
        asset::IImage::E_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::IImage::E_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::IImage::SImageCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_BUFFER_TO_IMAGE)
    {
        core::smart_refctd_ptr<const IGPUBuffer> srcBuffer;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::IImage::E_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::IImage::SBufferCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_IMAGE_TO_BUFFER)
    {
        core::smart_refctd_ptr<const IGPUImage> srcImage;
        asset::IImage::E_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUBuffer> dstBuffer;
        uint32_t regionCount;
        const asset::IImage::SBufferCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BLIT_IMAGE)
    {
        core::smart_refctd_ptr<const IGPUImage> srcImage;
        asset::IImage::E_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::IImage::E_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::SImageBlit* regions;
        asset::ISampler::E_TEXTURE_FILTER filter;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_RESOLVE_IMAGE)
    {
        core::smart_refctd_ptr<const IGPUImage> srcImage;
        asset::IImage::E_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::IImage::E_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::SImageResolve* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_VERTEX_BUFFERS)
    {
        uint32_t first;
        uint32_t count;
        core::smart_refctd_ptr<const IGPUBuffer> buffers[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
        size_t offsets[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_SCISSORS)
    {
        uint32_t firstScissor;
        uint32_t scissorCount;
        VkRect2D* scissors;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_DEPTH_BOUNDS)
    {
        float minDepthBounds;
        float maxDepthBounds;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_STENCIL_COMPARE_MASK)
    {
        asset::E_STENCIL_FACE_FLAGS faceMask;
        uint32_t cmpMask;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_STENCIL_WRITE_MASK)
    {
        asset::E_STENCIL_FACE_FLAGS faceMask;
        uint32_t writeMask;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_STENCIL_REFERENCE)
    {
        asset::E_STENCIL_FACE_FLAGS faceMask;
        uint32_t reference;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DISPATCH)
    {
        uint32_t groupCountX, groupCountY, groupCountZ;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DISPATCH_INDIRECT)
    {
        core::smart_refctd_ptr<const IGPUBuffer> buffer;
        size_t offset;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DISPATCH_BASE)
    {
        uint32_t baseGroupX, baseGroupY, baseGroupZ;
        uint32_t groupCountX, groupCountY, groupCountZ;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_EVENT)
    {
        core::smart_refctd_ptr<IGPUEvent> event;
        GLbitfield barrierBits;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_RESET_EVENT)
    {
        core::smart_refctd_ptr<IGPUEvent> event;
        asset::E_PIPELINE_STAGE_FLAGS stageMask;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_WAIT_EVENTS)
    {
        GLbitfield barrier;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_PIPELINE_BARRIER)
    {
        GLbitfield barrier;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BEGIN_RENDERPASS)
    {
        IGPUCommandBuffer::SRenderpassBeginInfo renderpassBegin;
        asset::E_SUBPASS_CONTENTS content;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_NEXT_SUBPASS)
    {
        asset::E_SUBPASS_CONTENTS contents;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_END_RENDERPASS)
    {
        // no parameters
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_DEVICE_MASK)
    {
        uint32_t deviceMask;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_GRAPHICS_PIPELINE)
    {
        core::smart_refctd_ptr<const IGPUGraphicsPipeline> pipeline;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_COMPUTE_PIPELINE)
    {
        core::smart_refctd_ptr<const IGPUComputePipeline> pipeline;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_RESET_QUERY_POOL)
    {
        core::smart_refctd_ptr<IQueryPool> queryPool;
        uint32_t query;
        uint32_t queryCount;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BEGIN_QUERY)
    {
        core::smart_refctd_ptr<const IQueryPool> queryPool;
        uint32_t query;
        core::bitflag<IQueryPool::E_QUERY_CONTROL_FLAGS> flags;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_END_QUERY)
    {
        core::smart_refctd_ptr<const IQueryPool> queryPool;
        uint32_t query;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_QUERY_POOL_RESULTS)
    {
        core::smart_refctd_ptr<const IQueryPool> queryPool;
        uint32_t firstQuery;
        uint32_t queryCount;
        core::smart_refctd_ptr<const IGPUBuffer> dstBuffer;
        size_t dstOffset;
        size_t stride;
        core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_WRITE_TIMESTAMP)
    {
        core::smart_refctd_ptr<const IQueryPool> queryPool;
        asset::E_PIPELINE_STAGE_FLAGS pipelineStage;
        uint32_t query;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_DESCRIPTOR_SETS)
    {
        asset::E_PIPELINE_BIND_POINT pipelineBindPoint;
        core::smart_refctd_ptr<const IGPUPipelineLayout> layout;
        uint32_t firstSet;
        uint32_t dsCount;
        core::smart_refctd_ptr<const IGPUDescriptorSet> descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
        static inline constexpr uint32_t MaxDynamicOffsets = SOpenGLState::MaxDynamicOffsets*IGPUPipelineLayout::DESCRIPTOR_SET_COUNT;
        uint32_t dynamicOffsets[MaxDynamicOffsets];
        uint32_t dynamicOffsetCount;

        SCmd() = default;
        SCmd<ECT_BIND_DESCRIPTOR_SETS>& operator=(SCmd<ECT_BIND_DESCRIPTOR_SETS>&& rhs)
        {
            pipelineBindPoint = rhs.pipelineBindPoint;
            layout = std::move(rhs.layout);
            firstSet = rhs.firstSet;
            dsCount = rhs.dsCount;
            dynamicOffsetCount = rhs.dynamicOffsetCount;
            std::move(rhs.descriptorSets,rhs.descriptorSets+IGPUPipelineLayout::DESCRIPTOR_SET_COUNT,descriptorSets);
            std::copy_n(rhs.dynamicOffsets,MaxDynamicOffsets,dynamicOffsets);
            return *this;
        }
        SCmd(SCmd<ECT_BIND_DESCRIPTOR_SETS>&& rhs)
        {
            operator=(std::move(rhs));
        }
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_PUSH_CONSTANTS)
    {
        constexpr static inline uint32_t MAX_PUSH_CONSTANT_BYTESIZE = 128u;

        core::smart_refctd_ptr<const IGPUPipelineLayout> layout;
        core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags;
        uint32_t offset;
        uint32_t size;
        uint8_t values[MAX_PUSH_CONSTANT_BYTESIZE];
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_CLEAR_COLOR_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> image;
        asset::IImage::E_LAYOUT imageLayout;
        asset::SClearColorValue color;
        uint32_t rangeCount;
        const asset::IImage::SSubresourceRange* ranges;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_CLEAR_DEPTH_STENCIL_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> image;
        asset::IImage::E_LAYOUT imageLayout;
        asset::SClearDepthStencilValue depthStencil;
        uint32_t rangeCount;
        const asset::IImage::SSubresourceRange* ranges;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_CLEAR_ATTACHMENTS)
    {
        uint32_t attachmentCount;
        const asset::SClearAttachment* attachments;
        uint32_t rectCount;
        const asset::SClearRect* rects;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_FILL_BUFFER)
    {
        core::smart_refctd_ptr<IGPUBuffer> dstBuffer;
        size_t dstOffset;
        size_t size;
        uint32_t data;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_UPDATE_BUFFER)
    {
        core::smart_refctd_ptr<IGPUBuffer> dstBuffer;
        size_t dstOffset;
        size_t dataSize;
        const void* data;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_EXECUTE_COMMANDS)
    {
        core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf; // secondary!!!
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_REGENERATE_MIPMAPS)
    {
        core::smart_refctd_ptr<IGPUImage> imgview;
    };

#undef _NBL_DEFINE_SCMD_SPEC
} //namespace impl

class COpenGLCommandBuffer final : public IGPUCommandBuffer
{
protected:
    void freeSpaceInCmdPool();

    ~COpenGLCommandBuffer();

    template <impl::E_COMMAND_TYPE ECT>
    using SCmd = impl::SCmd<ECT>;

    //NBL_FOREACH(NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR,__VA_ARGS__);
#define _NBL_SCMD_TYPE_FOR_ECT(ECT) SCmd<impl::ECT>,
    struct SCommand
    {
        impl::E_COMMAND_TYPE type;
        std::variant<
            NBL_FOREACH(_NBL_SCMD_TYPE_FOR_ECT, _NBL_COMMAND_TYPES_LIST)
            int
        > variant;

        template <impl::E_COMMAND_TYPE ECT>
        explicit SCommand(SCmd<ECT>&& cmd) : type(ECT), variant(std::move(cmd)) {}

        template <impl::E_COMMAND_TYPE ECT>
        SCmd<ECT>& get() { return std::get<SCmd<ECT>>(variant); }
        template <impl::E_COMMAND_TYPE ECT>
        const SCmd<ECT>& get() const { return std::get<SCmd<ECT>>(variant); }
    };
#undef _NBL_SCMD_TYPE_FOR_ECT
#undef _NBL_COMMAND_TYPES_LIST

    // TODO(achal): Remove.
    static void copyBufferToImage(const SCmd<impl::ECT_COPY_BUFFER_TO_IMAGE>& c, IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger);

    // TODO(achal): Remove.
    static void copyImageToBuffer(const SCmd<impl::ECT_COPY_IMAGE_TO_BUFFER>& c, IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const system::logger_opt_ptr logger);


    static void clearAttachments(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t count, const asset::SClearAttachment* attachments);

    static bool pushConstants_validate(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values);

    static void blit(IOpenGL_FunctionTable* gl, GLuint src, GLuint dst, const asset::VkOffset3D srcOffsets[2], const asset::VkOffset3D dstOffsets[2], asset::ISampler::E_TEXTURE_FILTER filter);

    static inline GLbitfield barriersToMemBarrierBits(
        const SOpenGLBarrierHelper& helper,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers
    )
    {
        const GLbitfield bufferBits = helper.AllBarrierBits^SOpenGLBarrierHelper::ImageTransferBits;
        constexpr GLbitfield imageBits = SOpenGLBarrierHelper::ImageDescriptorAccessBits|SOpenGLBarrierHelper::ImageTransferBits;

        // ignoring source access flags
        GLbitfield bufaccess = 0u;
        for (uint32_t i = 0u; i < bufferMemoryBarrierCount; ++i)
            bufaccess |= helper.accessFlagsToMemoryBarrierBits(pBufferMemoryBarriers[i].barrier);
        GLbitfield imgaccess = 0u;
        for (uint32_t i = 0u; i < imageMemoryBarrierCount; ++i)
            imgaccess |= helper.accessFlagsToMemoryBarrierBits(pImageMemoryBarriers[i].barrier);
        GLbitfield membarrier = 0u;

        for (uint32_t i = 0u; i < memoryBarrierCount; ++i)
            membarrier |= helper.accessFlagsToMemoryBarrierBits(pMemoryBarriers[i]);

        GLbitfield bufbarrier = bufferBits & bufaccess;
        GLbitfield imgbarrier = imageBits & imgaccess;

        return bufbarrier | imgbarrier | membarrier;
    }

    

    COpenGLCommandPool* getGLCommandPool() const { return static_cast<COpenGLCommandPool*>(m_cmdpool.get()); }

    template <impl::E_COMMAND_TYPE ECT>
    void pushCommand(SCmd<ECT>&& cmd)
    {
        m_commands.emplace_back(std::move(cmd));
    }
    core::vector<SCommand> m_commands; // TODO: embed in the command pool via the use of linked list
    const COpenGLFeatureMap* m_features;
    mutable core::bitflag<IQueryPool::E_QUERY_TYPE> queriesActive;
    mutable std::tuple<IQueryPool const *,uint32_t/*query ix*/,renderpass_t const *,uint32_t/*subpass ix*/> currentlyRecordingQueries[IQueryPool::EQT_COUNT];

public:
    static void beginRenderpass_clearAttachments(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid, const SRenderpassBeginInfo& info, GLuint fbo, const system::logger_opt_ptr logger);
    static bool beginRenderpass_clearAttachments(SOpenGLContextLocalCache* stateCache, const SRenderpassBeginInfo& info, const system::logger_opt_ptr logger, IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment::Iterator& segmentListHeadItr, IGPUCommandPool::CCommandSegment*& segmentListTail, const E_API_TYPE apiType, const COpenGLFeatureMap* features);

    static inline GLenum getGLprimitiveType(asset::E_PRIMITIVE_TOPOLOGY pt)
    {
        using namespace asset;
        switch (pt)
        {
        case EPT_POINT_LIST:
            return GL_POINTS;
        case EPT_LINE_LIST:
            return GL_LINES;
        case EPT_LINE_STRIP:
            return GL_LINE_STRIP;
        case EPT_TRIANGLE_LIST:
            return GL_TRIANGLES;
        case EPT_TRIANGLE_STRIP:
            return GL_TRIANGLE_STRIP;
        case EPT_TRIANGLE_FAN:
            return GL_TRIANGLE_FAN;
        case EPT_LINE_LIST_WITH_ADJACENCY:
            return GL_LINES_ADJACENCY;
        case EPT_LINE_STRIP_WITH_ADJACENCY:
            return GL_LINE_STRIP_ADJACENCY;
        case EPT_TRIANGLE_LIST_WITH_ADJACENCY:
            return GL_TRIANGLES_ADJACENCY;
        case EPT_TRIANGLE_STRIP_WITH_ADJACENCY:
            return GL_TRIANGLE_STRIP_ADJACENCY;
        case EPT_PATCH_LIST:
            return GL_PATCHES;
        default:
            return GL_INVALID_ENUM;
        }
    }

    mutable renderpass_t const * currentlyRecordingRenderPass = nullptr;

    void executeAll(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueLocal, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid) const;

    COpenGLCommandBuffer(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_LEVEL lvl, core::smart_refctd_ptr<IGPUCommandPool>&& _cmdpool, system::logger_opt_smart_ptr&& logger, const COpenGLFeatureMap* _features);

    bool begin_impl(core::bitflag<E_USAGE> flags, const SInheritanceInfo* inheritanceInfo) override final;

    bool end_impl() override final
    {
        assert(queriesActive.value == 0u); // No Queries should be active when command buffer ends
        return true;
    }

    void releaseResourcesBackToPool_impl() override final;

    void bindIndexBuffer_impl(const buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override
    {
        auto* glbuffer = static_cast<const COpenGLBuffer*>(buffer);
        m_stateCache.nextState.vertexInputParams.vaoval.idxBinding = { offset, core::smart_refctd_ptr<const COpenGLBuffer>(glbuffer) };
        m_stateCache.nextState.vertexInputParams.vaoval.idxType = indexType;

        SCmd<impl::ECT_BIND_INDEX_BUFFER> cmd;
        cmd.buffer = core::smart_refctd_ptr<const buffer_t>(buffer);
        cmd.indexType = indexType;
        cmd.offset = offset;
        pushCommand<impl::ECT_BIND_INDEX_BUFFER>(std::move(cmd));
    }

    bool draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override
    {
        if (!m_stateCache.flushStateGraphics(SOpenGLContextLocalCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        {
            assert(false);
            return false;
        }

        const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
        GLenum glpt = getGLprimitiveType(primType);

        if (!m_cmdpool->emplace<COpenGLCommandPool::CDrawArraysInstancedBaseInstanceCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glpt, firstVertex, vertexCount, instanceCount, firstInstance))
            return false;

        SCmd<impl::ECT_DRAW> cmd;
        cmd.vertexCount = vertexCount;
        cmd.instanceCount = instanceCount;
        cmd.firstVertex = firstVertex;
        cmd.firstInstance = firstInstance;
        pushCommand(std::move(cmd));
        return true;
    }
    bool drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override
    {
        if (!m_stateCache.flushStateGraphics(SOpenGLContextLocalCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        {
            assert(false);
            return false;
        }

        const asset::E_PRIMITIVE_TOPOLOGY primType = m_stateCache.currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getPrimitiveAssemblyParams().primitiveType;
        GLenum glpt = getGLprimitiveType(primType);
        GLenum idxType = GL_INVALID_ENUM;
        switch (m_stateCache.currentState.vertexInputParams.vaoval.idxType)
        {
        case asset::EIT_16BIT:
            idxType = GL_UNSIGNED_SHORT;
            break;
        case asset::EIT_32BIT:
            idxType = GL_UNSIGNED_INT;
            break;
        default: break;
        }

        if (idxType != GL_INVALID_ENUM)
        {
            const GLuint64 ixsz = idxType == GL_UNSIGNED_INT ? 4u : 2u;

            GLuint64 idxBufOffset = m_stateCache.currentState.vertexInputParams.vaoval.idxBinding.offset + ixsz * firstIndex;
            static_assert(sizeof(idxBufOffset) == sizeof(void*), "Bad reinterpret_cast");
            if (!m_cmdpool->emplace<COpenGLCommandPool::CDrawElementsInstancedBaseVertexBaseInstanceCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, glpt, indexCount, idxType, idxBufOffset, instanceCount, vertexOffset, firstInstance))
                return false;
        }

        SCmd<impl::ECT_DRAW_INDEXED> cmd;
        cmd.indexCount = indexCount;
        cmd.instanceCount = instanceCount;
        cmd.firstIndex = firstIndex;
        cmd.vertexOffset = vertexOffset;
        cmd.firstInstance = firstInstance;
        pushCommand(std::move(cmd));
        return true;
    }
    bool drawIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override;
    bool drawIndexedIndirect_impl(const buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override;
    bool drawIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override;
    bool drawIndexedIndirectCount_impl(const buffer_t* buffer, size_t offset, const buffer_t* countBuffer, size_t countBufferOffset, uint32_t maxDrawCount, uint32_t stride) override;

    bool setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports) override
    {
        if (viewportCount == 0u)
            return false;

        if (firstViewport >= SOpenGLState::MAX_VIEWPORT_COUNT)
            return false;

        uint32_t count = std::min(viewportCount, SOpenGLState::MAX_VIEWPORT_COUNT);
        if (firstViewport + count > SOpenGLState::MAX_VIEWPORT_COUNT)
            count = SOpenGLState::MAX_VIEWPORT_COUNT - firstViewport;

        uint32_t first = firstViewport;
        for (uint32_t i = 0u; i < count; ++i)
        {
            auto& vp = m_stateCache.nextState.rasterParams.viewport[first + i];
            auto& vpd = m_stateCache.nextState.rasterParams.viewport_depth[first + i];

            vp.x = pViewports[i].x;
            vp.y = pViewports[i].y;
            vp.width = pViewports[i].width;
            vp.height = pViewports[i].height;
            vpd.minDepth = pViewports[i].minDepth;
            vpd.maxDepth = pViewports[i].maxDepth;
        }

        SCmd<impl::ECT_SET_VIEWPORT> cmd;
        cmd.firstViewport = firstViewport;
        cmd.viewportCount = viewportCount;
        auto* viewports = getGLCommandPool()->emplace_n<asset::SViewport>(cmd.viewportCount, pViewports[0]);
        if (!viewports)
            return false;
        for (uint32_t i = 0u; i < viewportCount; ++i)
            viewports[i] = pViewports[i];
        cmd.viewports = viewports;
        pushCommand(std::move(cmd));

        return true;
    }

    bool setLineWidth(float lineWidth) override
    {
        TODO_CMD;

        SCmd<impl::ECT_SET_LINE_WIDTH> cmd;
        cmd.lineWidth = lineWidth;
        pushCommand(std::move(cmd));
        return true;
    }
    bool setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) override
    {
        TODO_CMD;

        SCmd<impl::ECT_SET_DEPTH_BIAS> cmd;
        cmd.depthBiasConstantFactor;
        cmd.depthBiasClamp = depthBiasClamp;
        cmd.depthBiasSlopeFactor = depthBiasSlopeFactor;
        pushCommand(std::move(cmd));
        return true;
    }

    bool setBlendConstants(const float blendConstants[4]) override
    {
        SCmd<impl::ECT_SET_BLEND_CONSTANTS> cmd;
        memcpy(cmd.blendConstants, blendConstants, 4*sizeof(float));
        pushCommand(std::move(cmd));
        return true;
    }

    bool copyBuffer_impl(const buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override;
    
    bool copyImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override
    {
        TODO_CMD;

        if (!this->isCompatibleDevicewise(srcImage))
            return false;
        if (!this->isCompatibleDevicewise(dstImage))
            return false;
        SCmd<impl::ECT_COPY_IMAGE> cmd;
        cmd.srcImage = core::smart_refctd_ptr<const image_t>(srcImage);
        cmd.srcImageLayout = srcImageLayout;
        cmd.dstImage = core::smart_refctd_ptr<image_t>(dstImage);
        cmd.dstImageLayout = dstImageLayout;
        cmd.regionCount = regionCount;
        auto* regions = getGLCommandPool()->emplace_n<asset::IImage::SImageCopy>(regionCount, pRegions[0]);
        if (!regions)
            return false;
        for (uint32_t i = 0u; i < regionCount; ++i)
            regions[i] = pRegions[i];
        cmd.regions = regions;
        pushCommand(std::move(cmd));
        return true;
    }
    bool copyBufferToImage_impl(const buffer_t* srcBuffer, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override;
    bool copyImageToBuffer_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override;
    bool blitImage_impl(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) override;

    bool resolveImage(const image_t* srcImage, asset::IImage::E_LAYOUT srcImageLayout, image_t* dstImage, asset::IImage::E_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override
    {
        TODO_CMD;

        if (!this->isCompatibleDevicewise(srcImage))
            return false;
        SCmd<impl::ECT_RESOLVE_IMAGE> cmd;
        cmd.srcImage = core::smart_refctd_ptr<const image_t>(srcImage);
        cmd.srcImageLayout = srcImageLayout;
        cmd.dstImage = core::smart_refctd_ptr<image_t>(dstImage);
        cmd.dstImageLayout = dstImageLayout;
        cmd.regionCount = regionCount;
        auto* regions = getGLCommandPool()->emplace_n<asset::SImageResolve>(regionCount, pRegions[0]);
        if (!regions)
            return false;
        for (uint32_t i = 0u; i < regionCount; ++i)
            regions[i] = pRegions[i];
        cmd.regions = regions;
        pushCommand(std::move(cmd));
        return true;
    }

    void bindVertexBuffers_impl(uint32_t firstBinding, uint32_t bindingCount, const buffer_t* const* const pBuffers, const size_t* pOffsets) override;

    bool setScissor(uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors) override
    {
        // TODO ?

        SCmd<impl::ECT_SET_SCISSORS> cmd;
        cmd.firstScissor = firstScissor;
        cmd.scissorCount = scissorCount;
        auto* scissors = getGLCommandPool()->emplace_n<VkRect2D>(scissorCount, pScissors[0]);
        if (!scissors)
            return false;
        for (uint32_t i = 0u; i < scissorCount; ++i)
            scissors[i] = pScissors[i];
        cmd.scissors = scissors;
        pushCommand(std::move(cmd));
        return true;
    }
    bool setDepthBounds(float minDepthBounds, float maxDepthBounds) override
    {
        SCmd<impl::ECT_SET_DEPTH_BOUNDS> cmd;
        cmd.minDepthBounds = minDepthBounds;
        cmd.maxDepthBounds = maxDepthBounds;
        pushCommand(std::move(cmd));
        return true;
    }
    bool setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) override
    {
        TODO_CMD;

        SCmd<impl::ECT_SET_STENCIL_COMPARE_MASK> cmd;
        cmd.faceMask = faceMask;
        cmd.cmpMask = compareMask;
        pushCommand(std::move(cmd));
        return true;
    }
    bool setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) override
    {
        TODO_CMD;

        SCmd<impl::ECT_SET_STENCIL_WRITE_MASK> cmd;
        cmd.faceMask = faceMask;
        cmd.writeMask = writeMask;
        pushCommand(std::move(cmd));
        return true;
    }
    bool setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) override
    {
        TODO_CMD;

        SCmd<impl::ECT_SET_STENCIL_REFERENCE> cmd;
        cmd.faceMask = faceMask;
        cmd.reference = reference;
        pushCommand(std::move(cmd));
        return true;
    }

    bool dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        if (!m_stateCache.flushStateCompute(SOpenGLContextLocalCache::GSB_ALL, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, m_features))
            return false;

        if (!m_cmdpool->emplace<COpenGLCommandPool::CDispatchComputeCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, groupCountX, groupCountY, groupCountZ))
            return false;

        SCmd<impl::ECT_DISPATCH> cmd;
        cmd.groupCountX = groupCountX;
        cmd.groupCountY = groupCountY;
        cmd.groupCountZ = groupCountZ;
        pushCommand(std::move(cmd));

        return true;
    }
    bool dispatchIndirect(const buffer_t* buffer, size_t offset) override
    {
        TODO_CMD;

        if (!this->isCompatibleDevicewise(buffer))
            return false;
        SCmd<impl::ECT_DISPATCH_INDIRECT> cmd;
        cmd.buffer = core::smart_refctd_ptr<const buffer_t>(buffer);
        cmd.offset = offset;
        pushCommand(std::move(cmd));
        return true;
    }
    bool dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        SCmd<impl::ECT_DISPATCH_BASE> cmd;
        cmd.baseGroupX = baseGroupX;
        cmd.baseGroupY = baseGroupY;
        cmd.baseGroupZ = baseGroupZ;
        cmd.groupCountX = groupCountX;
        cmd.groupCountY = groupCountY;
        cmd.groupCountZ = groupCountZ;
        pushCommand(std::move(cmd));
        return true;
    }

    bool setEvent(event_t* event, const SDependencyInfo& depInfo) override
    {
        if (!this->isCompatibleDevicewise(event))
            return false;
        SCmd<impl::ECT_SET_EVENT> cmd;
        cmd.event = core::smart_refctd_ptr<event_t>(event);
        cmd.barrierBits = barriersToMemBarrierBits(SOpenGLBarrierHelper(m_features),depInfo.memBarrierCount, depInfo.memBarriers, depInfo.bufBarrierCount, depInfo.bufBarriers, depInfo.imgBarrierCount, depInfo.imgBarriers);
        pushCommand(std::move(cmd));
        return true;
    }
    bool resetEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override
    {
        if (!this->isCompatibleDevicewise(event))
            return false;
        SCmd<impl::ECT_RESET_EVENT> cmd;
        cmd.event = core::smart_refctd_ptr<event_t>(event);
        cmd.stageMask = stageMask;
        pushCommand(std::move(cmd));
        return true;
    }

    bool waitEvents(uint32_t eventCount, event_t*const *const pEvents, const SDependencyInfo* depInfos) override
    {
        TODO_CMD;

        if (eventCount == 0u)
            return false;
        for (uint32_t i = 0u; i < eventCount; ++i)
            if (!this->isCompatibleDevicewise(pEvents[i]))
                return false;
        SCmd<impl::ECT_WAIT_EVENTS> cmd;
        cmd.barrier = 0;
        for (uint32_t i = 0u; i < eventCount; ++i)
        {
            auto& dep = depInfos[i];
            cmd.barrier |= barriersToMemBarrierBits(SOpenGLBarrierHelper(m_features),dep.memBarrierCount, dep.memBarriers, dep.bufBarrierCount, dep.bufBarriers, dep.imgBarrierCount, dep.imgBarriers);
        }
        pushCommand(std::move(cmd));
        return true;
    }

    bool pipelineBarrier_impl(core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        core::bitflag<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override
    {
        const SOpenGLBarrierHelper helper(m_features);
        GLbitfield barrier = helper.pipelineStageFlagsToMemoryBarrierBits(srcStageMask.value, dstStageMask.value);
        barrier &= barriersToMemBarrierBits(helper,memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);

        if (!m_cmdpool->emplace<COpenGLCommandPool::CMemoryBarrierCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, barrier))
            return false;

        return true;
    }

    bool beginRenderPass_impl(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override
    {
        m_stateCache.nextState.framebuffer.hash = static_cast<const COpenGLFramebuffer*>(pRenderPassBegin->framebuffer.get())->getHashValue();
        m_stateCache.nextState.framebuffer.fbo = core::smart_refctd_ptr_static_cast<const COpenGLFramebuffer>(pRenderPassBegin->framebuffer);
        if (!m_stateCache.flushStateGraphics(SOpenGLContextLocalCache::GSB_FRAMEBUFFER, m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        {
            assert(false);
            return false;
        }

        if (!beginRenderpass_clearAttachments(&m_stateCache, *pRenderPassBegin, m_logger.getOptRawPtr(), m_cmdpool.get(), m_GLSegmentListHeadItr, m_GLSegmentListTail, getAPIType(), m_features))
        {
            assert(false);
            return false;
        }

        // This is most likely only required to do some checks for the query pool which can be safely done on the main thread at command record time.
        currentlyRecordingRenderPass = pRenderPassBegin->renderpass.get();

        SCmd<impl::ECT_BEGIN_RENDERPASS> cmd;
        cmd.renderpassBegin = pRenderPassBegin[0];
        if (cmd.renderpassBegin.clearValueCount > 0u)
        {
            auto* clearVals = getGLCommandPool()->emplace_n<asset::SClearValue>(cmd.renderpassBegin.clearValueCount, cmd.renderpassBegin.clearValues[0]);
            memcpy(clearVals, pRenderPassBegin->clearValues, cmd.renderpassBegin.clearValueCount*sizeof(asset::SClearValue));
            cmd.renderpassBegin.clearValues = clearVals;
        }
        cmd.content = content;
        pushCommand(std::move(cmd));
        return true;
    }
    bool nextSubpass(asset::E_SUBPASS_CONTENTS contents) override
    {
        SCmd<impl::ECT_NEXT_SUBPASS> cmd;
        cmd.contents = contents;
        pushCommand(std::move(cmd));
        return true;
    }
    bool endRenderPass() override
    {
        m_stateCache.nextState.framebuffer.hash = SOpenGLState::NULL_FBO_HASH;
        m_stateCache.nextState.framebuffer.GLname = 0u;
        m_stateCache.nextState.framebuffer.fbo = nullptr;

        currentlyRecordingRenderPass = nullptr;

        SCmd<impl::ECT_END_RENDERPASS> cmd;
        pushCommand(std::move(cmd));
        return true;
    }

    bool setDeviceMask(uint32_t deviceMask) override
    { 
        // theres no need to add this command to buffer in GL backend
        assert(false); //make calling this an error
        return IGPUCommandBuffer::setDeviceMask(deviceMask);
    }

    bool bindGraphicsPipeline_impl(const graphics_pipeline_t* pipeline) override;
    void bindComputePipeline_impl(const compute_pipeline_t* pipeline) override;
        
    bool resetQueryPool_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount) override;
    bool beginQuery_impl(IQueryPool* queryPool, uint32_t query, core::bitflag<video::IQueryPool::E_QUERY_CONTROL_FLAGS> flags) override;
    bool endQuery_impl(IQueryPool* queryPool, uint32_t query) override;
    bool copyQueryPoolResults_impl(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, buffer_t* dstBuffer, size_t dstOffset, size_t stride, core::bitflag<video::IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override;
    bool writeTimestamp_impl(asset::E_PIPELINE_STAGE_FLAGS pipelineStage, IQueryPool* queryPool, uint32_t query) override;

    bool writeAccelerationStructureProperties(const core::SRange<IGPUAccelerationStructure>& pAccelerationStructures, IQueryPool::E_QUERY_TYPE queryType, IQueryPool* queryPool, uint32_t firstQuery)
    {
        return false;
    }

    bool bindDescriptorSets_impl(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, const pipeline_layout_t* layout_, uint32_t firstSet_, const uint32_t descriptorSetCount_,
        const descriptor_set_t* const* const descriptorSets_, const uint32_t dynamicOffsetCount_ = 0u, const uint32_t* dynamicOffsets_ = nullptr) override;

    bool pushConstants_impl(const pipeline_layout_t* layout, core::bitflag<asset::IShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override;

    bool clearColorImage(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        TODO_CMD;

        if (!this->isCompatibleDevicewise(image))
            return false;
        SCmd<impl::ECT_CLEAR_COLOR_IMAGE> cmd;
        cmd.image = core::smart_refctd_ptr<image_t>(image);
        cmd.imageLayout = imageLayout;
        cmd.color = pColor[0];
        cmd.rangeCount = rangeCount;
        auto* ranges = getGLCommandPool()->emplace_n<asset::IImage::SSubresourceRange>(rangeCount, pRanges[0]);
        if (!ranges)
            return false;
        for (uint32_t i = 0u; i < rangeCount; ++i)
            ranges[i] = pRanges[i];
        cmd.ranges = ranges;
        pushCommand(std::move(cmd));
        return true;
    }
    bool clearDepthStencilImage(image_t* image, asset::IImage::E_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        TODO_CMD;

        if (!this->isCompatibleDevicewise(image))
            return false;
        SCmd<impl::ECT_CLEAR_DEPTH_STENCIL_IMAGE> cmd;
        cmd.image = core::smart_refctd_ptr<image_t>(image);
        cmd.imageLayout = imageLayout;
        cmd.depthStencil = pDepthStencil[0];
        cmd.rangeCount = rangeCount;
        auto* ranges = getGLCommandPool()->emplace_n<asset::IImage::SSubresourceRange>(rangeCount, pRanges[0]);
        if (!ranges)
            return false;
        for (uint32_t i = 0u; i < rangeCount; ++i)
            ranges[i] = pRanges[i];
        cmd.ranges = ranges;
        pushCommand(std::move(cmd));
        return true;
    }
    bool clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) override
    {
        TODO_CMD;

        if (attachmentCount==0u || rectCount==0u)
            return false;
        SCmd<impl::ECT_CLEAR_ATTACHMENTS> cmd;
        cmd.attachmentCount = attachmentCount;
        auto* attachments = getGLCommandPool()->emplace_n<asset::SClearAttachment>(attachmentCount, pAttachments[0]);
        if (!attachments)
            return false;
        for (uint32_t i = 0u; i < attachmentCount; ++i)
            attachments[i] = pAttachments[i];
        cmd.attachments = attachments;
        cmd.rectCount = rectCount;
        auto* rects = getGLCommandPool()->emplace_n<asset::SClearRect>(rectCount, pRects[0]);
        if (!rects)
            return false;
        for (uint32_t i = 0u; i < rectCount; ++i)
            rects[i] = pRects[i];
        cmd.rects = rects;
        pushCommand(std::move(cmd));
        return true;
    }
    bool fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override
    {
        TODO_CMD;

        if (!this->isCompatibleDevicewise(dstBuffer))
            return false;
        SCmd<impl::ECT_FILL_BUFFER> cmd;
        cmd.dstBuffer = core::smart_refctd_ptr<buffer_t>(dstBuffer);
        cmd.dstOffset = dstOffset;
        cmd.size = size;
        cmd.data = data;
        pushCommand(std::move(cmd));
        return true;
    }
    bool updateBuffer_impl(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override
    {
        GLuint buf = static_cast<const COpenGLBuffer*>(dstBuffer)->getOpenGLName();

        if (!m_cmdpool->emplace<COpenGLCommandPool::CNamedBufferSubDataCmd>(m_GLSegmentListHeadItr, m_GLSegmentListTail, buf, dstOffset, dataSize, pData))
            return false;

        SCmd<impl::ECT_UPDATE_BUFFER> cmd;
        uint8_t* data = getGLCommandPool()->emplace_n<uint8_t>(dataSize);
        if (!data)
            return false;
        memcpy(data, pData, dataSize);
        cmd.dstBuffer = core::smart_refctd_ptr<buffer_t>(dstBuffer);
        cmd.dstOffset = dstOffset;
        cmd.dataSize = dataSize;
        cmd.data = data;
        pushCommand(std::move(cmd));
        return true;
    }
    bool executeCommands_impl(uint32_t count, IGPUCommandBuffer* const* const cmdbufs) override;
    bool regenerateMipmaps(image_t* imgview, uint32_t lastReadyMip, asset::IImage::E_ASPECT_FLAGS aspect) override
    {
        TODO_CMD;

        SCmd<impl::ECT_REGENERATE_MIPMAPS> cmd;
        cmd.imgview = core::smart_refctd_ptr<image_t>(imgview);

        pushCommand(std::move(cmd));

        return true;
    }

    inline const void* getNativeHandle() const override {return nullptr;}

private:
    SOpenGLContextLocalCache m_stateCache;
    IGPUCommandPool::CCommandSegment::Iterator m_GLSegmentListHeadItr = {};
    IGPUCommandPool::CCommandSegment* m_GLSegmentListTail = nullptr;
};

}

#endif

#ifndef __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_BUFFER_H_INCLUDED__

#include <variant>
#include "nbl/video/IGPUCommandBuffer.h"
#include "nbl/core/Types.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/SOpenGLContextLocalCache.h"
#include "nbl/video/IGPUMeshBuffer.h"

namespace nbl {
namespace video
{

class COpenGLCommandBuffer : public virtual IGPUCommandBuffer
{
protected:
    ~COpenGLCommandBuffer() = default;

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
    ECT_EXECUTE_COMMANDS

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
        core::smart_refctd_ptr<IGPUBuffer> buffer;
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
        core::smart_refctd_ptr<IGPUBuffer> buffer;
        size_t offset;
        uint32_t drawCount;
        uint32_t stride;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_DRAW_INDEXED_INDIRECT)
    {
        core::smart_refctd_ptr<IGPUBuffer> buffer;
        size_t offset;
        uint32_t drawCount;
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
        core::smart_refctd_ptr<IGPUBuffer> srcBuffer;
        core::smart_refctd_ptr<IGPUBuffer> dstBuffer;
        uint32_t regionCount;
        const asset::SBufferCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> srcImage;
        asset::E_IMAGE_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::E_IMAGE_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::IImage::SImageCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_BUFFER_TO_IMAGE)
    {
        core::smart_refctd_ptr<IGPUBuffer> srcBuffer;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::E_IMAGE_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::IImage::SBufferCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_IMAGE_TO_BUFFER)
    {
        core::smart_refctd_ptr<IGPUImage> srcImage;
        asset::E_IMAGE_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUBuffer> dstBuffer;
        uint32_t regionCount;
        const asset::IImage::SBufferCopy* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BLIT_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> srcImage;
        asset::E_IMAGE_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::E_IMAGE_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::SImageBlit* regions;
        asset::ISampler::E_TEXTURE_FILTER filter;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_RESOLVE_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> srcImage;
        asset::E_IMAGE_LAYOUT srcImageLayout;
        core::smart_refctd_ptr<IGPUImage> dstImage;
        asset::E_IMAGE_LAYOUT dstImageLayout;
        uint32_t regionCount;
        const asset::SImageResolve* regions;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_VERTEX_BUFFERS)
    {
        uint32_t first;
        uint32_t count;
        core::smart_refctd_ptr<IGPUBuffer> buffers[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
        const size_t* offsets;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_SET_SCISSORS)
    {
        uint32_t firstScissor;
        uint32_t scissorCount;
        const asset::VkRect2D* scissors;
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
        core::smart_refctd_ptr<IGPUBuffer> buffer;
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
        asset::E_PIPELINE_STAGE_FLAGS stageMask;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_RESET_EVENT)
    {
        core::smart_refctd_ptr<IGPUEvent> event;
        asset::E_PIPELINE_STAGE_FLAGS stageMask;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_WAIT_EVENTS)
    {
        uint32_t eventCount;
        core::smart_refctd_ptr<IGPUEvent>* events;
        std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask;
        std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask;
        uint32_t memoryBarrierCount;
        const asset::SMemoryBarrier* pMemoryBarriers;
        uint32_t bufferMemoryBarrierCount;
        const SBufferMemoryBarrier* pBufferMemoryBarriers;
        uint32_t imageMemoryBarrierCount;
        const SImageMemoryBarrier* pImageMemoryBarriers;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_PIPELINE_BARRIER)
    {
        uint32_t eventCount;
        core::smart_refctd_ptr<IGPUEvent>* events;
        std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask;
        std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask;
        std::underlying_type_t<asset::E_DEPENDENCY_FLAGS> dependencyFlags;
        uint32_t memoryBarrierCount;
        const asset::SMemoryBarrier* pMemoryBarriers;
        uint32_t bufferMemoryBarrierCount;
        const SBufferMemoryBarrier* pBufferMemoryBarriers;
        uint32_t imageMemoryBarrierCount;
        const SImageMemoryBarrier* pImageMemoryBarriers;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BEGIN_RENDERPASS)
    {
        SRenderpassBeginInfo renderpassBegin;
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
        core::smart_refctd_ptr<IGPUGraphicsPipeline> pipeline;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_COMPUTE_PIPELINE)
    {
        core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_RESET_QUERY_POOL)
    {
        // TODO
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BEGIN_QUERY)
    {
        // TODO
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_END_QUERY)
    {
        // TODO
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_COPY_QUERY_POOL_RESULTS)
    {
        // TODO
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_WRITE_TIMESTAMP)
    {
        // TODO
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_BIND_DESCRIPTOR_SETS)
    {
        asset::E_PIPELINE_BIND_POINT pipelineBindPoint;
        core::smart_refctd_ptr<IGPUPipelineLayout> layout;
        uint32_t firstSet;
        uint32_t dsCount;
        core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
        core::smart_refctd_dynamic_array<uint32_t> dynamicOffsets;

        SCmd() = default;
        SCmd<ECT_BIND_DESCRIPTOR_SETS>& operator=(SCmd<ECT_BIND_DESCRIPTOR_SETS>&& rhs)
        {
            pipelineBindPoint = rhs.pipelineBindPoint;
            layout = std::move(rhs.layout);
            firstSet = rhs.firstSet;
            dsCount = rhs.dsCount;
            dynamicOffsets = std::move(dynamicOffsets);
            for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
                descriptorSets[i] = std::move(rhs.descriptorSets[i]);
            return *this;
        }
        SCmd(SCmd<ECT_BIND_DESCRIPTOR_SETS>&& rhs)
        {
            operator=(std::move(rhs));
        }
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_PUSH_CONSTANTS)
    {
        core::smart_refctd_ptr<IGPUPipelineLayout> layout;
        std::underlying_type_t<asset::ISpecializedShader::E_SHADER_STAGE> stageFlags;
        uint32_t offset;
        uint32_t size;
        const void* values;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_CLEAR_COLOR_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> image;
        asset::E_IMAGE_LAYOUT imageLayout;
        asset::SClearColorValue color;
        uint32_t rangeCount;
        const asset::IImage::SSubresourceRange* ranges;
    };
    _NBL_DEFINE_SCMD_SPEC(ECT_CLEAR_DEPTH_STENCIL_IMAGE)
    {
        core::smart_refctd_ptr<IGPUImage> image;
        asset::E_IMAGE_LAYOUT imageLayout;
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

#undef _NBL_DEFINE_SCMD_SPEC

    //NBL_FOREACH(NBL_SYSTEM_DECLARE_DYNLIB_FUNCPTR,__VA_ARGS__);
#define _NBL_SCMD_TYPE_FOR_ECT(ECT) SCmd<ECT>,
    struct SCommand
    {
        E_COMMAND_TYPE type;
        std::variant<
            NBL_FOREACH(_NBL_SCMD_TYPE_FOR_ECT, _NBL_COMMAND_TYPES_LIST)
        > variant;

        template <E_COMMAND_TYPE ECT>
        explicit SCommand(SCmd<ECT>&& cmd) : type(ECT), variant(std::move(cmd)) {}

        template <E_COMMAND_TYPE ECT>
        SCmd<ECT>& get() { return std::get<SCmd<ECT>>(variant); }
        template <E_COMMAND_TYPE ECT>
        const SCmd<ECT>& get() const { return std::get<SCmd<ECT>>(variant); }
    };
#undef _NBL_SCMD_TYPE_FOR_ECT
#undef _NBL_COMMAND_TYPES_LIST

    static void copyBufferToImage(const SCmd<ECT_COPY_BUFFER_TO_IMAGE>& c, IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid);

    static void copyImageToBuffer(const SCmd<ECT_COPY_IMAGE_TO_BUFFER>& c, IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid);

    static void beginRenderpass_clearAttachments(IOpenGL_FunctionTable* gl, const SRenderpassBeginInfo& info, GLuint fbo);

    static bool pushConstants_validate(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values);


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

    template <E_COMMAND_TYPE ECT>
    void pushCommand(SCmd<ECT>&& cmd)
    {
        m_commands.emplace_back(std::move(cmd));
    }
    core::vector<SCommand> m_commands;

public:
    void executeAll(IOpenGL_FunctionTable* gl, SOpenGLContextLocalCache* ctxlocal, uint32_t ctxid) const;


    // this init of IGPUCommandBuffer will be always ignored by compiler since COpenGLCommandBuffer will never be most derived class
    COpenGLCommandBuffer() : IGPUCommandBuffer(nullptr) {}


    void bindIndexBuffer(buffer_t* buffer, size_t offset, asset::E_INDEX_TYPE indexType) override
    {
        SCmd<ECT_BIND_INDEX_BUFFER> cmd;
        cmd.buffer = core::smart_refctd_ptr<buffer_t>(buffer);
        cmd.indexType = indexType;
        cmd.offset = offset;
        pushCommand(std::move(cmd));
    }
    void draw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) override
    {
        SCmd<ECT_DRAW> cmd;
        cmd.vertexCount = vertexCount;
        cmd.instanceCount = instanceCount;
        cmd.firstVertex = firstVertex;
        cmd.firstInstance = firstInstance;
        pushCommand(std::move(cmd));
    }
    void drawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) override
    {
        SCmd<ECT_DRAW_INDEXED> cmd;
        cmd.indexCount = indexCount;
        cmd.instanceCount = instanceCount;
        cmd.firstIndex = firstIndex;
        cmd.vertexOffset = vertexOffset;
        cmd.firstInstance = firstInstance;
        pushCommand(std::move(cmd));
    }
    void drawIndirect(buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override
    {
        SCmd<ECT_DRAW_INDIRECT> cmd;
        cmd.buffer = core::smart_refctd_ptr<buffer_t>(buffer);
        cmd.offset = offset;
        cmd.drawCount = drawCount;
        cmd.stride = stride;
        pushCommand(std::move(cmd));
    }
    void drawIndexedIndirect(buffer_t* buffer, size_t offset, uint32_t drawCount, uint32_t stride) override
    {
        SCmd<ECT_DRAW_INDEXED_INDIRECT> cmd;
        cmd.buffer = core::smart_refctd_ptr<buffer_t>(buffer);
        cmd.offset = offset;
        cmd.drawCount = drawCount;
        cmd.stride = stride;
        pushCommand(std::move(cmd));
    }

    void setViewport(uint32_t firstViewport, uint32_t viewportCount, const asset::SViewport* pViewports) override
    {
        SCmd<ECT_SET_VIEWPORT> cmd;
        cmd.firstViewport = firstViewport;
        cmd.viewportCount = viewportCount;
        cmd.viewports = pViewports;
        pushCommand(std::move(cmd));
    }

    void setLineWidth(float lineWidth) override
    {
        SCmd<ECT_SET_LINE_WIDTH> cmd;
        cmd.lineWidth = lineWidth;
    }
    void setDepthBias(float depthBiasConstantFactor, float depthBiasClamp, float depthBiasSlopeFactor) override
    {
        SCmd<ECT_SET_DEPTH_BIAS> cmd;
        cmd.depthBiasConstantFactor;
        cmd.depthBiasClamp = depthBiasClamp;
        cmd.depthBiasSlopeFactor = depthBiasSlopeFactor;
        pushCommand(std::move(cmd));
    }

    void setBlendConstants(const float blendConstants[4]) override
    {
        SCmd<ECT_SET_BLEND_CONSTANTS> cmd;
        memcpy(cmd.blendConstants, blendConstants, 4*sizeof(float));
        pushCommand(std::move(cmd));
    }

    void copyBuffer(buffer_t* srcBuffer, buffer_t* dstBuffer, uint32_t regionCount, const asset::SBufferCopy* pRegions) override
    {
        SCmd<ECT_COPY_BUFFER> cmd;
        cmd.srcBuffer = core::smart_refctd_ptr<buffer_t>(srcBuffer);
        cmd.dstBuffer = core::smart_refctd_ptr<buffer_t>(dstBuffer);
        cmd.regionCount = regionCount;
        cmd.regions = pRegions;
        pushCommand(std::move(cmd));
    }
    void copyImage(image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SImageCopy* pRegions) override
    {
        SCmd<ECT_COPY_IMAGE> cmd;
        cmd.srcImage = core::smart_refctd_ptr<image_t>(srcImage);
        cmd.srcImageLayout = srcImageLayout;
        cmd.dstImage = core::smart_refctd_ptr<image_t>(dstImage);
        cmd.dstImageLayout = dstImageLayout;
        cmd.regionCount = regionCount;
        cmd.regions = pRegions;
        pushCommand(std::move(cmd));
    }
    void copyBufferToImage(buffer_t* srcBuffer, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        SCmd<ECT_COPY_BUFFER_TO_IMAGE> cmd;
        cmd.srcBuffer = core::smart_refctd_ptr<buffer_t>(srcBuffer);
        cmd.dstImage = core::smart_refctd_ptr<image_t>(dstImage);
        cmd.dstImageLayout = dstImageLayout;
        cmd.regionCount = regionCount;
        cmd.regions = pRegions;
        pushCommand(std::move(cmd));
    }
    void copyImageToBuffer(buffer_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, buffer_t* dstBuffer, uint32_t regionCount, const asset::IImage::SBufferCopy* pRegions) override
    {
        SCmd<ECT_COPY_IMAGE_TO_BUFFER> cmd;
        cmd.srcImage = core::smart_refctd_ptr<image_t>(srcImage);
        cmd.srcImageLayout = srcImageLayout;
        cmd.dstBuffer = core::smart_refctd_ptr<buffer_t>(dstBuffer);
        cmd.regionCount = regionCount;
        cmd.regions = pRegions;
        pushCommand(std::move(cmd));
    }
    void blitImage(image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageBlit* pRegions, asset::ISampler::E_TEXTURE_FILTER filter) override
    {
        SCmd<ECT_BLIT_IMAGE> cmd;
        cmd.srcImage = core::smart_refctd_ptr<image_t>(srcImage);
        cmd.srcImageLayout = srcImageLayout;
        cmd.dstImage = core::smart_refctd_ptr<image_t>(dstImage);
        cmd.regionCount = regionCount;
        cmd.regions = pRegions;
        cmd.filter = filter;
        pushCommand(std::move(cmd));
    }
    void resolveImage(image_t* srcImage, asset::E_IMAGE_LAYOUT srcImageLayout, image_t* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout, uint32_t regionCount, const asset::SImageResolve* pRegions) override
    {
        SCmd<ECT_RESOLVE_IMAGE> cmd;
        cmd.srcImage = core::smart_refctd_ptr<image_t>(srcImage);
        cmd.srcImageLayout = srcImageLayout;
        cmd.dstImage = core::smart_refctd_ptr<image_t>(dstImage);
        cmd.dstImageLayout = dstImageLayout;
        cmd.regionCount = regionCount;
        cmd.regions = pRegions;
        pushCommand(std::move(cmd));
    }

    void bindVertexBuffers(uint32_t firstBinding, uint32_t bindingCount, buffer_t** pBuffers, const size_t* pOffsets) override
    {
        SCmd<ECT_BIND_VERTEX_BUFFERS> cmd;
        cmd.first = firstBinding;
        cmd.count = bindingCount;
        cmd.offsets = pOffsets;
        for (uint32_t i = 0u; i < cmd.count; ++i)
        {
            buffer_t* b = pBuffers[i];
            cmd.buffers[i] = core::smart_refctd_ptr<buffer_t>(b);
        }
        pushCommand(std::move(cmd));
    }

    void setScissor(uint32_t firstScissor, uint32_t scissorCount, const asset::VkRect2D* pScissors) override
    {
        SCmd<ECT_SET_SCISSORS> cmd;
        cmd.firstScissor = firstScissor;
        cmd.scissorCount = scissorCount;
        cmd.scissors = pScissors;
        pushCommand(std::move(cmd));
    }
    void setDepthBounds(float minDepthBounds, float maxDepthBounds) override
    {
        SCmd<ECT_SET_DEPTH_BOUNDS> cmd;
        cmd.minDepthBounds = minDepthBounds;
        cmd.maxDepthBounds = maxDepthBounds;
        pushCommand(std::move(cmd));
    }
    void setStencilCompareMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t compareMask) override
    {
        SCmd<ECT_SET_STENCIL_COMPARE_MASK> cmd;
        cmd.faceMask = faceMask;
        cmd.cmpMask = compareMask;
        pushCommand(std::move(cmd));
    }
    void setStencilWriteMask(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t writeMask) override
    {
        SCmd<ECT_SET_STENCIL_WRITE_MASK> cmd;
        cmd.faceMask = faceMask;
        cmd.writeMask = writeMask;
        pushCommand(std::move(cmd));
    }
    void setStencilReference(asset::E_STENCIL_FACE_FLAGS faceMask, uint32_t reference) override
    {
        SCmd<ECT_SET_STENCIL_REFERENCE> cmd;
        cmd.faceMask = faceMask;
        cmd.reference = reference;
        pushCommand(std::move(cmd));
    }

    void dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        SCmd<ECT_DISPATCH> cmd;
        cmd.groupCountX = groupCountX;
        cmd.groupCountY = groupCountY;
        cmd.groupCountZ = groupCountZ;
        pushCommand(std::move(cmd));
    }
    void dispatchIndirect(buffer_t* buffer, size_t offset) override
    {
        SCmd<ECT_DISPATCH_INDIRECT> cmd;
        cmd.buffer = core::smart_refctd_ptr<buffer_t>(buffer);
        cmd.offset = offset;
        pushCommand(std::move(cmd));
    }
    void dispatchBase(uint32_t baseGroupX, uint32_t baseGroupY, uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) override
    {
        SCmd<ECT_DISPATCH_BASE> cmd;
        cmd.baseGroupX = baseGroupX;
        cmd.baseGroupY = baseGroupY;
        cmd.baseGroupZ = baseGroupZ;
        cmd.groupCountX = groupCountX;
        cmd.groupCountY = groupCountY;
        cmd.groupCountZ = groupCountZ;
        pushCommand(std::move(cmd));
    }

    void setEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override
    {
        SCmd<ECT_SET_EVENT> cmd;
        cmd.event = core::smart_refctd_ptr<event_t>(event);
        cmd.stageMask = stageMask;
        pushCommand(std::move(cmd));
    }
    void resetEvent(event_t* event, asset::E_PIPELINE_STAGE_FLAGS stageMask) override
    {
        SCmd<ECT_RESET_EVENT> cmd;
        cmd.event = core::smart_refctd_ptr<event_t>(event);
        cmd.stageMask = stageMask;
        pushCommand(std::move(cmd));
    }

    void waitEvents(uint32_t eventCount, event_t** pEvents, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers
    ) override
    {
        SCmd<ECT_WAIT_EVENTS> cmd;
        cmd.eventCount = eventCount;
        cmd.events = reinterpret_cast<core::smart_refctd_ptr<event_t>*>(pEvents); // TODO
        cmd.srcStageMask = srcStageMask;
        cmd.dstStageMask = dstStageMask;
        cmd.memoryBarrierCount = memoryBarrierCount;
        cmd.pMemoryBarriers = pMemoryBarriers;
        cmd.bufferMemoryBarrierCount = bufferMemoryBarrierCount;
        cmd.pBufferMemoryBarriers = pBufferMemoryBarriers;
        cmd.imageMemoryBarrierCount = imageMemoryBarrierCount;
        cmd.pImageMemoryBarriers = pImageMemoryBarriers;
        pushCommand(std::move(cmd));
    }

    void pipelineBarrier(uint32_t eventCount, const event_t** pEvents, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> srcStageMask, std::underlying_type_t<asset::E_PIPELINE_STAGE_FLAGS> dstStageMask,
        std::underlying_type_t<asset::E_DEPENDENCY_FLAGS> dependencyFlags,
        uint32_t memoryBarrierCount, const asset::SMemoryBarrier* pMemoryBarriers,
        uint32_t bufferMemoryBarrierCount, const SBufferMemoryBarrier* pBufferMemoryBarriers,
        uint32_t imageMemoryBarrierCount, const SImageMemoryBarrier* pImageMemoryBarriers) override
    {
        SCmd<ECT_PIPELINE_BARRIER> cmd;
        cmd.eventCount = eventCount;
        cmd.events = reinterpret_cast<core::smart_refctd_ptr<event_t>*>(pEvents); // TODO
        cmd.srcStageMask = srcStageMask;
        cmd.dstStageMask = dstStageMask;
        cmd.dependencyFlags = dependencyFlags;
        cmd.memoryBarrierCount = memoryBarrierCount;
        cmd.pMemoryBarriers = pMemoryBarriers;
        cmd.bufferMemoryBarrierCount = bufferMemoryBarrierCount;
        cmd.pBufferMemoryBarriers = pBufferMemoryBarriers;
        cmd.imageMemoryBarrierCount = imageMemoryBarrierCount;
        cmd.pImageMemoryBarriers = pImageMemoryBarriers;
        pushCommand(std::move(cmd));
    }

    void beginRenderPass(const SRenderpassBeginInfo* pRenderPassBegin, asset::E_SUBPASS_CONTENTS content) override
    {
        SCmd<ECT_BEGIN_RENDERPASS> cmd;
        cmd.renderpassBegin = pRenderPassBegin[0];
        cmd.content = content;
        pushCommand(std::move(cmd));
    }
    void nextSubpass(asset::E_SUBPASS_CONTENTS contents) override
    {
        SCmd<ECT_NEXT_SUBPASS> cmd;
        cmd.contents = contents;
        pushCommand(std::move(cmd));
    }
    void endRenderPass() override
    {
        SCmd<ECT_END_RENDERPASS> cmd;
        pushCommand(std::move(cmd));
    }

    void setDeviceMask(uint32_t deviceMask) override
    { 
        // theres no need to add this command to buffer in GL backend
        IGPUCommandBuffer::setDeviceMask(deviceMask);
    }

    void bindGraphicsPipeline(graphics_pipeline_t* pipeline) override
    {
        SCmd<ECT_BIND_GRAPHICS_PIPELINE> cmd;
        cmd.pipeline = core::smart_refctd_ptr<graphics_pipeline_t>(pipeline);
        pushCommand(std::move(cmd));
    }
    void bindComputePipeline(compute_pipeline_t* pipeline) override
    {
        SCmd<ECT_BIND_COMPUTE_PIPELINE> cmd;
        cmd.pipeline = core::smart_refctd_ptr<compute_pipeline_t>(pipeline);
        pushCommand(std::move(cmd));
    }

    void bindDescriptorSets(asset::E_PIPELINE_BIND_POINT pipelineBindPoint, pipeline_layout_t* layout, uint32_t firstSet, uint32_t descriptorSetCount,
        descriptor_set_t** pDescriptorSets, core::smart_refctd_dynamic_array<uint32_t> dynamicOffsets
    ) override
    {
        SCmd<ECT_BIND_DESCRIPTOR_SETS> cmd;
        cmd.pipelineBindPoint = pipelineBindPoint;
        cmd.layout = core::smart_refctd_ptr<pipeline_layout_t>(layout);
        cmd.firstSet = firstSet;
        cmd.dsCount = descriptorSetCount;
        for (uint32_t i = 0u; i < cmd.dsCount; ++i)
            cmd.descriptorSets[i] = core::smart_refctd_ptr<IGPUDescriptorSet>(pDescriptorSets[i]);
        cmd.dynamicOffsets = std::move(dynamicOffsets);
        pushCommand(std::move(cmd));
    }
    void pushConstants(pipeline_layout_t* layout, std::underlying_type_t<asset::ISpecializedShader::E_SHADER_STAGE> stageFlags, uint32_t offset, uint32_t size, const void* pValues) override
    {
        SCmd<ECT_PUSH_CONSTANTS> cmd;
        cmd.layout = core::smart_refctd_ptr<pipeline_layout_t>(layout);
        cmd.stageFlags = stageFlags;
        cmd.offset = offset;
        cmd.size = size;
        cmd.values = pValues;
        pushCommand(std::move(cmd));
    }

    void clearColorImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const asset::SClearColorValue* pColor, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        SCmd<ECT_CLEAR_COLOR_IMAGE> cmd;
        cmd.image = core::smart_refctd_ptr<image_t>(image);
        cmd.imageLayout = imageLayout;
        cmd.color = pColor[0];
        cmd.rangeCount = rangeCount;
        cmd.ranges = pRanges;
        pushCommand(std::move(cmd));
    }
    void clearDepthStencilImage(image_t* image, asset::E_IMAGE_LAYOUT imageLayout, const asset::SClearDepthStencilValue* pDepthStencil, uint32_t rangeCount, const asset::IImage::SSubresourceRange* pRanges) override
    {
        SCmd<ECT_CLEAR_DEPTH_STENCIL_IMAGE> cmd;
        cmd.image = core::smart_refctd_ptr<image_t>(image);
        cmd.imageLayout = imageLayout;
        cmd.depthStencil = pDepthStencil[0];
        cmd.rangeCount = rangeCount;
        cmd.ranges = pRanges;
        pushCommand(std::move(cmd));
    }
    void clearAttachments(uint32_t attachmentCount, const asset::SClearAttachment* pAttachments, uint32_t rectCount, const asset::SClearRect* pRects) override
    {
        SCmd<ECT_CLEAR_ATTACHMENTS> cmd;
        cmd.attachmentCount = attachmentCount;
        cmd.attachments = pAttachments;
        cmd.rectCount = rectCount;
        cmd.rects = pRects;
        pushCommand(std::move(cmd));
    }
    void fillBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t size, uint32_t data) override
    {
        SCmd<ECT_FILL_BUFFER> cmd;
        cmd.dstBuffer = core::smart_refctd_ptr<buffer_t>(dstBuffer);
        cmd.dstOffset = dstOffset;
        cmd.size = size;
        cmd.data = data;
        pushCommand(std::move(cmd));
    }
    void updateBuffer(buffer_t* dstBuffer, size_t dstOffset, size_t dataSize, const void* pData) override
    {
        SCmd<ECT_UPDATE_BUFFER> cmd;
        cmd.dstBuffer = core::smart_refctd_ptr<buffer_t>(dstBuffer);
        cmd.dstOffset = dstOffset;
        cmd.dataSize = dataSize;
        cmd.data = pData;
        pushCommand(std::move(cmd));
    }
    bool executeCommands(uint32_t count, IGPUCommandBuffer** cmdbufs) override
    {
        if (!IGPUCommandBuffer::executeCommands(count, cmdbufs))
            return false;

        for (uint32_t i = 0u; i < count; ++i)
        {
            SCmd<ECT_EXECUTE_COMMANDS> cmd;
            cmd.cmdbuf = core::smart_refctd_ptr<IGPUCommandBuffer>(cmdbufs[i]);

            pushCommand(std::move(cmd));
        }
        return true;
    }
};

}
}

#endif

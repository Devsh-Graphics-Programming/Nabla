#ifndef __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__
#define __NBL_C_OPENGL_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"
#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"
#include <mutex>

namespace nbl::video
{
class COpenGLCommandBuffer;
class COpenGLQueryPool;

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

    class IOpenGLCommand : public IGPUCommandPool::ICommand
    {
    public:
        IOpenGLCommand(const uint32_t size) : ICommand(size) {}

        virtual void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger) = 0;
    };

    template<typename CRTP>
    class IOpenGLFixedSizeCommand : public IOpenGLCommand
    {
    public:
        template <typename... Args>
        static uint32_t calc_size(const Args&...)
        {
            return core::alignUp(sizeof(CRTP), alignof(CRTP));
        }
    protected:
        IOpenGLFixedSizeCommand() : IOpenGLCommand(calc_size()) {}
    };

    class CBindFramebufferCmd;
    class CBlitNamedFramebufferCmd;
    class CClearNamedFramebufferCmd;
    class CViewportArrayVCmd;
    class CDepthRangeArrayVCmd;
    class CPolygonModeCmd;
    class CEnableCmd;
    class CDisableCmd;
    class CCullFaceCmd;
    class CStencilOpSeparateCmd;
    class CStencilFuncSeparateCmd;
    class CStencilMaskSeparateCmd;
    class CDepthFuncCmd;
    class CFrontFaceCmd;
    class CPolygonOffsetCmd;
    class CLineWidthCmd;
    class CMinSampleShadingCmd;
    class CSampleMaskICmd;
    class CDepthMaskCmd;
    class CLogicOpCmd;
    class CEnableICmd;
    class CDisableICmd;
    class CBlendFuncSeparateICmd;
    class CColorMaskICmd;
    class CMemoryBarrierCmd;
    class CBindPipelineComputeCmd;
    class CDispatchComputeCmd;

    // These does not correspond to a GL call
    class CSetUniformsImitatingPushConstantsComputeCmd; 
    class CSetUniformsImitatingPushConstantsGraphicsCmd;

    class CBindBufferCmd;
    class CBindImageTexturesCmd;
    class CBindTexturesCmd;
    class CBindSamplersCmd;
    class CBindBuffersRangeCmd;
    class CNamedBufferSubDataCmd;

    class CResetQueryCmd; // Does not correspond to a GL call, but some operation that needs to happen in a deferred way on COpenGLQueryPool.

    class CQueryCounterCmd;
    class CBeginQueryCmd;
    class CEndQueryCmd;
    class CGetQueryBufferObjectUICmd;
    class CBindPipelineGraphicsCmd;
    class CBindVertexArrayCmd;
    class CVertexArrayVertexBufferCmd;
    class CVertexArrayElementBufferCmd;
    class CPixelStoreICmd;
    class CDrawArraysInstancedBaseInstanceCmd;
    class CDrawElementsInstancedBaseVertexBaseInstanceCmd;

    template <asset::E_PIPELINE_BIND_POINT PBP>
    class CPushConstantsCmd; // Does not correspond to a GL call, but it is required that it runs in on the worker/queue thread because the push constant state is in the queue local cache

    class CCopyNamedBufferSubDataCmd;
    class CCompressedTextureSubImage2DCmd;
    class CCompressedTextureSubImage3DCmd;
    class CTextureSubImage2DCmd;
    class CTextureSubImage3DCmd;
    class CGetCompressedTextureSubImageCmd;
    class CGetTextureSubImageCmd;
    class CReadPixelsCmd;
    class CMultiDrawElementsIndirectCmd;
    class CMultiDrawElementsIndirectCountCmd;

    // This doesn't directly correspond to a GL call
    class CExecuteCommandsCmd;

    class CMultiDrawArraysIndirectCountCmd;
    class CMultiDrawArraysIndirectCmd;

private:
    std::mutex mutex;
    core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>,core::default_aligned_allocator,false,uint32_t> mempool;
};

class COpenGLCommandPool::CBindFramebufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindFramebufferCmd>
{
public:
    CBindFramebufferCmd(const COpenGLFramebuffer::hash_t& fboHash, const COpenGLFramebuffer* fbo) : m_fboHash(fboHash), m_fbo(fbo) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    COpenGLFramebuffer::hash_t m_fboHash;
    const COpenGLFramebuffer* m_fbo;
};

class COpenGLCommandPool::CBlitNamedFramebufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBlitNamedFramebufferCmd>
{
public:
    CBlitNamedFramebufferCmd(const COpenGLImage* srcImage, const COpenGLImage* dstImage, const uint32_t srcLevel, const uint32_t dstLevel,
        const uint32_t srcLayer, const uint32_t dstLayer, const asset::VkOffset3D* srcOffsets, const asset::VkOffset3D* dstOffsets, const asset::ISampler::E_TEXTURE_FILTER filter)
        : m_srcImage(srcImage), m_dstImage(dstImage), m_srcLevel(srcLevel), m_dstLevel(dstLevel), m_srcLayer(srcLayer), m_dstLayer(dstLayer), m_filter(filter)
    {
        memcpy(m_srcOffsets, srcOffsets, 2 * sizeof(asset::VkOffset3D));
        memcpy(m_dstOffsets, dstOffsets, 2 * sizeof(asset::VkOffset3D));
    }

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLImage* m_srcImage;
    const COpenGLImage* m_dstImage;
    const uint32_t m_srcLevel;
    const uint32_t m_dstLevel;
    const uint32_t m_srcLayer;
    const uint32_t m_dstLayer;
    asset::VkOffset3D m_srcOffsets[2];
    asset::VkOffset3D m_dstOffsets[2];
    asset::ISampler::E_TEXTURE_FILTER m_filter;
};

class COpenGLCommandPool::CClearNamedFramebufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CClearNamedFramebufferCmd>
{
public:
    CClearNamedFramebufferCmd(const COpenGLFramebuffer::hash_t& fboHash, const asset::E_FORMAT format, const GLenum bufferType, const asset::SClearValue& clearValue, const GLint drawBufferIndex)
        : m_fboHash(fboHash), m_format(format), m_bufferType(bufferType), m_clearValue(clearValue), m_drawBufferIndex(drawBufferIndex)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLFramebuffer::hash_t m_fboHash;
    const asset::E_FORMAT m_format;
    const GLenum m_bufferType;
    const asset::SClearValue m_clearValue;
    const GLint m_drawBufferIndex;
};

class COpenGLCommandPool::CViewportArrayVCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CViewportArrayVCmd>
{
public:
    CViewportArrayVCmd(const GLuint first, const GLsizei count, const GLfloat* params)
        : m_first(first), m_count(count)
    {
        memcpy(m_params, params, (m_count-m_first)*sizeof(GLfloat)*4ull);
    }

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_first;
    const GLsizei m_count;
    GLfloat m_params[SOpenGLState::MAX_VIEWPORT_COUNT * 4];
};

class COpenGLCommandPool::CDepthRangeArrayVCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDepthRangeArrayVCmd>
{
public:
    CDepthRangeArrayVCmd(const GLuint first, const GLsizei count, const GLdouble* params)
        : m_first(first), m_count(count)
    {
        memcpy(m_params, params, (m_count - m_first) * sizeof(GLdouble) * 2ull);
    }

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_first;
    const GLsizei m_count;
    GLdouble m_params[SOpenGLState::MAX_VIEWPORT_COUNT * 2];
};

class COpenGLCommandPool::CPolygonModeCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CPolygonModeCmd>
{
public:
    CPolygonModeCmd(const GLenum mode) : m_mode(mode) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
};

class COpenGLCommandPool::CEnableCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CEnableCmd>
{
public:
    CEnableCmd(const GLenum cap) : m_cap(cap) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_cap;
};

class COpenGLCommandPool::CDisableCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDisableCmd>
{
public:
    CDisableCmd(const GLenum cap) : m_cap(cap) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_cap;
};

class COpenGLCommandPool::CCullFaceCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CCullFaceCmd>
{
public:
    CCullFaceCmd(const GLenum mode) : m_mode(mode) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
};

class COpenGLCommandPool::CStencilOpSeparateCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CStencilOpSeparateCmd>
{
public:
    CStencilOpSeparateCmd(const GLenum face, const GLenum sfail, const GLenum dpfail, const GLenum dppass)
        : m_face(face), m_sfail(sfail), m_dpfail(dpfail), m_dppass(dppass)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_face;
    const GLenum m_sfail;
    const GLenum m_dpfail;
    const GLenum m_dppass;
};

class COpenGLCommandPool::CStencilFuncSeparateCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CStencilFuncSeparateCmd>
{
public:
    CStencilFuncSeparateCmd(const GLenum face, const GLenum func, const GLint ref, const GLuint mask) : 
        m_face(face), m_func(func), m_ref(ref), m_mask(mask)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_face;
    const GLenum m_func;
    const GLint m_ref;
    const GLuint m_mask;
};

class COpenGLCommandPool::CStencilMaskSeparateCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CStencilMaskSeparateCmd>
{
public:
    CStencilMaskSeparateCmd(const GLenum face, const GLuint mask) : m_face(face), m_mask(mask) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_face;
    const GLuint m_mask;
};

class COpenGLCommandPool::CDepthFuncCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDepthFuncCmd>
{
public:
    CDepthFuncCmd(const GLenum func) : m_func(func) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_func;
};

class COpenGLCommandPool::CFrontFaceCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CFrontFaceCmd>
{
public:
    CFrontFaceCmd(const GLenum mode) : m_mode(mode) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
};

class COpenGLCommandPool::CPolygonOffsetCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CPolygonOffsetCmd>
{
public:
    CPolygonOffsetCmd(const GLfloat factor, const GLfloat units) : m_factor(factor), m_units(units) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLfloat m_factor;
    const GLfloat m_units;
};

class COpenGLCommandPool::CLineWidthCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CLineWidthCmd>
{
public:
    CLineWidthCmd(const GLfloat width) : m_width(width) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLfloat m_width;
};

class COpenGLCommandPool::CMinSampleShadingCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CMinSampleShadingCmd>
{
public:
    CMinSampleShadingCmd(const GLfloat value) : m_value(value) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLfloat m_value;
};

class COpenGLCommandPool::CSampleMaskICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CSampleMaskICmd>
{
public:
    CSampleMaskICmd(const GLuint maskNumber, const GLbitfield mask) : m_maskNumber(maskNumber), m_mask(mask) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_maskNumber;
    const GLbitfield m_mask;
};

class COpenGLCommandPool::CDepthMaskCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDepthMaskCmd>
{
public:
    CDepthMaskCmd(const GLboolean flag) : m_flag(flag) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLboolean m_flag;
};

class COpenGLCommandPool::CLogicOpCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CLogicOpCmd>
{
public:
    CLogicOpCmd(const GLenum opcode) : m_opcode(opcode) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_opcode;
};

class COpenGLCommandPool::CEnableICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CEnableICmd>
{
public:
    CEnableICmd(const GLenum cap, const GLuint index) : m_cap(cap), m_index(index) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_cap;
    const GLuint m_index;
};

class COpenGLCommandPool::CDisableICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDisableICmd>
{
public:
    CDisableICmd(const GLenum cap, const GLuint index) : m_cap(cap), m_index(index) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_cap;
    const GLuint m_index;
};

class COpenGLCommandPool::CBlendFuncSeparateICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBlendFuncSeparateICmd>
{
public:
    CBlendFuncSeparateICmd(const GLuint buf, const GLenum srcRGB, const GLenum dstRGB, const GLenum srcAlpha, const GLenum dstAlpha)
        : m_buf(buf), m_srcRGB(srcRGB), m_dstRGB(dstRGB), m_srcAlpha(srcAlpha), m_dstAlpha(dstAlpha)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_buf;
    const GLenum m_srcRGB;
    const GLenum m_dstRGB;
    const GLenum m_srcAlpha;
    const GLenum m_dstAlpha;
};

class COpenGLCommandPool::CColorMaskICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CColorMaskICmd>
{
public:
    CColorMaskICmd(const GLuint buf, const GLboolean red, const GLboolean green, const GLboolean blue, const GLboolean alpha)
        : m_buf(buf), m_red(red), m_green(green), m_blue(blue), m_alpha(alpha)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_buf;
    const GLboolean m_red;
    const GLboolean m_green;
    const GLboolean m_blue;
    const GLboolean m_alpha;
};

class COpenGLCommandPool::CMemoryBarrierCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CMemoryBarrierCmd>
{
public:
    CMemoryBarrierCmd(const GLbitfield barrierBits) : m_barrierBits(barrierBits) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLbitfield m_barrierBits;
};

class COpenGLCommandPool::CBindPipelineComputeCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindPipelineComputeCmd>
{
public:
    CBindPipelineComputeCmd(const COpenGLComputePipeline* pipeline) : m_glppln(pipeline) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLComputePipeline* m_glppln;
};

class COpenGLCommandPool::CDispatchComputeCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDispatchComputeCmd>
{
public:
    CDispatchComputeCmd(const GLuint numGroupsX, const GLuint numGroupsY, const GLuint numGroupsZ) : m_numGroupsX(numGroupsX), m_numGroupsY(numGroupsY), m_numGroupsZ(numGroupsZ) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_numGroupsX;
    const GLuint m_numGroupsY;
    const GLuint m_numGroupsZ;
};

class COpenGLCommandPool::CSetUniformsImitatingPushConstantsComputeCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CSetUniformsImitatingPushConstantsComputeCmd>
{
public:
    CSetUniformsImitatingPushConstantsComputeCmd(const COpenGLComputePipeline* pipeline) : m_pipeline(pipeline) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLComputePipeline* m_pipeline;
};

class COpenGLCommandPool::CSetUniformsImitatingPushConstantsGraphicsCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CSetUniformsImitatingPushConstantsGraphicsCmd>
{
public:
    CSetUniformsImitatingPushConstantsGraphicsCmd(const COpenGLRenderpassIndependentPipeline* pipeline) : m_pipeline(pipeline) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLRenderpassIndependentPipeline* m_pipeline;
};

class COpenGLCommandPool::CBindBufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindBufferCmd>
{
public:
    CBindBufferCmd(const GLenum target, const GLuint bufferGLName) : m_target(target), m_bufferGLName(bufferGLName) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_target;
    const GLuint m_bufferGLName;
};

class COpenGLCommandPool::CBindImageTexturesCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindImageTexturesCmd>
{
public:
    CBindImageTexturesCmd(const GLuint first, const GLsizei count, const GLuint* textures, const GLenum* formats)
        : m_first(first), m_count(count), m_textures(textures), m_formats(formats)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_first;
    const GLsizei m_count;

    // Pointers to memory owned by COpenGLDescriptorSet.
    const GLuint* m_textures;
    const GLenum* m_formats;
};

class COpenGLCommandPool::CBindTexturesCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindTexturesCmd>
{
public:
    CBindTexturesCmd(const GLuint first, const GLsizei count, const GLuint* textures, const GLenum* targets)
        : m_first(first), m_count(count), m_textures(textures), m_targets(targets)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_first;
    const GLsizei m_count;

    // Pointers to memory owned by COpenGLDescriptorSet.
    const GLuint* m_textures;
    const GLenum* m_targets;
};

class COpenGLCommandPool::CBindSamplersCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindSamplersCmd>
{
public:
    CBindSamplersCmd(const GLuint first, const GLsizei count, const GLuint* samplers) : m_first(first), m_count(count), m_samplers(samplers) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_first;
    const GLsizei m_count;

    // Pointer to memory owned by COpenGLDescriptorSet.
    const GLuint* m_samplers;
};

class COpenGLCommandPool::CBindBuffersRangeCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindBuffersRangeCmd>
{
    //not entirely sure those MAXes are right
    static inline constexpr size_t MaxUboCount = 96ull;
    static inline constexpr size_t MaxSsboCount = 91ull;
    static inline constexpr size_t MaxOffsets = std::max(MaxUboCount, MaxSsboCount);

public:
    CBindBuffersRangeCmd(const GLenum target, const GLuint first, const GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLintptr* sizes)
        : m_target(target), m_first(first), m_count(count), m_buffers(buffers)
    {
        memcpy(m_offsets, offsets, m_count*sizeof(GLintptr));
        memcpy(m_sizes, sizes, m_count*sizeof(GLintptr));
    }

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_target;
    const GLuint m_first;
    const GLsizei m_count;

    // Pointer to memory owned by COpenGLDescriptorSet.
    const GLuint* m_buffers;

    GLintptr m_offsets[MaxOffsets];
    GLintptr m_sizes[MaxOffsets];
};

class COpenGLCommandPool::CNamedBufferSubDataCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CNamedBufferSubDataCmd>
{
public:
    CNamedBufferSubDataCmd(const GLuint bufferGLName, const GLintptr offset, const GLsizeiptr size, const void* data)
        : m_bufferGLName(bufferGLName), m_offset(offset), m_size(size)
    {
        m_data.resize(size);
        memcpy(m_data.data(), data, m_size);
    }

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_bufferGLName;
    const GLintptr m_offset;
    const GLsizeiptr m_size;
    core::vector<uint8_t> m_data;
};

class COpenGLCommandPool::CResetQueryCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CResetQueryCmd>
{
public:
    CResetQueryCmd(core::smart_refctd_ptr<COpenGLQueryPool>&& queryPool, const uint32_t query) : m_queryPool(std::move(queryPool)), m_query(query) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    core::smart_refctd_ptr<COpenGLQueryPool> m_queryPool;
    const uint32_t m_query;
};

class COpenGLCommandPool::CQueryCounterCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CQueryCounterCmd>
{
public:
    CQueryCounterCmd(const uint32_t query, const GLenum target, core::smart_refctd_ptr<COpenGLQueryPool>&& queryPool) : m_query(query), m_target(target), m_queryPool(std::move(queryPool)) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const uint32_t m_query;
    const GLenum m_target;
    // TODO(achal): Raw pointer.
    core::smart_refctd_ptr<COpenGLQueryPool> m_queryPool;
};

class COpenGLCommandPool::CBeginQueryCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBeginQueryCmd>
{
public:
    CBeginQueryCmd(const uint32_t query, const GLenum target, core::smart_refctd_ptr<const COpenGLQueryPool>&& queryPool)
        : m_query(query), m_target(target), m_queryPool(std::move(queryPool))
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const uint32_t m_query;
    const GLenum m_target;
    // TODO(achal): Raw pointer.
    core::smart_refctd_ptr<const COpenGLQueryPool> m_queryPool;
};

class COpenGLCommandPool::CEndQueryCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CEndQueryCmd>
{
public:
    CEndQueryCmd(const GLenum target) : m_target(target) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_target;
};

class COpenGLCommandPool::CGetQueryBufferObjectUICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CGetQueryBufferObjectUICmd>
{
public:
    CGetQueryBufferObjectUICmd(const COpenGLQueryPool* queryPool, const uint32_t queryIdx, const bool use64Version, const GLuint buffer, const GLenum pname, const GLintptr offset)
        : m_queryPool(queryPool), m_queryIdx(queryIdx), m_use64Version(use64Version), m_buffer(buffer), m_pname(pname), m_offset(offset)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLQueryPool* m_queryPool;
    const uint32_t m_queryIdx;
    const bool m_use64Version;
    const GLuint m_buffer;
    const GLenum m_pname;
    const GLintptr m_offset;
};

class COpenGLCommandPool::CBindPipelineGraphicsCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindPipelineGraphicsCmd>
{
public:
    CBindPipelineGraphicsCmd(const COpenGLRenderpassIndependentPipeline* pipeline) : m_pipeline(pipeline) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLRenderpassIndependentPipeline* m_pipeline;

};

class COpenGLCommandPool::CBindVertexArrayCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindVertexArrayCmd>
{
    using SVAOCacheKey = COpenGLRenderpassIndependentPipeline::SVAOHash;

public:
    CBindVertexArrayCmd(const SVAOCacheKey& vaoKey) : m_vaoKey(vaoKey) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const SVAOCacheKey m_vaoKey;
};

class COpenGLCommandPool::CVertexArrayVertexBufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CVertexArrayVertexBufferCmd>
{
    using SVAOCacheKey = COpenGLRenderpassIndependentPipeline::SVAOHash;

public:
    CVertexArrayVertexBufferCmd(const SVAOCacheKey& vaoKey, const GLuint bindingIndex, const GLuint bufferGLName, const GLintptr offset, const GLsizei stride)
        : m_vaoKey(vaoKey), m_bindingIndex(bindingIndex), m_bufferGLName(bufferGLName), m_offset(offset), m_stride(stride)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const SVAOCacheKey m_vaoKey;
    const GLuint m_bindingIndex;
    const GLuint m_bufferGLName;
    const GLintptr m_offset;
    const GLsizei m_stride;
};

class COpenGLCommandPool::CVertexArrayElementBufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CVertexArrayElementBufferCmd>
{
    using SVAOCacheKey = COpenGLRenderpassIndependentPipeline::SVAOHash;

public:
    CVertexArrayElementBufferCmd(const SVAOCacheKey& vaoKey, const GLuint bufferGLName) : m_vaoKey(vaoKey), m_bufferGLName(bufferGLName) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const SVAOCacheKey m_vaoKey;
    const GLuint m_bufferGLName;
};

class COpenGLCommandPool::CPixelStoreICmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CPixelStoreICmd>
{
public:
    CPixelStoreICmd(const GLenum pname, const GLint param) : m_pname(pname), m_param(param) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_pname;
    const GLint m_param;
};

class COpenGLCommandPool::CDrawArraysInstancedBaseInstanceCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDrawArraysInstancedBaseInstanceCmd>
{
public:
    CDrawArraysInstancedBaseInstanceCmd(const GLenum mode, const GLint first, const GLsizei count, const GLsizei instancecount, const GLuint baseinstance)
        : m_mode(mode), m_first(first), m_count(count), m_instancecount(instancecount), m_baseinstance(baseinstance)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
    const GLint m_first;
    const GLsizei m_count;
    const GLsizei m_instancecount;
    const GLuint m_baseinstance;
};

class COpenGLCommandPool::CDrawElementsInstancedBaseVertexBaseInstanceCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CDrawElementsInstancedBaseVertexBaseInstanceCmd>
{
public:
    CDrawElementsInstancedBaseVertexBaseInstanceCmd(const GLenum mode, const GLsizei count, const GLenum type, const GLuint64 idxBufOffset, const GLsizei instancecount, const GLint basevertex, const GLuint baseinstance)
        : m_mode(mode), m_count(count), m_type(type), m_idxBufOffset(idxBufOffset), m_instancecount(instancecount), m_basevertex(basevertex), m_baseinstance(baseinstance)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
    const GLsizei m_count;
    const GLenum m_type;
    const GLuint64 m_idxBufOffset;
    const GLsizei m_instancecount;
    const GLint m_basevertex;
    const GLuint m_baseinstance;
};

template <asset::E_PIPELINE_BIND_POINT PBP>
class COpenGLCommandPool::CPushConstantsCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CPushConstantsCmd<PBP>>
{
public:
    CPushConstantsCmd(const COpenGLPipelineLayout* layout, const core::bitflag<asset::IShader::E_SHADER_STAGE> stages, const uint32_t offset, const uint32_t size, const void* values)
        : m_layout(layout), m_stages(stages), m_offset(offset), m_size(size)
    {
        memcpy(m_values, values, m_size);
    }

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override
    {
        //validation is done in pushConstants_validate() of command buffer GL impl (COpenGLCommandBuffer/COpenGLPrimaryCommandBuffer)
        //if arguments were invalid (dont comply Valid Usage section of vkCmdPushConstants docs), execution should not even get to this point

        if (queueCache.pushConstantsState<PBP>()->layout && !queueCache.pushConstantsState<PBP>()->layout->isCompatibleForPushConstants(m_layout))
        {
            //#ifdef _NBL_DEBUG
            constexpr size_t toFill = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE / sizeof(uint64_t);
            constexpr size_t bytesLeft = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE - (toFill * sizeof(uint64_t));
            constexpr uint64_t pattern = 0xdeadbeefDEADBEEFull;
            std::fill(reinterpret_cast<uint64_t*>(queueCache.pushConstantsState<PBP>()->data), reinterpret_cast<uint64_t*>(queueCache.pushConstantsState<PBP>()->data) + toFill, pattern);
            if constexpr (bytesLeft > 0ull)
                memcpy(reinterpret_cast<uint64_t*>(queueCache.pushConstantsState<PBP>()->data) + toFill, &pattern, bytesLeft);
            //#endif

            m_stages |= IGPUShader::ESS_ALL;
        }
        queueCache.pushConstantsState<PBP>()->incrementStamps(m_stages.value);

        queueCache.pushConstantsState<PBP>()->layout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(m_layout);
        memcpy(queueCache.pushConstantsState<PBP>()->data + m_offset, m_values, m_size);
    }

private:
    const COpenGLPipelineLayout* m_layout;
    core::bitflag<asset::IShader::E_SHADER_STAGE> m_stages;
    const uint32_t m_offset;
    const uint32_t m_size;
    uint8_t m_values[IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE];
};

class COpenGLCommandPool::CCopyNamedBufferSubDataCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CCopyNamedBufferSubDataCmd>
{
public:
    CCopyNamedBufferSubDataCmd(const GLuint readBufferGLName, const GLuint writeBufferGLName, const GLintptr readOffset, const GLintptr writeOffset, const GLsizeiptr size)
        : m_readBufferGLName(readBufferGLName), m_writeBufferGLName(writeBufferGLName), m_readOffset(readOffset), m_writeOffset(writeOffset), m_size(size)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_readBufferGLName;
    const GLuint m_writeBufferGLName;
    const GLintptr m_readOffset;
    const GLintptr m_writeOffset;
    const GLsizeiptr m_size;
};

class COpenGLCommandPool::CCompressedTextureSubImage2DCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CCompressedTextureSubImage2DCmd>
{
public:
    CCompressedTextureSubImage2DCmd(const GLuint texture, const GLenum target, const GLint level, const GLint xoffset, const GLint yoffset,
        const GLsizei width, const GLsizei height, const GLenum format, const GLsizei imageSize, const void* data) 
        : m_texture(texture), m_target(target), m_level(level), m_xoffset(xoffset), m_yoffset(yoffset), m_width(width), m_height(height), m_format(format), m_imageSize(imageSize), m_data(data)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_texture;
    const GLenum m_target;
    const GLint m_level;
    const GLint m_xoffset;
    const GLint m_yoffset;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLenum m_format;
    const GLsizei m_imageSize;
    const void* m_data;
};

class COpenGLCommandPool::CCompressedTextureSubImage3DCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CCompressedTextureSubImage3DCmd>
{
public:
    CCompressedTextureSubImage3DCmd(const GLuint texture, const GLenum target, const GLint level, const GLint xoffset, const GLint yoffset, const GLint zoffset,
        const GLsizei width, const GLsizei height, const GLsizei depth, const GLenum format, const GLsizei imageSize, const void* data)
        : m_texture(texture), m_target(target), m_level(level), m_xoffset(xoffset), m_yoffset(yoffset), m_zoffset(zoffset), m_width(width), m_height(height), m_depth(depth),
        m_format(format), m_imageSize(imageSize), m_data(data)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_texture;
    const GLenum m_target;
    const GLint m_level;
    const GLint m_xoffset;
    const GLint m_yoffset;
    const GLint m_zoffset;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLsizei m_depth;
    const GLenum m_format;
    const GLsizei m_imageSize;
    const void* m_data;

};

class COpenGLCommandPool::CTextureSubImage2DCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CTextureSubImage2DCmd>
{
public:
    CTextureSubImage2DCmd(const GLuint texture, const GLenum target, const GLint level, const GLint xoffset, const GLint yoffset,
        const GLsizei width, const GLsizei height, const GLenum format, const GLenum type, const void* pixels) 
        : m_texture(texture), m_target(target), m_level(level), m_xoffset(xoffset), m_yoffset(yoffset), m_width(width), m_height(height), m_format(format), m_type(type), m_pixels(pixels)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_texture;
    const GLenum m_target;
    const GLint m_level;
    const GLint m_xoffset;
    const GLint m_yoffset;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLenum m_format;
    const GLenum m_type;
    const void* m_pixels;
};

class COpenGLCommandPool::CTextureSubImage3DCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CTextureSubImage3DCmd>
{
public:
    CTextureSubImage3DCmd(const GLuint texture, const GLenum target, const GLint level, const GLint xoffset, const GLint yoffset, const GLint zoffset,
        const GLsizei width, const GLsizei height, const GLsizei depth, const GLenum format, const GLenum type, const void* pixels)
        : m_texture(texture), m_target(target), m_level(level), m_xoffset(xoffset), m_yoffset(yoffset), m_zoffset(zoffset), m_width(width), m_height(height),
        m_depth(depth), m_format(format), m_type(type), m_pixels(pixels)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_texture;
    const GLenum m_target;
    const GLint m_level;
    const GLint m_xoffset;
    const GLint m_yoffset;
    const GLint m_zoffset;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLsizei m_depth;
    const GLenum m_format;
    const GLenum m_type;
    const void* m_pixels;
};

class COpenGLCommandPool::CGetCompressedTextureSubImageCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CGetCompressedTextureSubImageCmd>
{
public:
    CGetCompressedTextureSubImageCmd(const GLuint texture, const GLint level, const GLint xoffset, const GLint yoffset, const GLint zoffset,
        const GLsizei width, const GLsizei height, const GLsizei depth, GLsizei bufSize, const size_t bufferOffset)
        : m_texture(texture), m_level(level), m_xoffset(xoffset), m_yoffset(yoffset), m_zoffset(zoffset), m_width(width), m_height(height), m_depth(depth),
        m_bufSize(bufSize), m_bufferOffset(bufferOffset)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_texture;
    const GLint m_level;
    const GLint m_xoffset;
    const GLint m_yoffset;
    const GLint m_zoffset;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLsizei m_depth;
    const GLsizei m_bufSize;
    const size_t m_bufferOffset;
};

class COpenGLCommandPool::CGetTextureSubImageCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CGetTextureSubImageCmd>
{
public:
    CGetTextureSubImageCmd(const GLuint texture, const GLint level, const GLint xoffset, const GLint yoffset, const GLint zoffset,
        const GLsizei width, const GLsizei height, const GLsizei depth, const GLenum format, const GLenum type, const GLsizei bufSize, const size_t bufferOffset)
        : m_texture(texture), m_level(level), m_xoffset(xoffset), m_yoffset(yoffset), m_zoffset(zoffset), m_width(width), m_height(height), m_depth(depth), m_format(format),
        m_type(type), m_bufSize(bufSize), m_bufferOffset(bufferOffset)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLuint m_texture;
    const GLint m_level;
    const GLint m_xoffset;
    const GLint m_yoffset;
    const GLint m_zoffset;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLsizei m_depth;
    const GLenum m_format;
    const GLenum m_type;
    const GLsizei m_bufSize;
    const size_t m_bufferOffset;
};

class COpenGLCommandPool::CReadPixelsCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CReadPixelsCmd>
{
public:
    CReadPixelsCmd(const COpenGLImage* image, const uint32_t level, const uint32_t layer, const GLint x, const GLint y, const GLsizei width, const GLsizei height,
        const GLenum format, const GLenum type, const size_t bufOffset)
        : m_image(image), m_level(level), m_layer(layer), m_x(x), m_y(y), m_width(width), m_height(height), m_format(format), m_type(type), m_bufOffset(bufOffset)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const COpenGLImage* m_image;
    const uint32_t m_level;
    const uint32_t m_layer;

    const GLint m_x;
    const GLint m_y;
    const GLsizei m_width;
    const GLsizei m_height;
    const GLenum m_format;
    const GLenum m_type;
    const size_t m_bufOffset;

};

class COpenGLCommandPool::CMultiDrawElementsIndirectCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CMultiDrawElementsIndirectCmd>
{
public:
    CMultiDrawElementsIndirectCmd(const GLenum mode, const GLenum type, const GLuint64 indirect, const GLsizei drawcount, const GLsizei stride)
        : m_mode(mode), m_type(type), m_indirect(indirect), m_drawcount(drawcount), m_stride(stride)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
    const GLenum m_type;
    const GLuint64 m_indirect;
    const GLsizei m_drawcount;
    const GLsizei m_stride;
};

class COpenGLCommandPool::CMultiDrawElementsIndirectCountCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CMultiDrawElementsIndirectCountCmd>
{
public:
    CMultiDrawElementsIndirectCountCmd(const GLenum mode, const GLenum type, const GLuint64 indirect, const GLintptr drawcount, const GLsizei stride)
        : m_mode(mode), m_type(type), m_indirect(indirect), m_drawcount(drawcount), m_stride(stride)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
    const GLenum m_type;
    const GLuint64 m_indirect;
    const GLintptr m_drawcount;
    const GLsizei m_stride;
};

class COpenGLCommandPool::CExecuteCommandsCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CExecuteCommandsCmd>
{
public:
    CExecuteCommandsCmd(const uint32_t count, IGPUCommandBuffer* const* const commandBuffers) : m_count(count), m_commandBuffers(commandBuffers) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const uint32_t m_count;
    IGPUCommandBuffer* const* const m_commandBuffers;
};

class COpenGLCommandPool::CMultiDrawArraysIndirectCountCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CMultiDrawArraysIndirectCountCmd>
{
public:
    CMultiDrawArraysIndirectCountCmd(const GLenum mode, const GLuint64 indirect, const GLintptr drawcount, const GLsizei stride)
        : m_mode(mode), m_indirect(indirect), m_drawcount(drawcount), m_stride(stride)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
    const GLuint64 m_indirect;
    const GLintptr m_drawcount;
    const GLsizei m_stride;
};

class COpenGLCommandPool::CMultiDrawArraysIndirectCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CMultiDrawArraysIndirectCmd>
{
public:
    CMultiDrawArraysIndirectCmd(const GLenum mode, const GLuint64 indirect, const GLintptr drawcount, const GLsizei stride)
        : m_mode(mode), m_indirect(indirect), m_drawcount(drawcount), m_stride(stride)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const GLenum m_mode;
    const GLuint64 m_indirect;
    const GLintptr m_drawcount;
    const GLsizei m_stride;
};

}


#endif

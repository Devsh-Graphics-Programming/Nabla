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

    template<size_t STAGE_COUNT> class CProgramUniformCmd;
    // This does not correspond to but to preserve the order of execution in setUniformsImitatingPushConstants I have to defer this as well.
    // Perhaps, we should combine this with CProgramUniformCmd and make a CSetUniformsImitatingPushConstantsCmd and call setUniformsImitatingPushConstants
    // on the worker thread instead.
    template <size_t STAGE_COUNT> class CAfterUniformsSetCmd;

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

private:
    std::mutex mutex;
    core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>,core::default_aligned_allocator,false,uint32_t> mempool;
};

class COpenGLCommandPool::CBindFramebufferCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CBindFramebufferCmd>
{
public:
    CBindFramebufferCmd(const COpenGLFramebuffer::hash_t& fboHash, core::smart_refctd_ptr<const COpenGLFramebuffer>&& fbo) : m_fboHash(fboHash), m_fbo(std::move(fbo)) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    COpenGLFramebuffer::hash_t m_fboHash;
    core::smart_refctd_ptr<const COpenGLFramebuffer> m_fbo;
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

template<size_t STAGE_COUNT>
class COpenGLCommandPool::CProgramUniformCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CProgramUniformCmd<STAGE_COUNT>>
{
    using UniformMemberType = asset::impl::SShaderMemoryBlock::SMember;
    using GLPipelineType = IOpenGLPipeline<STAGE_COUNT>;

    static inline constexpr uint32_t MaxDwordSize = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE / sizeof(uint32_t);

public:
    CProgramUniformCmd(core::smart_refctd_ptr<const GLPipelineType>&& pipeline, const uint32_t stageIx, const UniformMemberType& uniformMember,
        const uint8_t* baseOffset, const GLint location, const std::array<uint32_t, MaxDwordSize>& packedData)
        : m_pipeline(std::move(pipeline)), m_stageIx(stageIx), m_uniformMember(uniformMember), m_baseOffset(baseOffset), m_location(location), m_packedData(packedData)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override
    {
        const auto* glppln = m_pipeline.get();

        GLuint GLname = glppln->getShaderGLnameForCtx(m_stageIx, ctxid);
        uint8_t* state = glppln->getPushConstantsStateForStage(m_stageIx, ctxid);

        const auto& m = m_uniformMember;
        auto is_scalar_or_vec = [&m] { return (m.mtxRowCnt >= 1u && m.mtxColCnt == 1u); };
        auto is_mtx = [&m] { return (m.mtxRowCnt > 1u && m.mtxColCnt > 1u); };

        uint8_t* valueptr = state + m.offset;
        NBL_ASSUME_ALIGNED(valueptr, sizeof(float));

        uint32_t arrayStride = m.arrayStride;
        // in case of non-array types, m.arrayStride is irrelevant
        // we should compute it though, so that we dont have to branch in the loop 
        if (!m.isArray())
        {
            // 1N for scalar types, 2N for gvec2, 4N for gvec3 and gvec4
            // N==sizeof(float)
            // WARNING / TODO : need some touch in case when we want to support `double` push constants
            if (is_scalar_or_vec())
                arrayStride = (m.mtxRowCnt == 1u) ? m.size : core::roundUpToPoT(m.mtxRowCnt) * sizeof(float);
            // same as size in case of matrices
            else if (is_mtx())
                arrayStride = m.size;
        }
        assert(m.mtxStride == 0u || arrayStride % m.mtxStride == 0u);
        //NBL_ASSUME_ALIGNED(valueptr, arrayStride); // should get the std140/std430 alignment of the type instead

        auto* baseOffset = m_baseOffset;
        const uint32_t count = std::min<uint32_t>(m.count, MaxDwordSize / (m.mtxRowCnt * m.mtxColCnt));

        if (!std::equal(baseOffset, baseOffset + arrayStride * count, valueptr) || !glppln->haveUniformsBeenEverSet(m_stageIx, ctxid))
        {
            // TODO pointers to GL func (those arrays)
            if (is_mtx() && m.type == asset::EGVT_F32)
            {
                PFNGLPROGRAMUNIFORMMATRIX4FVPROC glProgramUniformMatrixNxMfv_fptr[3][3]
                { //N - num of columns, M - num of rows because of weird OpenGL naming convention
                    {&gl->glShader.pglProgramUniformMatrix2fv, &gl->glShader.pglProgramUniformMatrix2x3fv, &gl->glShader.pglProgramUniformMatrix2x4fv},//2xM
                    {&gl->glShader.pglProgramUniformMatrix3x2fv, &gl->glShader.pglProgramUniformMatrix3fv, &gl->glShader.pglProgramUniformMatrix3x4fv},//3xM
                    {&gl->glShader.pglProgramUniformMatrix4x2fv, &gl->glShader.pglProgramUniformMatrix4x3fv, &gl->glShader.pglProgramUniformMatrix4fv} //4xM
                };

                glProgramUniformMatrixNxMfv_fptr[m.mtxColCnt - 2u][m.mtxRowCnt - 2u](GLname, m_location, count, m.rowMajor ? GL_TRUE : GL_FALSE, reinterpret_cast<const GLfloat*>(m_packedData.data()));
            }
            else if (is_scalar_or_vec())
            {
                switch (m.type)
                {
                case asset::EGVT_F32:
                {
                    PFNGLPROGRAMUNIFORM1FVPROC glProgramUniformNfv_fptr[4]
                    {
                        &gl->glShader.pglProgramUniform1fv, &gl->glShader.pglProgramUniform2fv, &gl->glShader.pglProgramUniform3fv, &gl->glShader.pglProgramUniform4fv
                    };
                    glProgramUniformNfv_fptr[m.mtxRowCnt - 1u](GLname, m_location, count, reinterpret_cast<const GLfloat*>(m_packedData.data()));
                    break;
                }
                case asset::EGVT_I32:
                {
                    PFNGLPROGRAMUNIFORM1IVPROC glProgramUniformNiv_fptr[4]
                    {
                        &gl->glShader.pglProgramUniform1iv, &gl->glShader.pglProgramUniform2iv, &gl->glShader.pglProgramUniform3iv, &gl->glShader.pglProgramUniform4iv
                    };
                    glProgramUniformNiv_fptr[m.mtxRowCnt - 1u](GLname, m_location, count, reinterpret_cast<const GLint*>(m_packedData.data()));
                    break;
                }
                case asset::EGVT_U32:
                {
                    PFNGLPROGRAMUNIFORM1UIVPROC glProgramUniformNuiv_fptr[4]
                    {
                        &gl->glShader.pglProgramUniform1uiv, &gl->glShader.pglProgramUniform2uiv, &gl->glShader.pglProgramUniform3uiv, &gl->glShader.pglProgramUniform4uiv
                    };
                    glProgramUniformNuiv_fptr[m.mtxRowCnt - 1u](GLname, m_location, count, reinterpret_cast<const GLuint*>(m_packedData.data()));
                    break;
                }
                }
            }
            std::copy(baseOffset, baseOffset + arrayStride * count, valueptr);
        }
    }

private:
    core::smart_refctd_ptr<const GLPipelineType> m_pipeline;
    const uint32_t m_stageIx;
    UniformMemberType m_uniformMember;
    const uint8_t* m_baseOffset;
    const GLint m_location;
    std::array<uint32_t, MaxDwordSize> m_packedData;
};

template <size_t STAGE_COUNT>
class COpenGLCommandPool::CAfterUniformsSetCmd : public COpenGLCommandPool::IOpenGLFixedSizeCommand<CAfterUniformsSetCmd<STAGE_COUNT>>
{
    using GLPipelineType = IOpenGLPipeline<STAGE_COUNT>;

public:
    CAfterUniformsSetCmd(core::smart_refctd_ptr<const GLPipelineType>&& pipeline, const uint32_t stageIx) : m_pipeline(std::move(pipeline)), m_stageIx(stageIx) {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueLocalCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override
    {
        const auto* glppln = m_pipeline.get();
        glppln->afterUniformsSet(m_stageIx, ctxid);
    }

private:
    core::smart_refctd_ptr<const GLPipelineType> m_pipeline;
    const uint32_t m_stageIx;
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
    CGetQueryBufferObjectUICmd(const uint32_t queueIdx, const bool use64Version, const GLuint queryId, const GLuint buffer, const GLenum pname, const GLintptr offset)
        : m_queueIdx(queueIdx), m_use64Version(use64Version), m_queryId(queryId), m_buffer(buffer), m_pname(pname), m_offset(offset)
    {}

    void operator()(IOpenGL_FunctionTable* gl, SQueueLocalCache& queueCache, const uint32_t ctxid, const system::logger_opt_ptr logger) override;

private:
    const uint32_t m_queueIdx;
    const bool m_use64Version;

    const GLuint m_queryId;
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

}


#endif

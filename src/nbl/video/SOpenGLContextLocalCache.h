#ifndef _NBL_S_OPENGL_CONTEXT_LOCAL_CACHE_H_INCLUDED_
#define _NBL_S_OPENGL_CONTEXT_LOCAL_CACHE_H_INCLUDED_

#include "nbl/video/SOpenGLState.h"
#include "nbl/core/containers/LRUCache.h"

namespace nbl::video
{

namespace impl
{
    // GCC is special
    template<asset::E_PIPELINE_BIND_POINT>
    struct pipeline_for_bindpoint;
    template<> struct pipeline_for_bindpoint<asset::EPBP_COMPUTE>  { using type = COpenGLComputePipeline; };
    template<> struct pipeline_for_bindpoint<asset::EPBP_GRAPHICS> { using type = COpenGLRenderpassIndependentPipeline; };

    template<asset::E_PIPELINE_BIND_POINT PBP>
    using pipeline_for_bindpoint_t = typename pipeline_for_bindpoint<PBP>::type;
}

struct SOpenGLContextLocalCache
{
    enum GL_STATE_BITS : uint32_t
    {
        // has to be flushed before constants are pushed (before `extGlProgramUniform*`)
        GSB_PIPELINE = 1u << 0,
        GSB_FRAMEBUFFER = 1u << 1,
        GSB_RASTER_PARAMETERS = 1u << 2,
        // we want the two to happen together and just before a draw (set VAO first, then binding)
        GSB_VAO_AND_VERTEX_INPUT = 1u << 3,
        // flush just before dispatches or drawcalls
        GSB_DESCRIPTOR_SETS = 1u << 4,
        // GL_DISPATCH_INDIRECT_BUFFER 
        GSB_DISPATCH_INDIRECT = 1u << 5,
        GSB_PUSH_CONSTANTS = 1u << 6,
        GSB_PIXEL_PACK_UNPACK = 1u << 7,
        // flush everything
        GSB_ALL = ~0x0u
    };

    struct SPipelineCacheVal
    {
        GLuint GLname;
    };

    struct SBeforeClearStateBackup
    {
        GLuint stencilWrite_front;
        GLuint stencilWrite_back;
        GLboolean depthWrite;
        GLboolean colorWrite[4];
    };

    using vao_cache_t = core::LRUCache<SOpenGLState::SVAOCacheKey, GLuint, SOpenGLState::SVAOCacheKey::hash>;
    using pipeline_cache_t = core::LRUCache<SOpenGLState::SGraphicsPipelineHash, SPipelineCacheVal, SOpenGLState::SGraphicsPipelineHashFunc>;
    using fbo_cache_t = core::LRUCache<SOpenGLState::SFBOHash, GLuint, SOpenGLState::SFBOHashFunc>;

    static inline constexpr size_t maxVAOCacheSize = 0x1ull << 10; //make this cache configurable
    static inline constexpr size_t maxPipelineCacheSize = 0x1ull << 13;//8k
    static inline constexpr size_t maxFBOCacheSize = 0x1ull << 8; // 256

    struct VAODisposalFunc
    {
        VAODisposalFunc(IOpenGL_FunctionTable* _gl) : gl(_gl)
#ifdef _NBL_DEBUG
            , tid(std::this_thread::get_id())
#endif
        {}

        IOpenGL_FunctionTable* gl;
#ifdef _NBL_DEBUG
        const std::thread::id tid;
#endif

        void operator()(vao_cache_t::assoc_t& x) const
        {
#ifdef _NBL_DEBUG
            assert(std::this_thread::get_id() == tid);
            if (std::this_thread::get_id() == tid)
#endif
                gl->glVertex.pglDeleteVertexArrays(1, &x.second);
        }
    };
    struct PipelineDisposalFunc
    {
        PipelineDisposalFunc(IOpenGL_FunctionTable* _gl) : gl(_gl)
#ifdef _NBL_DEBUG
            , tid(std::this_thread::get_id())
#endif
        {}

        IOpenGL_FunctionTable* gl;
#ifdef _NBL_DEBUG
        const std::thread::id tid;
#endif

        void operator()(pipeline_cache_t::assoc_t& x) const
        {
#ifdef _NBL_DEBUG
            assert(std::this_thread::get_id() == tid);
            if (std::this_thread::get_id() == tid)
#endif
                gl->glShader.pglDeleteProgramPipelines(1, &x.second.GLname);
        }
    };
    struct FBODisposalFunc
    {
        FBODisposalFunc(IOpenGL_FunctionTable* _gl) : gl(_gl)
#ifdef _NBL_DEBUG
            , tid(std::this_thread::get_id())
#endif
        {}

        IOpenGL_FunctionTable* gl;
#ifdef _NBL_DEBUG
        const std::thread::id tid;
#endif

        void operator()(fbo_cache_t::assoc_t& x) const
        {
#ifdef _NBL_DEBUG
            assert(std::this_thread::get_id() == tid);
            if (std::this_thread::get_id() == tid)
#endif
                gl->glFramebuffer.pglDeleteFramebuffers(1, &x.second);
        }
    };

    SOpenGLContextLocalCache()
        : VAOMap(maxVAOCacheSize, vao_cache_t::disposal_func_t(VAODisposalFunc(nullptr))),
        GraphicsPipelineMap(maxPipelineCacheSize, pipeline_cache_t::disposal_func_t(PipelineDisposalFunc(nullptr))),
        FBOCache(maxFBOCacheSize, fbo_cache_t::disposal_func_t(FBODisposalFunc(nullptr)))
    {}

    explicit SOpenGLContextLocalCache(IOpenGL_FunctionTable* _gl) : 
        VAOMap(maxVAOCacheSize, vao_cache_t::disposal_func_t(VAODisposalFunc(_gl))),
        GraphicsPipelineMap(maxPipelineCacheSize, pipeline_cache_t::disposal_func_t(PipelineDisposalFunc(_gl))),
        FBOCache(maxFBOCacheSize, fbo_cache_t::disposal_func_t(FBODisposalFunc(_gl)))
    {
    }

    SOpenGLState currentState;
    SOpenGLState nextState;
    // represents descriptors currently flushed into GL state,
    // layout is needed to disambiguate descriptor sets due to translation into OpenGL descriptor indices
    struct {
        SOpenGLState::SDescSetBnd descSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
        core::smart_refctd_ptr<const COpenGLPipelineLayout> layout;
    } effectivelyBoundDescriptors;

    impl::pipeline_for_bindpoint_t<asset::EPBP_COMPUTE>::PushConstantsState pushConstantsStateCompute;
    impl::pipeline_for_bindpoint_t<asset::EPBP_GRAPHICS>::PushConstantsState pushConstantsStateGraphics;

    //push constants are tracked outside of next/currentState because there can be multiple pushConstants() calls and each of them kinda depends on the pervious one (layout compatibility)
    template<asset::E_PIPELINE_BIND_POINT PBP>
    typename impl::pipeline_for_bindpoint_t<PBP>::PushConstantsState* pushConstantsState()
    {
        if constexpr (PBP == asset::EPBP_COMPUTE)
            return &pushConstantsStateCompute;
        else if (PBP == asset::EPBP_GRAPHICS)
            return &pushConstantsStateGraphics;
        else
            return nullptr;
    }

    vao_cache_t VAOMap;
    pipeline_cache_t GraphicsPipelineMap;
    fbo_cache_t FBOCache;

    inline void removePipelineEntry(IOpenGL_FunctionTable* gl, const SOpenGLState::SGraphicsPipelineHash& key)
    {
        SOpenGLContextLocalCache::SPipelineCacheVal* found = GraphicsPipelineMap.peek(key);
        if (found)
        {
            GLuint GLname = found->GLname;

            GraphicsPipelineMap.erase(key);

            if (currentState.pipeline.graphics.usedPipeline == GLname)
            {
                currentState.pipeline.graphics.pipeline = nullptr;
                currentState.pipeline.graphics.usedPipeline = 0u;
                memset(currentState.pipeline.graphics.usedShadersHash.data(), 0, sizeof(SOpenGLState::SGraphicsPipelineHash));
            }
        }
    }
    inline void removeFBOEntry(IOpenGL_FunctionTable* gl, const SOpenGLState::SFBOHash& key)
    {
        GLuint* found = FBOCache.peek(key);
        if (found)
        {
            GLuint GLname = found[0];

            FBOCache.erase(key);

            // for safety
            if (currentState.framebuffer.GLname == GLname)
            {
                currentState.framebuffer.GLname = 0u;
                memset(currentState.framebuffer.hash.data(), 0, sizeof(SOpenGLState::SFBOHash));
            }
        }
    }

    inline GLuint getSingleColorAttachmentFBO(IOpenGL_FunctionTable* gl, const IGPUImage* img, uint32_t mip, int32_t layer)
    {
        auto hash = COpenGLFramebuffer::getHashColorImage(img, mip, layer);
        auto found = FBOCache.get(hash);
        if (found)
            return *found;
        GLuint fbo = COpenGLFramebuffer::getNameColorImage(gl, img, mip, layer);
        if (!fbo)
            return 0u;
        FBOCache.insert(hash, fbo);
        return fbo;
    }
    inline GLuint getDepthStencilAttachmentFBO(IOpenGL_FunctionTable* gl, const IGPUImage* img, uint32_t mip, int32_t layer)
    {
        auto hash = COpenGLFramebuffer::getHashDepthStencilImage(img, mip, layer);
        auto found = FBOCache.get(hash);
        if (found)
            return *found;
        GLuint fbo = COpenGLFramebuffer::getNameDepthStencilImage(gl, img, mip, layer);
        if (!fbo)
            return 0u;
        FBOCache.insert(hash, fbo);
        return fbo;
    }

    void updateNextState_pipelineAndRaster(IOpenGL_FunctionTable* gl, const IGPUGraphicsPipeline* _pipeline, uint32_t ctxid);

    template<asset::E_PIPELINE_BIND_POINT PBP>
    inline void pushConstants(const COpenGLPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values)
    {
        //validation is done in pushConstants_validate() of command buffer GL impl (COpenGLCommandBuffer/COpenGLPrimaryCommandBuffer)
        //if arguments were invalid (dont comply Valid Usage section of vkCmdPushConstants docs), execution should not even get to this point

        if (pushConstantsState<PBP>()->layout && !pushConstantsState<PBP>()->layout->isCompatibleForPushConstants(_layout))
        {
            //#ifdef _NBL_DEBUG
            constexpr size_t toFill = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE / sizeof(uint64_t);
            constexpr size_t bytesLeft = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE - (toFill * sizeof(uint64_t));
            constexpr uint64_t pattern = 0xdeadbeefDEADBEEFull;
            std::fill(reinterpret_cast<uint64_t*>(pushConstantsState<PBP>()->data), reinterpret_cast<uint64_t*>(pushConstantsState<PBP>()->data) + toFill, pattern);
            if constexpr (bytesLeft > 0ull)
                memcpy(reinterpret_cast<uint64_t*>(pushConstantsState<PBP>()->data) + toFill, &pattern, bytesLeft);
            //#endif

            _stages |= IGPUShader::ESS_ALL;
        }
        pushConstantsState<PBP>()->incrementStamps(_stages);

        pushConstantsState<PBP>()->layout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(_layout);
        memcpy(pushConstantsState<PBP>()->data + _offset, _values, _size);
    }

    // state flushing 
    void flushStateGraphics(IOpenGL_FunctionTable* gl, uint32_t stateBits, uint32_t ctxid); // TODO(achal): Temporary, just to make the old code still work.
    bool flushStateGraphics(const uint32_t stateBits, IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment::Iterator& segmentListHeadItr, IGPUCommandPool::CCommandSegment*& segmentListTail, const E_API_TYPE apiType, const COpenGLFeatureMap* features);
    void flushStateCompute(IOpenGL_FunctionTable* gl, uint32_t stateBits, uint32_t ctxid); // TODO(achal): Temporary, just to make the old code still work.
    bool flushStateCompute(uint32_t stateBits, IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment::Iterator& segmentListHeadItr, IGPUCommandPool::CCommandSegment*& segmentListTail, const COpenGLFeatureMap* features);

    inline SBeforeClearStateBackup backupAndFlushStateClear(IOpenGL_FunctionTable* gl, uint32_t ctxid, bool color, bool depth, bool stencil)
    {
        SBeforeClearStateBackup backup;
        memcpy(backup.colorWrite, currentState.rasterParams.drawbufferBlend[0].colorMask.colorWritemask, 4);
        backup.depthWrite = currentState.rasterParams.depthWriteEnable;
        backup.stencilWrite_back = currentState.rasterParams.stencilFunc_back.mask;
        backup.stencilWrite_front = currentState.rasterParams.stencilFunc_front.mask;
        //TODO dithering (? i think vulkan doesnt have dithering at all), scissor test (COpenGLCommandBuffer impl doesnt support scissor yet)
        // "The pixel ownership test, the scissor test, dithering, and the buffer writemasks affect the operation of glClear."

        if (color)
            std::fill_n(nextState.rasterParams.drawbufferBlend[0].colorMask.colorWritemask, 4u, GL_TRUE);
        if (depth)
            nextState.rasterParams.depthWriteEnable = GL_TRUE;
        if (stencil)
        {
            nextState.rasterParams.stencilFunc_back.mask = ~0u;
            nextState.rasterParams.stencilFunc_front.mask = ~0u;
        }

        flushStateGraphics(gl, GSB_RASTER_PARAMETERS, ctxid);

        return backup;
    }
    // TODO(achal): Temporary.
    inline SBeforeClearStateBackup backupAndFlushStateClear2(IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment::Iterator& segmentListHeadItr, IGPUCommandPool::CCommandSegment*& segmentListTail, const bool color, const bool depth, const bool stencil, const E_API_TYPE apiType, const COpenGLFeatureMap* features)
    {
        SBeforeClearStateBackup backup;
        memcpy(backup.colorWrite, currentState.rasterParams.drawbufferBlend[0].colorMask.colorWritemask, 4);
        backup.depthWrite = currentState.rasterParams.depthWriteEnable;
        backup.stencilWrite_back = currentState.rasterParams.stencilFunc_back.mask;
        backup.stencilWrite_front = currentState.rasterParams.stencilFunc_front.mask;
        //TODO dithering (? i think vulkan doesnt have dithering at all), scissor test (COpenGLCommandBuffer impl doesnt support scissor yet)
        // "The pixel ownership test, the scissor test, dithering, and the buffer writemasks affect the operation of glClear."

        if (color)
            std::fill_n(nextState.rasterParams.drawbufferBlend[0].colorMask.colorWritemask, 4u, GL_TRUE);
        if (depth)
            nextState.rasterParams.depthWriteEnable = GL_TRUE;
        if (stencil)
        {
            nextState.rasterParams.stencilFunc_back.mask = ~0u;
            nextState.rasterParams.stencilFunc_front.mask = ~0u;
        }

        flushStateGraphics(GSB_RASTER_PARAMETERS, cmdpool, segmentListHeadItr, segmentListTail, apiType, features);

        return backup;
    }
    inline void restoreStateAfterClear(const SBeforeClearStateBackup& backup)
    {
        memcpy(nextState.rasterParams.drawbufferBlend[0].colorMask.colorWritemask, backup.colorWrite, 4);
        nextState.rasterParams.depthWriteEnable = backup.depthWrite;
        nextState.rasterParams.stencilFunc_back.mask = backup.stencilWrite_back;
        nextState.rasterParams.stencilFunc_front.mask = backup.stencilWrite_front;
    }

private:
    uint64_t m_timestampCounter = 0u;

    void flushState_descriptors(IOpenGL_FunctionTable* gl, asset::E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout); // TODO(achal): Temporary.
    bool flushState_descriptors(asset::E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout, IGPUCommandPool* cmdpool, IGPUCommandPool::CCommandSegment::Iterator& segmentListHeadItr, IGPUCommandPool::CCommandSegment*& segmentListTail, const COpenGLFeatureMap* features);
    GLuint createGraphicsPipeline(IOpenGL_FunctionTable* gl, const SOpenGLState::SGraphicsPipelineHash& _hash);

    static inline GLenum getGLpolygonMode(asset::E_POLYGON_MODE pm)
    {
        const static GLenum glpm[3]{ GL_FILL, GL_LINE, GL_POINT };
        return glpm[pm];
    }
    static inline GLenum getGLcullFace(asset::E_FACE_CULL_MODE cf)
    {
        const static GLenum glcf[4]{ 0, GL_FRONT, GL_BACK, GL_FRONT_AND_BACK };
        return glcf[cf];
    }
    static inline GLenum getGLstencilOp(asset::E_STENCIL_OP so)
    {
        static const GLenum glso[]{ GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_DECR, GL_INVERT, GL_INCR_WRAP, GL_DECR_WRAP };
        return glso[so];
    }
    static inline GLenum getGLcmpFunc(asset::E_COMPARE_OP sf)
    {
        static const GLenum glsf[]{ GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS };
        return glsf[sf];
    }
    static inline GLenum getGLlogicOp(asset::E_LOGIC_OP lo)
    {
        static const GLenum gllo[]{ GL_CLEAR, GL_AND, GL_AND_REVERSE, GL_COPY, GL_AND_INVERTED, GL_NOOP, GL_XOR, GL_OR, GL_NOR, GL_EQUIV, GL_INVERT, GL_OR_REVERSE,
            GL_COPY_INVERTED, GL_OR_INVERTED, GL_NAND, GL_SET
        };
        return gllo[lo];
    }
    static inline GLenum getGLblendFunc(asset::E_BLEND_FACTOR bf)
    {
        static const GLenum glbf[]{ GL_ZERO , GL_ONE, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA,
            GL_SRC_ALPHA_SATURATE, GL_SRC1_COLOR, GL_ONE_MINUS_SRC1_COLOR, GL_SRC1_ALPHA, GL_ONE_MINUS_SRC1_ALPHA
        };
        return glbf[bf];
    }
    static inline GLenum getGLblendEq(asset::E_BLEND_OP bo)
    {
        GLenum glbo[]{ GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX };
        if (bo >= std::extent<decltype(glbo)>::value)
            return GL_INVALID_ENUM;
        return glbo[bo];
    }
};

}

#endif

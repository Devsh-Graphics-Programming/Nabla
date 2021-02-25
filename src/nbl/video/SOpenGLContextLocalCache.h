#ifndef __NBL_S_OPENGL_CONTEXT_LOCAL_CACHE_H_INCLUDED__
#define __NBL_S_OPENGL_CONTEXT_LOCAL_CACHE_H_INCLUDED__

#include "nbl/video/SOpenGLState.h"

namespace nbl {
namespace video
{

namespace impl
{
    // GCC is special
    template<asset::E_PIPELINE_BIND_POINT>
    struct pipeline_for_bindpoint;
    template<> struct pipeline_for_bindpoint<asset::EPBP_COMPUTE > { using type = COpenGLComputePipeline; };
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
        GSB_RASTER_PARAMETERS = 1u << 1,
        // we want the two to happen together and just before a draw (set VAO first, then binding)
        GSB_VAO_AND_VERTEX_INPUT = 1u << 2,
        // flush just before (indirect)dispatch or (multi)(indirect)draw, textures and samplers first, then storage image, then SSBO, finally UBO
        GSB_DESCRIPTOR_SETS = 1u << 3,
        // GL_DISPATCH_INDIRECT_BUFFER 
        GSB_DISPATCH_INDIRECT = 1u << 4,
        GSB_PUSH_CONSTANTS = 1u << 5,
        GSB_PIXEL_PACK_UNPACK = 1u << 6,
        // flush everything
        GSB_ALL = ~0x0u
    };

    struct SPipelineCacheVal
    {
        GLuint GLname;
        core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline> object;//so that it holds shaders which concerns hash
        uint64_t lastUsed;
    };

    static inline constexpr size_t maxVAOCacheSize = 0x1ull << 10; //make this cache configurable
    static inline constexpr size_t maxPipelineCacheSize = 0x1ull << 13;//8k

    SOpenGLContextLocalCache()
    {
        VAOMap.reserve(maxVAOCacheSize);
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

    core::vector<SOpenGLState::HashVAOPair> VAOMap;
    struct HashPipelinePair
    {
        SOpenGLState::SGraphicsPipelineHash first;
        SPipelineCacheVal second;

        inline bool operator<(const HashPipelinePair& rhs) const { return first < rhs.first; }
    };
    core::vector<HashPipelinePair> GraphicsPipelineMap;


    void updateNextState_pipelineAndRaster(const IGPURenderpassIndependentPipeline* _pipeline, uint32_t ctxid);

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

            _stages |= IGPUSpecializedShader::ESS_ALL;
        }
        pushConstantsState<PBP>()->incrementStamps(_stages);

        pushConstantsState<PBP>()->layout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(_layout);
        memcpy(pushConstantsState<PBP>()->data + _offset, _values, _size);
    }

    // state flushing 
    void flushStateGraphics(IOpenGL_FunctionTable* gl, uint32_t stateBits, uint32_t ctxid);
    void flushStateCompute(IOpenGL_FunctionTable* gl, uint32_t stateBits, uint32_t ctxid);

private:
    void flushState_descriptors(IOpenGL_FunctionTable* gl, asset::E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout);
    GLuint createGraphicsPipeline(IOpenGL_FunctionTable* gl, const SOpenGLState::SGraphicsPipelineHash& _hash);

    inline void freeUpVAOCache(IOpenGL_FunctionTable* gl, bool exitOnFirstDelete)
    {
        for (auto it = VAOMap.begin(); VAOMap.size() > maxVAOCacheSize && it != VAOMap.end();)
        {
            if (it->first == currentState.vertexInputParams.vao.first)
                continue;

            if (CNullDriver::ReallocationCounter - it->second.lastUsed > 1000) //maybe make this configurable
            {
                gl->glVertex.pglDeleteVertexArrays(1, &it->second.GLname);
                it = VAOMap.erase(it);
                if (exitOnFirstDelete)
                    return;
            }
            else
                it++;
        }
    }
    //TODO DRY
    inline void freeUpGraphicsPipelineCache(IOpenGL_FunctionTable* gl, bool exitOnFirstDelete)
    {
        for (auto it = GraphicsPipelineMap.begin(); GraphicsPipelineMap.size() > maxPipelineCacheSize && it != GraphicsPipelineMap.end();)
        {
            if (it->first == currentState.pipeline.graphics.usedShadersHash)
                continue;

            if (CNullDriver::ReallocationCounter - it->second.lastUsed > 1000) //maybe make this configurable
            {
                gl->glShader.pglDeleteProgramPipelines(1, &it->second.GLname);
                it = GraphicsPipelineMap.erase(it);
                if (exitOnFirstDelete)
                    return;
            }
            else
                it++;
        }
    }

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
}

#endif

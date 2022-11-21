// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl/asset/utils/CGLSLCompiler.h"
#include "nbl/asset/utils/CShaderIntrospector.h"
#include "nbl/asset/utils/spvUtils.h"

#ifdef OLD_CODE

#include "nbl/core/core.h"
#include "nbl/system/compile_config.h"

#include "SIrrCreationParameters.h"

namespace nbl
{
	class CIrrDeviceStub;
}

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "EGL/egl.h"

#include "IDeviceMemoryAllocation.h"
#include "nbl/video/COpenGLSpecializedShader.h"
#include "nbl/video/COpenGLRenderpassIndependentPipeline.h"
#include "nbl/video/COpenGLDescriptorSet.h"
#include "nbl/video/COpenGLPipelineLayout.h"
#include "nbl/video/COpenGLComputePipeline.h"

#include "CNullDriver.h"
// also includes the OpenGL stuff
#include "COpenGLFrameBuffer.h"
#include "COpenGLDriverFence.h"
#include "nbl/video/CCUDAHandler.h"
#include "COpenCLHandler.h"

namespace nbl::video
{


struct SOpenGLState
{
    struct SVAO {
        GLuint GLname;
        uint64_t lastUsed;
    };
    struct HashVAOPair
    {
		COpenGLRenderpassIndependentPipeline::SVAOHash first = {};
		SVAO second = { 0u,0ull };
		//extra vao state being cached
		std::array<asset::SBufferBinding<const COpenGLBuffer>, IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT> vtxBindings;
		core::smart_refctd_ptr<const COpenGLBuffer> idxBinding;

        inline bool operator<(const HashVAOPair& rhs) const { return first < rhs.first; }
    };

    using SGraphicsPipelineHash = std::array<GLuint, COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT>;

    struct {
        struct {
            core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline> pipeline;
            SGraphicsPipelineHash usedShadersHash = { 0u, 0u, 0u, 0u, 0u };
			GLuint usedPipeline = 0u;
        } graphics;
        struct {
            core::smart_refctd_ptr<const COpenGLComputePipeline> pipeline;
            GLuint usedShader = 0u;
        } compute;
    } pipeline;

    struct {
        core::smart_refctd_ptr<const COpenGLBuffer> buffer;
    } dispatchIndirect;

    struct {
        //in GL it is possible to set polygon mode separately for back- and front-faces, but in VK it's one setting for both
        GLenum polygonMode = GL_FILL;
        GLenum faceCullingEnable = 0;
        GLenum cullFace = GL_BACK;
        //in VK stencil params (both: stencilOp and stencilFunc) are 2 distinct for back- and front-faces, but in GL it's one for both
        struct SStencilOp {
            GLenum sfail = GL_KEEP;
            GLenum dpfail = GL_KEEP;
            GLenum dppass = GL_KEEP;
            bool operator!=(const SStencilOp& rhs) const { return sfail!=rhs.sfail || dpfail!=rhs.dpfail || dppass!=rhs.dppass; }
        };
        SStencilOp stencilOp_front, stencilOp_back;
        struct SStencilFunc {
            GLenum func = GL_ALWAYS;
            GLint ref = 0;
            GLuint mask = ~static_cast<GLuint>(0u);
            bool operator!=(const SStencilFunc& rhs) const { return func!=rhs.func || ref!=rhs.ref || mask!=rhs.mask; }
        };
        SStencilFunc stencilFunc_front, stencilFunc_back;
        GLenum depthFunc = GL_LESS;
        GLenum frontFace = GL_CCW;
        GLboolean depthClampEnable = 0;
        GLboolean rasterizerDiscardEnable = 0;
        GLboolean polygonOffsetEnable = 0;
        struct SPolyOffset {
            GLfloat factor = 0.f;//depthBiasSlopeFactor 
            GLfloat units = 0.f;//depthBiasConstantFactor 
            bool operator!=(const SPolyOffset& rhs) const { return factor!=rhs.factor || units!=rhs.units; }
        } polygonOffset;
        GLfloat lineWidth = 1.f;
        GLboolean sampleShadingEnable = 0;
        GLfloat minSampleShading = 0.f;
        GLboolean sampleMaskEnable = 0;
        GLbitfield sampleMask[2]{~static_cast<GLbitfield>(0), ~static_cast<GLbitfield>(0)};
        GLboolean sampleAlphaToCoverageEnable = 0;
        GLboolean sampleAlphaToOneEnable = 0;
        GLboolean depthTestEnable = 0;
        GLboolean depthWriteEnable = 1;
        //GLboolean depthBoundsTestEnable;
        GLboolean stencilTestEnable = 0;
        GLboolean multisampleEnable = 1;
        GLboolean primitiveRestartEnable = 0;

        GLboolean logicOpEnable = 0;
        GLenum logicOp = GL_COPY;
        struct SDrawbufferBlending
        {
            GLboolean blendEnable = 0;
            struct SBlendFunc {
                GLenum srcRGB = GL_ONE;
                GLenum dstRGB = GL_ZERO;
                GLenum srcAlpha = GL_ONE;
                GLenum dstAlpha = GL_ZERO;
                bool operator!=(const SBlendFunc& rhs) const { return srcRGB!=rhs.srcRGB || dstRGB!=rhs.dstRGB || srcAlpha!=rhs.srcAlpha || dstAlpha!=rhs.dstAlpha; }
            } blendFunc;
            struct SBlendEq {
                GLenum modeRGB = GL_FUNC_ADD;
                GLenum modeAlpha = GL_FUNC_ADD;
                bool operator!=(const SBlendEq& rhs) const { return modeRGB!=rhs.modeRGB || modeAlpha!=rhs.modeAlpha; }
            } blendEquation;
            struct SColorWritemask {
                GLboolean colorWritemask[4]{ 1,1,1,1 };
                bool operator!=(const SColorWritemask& rhs) const { return memcmp(colorWritemask, rhs.colorWritemask, 4); }
            } colorMask;
        } drawbufferBlend[asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT];
    } rasterParams;

    struct {
		HashVAOPair vao = {};

        //putting it here because idk where else
        core::smart_refctd_ptr<const COpenGLBuffer> indirectDrawBuf;
        core::smart_refctd_ptr<const COpenGLBuffer> parameterBuf;//GL>=4.6
    } vertexInputParams;

    struct {
        SDescSetBnd descSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
    } descriptorsParams[E_PIPELINE_BIND_POINT::EPBP_COUNT];

    struct SPixelPackUnpack {
        core::smart_refctd_ptr<const COpenGLBuffer> buffer;
        GLint alignment = 4;
        GLint rowLength = 0;
        GLint imgHeight = 0;
        GLint BCwidth = 0;
        GLint BCheight = 0;
        GLint BCdepth = 0;
    };
    SPixelPackUnpack pixelPack;
    SPixelPackUnpack pixelUnpack;
};



class COpenGLDriver final : public CNullDriver, public COpenGLExtensionHandler
{
        uint16_t retrieveDisplayRefreshRate() const override;

		void fillBuffer(IGPUBuffer* buffer, size_t offset, size_t length, uint32_t value) override;

		void copyImage(IGPUImage* srcImage, IGPUImage* dstImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions) override;

		//!
		virtual void issueGPUTextureBarrier() {COpenGLExtensionHandler::extGlTextureBarrier();}


    private:
        void clearColor_gatherAndOverrideState(SAuxContext* found, uint32_t _attIx, GLboolean* _rasterDiscard, GLboolean* _colorWmask);
        void clearColor_bringbackState(SAuxContext* found, uint32_t _attIx, GLboolean _rasterDiscard, const GLboolean* _colorWmask);

    public:
		virtual void clearStencilBuffer(const int32_t &stencil);

		virtual void clearZStencilBuffers(const float &depth, const int32_t &stencil);

		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals);

		//! Enable/disable a clipping plane.
		//! There are at least 6 clipping planes available for the user to set at will.
		//! \param index: The plane index. Must be between 0 and MaxUserClipPlanes.
		//! \param enable: If true, enable the clipping plane else disable it.
		virtual void enableClipPlane(uint32_t index, bool enable);

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo() {return VendorName;}

		//!
		const size_t& getMaxConcurrentShaderInvocations() const {return maxConcurrentShaderInvocations;}

		//!
		const uint32_t& getMaxShaderComputeUnits() const {return maxShaderComputeUnits;}

		//!
		const size_t& getMaxShaderInvocationsPerALU() const {return maxALUShaderInvocations;}

        struct SAuxContext
        {
        //public:
            struct SPipelineCacheVal
            {
                GLuint GLname;
                core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline> object;//so that it holds shaders which concerns hash
                uint64_t lastUsed;
            };

            _NBL_STATIC_INLINE_CONSTEXPR size_t maxVAOCacheSize = 0x1u<<10; //make this cache configurable
            _NBL_STATIC_INLINE_CONSTEXPR size_t maxPipelineCacheSize = 0x1u<<13;//8k

            SAuxContext() : threadId(std::thread::id()), ctx(NULL),
                            CurrentFBO(0), CurrentRendertargetSize(0,0)
            {
                VAOMap.reserve(maxVAOCacheSize);
            }

            void flushState_descriptors(E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout);
            void flushStateGraphics(uint32_t stateBits);
            void flushStateCompute(uint32_t stateBits);

            SOpenGLState currentState;
            SOpenGLState nextState;
			// represents descriptors currently flushed into GL state,
			// layout is needed to disambiguate descriptor sets due to translation into OpenGL descriptor indices
            struct {
                SOpenGLState::SDescSetBnd descSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
				core::smart_refctd_ptr<const COpenGLPipelineLayout> layout;
            } effectivelyBoundDescriptors;


        //private:
            //! FBOs
            core::vector<IFrameBuffer*>  FrameBuffers;
            COpenGLFrameBuffer*         CurrentFBO;
            core::dimension2d<uint32_t> CurrentRendertargetSize; // @Crisspl TODO: Fold this into SOpenGLState, as well as the Vulkan dynamic state (scissor rect, viewport, etc.)

            //!
            core::vector<SOpenGLState::HashVAOPair> VAOMap;
            struct HashPipelinePair
            {
                SOpenGLState::SGraphicsPipelineHash first;
                SPipelineCacheVal second;

                inline bool operator<(const HashPipelinePair& rhs) const { return first < rhs.first; }
            };
            core::vector<HashPipelinePair> GraphicsPipelineMap;

            GLuint createGraphicsPipeline(const SOpenGLState::SGraphicsPipelineHash& _hash);

            void updateNextState_pipelineAndRaster(const IGPURenderpassIndependentPipeline* _pipeline);
            //! Must be called AFTER updateNextState_pipelineAndRaster() if pipeline and raster params have to be modified at all in this pass
            void updateNextState_vertexInput(
                const asset::SBufferBinding<const IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                const IGPUBuffer* _indexBuffer,
                const IGPUBuffer* _indirectDrawBuffer,
                const IGPUBuffer* _paramBuffer
            );

            inline size_t getVAOCacheSize() const
            {
                return VAOMap.size();
            }

            inline void freeUpVAOCache(bool exitOnFirstDelete)
            {
                for(auto it = VAOMap.begin(); VAOMap.size()>maxVAOCacheSize&&it!=VAOMap.end();)
                {
                    if (it->first==currentState.vertexInputParams.vao.first)
                        continue;

                    if (CNullDriver::ReallocationCounter-it->second.lastUsed>1000) //maybe make this configurable
                    {
                        COpenGLExtensionHandler::extGlDeleteVertexArrays(1, &it->second.GLname);
                        it = VAOMap.erase(it);
                        if (exitOnFirstDelete)
                            return;
                    }
                    else
                        it++;
                }
            }
            //TODO DRY
            inline void freeUpGraphicsPipelineCache(bool exitOnFirstDelete)
            {
                for (auto it = GraphicsPipelineMap.begin(); GraphicsPipelineMap.size() > maxPipelineCacheSize&&it != GraphicsPipelineMap.end();)
                {
                    if (it->first == currentState.pipeline.graphics.usedShadersHash)
                        continue;

                    if (CNullDriver::ReallocationCounter-it->second.lastUsed > 1000) //maybe make this configurable
                    {
                        COpenGLExtensionHandler::extGlDeleteProgramPipelines(1, &it->second.GLname);
                        it = GraphicsPipelineMap.erase(it);
                        if (exitOnFirstDelete)
                            return;
                    }
                    else
                        it++;
                }
            }
        };


		//!
		virtual uint32_t getRequiredUBOAlignment() const {return COpenGLExtensionHandler::reqUBOAlignment;}

		//!
		virtual uint32_t getRequiredSSBOAlignment() const {return COpenGLExtensionHandler::reqSSBOAlignment;}

		//!
		virtual uint32_t getRequiredTBOAlignment() const {return COpenGLExtensionHandler::reqTBOAlignment;}

		//!
		virtual uint32_t getMinimumMemoryMapAlignment() const {return COpenGLExtensionHandler::minMemoryMapAlignment;}

        //!
        virtual uint32_t getMaxComputeWorkGroupSize(uint32_t _dimension) const { return COpenGLExtensionHandler::MaxComputeWGSize[_dimension]; }

        //!
        virtual uint64_t getMaxUBOSize() const override { return COpenGLExtensionHandler::maxUBOSize; }

        //!
        virtual uint64_t getMaxSSBOSize() const override { return COpenGLExtensionHandler::maxSSBOSize; }

        //!
        virtual uint64_t getMaxTBOSizeInTexels() const override { return COpenGLExtensionHandler::maxTBOSizeInTexels; }

        //!
        virtual uint64_t getMaxBufferSize() const override { return COpenGLExtensionHandler::maxBufferSize; }

        uint32_t getMaxUBOBindings() const override { return COpenGLExtensionHandler::maxUBOBindings; }
        uint32_t getMaxSSBOBindings() const override { return COpenGLExtensionHandler::maxSSBOBindings; }
        uint32_t getMaxTextureBindings() const override { return COpenGLExtensionHandler::maxTextureBindings; }
        uint32_t getMaxTextureBindingsCompute() const override { return COpenGLExtensionHandler::maxTextureBindingsCompute; }
        uint32_t getMaxImageBindings() const override { return COpenGLExtensionHandler::maxImageBindings; }

		//!
		bool runningInRenderdoc() const { return runningInRenderDoc; }

    private:
        SAuxContext* getThreadContext_helper(const bool& alreadyLockedMutex, const std::thread::id& tid = std::this_thread::get_id());

        void cleanUpContextBeforeDelete();


        //COpenGLDriver::CGPUObjectFromAssetConverter
        class CGPUObjectFromAssetConverter;
        friend class CGPUObjectFromAssetConverter;

        using PipelineMapKeyT = std::pair<std::array<core::smart_refctd_ptr<IGPUSpecializedShader>, 5u>, std::thread::id>;
        core::map<PipelineMapKeyT, GLuint> Pipelines;

        bool runningInRenderDoc;

		void createMaterialRenderers();

		core::stringw Name;

		std::string VendorName;

        mutable core::smart_refctd_dynamic_array<std::string> m_supportedGLSLExtsNames;

		EGLDisplay Display;
		EGLNativeWindowType Window;

        size_t maxALUShaderInvocations;
        size_t maxConcurrentShaderInvocations;
        uint32_t maxShaderComputeUnits;

        std::mutex glContextMutex;
		SAuxContext* AuxContexts;
};

} // end namespace nbl::video


#endif // _NBL_COMPILE_WITH_OPENGL_
#endif

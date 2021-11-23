#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"

#include "nbl/video/SOpenGLContextLocalCache.h"

#include "nbl/video/IOpenGL_LogicalDevice.h"

namespace nbl::video
{

void SOpenGLContextLocalCache::updateNextState_pipelineAndRaster(const IGPUGraphicsPipeline* _pipeline, uint32_t ctxid)
{
    nextState.pipeline.graphics.pipeline = core::smart_refctd_ptr<const IGPUGraphicsPipeline>(
        _pipeline
        );
    if (!_pipeline)
    {
        SOpenGLState::SGraphicsPipelineHash hash;
        std::fill(hash.begin(), hash.end(), 0u);
        nextState.pipeline.graphics.usedShadersHash = hash;
        return;
    }

    const auto* ppln = static_cast<const COpenGLRenderpassIndependentPipeline*>(nextState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline());

    SOpenGLState::SGraphicsPipelineHash hash = ppln->getPipelineHash(ctxid);
    nextState.pipeline.graphics.usedShadersHash = hash;


    const auto& raster_src = ppln->getRasterizationParams();
    auto& raster_dst = nextState.rasterParams;

    raster_dst.polygonMode = getGLpolygonMode(raster_src.polygonMode);
    if (raster_src.faceCullingMode == asset::EFCM_NONE)
        raster_dst.faceCullingEnable = 0;
    else {
        raster_dst.faceCullingEnable = 1;
        raster_dst.cullFace = getGLcullFace(raster_src.faceCullingMode);
    }

    const asset::SStencilOpParams* stencil_src[2]{ &raster_src.frontStencilOps, &raster_src.backStencilOps };
    decltype(raster_dst.stencilOp_front)* stencilo_dst[2]{ &raster_dst.stencilOp_front, &raster_dst.stencilOp_back };
    for (uint32_t i = 0u; i < 2u; ++i) {
        stencilo_dst[i]->sfail = getGLstencilOp(stencil_src[i]->failOp);
        stencilo_dst[i]->dpfail = getGLstencilOp(stencil_src[i]->depthFailOp);
        stencilo_dst[i]->dppass = getGLstencilOp(stencil_src[i]->passOp);
    }

    decltype(raster_dst.stencilFunc_front)* stencilf_dst[2]{ &raster_dst.stencilFunc_front, &raster_dst.stencilFunc_back };
    for (uint32_t i = 0u; i < 2u; ++i) {
        stencilf_dst[i]->func = getGLcmpFunc(stencil_src[i]->compareOp);
        stencilf_dst[i]->ref = stencil_src[i]->reference;
        stencilf_dst[i]->mask = stencil_src[i]->writeMask;
    }

    raster_dst.depthFunc = getGLcmpFunc(raster_src.depthCompareOp);
    raster_dst.frontFace = raster_src.frontFaceIsCCW ? GL_CCW : GL_CW;
    raster_dst.depthClampEnable = raster_src.depthClampEnable;
    raster_dst.rasterizerDiscardEnable = raster_src.rasterizerDiscard;

    raster_dst.polygonOffsetEnable = raster_src.depthBiasEnable;
    raster_dst.polygonOffset.factor = raster_src.depthBiasSlopeFactor;
    raster_dst.polygonOffset.units = raster_src.depthBiasConstantFactor;

    raster_dst.sampleShadingEnable = raster_src.sampleShadingEnable;
    raster_dst.minSampleShading = raster_src.minSampleShading;

    //raster_dst.sampleMaskEnable = ???
    raster_dst.sampleMask[0] = raster_src.sampleMask[0];
    raster_dst.sampleMask[1] = raster_src.sampleMask[1];

    raster_dst.sampleAlphaToCoverageEnable = raster_src.alphaToCoverageEnable;
    raster_dst.sampleAlphaToOneEnable = raster_src.alphaToOneEnable;

    raster_dst.depthTestEnable = raster_src.depthTestEnable;
    raster_dst.depthWriteEnable = raster_src.depthWriteEnable;
    raster_dst.stencilTestEnable = raster_src.stencilTestEnable;

    raster_dst.multisampleEnable = (raster_src.rasterizationSamplesHint > asset::IImage::ESCF_1_BIT);

    const auto& blend_src = ppln->getBlendParams();
    raster_dst.logicOpEnable = blend_src.logicOpEnable;
    raster_dst.logicOp = getGLlogicOp(static_cast<asset::E_LOGIC_OP>(blend_src.logicOp));

    for (size_t i = 0ull; i < asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; ++i) {
        const auto& attach_src = blend_src.blendParams[i];
        auto& attach_dst = raster_dst.drawbufferBlend[i];

        attach_dst.blendEnable = attach_src.blendEnable;
        attach_dst.blendFunc.srcRGB = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.srcColorFactor));
        attach_dst.blendFunc.dstRGB = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.dstColorFactor));
        attach_dst.blendFunc.srcAlpha = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.srcAlphaFactor));
        attach_dst.blendFunc.dstAlpha = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.dstAlphaFactor));

        attach_dst.blendEquation.modeRGB = getGLblendEq(static_cast<asset::E_BLEND_OP>(attach_src.colorBlendOp));
        assert(attach_dst.blendEquation.modeRGB != GL_INVALID_ENUM);
        attach_dst.blendEquation.modeAlpha = getGLblendEq(static_cast<asset::E_BLEND_OP>(attach_src.alphaBlendOp));
        assert(attach_dst.blendEquation.modeAlpha != GL_INVALID_ENUM);

        for (uint32_t j = 0u; j < 4u; ++j)
            attach_dst.colorMask.colorWritemask[j] = (attach_src.colorWriteMask >> j) & 1u;
    }
}

void SOpenGLContextLocalCache::flushState_descriptors(IOpenGL_FunctionTable* gl, asset::E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout)
{
    const COpenGLPipelineLayout* prevLayout = effectivelyBoundDescriptors.layout.get();
    //bind new descriptor sets
    int32_t compatibilityLimit = 0u;
    if (prevLayout && _currentLayout)
        compatibilityLimit = prevLayout->isCompatibleUpToSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _currentLayout) + 1u;
    if (!prevLayout && !_currentLayout)
        compatibilityLimit = static_cast<int32_t>(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT);

    int64_t newUboCount = 0u, newSsboCount = 0u, newTexCount = 0u, newImgCount = 0u;
    if (_currentLayout)
        for (uint32_t i = 0u; i < static_cast<int32_t>(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT); ++i)
        {
            const auto& first_count = _currentLayout->getMultibindParamsForDescSet(i);

            {
                GLsizei count{};

#define CLAMP_COUNT(resname,limit,printstr) \
count = (first_count.resname.count - std::max(0, static_cast<int32_t>(first_count.resname.first + first_count.resname.count)-static_cast<int32_t>(limit)))

                CLAMP_COUNT(ubos, gl->getFeatures()->maxUBOBindings, UBO);
                newUboCount = first_count.ubos.first + count;
                CLAMP_COUNT(ssbos, gl->getFeatures()->maxSSBOBindings, SSBO);
                newSsboCount = first_count.ssbos.first + count;
                CLAMP_COUNT(textures, gl->getFeatures()->maxTextureBindings, texture); //TODO should use maxTextureBindingsCompute for compute
                newTexCount = first_count.textures.first + count;
                CLAMP_COUNT(textureImages, gl->getFeatures()->maxImageBindings, image);
                newImgCount = first_count.textureImages.first + count;
#undef CLAMP_COUNT
            }

            const auto* nextSet = nextState.descriptorsParams[_pbp].descSets[i].set.get();
            const auto dynamicOffsetCount = nextSet ? nextSet->getDynamicOffsetCount():0u;
            const auto nextStateDynamicOffsets = nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets;
            //if prev and curr pipeline layouts are compatible for set N, currState.set[N]==nextState.set[N] and the sets were bound with same dynamic offsets, then binding set N would be redundant
            if (
                    (i < compatibilityLimit) && (effectivelyBoundDescriptors.descSets[i].set.get() == nextSet) &&
                    std::equal(nextStateDynamicOffsets,nextStateDynamicOffsets+dynamicOffsetCount,effectivelyBoundDescriptors.descSets[i].dynamicOffsets) &&
                    (effectivelyBoundDescriptors.descSets[i].revision == nextState.descriptorsParams[_pbp].descSets[i].revision)
                )
            {
                continue;
            }

            const auto multibind_params = nextSet ? nextSet->getMultibindParams():COpenGLDescriptorSet::SMultibindParams{}; // all nullptr if null set

            const GLsizei localStorageImageCount = newImgCount - first_count.textureImages.first;
            if (localStorageImageCount)
            {
                assert(multibind_params.textureImages.textures);
                //formats must be provided since we dont have ARB_multi_bind on ES
                gl->extGlBindImageTextures(first_count.textureImages.first, localStorageImageCount, multibind_params.textureImages.textures, multibind_params.textureImages.formats);
            }

            const GLsizei localTextureCount = newTexCount - first_count.textures.first;
            if (localTextureCount)
            {
                assert(multibind_params.textures.textures && multibind_params.textures.samplers);
                //targets must be provided since we dont have ARB_multi_bind on ES
                gl->extGlBindTextures(first_count.textures.first, localTextureCount, multibind_params.textures.textures, multibind_params.textures.targets);
                gl->extGlBindSamplers(first_count.textures.first, localTextureCount, multibind_params.textures.samplers);
            }

            //not entirely sure those MAXes are right
            constexpr size_t MAX_UBO_COUNT = 96ull;
            constexpr size_t MAX_SSBO_COUNT = 91ull;
            constexpr size_t MAX_OFFSETS = MAX_UBO_COUNT > MAX_SSBO_COUNT ? MAX_UBO_COUNT : MAX_SSBO_COUNT;
            GLintptr offsetsArray[MAX_OFFSETS]{};
            GLintptr sizesArray[MAX_OFFSETS]{};

            const GLsizei localSsboCount = newSsboCount - first_count.ssbos.first;//"local" as in this DS
            if (localSsboCount)
            {
                if (nextSet)
                    for (GLsizei s = 0u; s < localSsboCount; ++s)
                    {
                        offsetsArray[s] = multibind_params.ssbos.offsets[s];
                        sizesArray[s] = multibind_params.ssbos.sizes[s];
                        //if it crashes below, it means that there are dynamic Buffer Objects in the DS, but the DS was bound with no (or not enough) dynamic offsets
                        //or for some weird reason (bug) descSets[i].set is nullptr, but descSets[i].dynamicOffsets is not
                        if (multibind_params.ssbos.dynOffsetIxs[s] < dynamicOffsetCount)
                            offsetsArray[s] += nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets[multibind_params.ssbos.dynOffsetIxs[s]];
                        if (sizesArray[s] == IGPUBufferView::whole_buffer)
                            sizesArray[s] = nextSet->getSSBO(s)->getSize() - offsetsArray[s];
                    }
                assert(multibind_params.ssbos.buffers);
                gl->extGlBindBuffersRange(GL_SHADER_STORAGE_BUFFER, first_count.ssbos.first, localSsboCount, multibind_params.ssbos.buffers, nextSet ? offsetsArray : nullptr, nextSet ? sizesArray : nullptr);
            }

            const GLsizei localUboCount = (newUboCount - first_count.ubos.first);//"local" as in this DS
            if (localUboCount)
            {
                if (nextSet)
                    for (GLsizei s = 0u; s < localUboCount; ++s)
                    {
                        offsetsArray[s] = multibind_params.ubos.offsets[s];
                        sizesArray[s] = multibind_params.ubos.sizes[s];
                        //if it crashes below, it means that there are dynamic Buffer Objects in the DS, but the DS was bound with no (or not enough) dynamic offsets
                        //or for some weird reason (bug) descSets[i].set is nullptr, but descSets[i].dynamicOffsets is not
                        if (multibind_params.ubos.dynOffsetIxs[s] < dynamicOffsetCount)
                            offsetsArray[s] += nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets[multibind_params.ubos.dynOffsetIxs[s]];
                        if (sizesArray[s] == IGPUBufferView::whole_buffer)
                            sizesArray[s] = nextSet->getUBO(s)->getSize() - offsetsArray[s];
                    }
                assert(multibind_params.ubos.buffers);
                gl->extGlBindBuffersRange(GL_UNIFORM_BUFFER, first_count.ubos.first, localUboCount, multibind_params.ubos.buffers, nextSet ? offsetsArray : nullptr, nextSet ? sizesArray : nullptr);
            }
        }

    //unbind previous descriptors if needed (if bindings not replaced by new multibind calls)
    if (prevLayout)//if previous pipeline was nullptr, then no descriptors were bound
    {
        int64_t prevUboCount = 0u, prevSsboCount = 0u, prevTexCount = 0u, prevImgCount = 0u;
        const auto& first_count = prevLayout->getMultibindParamsForDescSet(video::IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u);

        prevUboCount = first_count.ubos.first + first_count.ubos.count;
        prevSsboCount = first_count.ssbos.first + first_count.ssbos.count;
        prevTexCount = first_count.textures.first + first_count.textures.count;
        prevImgCount = first_count.textureImages.first + first_count.textureImages.count;

        int64_t diff = 0LL;
        if ((diff = prevUboCount - newUboCount) > 0LL)
            gl->extGlBindBuffersRange(GL_UNIFORM_BUFFER, newUboCount, diff, nullptr, nullptr, nullptr);
        if ((diff = prevSsboCount - newSsboCount) > 0LL)
            gl->extGlBindBuffersRange(GL_SHADER_STORAGE_BUFFER, newSsboCount, diff, nullptr, nullptr, nullptr);
        if ((diff = prevTexCount - newTexCount) > 0LL) {
            gl->extGlBindTextures(newTexCount, diff, nullptr, nullptr);
            gl->extGlBindSamplers(newTexCount, diff, nullptr);
        }
        if ((diff = prevImgCount - newImgCount) > 0LL)
            gl->extGlBindImageTextures(newImgCount, diff, nullptr, nullptr);
    }

    //update state in state tracker
    effectivelyBoundDescriptors.layout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(_currentLayout);
    for (uint32_t i = 0u; i < video::IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        currentState.descriptorsParams[_pbp].descSets[i] = nextState.descriptorsParams[_pbp].descSets[i];
        effectivelyBoundDescriptors.descSets[i] = nextState.descriptorsParams[_pbp].descSets[i];
    }
}

void SOpenGLContextLocalCache::flushStateGraphics(IOpenGL_FunctionTable* gl, uint32_t stateBits, uint32_t ctxid)
{
    if (stateBits & GSB_PIPELINE)
    {
        if (nextState.pipeline.graphics.pipeline != currentState.pipeline.graphics.pipeline)
        {
            if (nextState.pipeline.graphics.usedShadersHash != currentState.pipeline.graphics.usedShadersHash)
            {
                currentState.pipeline.graphics.usedPipeline = 0u;
#ifndef _NBL_DEBUG
                assert(nextState.pipeline.graphics.usedPipeline == 0u);
#endif

                constexpr SOpenGLState::SGraphicsPipelineHash NULL_HASH = { 0u, 0u, 0u, 0u, 0u };

                const SOpenGLState::SGraphicsPipelineHash lookingFor = nextState.pipeline.graphics.usedShadersHash;
                if (lookingFor != NULL_HASH)
                {
                    auto found = GraphicsPipelineMap.get(lookingFor);
                    if (found)
                    {
                        currentState.pipeline.graphics.usedPipeline = found->GLname;
                    }
                    else
                    {
                        GLuint pipeline = createGraphicsPipeline(gl, lookingFor);
                        SOpenGLState::SGraphicsPipelineHash hash = lookingFor;
                        SPipelineCacheVal val;
                        currentState.pipeline.graphics.usedPipeline = val.GLname = pipeline;
                        GraphicsPipelineMap.insert(std::move(hash), std::move(val));
                    }
                }
                gl->glShader.pglBindProgramPipeline(currentState.pipeline.graphics.usedPipeline);

                currentState.pipeline.graphics.usedShadersHash = nextState.pipeline.graphics.usedShadersHash;
            }

            currentState.pipeline.graphics.pipeline = nextState.pipeline.graphics.pipeline;
        }
    }
    

    // this needs to be here to make sure interleaving the same compute pipeline with the same gfx pipeline works
    if (currentState.pipeline.graphics.usedPipeline && currentState.pipeline.compute.usedShader)
    {
        currentState.pipeline.compute.pipeline = nullptr;
        currentState.pipeline.compute.usedShader = 0u;
        gl->glShader.pglUseProgram(0);
    }

#define STATE_NEQ(member) (nextState.member != currentState.member)
#define UPDATE_STATE(member) (currentState.member = nextState.member)
    if (stateBits & GSB_FRAMEBUFFER)
    {
        if (STATE_NEQ(framebuffer.hash))
        {
            GLuint GLname = 0u;
            if (nextState.framebuffer.hash != SOpenGLState::NULL_FBO_HASH)
            {
                auto* found = FBOCache.get(nextState.framebuffer.hash);
                if (found)
                    GLname = *found;
                else
                {
                    GLname = nextState.framebuffer.fbo->createGLFBO(gl);
                    if (GLname)
                        FBOCache.insert(nextState.framebuffer.hash, GLname);
                }

                assert(GLname != 0u); // TODO uncomment this
            }

            currentState.framebuffer.GLname = GLname;
            UPDATE_STATE(framebuffer.hash);
            UPDATE_STATE(framebuffer.fbo);

            gl->glFramebuffer.pglBindFramebuffer(GL_FRAMEBUFFER, GLname);
        }
    }

    if (stateBits & GSB_RASTER_PARAMETERS)
    {
#define DISABLE_ENABLE(able, what)\
    {\
        if (able)\
            gl->glGeneral.pglEnable(what);\
        else\
            gl->glGeneral.pglDisable(what);\
    }

        // viewports
        {
            GLfloat vpparams[SOpenGLState::MAX_VIEWPORT_COUNT*4];
            for (uint32_t i = 0u; i < SOpenGLState::MAX_VIEWPORT_COUNT; ++i)
            {
                auto& viewport = nextState.rasterParams.viewport[i];
                vpparams[4*i + 0] = viewport.x;
                vpparams[4*i + 1] = viewport.y;
                vpparams[4*i + 2] = viewport.width;
                vpparams[4*i + 3] = viewport.height;
            }

            double vpdparams[SOpenGLState::MAX_VIEWPORT_COUNT*4];
            for (uint32_t i = 0u; i < SOpenGLState::MAX_VIEWPORT_COUNT; ++i)
            {
                auto& vpdc = nextState.rasterParams.viewport_depth[i];
                vpdparams[2*i + 0] = vpdc.minDepth;
                vpdparams[2*i + 1] = vpdc.maxDepth;
            }

            {
                uint32_t first = 0u;
                uint32_t count = 0u;
                for (uint32_t i = 0u; i < SOpenGLState::MAX_VIEWPORT_COUNT; ++i)
                {
                    if (STATE_NEQ(rasterParams.viewport[i]))
                    {
                        if ((count++) == 0u)
                            first = i;
                    }
                    else if (count)
                    {
                        gl->extGlViewportArrayv(first, count, vpparams + (4u * first));
                        count = 0;
                    }
                }
                if (count)
                    gl->extGlViewportArrayv(first, count, vpparams + (4u * first));
            }
            {
                uint32_t first = 0u;
                uint32_t count = 0u;
                for (uint32_t i = 0u; i < SOpenGLState::MAX_VIEWPORT_COUNT; ++i)
                {
                    if (STATE_NEQ(rasterParams.viewport_depth[i]))
                    {
                        if ((count++) == 0u)
                            first = i;
                    }
                    else if (count)
                    {
                        gl->extGlDepthRangeArrayv(first, count, vpdparams + (2u * first));
                        count = 0u;
                    }
                }
                if (count)
                    gl->extGlDepthRangeArrayv(first, count, vpdparams + (2u * first));
            }

            // update all
            for (uint32_t i = 0u; i < SOpenGLState::MAX_VIEWPORT_COUNT; ++i)
            {
                UPDATE_STATE(rasterParams.viewport[i]);
                UPDATE_STATE(rasterParams.viewport_depth[i]);
            }
        }
        // end viewports

        if (STATE_NEQ(rasterParams.polygonMode)) {
            if (!gl->isGLES())
            {
                gl->extGlPolygonMode(GL_FRONT_AND_BACK, nextState.rasterParams.polygonMode);
                UPDATE_STATE(rasterParams.polygonMode);
            }
        }
        if (STATE_NEQ(rasterParams.faceCullingEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.faceCullingEnable, GL_CULL_FACE);
            UPDATE_STATE(rasterParams.faceCullingEnable);
        }
        if (STATE_NEQ(rasterParams.cullFace)) {
            gl->glShader.pglCullFace(nextState.rasterParams.cullFace);
            UPDATE_STATE(rasterParams.cullFace);
        }
        if (STATE_NEQ(rasterParams.stencilTestEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.stencilTestEnable, GL_STENCIL_TEST);
            UPDATE_STATE(rasterParams.stencilTestEnable);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilOp_front)) {
            gl->glFragment.pglStencilOpSeparate(GL_FRONT, nextState.rasterParams.stencilOp_front.sfail, nextState.rasterParams.stencilOp_front.dpfail, nextState.rasterParams.stencilOp_front.dppass);
            UPDATE_STATE(rasterParams.stencilOp_front);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilOp_back)) {
            gl->glFragment.pglStencilOpSeparate(GL_BACK, nextState.rasterParams.stencilOp_back.sfail, nextState.rasterParams.stencilOp_back.dpfail, nextState.rasterParams.stencilOp_back.dppass);
            UPDATE_STATE(rasterParams.stencilOp_back);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilFunc_front)) {
            gl->glFragment.pglStencilFuncSeparate(GL_FRONT, nextState.rasterParams.stencilFunc_front.func, nextState.rasterParams.stencilFunc_front.ref, nextState.rasterParams.stencilFunc_front.mask);
            UPDATE_STATE(rasterParams.stencilFunc_front);
        }
        if (STATE_NEQ(rasterParams.stencilWriteMask_front)) {
            gl->glFragment.pglStencilMaskSeparate(GL_FRONT, nextState.rasterParams.stencilWriteMask_front);
            UPDATE_STATE(rasterParams.stencilWriteMask_front);
        }
        if (STATE_NEQ(rasterParams.stencilWriteMask_back)) {
            gl->glFragment.pglStencilMaskSeparate(GL_BACK, nextState.rasterParams.stencilWriteMask_back);
            UPDATE_STATE(rasterParams.stencilWriteMask_back);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilFunc_back)) {
            gl->glFragment.pglStencilFuncSeparate(GL_FRONT, nextState.rasterParams.stencilFunc_back.func, nextState.rasterParams.stencilFunc_back.ref, nextState.rasterParams.stencilFunc_back.mask);
            UPDATE_STATE(rasterParams.stencilFunc_back);
        }
        if (STATE_NEQ(rasterParams.depthTestEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.depthTestEnable, GL_DEPTH_TEST);
            UPDATE_STATE(rasterParams.depthTestEnable);
        }
        if (nextState.rasterParams.depthTestEnable && STATE_NEQ(rasterParams.depthFunc)) {
            gl->glShader.pglDepthFunc(nextState.rasterParams.depthFunc);
            UPDATE_STATE(rasterParams.depthFunc);
        }
        if (STATE_NEQ(rasterParams.frontFace)) {
            gl->glShader.pglFrontFace(nextState.rasterParams.frontFace);
            UPDATE_STATE(rasterParams.frontFace);
        }
        if (STATE_NEQ(rasterParams.depthClampEnable)) {
            if (gl->isGLES())
            {
                if (gl->getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_depth_clamp))
                {
                    DISABLE_ENABLE(nextState.rasterParams.depthClampEnable, IOpenGL_FunctionTable::DEPTH_CLAMP);
                    UPDATE_STATE(rasterParams.depthClampEnable);
                }
            }
            else
            {
                DISABLE_ENABLE(nextState.rasterParams.depthClampEnable, IOpenGL_FunctionTable::DEPTH_CLAMP);
            }
        }
        if (STATE_NEQ(rasterParams.rasterizerDiscardEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.rasterizerDiscardEnable, GL_RASTERIZER_DISCARD);
            UPDATE_STATE(rasterParams.rasterizerDiscardEnable);
        }
        if (STATE_NEQ(rasterParams.polygonOffsetEnable)) {
            if (!gl->isGLES())
            {
                DISABLE_ENABLE(nextState.rasterParams.polygonOffsetEnable, IOpenGL_FunctionTable::POLYGON_OFFSET_POINT);
                DISABLE_ENABLE(nextState.rasterParams.polygonOffsetEnable, IOpenGL_FunctionTable::POLYGON_OFFSET_LINE);
            }
            DISABLE_ENABLE(nextState.rasterParams.polygonOffsetEnable, GL_POLYGON_OFFSET_FILL);
            UPDATE_STATE(rasterParams.polygonOffsetEnable);
        }
        if (STATE_NEQ(rasterParams.polygonOffset)) {
            gl->glShader.pglPolygonOffset(nextState.rasterParams.polygonOffset.factor, nextState.rasterParams.polygonOffset.units);
            UPDATE_STATE(rasterParams.polygonOffset);
        }
        if (STATE_NEQ(rasterParams.lineWidth)) {
            gl->glShader.pglLineWidth(nextState.rasterParams.lineWidth);
            UPDATE_STATE(rasterParams.lineWidth);
        }
        if (STATE_NEQ(rasterParams.sampleShadingEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.sampleShadingEnable, GL_SAMPLE_SHADING);
            UPDATE_STATE(rasterParams.sampleShadingEnable);
        }
        if (nextState.rasterParams.sampleShadingEnable && STATE_NEQ(rasterParams.minSampleShading)) {
            gl->extGlMinSampleShading(nextState.rasterParams.minSampleShading);
            UPDATE_STATE(rasterParams.minSampleShading);
        }
        if (STATE_NEQ(rasterParams.sampleMaskEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.sampleMaskEnable, GL_SAMPLE_MASK);
            UPDATE_STATE(rasterParams.sampleMaskEnable);
        }
        if (nextState.rasterParams.sampleMaskEnable && STATE_NEQ(rasterParams.sampleMask[0])) {
            gl->glFragment.pglSampleMaski(0u, nextState.rasterParams.sampleMask[0]);
            UPDATE_STATE(rasterParams.sampleMask[0]);
        }
        if (nextState.rasterParams.sampleMaskEnable && STATE_NEQ(rasterParams.sampleMask[1])) {
            gl->glFragment.pglSampleMaski(1u, nextState.rasterParams.sampleMask[1]);
            UPDATE_STATE(rasterParams.sampleMask[1]);
        }
        if (STATE_NEQ(rasterParams.depthWriteEnable)) {
            gl->glShader.pglDepthMask(nextState.rasterParams.depthWriteEnable);
            UPDATE_STATE(rasterParams.depthWriteEnable);
        }
        if (!gl->isGLES() && STATE_NEQ(rasterParams.multisampleEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.multisampleEnable, IOpenGL_FunctionTable::MULTISAMPLE);
            UPDATE_STATE(rasterParams.multisampleEnable);
        }
        if (!gl->isGLES() && STATE_NEQ(rasterParams.primitiveRestartEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.primitiveRestartEnable, GL_PRIMITIVE_RESTART_FIXED_INDEX);
            UPDATE_STATE(rasterParams.primitiveRestartEnable);
        }


        if (!gl->isGLES() && STATE_NEQ(rasterParams.logicOpEnable)) {
            DISABLE_ENABLE(nextState.rasterParams.logicOpEnable, IOpenGL_FunctionTable::COLOR_LOGIC_OP);
            UPDATE_STATE(rasterParams.logicOpEnable);
        }
        if (!gl->isGLES() && STATE_NEQ(rasterParams.logicOp)) {
            gl->extGlLogicOp(nextState.rasterParams.logicOp);
            UPDATE_STATE(rasterParams.logicOp);
        }
        for (GLuint i = 0u; i < asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
        {
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].blendEnable)) {
                bool enable = nextState.rasterParams.drawbufferBlend[i].blendEnable;
                if (enable)
                    gl->extGlEnablei(GL_BLEND, i);
                else
                    gl->extGlDisablei(GL_BLEND, i);
                UPDATE_STATE(rasterParams.drawbufferBlend[i].blendEnable);
            }
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].blendFunc)) {
                gl->extGlBlendFuncSeparatei(i,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.srcRGB,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.dstRGB,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.srcAlpha,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.dstAlpha
                );
                UPDATE_STATE(rasterParams.drawbufferBlend[i].blendFunc);
            }
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].blendEquation)) {
                gl->extGlBlendEquationSeparatei(i,
                    nextState.rasterParams.drawbufferBlend[i].blendEquation.modeRGB,
                    nextState.rasterParams.drawbufferBlend[i].blendEquation.modeAlpha
                );
                UPDATE_STATE(rasterParams.drawbufferBlend[i].blendEquation);
            }
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].colorMask)) {
                gl->extGlColorMaski(i,
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[0],
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[1],
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[2],
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[3]
                );
                UPDATE_STATE(rasterParams.drawbufferBlend[i].colorMask);
            }
        }
#undef DISABLE_ENABLE
    }
    if (stateBits & GSB_DESCRIPTOR_SETS)
    {
        const COpenGLPipelineLayout* currLayout = static_cast<const COpenGLPipelineLayout*>(currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline()->getLayout());
        flushState_descriptors(gl, asset::EPBP_GRAPHICS, currLayout);
    }
    if ((stateBits & GSB_VAO_AND_VERTEX_INPUT) && currentState.pipeline.graphics.pipeline)
    {
        bool brandNewVAO = false;//if VAO is taken from cache we don't have to modify VAO state that is part of hashval (everything except index and vertex buf bindings)
        if (STATE_NEQ(vertexInputParams.vaokey))
        {
            auto hashVal = nextState.vertexInputParams.vaokey;
            auto it = VAOMap.get(hashVal);

            currentState.vertexInputParams.vaokey = hashVal;
            if (it) {
                currentState.vertexInputParams.vaoval.GLname = *it;
            }
            else
            {
                GLuint GLvao;
                gl->extGlCreateVertexArrays(1u, &GLvao);
                SOpenGLState::SVAO val;
                val.GLname = GLvao;
                val.idxType = nextState.vertexInputParams.vaoval.idxType;
                //intentionally leaving val.vtxBindings,idxBinding untouched in currentState so that STATE_NEQ gives true and they get bound
                currentState.vertexInputParams.vaoval = val;
                //bindings in cached object will be updated/filled later
                VAOMap.insert(hashVal, GLvao);
                brandNewVAO = true;
            }

            GLuint vao = currentState.vertexInputParams.vaoval.GLname;

            gl->glVertex.pglBindVertexArray(vao);

            bool updatedBindings[asset::SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT]{};
            for (uint32_t attr = 0u; attr < asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++attr)
            {
                if (hashVal.attribFormatAndComponentCount[attr] != asset::EF_UNKNOWN) {
                    if (brandNewVAO)
                    {
                        gl->extGlEnableVertexArrayAttrib(vao, attr);
                    }
                }
                else
                    continue;

                const uint32_t bnd = hashVal.getBindingForAttrib(attr);

                if (brandNewVAO)
                {
                    gl->extGlVertexArrayAttribBinding(vao, attr, bnd);

                    const asset::E_FORMAT format = static_cast<asset::E_FORMAT>(hashVal.attribFormatAndComponentCount[attr]);

                    if (!gl->isGLES() && isFloatingPointFormat(format) && getTexelOrBlockBytesize(format) == getFormatChannelCount(format) * sizeof(double))//DOUBLE
                        gl->extGlVertexArrayAttribLFormat(vao, attr, getFormatChannelCount(format), GL_DOUBLE, hashVal.getRelativeOffsetForAttrib(attr));
                    else if (isFloatingPointFormat(format) || isScaledFormat(format) || isNormalizedFormat(format))//FLOATING-POINT, SCALED ("weak integer"), NORMALIZED
                    {
                        gl->extGlVertexArrayAttribFormat(vao, attr, isBGRALayoutFormat(format) ? GL_BGRA : getFormatChannelCount(format), formatEnumToGLenum(gl, format), isNormalizedFormat(format) ? GL_TRUE : GL_FALSE, hashVal.getRelativeOffsetForAttrib(attr));
                    }
                    else if (isIntegerFormat(format))//INTEGERS
                        gl->extGlVertexArrayAttribIFormat(vao, attr, getFormatChannelCount(format), formatEnumToGLenum(gl, format), hashVal.getRelativeOffsetForAttrib(attr));

                    if (!updatedBindings[bnd]) {
                        gl->extGlVertexArrayBindingDivisor(vao, bnd, hashVal.getDivisorForBinding(bnd));
                        updatedBindings[bnd] = true;
                    }
                }
            }
            //vertex and index buffer bindings are done outside this if-statement because no change in hash doesn't imply no change in those bindings
        }

        UPDATE_STATE(vertexInputParams.vaoval.idxType);

        GLuint GLvao = currentState.vertexInputParams.vaoval.GLname;
        assert(GLvao);
        {
            // changing buffer binding in vao is practically free, so we're always doing that and not caching this part of vao state
            for (uint32_t i = 0u; i < asset::SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++i)
            {
                const auto& hash = currentState.vertexInputParams.vaokey;
                if (hash.attribFormatAndComponentCount[i] == asset::EF_UNKNOWN)
                    continue;

                const uint32_t bnd = hash.getBindingForAttrib(i);

                assert(nextState.vertexInputParams.vaoval.vtxBindings[bnd].buffer);//something went wrong
                uint32_t stride = hash.getStrideForBinding(bnd);
                assert(stride != 0u);
                gl->extGlVertexArrayVertexBuffer(GLvao, bnd, nextState.vertexInputParams.vaoval.vtxBindings[bnd].buffer->getOpenGLName(), nextState.vertexInputParams.vaoval.vtxBindings[bnd].offset, stride);
                UPDATE_STATE(vertexInputParams.vaoval.vtxBindings[bnd]);
            }
            GLuint GLidxbuf = nextState.vertexInputParams.vaoval.idxBinding.buffer ? nextState.vertexInputParams.vaoval.idxBinding.buffer->getOpenGLName() : 0u;
            gl->extGlVertexArrayElementBuffer(GLvao, GLidxbuf);
            UPDATE_STATE(vertexInputParams.vaoval.idxBinding);
        }
        if (STATE_NEQ(vertexInputParams.indirectDrawBuf))
        {
            gl->glBuffer.pglBindBuffer(GL_DRAW_INDIRECT_BUFFER, nextState.vertexInputParams.indirectDrawBuf ? nextState.vertexInputParams.indirectDrawBuf->getOpenGLName() : 0u);
            UPDATE_STATE(vertexInputParams.indirectDrawBuf);
        }
        if (STATE_NEQ(vertexInputParams.parameterBuf))
        {
            gl->glBuffer.pglBindBuffer(GL_PARAMETER_BUFFER, nextState.vertexInputParams.parameterBuf ? nextState.vertexInputParams.parameterBuf->getOpenGLName() : 0u);
            UPDATE_STATE(vertexInputParams.parameterBuf);
        }
    }
    if ((stateBits & GSB_PUSH_CONSTANTS) && currentState.pipeline.graphics.pipeline)
    {
        const auto* glppln = static_cast<const COpenGLRenderpassIndependentPipeline*>(currentState.pipeline.graphics.pipeline->getRenderpassIndependentPipeline());
        //pipeline must be flushed before push constants so taking pipeline from currentState
        glppln->setUniformsImitatingPushConstants(gl, ctxid, pushConstantsStateGraphics);
    }
    if (stateBits & GSB_PIXEL_PACK_UNPACK)
    {
        //PACK
        if (STATE_NEQ(pixelPack.buffer))
        {
            gl->glBuffer.pglBindBuffer(GL_PIXEL_PACK_BUFFER, nextState.pixelPack.buffer ? nextState.pixelPack.buffer->getOpenGLName() : 0u);
            UPDATE_STATE(pixelPack.buffer);
        }
        if (STATE_NEQ(pixelPack.alignment))
        {
            gl->glShader.pglPixelStorei(GL_PACK_ALIGNMENT, nextState.pixelPack.alignment);
            UPDATE_STATE(pixelPack.alignment);
        }
        if (STATE_NEQ(pixelPack.rowLength))
        {
            gl->glShader.pglPixelStorei(GL_PACK_ROW_LENGTH, nextState.pixelPack.rowLength);
            UPDATE_STATE(pixelPack.rowLength);
        }
        if (STATE_NEQ(pixelPack.imgHeight))
        {
            gl->glShader.pglPixelStorei(GL_PACK_IMAGE_HEIGHT, nextState.pixelPack.imgHeight);
            UPDATE_STATE(pixelPack.imgHeight);
        }
        if (STATE_NEQ(pixelPack.BCwidth))
        {
            gl->glShader.pglPixelStorei(GL_PACK_COMPRESSED_BLOCK_WIDTH, nextState.pixelPack.BCwidth);
            UPDATE_STATE(pixelPack.BCwidth);
        }
        if (STATE_NEQ(pixelPack.BCheight))
        {
            gl->glShader.pglPixelStorei(GL_PACK_COMPRESSED_BLOCK_HEIGHT, nextState.pixelPack.BCheight);
            UPDATE_STATE(pixelPack.BCheight);
        }
        if (STATE_NEQ(pixelPack.BCdepth))
        {
            gl->glShader.pglPixelStorei(GL_PACK_COMPRESSED_BLOCK_DEPTH, nextState.pixelPack.BCdepth);
            UPDATE_STATE(pixelPack.BCdepth);
        }

        //UNPACK
        if (STATE_NEQ(pixelUnpack.buffer))
        {
            gl->glBuffer.pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, nextState.pixelUnpack.buffer ? nextState.pixelUnpack.buffer->getOpenGLName() : 0u);
            UPDATE_STATE(pixelUnpack.buffer);
        }
        if (STATE_NEQ(pixelUnpack.alignment))
        {
            gl->glShader.pglPixelStorei(GL_UNPACK_ALIGNMENT, nextState.pixelUnpack.alignment);
            UPDATE_STATE(pixelUnpack.alignment);
        }
        if (STATE_NEQ(pixelUnpack.rowLength))
        {
            gl->glShader.pglPixelStorei(GL_UNPACK_ROW_LENGTH, nextState.pixelUnpack.rowLength);
            UPDATE_STATE(pixelUnpack.rowLength);
        }
        if (STATE_NEQ(pixelUnpack.imgHeight))
        {
            gl->glShader.pglPixelStorei(GL_UNPACK_IMAGE_HEIGHT, nextState.pixelUnpack.imgHeight);
            UPDATE_STATE(pixelUnpack.imgHeight);
        }
        if (STATE_NEQ(pixelUnpack.BCwidth))
        {
            gl->glShader.pglPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_WIDTH, nextState.pixelUnpack.BCwidth);
            UPDATE_STATE(pixelUnpack.BCwidth);
        }
        if (STATE_NEQ(pixelUnpack.BCheight))
        {
            gl->glShader.pglPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_HEIGHT, nextState.pixelUnpack.BCheight);
            UPDATE_STATE(pixelUnpack.BCheight);
        }
        if (STATE_NEQ(pixelUnpack.BCdepth))
        {
            gl->glShader.pglPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_DEPTH, nextState.pixelUnpack.BCdepth);
            UPDATE_STATE(pixelUnpack.BCdepth);
        }
    }
#undef STATE_NEQ
#undef UPDATE_STATE
}

void SOpenGLContextLocalCache::flushStateCompute(IOpenGL_FunctionTable* gl, uint32_t stateBits, uint32_t ctxid)
{
    if (stateBits & GSB_PIPELINE)
    {
        if (nextState.pipeline.compute.usedShader != currentState.pipeline.compute.usedShader)
        {
            const GLuint GLname = nextState.pipeline.compute.usedShader;
            gl->glShader.pglUseProgram(GLname);
            currentState.pipeline.compute.usedShader = GLname;
        }
        if (nextState.pipeline.compute.pipeline != currentState.pipeline.compute.pipeline)
        {
            currentState.pipeline.compute.pipeline = nextState.pipeline.compute.pipeline;
        }
    }
    if ((stateBits & GSB_PUSH_CONSTANTS) && currentState.pipeline.compute.pipeline)
    {
        assert(currentState.pipeline.compute.pipeline->containsShader());
        currentState.pipeline.compute.pipeline->setUniformsImitatingPushConstants(gl, ctxid, pushConstantsStateCompute);
    }
    if (stateBits & GSB_DISPATCH_INDIRECT)
    {
        if (currentState.dispatchIndirect.buffer != nextState.dispatchIndirect.buffer)
        {
            const GLuint GLname = nextState.dispatchIndirect.buffer ? nextState.dispatchIndirect.buffer->getOpenGLName() : 0u;
            gl->glBuffer.pglBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, GLname);
            currentState.dispatchIndirect.buffer = nextState.dispatchIndirect.buffer;
        }
    }
    if (stateBits & GSB_DESCRIPTOR_SETS)
    {
        const COpenGLPipelineLayout* currLayout = static_cast<const COpenGLPipelineLayout*>(currentState.pipeline.compute.pipeline->getLayout());
        flushState_descriptors(gl, asset::EPBP_COMPUTE, currLayout);
    }
}

GLuint SOpenGLContextLocalCache::createGraphicsPipeline(IOpenGL_FunctionTable* gl, const SOpenGLState::SGraphicsPipelineHash& _hash)
{
    constexpr size_t STAGE_CNT = COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT;
    static_assert(STAGE_CNT == 5u, "SHADER_STAGE_COUNT is expected to be 5");
    const GLenum stages[5]{ GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const GLenum stageFlags[5]{ GL_VERTEX_SHADER_BIT, GL_TESS_CONTROL_SHADER_BIT, GL_TESS_EVALUATION_SHADER_BIT, GL_GEOMETRY_SHADER_BIT, GL_FRAGMENT_SHADER_BIT };

    GLuint GLpipeline = 0u;
    gl->glShader.pglGenProgramPipelines(1u, &GLpipeline);

    for (uint32_t ix = 0u; ix < STAGE_CNT; ++ix) {
        GLuint progName = _hash[ix];

        if (progName)
            gl->glShader.pglUseProgramStages(GLpipeline, stageFlags[ix], progName);
    }

    return GLpipeline;
}

}
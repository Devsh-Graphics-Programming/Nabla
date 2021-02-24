#include "nbl/video/SOpenGLContextLocalCache.h"

namespace nbl {
namespace video
{

void SOpenGLContextLocalCache::updateNextState_pipelineAndRaster(const IGPURenderpassIndependentPipeline* _pipeline, uint32_t ctxid)
{
    nextState.pipeline.graphics.pipeline = core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline>(
        static_cast<const COpenGLRenderpassIndependentPipeline*>(_pipeline)
        );
    if (!_pipeline)
    {
        SOpenGLState::SGraphicsPipelineHash hash;
        std::fill(hash.begin(), hash.end(), 0u);
        nextState.pipeline.graphics.usedShadersHash = hash;
        return;
    }
    SOpenGLState::SGraphicsPipelineHash hash;
    for (uint32_t i = 0u; i < COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++i)
    {
        hash[i] = nextState.pipeline.graphics.pipeline->getShaderAtIndex(i) ?
            nextState.pipeline.graphics.pipeline->getShaderGLnameForCtx(i, ctxid) :
            0u;
    }
    nextState.pipeline.graphics.usedShadersHash = hash;

    const auto& ppln = nextState.pipeline.graphics.pipeline;

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
    raster_dst.polygonOffset.units = raster_src.depthBiasSlopeFactor;

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

}
}
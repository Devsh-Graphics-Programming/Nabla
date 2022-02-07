// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

///#include "COpenGLStateManager.h"

#ifndef __NBL_C_OPENGL_STATE_MANAGER_IMPLEMENTATION_H_INCLUDED__
#define __NBL_C_OPENGL_STATE_MANAGER_IMPLEMENTATION_H_INCLUDED__

#ifndef glIsEnabledi_MACRO
#define glIsEnabledi_MACRO glIsEnabledi
#endif  // glIsEnabledi_MACRO
#ifndef glEnablei_MACRO
#define glEnablei_MACRO glEnablei
#endif  // glEnablei_MACRO
#ifndef glDisablei_MACRO
#define glDisablei_MACRO glDisablei
#endif  // glDisablei_MACRO

#ifndef glProvokingVertex_MACRO
#define glProvokingVertex_MACRO glProvokingVertex
#endif  // glProvokingVertex_MACRO

#ifndef glClampColor_MACRO
#define glClampColor_MACRO glClampColor
#endif  // glClampColor_MACRO

#ifndef glPrimitiveRestartIndex_MACRO
#define glPrimitiveRestartIndex_MACRO glPrimitiveRestartIndex
#endif  // glPrimitiveRestartIndex_MACRO

#ifndef glBindTransformFeedback_MACRO
#define glBindTransformFeedback_MACRO glBindTransformFeedback
#endif  // glBindTransformFeedback_MACRO

#ifndef glUseProgram_MACRO
#define glUseProgram_MACRO glUseProgram
#endif  // glUseProgram_MACRO
#ifndef glBindProgramPipeline_MACRO
#define glBindProgramPipeline_MACRO glBindProgramPipeline
#endif  // glBindProgramPipeline_MACRO

#ifndef glPatchParameteri_MACRO
#define glPatchParameteri_MACRO glPatchParameteri
#endif  // glPatchParameteri_MACRO
#ifndef glPatchParameterfv_MACRO
#define glPatchParameterfv_MACRO glPatchParameterfv
#endif  // glPatchParameterfv_MACRO

#ifndef glBindBuffer_MACRO
#define glBindBuffer_MACRO glBindBuffer
#endif  // glBindBuffer_MACRO
//! NEEDS AN URGENT CHANGE
#ifndef glBindBufferRange_MACRO
#define glBindBufferRange_MACRO glBindBufferRange
#endif  // glBindBufferRange_MACRO

#ifndef glGetNamedBufferParameteri64v_MACRO
#define glGetNamedBufferParameteri64v_MACRO glGetNamedBufferParameteri64v
#endif

#ifndef glDepthRangeIndexed_MACRO
#define glDepthRangeIndexed_MACRO glDepthRangeIndexed
#endif  // glDepthRangeIndexed_MACRO

#ifndef glViewportIndexedfv_MACRO
#define glViewportIndexedfv_MACRO glViewportIndexedfv
#endif  // glViewportIndexedfv_MACRO

#ifndef glScissorIndexedv_MACRO
#define glScissorIndexedv_MACRO glScissorIndexedv
#endif  // glScissorIndexedv_MACRO

#ifndef glSampleCoverage_MACRO
#define glSampleCoverage_MACRO glSampleCoverage
#endif  // glSampleCoverage_MACRO

#ifndef glSampleMaski_MACRO
#define glSampleMaski_MACRO glSampleMaski
#endif  // glSampleMaski_MACRO

#ifndef glMinSampleShading_MACRO
#define glMinSampleShading_MACRO glMinSampleShading
#endif  // glMinSampleShading_MACRO

#ifndef glBlendColor_MACRO
#define glBlendColor_MACRO glBlendColor
#endif  // glBlendColor_MACRO

#ifndef glBlendEquationSeparatei_MACRO
#define glBlendEquationSeparatei_MACRO glBlendEquationSeparatei
#endif  // glBlendEquationSeparatei_MACRO

#ifndef glBlendFuncSeparatei_MACRO
#define glBlendFuncSeparatei_MACRO glBlendFuncSeparatei
#endif  // glBlendFuncSeparatei_MACRO

//! NEEDS AN URGENT CHANGE
#ifndef glBindImageTexture_MACRO
#define glBindImageTexture_MACRO glBindImageTexture
#endif  // glBindImageTexture_MACRO

#ifndef glActiveTexture_MACRO
#define glActiveTexture_MACRO glActiveTexture
#endif  // glActiveTexture_MACRO

//! NEEDS AN URGENT CHANGE
#ifndef SPECIAL_glBindTextureUnit_MACRO
#error "No Override for Active Texture Setting"
#endif  // SPECIAL_glBindTextureUnit_MACRO

//! NEEDS AN URGENT CHANGE
#ifndef glBindSampler_MACRO
#define glBindSampler_MACRO glBindSampler
#endif  // glBindSampler_MACRO

#ifndef glBindVertexArray_MACRO
#define glBindVertexArray_MACRO glBindVertexArray
#endif  // glBindVertexArray_MACRO

COpenGLState COpenGLState::collectGLState(bool careAboutHints,  //should be default false
    bool careAboutFBOs,
    bool careAboutPolygonOffset,  //should be default false
    bool careAboutPixelXferOps,
    bool careAboutSSBOAndAtomicCounters,
    bool careAboutXFormFeedback,
    bool careAboutProgram,
    bool careAboutPipeline,
    bool careAboutTesellationParams,
    bool careAboutViewports,
    bool careAboutDrawIndirectBuffers,
    bool careAboutPointSize,
    bool careAboutLineWidth,
    bool careAboutLogicOp,
    bool careAboutMultisampling,
    bool careAboutBlending,
    bool careAboutColorWriteMasks,
    bool careAboutStencilFunc,
    bool careAboutStencilOp,
    bool careAboutStencilMask,
    bool careAboutDepthFunc,
    bool careAboutDepthMask,
    bool careAboutImages,
    bool careAboutTextures,
    bool careAboutFaceOrientOrCull,
    bool careAboutVAO,
    bool careAboutUBO)
{
    COpenGLState retval;

    if(careAboutHints)
    {
        glGetIntegerv(GL_FRAGMENT_SHADER_DERIVATIVE_HINT, reinterpret_cast<GLint*>(retval.glHint_vals + EGHB_FRAGMENT_SHADER_DERIVATIVE_HINT));
        glGetIntegerv(GL_LINE_SMOOTH_HINT, reinterpret_cast<GLint*>(retval.glHint_vals + EGHB_LINE_SMOOTH_HINT));
        glGetIntegerv(GL_POLYGON_SMOOTH_HINT, reinterpret_cast<GLint*>(retval.glHint_vals + EGHB_POLYGON_SMOOTH_HINT));
        glGetIntegerv(GL_TEXTURE_COMPRESSION_HINT, reinterpret_cast<GLint*>(retval.glHint_vals + EGHB_TEXTURE_COMPRESSION_HINT));
    }

    glGetIntegerv(GL_PROVOKING_VERTEX, reinterpret_cast<GLint*>(&retval.glProvokingVertex_val));

    for(size_t i = 0; i < EGEB_COUNT; i++)
        retval.setGlEnableBit((E_GL_ENABLE_BIT)i, glIsEnabled(glEnableBitToGLenum(i)));

    int32_t maxDrawBuffers, maxViewports;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxDrawBuffers);
    glGetIntegerv(GL_MAX_VIEWPORTS, &maxViewports);
    if(maxDrawBuffers > OGL_STATE_MAX_DRAW_BUFFERS)
        maxDrawBuffers = OGL_STATE_MAX_DRAW_BUFFERS;
    if(maxViewports > OGL_STATE_MAX_VIEWPORTS)
        maxViewports = OGL_STATE_MAX_VIEWPORTS;

    for(size_t i = 0; i < EGEIB_COUNT; i++)
    {
        int32_t maxIx = 0;
        GLenum flag = GL_INVALID_ENUM;
        switch(i)
        {
            case EGEIB_BLEND:
                maxIx = maxDrawBuffers;
                flag = GL_BLEND;
                break;
            case EGEIB_SCISSOR_TEST:
                maxIx = maxViewports;
                flag = GL_SCISSOR_TEST;
                break;
        }
        for(int32_t index = 0; index < maxIx; index++)
            retval.setGlEnableiBit((E_GL_ENABLE_INDEX_BIT)i, index, glIsEnabledi_MACRO(flag, index));
    }

    if(careAboutFBOs)
    {
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(retval.glBindFramebuffer_vals + 0));
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, reinterpret_cast<GLint*>(retval.glBindFramebuffer_vals + 1));
    }

    if(careAboutPolygonOffset)
    {
        glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &retval.glPolygonOffset_factor);
        glGetFloatv(GL_POLYGON_OFFSET_UNITS, &retval.glPolygonOffset_units);
    }

    //
    if(careAboutPixelXferOps)
    {
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, reinterpret_cast<GLint*>(retval.boundBuffers + EGBT_PIXEL_PACK_BUFFER));
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, reinterpret_cast<GLint*>(retval.boundBuffers + EGBT_PIXEL_UNPACK_BUFFER));
    }

    if(careAboutPointSize)
        glGetFloatv(GL_POINT_SIZE, retval.glPrimitiveSize + 0);
    if(careAboutLineWidth)
        glGetFloatv(GL_LINE_WIDTH, retval.glPrimitiveSize + 1);

    if(careAboutPixelXferOps)
    {
        glGetIntegerv(GL_CLAMP_READ_COLOR, reinterpret_cast<GLint*>(&retval.glClampColor_val));
        for(size_t i = 0; i < EGPP_COUNT; i++)
            glGetIntegerv(COpenGLState::glPackParamToGLenum(i), reinterpret_cast<GLint*>(retval.glPixelStorei_vals[0] + i));
        for(size_t i = EGPP_COUNT; i < 2 * EGPP_COUNT; i++)
            glGetIntegerv(COpenGLState::glPackParamToGLenum(i), reinterpret_cast<GLint*>(retval.glPixelStorei_vals[1] + i - EGPP_COUNT));
    }

    glGetIntegerv(GL_PRIMITIVE_RESTART_INDEX, reinterpret_cast<GLint*>(&retval.glPrimitiveRestartIndex_val));

    if(careAboutXFormFeedback)
        glGetIntegerv(GL_TRANSFORM_FEEDBACK_BINDING, reinterpret_cast<GLint*>(&retval.glBindTransformFeedback_val));

    if(careAboutProgram)
        glGetIntegerv(GL_CURRENT_PROGRAM, reinterpret_cast<GLint*>(&retval.glUseProgram_val));
    if(careAboutPipeline)
        glGetIntegerv(GL_PROGRAM_PIPELINE_BINDING, reinterpret_cast<GLint*>(&retval.glBindProgramPipeline_val));

    if(careAboutTesellationParams)
    {
        glGetIntegerv(GL_PATCH_VERTICES, &retval.glPatchParameteri_val);
        glGetFloatv(GL_PATCH_DEFAULT_INNER_LEVEL, retval.glPatchParameterfv_inner);
        glGetFloatv(GL_PATCH_DEFAULT_OUTER_LEVEL, retval.glPatchParameterfv_outer);
    }

    if(careAboutViewports)
    {
        for(uint32_t i = 0; i < maxViewports; i++)
        {
            glGetFloati_v_MACRO(GL_DEPTH_RANGE, i, retval.glDepthRangeArray_vals[i]);
            glGetFloati_v_MACRO(GL_VIEWPORT, i, retval.glViewportArray_vals[i]);
            glGetIntegeri_v_MACRO(GL_SCISSOR_BOX, i, retval.glScissorArray_vals[i]);
        }
    }

    if(careAboutLogicOp)
        glGetIntegerv(GL_LOGIC_OP_MODE, reinterpret_cast<GLint*>(&retval.glLogicOp_val));

    if(careAboutMultisampling)
    {
        glGetFloatv(GL_SAMPLE_COVERAGE_VALUE, &retval.glSampleCoverage_val);
        GLboolean tmpBool;
        glGetBooleanv(GL_SAMPLE_COVERAGE_INVERT, &tmpBool);
        retval.glSampleCoverage_invert = tmpBool != GL_FALSE;

        int32_t maxSampleMasks;
        glGetIntegerv(GL_MAX_SAMPLE_MASK_WORDS, &maxSampleMasks);
        if(maxSampleMasks > OGL_STATE_MAX_SAMPLE_MASK_WORDS)
            maxSampleMasks = OGL_STATE_MAX_SAMPLE_MASK_WORDS;

        for(uint32_t i = 0; i < maxSampleMasks; i++)
            glGetIntegeri_v_MACRO(GL_SAMPLE_MASK_VALUE, i, reinterpret_cast<GLint*>(retval.glSampleMaski_vals + i));

        glGetFloatv(GL_MIN_SAMPLE_SHADING_VALUE, &retval.glMinSampleShading_val);
    }

    if(careAboutBlending)
    {
        glGetFloatv(GL_BLEND_COLOR, retval.glBlendColor_vals);
        //assert(retval.glColorMaski_vals[0]==(~uint64_t(0)));
        for(int32_t i = 0; i < maxDrawBuffers; i++)
        {
            glGetIntegeri_v_MACRO(GL_BLEND_EQUATION_RGB, i, reinterpret_cast<GLint*>(retval.glBlendEquationSeparatei_vals[i] + 0));
            glGetIntegeri_v_MACRO(GL_BLEND_EQUATION_ALPHA, i, reinterpret_cast<GLint*>(retval.glBlendEquationSeparatei_vals[i] + 1));

            glGetIntegeri_v_MACRO(GL_BLEND_SRC_RGB, i, reinterpret_cast<GLint*>(retval.glBlendFuncSeparatei_vals[i] + 0));
            glGetIntegeri_v_MACRO(GL_BLEND_DST_RGB, i, reinterpret_cast<GLint*>(retval.glBlendFuncSeparatei_vals[i] + 1));
            glGetIntegeri_v_MACRO(GL_BLEND_SRC_ALPHA, i, reinterpret_cast<GLint*>(retval.glBlendFuncSeparatei_vals[i] + 2));
            glGetIntegeri_v_MACRO(GL_BLEND_DST_ALPHA, i, reinterpret_cast<GLint*>(retval.glBlendFuncSeparatei_vals[i] + 3));

            GLboolean bools[4];
            glGetBooleani_v_MACRO(GL_COLOR_WRITEMASK, i, bools);
            for(size_t j = 0; j < 4; j++)
                retval.glColorMaski_vals[i / 16] ^= bools[j] ? 0x0u : (0x1ull << (j + 4 * i));
        }
    }

    if(careAboutStencilFunc)
    {
        glGetIntegerv(GL_STENCIL_FUNC, reinterpret_cast<GLint*>(retval.glStencilFuncSeparate_func + 0));
        glGetIntegerv(GL_STENCIL_REF, reinterpret_cast<GLint*>(retval.glStencilFuncSeparate_ref + 0));
        glGetIntegerv(GL_STENCIL_VALUE_MASK, reinterpret_cast<GLint*>(retval.glStencilFuncSeparate_mask + 0));
        glGetIntegerv(GL_STENCIL_BACK_FUNC, reinterpret_cast<GLint*>(retval.glStencilFuncSeparate_func + 1));
        glGetIntegerv(GL_STENCIL_BACK_REF, reinterpret_cast<GLint*>(retval.glStencilFuncSeparate_ref + 1));
        glGetIntegerv(GL_STENCIL_BACK_VALUE_MASK, reinterpret_cast<GLint*>(retval.glStencilFuncSeparate_mask + 1));
    }

    if(careAboutStencilOp)
    {
        glGetIntegerv(GL_STENCIL_FAIL, reinterpret_cast<GLint*>(retval.glStencilOpSeparate_sfail + 0));
        glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, reinterpret_cast<GLint*>(retval.glStencilOpSeparate_dpfail + 0));
        glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, reinterpret_cast<GLint*>(retval.glStencilOpSeparate_dppass + 0));
        glGetIntegerv(GL_STENCIL_BACK_FAIL, reinterpret_cast<GLint*>(retval.glStencilOpSeparate_sfail + 1));
        glGetIntegerv(GL_STENCIL_BACK_PASS_DEPTH_FAIL, reinterpret_cast<GLint*>(retval.glStencilOpSeparate_dpfail + 1));
        glGetIntegerv(GL_STENCIL_BACK_PASS_DEPTH_PASS, reinterpret_cast<GLint*>(retval.glStencilOpSeparate_dppass + 1));
    }

    if(careAboutStencilMask)
    {
        glGetIntegerv(GL_STENCIL_WRITEMASK, reinterpret_cast<GLint*>(retval.glStencilMaskSeparate_mask + 0));
        glGetIntegerv(GL_STENCIL_BACK_WRITEMASK, reinterpret_cast<GLint*>(retval.glStencilMaskSeparate_mask + 1));
    }

    if(careAboutDepthFunc)
        glGetIntegerv(GL_DEPTH_FUNC, reinterpret_cast<GLint*>(&retval.glDepthFunc_val));

    if(careAboutDepthMask)
    {
        GLboolean boolval;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &boolval);
        retval.glDepthMask_val = boolval;
    }

    if(careAboutDrawIndirectBuffers)
    {
        glGetIntegerv(GL_DRAW_INDIRECT_BUFFER_BINDING, reinterpret_cast<GLint*>(retval.boundBuffers + EGBT_DRAW_INDIRECT_BUFFER));
        glGetIntegerv(GL_DISPATCH_INDIRECT_BUFFER_BINDING, reinterpret_cast<GLint*>(retval.boundBuffers + EGBT_DISPATCH_INDIRECT_BUFFER));
    }

    if(careAboutSSBOAndAtomicCounters)
    {
        int32_t maxSSBOs = 0;
        glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxSSBOs);
        if(maxSSBOs > OGL_MAX_BUFFER_BINDINGS)
            maxSSBOs = OGL_MAX_BUFFER_BINDINGS;

        for(int32_t i = 0; i < maxSSBOs; i++)
        {
            glGetIntegeri_v_MACRO(GL_SHADER_STORAGE_BUFFER_BINDING, i, reinterpret_cast<GLint*>(&retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].object));
            if(retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].object)
            {
                glGetIntegeri_v_MACRO(GL_SHADER_STORAGE_BUFFER_SIZE, i, reinterpret_cast<GLint*>(&retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].size));
                if(retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].size)
                    glGetIntegeri_v_MACRO(GL_SHADER_STORAGE_BUFFER_START, i, reinterpret_cast<GLint*>(&retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].offset));
                else
                {
                    glGetNamedBufferParameteri64v_MACRO(retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].object, GL_BUFFER_SIZE,
                        reinterpret_cast<GLint64*>(&retval.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][i].size));
                }
            }
        }
    }

    if(careAboutImages)
    {
        int32_t maxImageUnits;
        glGetIntegerv(GL_MAX_IMAGE_UNITS, &maxImageUnits);
        if(maxImageUnits > OGL_STATE_MAX_IMAGES)
            maxImageUnits = OGL_STATE_MAX_IMAGES;

        for(int32_t i = 0; i < maxImageUnits; i++)
        {
            glGetIntegeri_v_MACRO(GL_IMAGE_BINDING_NAME, i, reinterpret_cast<GLint*>(&retval.glBindImageTexture_texture[i]));
            glGetIntegeri_v_MACRO(GL_IMAGE_BINDING_LEVEL, i, reinterpret_cast<GLint*>(&retval.glBindImageTexture_level[i]));
            glGetIntegeri_v_MACRO(GL_IMAGE_BINDING_LAYERED, i, reinterpret_cast<GLint*>(&retval.glBindImageTexture_layered[i]));
            glGetIntegeri_v_MACRO(GL_IMAGE_BINDING_LAYER, i, reinterpret_cast<GLint*>(&retval.glBindImageTexture_layer[i]));
            glGetIntegeri_v_MACRO(GL_IMAGE_BINDING_ACCESS, i, reinterpret_cast<GLint*>(&retval.glBindImageTexture_access[i]));
            glGetIntegeri_v_MACRO(GL_IMAGE_BINDING_FORMAT, i, reinterpret_cast<GLint*>(&retval.glBindImageTexture_format[i]));
        }
    }

    if(careAboutTextures)
    {
        int32_t maxTextureUnits;
        glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &maxTextureUnits);
        if(maxTextureUnits > OGL_STATE_MAX_TEXTURES)
            maxTextureUnits = OGL_STATE_MAX_TEXTURES;

        int32_t activeTex = 0;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
        for(int32_t i = 0; i < maxTextureUnits; i++)
        {
            glActiveTexture_MACRO(GL_TEXTURE0 + i);

            for(uint32_t j = 0; j < EGTT_COUNT; j++)
                glGetIntegerv(glTextureTypeToBindingGLenum(j), reinterpret_cast<GLint*>(retval.boundTextures[i] + j));

            glGetIntegerv(GL_SAMPLER_BINDING, reinterpret_cast<GLint*>(retval.boundSamplers + i));
        }
        glActiveTexture_MACRO(activeTex);
    }

    if(careAboutVAO)
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, reinterpret_cast<GLint*>(&retval.boundVAO));

    if(careAboutFaceOrientOrCull)
    {
        glGetIntegerv(GL_POLYGON_MODE, reinterpret_cast<GLint*>(&retval.glPolygonMode_mode));
        glGetIntegerv(GL_FRONT_FACE, reinterpret_cast<GLint*>(&retval.glFrontFace_val));
        glGetIntegerv(GL_CULL_FACE_MODE, reinterpret_cast<GLint*>(&retval.glCullFace_val));
    }

    if(careAboutUBO)
    {
        uint32_t maxUBOs = 0;
        glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, reinterpret_cast<int32_t*>(&maxUBOs));
        if(maxUBOs > OGL_MAX_BUFFER_BINDINGS)
            maxUBOs = OGL_MAX_BUFFER_BINDINGS;

        for(int32_t i = 0; i < maxUBOs; i++)
        {
            glGetIntegeri_v_MACRO(GL_UNIFORM_BUFFER_BINDING, i, reinterpret_cast<GLint*>(&retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].object));
            if(retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].object)
            {
                glGetIntegeri_v_MACRO(GL_UNIFORM_BUFFER_SIZE, i, reinterpret_cast<GLint*>(&retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].size));
                if(retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].size)
                    glGetIntegeri_v_MACRO(GL_UNIFORM_BUFFER_START, i, reinterpret_cast<GLint*>(&retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].offset));
                else
                {
                    glGetNamedBufferParameteri64v_MACRO(retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].object, GL_BUFFER_SIZE,
                        reinterpret_cast<GLint64*>(&retval.boundBufferRanges[EGRBT_UNIFORM_BUFFER][i].size));
                }
            }
        }
    }

    return retval;
}

void executeGLDiff(const COpenGLStateDiff& diff)
{
    //set hints
    for(uint8_t i = 0; i < diff.hintsToSet; i++)
        glHint(diff.glHint_pair[i][0], diff.glHint_pair[i][1]);

    if(diff.glProvokingVertex_val != GL_INVALID_ENUM)
        glProvokingVertex_MACRO(diff.glProvokingVertex_val);

    //enable/disable
    for(uint32_t i = 0; i < diff.glDisableCount; i++)
        glDisable(diff.glDisables[i]);
    for(uint32_t i = 0; i < diff.glEnableCount; i++)
        glEnable(diff.glEnables[i]);

    for(uint32_t i = 0; i < diff.glDisableiCount; i++)
        for(uint32_t j = 0; j < OGL_MAX_ENDISABLEI_INDICES; j++)
        {
            if(diff.glDisableis[i].indices & (0x1u << j))
                glDisablei_MACRO(diff.glDisableis[i].flag, j);
        }
    for(uint32_t i = 0; i < diff.glEnableiCount; i++)
        for(uint32_t j = 0; j < OGL_MAX_ENDISABLEI_INDICES; j++)
        {
            if(diff.glEnableis[i].indices & (0x1u << j))
                glEnablei_MACRO(diff.glEnableis[i].flag, j);
        }

    //change FBO
    if(diff.bindFramebuffers & 0x1u)
        glBindFramebuffer_MACRO(GL_DRAW_FRAMEBUFFER, diff.glBindFramebuffer_vals[0]);
    if(diff.bindFramebuffers & 0x2u)
        glBindFramebuffer_MACRO(GL_READ_FRAMEBUFFER, diff.glBindFramebuffer_vals[1]);

    if(diff.glClampColor_val != GL_INVALID_ENUM)
        glClampColor_MACRO(GL_CLAMP_READ_COLOR, diff.glClampColor_val);

    for(size_t i = 0; i < diff.glPixelStoreiCount; i++)
        glPixelStorei(diff.glPixelStorei_vals[i].first, diff.glPixelStorei_vals[i].second);

    if(diff.setBuffers[EGBT_PIXEL_PACK_BUFFER])
        glBindBuffer_MACRO(GL_PIXEL_PACK_BUFFER, diff.bindBuffer[EGBT_PIXEL_PACK_BUFFER]);
    if(diff.setBuffers[EGBT_PIXEL_UNPACK_BUFFER])
        glBindBuffer_MACRO(GL_PIXEL_UNPACK_BUFFER, diff.bindBuffer[EGBT_PIXEL_UNPACK_BUFFER]);

    if(diff.setPrimitiveRestartIndex)
        glPrimitiveRestartIndex_MACRO(diff.glPrimitiveRestartIndex_val);

    //change XFB
    if(diff.changeXFormFeedback)
        glBindTransformFeedback_MACRO(GL_TRANSFORM_FEEDBACK, diff.glBindTransformFeedback_val);

    //change Shader
    if(diff.changeGlProgram)
        glUseProgram_MACRO(diff.glUseProgram_val);
    if(diff.changeGlProgramPipeline)  //!this has no effect with an active program except state modification
        glBindProgramPipeline_MACRO(diff.glBindProgramPipeline_val);

    //
    if(diff.glPatchParameteri_val)
        glPatchParameteri_MACRO(GL_PATCH_VERTICES, diff.glPatchParameteri_val);
    if(diff.glPatchParameterfv_inner[0] > -FLT_MAX)
        glPatchParameterfv_MACRO(GL_PATCH_DEFAULT_INNER_LEVEL, diff.glPatchParameterfv_inner);
    if(diff.glPatchParameterfv_outer[0] > -FLT_MAX)
        glPatchParameterfv_MACRO(GL_PATCH_DEFAULT_OUTER_LEVEL, diff.glPatchParameterfv_outer);

    //
    for(uint32_t i = 0; i < diff.setBufferRanges; i++)
    {
        if(diff.bindBufferRange[i].object)
        {
            glBindBufferRange_MACRO(diff.bindBufferRange[i].bindPoint, diff.bindBufferRange[i].index,
                diff.bindBufferRange[i].object,
                diff.bindBufferRange[i].offset, diff.bindBufferRange[i].size);
        }
        else
        {
            COpenGLExtensionHandler::extGlBindBuffersRange(diff.bindBufferRange[i].bindPoint,
                diff.bindBufferRange[i].index, 1,
                NULL, NULL, NULL);
        }
    }

    //change ROP
    if(diff.resetPolygonOffset)
        glPolygonOffset(diff.glPolygonOffset_factor, diff.glPolygonOffset_units);

    {
        size_t j = 0;
        for(uint16_t i = 0; i < OGL_STATE_MAX_VIEWPORTS; i++)
        {
            if(diff.setDepthRange & (uint16_t(1) << i))
            {
                glDepthRangeIndexed_MACRO(i, diff.glDepthRangeArray_vals[j][0], diff.glDepthRangeArray_vals[j][1]);
                j++;
            }
        }
        j = 0;
        for(uint16_t i = 0; i < OGL_STATE_MAX_VIEWPORTS; i++)
        {
            if(diff.setViewportArray & (uint16_t(1) << i))
                glViewportIndexedfv_MACRO(i, diff.glViewportArray_vals[j++]);
        }
        j = 0;
        for(uint16_t i = 0; i < OGL_STATE_MAX_VIEWPORTS; i++)
        {
            if(diff.setScissorBox & (uint16_t(1) << i))
                glScissorIndexedv_MACRO(i, diff.glScissorArray_vals[j++]);
        }
    }

    if(diff.glPrimitiveSize[0] > -FLT_MAX)
    {
        glPointSize(diff.glPrimitiveSize[0]);
    }
    if(diff.glPrimitiveSize[1] > -FLT_MAX)
    {
        glLineWidth(diff.glPrimitiveSize[1]);
    }

    if(diff.glLogicOp_val != GL_INVALID_ENUM)
        glLogicOp(diff.glLogicOp_val);

    if(diff.glSampleCoverage_val > -FLT_MAX)
        glSampleCoverage_MACRO(diff.glSampleCoverage_val, diff.glSampleCoverage_invert);

    for(uint32_t i = 0; i < diff.setSampleMask; i++)
        glSampleMaski_MACRO(i, diff.glSampleMaski_vals[i]);

    if(diff.glMinSampleShading_val > -FLT_MAX)
        glMinSampleShading_MACRO(diff.glMinSampleShading_val);

    if(diff.setBlendColor)
        glBlendColor_MACRO(diff.glBlendColor_vals[0], diff.glBlendColor_vals[1], diff.glBlendColor_vals[2], diff.glBlendColor_vals[3]);

    {
        uint32_t j = 0;
        for(uint32_t i = 0; i < OGL_STATE_MAX_DRAW_BUFFERS; i++)
        {
            if(diff.setBlendEquation & (uint16_t(1) << i))
            {
                glBlendEquationSeparatei_MACRO(i, diff.glBlendEquationSeparatei_vals[j][0], diff.glBlendEquationSeparatei_vals[j][1]);
                j++;
            }
        }
        j = 0;
        for(uint32_t i = 0; i < OGL_STATE_MAX_DRAW_BUFFERS; i++)
        {
            if(diff.setBlendFunc & (uint16_t(1) << i))
            {
                glBlendFuncSeparatei_MACRO(i, diff.glBlendFuncSeparatei_vals[j][0], diff.glBlendFuncSeparatei_vals[j][1], diff.glBlendFuncSeparatei_vals[j][2], diff.glBlendFuncSeparatei_vals[j][3]);
                j++;
            }
        }
    }

    if(diff.setColorMask)
    {
        for(uint32_t i = 0; i < OGL_STATE_MAX_DRAW_BUFFERS; i++)
        {
            if(diff.setColorMask & (uint16_t(1) << i))
            {
                uint64_t compareShift = (i % 16) * 4;
                glColorMaski_MACRO(i, diff.glColorMaski_vals[i / 16] & (0x1ull << compareShift) ? GL_TRUE : GL_FALSE,
                    diff.glColorMaski_vals[i / 16] & (0x2ull << compareShift) ? GL_TRUE : GL_FALSE,
                    diff.glColorMaski_vals[i / 16] & (0x4ull << compareShift) ? GL_TRUE : GL_FALSE,
                    diff.glColorMaski_vals[i / 16] & (0x8ull << compareShift) ? GL_TRUE : GL_FALSE);
            }
        }
    }

    //front
    if(diff.setStencilFunc & 0x1u)
        glStencilFuncSeparate_MACRO(GL_FRONT, diff.glStencilFuncSeparate_func[0], diff.glStencilFuncSeparate_ref[0], diff.glStencilFuncSeparate_mask[0]);
    //back
    if(diff.setStencilFunc & 0x2u)
        glStencilFuncSeparate_MACRO(GL_BACK, diff.glStencilFuncSeparate_func[1], diff.glStencilFuncSeparate_ref[1], diff.glStencilFuncSeparate_mask[1]);

    //front
    if(diff.setStencilOp & 0x1u)
        glStencilOpSeparate_MACRO(GL_FRONT, diff.glStencilOpSeparate_sfail[0], diff.glStencilOpSeparate_dpfail[0], diff.glStencilOpSeparate_dppass[0]);
    //back
    if(diff.setStencilOp & 0x2u)
        glStencilOpSeparate_MACRO(GL_BACK, diff.glStencilOpSeparate_sfail[1], diff.glStencilOpSeparate_dpfail[1], diff.glStencilOpSeparate_dppass[1]);

    //
    if(diff.setStencilMask & 0x1u)
        glStencilMaskSeparate_MACRO(GL_FRONT, diff.glStencilMaskSeparate_mask[0]);
    //
    if(diff.setStencilMask & 0x2u)
        glStencilMaskSeparate_MACRO(GL_BACK, diff.glStencilMaskSeparate_mask[1]);

    if(diff.glDepthFunc_val != GL_INVALID_ENUM)
        glDepthFunc(diff.glDepthFunc_val);

    if(diff.setDepthMask)
        glDepthMask(diff.setDepthMask & 0x1u);

    if(diff.setBuffers[EGBT_DRAW_INDIRECT_BUFFER])
        glBindBuffer_MACRO(GL_DRAW_INDIRECT_BUFFER, diff.bindBuffer[EGBT_DRAW_INDIRECT_BUFFER]);

    if(diff.setBuffers[EGBT_DISPATCH_INDIRECT_BUFFER])
        glBindBuffer_MACRO(GL_DISPATCH_INDIRECT_BUFFER, diff.bindBuffer[EGBT_DISPATCH_INDIRECT_BUFFER]);

    //change Images
    for(size_t i = 0; i < diff.setImageBindings; i++)
    {
        const COpenGLStateDiff::ImageToBind& imb = diff.bindImages[i];
        glBindImageTexture_MACRO(imb.index, imb.obj, imb.level, imb.layered, imb.layer, imb.access, imb.format);
    }

    //change Texture
    for(size_t i = 0; i < diff.texturesToBind; i++)
        SPECIAL_glBindTextureUnit_MACRO(diff.bindTextures[i].index, diff.bindTextures[i].obj, diff.bindTextures[i].target);
    for(size_t i = 0; i < diff.samplersToBind; i++)
        glBindSampler_MACRO(diff.bindSamplers[i].index, diff.bindSamplers[i].obj);

    //change VAO
    if(diff.setVAO)
        glBindVertexArray_MACRO(diff.bindVAO);

    if(diff.glPolygonMode_mode != GL_INVALID_ENUM)
        glPolygonMode(GL_FRONT_AND_BACK, diff.glPolygonMode_mode);
    if(diff.glFrontFace_val != GL_INVALID_ENUM)
        glFrontFace(diff.glFrontFace_val);
    if(diff.glCullFace_val != GL_INVALID_ENUM)
        glCullFace(diff.glCullFace_val);
}

#endif

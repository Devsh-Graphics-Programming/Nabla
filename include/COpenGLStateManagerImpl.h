// Copyright (C) 2017- Mateusz Kielan
// This file is part of the "IrrlichtBAW".
#include "COpenGLStateManager.h"

#ifndef __C_OPENGL_STATE_MANAGER_IMPLEMENTATION_H_INCLUDED__
#define __C_OPENGL_STATE_MANAGER_IMPLEMENTATION_H_INCLUDED__

#ifndef glHint_MACRO
#define glHint_MACRO glHint
#endif // glHint_MACRO

#ifndef glProvokingVertex_MACRO
#define glProvokingVertex_MACRO glProvokingVertex
#endif // glProvokingVertex_MACRO

#ifndef glEnable_MACRO
#define glEnable_MACRO glEnable
#endif // glEnable_MACRO
#ifndef glDisable_MACRO
#define glDisable_MACRO glDisable
#endif // glDisable_MACRO

#ifndef glEnablei_MACRO
#define glEnablei_MACRO glEnablei
#endif // glEnablei_MACRO
#ifndef glDisablei_MACRO
#define glDisablei_MACRO glDisablei
#endif // glDisablei_MACRO


#ifndef glUseProgram_MACRO
#define glUseProgram_MACRO glUseProgram
#endif // glUseProgram_MACRO
#ifndef glBindProgramPipeline_MACRO
#define glBindProgramPipeline_MACRO glBindProgramPipeline
#endif // glBindProgramPipeline_MACRO


#ifndef glPolygonOffset_MACRO
#define glPolygonOffset_MACRO glPolygonOffset
#endif // glPolygonOffset_MACRO


#ifndef glDepthRangeIndexed_MACRO
#define glDepthRangeIndexed_MACRO glDepthRangeIndexed
#endif // glDepthRangeIndexed_MACRO


#ifndef glViewportIndexedv_MACRO
#define glViewportIndexedv_MACRO glViewportIndexedv
#endif // glViewportIndexedv_MACRO


#ifndef glScissorIndexedv_MACRO
#define glScissorIndexedv_MACRO glScissorIndexedv
#endif // glScissorIndexedv_MACRO


#ifndef glPointSize_MACRO
#define glPointSize_MACRO glPointSize
#endif // glPointSize_MACRO


#ifndef glLineWidth_MACRO
#define glLineWidth_MACRO glLineWidth
#endif // glLineWidth_MACRO


#ifndef glLogicOp_MACRO
#define glLogicOp_MACRO glLogicOp
#endif // glLogicOp_MACRO

#ifndef glBlendColor_MACRO
#define glBlendColor_MACRO glBlendColor
#endif // glBlendColor_MACRO


#ifndef glBlendEquationSeparatei_MACRO
#define glBlendEquationSeparatei_MACRO glBlendEquationSeparatei
#endif // glBlendEquationSeparatei_MACRO


#ifndef glBlendFuncSeparatei_MACRO
#define glBlendFuncSeparatei_MACRO glBlendFuncSeparatei
#endif // glBlendFuncSeparatei_MACRO


#ifndef SPECIAL_glBindTextureUnit_MACRO
#error "No Override for Active Texture Setting"
#endif // SPECIAL_glBindTextureUnit_MACRO


#ifndef glBindSampler_MACRO
#define glBindSampler_MACRO glBindSampler
#endif // glBindSampler_MACRO

/*
#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO


#ifndef _MACRO
#define _MACRO
#endif // _MACRO
*/

namespace irr
{
namespace video
{

inline void executeGLDiff(const COpenGLStateDiff& diff)
{
    //set hints
    for (uint8_t i=0; i<diff.hintsToSet; i++)
        glHint_MACRO(diff.glHint_pair[i][0],diff.glHint_pair[i][1]);

    if (diff.glProvokingVertex_val!=GL_INVALID_ENUM)
        glProvokingVertex_MACRO(diff.glProvokingVertex_val);

    //enable/disable
    for (uint32_t i=0; i<diff.glDisableCount; i++)
        glDisable_MACRO(diff.glDisables[i]);
    for (uint32_t i=0; i<diff.glEnableCount; i++)
        glEnable_MACRO(diff.glEnables[i]);

    for (uint32_t i=0; i<diff.glDisableiCount; i++)
    for (uint8_t j=0; j<OGL_MAX_ENDISABLEI_INDICES; j++)
    {
        if (diff.glDisableis[i].indices&(0x1u<<j))
            glDisablei_MACRO(diff.glDisableis[i].flag,j);
    }
    for (uint32_t i=0; i<diff.glEnableiCount; i++)
    for (uint32_t j=0; j<OGL_MAX_ENDISABLEI_INDICES; j++)
    {
        if (diff.glEnableis[i].indices&(0x1u<<j))
            glEnablei_MACRO(diff.glEnableis[i].flag,j);
    }

    //change FBO

    //change Shader
    if (diff.changeGlProgram)
        glUseProgram_MACRO(diff.glUseProgram_val);
    if (diff.changeGlProgramPipeline) //!this has no effect with an active program except state modification
        glBindProgramPipeline_MACRO(diff.glBindProgramPipeline_val);

    //change ROP
    if (diff.resetPolygonOffset)
        glPolygonOffset_MACRO(diff.glPolygonOffset_factor,diff.glPolygonOffset_units);

    {
        size_t j=0;
        for (uint8_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
        {
            if (diff.setDepthRange&(uint8_t(1)<<i))
            {
                glDepthRangeIndexed_MACRO(i,diff.glDepthRangeArray_vals[j][0],diff.glDepthRangeArray_vals[j][1]);
                j++;
            }
        }
        j=0;
        for (uint8_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
        {
            if (diff.setViewportArray&(uint8_t(1)<<i))
                glViewportIndexedv_MACRO(i,diff.glViewportArray_vals[j++]);
        }
        j=0;
        for (uint8_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
        {
            if (diff.setScissorBox&(uint8_t(1)<<i))
                glScissorIndexedv_MACRO(i,diff.glScissorArray_vals[j++]);
        }
    }

    if (diff.glPrimitiveSize[0]==diff.glPrimitiveSize[0])
        glPointSize_MACRO(diff.glPrimitiveSize[0]);
    if (diff.glPrimitiveSize[1]==diff.glPrimitiveSize[1])
        glLineWidth_MACRO(diff.glPrimitiveSize[1]);

    if (diff.glLogicOp_val!=GL_INVALID_ENUM)
        glLogicOp_MACRO(diff.glLogicOp_val);

    if (diff.setBlendColor)
        glBlendColor_MACRO(diff.glBlendColor_vals[0],diff.glBlendColor_vals[1],diff.glBlendColor_vals[2],diff.glBlendColor_vals[3]);

    {
        size_t j=0;
        for (size_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
        {
            if (diff.setBlendEquation&(uint8_t(1)<<i))
            {
                glBlendEquationSeparatei_MACRO(i,diff.glBlendEquationSeparatei_vals[0],diff.glBlendEquationSeparatei_vals[1]);
                j++;
            }
        }
        j=0;
        for (size_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
        {
            if (diff.setBlendFunc&(uint8_t(1)<<i))
            {
                glBlendFuncSeparatei_MACRO(i,diff.glBlendEquationSeparatei_vals[0],diff.glBlendEquationSeparatei_vals[1],diff.glBlendEquationSeparatei_vals[2],diff.glBlendEquationSeparatei_vals[3]);
                j++;
            }
        }
    }


    //change Texture
    for (size_t i=0; i<texturesToBind; i++)
        SPECIAL_glBindTextureUnit_MACRO(diff.bindTextures[i].index,diff.bindTextures[i].obj,diff.bindTextures[i].target);
    for (size_t i=0; i<samplersToBind; i++)
        glBindSampler_MACRO(diff.bindSamplers[i].index,diff.bindSamplers[i].obj);

    //change VAO

    //change UBO
}


} // end namespace video
} // end namespace irr

#endif


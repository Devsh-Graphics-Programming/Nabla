// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPENGL_SHADER_LANGUAGE_MATERIAL_RENDERER_H_INCLUDED__
#define __C_OPENGL_SHADER_LANGUAGE_MATERIAL_RENDERER_H_INCLUDED__

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLExtensionHandler.h"


#include "IMaterialRenderer.h"
#include "IMaterialRendererServices.h"
#include "IShaderConstantSetCallBack.h"
#include "IGPUProgrammingServices.h"



namespace irr
{
namespace video
{

class COpenGLDriver;
class IShaderConstantSetCallBack;

//! Class for using GLSL shaders with OpenGL
//! Please note: This renderer implements its own IMaterialRendererServices
class COpenGLSLMaterialRenderer : public IMaterialRenderer, public IMaterialRendererServices
{
    protected:
        //! Destructor
        virtual ~COpenGLSLMaterialRenderer();

    public:
        //! Constructor
        COpenGLSLMaterialRenderer(
            COpenGLDriver* driver,
            int32_t& outMaterialTypeNr,
            const char* vertexShaderProgram = 0,
            const char* vertexShaderEntryPointName = 0,
            const char* pixelShaderProgram = 0,
            const char* pixelShaderEntryPointName = 0,
            const char* geometryShaderProgram = 0,
            const char* geometryShaderEntryPointName = "main",
            const char* controlShaderProgram = 0,
            const char* controlShaderEntryPointName="main",
            const char* evaluationShaderProgram = 0,
            const char* evaluationShaderEntryPointName="main",
            uint32_t patchVertices=3,
            IShaderConstantSetCallBack* callback = 0,
            E_MATERIAL_TYPE baseMaterial = EMT_SOLID,
            const char** xformFeedbackOutputs = NULL,
            const uint32_t& xformFeedbackOutputCount = 0,
            int32_t userData = 0);

        virtual void OnSetMaterial(const SGPUMaterial& material, const SGPUMaterial& lastMaterial,
            bool resetAllRenderstates, IMaterialRendererServices* services);

        virtual bool OnRender(IMaterialRendererServices* service);

        virtual void OnUnsetMaterial();

        virtual bool isTransparent() const; //depr

        //! Returns if it's a tessellation shader.
        virtual bool isTessellation() const {return tessellationPatchVertices!=-1;}

        // implementations for the render services
        virtual void setBasicRenderStates(const SGPUMaterial& material, const SGPUMaterial& lastMaterial, bool resetAllRenderstates);
        virtual void setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number=1);
        virtual IVideoDriver* getVideoDriver();

    protected:

        void init(int32_t& outMaterialTypeNr,
            const char* vertexShaderProgram,
            const char* pixelShaderProgram,
            const char* geometryShaderProgram,
            const char* controlShaderProgram,
            const char* evaluationShaderProgram,
            uint32_t patchVertices=3,
            const char** xformFeedbackOutputs = NULL,
            const uint32_t& xformFeedbackOutputCount = 0);

        bool createProgram();
        bool createShader(GLenum shaderType, const char* shader);
        bool linkProgram();

        COpenGLDriver* Driver;
        IShaderConstantSetCallBack* CallBack;
        E_MATERIAL_TYPE BaseMaterial;

        int32_t tessellationPatchVertices;//is a Tesselation Shader?

        GLuint Program2;
        GLint activeUniformCount;
        int32_t UserData;
    #ifdef _IRR_DEBUG
        core::vector<SConstantLocationNamePair> debugConstants;
        core::vector<GLuint> debugConstantIndices;
    #endif
};


} // end namespace video
} // end namespace irr

#endif // compile with OpenGL
#endif // if included


// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

// This file was originally written by William Finlayson.  I (Nikolaus
// Gebhardt) did some minor modifications and changes to it and integrated it
// into Irrlicht. Thanks a lot to William for his work on this and that he gave
// me his permission to add it into Irrlicht using the zlib license.

// After Irrlicht 0.12, Michael Zoech did some improvements to this renderer, I
// merged this into Irrlicht 0.14, thanks to him for his work.

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLSLMaterialRenderer.h"
#include "IGPUProgrammingServices.h"
#include "IMaterialRendererServices.h"
#include "IVideoDriver.h"
#include "os.h"
#include "COpenGLDriver.h"

namespace irr
{
namespace video
{


//! Constructor
COpenGLSLMaterialRenderer::COpenGLSLMaterialRenderer(video::COpenGLDriver* driver,
		int32_t& outMaterialTypeNr, const char* vertexShaderProgram,
		const char* vertexShaderEntryPointName,
		const char* pixelShaderProgram,
		const char* pixelShaderEntryPointName,
		const char* geometryShaderProgram,
		const char* geometryShaderEntryPointName,
		const char* controlShaderProgram,
		const char* controlShaderEntryPointName,
		const char* evaluationShaderProgram,
		const char* evaluationShaderEntryPointName,
		uint32_t patchVertices,
		IShaderConstantSetCallBack* callback,
		E_MATERIAL_TYPE baseMaterial,
        const char** xformFeedbackOutputs,
        const uint32_t& xformFeedbackOutputCount,
		int32_t userData)
	: Driver(driver), UserData(userData), tessellationPatchVertices(patchVertices)
{
	#ifdef _IRR_DEBUG
	setDebugName("COpenGLSLMaterialRenderer");
	#endif

	//entry points must always be main, and the compile target isn't selectable
	//it is fine to ignore what has been asked for, as the compiler should spot anything wrong
	//just check that GLSL is available

	//init(outMaterialTypeNr, vertexShaderProgram, pixelShaderProgram, geometryShaderProgram,
    //    controlShaderProgram,evaluationShaderProgram,patchVertices,xformFeedbackOutputs,xformFeedbackOutputCount);
}


//! Destructor
COpenGLSLMaterialRenderer::~COpenGLSLMaterialRenderer()
{

}


void COpenGLSLMaterialRenderer::init(int32_t& outMaterialTypeNr,
		const char* vertexShaderProgram,
		const char* pixelShaderProgram,
		const char* geometryShaderProgram,
		const char* controlShaderProgram,
		const char* evaluationShaderProgram,
		uint32_t patchVertices,
        const char** xformFeedbackOutputs,
        const uint32_t& xformFeedbackOutputCount)
{
    /*
	outMaterialTypeNr = -1;

	if (!createProgram())
		return;

	if (vertexShaderProgram)
		if (!createShader(GL_VERTEX_SHADER, vertexShaderProgram))
			return;

	if (pixelShaderProgram)
		if (!createShader(GL_FRAGMENT_SHADER, pixelShaderProgram))
			return;

	if (geometryShaderProgram)
	{
		if (!createShader(GL_GEOMETRY_SHADER, geometryShaderProgram))
			return;
	}

    if (controlShaderProgram && evaluationShaderProgram)
    {
        if (!createShader(GL_TESS_CONTROL_SHADER,controlShaderProgram))
            return;
        if (!createShader(GL_TESS_EVALUATION_SHADER,evaluationShaderProgram))
            return;
        tessellationPatchVertices = patchVertices;
    }
    */

    //if (xformFeedbackOutputCount>0&&xformFeedbackOutputs)
    //    COpenGLExtensionHandler::extGlTransformFeedbackVaryings(Program2,xformFeedbackOutputCount,xformFeedbackOutputs,GL_INTERLEAVED_ATTRIBS);

	//if (!linkProgram())
	//	return;


    //THIS IS SICK THAT IT'S NOT IN DRIVER CODE XD I WAS LOOKING FOR THIS LIKE HALF AN HOUR
	// register myself as new material
	//outMaterialTypeNr = Driver->addMaterialRenderer(this);
}


bool COpenGLSLMaterialRenderer::OnRender(IMaterialRendererServices* service)
{
    // TODO WARNING tesselation is not really operational now because of this. Maybe we can put `tessellationPatchVertices` into SMaterial?

    /*if (tessellationPatchVertices!=-1)
    {
        COpenGLExtensionHandler::extGlPatchParameteri(GL_PATCH_VERTICES, tessellationPatchVertices);
        GLfloat outer[] = {1.f,1.f,1.f,1.f};
        COpenGLExtensionHandler::extGlPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL,outer);
        GLfloat inner[] = {1.f,1.f};
        COpenGLExtensionHandler::extGlPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL,inner);
    }*/

	return true;
}


void COpenGLSLMaterialRenderer::OnSetMaterial(const video::SGPUMaterial& material,
				const video::SGPUMaterial& lastMaterial,
				bool resetAllRenderstates,
				video::IMaterialRendererServices* services)
{
    Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);

    //TODO we have to come up with some way to tell engine which blend settings to set without those IMaterialRenderers

	/*if (material.Pipeline != lastMaterial.Pipeline || resetAllRenderstates)
	{
		if (BaseMaterial==EMT_TRANSPARENT_ADD_COLOR)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        else if (BaseMaterial==EMT_TRANSPARENT_ALPHA_CHANNEL||BaseMaterial==EMT_TRANSPARENT_VERTEX_ALPHA)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}*/
}


void COpenGLSLMaterialRenderer::OnUnsetMaterial()
{
}


void COpenGLSLMaterialRenderer::setBasicRenderStates(const SGPUMaterial& material,
						const SGPUMaterial& lastMaterial,
						bool resetAllRenderstates)
{
	// forward
	Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);
}

void COpenGLSLMaterialRenderer::setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number)
{
}

IVideoDriver* COpenGLSLMaterialRenderer::getVideoDriver()
{
	return Driver;
}



//! Returns if the material is transparent.
bool COpenGLSLMaterialRenderer::isTransparent() const
{
	return false;
}

} // end namespace video
} // end namespace irr


#endif


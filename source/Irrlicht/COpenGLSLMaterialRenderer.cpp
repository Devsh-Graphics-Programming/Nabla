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
	: Driver(driver), CallBack(callback), BaseMaterial(baseMaterial),
		Program2(0), UserData(userData), tessellationPatchVertices(-1), activeUniformCount(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("COpenGLSLMaterialRenderer");
	#endif

	//entry points must always be main, and the compile target isn't selectable
	//it is fine to ignore what has been asked for, as the compiler should spot anything wrong
	//just check that GLSL is available

	if (CallBack)
		CallBack->grab();

	init(outMaterialTypeNr, vertexShaderProgram, pixelShaderProgram, geometryShaderProgram,
        controlShaderProgram,evaluationShaderProgram,patchVertices,xformFeedbackOutputs,xformFeedbackOutputCount);
}


//! Destructor
COpenGLSLMaterialRenderer::~COpenGLSLMaterialRenderer()
{
	if (CallBack)
		CallBack->drop();

	if (Program2)
	{
		GLuint shaders[8];
		GLint count;
		COpenGLExtensionHandler::extGlGetAttachedShaders(Program2, 8, &count, shaders);
		// avoid bugs in some drivers, which return larger numbers
		count=core::min_(count,8);
		for (GLint i=0; i<count; ++i)
			COpenGLExtensionHandler::extGlDeleteShader(shaders[i]);
		COpenGLExtensionHandler::extGlDeleteProgram(Program2);
		Program2 = 0;
	}
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

    if (CallBack)
        CallBack->PreLink(Program2);

    if (xformFeedbackOutputCount>0&&xformFeedbackOutputs)
        COpenGLExtensionHandler::extGlTransformFeedbackVaryings(Program2,xformFeedbackOutputCount,xformFeedbackOutputs,GL_INTERLEAVED_ATTRIBS);

	if (!linkProgram())
		return;



	// register myself as new material
	outMaterialTypeNr = Driver->addMaterialRenderer(this);

    activeUniformCount = 0;
    //get uniforms
    COpenGLExtensionHandler::extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORMS, &activeUniformCount);

    if (activeUniformCount == 0)
    {
        // no uniforms
        return;
    }

    GLint maxlen = 0;
    COpenGLExtensionHandler::extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxlen);

    if (maxlen < 0)
    {
        os::Printer::log("GLSL: failed to retrieve uniform information", ELL_ERROR);
        return;
    }
    maxlen = core::max_(maxlen,36); // gl_MVPInv for Intel drivers (irretards)

    // seems that some implementations use an extra null terminator
    ++maxlen;

    core::vector<SConstantLocationNamePair> constants;


    char *buf = new char[maxlen];
    for (GLuint i=0; i < activeUniformCount; ++i)
    {
        SConstantLocationNamePair pr;
        GLint length;
        GLenum type;

        COpenGLExtensionHandler::extGlGetActiveUniform(Program2, i, maxlen, NULL, &length, &type, reinterpret_cast<GLchar*>(buf));

        pr.name = buf;
		GLint index = COpenGLExtensionHandler::extGlGetUniformLocation(Program2,pr.name.c_str());
        pr.location = index;
        pr.length = length;
        pr.type = getIrrUniformType(type);
        constants.push_back(pr);
#ifdef _IRR_DEBUG
        debugConstantIndices.push_back(i);
#endif
    }

    delete [] buf;


    if (CallBack)
    {
        GLint oldProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM,&oldProgram);
		COpenGLExtensionHandler::extGlUseProgram(Program2);
        CallBack->PostLink(this,(E_MATERIAL_TYPE)outMaterialTypeNr,constants);
		COpenGLExtensionHandler::extGlUseProgram(oldProgram);
    }

#ifdef _IRR_DEBUG
    debugConstants = constants;
#endif
}


bool COpenGLSLMaterialRenderer::OnRender(IMaterialRendererServices* service)
{
	// call callback to set shader constants
	if (CallBack && Program2)
		CallBack->OnSetConstants(this, UserData);

    if (tessellationPatchVertices!=-1)
    {
        COpenGLExtensionHandler::extGlPatchParameteri(GL_PATCH_VERTICES, tessellationPatchVertices);
        GLfloat outer[] = {1.f,1.f,1.f,1.f};
        COpenGLExtensionHandler::extGlPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL,outer);
        GLfloat inner[] = {1.f,1.f};
        COpenGLExtensionHandler::extGlPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL,inner);
    }

	return true;
}


void COpenGLSLMaterialRenderer::OnSetMaterial(const video::SGPUMaterial& material,
				const video::SGPUMaterial& lastMaterial,
				bool resetAllRenderstates,
				video::IMaterialRendererServices* services)
{
    Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);

	if (material.MaterialType != lastMaterial.MaterialType || resetAllRenderstates)
	{
		COpenGLExtensionHandler::extGlUseProgram(Program2);

		if (BaseMaterial==EMT_TRANSPARENT_ADD_COLOR)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        else if (BaseMaterial==EMT_TRANSPARENT_ALPHA_CHANNEL||BaseMaterial==EMT_TRANSPARENT_VERTEX_ALPHA)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	//let callback know used material
	if (CallBack)
		CallBack->OnSetMaterial(this,material,lastMaterial);
}


void COpenGLSLMaterialRenderer::OnUnsetMaterial()
{
    if (CallBack)
        CallBack->OnUnsetMaterial();
}


bool COpenGLSLMaterialRenderer::createProgram()
{
	Program2 = COpenGLExtensionHandler::extGlCreateProgram();

	return true;
}


bool COpenGLSLMaterialRenderer::createShader(GLenum shaderType, const char* shader)
{
    GLuint shaderHandle = COpenGLExtensionHandler::extGlCreateShader(shaderType);
    COpenGLExtensionHandler::extGlShaderSource(shaderHandle, 1, &shader, NULL);
    COpenGLExtensionHandler::extGlCompileShader(shaderHandle);

    GLint status = 0;

    COpenGLExtensionHandler::extGlGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE)
    {
        os::Printer::log("GLSL shader failed to compile", ELL_ERROR);
        // check error message and log it
        GLint maxLength=0;
        GLint length;
        COpenGLExtensionHandler::extGlGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH,
                &maxLength);

        if (maxLength)
        {
            GLchar *infoLog = new GLchar[maxLength];
            COpenGLExtensionHandler::extGlGetShaderInfoLog(shaderHandle, maxLength, &length, infoLog);
            os::Printer::log(reinterpret_cast<const char*>(infoLog), ELL_ERROR);
            delete [] infoLog;
        }

        COpenGLExtensionHandler::extGlDeleteShader(shaderHandle);

        return false;
    }

    COpenGLExtensionHandler::extGlAttachShader(Program2, shaderHandle);

	return true;
}


bool COpenGLSLMaterialRenderer::linkProgram()
{
    COpenGLExtensionHandler::extGlLinkProgram(Program2);

    GLint status = 0;

    COpenGLExtensionHandler::extGlGetProgramiv(Program2, GL_LINK_STATUS, &status);

    if (!status)
    {
        os::Printer::log("GLSL shader program failed to link", ELL_ERROR);
        // check error message and log it
        GLint maxLength=0;
        GLsizei length;
        COpenGLExtensionHandler::extGlGetProgramiv(Program2, GL_INFO_LOG_LENGTH, &maxLength);

        if (maxLength)
        {
            GLchar *infoLog = new GLchar[maxLength];
            COpenGLExtensionHandler::extGlGetProgramInfoLog(Program2, maxLength, &length, infoLog);
            os::Printer::log(reinterpret_cast<const char*>(infoLog), ELL_ERROR);
            delete [] infoLog;
        }

        return false;
    }



    // get uniforms information

    activeUniformCount = 0;
    COpenGLExtensionHandler::extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORMS, &activeUniformCount);

    if (activeUniformCount == 0)
    {
        // no uniforms
        return true;
    }

    GLint maxlen = 0;
    COpenGLExtensionHandler::extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxlen);

    if (maxlen < 0)
    {
        os::Printer::log("GLSL: failed to retrieve uniform information", ELL_ERROR);
        return false;
    }




	return true;
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
    if (location<0)
    {
#ifdef _IRR_DEBUG
        os::Printer::log("Cannot set shader constant, uniform index out of range.", ELL_ERROR);
#endif
        return;
    }

    GLsizei cnt = int32_t(number);
    GLint loc = int32_t(location);

    constexpr bool isRowMajor = true;

    switch (type)
    {
    case ESCT_FLOAT:
        COpenGLExtensionHandler::extGlProgramUniform1fv(Program2,loc,cnt,(GLfloat*)data);
        break;
    case ESCT_FLOAT_VEC2:
        COpenGLExtensionHandler::extGlProgramUniform2fv(Program2,loc,cnt,(GLfloat*)data);
        break;
    case ESCT_FLOAT_VEC3:
        COpenGLExtensionHandler::extGlProgramUniform3fv(Program2,loc,cnt,(GLfloat*)data);
        break;
    case ESCT_FLOAT_VEC4:
        COpenGLExtensionHandler::extGlProgramUniform4fv(Program2,loc,cnt,(GLfloat*)data);
        break;
    case ESCT_INT:
        COpenGLExtensionHandler::extGlProgramUniform1iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_INT_VEC2:
        COpenGLExtensionHandler::extGlProgramUniform2iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_INT_VEC3:
        COpenGLExtensionHandler::extGlProgramUniform3iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_INT_VEC4:
        COpenGLExtensionHandler::extGlProgramUniform4iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_UINT:
        COpenGLExtensionHandler::extGlProgramUniform1uiv(Program2,loc,cnt,(GLuint*)data);
        break;
    case ESCT_UINT_VEC2:
        COpenGLExtensionHandler::extGlProgramUniform2uiv(Program2,loc,cnt,(GLuint*)data);
        break;
    case ESCT_UINT_VEC3:
        COpenGLExtensionHandler::extGlProgramUniform3uiv(Program2,loc,cnt,(GLuint*)data);
        break;
    case ESCT_UINT_VEC4:
        COpenGLExtensionHandler::extGlProgramUniform4uiv(Program2,loc,cnt,(GLuint*)data);
        break;
    case ESCT_BOOL:
        COpenGLExtensionHandler::extGlProgramUniform1iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_BOOL_VEC2:
        COpenGLExtensionHandler::extGlProgramUniform2iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_BOOL_VEC3:
        COpenGLExtensionHandler::extGlProgramUniform3iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_BOOL_VEC4:
        COpenGLExtensionHandler::extGlProgramUniform4iv(Program2,loc,cnt,(GLint*)data);
        break;
    case ESCT_FLOAT_MAT2:
        COpenGLExtensionHandler::extGlProgramUniformMatrix2fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT3:
        COpenGLExtensionHandler::extGlProgramUniformMatrix3fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT4:
        COpenGLExtensionHandler::extGlProgramUniformMatrix4fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT2x3:
        COpenGLExtensionHandler::extGlProgramUniformMatrix2x3fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT2x4:
        COpenGLExtensionHandler::extGlProgramUniformMatrix2x4fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT3x2:
        COpenGLExtensionHandler::extGlProgramUniformMatrix3x2fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT3x4:
        COpenGLExtensionHandler::extGlProgramUniformMatrix3x4fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT4x2:
        COpenGLExtensionHandler::extGlProgramUniformMatrix4x2fv(Program2,loc,cnt,isRowMajor,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT4x3:
        COpenGLExtensionHandler::extGlProgramUniformMatrix4x3fv(Program2,loc,cnt,false,(GLfloat*)data); // not yet because core::matrix3x4 is still in use
        break;
#ifdef _IRR_DEBUG
    default:
        os::Printer::log("Cannot set shader constant, wrong uniform type or wrong call type used.", ELL_ERROR);
        return;
#else
    default:
        return;
#endif
    }
}

IVideoDriver* COpenGLSLMaterialRenderer::getVideoDriver()
{
	return Driver;
}



//! Returns if the material is transparent.
bool COpenGLSLMaterialRenderer::isTransparent() const
{
	return BaseMaterial>=EMT_TRANSPARENT_ADD_COLOR&&BaseMaterial<=EMT_TRANSPARENT_VERTEX_ALPHA;
}

} // end namespace video
} // end namespace irr


#endif


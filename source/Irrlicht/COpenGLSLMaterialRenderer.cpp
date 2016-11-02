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
		s32& outMaterialTypeNr, const c8* vertexShaderProgram,
		const c8* vertexShaderEntryPointName,
		const c8* pixelShaderProgram,
		const c8* pixelShaderEntryPointName,
		const c8* geometryShaderProgram,
		const c8* geometryShaderEntryPointName,
		const c8* controlShaderProgram,
		const c8* controlShaderEntryPointName,
		const c8* evaluationShaderProgram,
		const c8* evaluationShaderEntryPointName,
		u32 patchVertices,
		IShaderConstantSetCallBack* callback,
		E_MATERIAL_TYPE baseMaterial,
        const char** xformFeedbackOutputs,
        const uint32_t& xformFeedbackOutputCount,
        const E_XFORM_FEEDBACK_ATTRIBUTE_MODE& attribLayout,
		s32 userData)
	: Driver(driver), CallBack(callback), BaseMaterial(baseMaterial),
		Program2(0), UserData(userData), tessellationPatchVertices(-1), activeUniformCount(0)
{
	#ifdef _DEBUG
	setDebugName("COpenGLSLMaterialRenderer");
	#endif

	//entry points must always be main, and the compile target isn't selectable
	//it is fine to ignore what has been asked for, as the compiler should spot anything wrong
	//just check that GLSL is available

	if (CallBack)
		CallBack->grab();

	init(outMaterialTypeNr, vertexShaderProgram, pixelShaderProgram, geometryShaderProgram,
        controlShaderProgram,evaluationShaderProgram,patchVertices,xformFeedbackOutputs,xformFeedbackOutputCount,attribLayout);
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
		Driver->extGlGetAttachedShaders(Program2, 8, &count, shaders);
		// avoid bugs in some drivers, which return larger numbers
		count=core::min_(count,8);
		for (GLint i=0; i<count; ++i)
			Driver->extGlDeleteShader(shaders[i]);
		Driver->extGlDeleteProgram(Program2);
		Program2 = 0;
	}
}


void COpenGLSLMaterialRenderer::init(s32& outMaterialTypeNr,
		const c8* vertexShaderProgram,
		const c8* pixelShaderProgram,
		const c8* geometryShaderProgram,
		const c8* controlShaderProgram,
		const c8* evaluationShaderProgram,
		u32 patchVertices,
        const char** xformFeedbackOutputs,
        const uint32_t& xformFeedbackOutputCount,
        const E_XFORM_FEEDBACK_ATTRIBUTE_MODE& attribLayout)
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

    if (controlShaderProgram && evaluationShaderProgram && Driver->queryFeature(EVDF_TESSELLATION_SHADER))
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
    {
        if (attribLayout==EXFAM_INTERLEAVED)
            Driver->extGlTransformFeedbackVaryings(Program2,xformFeedbackOutputCount,xformFeedbackOutputs,GL_INTERLEAVED_ATTRIBS);
        else if (attribLayout==EXFAM_SEPARATE)
            Driver->extGlTransformFeedbackVaryings(Program2,xformFeedbackOutputCount,xformFeedbackOutputs,GL_SEPARATE_ATTRIBS);
    }

	if (!linkProgram())
		return;



	// register myself as new material
	outMaterialTypeNr = Driver->addMaterialRenderer(this);

    activeUniformCount = 0;
    //get uniforms
    Driver->extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORMS, &activeUniformCount);

    if (activeUniformCount == 0)
    {
        // no uniforms
        return;
    }

    GLint maxlen = 0;
    Driver->extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxlen);

    if (maxlen < 0)
    {
        os::Printer::log("GLSL: failed to retrieve uniform information", ELL_ERROR);
        return;
    }
    maxlen = core::max_(maxlen,36); // gl_MVPInv for Intel drivers (irretards)

    // seems that some implementations use an extra null terminator
    ++maxlen;

    core::array<SConstantLocationNamePair> constants;


    c8 *buf = new c8[maxlen];
    for (GLuint i=0; i < activeUniformCount; ++i)
    {
        SConstantLocationNamePair pr;
        GLint length;
        GLenum type;

        Driver->extGlGetActiveUniform(Program2, i, maxlen, NULL, &length, &type, reinterpret_cast<GLchar*>(buf));

        pr.name = buf;
		GLint index = Driver->extGlGetUniformLocation(Program2,pr.name.c_str());
        pr.location = index;
        pr.length = length;
        pr.type = getIrrUniformType(type);
        constants.push_back(pr);
#if _DEBUG
        debugConstantIndices.push_back(i);
#endif
    }

    delete [] buf;


    if (CallBack)
    {
        GLint oldProgram;
        glGetIntegerv(GL_CURRENT_PROGRAM,&oldProgram);
		Driver->extGlUseProgram(Program2);
        CallBack->PostLink(this,(E_MATERIAL_TYPE)outMaterialTypeNr,constants);
		Driver->extGlUseProgram(oldProgram);
    }

#if _DEBUG
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
        Driver->extGlPatchParameteri(GL_PATCH_VERTICES, tessellationPatchVertices);
        GLfloat outer[] = {1.f,1.f,1.f,1.f};
        Driver->extGlPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL,outer);
        GLfloat inner[] = {1.f,1.f};
        Driver->extGlPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL,inner);
    }

	return true;
}


void COpenGLSLMaterialRenderer::OnSetMaterial(const video::SMaterial& material,
				const video::SMaterial& lastMaterial,
				bool resetAllRenderstates,
				video::IMaterialRendererServices* services)
{
    Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);

	if (material.MaterialType != lastMaterial.MaterialType || resetAllRenderstates)
	{
		Driver->extGlUseProgram(Program2);

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


//! Returns if the material is transparent.
bool COpenGLSLMaterialRenderer::isTransparent() const
{
	return BaseMaterial>=EMT_TRANSPARENT_ADD_COLOR&&BaseMaterial<=EMT_TRANSPARENT_VERTEX_ALPHA;
}


bool COpenGLSLMaterialRenderer::createProgram()
{
	Program2 = Driver->extGlCreateProgram();

	return true;
}


bool COpenGLSLMaterialRenderer::createShader(GLenum shaderType, const char* shader)
{
    GLuint shaderHandle = Driver->extGlCreateShader(shaderType);
    Driver->extGlShaderSource(shaderHandle, 1, &shader, NULL);
    Driver->extGlCompileShader(shaderHandle);

    GLint status = 0;

    Driver->extGlGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE)
    {
        os::Printer::log("GLSL shader failed to compile", ELL_ERROR);
        // check error message and log it
        GLint maxLength=0;
        GLint length;
        Driver->extGlGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH,
                &maxLength);

        if (maxLength)
        {
            GLchar *infoLog = new GLchar[maxLength];
            Driver->extGlGetShaderInfoLog(shaderHandle, maxLength, &length, infoLog);
            os::Printer::log(reinterpret_cast<const c8*>(infoLog), ELL_ERROR);
            delete [] infoLog;
        }

        Driver->extGlDeleteShader(shaderHandle);

        return false;
    }

    Driver->extGlAttachShader(Program2, shaderHandle);

	return true;
}


bool COpenGLSLMaterialRenderer::linkProgram()
{
    Driver->extGlLinkProgram(Program2);

    GLint status = 0;

    Driver->extGlGetProgramiv(Program2, GL_LINK_STATUS, &status);

    if (!status)
    {
        os::Printer::log("GLSL shader program failed to link", ELL_ERROR);
        // check error message and log it
        GLint maxLength=0;
        GLsizei length;
        Driver->extGlGetProgramiv(Program2, GL_INFO_LOG_LENGTH, &maxLength);

        if (maxLength)
        {
            GLchar *infoLog = new GLchar[maxLength];
            Driver->extGlGetProgramInfoLog(Program2, maxLength, &length, infoLog);
            os::Printer::log(reinterpret_cast<const c8*>(infoLog), ELL_ERROR);
            delete [] infoLog;
        }

        return false;
    }



    // get uniforms information

    activeUniformCount = 0;
    Driver->extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORMS, &activeUniformCount);

    if (activeUniformCount == 0)
    {
        // no uniforms
        return true;
    }

    GLint maxlen = 0;
    Driver->extGlGetProgramiv(Program2, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxlen);

    if (maxlen < 0)
    {
        os::Printer::log("GLSL: failed to retrieve uniform information", ELL_ERROR);
        return false;
    }




	return true;
}


void COpenGLSLMaterialRenderer::setBasicRenderStates(const SMaterial& material,
						const SMaterial& lastMaterial,
						bool resetAllRenderstates)
{
	// forward
	Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);
}

void COpenGLSLMaterialRenderer::setShaderConstant(const void* data, s32 location, E_SHADER_CONSTANT_TYPE type, u32 number)
{
    if (location<0)
    {
#ifdef _DEBUG
        os::Printer::log("Cannot set shader constant, uniform index out of range.", ELL_ERROR);
#endif
        return;
    }
/*
#ifdef _DEBUG
    GLuint index = 0;
    bool found = false;
    for (u32 i=0; i<debugConstants.size(); i++)
    {
        if (debugConstants[i].location==location)
        {
            found = true;
            index = debugConstantIndices[i];
            break;
        }
    }

    if (!found)
    {
        os::Printer::log("Uniform not found.", ELL_ERROR);
        return;
    }
    GLint length;
    GLenum type2;
    GLchar buf[1024];
    Driver->extGlGetActiveUniform(Program2, index, 1023, NULL, &length, &type2, buf);
    if (number>length||type!=getIrrUniformType(type2))
    {
        os::Printer::log("Number of elements or uniform type are ALL WRONG.", ELL_ERROR);
        return;
    }
#endif
*/

    GLsizei cnt = int32_t(number);
    GLint loc = int32_t(location);

    switch (type)
    {
    case ESCT_FLOAT:
        Driver->extGlUniform1fv(loc,cnt,(GLfloat*)data);
        break;
    case ESCT_FLOAT_VEC2:
        Driver->extGlUniform2fv(loc,cnt,(GLfloat*)data);
        break;
    case ESCT_FLOAT_VEC3:
        Driver->extGlUniform3fv(loc,cnt,(GLfloat*)data);
        break;
    case ESCT_FLOAT_VEC4:
        Driver->extGlUniform4fv(loc,cnt,(GLfloat*)data);
        break;
    case ESCT_INT:
        Driver->extGlUniform1iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_INT_VEC2:
        Driver->extGlUniform2iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_INT_VEC3:
        Driver->extGlUniform3iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_INT_VEC4:
        Driver->extGlUniform4iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_UINT:
        Driver->extGlUniform1uiv(loc,cnt,(GLuint*)data);
        break;
    case ESCT_UINT_VEC2:
        Driver->extGlUniform2uiv(loc,cnt,(GLuint*)data);
        break;
    case ESCT_UINT_VEC3:
        Driver->extGlUniform3uiv(loc,cnt,(GLuint*)data);
        break;
    case ESCT_UINT_VEC4:
        Driver->extGlUniform4uiv(loc,cnt,(GLuint*)data);
        break;
    case ESCT_BOOL:
        Driver->extGlUniform1iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_BOOL_VEC2:
        Driver->extGlUniform2iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_BOOL_VEC3:
        Driver->extGlUniform3iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_BOOL_VEC4:
        Driver->extGlUniform4iv(loc,cnt,(GLint*)data);
        break;
    case ESCT_FLOAT_MAT2:
        Driver->extGlUniformMatrix2fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT3:
        Driver->extGlUniformMatrix3fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT4:
        Driver->extGlUniformMatrix4fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT2x3:
        Driver->extGlUniformMatrix2x3fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT2x4:
        Driver->extGlUniformMatrix2x4fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT3x2:
        Driver->extGlUniformMatrix3x2fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT3x4:
        Driver->extGlUniformMatrix3x4fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT4x2:
        Driver->extGlUniformMatrix4x2fv(loc,cnt,false,(GLfloat*)data);
        break;
    case ESCT_FLOAT_MAT4x3:
        Driver->extGlUniformMatrix4x3fv(loc,cnt,false,(GLfloat*)data);
        break;
#ifdef _DEBUG
    default:
        os::Printer::log("Cannot set shader constant, wrong uniform type or wrong call type used.", ELL_ERROR);
        return;
#else
    default:
        return;
#endif
    }
}

void COpenGLSLMaterialRenderer::setShaderTextures(const s32* textureIndices, s32 location, E_SHADER_CONSTANT_TYPE type, u32 number)
{
    if (location<0)
    {
#ifdef _DEBUG
        os::Printer::log("Cannot set shader constant, uniform index out of range.", ELL_ERROR);
#endif
        return;
    }
    /*
#ifdef _DEBUG
    GLuint index = 0;
    bool found = false;
    for (u32 i=0; i<debugConstants.size(); i++)
    {
        if (debugConstants[i].location==location)
        {
            found = true;
            index = debugConstantIndices[i];
            break;
        }
    }

    if (!found)
    {
        os::Printer::log("Uniform not found.", ELL_ERROR);
        return;
    }
    GLint length;
    GLenum type2;
    GLchar buf[1024];
    Driver->extGlGetActiveUniform(Program2, index, 1023, NULL, &length, &type2, buf);
    if (number>length||type!=getIrrUniformType(type2))
    {
        os::Printer::log("Number of elements or uniform type are ALL WRONG.", ELL_ERROR);
        return;
    }
#endif
*/

    GLsizei cnt = number;
    GLint loc = location;

    switch (type)
    {
    case ESCT_SAMPLER_1D:
    case ESCT_SAMPLER_2D:
    case ESCT_SAMPLER_3D:
    case ESCT_SAMPLER_CUBE:
    case ESCT_SAMPLER_1D_SHADOW:
    case ESCT_SAMPLER_2D_SHADOW:
    case ESCT_SAMPLER_1D_ARRAY:
    case ESCT_SAMPLER_2D_ARRAY:
    case ESCT_SAMPLER_1D_ARRAY_SHADOW:
    case ESCT_SAMPLER_2D_ARRAY_SHADOW:
    case ESCT_SAMPLER_2D_MULTISAMPLE:
    case ESCT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case ESCT_SAMPLER_CUBE_SHADOW:
    case ESCT_SAMPLER_BUFFER:
    case ESCT_SAMPLER_2D_RECT:
    case ESCT_SAMPLER_2D_RECT_SHADOW:
    case ESCT_INT_SAMPLER_1D:
    case ESCT_INT_SAMPLER_2D:
    case ESCT_INT_SAMPLER_3D:
    case ESCT_INT_SAMPLER_CUBE:
    case ESCT_INT_SAMPLER_1D_ARRAY:
    case ESCT_INT_SAMPLER_2D_ARRAY:
    case ESCT_INT_SAMPLER_2D_MULTISAMPLE:
    case ESCT_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case ESCT_INT_SAMPLER_BUFFER:
    case ESCT_UINT_SAMPLER_1D:
    case ESCT_UINT_SAMPLER_2D:
    case ESCT_UINT_SAMPLER_3D:
    case ESCT_UINT_SAMPLER_CUBE:
    case ESCT_UINT_SAMPLER_1D_ARRAY:
    case ESCT_UINT_SAMPLER_2D_ARRAY:
    case ESCT_UINT_SAMPLER_2D_MULTISAMPLE:
    case ESCT_UINT_SAMPLER_2D_MULTISAMPLE_ARRAY:
    case ESCT_UINT_SAMPLER_BUFFER:
        Driver->extGlUniform1iv(loc,cnt,(GLint*)textureIndices);
        break;
#ifdef _DEBUG
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

} // end namespace video
} // end namespace irr


#endif


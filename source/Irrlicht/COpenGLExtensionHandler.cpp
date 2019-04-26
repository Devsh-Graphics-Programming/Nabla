// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLExtensionHandler.h"
#include "SMaterial.h" // for MATERIAL_MAX_TEXTURES
#include "coreutil.h"

namespace irr
{
namespace video
{


#define glIsEnabledi_MACRO COpenGLExtensionHandler::extGlIsEnabledi
#define glEnablei_MACRO COpenGLExtensionHandler::extGlEnablei
#define glDisablei_MACRO COpenGLExtensionHandler::extGlDisablei

#define glGetBooleani_v_MACRO COpenGLExtensionHandler::extGlGetBooleani_v

#define glGetFloati_v_MACRO COpenGLExtensionHandler::extGlGetFloati_v
#define glGetIntegeri_v_MACRO COpenGLExtensionHandler::extGlGetIntegeri_v

#define glProvokingVertex_MACRO COpenGLExtensionHandler::extGlProvokingVertex

#define glBindFramebuffer_MACRO COpenGLExtensionHandler::extGlBindFramebuffer

#define glClampColor_MACRO COpenGLExtensionHandler::extGlClampColor
#define glPrimitiveRestartIndex_MACRO COpenGLExtensionHandler::extGlPrimitiveRestartIndex

#define glBindTransformFeedback_MACRO COpenGLExtensionHandler::extGlBindTransformFeedback

#define glUseProgram_MACRO COpenGLExtensionHandler::extGlUseProgram
#define glBindProgramPipeline_MACRO COpenGLExtensionHandler::extGlBindProgramPipeline

#define glPatchParameteri_MACRO COpenGLExtensionHandler::extGlPatchParameteri
#define glPatchParameterfv_MACRO COpenGLExtensionHandler::extGlPatchParameterfv

#define glBindBuffer_MACRO COpenGLExtensionHandler::extGlBindBuffer
#define glBindBufferRange_MACRO(target,index,buffer,offset,size) COpenGLExtensionHandler::extGlBindBuffersRange(target,index,1,&buffer,&offset,&size)

#define glGetNamedBufferParameteri64v_MACRO COpenGLExtensionHandler::extGlGetNamedBufferParameteri64v

#define glDepthRangeIndexed_MACRO COpenGLExtensionHandler::extGlDepthRangeIndexed
#define glViewportIndexedfv_MACRO COpenGLExtensionHandler::extGlViewportIndexedfv
#define glScissorIndexedv_MACRO COpenGLExtensionHandler::extGlScissorIndexedv

#define glSampleCoverage_MACRO COpenGLExtensionHandler::extGlSampleCoverage
#define glSampleMaski_MACRO COpenGLExtensionHandler::extGlSampleMaski
#define glMinSampleShading_MACRO COpenGLExtensionHandler::extGlMinSampleShading

#define glBlendColor_MACRO COpenGLExtensionHandler::extGlBlendColor
#define glBlendEquationSeparatei_MACRO COpenGLExtensionHandler::extGlBlendEquationSeparatei
#define glBlendFuncSeparatei_MACRO COpenGLExtensionHandler::extGlBlendFuncSeparatei
#define glColorMaski_MACRO COpenGLExtensionHandler::extGlColorMaski

#define glStencilFuncSeparate_MACRO COpenGLExtensionHandler::extGlStencilFuncSeparate
#define glStencilOpSeparate_MACRO COpenGLExtensionHandler::extGlStencilOpSeparate
#define glStencilMaskSeparate_MACRO COpenGLExtensionHandler::extGlStencilMaskSeparate

#define glBindImageTexture_MACRO COpenGLExtensionHandler::extGlBindImageTexture

#define glActiveTexture_MACRO COpenGLExtensionHandler::extGlActiveTexture
#define SPECIAL_glBindTextureUnit_MACRO(index,texture,target) COpenGLExtensionHandler::extGlBindTextures(index,1,&texture,&target)
#define glBindSampler_MACRO(index,sampler) COpenGLExtensionHandler::extGlBindSamplers(index,1,&sampler)

#define glBindVertexArray_MACRO COpenGLExtensionHandler::extGlBindVertexArray

#include "COpenGLStateManagerImpl.h"




E_SHADER_CONSTANT_TYPE getIrrUniformType(GLenum oglType)
{
    switch (oglType)
    {
    case GL_FLOAT:
        return ESCT_FLOAT;
    case GL_FLOAT_VEC2:
        return ESCT_FLOAT_VEC2;
    case GL_FLOAT_VEC3:
        return ESCT_FLOAT_VEC3;
    case GL_FLOAT_VEC4:
        return ESCT_FLOAT_VEC4;
    case GL_INT:
        return ESCT_INT;
    case GL_INT_VEC2:
        return ESCT_INT_VEC2;
    case GL_INT_VEC3:
        return ESCT_INT_VEC3;
    case GL_INT_VEC4:
        return ESCT_INT_VEC4;
    case GL_UNSIGNED_INT:
        return ESCT_UINT;
    case GL_UNSIGNED_INT_VEC2:
        return ESCT_UINT_VEC2;
    case GL_UNSIGNED_INT_VEC3:
        return ESCT_UINT_VEC3;
    case GL_UNSIGNED_INT_VEC4:
        return ESCT_UINT_VEC4;
    case GL_BOOL:
        return ESCT_BOOL;
    case GL_BOOL_VEC2:
        return ESCT_BOOL_VEC2;
    case GL_BOOL_VEC3:
        return ESCT_BOOL_VEC3;
    case GL_BOOL_VEC4:
        return ESCT_BOOL_VEC4;
    case GL_FLOAT_MAT2:
        return ESCT_FLOAT_MAT2;
    case GL_FLOAT_MAT3:
        return ESCT_FLOAT_MAT3;
    case GL_FLOAT_MAT4:
        return ESCT_FLOAT_MAT4;
    case GL_FLOAT_MAT2x3:
        return ESCT_FLOAT_MAT2x3;
    case GL_FLOAT_MAT2x4:
        return ESCT_FLOAT_MAT2x4;
    case GL_FLOAT_MAT3x2:
        return ESCT_FLOAT_MAT3x2;
    case GL_FLOAT_MAT3x4:
        return ESCT_FLOAT_MAT3x4;
    case GL_FLOAT_MAT4x2:
        return ESCT_FLOAT_MAT4x2;
    case GL_FLOAT_MAT4x3:
        return ESCT_FLOAT_MAT4x3;
    case GL_SAMPLER_1D:
    default:
        return ESCT_INVALID_COUNT;
    }

    return ESCT_INVALID_COUNT;
}


uint16_t COpenGLExtensionHandler::Version = 0;
uint16_t COpenGLExtensionHandler::ShaderLanguageVersion = 0;
bool COpenGLExtensionHandler::functionsAlreadyLoaded = false;
int32_t COpenGLExtensionHandler::pixelUnpackAlignment = 2;
bool COpenGLExtensionHandler::FeatureAvailable[] = {false};

int32_t COpenGLExtensionHandler::reqUBOAlignment = 0;
int32_t COpenGLExtensionHandler::reqSSBOAlignment = 0;
int32_t COpenGLExtensionHandler::reqTBOAlignment = 0;

uint64_t COpenGLExtensionHandler::maxUBOSize = 0;
uint64_t COpenGLExtensionHandler::maxSSBOSize = 0;
uint64_t COpenGLExtensionHandler::maxTBOSize = 0;
uint64_t COpenGLExtensionHandler::maxBufferSize = 0;

int32_t COpenGLExtensionHandler::minMemoryMapAlignment = 0;

int32_t COpenGLExtensionHandler::MaxComputeWGSize[3]{0, 0, 0};

uint32_t COpenGLExtensionHandler::MaxArrayTextureLayers = 2048;
uint8_t COpenGLExtensionHandler::MaxTextureUnits = 96;
uint8_t COpenGLExtensionHandler::MaxAnisotropy = 8;
uint8_t COpenGLExtensionHandler::MaxUserClipPlanes = 8;
uint8_t COpenGLExtensionHandler::MaxMultipleRenderTargets = 4;
uint32_t COpenGLExtensionHandler::MaxIndices = 65535;
uint32_t COpenGLExtensionHandler::MaxVertices = 0xffffffffu;
uint32_t COpenGLExtensionHandler::MaxVertexStreams = 1;
uint32_t COpenGLExtensionHandler::MaxXFormFeedbackComponents = 64;
uint32_t COpenGLExtensionHandler::MaxGPUWaitTimeout = 0;
uint32_t COpenGLExtensionHandler::InvocationSubGroupSize[2] = {32,64};
uint32_t COpenGLExtensionHandler::MaxGeometryVerticesOut = 65535;
float COpenGLExtensionHandler::MaxTextureLODBias = 0.f;

//uint32_t COpenGLExtensionHandler::MaxXFormFeedbackInterleavedAttributes = GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS;
//uint32_t COpenGLExtensionHandler::MaxXFormFeedbackSeparateAttributes = GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS;

bool COpenGLExtensionHandler::IsIntelGPU = false;
bool COpenGLExtensionHandler::needsDSAFramebufferHack = true;

//
PFNGLISENABLEDIPROC COpenGLExtensionHandler::pGlIsEnabledi = nullptr;
PFNGLENABLEIPROC COpenGLExtensionHandler::pGlEnablei = nullptr;
PFNGLDISABLEIPROC COpenGLExtensionHandler::pGlDisablei = nullptr;
PFNGLGETBOOLEANI_VPROC COpenGLExtensionHandler::pGlGetBooleani_v = nullptr;
PFNGLGETFLOATI_VPROC COpenGLExtensionHandler::pGlGetFloati_v = nullptr;
PFNGLGETINTEGER64VPROC COpenGLExtensionHandler::pGlGetInteger64v = nullptr;
PFNGLGETINTEGERI_VPROC COpenGLExtensionHandler::pGlGetIntegeri_v = nullptr;
PFNGLGETSTRINGIPROC COpenGLExtensionHandler::pGlGetStringi = nullptr;
PFNGLPROVOKINGVERTEXPROC COpenGLExtensionHandler::pGlProvokingVertex = nullptr;
PFNGLCLIPCONTROLPROC COpenGLExtensionHandler::pGlClipControl = nullptr;

//fences
PFNGLFENCESYNCPROC COpenGLExtensionHandler::pGlFenceSync = nullptr;
PFNGLDELETESYNCPROC COpenGLExtensionHandler::pGlDeleteSync = nullptr;
PFNGLCLIENTWAITSYNCPROC COpenGLExtensionHandler::pGlClientWaitSync = nullptr;
PFNGLWAITSYNCPROC COpenGLExtensionHandler::pGlWaitSync = nullptr;

        //textures
PFNGLACTIVETEXTUREPROC COpenGLExtensionHandler::pGlActiveTexture = nullptr;
PFNGLBINDTEXTURESPROC COpenGLExtensionHandler::pGlBindTextures = nullptr;
PFNGLCREATETEXTURESPROC COpenGLExtensionHandler::pGlCreateTextures = nullptr;
PFNGLTEXSTORAGE1DPROC COpenGLExtensionHandler::pGlTexStorage1D = nullptr;
PFNGLTEXSTORAGE2DPROC COpenGLExtensionHandler::pGlTexStorage2D = nullptr;
PFNGLTEXSTORAGE3DPROC COpenGLExtensionHandler::pGlTexStorage3D = nullptr;
PFNGLTEXSTORAGE2DMULTISAMPLEPROC COpenGLExtensionHandler::pGlTexStorage2DMultisample = nullptr;
PFNGLTEXSTORAGE3DMULTISAMPLEPROC COpenGLExtensionHandler::pGlTexStorage3DMultisample = nullptr;
PFNGLTEXBUFFERPROC COpenGLExtensionHandler::pGlTexBuffer = nullptr;
PFNGLTEXBUFFERRANGEPROC COpenGLExtensionHandler::pGlTexBufferRange = nullptr;
PFNGLTEXTURESTORAGE1DPROC COpenGLExtensionHandler::pGlTextureStorage1D = nullptr;
PFNGLTEXTURESTORAGE2DPROC COpenGLExtensionHandler::pGlTextureStorage2D = nullptr;
PFNGLTEXTURESTORAGE3DPROC COpenGLExtensionHandler::pGlTextureStorage3D = nullptr;
PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC COpenGLExtensionHandler::pGlTextureStorage2DMultisample = nullptr;
PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC COpenGLExtensionHandler::pGlTextureStorage3DMultisample = nullptr;
PFNGLTEXTUREBUFFERPROC COpenGLExtensionHandler::pGlTextureBuffer = nullptr;
PFNGLTEXTUREBUFFERRANGEPROC COpenGLExtensionHandler::pGlTextureBufferRange = nullptr;
PFNGLTEXTURESTORAGE1DEXTPROC COpenGLExtensionHandler::pGlTextureStorage1DEXT = nullptr;
PFNGLTEXTURESTORAGE2DEXTPROC COpenGLExtensionHandler::pGlTextureStorage2DEXT = nullptr;
PFNGLTEXTURESTORAGE3DEXTPROC COpenGLExtensionHandler::pGlTextureStorage3DEXT = nullptr;
PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC COpenGLExtensionHandler::pGlTextureStorage2DMultisampleEXT = nullptr;
PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC COpenGLExtensionHandler::pGlTextureStorage3DMultisampleEXT = nullptr;
PFNGLTEXTUREBUFFEREXTPROC COpenGLExtensionHandler::pGlTextureBufferEXT = nullptr;
PFNGLTEXTUREBUFFERRANGEEXTPROC COpenGLExtensionHandler::pGlTextureBufferRangeEXT = nullptr;
PFNGLGETTEXTURESUBIMAGEPROC COpenGLExtensionHandler::pGlGetTextureSubImage = nullptr;
PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC COpenGLExtensionHandler::pGlGetCompressedTextureSubImage = nullptr;
PFNGLGETTEXTUREIMAGEPROC COpenGLExtensionHandler::pGlGetTextureImage = nullptr;
PFNGLGETTEXTUREIMAGEEXTPROC COpenGLExtensionHandler::pGlGetTextureImageEXT = nullptr;
PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC COpenGLExtensionHandler::pGlGetCompressedTextureImage = nullptr;
PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC COpenGLExtensionHandler::pGlGetCompressedTextureImageEXT = nullptr;
PFNGLGETCOMPRESSEDTEXIMAGEPROC COpenGLExtensionHandler::pGlGetCompressedTexImage = nullptr;
PFNGLTEXSUBIMAGE3DPROC COpenGLExtensionHandler::pGlTexSubImage3D = nullptr;
PFNGLMULTITEXSUBIMAGE1DEXTPROC COpenGLExtensionHandler::pGlMultiTexSubImage1DEXT = nullptr;
PFNGLMULTITEXSUBIMAGE2DEXTPROC COpenGLExtensionHandler::pGlMultiTexSubImage2DEXT = nullptr;
PFNGLMULTITEXSUBIMAGE3DEXTPROC COpenGLExtensionHandler::pGlMultiTexSubImage3DEXT = nullptr;
PFNGLTEXTURESUBIMAGE1DPROC COpenGLExtensionHandler::pGlTextureSubImage1D = nullptr;
PFNGLTEXTURESUBIMAGE2DPROC COpenGLExtensionHandler::pGlTextureSubImage2D = nullptr;
PFNGLTEXTURESUBIMAGE3DPROC COpenGLExtensionHandler::pGlTextureSubImage3D = nullptr;
PFNGLTEXTURESUBIMAGE1DEXTPROC COpenGLExtensionHandler::pGlTextureSubImage1DEXT = nullptr;
PFNGLTEXTURESUBIMAGE2DEXTPROC COpenGLExtensionHandler::pGlTextureSubImage2DEXT = nullptr;
PFNGLTEXTURESUBIMAGE3DEXTPROC COpenGLExtensionHandler::pGlTextureSubImage3DEXT = nullptr;
PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC COpenGLExtensionHandler::pGlCompressedTexSubImage1D = nullptr;
PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC COpenGLExtensionHandler::pGlCompressedTexSubImage2D = nullptr;
PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC COpenGLExtensionHandler::pGlCompressedTexSubImage3D = nullptr;
PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage1D = nullptr;
PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage2D = nullptr;
PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage3D = nullptr;
PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage1DEXT = nullptr;
PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage2DEXT = nullptr;
PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage3DEXT = nullptr;
PFNGLCOPYTEXSUBIMAGE3DPROC COpenGLExtensionHandler::pGlCopyTexSubImage3D = nullptr;
PFNGLCOPYTEXTURESUBIMAGE1DPROC COpenGLExtensionHandler::pGlCopyTextureSubImage1D = nullptr;
PFNGLCOPYTEXTURESUBIMAGE2DPROC COpenGLExtensionHandler::pGlCopyTextureSubImage2D = nullptr;
PFNGLCOPYTEXTURESUBIMAGE3DPROC COpenGLExtensionHandler::pGlCopyTextureSubImage3D = nullptr;
PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC COpenGLExtensionHandler::pGlCopyTextureSubImage1DEXT = nullptr;
PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC COpenGLExtensionHandler::pGlCopyTextureSubImage2DEXT = nullptr;
PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC COpenGLExtensionHandler::pGlCopyTextureSubImage3DEXT = nullptr;
PFNGLGENERATEMIPMAPPROC COpenGLExtensionHandler::pGlGenerateMipmap = nullptr;
PFNGLGENERATETEXTUREMIPMAPPROC COpenGLExtensionHandler::pGlGenerateTextureMipmap = nullptr;
PFNGLGENERATETEXTUREMIPMAPEXTPROC COpenGLExtensionHandler::pGlGenerateTextureMipmapEXT = nullptr;
PFNGLCLAMPCOLORPROC COpenGLExtensionHandler::pGlClampColor = nullptr;

        //samplers
PFNGLGENSAMPLERSPROC COpenGLExtensionHandler::pGlGenSamplers = nullptr;
PFNGLCREATESAMPLERSPROC COpenGLExtensionHandler::pGlCreateSamplers = nullptr;
PFNGLDELETESAMPLERSPROC COpenGLExtensionHandler::pGlDeleteSamplers = nullptr;
PFNGLBINDSAMPLERPROC COpenGLExtensionHandler::pGlBindSampler = nullptr;
PFNGLBINDSAMPLERSPROC COpenGLExtensionHandler::pGlBindSamplers = nullptr;
PFNGLSAMPLERPARAMETERIPROC COpenGLExtensionHandler::pGlSamplerParameteri = nullptr;
PFNGLSAMPLERPARAMETERFPROC COpenGLExtensionHandler::pGlSamplerParameterf = nullptr;

//
PFNGLBINDIMAGETEXTUREPROC COpenGLExtensionHandler::pGlBindImageTexture = nullptr;

        //stuff
PFNGLBINDBUFFERBASEPROC COpenGLExtensionHandler::pGlBindBufferBase = nullptr;
PFNGLBINDBUFFERRANGEPROC COpenGLExtensionHandler::pGlBindBufferRange = nullptr;
PFNGLBINDBUFFERSBASEPROC COpenGLExtensionHandler::pGlBindBuffersBase = nullptr;
PFNGLBINDBUFFERSRANGEPROC COpenGLExtensionHandler::pGlBindBuffersRange = nullptr;


        //shaders
PFNGLBINDATTRIBLOCATIONPROC COpenGLExtensionHandler::pGlBindAttribLocation = nullptr;
PFNGLCREATEPROGRAMPROC COpenGLExtensionHandler::pGlCreateProgram = nullptr;
PFNGLUSEPROGRAMPROC COpenGLExtensionHandler::pGlUseProgram = nullptr;
PFNGLDELETEPROGRAMPROC COpenGLExtensionHandler::pGlDeleteProgram = nullptr;
PFNGLDELETESHADERPROC COpenGLExtensionHandler::pGlDeleteShader = nullptr;
PFNGLGETATTACHEDSHADERSPROC COpenGLExtensionHandler::pGlGetAttachedShaders = nullptr;
PFNGLCREATESHADERPROC COpenGLExtensionHandler::pGlCreateShader = nullptr;
PFNGLSHADERSOURCEPROC COpenGLExtensionHandler::pGlShaderSource = nullptr;
PFNGLCOMPILESHADERPROC COpenGLExtensionHandler::pGlCompileShader = nullptr;
PFNGLATTACHSHADERPROC COpenGLExtensionHandler::pGlAttachShader = nullptr;
PFNGLTRANSFORMFEEDBACKVARYINGSPROC COpenGLExtensionHandler::pGlTransformFeedbackVaryings = nullptr;
PFNGLLINKPROGRAMPROC COpenGLExtensionHandler::pGlLinkProgram = nullptr;
PFNGLGETSHADERINFOLOGPROC COpenGLExtensionHandler::pGlGetShaderInfoLog = nullptr;
PFNGLGETPROGRAMINFOLOGPROC COpenGLExtensionHandler::pGlGetProgramInfoLog = nullptr;
PFNGLGETSHADERIVPROC COpenGLExtensionHandler::pGlGetShaderiv = nullptr;
PFNGLGETSHADERIVPROC COpenGLExtensionHandler::pGlGetProgramiv = nullptr;
PFNGLGETUNIFORMLOCATIONPROC COpenGLExtensionHandler::pGlGetUniformLocation = nullptr;
//
PFNGLPROGRAMUNIFORM1FVPROC COpenGLExtensionHandler::pGlProgramUniform1fv = nullptr;
PFNGLPROGRAMUNIFORM2FVPROC COpenGLExtensionHandler::pGlProgramUniform2fv = nullptr;
PFNGLPROGRAMUNIFORM3FVPROC COpenGLExtensionHandler::pGlProgramUniform3fv = nullptr;
PFNGLPROGRAMUNIFORM4FVPROC COpenGLExtensionHandler::pGlProgramUniform4fv = nullptr;
PFNGLPROGRAMUNIFORM1IVPROC COpenGLExtensionHandler::pGlProgramUniform1iv = nullptr;
PFNGLPROGRAMUNIFORM2IVPROC COpenGLExtensionHandler::pGlProgramUniform2iv = nullptr;
PFNGLPROGRAMUNIFORM3IVPROC COpenGLExtensionHandler::pGlProgramUniform3iv = nullptr;
PFNGLPROGRAMUNIFORM4IVPROC COpenGLExtensionHandler::pGlProgramUniform4iv = nullptr;
PFNGLPROGRAMUNIFORM1UIVPROC COpenGLExtensionHandler::pGlProgramUniform1uiv = nullptr;
PFNGLPROGRAMUNIFORM2UIVPROC COpenGLExtensionHandler::pGlProgramUniform2uiv = nullptr;
PFNGLPROGRAMUNIFORM3UIVPROC COpenGLExtensionHandler::pGlProgramUniform3uiv = nullptr;
PFNGLPROGRAMUNIFORM4UIVPROC COpenGLExtensionHandler::pGlProgramUniform4uiv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX2FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix2fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX3FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix3fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX4FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix4fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix2x3fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix2x4fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix3x2fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix3x4fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix4x2fv = nullptr;
PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC COpenGLExtensionHandler::pGlProgramUniformMatrix4x3fv = nullptr;
//
PFNGLGETACTIVEUNIFORMPROC COpenGLExtensionHandler::pGlGetActiveUniform = nullptr;
PFNGLBINDPROGRAMPIPELINEPROC COpenGLExtensionHandler::pGlBindProgramPipeline = nullptr;

// Criss
PFNGLMEMORYBARRIERPROC COpenGLExtensionHandler::pGlMemoryBarrier = nullptr;
PFNGLDISPATCHCOMPUTEPROC COpenGLExtensionHandler::pGlDispatchCompute = nullptr;
PFNGLDISPATCHCOMPUTEINDIRECTPROC COpenGLExtensionHandler::pGlDispatchComputeIndirect = nullptr;
//
PFNGLPOINTPARAMETERFPROC COpenGLExtensionHandler:: pGlPointParameterf = nullptr;
PFNGLPOINTPARAMETERFVPROC COpenGLExtensionHandler::pGlPointParameterfv = nullptr;

//ROP
PFNGLBLENDCOLORPROC COpenGLExtensionHandler::pGlBlendColor = nullptr;
PFNGLDEPTHRANGEINDEXEDPROC COpenGLExtensionHandler::pGlDepthRangeIndexed = nullptr;
PFNGLVIEWPORTINDEXEDFVPROC COpenGLExtensionHandler::pGlViewportIndexedfv = nullptr;
PFNGLSCISSORINDEXEDVPROC COpenGLExtensionHandler::pGlScissorIndexedv = nullptr;
PFNGLSAMPLECOVERAGEPROC COpenGLExtensionHandler::pGlSampleCoverage = nullptr;
PFNGLSAMPLEMASKIPROC COpenGLExtensionHandler::pGlSampleMaski = nullptr;
PFNGLMINSAMPLESHADINGPROC COpenGLExtensionHandler::pGlMinSampleShading = nullptr;
PFNGLBLENDEQUATIONSEPARATEIPROC COpenGLExtensionHandler::pGlBlendEquationSeparatei = nullptr;
PFNGLBLENDFUNCSEPARATEIPROC COpenGLExtensionHandler::pGlBlendFuncSeparatei = nullptr;
PFNGLCOLORMASKIPROC COpenGLExtensionHandler::pGlColorMaski = nullptr;
PFNGLSTENCILFUNCSEPARATEPROC COpenGLExtensionHandler::pGlStencilFuncSeparate = nullptr;
PFNGLSTENCILOPSEPARATEPROC COpenGLExtensionHandler::pGlStencilOpSeparate = nullptr;
PFNGLSTENCILMASKSEPARATEPROC COpenGLExtensionHandler::pGlStencilMaskSeparate = nullptr;


		// ARB framebuffer object
PFNGLBLITNAMEDFRAMEBUFFERPROC COpenGLExtensionHandler::pGlBlitNamedFramebuffer = nullptr;
PFNGLBLITFRAMEBUFFERPROC COpenGLExtensionHandler::pGlBlitFramebuffer = nullptr;
PFNGLDELETEFRAMEBUFFERSPROC COpenGLExtensionHandler::pGlDeleteFramebuffers = nullptr;
PFNGLCREATEFRAMEBUFFERSPROC COpenGLExtensionHandler::pGlCreateFramebuffers = nullptr;
PFNGLGENFRAMEBUFFERSPROC COpenGLExtensionHandler::pGlGenFramebuffers = nullptr;
PFNGLBINDFRAMEBUFFERPROC COpenGLExtensionHandler::pGlBindFramebuffer = nullptr;
PFNGLCHECKFRAMEBUFFERSTATUSPROC COpenGLExtensionHandler::pGlCheckFramebufferStatus = nullptr;
PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC COpenGLExtensionHandler::pGlCheckNamedFramebufferStatus = nullptr;
PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC COpenGLExtensionHandler::pGlCheckNamedFramebufferStatusEXT = nullptr;
PFNGLFRAMEBUFFERTEXTUREPROC COpenGLExtensionHandler::pGlFramebufferTexture = nullptr;
PFNGLNAMEDFRAMEBUFFERTEXTUREPROC COpenGLExtensionHandler::pGlNamedFramebufferTexture = nullptr;
PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC COpenGLExtensionHandler::pGlNamedFramebufferTextureEXT = nullptr;
PFNGLFRAMEBUFFERTEXTURELAYERPROC COpenGLExtensionHandler::pGlFramebufferTextureLayer = nullptr;
PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC COpenGLExtensionHandler::pGlNamedFramebufferTextureLayer = nullptr;
PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC COpenGLExtensionHandler::pGlNamedFramebufferTextureLayerEXT = nullptr;
PFNGLFRAMEBUFFERTEXTURE2DPROC COpenGLExtensionHandler::pGlFramebufferTexture2D = nullptr;
PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC COpenGLExtensionHandler::pGlNamedFramebufferTexture2DEXT = nullptr;
		// EXT framebuffer object
PFNGLACTIVESTENCILFACEEXTPROC COpenGLExtensionHandler::pGlActiveStencilFaceEXT = nullptr;
PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC COpenGLExtensionHandler::pGlNamedFramebufferReadBuffer = nullptr;
PFNGLFRAMEBUFFERREADBUFFEREXTPROC COpenGLExtensionHandler::pGlFramebufferReadBufferEXT = nullptr;
PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC COpenGLExtensionHandler::pGlNamedFramebufferDrawBuffer = nullptr;
PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC COpenGLExtensionHandler::pGlFramebufferDrawBufferEXT = nullptr;
PFNGLDRAWBUFFERSPROC COpenGLExtensionHandler::pGlDrawBuffers = nullptr;
PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC COpenGLExtensionHandler::pGlNamedFramebufferDrawBuffers = nullptr;
PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC COpenGLExtensionHandler::pGlFramebufferDrawBuffersEXT = nullptr;
PFNGLCLEARNAMEDFRAMEBUFFERIVPROC COpenGLExtensionHandler::pGlClearNamedFramebufferiv = nullptr;
PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC COpenGLExtensionHandler::pGlClearNamedFramebufferuiv = nullptr;
PFNGLCLEARNAMEDFRAMEBUFFERFVPROC COpenGLExtensionHandler::pGlClearNamedFramebufferfv = nullptr;
PFNGLCLEARNAMEDFRAMEBUFFERFIPROC COpenGLExtensionHandler::pGlClearNamedFramebufferfi = nullptr;
PFNGLCLEARBUFFERIVPROC COpenGLExtensionHandler::pGlClearBufferiv = nullptr;
PFNGLCLEARBUFFERUIVPROC COpenGLExtensionHandler::pGlClearBufferuiv = nullptr;
PFNGLCLEARBUFFERFVPROC COpenGLExtensionHandler::pGlClearBufferfv = nullptr;
PFNGLCLEARBUFFERFIPROC COpenGLExtensionHandler::pGlClearBufferfi = nullptr;

//
PFNGLGENBUFFERSPROC COpenGLExtensionHandler::pGlGenBuffers = nullptr;
PFNGLCREATEBUFFERSPROC COpenGLExtensionHandler::pGlCreateBuffers = nullptr;
PFNGLBINDBUFFERPROC COpenGLExtensionHandler::pGlBindBuffer = nullptr;
PFNGLDELETEBUFFERSPROC COpenGLExtensionHandler::pGlDeleteBuffers = nullptr;
PFNGLBUFFERSTORAGEPROC COpenGLExtensionHandler::pGlBufferStorage = nullptr;
PFNGLNAMEDBUFFERSTORAGEPROC COpenGLExtensionHandler::pGlNamedBufferStorage = nullptr;
PFNGLNAMEDBUFFERSTORAGEEXTPROC COpenGLExtensionHandler::pGlNamedBufferStorageEXT = nullptr;
PFNGLBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlBufferSubData = nullptr;
PFNGLNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlNamedBufferSubData = nullptr;
PFNGLNAMEDBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlNamedBufferSubDataEXT = nullptr;
PFNGLGETBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlGetBufferSubData = nullptr;
PFNGLGETNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlGetNamedBufferSubData = nullptr;
PFNGLGETNAMEDBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlGetNamedBufferSubDataEXT = nullptr;
PFNGLMAPBUFFERPROC COpenGLExtensionHandler::pGlMapBuffer = nullptr;
PFNGLMAPNAMEDBUFFERPROC COpenGLExtensionHandler::pGlMapNamedBuffer = nullptr;
PFNGLMAPNAMEDBUFFEREXTPROC COpenGLExtensionHandler::pGlMapNamedBufferEXT = nullptr;
PFNGLMAPBUFFERRANGEPROC COpenGLExtensionHandler::pGlMapBufferRange = nullptr;
PFNGLMAPNAMEDBUFFERRANGEPROC COpenGLExtensionHandler::pGlMapNamedBufferRange = nullptr;
PFNGLMAPNAMEDBUFFERRANGEEXTPROC COpenGLExtensionHandler::pGlMapNamedBufferRangeEXT = nullptr;
PFNGLFLUSHMAPPEDBUFFERRANGEPROC COpenGLExtensionHandler::pGlFlushMappedBufferRange = nullptr;
PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC COpenGLExtensionHandler::pGlFlushMappedNamedBufferRange = nullptr;
PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC COpenGLExtensionHandler::pGlFlushMappedNamedBufferRangeEXT = nullptr;
PFNGLUNMAPBUFFERPROC COpenGLExtensionHandler::pGlUnmapBuffer = nullptr;
PFNGLUNMAPNAMEDBUFFERPROC COpenGLExtensionHandler::pGlUnmapNamedBuffer = nullptr;
PFNGLUNMAPNAMEDBUFFEREXTPROC COpenGLExtensionHandler::pGlUnmapNamedBufferEXT = nullptr;
PFNGLCLEARBUFFERDATAPROC COpenGLExtensionHandler::pGlClearBufferData = nullptr;
PFNGLCLEARNAMEDBUFFERDATAPROC COpenGLExtensionHandler::pGlClearNamedBufferData = nullptr;
PFNGLCLEARNAMEDBUFFERDATAEXTPROC COpenGLExtensionHandler::pGlClearNamedBufferDataEXT = nullptr;
PFNGLCLEARBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlClearBufferSubData = nullptr;
PFNGLCLEARNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlClearNamedBufferSubData = nullptr;
PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlClearNamedBufferSubDataEXT = nullptr;
PFNGLCOPYBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlCopyBufferSubData = nullptr;
PFNGLCOPYNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlCopyNamedBufferSubData = nullptr;
PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlNamedCopyBufferSubDataEXT = nullptr;
PFNGLISBUFFERPROC COpenGLExtensionHandler::pGlIsBuffer = nullptr;
PFNGLGETNAMEDBUFFERPARAMETERI64VPROC COpenGLExtensionHandler::pGlGetNamedBufferParameteri64v = nullptr;
PFNGLGETBUFFERPARAMETERI64VPROC COpenGLExtensionHandler::pGlGetBufferParameteri64v = nullptr;
PFNGLGETNAMEDBUFFERPARAMETERIVPROC COpenGLExtensionHandler::pGlGetNamedBufferParameteriv = nullptr;
PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC COpenGLExtensionHandler::pGlGetNamedBufferParameterivEXT = nullptr;
PFNGLGETBUFFERPARAMETERIVPROC COpenGLExtensionHandler::pGlGetBufferParameteriv = nullptr;
//vao
PFNGLGENVERTEXARRAYSPROC COpenGLExtensionHandler::pGlGenVertexArrays = nullptr;
PFNGLCREATEVERTEXARRAYSPROC COpenGLExtensionHandler::pGlCreateVertexArrays = nullptr;
PFNGLDELETEVERTEXARRAYSPROC COpenGLExtensionHandler::pGlDeleteVertexArrays = nullptr;
PFNGLBINDVERTEXARRAYPROC COpenGLExtensionHandler::pGlBindVertexArray = nullptr;
PFNGLVERTEXARRAYELEMENTBUFFERPROC COpenGLExtensionHandler::pGlVertexArrayElementBuffer = nullptr;
PFNGLBINDVERTEXBUFFERPROC COpenGLExtensionHandler::pGlBindVertexBuffer = nullptr;
PFNGLVERTEXARRAYVERTEXBUFFERPROC COpenGLExtensionHandler::pGlVertexArrayVertexBuffer = nullptr;
PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC COpenGLExtensionHandler::pGlVertexArrayBindVertexBufferEXT = nullptr;
PFNGLVERTEXATTRIBBINDINGPROC COpenGLExtensionHandler::pGlVertexAttribBinding = nullptr;
PFNGLVERTEXARRAYATTRIBBINDINGPROC COpenGLExtensionHandler::pGlVertexArrayAttribBinding = nullptr;
PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribBindingEXT = nullptr;
PFNGLENABLEVERTEXATTRIBARRAYPROC COpenGLExtensionHandler::pGlEnableVertexAttribArray = nullptr;
PFNGLENABLEVERTEXARRAYATTRIBPROC COpenGLExtensionHandler::pGlEnableVertexArrayAttrib = nullptr;
PFNGLENABLEVERTEXARRAYATTRIBEXTPROC COpenGLExtensionHandler::pGlEnableVertexArrayAttribEXT = nullptr;
PFNGLDISABLEVERTEXATTRIBARRAYPROC COpenGLExtensionHandler::pGlDisableVertexAttribArray = nullptr;
PFNGLDISABLEVERTEXARRAYATTRIBPROC COpenGLExtensionHandler::pGlDisableVertexArrayAttrib = nullptr;
PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC COpenGLExtensionHandler::pGlDisableVertexArrayAttribEXT = nullptr;
PFNGLVERTEXATTRIBFORMATPROC COpenGLExtensionHandler::pGlVertexAttribFormat = nullptr;
PFNGLVERTEXATTRIBIFORMATPROC COpenGLExtensionHandler::pGlVertexAttribIFormat = nullptr;
PFNGLVERTEXATTRIBLFORMATPROC COpenGLExtensionHandler::pGlVertexAttribLFormat = nullptr;
PFNGLVERTEXARRAYATTRIBFORMATPROC COpenGLExtensionHandler::pGlVertexArrayAttribFormat = nullptr;
PFNGLVERTEXARRAYATTRIBIFORMATPROC COpenGLExtensionHandler::pGlVertexArrayAttribIFormat = nullptr;
PFNGLVERTEXARRAYATTRIBLFORMATPROC COpenGLExtensionHandler::pGlVertexArrayAttribLFormat = nullptr;
PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribFormatEXT = nullptr;
PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribIFormatEXT = nullptr;
PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribLFormatEXT = nullptr;
PFNGLVERTEXARRAYBINDINGDIVISORPROC COpenGLExtensionHandler::pGlVertexArrayBindingDivisor = nullptr;
PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexBindingDivisorEXT = nullptr;
PFNGLVERTEXBINDINGDIVISORPROC COpenGLExtensionHandler::pGlVertexBindingDivisor = nullptr;
//
PFNGLPRIMITIVERESTARTINDEXPROC COpenGLExtensionHandler::pGlPrimitiveRestartIndex = nullptr;
PFNGLDRAWARRAYSINSTANCEDPROC COpenGLExtensionHandler::pGlDrawArraysInstanced = nullptr;
PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC COpenGLExtensionHandler::pGlDrawArraysInstancedBaseInstance = nullptr;
PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC COpenGLExtensionHandler::pGlDrawElementsInstancedBaseVertex = nullptr;
PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC COpenGLExtensionHandler::pGlDrawElementsInstancedBaseVertexBaseInstance = nullptr;
PFNGLDRAWTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlDrawTransformFeedback = nullptr;
PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC COpenGLExtensionHandler::pGlDrawTransformFeedbackInstanced = nullptr;
PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC COpenGLExtensionHandler::pGlDrawTransformFeedbackStream = nullptr;
PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC COpenGLExtensionHandler::pGlDrawTransformFeedbackStreamInstanced = nullptr;
PFNGLDRAWARRAYSINDIRECTPROC COpenGLExtensionHandler::pGlDrawArraysIndirect = nullptr;
PFNGLDRAWELEMENTSINDIRECTPROC COpenGLExtensionHandler::pGlDrawElementsIndirect = nullptr;
PFNGLMULTIDRAWARRAYSINDIRECTPROC COpenGLExtensionHandler::pGlMultiDrawArraysIndirect = nullptr;
PFNGLMULTIDRAWELEMENTSINDIRECTPROC COpenGLExtensionHandler::pGlMultiDrawElementsIndirect = nullptr;
//
PFNGLCREATETRANSFORMFEEDBACKSPROC COpenGLExtensionHandler::pGlCreateTransformFeedbacks = nullptr;
PFNGLGENTRANSFORMFEEDBACKSPROC COpenGLExtensionHandler::pGlGenTransformFeedbacks = nullptr;
PFNGLDELETETRANSFORMFEEDBACKSPROC COpenGLExtensionHandler::pGlDeleteTransformFeedbacks = nullptr;
PFNGLBINDTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlBindTransformFeedback = nullptr;
PFNGLBEGINTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlBeginTransformFeedback = nullptr;
PFNGLPAUSETRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlPauseTransformFeedback = nullptr;
PFNGLRESUMETRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlResumeTransformFeedback = nullptr;
PFNGLENDTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlEndTransformFeedback = nullptr;
PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC COpenGLExtensionHandler::pGlTransformFeedbackBufferBase = nullptr;
PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC COpenGLExtensionHandler::pGlTransformFeedbackBufferRange = nullptr;
//
PFNGLBLENDFUNCSEPARATEPROC COpenGLExtensionHandler::pGlBlendFuncSeparate = nullptr;
PFNGLBLENDFUNCINDEXEDAMDPROC COpenGLExtensionHandler::pGlBlendFuncIndexedAMD = nullptr;
PFNGLBLENDFUNCIPROC COpenGLExtensionHandler::pGlBlendFunciARB = nullptr;
PFNGLBLENDEQUATIONINDEXEDAMDPROC COpenGLExtensionHandler::pGlBlendEquationIndexedAMD = nullptr;
PFNGLBLENDEQUATIONIPROC COpenGLExtensionHandler::pGlBlendEquationiARB = nullptr;
PFNGLPROGRAMPARAMETERIPROC COpenGLExtensionHandler::pGlProgramParameteri = nullptr;
PFNGLPATCHPARAMETERIPROC COpenGLExtensionHandler::pGlPatchParameteri = nullptr;
PFNGLPATCHPARAMETERFVPROC COpenGLExtensionHandler::pGlPatchParameterfv = nullptr;
//
PFNGLCREATEQUERIESPROC COpenGLExtensionHandler::pGlCreateQueries = nullptr;
PFNGLGENQUERIESPROC COpenGLExtensionHandler::pGlGenQueries = nullptr;
PFNGLDELETEQUERIESPROC COpenGLExtensionHandler::pGlDeleteQueries = nullptr;
PFNGLISQUERYPROC COpenGLExtensionHandler::pGlIsQuery = nullptr;
PFNGLBEGINQUERYPROC COpenGLExtensionHandler::pGlBeginQuery = nullptr;
PFNGLENDQUERYPROC COpenGLExtensionHandler::pGlEndQuery = nullptr;
PFNGLBEGINQUERYINDEXEDPROC COpenGLExtensionHandler::pGlBeginQueryIndexed = nullptr;
PFNGLENDQUERYINDEXEDPROC COpenGLExtensionHandler::pGlEndQueryIndexed = nullptr;
PFNGLGETQUERYIVPROC COpenGLExtensionHandler::pGlGetQueryiv = nullptr;
PFNGLGETQUERYOBJECTUIVPROC COpenGLExtensionHandler::pGlGetQueryObjectuiv = nullptr;
PFNGLGETQUERYOBJECTUI64VPROC COpenGLExtensionHandler::pGlGetQueryObjectui64v = nullptr;
PFNGLGETQUERYBUFFEROBJECTUIVPROC COpenGLExtensionHandler::pGlGetQueryBufferObjectuiv = nullptr;
PFNGLGETQUERYBUFFEROBJECTUI64VPROC COpenGLExtensionHandler::pGlGetQueryBufferObjectui64v = nullptr;
PFNGLQUERYCOUNTERPROC COpenGLExtensionHandler::pGlQueryCounter = nullptr;
PFNGLBEGINCONDITIONALRENDERPROC COpenGLExtensionHandler::pGlBeginConditionalRender = nullptr;
PFNGLENDCONDITIONALRENDERPROC COpenGLExtensionHandler::pGlEndConditionalRender = nullptr;
//
PFNGLTEXTUREBARRIERPROC COpenGLExtensionHandler::pGlTextureBarrier = nullptr;
PFNGLTEXTUREBARRIERNVPROC COpenGLExtensionHandler::pGlTextureBarrierNV = nullptr;
//
PFNGLBLENDEQUATIONEXTPROC COpenGLExtensionHandler::pGlBlendEquationEXT = nullptr;
PFNGLBLENDEQUATIONPROC COpenGLExtensionHandler::pGlBlendEquation = nullptr;

PFNGLGETINTERNALFORMATIVPROC COpenGLExtensionHandler::pGlGetInternalformativ = nullptr;
PFNGLGETINTERNALFORMATI64VPROC COpenGLExtensionHandler::pGlGetInternalformati64v = nullptr;

PFNGLDEBUGMESSAGECONTROLPROC COpenGLExtensionHandler::pGlDebugMessageControl = nullptr;
PFNGLDEBUGMESSAGECONTROLARBPROC COpenGLExtensionHandler::pGlDebugMessageControlARB = nullptr;
PFNGLDEBUGMESSAGECALLBACKPROC COpenGLExtensionHandler::pGlDebugMessageCallback = nullptr;
PFNGLDEBUGMESSAGECALLBACKARBPROC COpenGLExtensionHandler::pGlDebugMessageCallbackARB = nullptr;

    #if defined(WGL_EXT_swap_control)
        PFNWGLSWAPINTERVALEXTPROC COpenGLExtensionHandler::pWglSwapIntervalEXT = nullptr;
    #endif
    #if defined(GLX_SGI_swap_control)
        PFNGLXSWAPINTERVALSGIPROC COpenGLExtensionHandler::pGlxSwapIntervalSGI = nullptr;
    #endif
    #if defined(GLX_EXT_swap_control)
        PFNGLXSWAPINTERVALEXTPROC COpenGLExtensionHandler::pGlxSwapIntervalEXT = nullptr;
    #endif
    #if defined(GLX_MESA_swap_control)
        PFNGLXSWAPINTERVALMESAPROC COpenGLExtensionHandler::pGlxSwapIntervalMESA = nullptr;
    #endif


core::LeakDebugger COpenGLExtensionHandler::bufferLeaker("GLBuffer");
core::LeakDebugger COpenGLExtensionHandler::textureLeaker("GLTex");



COpenGLExtensionHandler::COpenGLExtensionHandler() :
		StencilBuffer(false),
		TextureCompressionExtension(false)
{
	DimAliasedLine[0]=1.f;
	DimAliasedLine[1]=1.f;
	DimAliasedPoint[0]=1.f;
	DimAliasedPoint[1]=1.f;
	DimSmoothedLine[0]=1.f;
	DimSmoothedLine[1]=1.f;
	DimSmoothedPoint[0]=1.f;
	DimSmoothedPoint[1]=1.f;
}


void COpenGLExtensionHandler::dump(std::string* outStr, bool onlyAvailable) const
{
    if (onlyAvailable)
    {
        for (uint32_t i=0; i<IRR_OpenGL_Feature_Count; ++i)
        {
            if (FeatureAvailable[i])
            {
                if (outStr)
                {
                    (*outStr) += OpenGLFeatureStrings[i];
                    (*outStr) += "\n";
                }
                else
                    os::Printer::log(OpenGLFeatureStrings[i]);
            }
        }
    }
    else
    {
        for (uint32_t i=0; i<IRR_OpenGL_Feature_Count; ++i)
        {
            if (outStr)
            {
                (*outStr) += OpenGLFeatureStrings[i];
                (*outStr) += FeatureAvailable[i] ? " true\n":" false\n";
            }
            else
                os::Printer::log(OpenGLFeatureStrings[i], FeatureAvailable[i]?" true":" false");
        }
    }
}


void COpenGLExtensionHandler::dumpFramebufferFormats() const
{
#ifdef _IRR_WINDOWS_API_
	HDC hdc=wglGetCurrentDC();
	std::string wglExtensions;
#ifdef WGL_ARB_extensions_string
	PFNWGLGETEXTENSIONSSTRINGARBPROC irrGetExtensionsString = (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
	if (irrGetExtensionsString)
		wglExtensions = irrGetExtensionsString(hdc);
#elif defined(WGL_EXT_extensions_string)
	PFNWGLGETEXTENSIONSSTRINGEXTPROC irrGetExtensionsString = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");
	if (irrGetExtensionsString)
		wglExtensions = irrGetExtensionsString(hdc);
#endif
	const bool pixel_format_supported = (wglExtensions.find("WGL_ARB_pixel_format") != std::string::npos);
	const bool multi_sample_supported = ((wglExtensions.find("WGL_ARB_multisample") != std::string::npos) ||
		(wglExtensions.find("WGL_EXT_multisample") != std::string::npos) || (wglExtensions.find("WGL_3DFX_multisample") != std::string::npos) );
#ifdef _IRR_DEBUG
	os::Printer::log("WGL_extensions", wglExtensions);
#endif

#ifdef WGL_ARB_pixel_format
	PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormat_ARB = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
	if (pixel_format_supported && wglChoosePixelFormat_ARB)
	{
		// This value determines the number of samples used for antialiasing
		// My experience is that 8 does not show a big
		// improvement over 4, but 4 shows a big improvement
		// over 2.

		PFNWGLGETPIXELFORMATATTRIBIVARBPROC wglGetPixelFormatAttribiv_ARB = (PFNWGLGETPIXELFORMATATTRIBIVARBPROC)wglGetProcAddress("wglGetPixelFormatAttribivARB");
		if (wglGetPixelFormatAttribiv_ARB)
		{
			int vals[128];
			int atts[] = {
				WGL_NUMBER_PIXEL_FORMATS_ARB,
				WGL_DRAW_TO_BITMAP_ARB,
				WGL_ACCELERATION_ARB,
				WGL_NEED_PALETTE_ARB,
				WGL_NEED_SYSTEM_PALETTE_ARB,
				WGL_SWAP_LAYER_BUFFERS_ARB,
				WGL_SWAP_METHOD_ARB,
				WGL_NUMBER_OVERLAYS_ARB,
				WGL_NUMBER_UNDERLAYS_ARB,
				WGL_TRANSPARENT_ARB,
				WGL_TRANSPARENT_RED_VALUE_ARB,
				WGL_TRANSPARENT_GREEN_VALUE_ARB,
				WGL_TRANSPARENT_BLUE_VALUE_ARB,
				WGL_TRANSPARENT_ALPHA_VALUE_ARB,
				WGL_TRANSPARENT_INDEX_VALUE_ARB,
				WGL_SHARE_DEPTH_ARB,
				WGL_SHARE_STENCIL_ARB,
				WGL_SHARE_ACCUM_ARB,
				WGL_SUPPORT_GDI_ARB,
				WGL_SUPPORT_OPENGL_ARB,
				WGL_DOUBLE_BUFFER_ARB,
				WGL_STEREO_ARB,
				WGL_PIXEL_TYPE_ARB,
				WGL_COLOR_BITS_ARB,
				WGL_RED_BITS_ARB,
				WGL_RED_SHIFT_ARB,
				WGL_GREEN_BITS_ARB,
				WGL_GREEN_SHIFT_ARB,
				WGL_BLUE_BITS_ARB,
				WGL_BLUE_SHIFT_ARB,
				WGL_ALPHA_BITS_ARB,
				WGL_ALPHA_SHIFT_ARB,
				WGL_ACCUM_BITS_ARB,
				WGL_ACCUM_RED_BITS_ARB,
				WGL_ACCUM_GREEN_BITS_ARB,
				WGL_ACCUM_BLUE_BITS_ARB,
				WGL_ACCUM_ALPHA_BITS_ARB,
				WGL_DEPTH_BITS_ARB,
				WGL_STENCIL_BITS_ARB,
				WGL_AUX_BUFFERS_ARB
#ifdef WGL_ARB_render_texture
				,WGL_BIND_TO_TEXTURE_RGB_ARB //40
				,WGL_BIND_TO_TEXTURE_RGBA_ARB
#endif
#ifdef WGL_ARB_pbuffer
				,WGL_DRAW_TO_PBUFFER_ARB //42
				,WGL_MAX_PBUFFER_PIXELS_ARB
				,WGL_MAX_PBUFFER_WIDTH_ARB
				,WGL_MAX_PBUFFER_HEIGHT_ARB
#endif
#ifdef WGL_ARB_framebuffer_sRGB
				,WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB //46
#endif
#ifdef WGL_ARB_multisample
				,WGL_SAMPLES_ARB //47
				,WGL_SAMPLE_BUFFERS_ARB
#endif
#ifdef WGL_EXT_depth_float
				,WGL_DEPTH_FLOAT_EXT //49
#endif
				,0,0,0,0
			};
			size_t nums = sizeof(atts)/sizeof(int);
			const bool depth_float_supported= (wglExtensions.find("WGL_EXT_depth_float") != std::string::npos);
			if (!depth_float_supported)
			{
				memmove(&atts[49], &atts[50], (nums-50)*sizeof(int));
				nums -= 1;
			}
			if (!multi_sample_supported)
			{
				memmove(&atts[47], &atts[49], (nums-49)*sizeof(int));
				nums -= 2;
			}
			const bool framebuffer_sRGB_supported= (wglExtensions.find("WGL_ARB_framebuffer_sRGB") != std::string::npos);
			if (!framebuffer_sRGB_supported)
			{
				memmove(&atts[46], &atts[47], (nums-47)*sizeof(int));
				nums -= 1;
			}
			const bool pbuffer_supported = (wglExtensions.find("WGL_ARB_pbuffer") != std::string::npos);
			if (!pbuffer_supported)
			{
				memmove(&atts[42], &atts[46], (nums-46)*sizeof(int));
				nums -= 4;
			}
			const bool render_texture_supported = (wglExtensions.find("WGL_ARB_render_texture") != std::string::npos);
			if (!render_texture_supported)
			{
				memmove(&atts[40], &atts[42], (nums-42)*sizeof(int));
				nums -= 2;
			}
			wglGetPixelFormatAttribiv_ARB(hdc,0,0,1,atts,vals);
			const int count = vals[0];
			atts[0]=WGL_DRAW_TO_WINDOW_ARB;
			for (int i=1; i<count; ++i)
			{
				memset(vals,0,sizeof(vals));
#define tmplog(x,y) os::Printer::log(x, std::to_string(y))
				const BOOL res = wglGetPixelFormatAttribiv_ARB(hdc,i,0,nums,atts,vals);
				if (FALSE==res)
					continue;
				tmplog("Pixel format ",i);
				uint32_t j=0;
				tmplog("Draw to window " , vals[j]);
				tmplog("Draw to bitmap " , vals[++j]);
				++j;
				os::Printer::log("Acceleration " , (vals[j]==WGL_NO_ACCELERATION_ARB?"No":
					vals[j]==WGL_GENERIC_ACCELERATION_ARB?"Generic":vals[j]==WGL_FULL_ACCELERATION_ARB?"Full":"ERROR"));
				tmplog("Need palette " , vals[++j]);
				tmplog("Need system palette " , vals[++j]);
				tmplog("Swap layer buffers " , vals[++j]);
				++j;
				os::Printer::log("Swap method " , (vals[j]==WGL_SWAP_EXCHANGE_ARB?"Exchange":
					vals[j]==WGL_SWAP_COPY_ARB?"Copy":vals[j]==WGL_SWAP_UNDEFINED_ARB?"Undefined":"ERROR"));
				tmplog("Number of overlays " , vals[++j]);
				tmplog("Number of underlays " , vals[++j]);
				tmplog("Transparent " , vals[++j]);
				tmplog("Transparent red value " , vals[++j]);
				tmplog("Transparent green value " , vals[++j]);
				tmplog("Transparent blue value " , vals[++j]);
				tmplog("Transparent alpha value " , vals[++j]);
				tmplog("Transparent index value " , vals[++j]);
				tmplog("Share depth " , vals[++j]);
				tmplog("Share stencil " , vals[++j]);
				tmplog("Share accum " , vals[++j]);
				tmplog("Support GDI " , vals[++j]);
				tmplog("Support OpenGL " , vals[++j]);
				tmplog("Double Buffer " , vals[++j]);
				tmplog("Stereo Buffer " , vals[++j]);
				tmplog("Pixel type " , vals[++j]);
				tmplog("Color bits" , vals[++j]);
				tmplog("Red bits " , vals[++j]);
				tmplog("Red shift " , vals[++j]);
				tmplog("Green bits " , vals[++j]);
				tmplog("Green shift " , vals[++j]);
				tmplog("Blue bits " , vals[++j]);
				tmplog("Blue shift " , vals[++j]);
				tmplog("Alpha bits " , vals[++j]);
				tmplog("Alpha Shift " , vals[++j]);
				tmplog("Accum bits " , vals[++j]);
				tmplog("Accum red bits " , vals[++j]);
				tmplog("Accum green bits " , vals[++j]);
				tmplog("Accum blue bits " , vals[++j]);
				tmplog("Accum alpha bits " , vals[++j]);
				tmplog("Depth bits " , vals[++j]);
				tmplog("Stencil bits " , vals[++j]);
				tmplog("Aux buffers " , vals[++j]);
				if (render_texture_supported)
				{
					tmplog("Bind to texture RGB" , vals[++j]);
					tmplog("Bind to texture RGBA" , vals[++j]);
				}
				if (pbuffer_supported)
				{
					tmplog("Draw to pbuffer" , vals[++j]);
					tmplog("Max pbuffer pixels " , vals[++j]);
					tmplog("Max pbuffer width" , vals[++j]);
					tmplog("Max pbuffer height" , vals[++j]);
				}
				if (framebuffer_sRGB_supported)
					tmplog("Framebuffer sRBG capable" , vals[++j]);
				if (multi_sample_supported)
				{
					tmplog("Samples " , vals[++j]);
					tmplog("Sample buffers " , vals[++j]);
				}
				if (depth_float_supported)
					tmplog("Depth float" , vals[++j]);
#undef tmplog
			}
		}
	}
#endif
#elif defined(IRR_LINUX_DEVICE)
#endif
}


void COpenGLExtensionHandler::initExtensions(bool stencilBuffer)
{
    core::stringc vendorString = (char*)glGetString(GL_VENDOR);
    if (vendorString.find("Intel")!=-1 || vendorString.find("INTEL")!=-1)
	    IsIntelGPU = true;


	loadFunctions();


	TextureCompressionExtension = FeatureAvailable[IRR_ARB_texture_compression];
	StencilBuffer = stencilBuffer;


	GLint num = 0;

	glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &reqUBOAlignment);
	glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &reqSSBOAlignment);
	glGetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &reqTBOAlignment);

    extGlGetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, reinterpret_cast<GLint64*>(&maxUBOSize));
    extGlGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, reinterpret_cast<GLint64*>(&maxSSBOSize));
    extGlGetInteger64v(GL_MAX_TEXTURE_BUFFER_SIZE, reinterpret_cast<GLint64*>(&maxTBOSize));
    maxBufferSize = std::max(maxUBOSize, std::max(maxSSBOSize, maxTBOSize));

	glGetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, &minMemoryMapAlignment);

    extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, MaxComputeWGSize);
    extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, MaxComputeWGSize+1);
    extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, MaxComputeWGSize+2);


	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &num);
	MaxArrayTextureLayers = num;

	if (FeatureAvailable[IRR_EXT_texture_filter_anisotropic])
	{
		glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &num);
		MaxAnisotropy = static_cast<uint8_t>(num);
	}


    if (FeatureAvailable[IRR_ARB_geometry_shader4])
    {
        glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &num);
        MaxGeometryVerticesOut = static_cast<uint32_t>(num);
    }

	if (FeatureAvailable[IRR_EXT_texture_lod_bias])
		glGetFloatv(GL_MAX_TEXTURE_LOD_BIAS_EXT, &MaxTextureLODBias);


	glGetIntegerv(GL_MAX_CLIP_DISTANCES, &num);
	MaxUserClipPlanes=static_cast<uint8_t>(num);
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &num);
    MaxMultipleRenderTargets = static_cast<uint8_t>(num);

	glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, DimAliasedLine);
	glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE, DimAliasedPoint);
	glGetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE, DimSmoothedLine);
	glGetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, DimSmoothedPoint);

    const GLubyte* shaderVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    float sl_ver;
    sscanf(reinterpret_cast<const char*>(shaderVersion),"%f",&sl_ver);
    ShaderLanguageVersion = static_cast<uint16_t>(core::round32(sl_ver*100.0f));


	//! For EXT-DSA testing
	if (IsIntelGPU)
	{
		Version = 440;
		FeatureAvailable[IRR_ARB_direct_state_access] = false;
		pGlCreateTextures = nullptr;
		pGlTextureStorage1D = nullptr;
		pGlTextureStorage2D = nullptr;
		pGlTextureStorage3D = nullptr;
		pGlTextureStorage2DMultisample = nullptr;
		pGlTextureStorage3DMultisample = nullptr;
		pGlTextureSubImage1D = nullptr;
		pGlTextureSubImage2D = nullptr;
		pGlTextureSubImage3D = nullptr;
		pGlCompressedTextureSubImage1D = nullptr;
		pGlCompressedTextureSubImage2D = nullptr;
		pGlCompressedTextureSubImage3D = nullptr;
		pGlCopyTextureSubImage1D = nullptr;
		pGlCopyTextureSubImage2D = nullptr;
		pGlCopyTextureSubImage3D = nullptr;
		pGlGenerateTextureMipmap = nullptr;
		pGlCreateSamplers = nullptr;
		pGlBindAttribLocation = nullptr;
		pGlBlitNamedFramebuffer = nullptr;
		pGlCreateFramebuffers = nullptr;
		pGlCheckNamedFramebufferStatus = nullptr;
		pGlNamedFramebufferTexture = nullptr;
		pGlNamedFramebufferTextureLayer = nullptr;
		pGlActiveStencilFaceEXT = nullptr;
		pGlNamedFramebufferReadBuffer = nullptr;
		pGlNamedFramebufferDrawBuffer = nullptr;
		pGlNamedFramebufferDrawBuffers = nullptr;
		pGlClearNamedFramebufferiv = nullptr;
		pGlClearNamedFramebufferuiv = nullptr;
		pGlClearNamedFramebufferfv = nullptr;
		pGlClearNamedFramebufferfi = nullptr;
		pGlCreateBuffers = nullptr;
		pGlNamedBufferStorage = nullptr;
		pGlNamedBufferSubData = nullptr;
		pGlGetNamedBufferSubData = nullptr;
		pGlMapNamedBuffer = nullptr;
		pGlMapNamedBufferRange = nullptr;
		pGlFlushMappedNamedBufferRange = nullptr;
		pGlUnmapNamedBuffer = nullptr;
		pGlClearNamedBufferData = nullptr;
		pGlClearNamedBufferSubData = nullptr;
		pGlCopyNamedBufferSubData = nullptr;
		pGlCreateVertexArrays = nullptr;
		pGlVertexArrayElementBuffer = nullptr;
		pGlVertexArrayVertexBuffer = nullptr;
		pGlVertexArrayAttribBinding = nullptr;
		pGlEnableVertexArrayAttrib = nullptr;
		pGlDisableVertexArrayAttrib = nullptr;
		pGlVertexArrayAttribFormat = nullptr;
		pGlVertexArrayAttribIFormat = nullptr;
		pGlVertexArrayAttribLFormat = nullptr;
		pGlVertexArrayBindingDivisor = nullptr;
		pGlBlendFuncIndexedAMD = nullptr;
		pGlBlendEquationIndexedAMD = nullptr;
		pGlBlendEquationiARB = nullptr;
		pGlCreateQueries = nullptr;
	}/*
    //! Non-DSA testing
    Version = 430;
    FeatureAvailable[IRR_EXT_direct_state_access] = FeatureAvailable[IRR_ARB_direct_state_access] = false;
    pGlTextureStorage1DEXT = nullptr;
    pGlTextureStorage2DEXT = nullptr;
    pGlTextureStorage3DEXT = nullptr;
    pGlTextureStorage2DMultisampleEXT = nullptr;
    pGlTextureStorage3DMultisampleEXT = nullptr;
    pGlTextureSubImage1DEXT = nullptr;
    pGlTextureSubImage2DEXT = nullptr;
    pGlTextureSubImage3DEXT = nullptr;
    pGlCompressedTextureSubImage1DEXT = nullptr;
    pGlCompressedTextureSubImage2DEXT = nullptr;
    pGlCompressedTextureSubImage3DEXT = nullptr;
    pGlCopyTextureSubImage1DEXT = nullptr;
    pGlCopyTextureSubImage2DEXT = nullptr;
    pGlCopyTextureSubImage3DEXT = nullptr;
    pGlGenerateTextureMipmapEXT = nullptr;
    pGlCheckNamedFramebufferStatusEXT = nullptr;
    pGlNamedFramebufferTextureEXT = nullptr;
    pGlNamedFramebufferTextureLayerEXT = nullptr;
    pGlFramebufferReadBufferEXT = nullptr;
    pGlFramebufferDrawBufferEXT = nullptr;
    pGlFramebufferDrawBuffersEXT = nullptr;
    pGlNamedBufferStorageEXT = nullptr;
    pGlNamedBufferSubDataEXT = nullptr;
    pGlGetNamedBufferSubDataEXT = nullptr;
    pGlMapNamedBufferEXT = nullptr;
    pGlMapNamedBufferRangeEXT = nullptr;
    pGlFlushMappedNamedBufferRangeEXT = nullptr;
    pGlUnmapNamedBufferEXT = nullptr;
    pGlClearNamedBufferDataEXT = nullptr;
    pGlClearNamedBufferSubDataEXT = nullptr;
    pGlNamedCopyBufferSubDataEXT = nullptr;
    pGlVertexArrayBindVertexBufferEXT = nullptr;
    pGlVertexArrayVertexAttribBindingEXT = nullptr;
    pGlEnableVertexArrayAttribEXT = nullptr;
    pGlDisableVertexArrayAttribEXT = nullptr;
    pGlVertexArrayVertexAttribFormatEXT = nullptr;
    pGlVertexArrayVertexAttribIFormatEXT = nullptr;
    pGlVertexArrayVertexAttribLFormatEXT = nullptr;
    pGlVertexArrayVertexBindingDivisorEXT = nullptr;
    pGlCreateQueries = nullptr;*/


    num=0;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &num);
	MaxTextureUnits = core::min_(static_cast<uint8_t>(num), static_cast<uint8_t>(MATERIAL_MAX_TEXTURES));

    //num=100000000u;
	//glGetIntegerv(GL_MAX_ELEMENTS_INDICES,&num);
#ifdef WIN32
#ifdef _IRR_DEBUG
	if (FeatureAvailable[IRR_NVX_gpu_memory_info])
	{
		// undocumented flags, so use the RAW values
		GLint val;
		glGetIntegerv(0x9047, &val);
		os::Printer::log("Dedicated video memory (kB)", std::to_string(val));
		glGetIntegerv(0x9048, &val);
		os::Printer::log("Total video memory (kB)", std::to_string(val));
		glGetIntegerv(0x9049, &val);
		os::Printer::log("Available video memory (kB)", std::to_string(val));
	}
	if (FeatureAvailable[IRR_ATI_meminfo])
	{
		GLint val[4];
		glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, val);
		os::Printer::log("Free texture memory (kB)", std::to_string(val[0]));
		glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, val);
		os::Printer::log("Free VBO memory (kB)", std::to_string(val[0]));
		glGetIntegerv(GL_RENDERBUFFER_FREE_MEMORY_ATI, val);
		os::Printer::log("Free render buffer memory (kB)", std::to_string(val[0]));
	}
#endif
#endif
}

void COpenGLExtensionHandler::loadFunctions()
{
    if (functionsAlreadyLoaded)
        return;

	for (uint32_t i=0; i<IRR_OpenGL_Feature_Count; ++i)
		FeatureAvailable[i]=false;



#ifdef _IRR_WINDOWS_API_
	#define IRR_OGL_LOAD_EXTENSION(x) wglGetProcAddress(reinterpret_cast<const char*>(x))
#elif defined(_IRR_COMPILE_WITH_SDL_DEVICE_) && !defined(_IRR_COMPILE_WITH_X11_DEVICE_)
	#define IRR_OGL_LOAD_EXTENSION(x) SDL_GL_GetProcAddress(reinterpret_cast<const char*>(x))
#else
    #define IRR_OGL_LOAD_EXTENSION(X) glXGetProcAddress(reinterpret_cast<const GLubyte*>(X))
#endif // Windows, SDL, or Linux

    pGlIsEnabledi = (PFNGLISENABLEDIPROC) IRR_OGL_LOAD_EXTENSION("glIsEnabledi");
    pGlEnablei = (PFNGLENABLEIPROC) IRR_OGL_LOAD_EXTENSION("glEnablei");
    pGlDisablei = (PFNGLDISABLEIPROC) IRR_OGL_LOAD_EXTENSION("glDisablei");
    pGlGetBooleani_v = (PFNGLGETBOOLEANI_VPROC) IRR_OGL_LOAD_EXTENSION("glGetBooleani_v");
    pGlGetFloati_v = (PFNGLGETFLOATI_VPROC) IRR_OGL_LOAD_EXTENSION("glGetFloati_v");
    pGlGetInteger64v = (PFNGLGETINTEGER64VPROC)IRR_OGL_LOAD_EXTENSION("glGetInteger64v");
    pGlGetIntegeri_v = (PFNGLGETINTEGERI_VPROC) IRR_OGL_LOAD_EXTENSION("glGetIntegeri_v");
    pGlGetStringi = (PFNGLGETSTRINGIPROC) IRR_OGL_LOAD_EXTENSION("glGetStringi");

    GLint extensionCount;
    glGetIntegerv(GL_NUM_EXTENSIONS,&extensionCount);
    for (GLint i=0; i<extensionCount; ++i)
    {
        const char* extensionName = reinterpret_cast<const char*>(pGlGetStringi(GL_EXTENSIONS,i));

        for (uint32_t j=0; j<IRR_OpenGL_Feature_Count; ++j)
        {
            if (!strcmp(OpenGLFeatureStrings[j], extensionName))
            {
                FeatureAvailable[j] = true;
                break;
            }
        }
    }


	float ogl_ver;
	sscanf(reinterpret_cast<const char*>(glGetString(GL_VERSION)),"%f",&ogl_ver);
	Version = static_cast<uint16_t>(core::round32(ogl_ver*100.0f));

	GLint num=0;
	glGetIntegerv(GL_MAX_ELEMENTS_INDICES, &num);
    MaxIndices=num;
	glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &num);
    MaxVertices=num;
	glGetIntegerv(GL_MAX_VERTEX_STREAMS, &num);
    MaxVertexStreams=num;
	glGetIntegerv(GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS, &num);
    MaxXFormFeedbackComponents = num;
	glGetIntegerv(GL_MAX_SERVER_WAIT_TIMEOUT, &num);
    MaxGPUWaitTimeout = reinterpret_cast<const uint32_t&>(num);

    if (FeatureAvailable[IRR_NV_shader_thread_group])
    {
        glGetIntegerv(GL_WARP_SIZE_NV, &num);
        InvocationSubGroupSize[0] = InvocationSubGroupSize[1] = reinterpret_cast<const uint32_t&>(num);
    }
    else if (IsIntelGPU)
    {
        InvocationSubGroupSize[0] = 4;
        InvocationSubGroupSize[1] = 32;
    }

    /**
    pGl = () IRR_OGL_LOAD_EXTENSION("gl");
    **/
    pGlProvokingVertex = (PFNGLPROVOKINGVERTEXPROC) IRR_OGL_LOAD_EXTENSION("glProvokingVertex");
    pGlClipControl = (PFNGLCLIPCONTROLPROC) IRR_OGL_LOAD_EXTENSION("glClipControl");

    //fences
    pGlFenceSync = (PFNGLFENCESYNCPROC) IRR_OGL_LOAD_EXTENSION("glFenceSync");
    pGlDeleteSync = (PFNGLDELETESYNCPROC) IRR_OGL_LOAD_EXTENSION("glDeleteSync");
    pGlClientWaitSync = (PFNGLCLIENTWAITSYNCPROC) IRR_OGL_LOAD_EXTENSION("glClientWaitSync");
    pGlWaitSync = (PFNGLWAITSYNCPROC) IRR_OGL_LOAD_EXTENSION("glWaitSync");

	// get multitexturing function pointers
    pGlActiveTexture = (PFNGLACTIVETEXTUREPROC) IRR_OGL_LOAD_EXTENSION("glActiveTexture");
	pGlBindTextures = (PFNGLBINDTEXTURESPROC) IRR_OGL_LOAD_EXTENSION("glBindTextures");
    pGlCreateTextures = (PFNGLCREATETEXTURESPROC) IRR_OGL_LOAD_EXTENSION("glCreateTextures");
    pGlTexStorage1D = (PFNGLTEXSTORAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage1D");
    pGlTexStorage2D = (PFNGLTEXSTORAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage2D");
    pGlTexStorage3D = (PFNGLTEXSTORAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage3D");
    pGlTexStorage2DMultisample = (PFNGLTEXSTORAGE2DMULTISAMPLEPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage2DMultisample");
    pGlTexStorage3DMultisample = (PFNGLTEXSTORAGE3DMULTISAMPLEPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage3DMultisample");
    pGlTexBuffer = (PFNGLTEXBUFFERPROC) IRR_OGL_LOAD_EXTENSION( "glTexBuffer");
    pGlTexBufferRange = (PFNGLTEXBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION( "glTexBufferRange");
    pGlTextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage1D");
    pGlTextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2D");
    pGlTextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3D");
    pGlTextureStorage2DMultisample = (PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2DMultisample");
    pGlTextureStorage3DMultisample = (PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3DMultisample");
    pGlTextureBuffer = (PFNGLTEXTUREBUFFERPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBuffer");
    pGlTextureBufferRange = (PFNGLTEXTUREBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBufferRange");
    pGlTextureStorage1DEXT = (PFNGLTEXTURESTORAGE1DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage1DEXT");
    pGlTextureStorage2DEXT = (PFNGLTEXTURESTORAGE2DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2DEXT");
    pGlTextureStorage3DEXT = (PFNGLTEXTURESTORAGE3DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3DEXT");
    pGlTextureBufferEXT = (PFNGLTEXTUREBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBufferEXT");
    pGlTextureBufferRangeEXT = (PFNGLTEXTUREBUFFERRANGEEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBufferRangeEXT");
    pGlTextureStorage2DMultisampleEXT = (PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2DMultisampleEXT");
    pGlTextureStorage3DMultisampleEXT = (PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3DMultisampleEXT");
	pGlGetTextureSubImage = (PFNGLGETTEXTURESUBIMAGEPROC)IRR_OGL_LOAD_EXTENSION("glGetTextureSubImage");
	pGlGetCompressedTextureSubImage = (PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC)IRR_OGL_LOAD_EXTENSION("glGetCompressedTextureSubImage");
	pGlGetTextureImage = (PFNGLGETTEXTUREIMAGEPROC)IRR_OGL_LOAD_EXTENSION("glGetTextureImage");
	pGlGetTextureImageEXT = (PFNGLGETTEXTUREIMAGEEXTPROC)IRR_OGL_LOAD_EXTENSION("glGetTextureImageEXT");
	pGlGetCompressedTextureImage = (PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC)IRR_OGL_LOAD_EXTENSION("glGetCompressedTextureImage");
	pGlGetCompressedTextureImageEXT = (PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC)IRR_OGL_LOAD_EXTENSION("glGetCompressedTextureImageEXT");
	pGlGetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGEPROC)IRR_OGL_LOAD_EXTENSION("glGetCompressedTexImage");
    pGlTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTexSubImage3D");
    pGlMultiTexSubImage1DEXT = (PFNGLMULTITEXSUBIMAGE1DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glMultiTexSubImage1DEXT");
    pGlMultiTexSubImage2DEXT = (PFNGLMULTITEXSUBIMAGE2DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glMultiTexSubImage2DEXT");
    pGlMultiTexSubImage3DEXT = (PFNGLMULTITEXSUBIMAGE3DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glMultiTexSubImage3DEXT");
    pGlTextureSubImage1D = (PFNGLTEXTURESUBIMAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureSubImage1D");
    pGlTextureSubImage2D = (PFNGLTEXTURESUBIMAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureSubImage2D");
    pGlTextureSubImage3D = (PFNGLTEXTURESUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureSubImage3D");
    pGlTextureSubImage1DEXT = (PFNGLTEXTURESUBIMAGE1DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureSubImage1DEXT");
    pGlTextureSubImage2DEXT = (PFNGLTEXTURESUBIMAGE2DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureSubImage2DEXT");
    pGlTextureSubImage3DEXT = (PFNGLTEXTURESUBIMAGE3DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureSubImage3DEXT");
    pGlCompressedTexSubImage1D = (PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTexSubImage1D");
    pGlCompressedTexSubImage2D = (PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTexSubImage2D");
    pGlCompressedTexSubImage3D = (PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTexSubImage3D");
    pGlCompressedTextureSubImage1D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage1D");
    pGlCompressedTextureSubImage2D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage2D");
    pGlCompressedTextureSubImage3D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage3D");
    pGlCompressedTextureSubImage1DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage1DEXT");
    pGlCompressedTextureSubImage2DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage2DEXT");
    pGlCompressedTextureSubImage3DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage3DEXT");
    pGlCopyTexSubImage3D = (PFNGLCOPYTEXSUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTexSubImage3D");
    pGlCopyTextureSubImage1D = (PFNGLCOPYTEXTURESUBIMAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTextureSubImage1D");
    pGlCopyTextureSubImage2D = (PFNGLCOPYTEXTURESUBIMAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTextureSubImage2D");
    pGlCopyTextureSubImage3D = (PFNGLCOPYTEXTURESUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTextureSubImage3D");
    pGlCopyTextureSubImage1DEXT = (PFNGLCOPYTEXTURESUBIMAGE1DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTextureSubImage1DEXT");
    pGlCopyTextureSubImage2DEXT = (PFNGLCOPYTEXTURESUBIMAGE2DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTextureSubImage2DEXT");
    pGlCopyTextureSubImage3DEXT = (PFNGLCOPYTEXTURESUBIMAGE3DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glCopyTextureSubImage3DEXT");
    pGlGenerateMipmap = (PFNGLGENERATEMIPMAPPROC) IRR_OGL_LOAD_EXTENSION( "glGenerateMipmap");
    pGlGenerateTextureMipmap = (PFNGLGENERATETEXTUREMIPMAPPROC) IRR_OGL_LOAD_EXTENSION( "glGenerateTextureMipmap");
    pGlGenerateTextureMipmapEXT = (PFNGLGENERATETEXTUREMIPMAPEXTPROC) IRR_OGL_LOAD_EXTENSION( "glGenerateTextureMipmapEXT");
    pGlClampColor = (PFNGLCLAMPCOLORPROC) IRR_OGL_LOAD_EXTENSION( "glClampColor");

    //samplers
    pGlGenSamplers = (PFNGLGENSAMPLERSPROC) IRR_OGL_LOAD_EXTENSION( "glGenSamplers");
    pGlDeleteSamplers = (PFNGLDELETESAMPLERSPROC) IRR_OGL_LOAD_EXTENSION( "glDeleteSamplers");
    pGlBindSampler = (PFNGLBINDSAMPLERPROC) IRR_OGL_LOAD_EXTENSION( "glBindSampler");
    pGlBindSamplers = (PFNGLBINDSAMPLERSPROC) IRR_OGL_LOAD_EXTENSION( "glBindSamplers");
    pGlSamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC) IRR_OGL_LOAD_EXTENSION( "glSamplerParameteri");
    pGlSamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC) IRR_OGL_LOAD_EXTENSION( "glSamplerParameterf");

    //
    pGlBindImageTexture = (PFNGLBINDIMAGETEXTUREPROC) IRR_OGL_LOAD_EXTENSION( "glBindImageTexture");


    //
    pGlBindBufferBase = (PFNGLBINDBUFFERBASEPROC) IRR_OGL_LOAD_EXTENSION("glBindBufferBase");
    pGlBindBufferRange = (PFNGLBINDBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glBindBufferRange");
    pGlBindBuffersBase = (PFNGLBINDBUFFERSBASEPROC) IRR_OGL_LOAD_EXTENSION("glBindBuffersBase");
    pGlBindBuffersRange = (PFNGLBINDBUFFERSRANGEPROC) IRR_OGL_LOAD_EXTENSION("glBindBuffersRange");

	// get fragment and vertex program function pointers
	pGlCreateShader = (PFNGLCREATESHADERPROC) IRR_OGL_LOAD_EXTENSION("glCreateShader");
	pGlShaderSource = (PFNGLSHADERSOURCEPROC) IRR_OGL_LOAD_EXTENSION("glShaderSource");
	pGlCompileShader = (PFNGLCOMPILESHADERPROC) IRR_OGL_LOAD_EXTENSION("glCompileShader");
	pGlCreateProgram = (PFNGLCREATEPROGRAMPROC) IRR_OGL_LOAD_EXTENSION("glCreateProgram");
	pGlAttachShader = (PFNGLATTACHSHADERPROC) IRR_OGL_LOAD_EXTENSION("glAttachShader");
	pGlTransformFeedbackVaryings = (PFNGLTRANSFORMFEEDBACKVARYINGSPROC) IRR_OGL_LOAD_EXTENSION("glTransformFeedbackVaryings");
	pGlLinkProgram = (PFNGLLINKPROGRAMPROC) IRR_OGL_LOAD_EXTENSION("glLinkProgram");
	pGlUseProgram = (PFNGLUSEPROGRAMPROC) IRR_OGL_LOAD_EXTENSION("glUseProgram");
	pGlDeleteProgram = (PFNGLDELETEPROGRAMPROC) IRR_OGL_LOAD_EXTENSION("glDeleteProgram");
	pGlDeleteShader = (PFNGLDELETESHADERPROC) IRR_OGL_LOAD_EXTENSION("glDeleteShader");
	pGlGetAttachedShaders = (PFNGLGETATTACHEDSHADERSPROC) IRR_OGL_LOAD_EXTENSION("glGetAttachedShaders");
	pGlGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC) IRR_OGL_LOAD_EXTENSION("glGetShaderInfoLog");
	pGlGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC) IRR_OGL_LOAD_EXTENSION("glGetProgramInfoLog");
	pGlGetShaderiv = (PFNGLGETSHADERIVPROC) IRR_OGL_LOAD_EXTENSION("glGetShaderiv");
	pGlGetProgramiv = (PFNGLGETPROGRAMIVPROC) IRR_OGL_LOAD_EXTENSION("glGetProgramiv");
	pGlGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC) IRR_OGL_LOAD_EXTENSION("glGetUniformLocation");
	pGlProgramUniform1fv = (PFNGLPROGRAMUNIFORM1FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform1fv");
	pGlProgramUniform2fv = (PFNGLPROGRAMUNIFORM2FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform2fv");
	pGlProgramUniform3fv = (PFNGLPROGRAMUNIFORM3FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform3fv");
	pGlProgramUniform4fv = (PFNGLPROGRAMUNIFORM4FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform4fv");
	pGlProgramUniform1iv = (PFNGLPROGRAMUNIFORM1IVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform1iv");
	pGlProgramUniform2iv = (PFNGLPROGRAMUNIFORM2IVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform2iv");
	pGlProgramUniform3iv = (PFNGLPROGRAMUNIFORM3IVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform3iv");
	pGlProgramUniform4iv = (PFNGLPROGRAMUNIFORM4IVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform4iv");
	pGlProgramUniform1uiv = (PFNGLPROGRAMUNIFORM1UIVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform1uiv");
	pGlProgramUniform2uiv = (PFNGLPROGRAMUNIFORM2UIVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform2uiv");
	pGlProgramUniform3uiv = (PFNGLPROGRAMUNIFORM3UIVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform3uiv");
	pGlProgramUniform4uiv = (PFNGLPROGRAMUNIFORM4UIVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniform4uiv");
	pGlProgramUniformMatrix2fv = (PFNGLPROGRAMUNIFORMMATRIX2FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix2fv");
	pGlProgramUniformMatrix3fv = (PFNGLPROGRAMUNIFORMMATRIX3FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix3fv");
	pGlProgramUniformMatrix4fv = (PFNGLPROGRAMUNIFORMMATRIX4FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix4fv");
	pGlProgramUniformMatrix2x3fv = (PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix2x3fv");
	pGlProgramUniformMatrix3x2fv = (PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix3x2fv");
	pGlProgramUniformMatrix4x2fv = (PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix4x2fv");
	pGlProgramUniformMatrix2x4fv = (PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix2x4fv");
	pGlProgramUniformMatrix3x4fv = (PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix3x4fv");
	pGlProgramUniformMatrix4x3fv = (PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC) IRR_OGL_LOAD_EXTENSION("glProgramUniformMatrix4x3fv");
	pGlGetActiveUniform = (PFNGLGETACTIVEUNIFORMPROC) IRR_OGL_LOAD_EXTENSION("glGetActiveUniform");
    pGlBindProgramPipeline = (PFNGLBINDPROGRAMPIPELINEPROC) IRR_OGL_LOAD_EXTENSION("glBindProgramPipeline");

	//Criss
	pGlMemoryBarrier = (PFNGLMEMORYBARRIERPROC) IRR_OGL_LOAD_EXTENSION("glMemoryBarrier");
	pGlDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC) IRR_OGL_LOAD_EXTENSION("glDispatchCompute");
	pGlDispatchComputeIndirect = (PFNGLDISPATCHCOMPUTEINDIRECTPROC) IRR_OGL_LOAD_EXTENSION("glDispatchComputeIndirect");

	// get point parameter extension
	pGlPointParameterf = (PFNGLPOINTPARAMETERFARBPROC) IRR_OGL_LOAD_EXTENSION("glPointParameterf");
	pGlPointParameterfv = (PFNGLPOINTPARAMETERFVARBPROC) IRR_OGL_LOAD_EXTENSION("glPointParameterfv");

    //ROP
	pGlBlendColor = (PFNGLBLENDCOLORPROC)IRR_OGL_LOAD_EXTENSION("glBlendColor");
    pGlDepthRangeIndexed = (PFNGLDEPTHRANGEINDEXEDPROC) IRR_OGL_LOAD_EXTENSION("glDepthRangeIndexed");
    pGlViewportIndexedfv = (PFNGLVIEWPORTINDEXEDFVPROC) IRR_OGL_LOAD_EXTENSION("glViewportIndexedfv");
    pGlScissorIndexedv = (PFNGLSCISSORINDEXEDVPROC) IRR_OGL_LOAD_EXTENSION("glScissorIndexedv");
    pGlSampleCoverage = (PFNGLSAMPLECOVERAGEPROC) IRR_OGL_LOAD_EXTENSION("glSampleCoverage");
	pGlSampleMaski = (PFNGLSAMPLEMASKIPROC) IRR_OGL_LOAD_EXTENSION("glSampleMaski");
	pGlMinSampleShading = (PFNGLMINSAMPLESHADINGPROC) IRR_OGL_LOAD_EXTENSION("glMinSampleShading");
    pGlBlendEquationSeparatei = (PFNGLBLENDEQUATIONSEPARATEIPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationSeparatei");
    pGlBlendFuncSeparatei = (PFNGLBLENDFUNCSEPARATEIPROC) IRR_OGL_LOAD_EXTENSION("glBlendFuncSeparatei");
    pGlColorMaski = (PFNGLCOLORMASKIPROC) IRR_OGL_LOAD_EXTENSION("glColorMaski");
	pGlStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC) IRR_OGL_LOAD_EXTENSION("glStencilFuncSeparate");
	pGlStencilOpSeparate = (PFNGLSTENCILOPSEPARATEPROC) IRR_OGL_LOAD_EXTENSION("glStencilOpSeparate");
	pGlStencilMaskSeparate = (PFNGLSTENCILMASKSEPARATEPROC) IRR_OGL_LOAD_EXTENSION("glStencilMaskSeparate");

	// ARB FrameBufferObjects
	pGlBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glBlitNamedFramebuffer");
	pGlBlitFramebuffer = (PFNGLBLITFRAMEBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glBlitFramebuffer");
	pGlDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glDeleteFramebuffers");
	pGlCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glCreateFramebuffers");
	pGlGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glGenFramebuffers");
	pGlBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glBindFramebuffer");
	pGlCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC) IRR_OGL_LOAD_EXTENSION("glCheckFramebufferStatus");
	pGlCheckNamedFramebufferStatus = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC) IRR_OGL_LOAD_EXTENSION("glCheckNamedFramebufferStatus");
	pGlCheckNamedFramebufferStatusEXT = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC) IRR_OGL_LOAD_EXTENSION("glCheckNamedFramebufferStatusEXT");
	pGlFramebufferTexture = (PFNGLFRAMEBUFFERTEXTUREPROC) IRR_OGL_LOAD_EXTENSION("glFramebufferTexture");
	pGlNamedFramebufferTexture = (PFNGLNAMEDFRAMEBUFFERTEXTUREPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferTexture");
	pGlNamedFramebufferTextureEXT = (PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferTextureEXT");
	pGlFramebufferTextureLayer = (PFNGLFRAMEBUFFERTEXTURELAYERPROC) IRR_OGL_LOAD_EXTENSION("glFramebufferTextureLayer");
	pGlNamedFramebufferTextureLayer = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferTextureLayer");
	pGlNamedFramebufferTextureLayerEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferTextureLayerEXT");
	pGlFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)IRR_OGL_LOAD_EXTENSION("glFramebufferTexture2D");
	pGlNamedFramebufferTexture2DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC)IRR_OGL_LOAD_EXTENSION("glNamedFramebufferTexture2DEXT");
	pGlDrawBuffers = (PFNGLDRAWBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glDrawBuffers");
	pGlNamedFramebufferDrawBuffers = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferDrawBuffers");
	pGlFramebufferDrawBuffersEXT = (PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC) IRR_OGL_LOAD_EXTENSION("glFramebufferDrawBuffersEXT");
	pGlNamedFramebufferDrawBuffer = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferDrawBuffer");
	pGlFramebufferDrawBufferEXT = (PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION("glFramebufferDrawBufferEXT");
	pGlNamedFramebufferReadBuffer = (PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferReadBuffer");
	pGlFramebufferReadBufferEXT = (PFNGLFRAMEBUFFERREADBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION("glFramebufferReadBufferEXT");
    pGlClearNamedFramebufferiv = (PFNGLCLEARNAMEDFRAMEBUFFERIVPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedFramebufferiv");
    pGlClearNamedFramebufferuiv = (PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedFramebufferuiv");
    pGlClearNamedFramebufferfv = (PFNGLCLEARNAMEDFRAMEBUFFERFVPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedFramebufferfv");
    pGlClearNamedFramebufferfi = (PFNGLCLEARNAMEDFRAMEBUFFERFIPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedFramebufferfi");
    pGlClearBufferiv = (PFNGLCLEARBUFFERIVPROC) IRR_OGL_LOAD_EXTENSION("glClearBufferiv");
    pGlClearBufferuiv = (PFNGLCLEARBUFFERUIVPROC) IRR_OGL_LOAD_EXTENSION("glClearBufferuiv");
    pGlClearBufferfv = (PFNGLCLEARBUFFERFVPROC) IRR_OGL_LOAD_EXTENSION("glClearBufferfv");
    pGlClearBufferfi = (PFNGLCLEARBUFFERFIPROC) IRR_OGL_LOAD_EXTENSION("glClearBufferfi");

	// get vertex buffer extension
	pGlGenBuffers = (PFNGLGENBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glGenBuffers");
    pGlCreateBuffers = (PFNGLCREATEBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glCreateBuffers");
	pGlBindBuffer = (PFNGLBINDBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glBindBuffer");
	pGlDeleteBuffers = (PFNGLDELETEBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glDeleteBuffers");
    pGlBufferStorage = (PFNGLBUFFERSTORAGEPROC) IRR_OGL_LOAD_EXTENSION("glBufferStorage");
    pGlNamedBufferStorage = (PFNGLNAMEDBUFFERSTORAGEPROC) IRR_OGL_LOAD_EXTENSION("glNamedBufferStorage");
    pGlNamedBufferStorageEXT = (PFNGLNAMEDBUFFERSTORAGEEXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedBufferStorageEXT");
    pGlBufferSubData = (PFNGLBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glBufferSubData");
    pGlNamedBufferSubData = (PFNGLNAMEDBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glNamedBufferSubData");
    pGlNamedBufferSubDataEXT = (PFNGLNAMEDBUFFERSUBDATAEXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedBufferSubDataEXT");
    pGlGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glGetBufferSubData");
    pGlGetNamedBufferSubData = (PFNGLGETNAMEDBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glGetNamedBufferSubData");
    pGlGetNamedBufferSubDataEXT = (PFNGLGETNAMEDBUFFERSUBDATAEXTPROC) IRR_OGL_LOAD_EXTENSION("glGetNamedBufferSubDataEXT");
    pGlMapBuffer = (PFNGLMAPBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glMapBuffer");
    pGlMapNamedBuffer = (PFNGLMAPNAMEDBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glMapNamedBuffer");
    pGlMapNamedBufferEXT = (PFNGLMAPNAMEDBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION("glMapNamedBufferEXT");
    pGlMapBufferRange = (PFNGLMAPBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glMapBufferRange");
    pGlMapNamedBufferRange = (PFNGLMAPNAMEDBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glMapNamedBufferRange");
    pGlMapNamedBufferRangeEXT = (PFNGLMAPNAMEDBUFFERRANGEEXTPROC) IRR_OGL_LOAD_EXTENSION("glMapNamedBufferRangeEXT");
    pGlFlushMappedBufferRange = (PFNGLFLUSHMAPPEDBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glFlushMappedBufferRange");
    pGlFlushMappedNamedBufferRange = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glFlushMappedNamedBufferRange");
    pGlFlushMappedNamedBufferRangeEXT = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC) IRR_OGL_LOAD_EXTENSION("glFlushMappedNamedBufferRangeEXT");
    pGlUnmapBuffer = (PFNGLUNMAPBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glUnmapBuffer");
    pGlUnmapNamedBuffer = (PFNGLUNMAPNAMEDBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glUnmapNamedBuffer");
    pGlUnmapNamedBufferEXT = (PFNGLUNMAPNAMEDBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION("glUnmapNamedBufferEXT");
    pGlClearBufferData = (PFNGLCLEARBUFFERDATAPROC) IRR_OGL_LOAD_EXTENSION("glClearBufferData");
    pGlClearNamedBufferData = (PFNGLCLEARNAMEDBUFFERDATAPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedBufferData");
    pGlClearNamedBufferDataEXT = (PFNGLCLEARNAMEDBUFFERDATAEXTPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedBufferDataEXT");
    pGlClearBufferSubData = (PFNGLCLEARBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glClearBufferSubData");
    pGlClearNamedBufferSubData = (PFNGLCLEARNAMEDBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedBufferSubData");
    pGlClearNamedBufferSubDataEXT = (PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC) IRR_OGL_LOAD_EXTENSION("glClearNamedBufferSubDataEXT");
    pGlCopyBufferSubData = (PFNGLCOPYBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glCopyBufferSubData");
    pGlCopyNamedBufferSubData = (PFNGLCOPYNAMEDBUFFERSUBDATAPROC) IRR_OGL_LOAD_EXTENSION("glCopyNamedBufferSubData");
    pGlNamedCopyBufferSubDataEXT = (PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedCopyBufferSubDataEXT");
	pGlIsBuffer = (PFNGLISBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glIsBuffer");
    pGlGetNamedBufferParameteri64v = (PFNGLGETNAMEDBUFFERPARAMETERI64VPROC) IRR_OGL_LOAD_EXTENSION("glGetNamedBufferParameteri64v");
    pGlGetBufferParameteri64v = (PFNGLGETBUFFERPARAMETERI64VPROC) IRR_OGL_LOAD_EXTENSION("glGetBufferParameteri64v");
    pGlGetNamedBufferParameteriv = (PFNGLGETNAMEDBUFFERPARAMETERIVPROC) IRR_OGL_LOAD_EXTENSION("glGetNamedBufferParameteriv");
    pGlGetNamedBufferParameterivEXT = (PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC) IRR_OGL_LOAD_EXTENSION("glGetNamedBufferParameterivEXT");
    pGlGetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIVPROC) IRR_OGL_LOAD_EXTENSION("glGetBufferParameteriv");
	//vao
    pGlGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC) IRR_OGL_LOAD_EXTENSION("glGenVertexArrays");
    pGlCreateVertexArrays = (PFNGLCREATEVERTEXARRAYSPROC) IRR_OGL_LOAD_EXTENSION("glCreateVertexArrays");
    pGlDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC) IRR_OGL_LOAD_EXTENSION("glDeleteVertexArrays");
    pGlBindVertexArray = (PFNGLBINDVERTEXARRAYPROC) IRR_OGL_LOAD_EXTENSION("glBindVertexArray");
    pGlVertexArrayElementBuffer = (PFNGLVERTEXARRAYELEMENTBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayElementBuffer");
    pGlBindVertexBuffer = (PFNGLBINDVERTEXBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glBindVertexBuffer");
    pGlVertexArrayVertexBuffer = (PFNGLVERTEXARRAYVERTEXBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayVertexBuffer");
    pGlVertexArrayBindVertexBufferEXT = (PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayBindVertexBufferEXT");
    pGlVertexAttribBinding = (PFNGLVERTEXATTRIBBINDINGPROC) IRR_OGL_LOAD_EXTENSION("glVertexAttribBinding");
    pGlVertexArrayAttribBinding = (PFNGLVERTEXARRAYATTRIBBINDINGPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayAttribBinding");
    pGlVertexArrayVertexAttribBindingEXT = (PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribBindingEXT");
    pGlEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC) IRR_OGL_LOAD_EXTENSION("glEnableVertexAttribArray");
    pGlEnableVertexArrayAttrib = (PFNGLENABLEVERTEXARRAYATTRIBPROC) IRR_OGL_LOAD_EXTENSION("glEnableVertexArrayAttrib");
    pGlEnableVertexArrayAttribEXT = (PFNGLENABLEVERTEXARRAYATTRIBEXTPROC) IRR_OGL_LOAD_EXTENSION("glEnableVertexArrayAttribEXT");
    pGlDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC) IRR_OGL_LOAD_EXTENSION("glDisableVertexAttribArray");
    pGlDisableVertexArrayAttrib = (PFNGLDISABLEVERTEXARRAYATTRIBPROC) IRR_OGL_LOAD_EXTENSION("glDisableVertexArrayAttrib");
    pGlDisableVertexArrayAttribEXT = (PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC) IRR_OGL_LOAD_EXTENSION("glDisableVertexArrayAttribEXT");
    pGlVertexAttribFormat = (PFNGLVERTEXATTRIBFORMATPROC) IRR_OGL_LOAD_EXTENSION("glVertexAttribFormat");
    pGlVertexAttribIFormat = (PFNGLVERTEXATTRIBIFORMATPROC) IRR_OGL_LOAD_EXTENSION("glVertexAttribIFormat");
    pGlVertexAttribLFormat = (PFNGLVERTEXATTRIBLFORMATPROC) IRR_OGL_LOAD_EXTENSION("glVertexAttribLFormat");
    pGlVertexArrayAttribFormat = (PFNGLVERTEXARRAYATTRIBFORMATPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayAttribFormat");
    pGlVertexArrayAttribIFormat = (PFNGLVERTEXARRAYATTRIBIFORMATPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayAttribIFormat");
    pGlVertexArrayAttribLFormat = (PFNGLVERTEXARRAYATTRIBLFORMATPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayAttribLFormat");
    pGlVertexArrayVertexAttribFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribFormatEXT");
    pGlVertexArrayVertexAttribIFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribIFormatEXT");
    pGlVertexArrayVertexAttribLFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribLFormatEXT");
    pGlVertexArrayBindingDivisor = (PFNGLVERTEXARRAYBINDINGDIVISORPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayBindingDivisor");
    pGlVertexArrayVertexBindingDivisorEXT = (PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC) IRR_OGL_LOAD_EXTENSION("glVertexArrayVertexBindingDivisorEXT");
    pGlVertexBindingDivisor = (PFNGLVERTEXBINDINGDIVISORPROC) IRR_OGL_LOAD_EXTENSION("glVertexBindingDivisor");
    //
    pGlPrimitiveRestartIndex = (PFNGLPRIMITIVERESTARTINDEXPROC) IRR_OGL_LOAD_EXTENSION("glPrimitiveRestartIndex");
    pGlDrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCEDPROC) IRR_OGL_LOAD_EXTENSION("glDrawArraysInstanced");
    pGlDrawArraysInstancedBaseInstance = (PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC) IRR_OGL_LOAD_EXTENSION("glDrawArraysInstancedBaseInstance");
    pGlDrawElementsInstancedBaseVertex = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC) IRR_OGL_LOAD_EXTENSION("glDrawElementsInstancedBaseVertex");
    pGlDrawElementsInstancedBaseVertexBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC) IRR_OGL_LOAD_EXTENSION("glDrawElementsInstancedBaseVertexBaseInstance");
    pGlDrawTransformFeedback = (PFNGLDRAWTRANSFORMFEEDBACKPROC) IRR_OGL_LOAD_EXTENSION("glDrawTransformFeedback");
    pGlDrawTransformFeedbackInstanced = (PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC) IRR_OGL_LOAD_EXTENSION("glDrawTransformFeedbackInstanced");
    pGlDrawTransformFeedbackStream = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC) IRR_OGL_LOAD_EXTENSION("glDrawTransformFeedbackStream");
    pGlDrawTransformFeedbackStreamInstanced = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC) IRR_OGL_LOAD_EXTENSION("glDrawTransformFeedbackStreamInstanced");
    pGlDrawArraysIndirect = (PFNGLDRAWARRAYSINDIRECTPROC) IRR_OGL_LOAD_EXTENSION("glDrawArraysIndirect");
    pGlDrawElementsIndirect = (PFNGLDRAWELEMENTSINDIRECTPROC) IRR_OGL_LOAD_EXTENSION("glDrawElementsIndirect");
    pGlMultiDrawArraysIndirect = (PFNGLMULTIDRAWARRAYSINDIRECTPROC) IRR_OGL_LOAD_EXTENSION("glMultiDrawArraysIndirect");
    pGlMultiDrawElementsIndirect = (PFNGLMULTIDRAWELEMENTSINDIRECTPROC) IRR_OGL_LOAD_EXTENSION("glMultiDrawElementsIndirect");
    //
	pGlCreateTransformFeedbacks = (PFNGLCREATETRANSFORMFEEDBACKSPROC) IRR_OGL_LOAD_EXTENSION("glCreateTransformFeedbacks");
	pGlGenTransformFeedbacks = (PFNGLGENTRANSFORMFEEDBACKSPROC) IRR_OGL_LOAD_EXTENSION("glGenTransformFeedbacks");
	pGlDeleteTransformFeedbacks = (PFNGLDELETETRANSFORMFEEDBACKSPROC) IRR_OGL_LOAD_EXTENSION("glDeleteTransformFeedbacks");
	pGlBindTransformFeedback = (PFNGLBINDTRANSFORMFEEDBACKPROC) IRR_OGL_LOAD_EXTENSION("glBindTransformFeedback");
	pGlBeginTransformFeedback = (PFNGLBEGINTRANSFORMFEEDBACKPROC) IRR_OGL_LOAD_EXTENSION("glBeginTransformFeedback");
	pGlPauseTransformFeedback = (PFNGLPAUSETRANSFORMFEEDBACKPROC) IRR_OGL_LOAD_EXTENSION("glPauseTransformFeedback");
	pGlResumeTransformFeedback = (PFNGLRESUMETRANSFORMFEEDBACKPROC) IRR_OGL_LOAD_EXTENSION("glResumeTransformFeedback");
	pGlEndTransformFeedback = (PFNGLENDTRANSFORMFEEDBACKPROC) IRR_OGL_LOAD_EXTENSION("glEndTransformFeedback");
	pGlTransformFeedbackBufferBase = (PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC) IRR_OGL_LOAD_EXTENSION("glTransformFeedbackBufferBase");
	pGlTransformFeedbackBufferRange = (PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glTransformFeedbackBufferRange");
	//
	pGlBlendFuncSeparate = (PFNGLBLENDFUNCSEPARATEPROC) IRR_OGL_LOAD_EXTENSION("glBlendFuncSeparate");
	pGlEnablei = (PFNGLENABLEIPROC) IRR_OGL_LOAD_EXTENSION("glEnablei");
	pGlDisablei = (PFNGLDISABLEIPROC) IRR_OGL_LOAD_EXTENSION("glDisablei");
	pGlBlendFuncIndexedAMD= (PFNGLBLENDFUNCINDEXEDAMDPROC) IRR_OGL_LOAD_EXTENSION("glBlendFuncIndexedAMD");
	pGlBlendFunciARB= (PFNGLBLENDFUNCIPROC) IRR_OGL_LOAD_EXTENSION("glBlendFunciARB");
	pGlBlendEquationIndexedAMD= (PFNGLBLENDEQUATIONINDEXEDAMDPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationIndexedAMD");
	pGlBlendEquationiARB= (PFNGLBLENDEQUATIONIPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationiARB");
	pGlProgramParameteri= (PFNGLPROGRAMPARAMETERIPROC) IRR_OGL_LOAD_EXTENSION("glProgramParameteri");
	pGlPatchParameterfv = (PFNGLPATCHPARAMETERFVPROC) IRR_OGL_LOAD_EXTENSION("glPatchParameterfv");
	pGlPatchParameteri = (PFNGLPATCHPARAMETERIPROC) IRR_OGL_LOAD_EXTENSION("glPatchParameteri");

	// occlusion query
	pGlCreateQueries = (PFNGLCREATEQUERIESPROC) IRR_OGL_LOAD_EXTENSION("glCreateQueries");
	pGlGenQueries = (PFNGLGENQUERIESPROC) IRR_OGL_LOAD_EXTENSION("glGenQueries");
	pGlDeleteQueries = (PFNGLDELETEQUERIESPROC) IRR_OGL_LOAD_EXTENSION("glDeleteQueries");
	pGlIsQuery = (PFNGLISQUERYPROC) IRR_OGL_LOAD_EXTENSION("glIsQuery");
	pGlBeginQuery = (PFNGLBEGINQUERYPROC) IRR_OGL_LOAD_EXTENSION("glBeginQuery");
	pGlEndQuery = (PFNGLENDQUERYPROC) IRR_OGL_LOAD_EXTENSION("glEndQuery");
	pGlBeginQueryIndexed = (PFNGLBEGINQUERYINDEXEDPROC) IRR_OGL_LOAD_EXTENSION("glBeginQueryIndexed");
	pGlEndQueryIndexed = (PFNGLENDQUERYINDEXEDPROC) IRR_OGL_LOAD_EXTENSION("glEndQueryIndexed");
	pGlGetQueryiv = (PFNGLGETQUERYIVPROC) IRR_OGL_LOAD_EXTENSION("glGetQueryiv");
	pGlGetQueryObjectuiv = (PFNGLGETQUERYOBJECTUIVPROC) IRR_OGL_LOAD_EXTENSION("glGetQueryObjectuiv");
	pGlGetQueryObjectui64v = (PFNGLGETQUERYOBJECTUI64VPROC) IRR_OGL_LOAD_EXTENSION("glGetQueryObjectui64v");
    pGlGetQueryBufferObjectuiv = (PFNGLGETQUERYBUFFEROBJECTUIVPROC) IRR_OGL_LOAD_EXTENSION("glGetQueryBufferObjectuiv");
    pGlGetQueryBufferObjectui64v = (PFNGLGETQUERYBUFFEROBJECTUI64VPROC) IRR_OGL_LOAD_EXTENSION("glGetQueryBufferObjectui64v");
	pGlQueryCounter = (PFNGLQUERYCOUNTERPROC) IRR_OGL_LOAD_EXTENSION("glQueryCounter");
	pGlBeginConditionalRender = (PFNGLBEGINCONDITIONALRENDERPROC) IRR_OGL_LOAD_EXTENSION("glBeginConditionalRender");
    pGlEndConditionalRender = (PFNGLENDCONDITIONALRENDERPROC) IRR_OGL_LOAD_EXTENSION("glEndConditionalRender");

    if (FeatureAvailable[IRR_ARB_texture_barrier])
        pGlTextureBarrier = (PFNGLTEXTUREBARRIERPROC) IRR_OGL_LOAD_EXTENSION("glTextureBarrier");
    else if (FeatureAvailable[IRR_NV_texture_barrier])
        pGlTextureBarrierNV = (PFNGLTEXTUREBARRIERNVPROC) IRR_OGL_LOAD_EXTENSION("glTextureBarrierNV");


    pGlDebugMessageControl = (PFNGLDEBUGMESSAGECONTROLPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageControl");
    pGlDebugMessageControlARB = (PFNGLDEBUGMESSAGECONTROLARBPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageControlARB");
    pGlDebugMessageCallback = (PFNGLDEBUGMESSAGECALLBACKPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageCallback");
    pGlDebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageCallbackARB");

	// blend equation
	pGlBlendEquationEXT = (PFNGLBLENDEQUATIONEXTPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationEXT");
	pGlBlendEquation = (PFNGLBLENDEQUATIONPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquation");

    pGlGetInternalformativ = (PFNGLGETINTERNALFORMATIVPROC) IRR_OGL_LOAD_EXTENSION("glGetInternalformativ");
    pGlGetInternalformati64v = (PFNGLGETINTERNALFORMATI64VPROC) IRR_OGL_LOAD_EXTENSION("glGetInternalformati64v");

	// get vsync extension
	#if defined(WGL_EXT_swap_control) && !defined(_IRR_COMPILE_WITH_SDL_DEVICE_)
		pWglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC) IRR_OGL_LOAD_EXTENSION("wglSwapIntervalEXT");
	#endif
	#if defined(GLX_SGI_swap_control) && !defined(_IRR_COMPILE_WITH_SDL_DEVICE_)
		pGlxSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC)IRR_OGL_LOAD_EXTENSION("glXSwapIntervalSGI");
	#endif
	#if defined(GLX_EXT_swap_control) && !defined(_IRR_COMPILE_WITH_SDL_DEVICE_)
		pGlxSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)IRR_OGL_LOAD_EXTENSION("glXSwapIntervalEXT");
	#endif
	#if defined(GLX_MESA_swap_control) && !defined(_IRR_COMPILE_WITH_SDL_DEVICE_)
		pGlxSwapIntervalMESA = (PFNGLXSWAPINTERVALMESAPROC)IRR_OGL_LOAD_EXTENSION("glXSwapIntervalMESA");
	#endif

	functionsAlreadyLoaded = true;
}


bool COpenGLExtensionHandler::isDeviceCompatibile(core::vector<std::string>* failedExtensions)
{
    bool retval = true;

    if (Version<430)
    {
        retval = false;
        std::string error = "OpenGL Version Lower Than 4.3\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[IRR_EXT_texture_filter_anisotropic]||Version>=460))
    {
        retval =  false;
        std::string error = "No anisotropic filtering\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[IRR_EXT_texture_compression_s3tc]))
    {
        retval =  false;
        std::string error = "DXTn compression missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[IRR_ARB_buffer_storage]||Version>=440))
    {
        retval =  false;
        std::string error = "GL_ARB_buffer_storage missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[IRR_ARB_clip_control]||Version>=450))
    {
        retval =  false;
        std::string error = "GL_ARB_clip_control missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

/*
    if (!(FeatureAvailable[IRR_NV_texture_barrier]||FeatureAvailable[IRR_ARB_texture_barrier]||Version>=450))
    {
        retval =  false;
        std::string error = "GL_NV_texture_barrier missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }
*/

    if (!(FeatureAvailable[IRR_ARB_direct_state_access] || FeatureAvailable[IRR_EXT_direct_state_access] || Version>=450))
    {
        retval =  false;
        std::string error = "Direct State Access Extension missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }


    return retval;
}

}
}

#endif

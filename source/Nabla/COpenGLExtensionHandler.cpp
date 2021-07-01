// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "COpenGLExtensionHandler.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
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

//#include "COpenGLStateManagerImpl.h"


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
uint64_t COpenGLExtensionHandler::maxTBOSizeInTexels = 0;
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
GLuint COpenGLExtensionHandler::SPIR_VextensionsCount = 0u;
core::smart_refctd_dynamic_array<const GLubyte*> COpenGLExtensionHandler::SPIR_Vextensions;
uint32_t COpenGLExtensionHandler::MaxGeometryVerticesOut = 65535;
float COpenGLExtensionHandler::MaxTextureLODBias = 0.f;

uint32_t COpenGLExtensionHandler::maxUBOBindings = 0u;
uint32_t COpenGLExtensionHandler::maxSSBOBindings = 0u;
uint32_t COpenGLExtensionHandler::maxTextureBindings = 0u;
uint32_t COpenGLExtensionHandler::maxTextureBindingsCompute = 0u;
uint32_t COpenGLExtensionHandler::maxImageBindings = 0u;

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
PFNGLTEXTUREVIEWPROC COpenGLExtensionHandler::pGlTextureView = nullptr;
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
PFNGLCOPYIMAGESUBDATAPROC COpenGLExtensionHandler::pGlCopyImageSubData = nullptr;
PFNGLTEXTUREPARAMETERIUIVPROC COpenGLExtensionHandler::pGlTextureParameterIuiv = nullptr;
PFNGLTEXTUREPARAMETERIUIVEXTPROC COpenGLExtensionHandler::pGlTextureParameterIuivEXT = nullptr;
PFNGLTEXPARAMETERIUIVPROC COpenGLExtensionHandler::pGlTexParameterIuiv = nullptr;
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
PFNGLSAMPLERPARAMETERFVPROC COpenGLExtensionHandler::pGlSamplerParameterfv = nullptr;

//
PFNGLBINDIMAGETEXTUREPROC COpenGLExtensionHandler::pGlBindImageTexture = nullptr;
PFNGLBINDIMAGETEXTURESPROC COpenGLExtensionHandler::pGlBindImageTextures = nullptr;

//bindless textures
//ARB
PFNGLGETTEXTUREHANDLEARBPROC COpenGLExtensionHandler::pGlGetTextureHandleARB = nullptr;
PFNGLGETTEXTURESAMPLERHANDLEARBPROC COpenGLExtensionHandler::pGlGetTextureSamplerHandleARB = nullptr;
PFNGLMAKETEXTUREHANDLERESIDENTARBPROC COpenGLExtensionHandler::pGlMakeTextureHandleResidentARB = nullptr;
PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC COpenGLExtensionHandler::pGlMakeTextureHandleNonResidentARB = nullptr;
PFNGLGETIMAGEHANDLEARBPROC COpenGLExtensionHandler::pGlGetImageHandleARB = nullptr;
PFNGLMAKEIMAGEHANDLERESIDENTARBPROC COpenGLExtensionHandler::pGlMakeImageHandleResidentARB = nullptr;
PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC COpenGLExtensionHandler::pGlMakeImageHandleNonResidentARB = nullptr;
PFNGLISTEXTUREHANDLERESIDENTARBPROC COpenGLExtensionHandler::pGlIsTextureHandleResidentARB = nullptr;
PFNGLISIMAGEHANDLERESIDENTARBPROC COpenGLExtensionHandler::pGlIsImageHandleResidentARB = nullptr;
//NV
PFNGLGETTEXTUREHANDLENVPROC COpenGLExtensionHandler::pGlGetTextureHandleNV = nullptr;
PFNGLGETTEXTURESAMPLERHANDLENVPROC COpenGLExtensionHandler::pGlGetTextureSamplerHandleNV = nullptr;
PFNGLMAKETEXTUREHANDLERESIDENTNVPROC COpenGLExtensionHandler::pGlMakeTextureHandleResidentNV = nullptr;
PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC COpenGLExtensionHandler::pGlMakeTextureHandleNonResidentNV = nullptr;
PFNGLGETIMAGEHANDLENVPROC COpenGLExtensionHandler::pGlGetImageHandleNV = nullptr;
PFNGLMAKEIMAGEHANDLERESIDENTNVPROC COpenGLExtensionHandler::pGlMakeImageHandleResidentNV = nullptr;
PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC COpenGLExtensionHandler::pGlMakeImageHandleNonResidentNV = nullptr;
PFNGLISTEXTUREHANDLERESIDENTNVPROC COpenGLExtensionHandler::pGlIsTextureHandleResidentNV = nullptr;
PFNGLISIMAGEHANDLERESIDENTNVPROC COpenGLExtensionHandler::pGlIsImageHandleResidentNV = nullptr;

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
PFNGLCREATESHADERPROGRAMVPROC COpenGLExtensionHandler::pGlCreateShaderProgramv = nullptr;
PFNGLCREATEPROGRAMPIPELINESPROC COpenGLExtensionHandler::pGlCreateProgramPipelines = nullptr;
PFNGLDELETEPROGRAMPIPELINESPROC COpenGLExtensionHandler::pGlDeleteProgramPipelines = nullptr;
PFNGLUSEPROGRAMSTAGESPROC COpenGLExtensionHandler::pGlUseProgramStages = nullptr;
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
PFNGLGETPROGRAMBINARYPROC COpenGLExtensionHandler::pGlGetProgramBinary = nullptr;
PFNGLPROGRAMBINARYPROC COpenGLExtensionHandler::pGlProgramBinary = nullptr;

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
PFNGLMULTIDRAWARRAYSINDIRECTCOUNTPROC COpenGLExtensionHandler::pGlMultiDrawArrysIndirectCount = nullptr;
PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTPROC COpenGLExtensionHandler::pGlMultiDrawElementsIndirectCount = nullptr;
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


core::CLeakDebugger COpenGLExtensionHandler::bufferLeaker("GLBuffer");
core::CLeakDebugger COpenGLExtensionHandler::textureLeaker("GLTex");



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
        for (uint32_t i=0; i<NBL_OpenGL_Feature_Count; ++i)
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
        for (uint32_t i=0; i<NBL_OpenGL_Feature_Count; ++i)
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
#ifdef _NBL_WINDOWS_API_
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
#ifdef _NBL_DEBUG
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
#elif defined(NBL_LINUX_DEVICE)
#endif
}


void COpenGLExtensionHandler::initExtensions(bool stencilBuffer)
{
    core::stringc vendorString = (char*)glGetString(GL_VENDOR);
    if (vendorString.find("Intel")!=-1 || vendorString.find("INTEL")!=-1)
	    IsIntelGPU = true;


	loadFunctions();


	TextureCompressionExtension = FeatureAvailable[NBL_ARB_texture_compression];
	StencilBuffer = stencilBuffer;


	GLint num = 0;

	glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &reqUBOAlignment);
    assert(core::is_alignment(reqUBOAlignment));
	glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &reqSSBOAlignment);
    assert(core::is_alignment(reqSSBOAlignment));
	glGetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &reqTBOAlignment);
    assert(core::is_alignment(reqTBOAlignment));

    extGlGetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, reinterpret_cast<GLint64*>(&maxUBOSize));
    extGlGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, reinterpret_cast<GLint64*>(&maxSSBOSize));
    extGlGetInteger64v(GL_MAX_TEXTURE_BUFFER_SIZE, reinterpret_cast<GLint64*>(&maxTBOSizeInTexels));
    maxBufferSize = std::max(maxUBOSize, maxSSBOSize);

    glGetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, reinterpret_cast<GLint*>(&maxUBOBindings));
    glGetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, reinterpret_cast<GLint*>(&maxSSBOBindings));
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&maxTextureBindings));
    glGetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&maxTextureBindingsCompute));
    glGetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, reinterpret_cast<GLint*>(&maxImageBindings));

	glGetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, &minMemoryMapAlignment);

    extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, MaxComputeWGSize);
    extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, MaxComputeWGSize+1);
    extGlGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, MaxComputeWGSize+2);


	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &num);
	MaxArrayTextureLayers = num;

	if (FeatureAvailable[NBL_EXT_texture_filter_anisotropic])
	{
		glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &num);
		MaxAnisotropy = static_cast<uint8_t>(num);
	}


    if (FeatureAvailable[NBL_ARB_geometry_shader4])
    {
        glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &num);
        MaxGeometryVerticesOut = static_cast<uint32_t>(num);
    }

	if (FeatureAvailable[NBL_EXT_texture_lod_bias])
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
    ShaderLanguageVersion = static_cast<uint16_t>(core::round(sl_ver*100.0f));
	
	//! For EXT-DSA testing
	if (IsIntelGPU)
	{
		Version = 440;
		FeatureAvailable[NBL_ARB_direct_state_access] = false;
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
    FeatureAvailable[NBL_EXT_direct_state_access] = FeatureAvailable[NBL_ARB_direct_state_access] = false;
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
	MaxTextureUnits = num;

    //num=100000000u;
	//glGetIntegerv(GL_MAX_ELEMENTS_INDICES,&num);
#ifdef WIN32
#ifdef _NBL_DEBUG
	if (FeatureAvailable[NBL_NVX_gpu_memory_info])
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
	if (FeatureAvailable[NBL_ATI_meminfo])
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

	for (uint32_t i=0; i<NBL_OpenGL_Feature_Count; ++i)
		FeatureAvailable[i]=false;



#ifdef _NBL_WINDOWS_API_
	#define NBL_OGL_LOAD_EXTENSION(x) wglGetProcAddress(reinterpret_cast<const char*>(x))
#elif defined(_NBL_COMPILE_WITH_SDL_DEVICE_) && !defined(_NBL_COMPILE_WITH_X11_DEVICE_)
	#define NBL_OGL_LOAD_EXTENSION(x) SDL_GL_GetProcAddress(reinterpret_cast<const char*>(x))
#else
    #define NBL_OGL_LOAD_EXTENSION(X) glXGetProcAddress(reinterpret_cast<const GLubyte*>(X))
#endif // Windows, SDL, or Linux

    pGlIsEnabledi = (PFNGLISENABLEDIPROC) NBL_OGL_LOAD_EXTENSION("glIsEnabledi");
    pGlEnablei = (PFNGLENABLEIPROC) NBL_OGL_LOAD_EXTENSION("glEnablei");
    pGlDisablei = (PFNGLDISABLEIPROC) NBL_OGL_LOAD_EXTENSION("glDisablei");
    pGlGetBooleani_v = (PFNGLGETBOOLEANI_VPROC) NBL_OGL_LOAD_EXTENSION("glGetBooleani_v");
    pGlGetFloati_v = (PFNGLGETFLOATI_VPROC) NBL_OGL_LOAD_EXTENSION("glGetFloati_v");
    pGlGetInteger64v = (PFNGLGETINTEGER64VPROC)NBL_OGL_LOAD_EXTENSION("glGetInteger64v");
    pGlGetIntegeri_v = (PFNGLGETINTEGERI_VPROC) NBL_OGL_LOAD_EXTENSION("glGetIntegeri_v");
    pGlGetStringi = (PFNGLGETSTRINGIPROC) NBL_OGL_LOAD_EXTENSION("glGetStringi");

	//should contain space-separated OpenGL extension names
	constexpr const char* OPENGL_EXTS_ENVVAR_NAME = "_NBL_OPENGL_EXTENSIONS_LIST";//move this to some top-level header?

	const char* envvar = std::getenv(OPENGL_EXTS_ENVVAR_NAME);
	if (!envvar)
	{
		GLint extensionCount;
		glGetIntegerv(GL_NUM_EXTENSIONS,&extensionCount);
		for (GLint i=0; i<extensionCount; ++i)
		{
			const char* extensionName = reinterpret_cast<const char*>(pGlGetStringi(GL_EXTENSIONS,i));

			for (uint32_t j=0; j<NBL_OpenGL_Feature_Count; ++j)
			{
				if (!strcmp(OpenGLFeatureStrings[j], extensionName))
				{
					FeatureAvailable[j] = true;
					break;
				}
			}
		}
	}
	else
	{
		std::stringstream ss{ std::string(envvar) };
		std::string extname;
		extname.reserve(100);
		while (std::getline(ss, extname))
		{
			for (uint32_t j=0; j<NBL_OpenGL_Feature_Count; ++j)
			{
				if (extname==OpenGLFeatureStrings[j])
				{
					FeatureAvailable[j] = true;
					break;
				}
			}
		}
	}

	float ogl_ver;
	sscanf(reinterpret_cast<const char*>(glGetString(GL_VERSION)),"%f",&ogl_ver);
	Version = static_cast<uint16_t>(core::round(ogl_ver*100.0f));

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

    if (FeatureAvailable[NBL_NV_shader_thread_group])
    {
        glGetIntegerv(GL_WARP_SIZE_NV, &num);
        InvocationSubGroupSize[0] = InvocationSubGroupSize[1] = reinterpret_cast<const uint32_t&>(num);
    }
    else if (IsIntelGPU)
    {
        InvocationSubGroupSize[0] = 4;
        InvocationSubGroupSize[1] = 32;
    }

    if (FeatureAvailable[NBL_ARB_spirv_extensions])
    {
        glGetIntegerv(GL_NUM_SPIR_V_EXTENSIONS, reinterpret_cast<GLint*>(&SPIR_VextensionsCount));
        if (SPIR_VextensionsCount)
            SPIR_Vextensions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<const GLubyte*> >(SPIR_VextensionsCount);
        for (GLuint i = 0u; i < SPIR_VextensionsCount; ++i)
            (*SPIR_Vextensions)[i] = pGlGetStringi(GL_SPIR_V_EXTENSIONS, i);
    }
    else
    {
        SPIR_VextensionsCount = 0u;
    }

    /**
    pGl = () NBL_OGL_LOAD_EXTENSION("gl");
    **/
    pGlProvokingVertex = (PFNGLPROVOKINGVERTEXPROC) NBL_OGL_LOAD_EXTENSION("glProvokingVertex");
    pGlClipControl = (PFNGLCLIPCONTROLPROC) NBL_OGL_LOAD_EXTENSION("glClipControl");

    //fences
    pGlFenceSync = (PFNGLFENCESYNCPROC) NBL_OGL_LOAD_EXTENSION("glFenceSync");
    pGlDeleteSync = (PFNGLDELETESYNCPROC) NBL_OGL_LOAD_EXTENSION("glDeleteSync");
    pGlClientWaitSync = (PFNGLCLIENTWAITSYNCPROC) NBL_OGL_LOAD_EXTENSION("glClientWaitSync");
    pGlWaitSync = (PFNGLWAITSYNCPROC) NBL_OGL_LOAD_EXTENSION("glWaitSync");

	// get multitexturing function pointers
    pGlActiveTexture = (PFNGLACTIVETEXTUREPROC) NBL_OGL_LOAD_EXTENSION("glActiveTexture");
	pGlBindTextures = (PFNGLBINDTEXTURESPROC) NBL_OGL_LOAD_EXTENSION("glBindTextures");
    pGlCreateTextures = (PFNGLCREATETEXTURESPROC) NBL_OGL_LOAD_EXTENSION("glCreateTextures");
    pGlTexStorage1D = (PFNGLTEXSTORAGE1DPROC) NBL_OGL_LOAD_EXTENSION( "glTexStorage1D");
    pGlTexStorage2D = (PFNGLTEXSTORAGE2DPROC) NBL_OGL_LOAD_EXTENSION( "glTexStorage2D");
    pGlTexStorage3D = (PFNGLTEXSTORAGE3DPROC) NBL_OGL_LOAD_EXTENSION( "glTexStorage3D");
    pGlTexStorage2DMultisample = (PFNGLTEXSTORAGE2DMULTISAMPLEPROC) NBL_OGL_LOAD_EXTENSION( "glTexStorage2DMultisample");
    pGlTexStorage3DMultisample = (PFNGLTEXSTORAGE3DMULTISAMPLEPROC) NBL_OGL_LOAD_EXTENSION( "glTexStorage3DMultisample");
    pGlTexBuffer = (PFNGLTEXBUFFERPROC) NBL_OGL_LOAD_EXTENSION( "glTexBuffer");
    pGlTexBufferRange = (PFNGLTEXBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION( "glTexBufferRange");
    pGlTextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage1D");
    pGlTextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage2D");
    pGlTextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage3D");
    pGlTextureStorage2DMultisample = (PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage2DMultisample");
    pGlTextureStorage3DMultisample = (PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage3DMultisample");
    pGlTextureBuffer = (PFNGLTEXTUREBUFFERPROC) NBL_OGL_LOAD_EXTENSION( "glTextureBuffer");
    pGlTextureBufferRange = (PFNGLTEXTUREBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION( "glTextureBufferRange");
	pGlTextureView = (PFNGLTEXTUREVIEWPROC) NBL_OGL_LOAD_EXTENSION( "glTextureView");
    pGlTextureStorage1DEXT = (PFNGLTEXTURESTORAGE1DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage1DEXT");
    pGlTextureStorage2DEXT = (PFNGLTEXTURESTORAGE2DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage2DEXT");
    pGlTextureStorage3DEXT = (PFNGLTEXTURESTORAGE3DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage3DEXT");
    pGlTextureBufferEXT = (PFNGLTEXTUREBUFFEREXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureBufferEXT");
    pGlTextureBufferRangeEXT = (PFNGLTEXTUREBUFFERRANGEEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureBufferRangeEXT");
    pGlTextureStorage2DMultisampleEXT = (PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage2DMultisampleEXT");
    pGlTextureStorage3DMultisampleEXT = (PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureStorage3DMultisampleEXT");
	pGlGetTextureSubImage = (PFNGLGETTEXTURESUBIMAGEPROC)NBL_OGL_LOAD_EXTENSION("glGetTextureSubImage");
	pGlGetCompressedTextureSubImage = (PFNGLGETCOMPRESSEDTEXTURESUBIMAGEPROC)NBL_OGL_LOAD_EXTENSION("glGetCompressedTextureSubImage");
	pGlGetTextureImage = (PFNGLGETTEXTUREIMAGEPROC)NBL_OGL_LOAD_EXTENSION("glGetTextureImage");
	pGlGetTextureImageEXT = (PFNGLGETTEXTUREIMAGEEXTPROC)NBL_OGL_LOAD_EXTENSION("glGetTextureImageEXT");
	pGlGetCompressedTextureImage = (PFNGLGETCOMPRESSEDTEXTUREIMAGEPROC)NBL_OGL_LOAD_EXTENSION("glGetCompressedTextureImage");
	pGlGetCompressedTextureImageEXT = (PFNGLGETCOMPRESSEDTEXTUREIMAGEEXTPROC)NBL_OGL_LOAD_EXTENSION("glGetCompressedTextureImageEXT");
	pGlGetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGEPROC)NBL_OGL_LOAD_EXTENSION("glGetCompressedTexImage");
    pGlTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC) NBL_OGL_LOAD_EXTENSION( "glTexSubImage3D");
    pGlMultiTexSubImage1DEXT = (PFNGLMULTITEXSUBIMAGE1DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glMultiTexSubImage1DEXT");
    pGlMultiTexSubImage2DEXT = (PFNGLMULTITEXSUBIMAGE2DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glMultiTexSubImage2DEXT");
    pGlMultiTexSubImage3DEXT = (PFNGLMULTITEXSUBIMAGE3DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glMultiTexSubImage3DEXT");
    pGlTextureSubImage1D = (PFNGLTEXTURESUBIMAGE1DPROC) NBL_OGL_LOAD_EXTENSION( "glTextureSubImage1D");
    pGlTextureSubImage2D = (PFNGLTEXTURESUBIMAGE2DPROC) NBL_OGL_LOAD_EXTENSION( "glTextureSubImage2D");
    pGlTextureSubImage3D = (PFNGLTEXTURESUBIMAGE3DPROC) NBL_OGL_LOAD_EXTENSION( "glTextureSubImage3D");
    pGlTextureSubImage1DEXT = (PFNGLTEXTURESUBIMAGE1DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureSubImage1DEXT");
    pGlTextureSubImage2DEXT = (PFNGLTEXTURESUBIMAGE2DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureSubImage2DEXT");
    pGlTextureSubImage3DEXT = (PFNGLTEXTURESUBIMAGE3DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureSubImage3DEXT");
    pGlCompressedTexSubImage1D = (PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTexSubImage1D");
    pGlCompressedTexSubImage2D = (PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTexSubImage2D");
    pGlCompressedTexSubImage3D = (PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTexSubImage3D");
    pGlCompressedTextureSubImage1D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage1D");
    pGlCompressedTextureSubImage2D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage2D");
    pGlCompressedTextureSubImage3D = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage3D");
    pGlCompressedTextureSubImage1DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage1DEXT");
    pGlCompressedTextureSubImage2DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage2DEXT");
    pGlCompressedTextureSubImage3DEXT = (PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC) NBL_OGL_LOAD_EXTENSION( "glCompressedTextureSubImage3DEXT");
	pGlCopyImageSubData = (PFNGLCOPYIMAGESUBDATAPROC) NBL_OGL_LOAD_EXTENSION( "glCopyImageSubData");
	pGlTextureParameterIuiv = (PFNGLTEXTUREPARAMETERIUIVPROC) NBL_OGL_LOAD_EXTENSION( "glTextureParameterIuiv");
	pGlTextureParameterIuivEXT = (PFNGLTEXTUREPARAMETERIUIVEXTPROC) NBL_OGL_LOAD_EXTENSION( "glTextureParameterIuivEXT");
	pGlTexParameterIuiv = (PFNGLTEXPARAMETERIUIVPROC) NBL_OGL_LOAD_EXTENSION( "glTexParameterIuiv");
    pGlGenerateMipmap = (PFNGLGENERATEMIPMAPPROC) NBL_OGL_LOAD_EXTENSION( "glGenerateMipmap");
    pGlGenerateTextureMipmap = (PFNGLGENERATETEXTUREMIPMAPPROC) NBL_OGL_LOAD_EXTENSION( "glGenerateTextureMipmap");
    pGlGenerateTextureMipmapEXT = (PFNGLGENERATETEXTUREMIPMAPEXTPROC) NBL_OGL_LOAD_EXTENSION( "glGenerateTextureMipmapEXT");
    pGlClampColor = (PFNGLCLAMPCOLORPROC) NBL_OGL_LOAD_EXTENSION( "glClampColor");

    //samplers
    pGlCreateSamplers = (PFNGLCREATESAMPLERSPROC)NBL_OGL_LOAD_EXTENSION("glCreateSamplers");
    pGlGenSamplers = (PFNGLGENSAMPLERSPROC) NBL_OGL_LOAD_EXTENSION( "glGenSamplers");
    pGlDeleteSamplers = (PFNGLDELETESAMPLERSPROC) NBL_OGL_LOAD_EXTENSION( "glDeleteSamplers");
    pGlBindSampler = (PFNGLBINDSAMPLERPROC) NBL_OGL_LOAD_EXTENSION( "glBindSampler");
    pGlBindSamplers = (PFNGLBINDSAMPLERSPROC) NBL_OGL_LOAD_EXTENSION( "glBindSamplers");
    pGlSamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC) NBL_OGL_LOAD_EXTENSION( "glSamplerParameteri");
    pGlSamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC) NBL_OGL_LOAD_EXTENSION( "glSamplerParameterf");
    pGlSamplerParameterfv = (PFNGLSAMPLERPARAMETERFVPROC)NBL_OGL_LOAD_EXTENSION("glSamplerParameterfv");

    //
    pGlBindImageTexture = (PFNGLBINDIMAGETEXTUREPROC) NBL_OGL_LOAD_EXTENSION( "glBindImageTexture");
    pGlBindImageTextures = (PFNGLBINDIMAGETEXTURESPROC) NBL_OGL_LOAD_EXTENSION( "glBindImageTextures" );

	//bindless texture
	//ARB
	pGlGetTextureHandleARB = (PFNGLGETTEXTUREHANDLEARBPROC) NBL_OGL_LOAD_EXTENSION("glGetTextureHandleARB");
	pGlGetTextureSamplerHandleARB = (PFNGLGETTEXTURESAMPLERHANDLEARBPROC) NBL_OGL_LOAD_EXTENSION("glGetTextureSamplerHandleARB");
	pGlMakeTextureHandleResidentARB = (PFNGLMAKETEXTUREHANDLERESIDENTARBPROC) NBL_OGL_LOAD_EXTENSION("glMakeTextureHandleResidentAR");
	pGlMakeTextureHandleNonResidentARB = (PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC) NBL_OGL_LOAD_EXTENSION("glMakeTextureHandleNonResidentARB");
	pGlGetImageHandleARB = (PFNGLGETIMAGEHANDLEARBPROC) NBL_OGL_LOAD_EXTENSION("glGetImageHandleARB");
	pGlMakeImageHandleResidentARB = (PFNGLMAKEIMAGEHANDLERESIDENTARBPROC) NBL_OGL_LOAD_EXTENSION("glMakeImageHandleResidentARB");
	pGlMakeImageHandleNonResidentARB = (PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC) NBL_OGL_LOAD_EXTENSION("glMakeImageHandleNonResidentARB");
	pGlIsTextureHandleResidentARB = (PFNGLISTEXTUREHANDLERESIDENTARBPROC) NBL_OGL_LOAD_EXTENSION("glIsTextureHandleResidentARB");
	pGlIsImageHandleResidentARB = (PFNGLISTEXTUREHANDLERESIDENTARBPROC) NBL_OGL_LOAD_EXTENSION("glIsImageHandleResidentARB");
	//NV
	pGlGetTextureHandleNV = (PFNGLGETTEXTUREHANDLENVPROC)NBL_OGL_LOAD_EXTENSION("glGetTextureHandleNV");
	pGlGetTextureSamplerHandleNV = (PFNGLGETTEXTURESAMPLERHANDLENVPROC)NBL_OGL_LOAD_EXTENSION("glGetTextureSamplerHandleNV");
	pGlMakeTextureHandleResidentNV = (PFNGLMAKETEXTUREHANDLERESIDENTNVPROC)NBL_OGL_LOAD_EXTENSION("glMakeTextureHandleResidentAR");
	pGlMakeTextureHandleNonResidentNV = (PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC)NBL_OGL_LOAD_EXTENSION("glMakeTextureHandleNonResidentNV");
	pGlGetImageHandleNV = (PFNGLGETIMAGEHANDLENVPROC)NBL_OGL_LOAD_EXTENSION("glGetImageHandleNV");
	pGlMakeImageHandleResidentNV = (PFNGLMAKEIMAGEHANDLERESIDENTNVPROC)NBL_OGL_LOAD_EXTENSION("glMakeImageHandleResidentNV");
	pGlMakeImageHandleNonResidentNV = (PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC)NBL_OGL_LOAD_EXTENSION("glMakeImageHandleNonResidentNV");
	pGlIsTextureHandleResidentNV = (PFNGLISTEXTUREHANDLERESIDENTNVPROC)NBL_OGL_LOAD_EXTENSION("glIsTextureHandleResidentNV");
	pGlIsImageHandleResidentNV = (PFNGLISTEXTUREHANDLERESIDENTNVPROC)NBL_OGL_LOAD_EXTENSION("glIsImageHandleResidentNV");

    //
    pGlBindBufferBase = (PFNGLBINDBUFFERBASEPROC) NBL_OGL_LOAD_EXTENSION("glBindBufferBase");
    pGlBindBufferRange = (PFNGLBINDBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION("glBindBufferRange");
    pGlBindBuffersBase = (PFNGLBINDBUFFERSBASEPROC) NBL_OGL_LOAD_EXTENSION("glBindBuffersBase");
    pGlBindBuffersRange = (PFNGLBINDBUFFERSRANGEPROC) NBL_OGL_LOAD_EXTENSION("glBindBuffersRange");

	// get fragment and vertex program function pointers
	pGlCreateShader = (PFNGLCREATESHADERPROC) NBL_OGL_LOAD_EXTENSION("glCreateShader");
    pGlCreateShaderProgramv = (PFNGLCREATESHADERPROGRAMVPROC) NBL_OGL_LOAD_EXTENSION("glCreateShaderProgramv");
    pGlCreateProgramPipelines = (PFNGLCREATEPROGRAMPIPELINESPROC) NBL_OGL_LOAD_EXTENSION("glCreateProgramPipelines");
    pGlDeleteProgramPipelines = (PFNGLDELETEPROGRAMPIPELINESPROC) NBL_OGL_LOAD_EXTENSION("glDeleteProgramPipelines");
    pGlUseProgramStages = (PFNGLUSEPROGRAMSTAGESPROC)NBL_OGL_LOAD_EXTENSION("glUseProgramStages");
	pGlShaderSource = (PFNGLSHADERSOURCEPROC) NBL_OGL_LOAD_EXTENSION("glShaderSource");
	pGlCompileShader = (PFNGLCOMPILESHADERPROC) NBL_OGL_LOAD_EXTENSION("glCompileShader");
	pGlCreateProgram = (PFNGLCREATEPROGRAMPROC) NBL_OGL_LOAD_EXTENSION("glCreateProgram");
	pGlAttachShader = (PFNGLATTACHSHADERPROC) NBL_OGL_LOAD_EXTENSION("glAttachShader");
	pGlTransformFeedbackVaryings = (PFNGLTRANSFORMFEEDBACKVARYINGSPROC) NBL_OGL_LOAD_EXTENSION("glTransformFeedbackVaryings");
	pGlLinkProgram = (PFNGLLINKPROGRAMPROC) NBL_OGL_LOAD_EXTENSION("glLinkProgram");
	pGlUseProgram = (PFNGLUSEPROGRAMPROC) NBL_OGL_LOAD_EXTENSION("glUseProgram");
	pGlDeleteProgram = (PFNGLDELETEPROGRAMPROC) NBL_OGL_LOAD_EXTENSION("glDeleteProgram");
	pGlDeleteShader = (PFNGLDELETESHADERPROC) NBL_OGL_LOAD_EXTENSION("glDeleteShader");
	pGlGetAttachedShaders = (PFNGLGETATTACHEDSHADERSPROC) NBL_OGL_LOAD_EXTENSION("glGetAttachedShaders");
	pGlGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC) NBL_OGL_LOAD_EXTENSION("glGetShaderInfoLog");
	pGlGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC) NBL_OGL_LOAD_EXTENSION("glGetProgramInfoLog");
	pGlGetShaderiv = (PFNGLGETSHADERIVPROC) NBL_OGL_LOAD_EXTENSION("glGetShaderiv");
	pGlGetProgramiv = (PFNGLGETPROGRAMIVPROC) NBL_OGL_LOAD_EXTENSION("glGetProgramiv");
	pGlGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC) NBL_OGL_LOAD_EXTENSION("glGetUniformLocation");
	pGlProgramUniform1fv = (PFNGLPROGRAMUNIFORM1FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform1fv");
	pGlProgramUniform2fv = (PFNGLPROGRAMUNIFORM2FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform2fv");
	pGlProgramUniform3fv = (PFNGLPROGRAMUNIFORM3FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform3fv");
	pGlProgramUniform4fv = (PFNGLPROGRAMUNIFORM4FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform4fv");
	pGlProgramUniform1iv = (PFNGLPROGRAMUNIFORM1IVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform1iv");
	pGlProgramUniform2iv = (PFNGLPROGRAMUNIFORM2IVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform2iv");
	pGlProgramUniform3iv = (PFNGLPROGRAMUNIFORM3IVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform3iv");
	pGlProgramUniform4iv = (PFNGLPROGRAMUNIFORM4IVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform4iv");
	pGlProgramUniform1uiv = (PFNGLPROGRAMUNIFORM1UIVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform1uiv");
	pGlProgramUniform2uiv = (PFNGLPROGRAMUNIFORM2UIVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform2uiv");
	pGlProgramUniform3uiv = (PFNGLPROGRAMUNIFORM3UIVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform3uiv");
	pGlProgramUniform4uiv = (PFNGLPROGRAMUNIFORM4UIVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniform4uiv");
	pGlProgramUniformMatrix2fv = (PFNGLPROGRAMUNIFORMMATRIX2FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix2fv");
	pGlProgramUniformMatrix3fv = (PFNGLPROGRAMUNIFORMMATRIX3FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix3fv");
	pGlProgramUniformMatrix4fv = (PFNGLPROGRAMUNIFORMMATRIX4FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix4fv");
	pGlProgramUniformMatrix2x3fv = (PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix2x3fv");
	pGlProgramUniformMatrix3x2fv = (PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix3x2fv");
	pGlProgramUniformMatrix4x2fv = (PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix4x2fv");
	pGlProgramUniformMatrix2x4fv = (PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix2x4fv");
	pGlProgramUniformMatrix3x4fv = (PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix3x4fv");
	pGlProgramUniformMatrix4x3fv = (PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC) NBL_OGL_LOAD_EXTENSION("glProgramUniformMatrix4x3fv");
	pGlGetActiveUniform = (PFNGLGETACTIVEUNIFORMPROC) NBL_OGL_LOAD_EXTENSION("glGetActiveUniform");
    pGlBindProgramPipeline = (PFNGLBINDPROGRAMPIPELINEPROC) NBL_OGL_LOAD_EXTENSION("glBindProgramPipeline");
    pGlGetProgramBinary = (PFNGLGETPROGRAMBINARYPROC) NBL_OGL_LOAD_EXTENSION("glGetProgramBinary");
    pGlProgramBinary = (PFNGLPROGRAMBINARYPROC) NBL_OGL_LOAD_EXTENSION("glProgramBinary");

	//Criss
	pGlMemoryBarrier = (PFNGLMEMORYBARRIERPROC) NBL_OGL_LOAD_EXTENSION("glMemoryBarrier");
	pGlDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC) NBL_OGL_LOAD_EXTENSION("glDispatchCompute");
	pGlDispatchComputeIndirect = (PFNGLDISPATCHCOMPUTEINDIRECTPROC) NBL_OGL_LOAD_EXTENSION("glDispatchComputeIndirect");

	// get point parameter extension
	pGlPointParameterf = (PFNGLPOINTPARAMETERFARBPROC) NBL_OGL_LOAD_EXTENSION("glPointParameterf");
	pGlPointParameterfv = (PFNGLPOINTPARAMETERFVARBPROC) NBL_OGL_LOAD_EXTENSION("glPointParameterfv");

    //ROP
	pGlBlendColor = (PFNGLBLENDCOLORPROC)NBL_OGL_LOAD_EXTENSION("glBlendColor");
    pGlDepthRangeIndexed = (PFNGLDEPTHRANGEINDEXEDPROC) NBL_OGL_LOAD_EXTENSION("glDepthRangeIndexed");
    pGlViewportIndexedfv = (PFNGLVIEWPORTINDEXEDFVPROC) NBL_OGL_LOAD_EXTENSION("glViewportIndexedfv");
    pGlScissorIndexedv = (PFNGLSCISSORINDEXEDVPROC) NBL_OGL_LOAD_EXTENSION("glScissorIndexedv");
    pGlSampleCoverage = (PFNGLSAMPLECOVERAGEPROC) NBL_OGL_LOAD_EXTENSION("glSampleCoverage");
	pGlSampleMaski = (PFNGLSAMPLEMASKIPROC) NBL_OGL_LOAD_EXTENSION("glSampleMaski");
	pGlMinSampleShading = (PFNGLMINSAMPLESHADINGPROC) NBL_OGL_LOAD_EXTENSION("glMinSampleShading");
    pGlBlendEquationSeparatei = (PFNGLBLENDEQUATIONSEPARATEIPROC) NBL_OGL_LOAD_EXTENSION("glBlendEquationSeparatei");
    pGlBlendFuncSeparatei = (PFNGLBLENDFUNCSEPARATEIPROC) NBL_OGL_LOAD_EXTENSION("glBlendFuncSeparatei");
    pGlColorMaski = (PFNGLCOLORMASKIPROC) NBL_OGL_LOAD_EXTENSION("glColorMaski");
	pGlStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC) NBL_OGL_LOAD_EXTENSION("glStencilFuncSeparate");
	pGlStencilOpSeparate = (PFNGLSTENCILOPSEPARATEPROC) NBL_OGL_LOAD_EXTENSION("glStencilOpSeparate");
	pGlStencilMaskSeparate = (PFNGLSTENCILMASKSEPARATEPROC) NBL_OGL_LOAD_EXTENSION("glStencilMaskSeparate");

	// ARB FrameBufferObjects
	pGlBlitNamedFramebuffer = (PFNGLBLITNAMEDFRAMEBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glBlitNamedFramebuffer");
	pGlBlitFramebuffer = (PFNGLBLITFRAMEBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glBlitFramebuffer");
	pGlDeleteFramebuffers = (PFNGLDELETEFRAMEBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glDeleteFramebuffers");
	pGlCreateFramebuffers = (PFNGLCREATEFRAMEBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glCreateFramebuffers");
	pGlGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glGenFramebuffers");
	pGlBindFramebuffer = (PFNGLBINDFRAMEBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glBindFramebuffer");
	pGlCheckFramebufferStatus = (PFNGLCHECKFRAMEBUFFERSTATUSPROC) NBL_OGL_LOAD_EXTENSION("glCheckFramebufferStatus");
	pGlCheckNamedFramebufferStatus = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC) NBL_OGL_LOAD_EXTENSION("glCheckNamedFramebufferStatus");
	pGlCheckNamedFramebufferStatusEXT = (PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC) NBL_OGL_LOAD_EXTENSION("glCheckNamedFramebufferStatusEXT");
	pGlFramebufferTexture = (PFNGLFRAMEBUFFERTEXTUREPROC) NBL_OGL_LOAD_EXTENSION("glFramebufferTexture");
	pGlNamedFramebufferTexture = (PFNGLNAMEDFRAMEBUFFERTEXTUREPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferTexture");
	pGlNamedFramebufferTextureEXT = (PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferTextureEXT");
	pGlFramebufferTextureLayer = (PFNGLFRAMEBUFFERTEXTURELAYERPROC) NBL_OGL_LOAD_EXTENSION("glFramebufferTextureLayer");
	pGlNamedFramebufferTextureLayer = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferTextureLayer");
	pGlNamedFramebufferTextureLayerEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferTextureLayerEXT");
	pGlFramebufferTexture2D = (PFNGLFRAMEBUFFERTEXTURE2DPROC)NBL_OGL_LOAD_EXTENSION("glFramebufferTexture2D");
	pGlNamedFramebufferTexture2DEXT = (PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC)NBL_OGL_LOAD_EXTENSION("glNamedFramebufferTexture2DEXT");
	pGlDrawBuffers = (PFNGLDRAWBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glDrawBuffers");
	pGlNamedFramebufferDrawBuffers = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferDrawBuffers");
	pGlFramebufferDrawBuffersEXT = (PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC) NBL_OGL_LOAD_EXTENSION("glFramebufferDrawBuffersEXT");
	pGlNamedFramebufferDrawBuffer = (PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferDrawBuffer");
	pGlFramebufferDrawBufferEXT = (PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC) NBL_OGL_LOAD_EXTENSION("glFramebufferDrawBufferEXT");
	pGlNamedFramebufferReadBuffer = (PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glNamedFramebufferReadBuffer");
	pGlFramebufferReadBufferEXT = (PFNGLFRAMEBUFFERREADBUFFEREXTPROC) NBL_OGL_LOAD_EXTENSION("glFramebufferReadBufferEXT");
    pGlClearNamedFramebufferiv = (PFNGLCLEARNAMEDFRAMEBUFFERIVPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedFramebufferiv");
    pGlClearNamedFramebufferuiv = (PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedFramebufferuiv");
    pGlClearNamedFramebufferfv = (PFNGLCLEARNAMEDFRAMEBUFFERFVPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedFramebufferfv");
    pGlClearNamedFramebufferfi = (PFNGLCLEARNAMEDFRAMEBUFFERFIPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedFramebufferfi");
    pGlClearBufferiv = (PFNGLCLEARBUFFERIVPROC) NBL_OGL_LOAD_EXTENSION("glClearBufferiv");
    pGlClearBufferuiv = (PFNGLCLEARBUFFERUIVPROC) NBL_OGL_LOAD_EXTENSION("glClearBufferuiv");
    pGlClearBufferfv = (PFNGLCLEARBUFFERFVPROC) NBL_OGL_LOAD_EXTENSION("glClearBufferfv");
    pGlClearBufferfi = (PFNGLCLEARBUFFERFIPROC) NBL_OGL_LOAD_EXTENSION("glClearBufferfi");

	// get vertex buffer extension
	pGlGenBuffers = (PFNGLGENBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glGenBuffers");
    pGlCreateBuffers = (PFNGLCREATEBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glCreateBuffers");
	pGlBindBuffer = (PFNGLBINDBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glBindBuffer");
	pGlDeleteBuffers = (PFNGLDELETEBUFFERSPROC) NBL_OGL_LOAD_EXTENSION("glDeleteBuffers");
    pGlBufferStorage = (PFNGLBUFFERSTORAGEPROC) NBL_OGL_LOAD_EXTENSION("glBufferStorage");
    pGlNamedBufferStorage = (PFNGLNAMEDBUFFERSTORAGEPROC) NBL_OGL_LOAD_EXTENSION("glNamedBufferStorage");
    pGlNamedBufferStorageEXT = (PFNGLNAMEDBUFFERSTORAGEEXTPROC) NBL_OGL_LOAD_EXTENSION("glNamedBufferStorageEXT");
    pGlBufferSubData = (PFNGLBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glBufferSubData");
    pGlNamedBufferSubData = (PFNGLNAMEDBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glNamedBufferSubData");
    pGlNamedBufferSubDataEXT = (PFNGLNAMEDBUFFERSUBDATAEXTPROC) NBL_OGL_LOAD_EXTENSION("glNamedBufferSubDataEXT");
    pGlGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glGetBufferSubData");
    pGlGetNamedBufferSubData = (PFNGLGETNAMEDBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glGetNamedBufferSubData");
    pGlGetNamedBufferSubDataEXT = (PFNGLGETNAMEDBUFFERSUBDATAEXTPROC) NBL_OGL_LOAD_EXTENSION("glGetNamedBufferSubDataEXT");
    pGlMapBuffer = (PFNGLMAPBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glMapBuffer");
    pGlMapNamedBuffer = (PFNGLMAPNAMEDBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glMapNamedBuffer");
    pGlMapNamedBufferEXT = (PFNGLMAPNAMEDBUFFEREXTPROC) NBL_OGL_LOAD_EXTENSION("glMapNamedBufferEXT");
    pGlMapBufferRange = (PFNGLMAPBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION("glMapBufferRange");
    pGlMapNamedBufferRange = (PFNGLMAPNAMEDBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION("glMapNamedBufferRange");
    pGlMapNamedBufferRangeEXT = (PFNGLMAPNAMEDBUFFERRANGEEXTPROC) NBL_OGL_LOAD_EXTENSION("glMapNamedBufferRangeEXT");
    pGlFlushMappedBufferRange = (PFNGLFLUSHMAPPEDBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION("glFlushMappedBufferRange");
    pGlFlushMappedNamedBufferRange = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION("glFlushMappedNamedBufferRange");
    pGlFlushMappedNamedBufferRangeEXT = (PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC) NBL_OGL_LOAD_EXTENSION("glFlushMappedNamedBufferRangeEXT");
    pGlUnmapBuffer = (PFNGLUNMAPBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glUnmapBuffer");
    pGlUnmapNamedBuffer = (PFNGLUNMAPNAMEDBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glUnmapNamedBuffer");
    pGlUnmapNamedBufferEXT = (PFNGLUNMAPNAMEDBUFFEREXTPROC) NBL_OGL_LOAD_EXTENSION("glUnmapNamedBufferEXT");
    pGlClearBufferData = (PFNGLCLEARBUFFERDATAPROC) NBL_OGL_LOAD_EXTENSION("glClearBufferData");
    pGlClearNamedBufferData = (PFNGLCLEARNAMEDBUFFERDATAPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedBufferData");
    pGlClearNamedBufferDataEXT = (PFNGLCLEARNAMEDBUFFERDATAEXTPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedBufferDataEXT");
    pGlClearBufferSubData = (PFNGLCLEARBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glClearBufferSubData");
    pGlClearNamedBufferSubData = (PFNGLCLEARNAMEDBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedBufferSubData");
    pGlClearNamedBufferSubDataEXT = (PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC) NBL_OGL_LOAD_EXTENSION("glClearNamedBufferSubDataEXT");
    pGlCopyBufferSubData = (PFNGLCOPYBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glCopyBufferSubData");
    pGlCopyNamedBufferSubData = (PFNGLCOPYNAMEDBUFFERSUBDATAPROC) NBL_OGL_LOAD_EXTENSION("glCopyNamedBufferSubData");
    pGlNamedCopyBufferSubDataEXT = (PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC) NBL_OGL_LOAD_EXTENSION("glNamedCopyBufferSubDataEXT");
	pGlIsBuffer = (PFNGLISBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glIsBuffer");
    pGlGetNamedBufferParameteri64v = (PFNGLGETNAMEDBUFFERPARAMETERI64VPROC) NBL_OGL_LOAD_EXTENSION("glGetNamedBufferParameteri64v");
    pGlGetBufferParameteri64v = (PFNGLGETBUFFERPARAMETERI64VPROC) NBL_OGL_LOAD_EXTENSION("glGetBufferParameteri64v");
    pGlGetNamedBufferParameteriv = (PFNGLGETNAMEDBUFFERPARAMETERIVPROC) NBL_OGL_LOAD_EXTENSION("glGetNamedBufferParameteriv");
    pGlGetNamedBufferParameterivEXT = (PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC) NBL_OGL_LOAD_EXTENSION("glGetNamedBufferParameterivEXT");
    pGlGetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIVPROC) NBL_OGL_LOAD_EXTENSION("glGetBufferParameteriv");
	//vao
    pGlGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC) NBL_OGL_LOAD_EXTENSION("glGenVertexArrays");
    pGlCreateVertexArrays = (PFNGLCREATEVERTEXARRAYSPROC) NBL_OGL_LOAD_EXTENSION("glCreateVertexArrays");
    pGlDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC) NBL_OGL_LOAD_EXTENSION("glDeleteVertexArrays");
    pGlBindVertexArray = (PFNGLBINDVERTEXARRAYPROC) NBL_OGL_LOAD_EXTENSION("glBindVertexArray");
    pGlVertexArrayElementBuffer = (PFNGLVERTEXARRAYELEMENTBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayElementBuffer");
    pGlBindVertexBuffer = (PFNGLBINDVERTEXBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glBindVertexBuffer");
    pGlVertexArrayVertexBuffer = (PFNGLVERTEXARRAYVERTEXBUFFERPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayVertexBuffer");
    pGlVertexArrayBindVertexBufferEXT = (PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayBindVertexBufferEXT");
    pGlVertexAttribBinding = (PFNGLVERTEXATTRIBBINDINGPROC) NBL_OGL_LOAD_EXTENSION("glVertexAttribBinding");
    pGlVertexArrayAttribBinding = (PFNGLVERTEXARRAYATTRIBBINDINGPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayAttribBinding");
    pGlVertexArrayVertexAttribBindingEXT = (PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribBindingEXT");
    pGlEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC) NBL_OGL_LOAD_EXTENSION("glEnableVertexAttribArray");
    pGlEnableVertexArrayAttrib = (PFNGLENABLEVERTEXARRAYATTRIBPROC) NBL_OGL_LOAD_EXTENSION("glEnableVertexArrayAttrib");
    pGlEnableVertexArrayAttribEXT = (PFNGLENABLEVERTEXARRAYATTRIBEXTPROC) NBL_OGL_LOAD_EXTENSION("glEnableVertexArrayAttribEXT");
    pGlDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC) NBL_OGL_LOAD_EXTENSION("glDisableVertexAttribArray");
    pGlDisableVertexArrayAttrib = (PFNGLDISABLEVERTEXARRAYATTRIBPROC) NBL_OGL_LOAD_EXTENSION("glDisableVertexArrayAttrib");
    pGlDisableVertexArrayAttribEXT = (PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC) NBL_OGL_LOAD_EXTENSION("glDisableVertexArrayAttribEXT");
    pGlVertexAttribFormat = (PFNGLVERTEXATTRIBFORMATPROC) NBL_OGL_LOAD_EXTENSION("glVertexAttribFormat");
    pGlVertexAttribIFormat = (PFNGLVERTEXATTRIBIFORMATPROC) NBL_OGL_LOAD_EXTENSION("glVertexAttribIFormat");
    pGlVertexAttribLFormat = (PFNGLVERTEXATTRIBLFORMATPROC) NBL_OGL_LOAD_EXTENSION("glVertexAttribLFormat");
    pGlVertexArrayAttribFormat = (PFNGLVERTEXARRAYATTRIBFORMATPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayAttribFormat");
    pGlVertexArrayAttribIFormat = (PFNGLVERTEXARRAYATTRIBIFORMATPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayAttribIFormat");
    pGlVertexArrayAttribLFormat = (PFNGLVERTEXARRAYATTRIBLFORMATPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayAttribLFormat");
    pGlVertexArrayVertexAttribFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribFormatEXT");
    pGlVertexArrayVertexAttribIFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribIFormatEXT");
    pGlVertexArrayVertexAttribLFormatEXT = (PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayVertexAttribLFormatEXT");
    pGlVertexArrayBindingDivisor = (PFNGLVERTEXARRAYBINDINGDIVISORPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayBindingDivisor");
    pGlVertexArrayVertexBindingDivisorEXT = (PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC) NBL_OGL_LOAD_EXTENSION("glVertexArrayVertexBindingDivisorEXT");
    pGlVertexBindingDivisor = (PFNGLVERTEXBINDINGDIVISORPROC) NBL_OGL_LOAD_EXTENSION("glVertexBindingDivisor");
    //
    pGlPrimitiveRestartIndex = (PFNGLPRIMITIVERESTARTINDEXPROC) NBL_OGL_LOAD_EXTENSION("glPrimitiveRestartIndex");
    pGlDrawArraysInstanced = (PFNGLDRAWARRAYSINSTANCEDPROC) NBL_OGL_LOAD_EXTENSION("glDrawArraysInstanced");
    pGlDrawArraysInstancedBaseInstance = (PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC) NBL_OGL_LOAD_EXTENSION("glDrawArraysInstancedBaseInstance");
    pGlDrawElementsInstancedBaseVertex = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC) NBL_OGL_LOAD_EXTENSION("glDrawElementsInstancedBaseVertex");
    pGlDrawElementsInstancedBaseVertexBaseInstance = (PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC) NBL_OGL_LOAD_EXTENSION("glDrawElementsInstancedBaseVertexBaseInstance");
    pGlDrawTransformFeedback = (PFNGLDRAWTRANSFORMFEEDBACKPROC) NBL_OGL_LOAD_EXTENSION("glDrawTransformFeedback");
    pGlDrawTransformFeedbackInstanced = (PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC) NBL_OGL_LOAD_EXTENSION("glDrawTransformFeedbackInstanced");
    pGlDrawTransformFeedbackStream = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC) NBL_OGL_LOAD_EXTENSION("glDrawTransformFeedbackStream");
    pGlDrawTransformFeedbackStreamInstanced = (PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC) NBL_OGL_LOAD_EXTENSION("glDrawTransformFeedbackStreamInstanced");
    pGlDrawArraysIndirect = (PFNGLDRAWARRAYSINDIRECTPROC) NBL_OGL_LOAD_EXTENSION("glDrawArraysIndirect");
    pGlDrawElementsIndirect = (PFNGLDRAWELEMENTSINDIRECTPROC) NBL_OGL_LOAD_EXTENSION("glDrawElementsIndirect");
    pGlMultiDrawArraysIndirect = (PFNGLMULTIDRAWARRAYSINDIRECTPROC) NBL_OGL_LOAD_EXTENSION("glMultiDrawArraysIndirect");
    pGlMultiDrawElementsIndirect = (PFNGLMULTIDRAWELEMENTSINDIRECTPROC) NBL_OGL_LOAD_EXTENSION("glMultiDrawElementsIndirect");
    if (Version >= 460)
    {
        pGlMultiDrawArrysIndirectCount = (PFNGLMULTIDRAWARRAYSINDIRECTCOUNTPROC) NBL_OGL_LOAD_EXTENSION("glMultiDrawArraysIndirectCount");
        pGlMultiDrawElementsIndirectCount = (PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTPROC) NBL_OGL_LOAD_EXTENSION("glMultiDrawElementsIndirectCount");
    }
    else if (FeatureAvailable[NBL_ARB_indirect_parameters])
    {
        pGlMultiDrawArrysIndirectCount = (PFNGLMULTIDRAWARRAYSINDIRECTCOUNTARBPROC) NBL_OGL_LOAD_EXTENSION("glMultiDrawArraysIndirectCountARB");
        pGlMultiDrawElementsIndirectCount = (PFNGLMULTIDRAWELEMENTSINDIRECTCOUNTARBPROC) NBL_OGL_LOAD_EXTENSION("glMultiDrawElementsIndirectCountARB");
    }
    //
	pGlCreateTransformFeedbacks = (PFNGLCREATETRANSFORMFEEDBACKSPROC) NBL_OGL_LOAD_EXTENSION("glCreateTransformFeedbacks");
	pGlGenTransformFeedbacks = (PFNGLGENTRANSFORMFEEDBACKSPROC) NBL_OGL_LOAD_EXTENSION("glGenTransformFeedbacks");
	pGlDeleteTransformFeedbacks = (PFNGLDELETETRANSFORMFEEDBACKSPROC) NBL_OGL_LOAD_EXTENSION("glDeleteTransformFeedbacks");
	pGlBindTransformFeedback = (PFNGLBINDTRANSFORMFEEDBACKPROC) NBL_OGL_LOAD_EXTENSION("glBindTransformFeedback");
	pGlBeginTransformFeedback = (PFNGLBEGINTRANSFORMFEEDBACKPROC) NBL_OGL_LOAD_EXTENSION("glBeginTransformFeedback");
	pGlPauseTransformFeedback = (PFNGLPAUSETRANSFORMFEEDBACKPROC) NBL_OGL_LOAD_EXTENSION("glPauseTransformFeedback");
	pGlResumeTransformFeedback = (PFNGLRESUMETRANSFORMFEEDBACKPROC) NBL_OGL_LOAD_EXTENSION("glResumeTransformFeedback");
	pGlEndTransformFeedback = (PFNGLENDTRANSFORMFEEDBACKPROC) NBL_OGL_LOAD_EXTENSION("glEndTransformFeedback");
	pGlTransformFeedbackBufferBase = (PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC) NBL_OGL_LOAD_EXTENSION("glTransformFeedbackBufferBase");
	pGlTransformFeedbackBufferRange = (PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC) NBL_OGL_LOAD_EXTENSION("glTransformFeedbackBufferRange");
	//
	pGlBlendFuncSeparate = (PFNGLBLENDFUNCSEPARATEPROC) NBL_OGL_LOAD_EXTENSION("glBlendFuncSeparate");
	pGlEnablei = (PFNGLENABLEIPROC) NBL_OGL_LOAD_EXTENSION("glEnablei");
	pGlDisablei = (PFNGLDISABLEIPROC) NBL_OGL_LOAD_EXTENSION("glDisablei");
	pGlBlendFuncIndexedAMD= (PFNGLBLENDFUNCINDEXEDAMDPROC) NBL_OGL_LOAD_EXTENSION("glBlendFuncIndexedAMD");
	pGlBlendFunciARB= (PFNGLBLENDFUNCIPROC) NBL_OGL_LOAD_EXTENSION("glBlendFunciARB");
	pGlBlendEquationIndexedAMD= (PFNGLBLENDEQUATIONINDEXEDAMDPROC) NBL_OGL_LOAD_EXTENSION("glBlendEquationIndexedAMD");
	pGlBlendEquationiARB= (PFNGLBLENDEQUATIONIPROC) NBL_OGL_LOAD_EXTENSION("glBlendEquationiARB");
	pGlProgramParameteri= (PFNGLPROGRAMPARAMETERIPROC) NBL_OGL_LOAD_EXTENSION("glProgramParameteri");
	pGlPatchParameterfv = (PFNGLPATCHPARAMETERFVPROC) NBL_OGL_LOAD_EXTENSION("glPatchParameterfv");
	pGlPatchParameteri = (PFNGLPATCHPARAMETERIPROC) NBL_OGL_LOAD_EXTENSION("glPatchParameteri");

	// occlusion query
	pGlCreateQueries = (PFNGLCREATEQUERIESPROC) NBL_OGL_LOAD_EXTENSION("glCreateQueries");
	pGlGenQueries = (PFNGLGENQUERIESPROC) NBL_OGL_LOAD_EXTENSION("glGenQueries");
	pGlDeleteQueries = (PFNGLDELETEQUERIESPROC) NBL_OGL_LOAD_EXTENSION("glDeleteQueries");
	pGlIsQuery = (PFNGLISQUERYPROC) NBL_OGL_LOAD_EXTENSION("glIsQuery");
	pGlBeginQuery = (PFNGLBEGINQUERYPROC) NBL_OGL_LOAD_EXTENSION("glBeginQuery");
	pGlEndQuery = (PFNGLENDQUERYPROC) NBL_OGL_LOAD_EXTENSION("glEndQuery");
	pGlBeginQueryIndexed = (PFNGLBEGINQUERYINDEXEDPROC) NBL_OGL_LOAD_EXTENSION("glBeginQueryIndexed");
	pGlEndQueryIndexed = (PFNGLENDQUERYINDEXEDPROC) NBL_OGL_LOAD_EXTENSION("glEndQueryIndexed");
	pGlGetQueryiv = (PFNGLGETQUERYIVPROC) NBL_OGL_LOAD_EXTENSION("glGetQueryiv");
	pGlGetQueryObjectuiv = (PFNGLGETQUERYOBJECTUIVPROC) NBL_OGL_LOAD_EXTENSION("glGetQueryObjectuiv");
	pGlGetQueryObjectui64v = (PFNGLGETQUERYOBJECTUI64VPROC) NBL_OGL_LOAD_EXTENSION("glGetQueryObjectui64v");
    pGlGetQueryBufferObjectuiv = (PFNGLGETQUERYBUFFEROBJECTUIVPROC) NBL_OGL_LOAD_EXTENSION("glGetQueryBufferObjectuiv");
    pGlGetQueryBufferObjectui64v = (PFNGLGETQUERYBUFFEROBJECTUI64VPROC) NBL_OGL_LOAD_EXTENSION("glGetQueryBufferObjectui64v");
	pGlQueryCounter = (PFNGLQUERYCOUNTERPROC) NBL_OGL_LOAD_EXTENSION("glQueryCounter");
	pGlBeginConditionalRender = (PFNGLBEGINCONDITIONALRENDERPROC) NBL_OGL_LOAD_EXTENSION("glBeginConditionalRender");
    pGlEndConditionalRender = (PFNGLENDCONDITIONALRENDERPROC) NBL_OGL_LOAD_EXTENSION("glEndConditionalRender");

    if (FeatureAvailable[NBL_ARB_texture_barrier])
        pGlTextureBarrier = (PFNGLTEXTUREBARRIERPROC) NBL_OGL_LOAD_EXTENSION("glTextureBarrier");
    else if (FeatureAvailable[NBL_NV_texture_barrier])
        pGlTextureBarrierNV = (PFNGLTEXTUREBARRIERNVPROC) NBL_OGL_LOAD_EXTENSION("glTextureBarrierNV");


    pGlDebugMessageControl = (PFNGLDEBUGMESSAGECONTROLPROC) NBL_OGL_LOAD_EXTENSION("glDebugMessageControl");
    pGlDebugMessageControlARB = (PFNGLDEBUGMESSAGECONTROLARBPROC) NBL_OGL_LOAD_EXTENSION("glDebugMessageControlARB");
    pGlDebugMessageCallback = (PFNGLDEBUGMESSAGECALLBACKPROC) NBL_OGL_LOAD_EXTENSION("glDebugMessageCallback");
    pGlDebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC) NBL_OGL_LOAD_EXTENSION("glDebugMessageCallbackARB");

	// blend equation
	pGlBlendEquationEXT = (PFNGLBLENDEQUATIONEXTPROC) NBL_OGL_LOAD_EXTENSION("glBlendEquationEXT");
	pGlBlendEquation = (PFNGLBLENDEQUATIONPROC) NBL_OGL_LOAD_EXTENSION("glBlendEquation");

    pGlGetInternalformativ = (PFNGLGETINTERNALFORMATIVPROC) NBL_OGL_LOAD_EXTENSION("glGetInternalformativ");
    pGlGetInternalformati64v = (PFNGLGETINTERNALFORMATI64VPROC) NBL_OGL_LOAD_EXTENSION("glGetInternalformati64v");

	// get vsync extension
	#if defined(WGL_EXT_swap_control) && !defined(_NBL_COMPILE_WITH_SDL_DEVICE_)
		pWglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC) NBL_OGL_LOAD_EXTENSION("wglSwapIntervalEXT");
	#endif
	#if defined(GLX_SGI_swap_control) && !defined(_NBL_COMPILE_WITH_SDL_DEVICE_)
		pGlxSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC)NBL_OGL_LOAD_EXTENSION("glXSwapIntervalSGI");
	#endif
	#if defined(GLX_EXT_swap_control) && !defined(_NBL_COMPILE_WITH_SDL_DEVICE_)
		pGlxSwapIntervalEXT = (PFNGLXSWAPINTERVALEXTPROC)NBL_OGL_LOAD_EXTENSION("glXSwapIntervalEXT");
	#endif
	#if defined(GLX_MESA_swap_control) && !defined(_NBL_COMPILE_WITH_SDL_DEVICE_)
		pGlxSwapIntervalMESA = (PFNGLXSWAPINTERVALMESAPROC)NBL_OGL_LOAD_EXTENSION("glXSwapIntervalMESA");
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

    if (!(FeatureAvailable[NBL_EXT_texture_filter_anisotropic]||Version>=460))
    {
        retval =  false;
        std::string error = "No anisotropic filtering\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[NBL_EXT_texture_compression_s3tc]))
    {
        retval =  false;
        std::string error = "DXTn compression missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[NBL_ARB_buffer_storage]||Version>=440))
    {
        retval =  false;
        std::string error = "GL_ARB_buffer_storage missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!(FeatureAvailable[NBL_ARB_clip_control]||Version>=450))
    {
        retval =  false;
        std::string error = "GL_ARB_clip_control missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

/*
    if (!(FeatureAvailable[NBL_NV_texture_barrier]||FeatureAvailable[NBL_ARB_texture_barrier]||Version>=450))
    {
        retval =  false;
        std::string error = "GL_NV_texture_barrier missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }
*/

    if (!(FeatureAvailable[NBL_ARB_direct_state_access] || FeatureAvailable[NBL_EXT_direct_state_access] || Version>=450))
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

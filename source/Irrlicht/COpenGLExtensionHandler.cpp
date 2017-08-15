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
        return ESCT_SAMPLER_1D;
    case GL_SAMPLER_2D:
        return ESCT_SAMPLER_2D;
    case GL_SAMPLER_3D:
        return ESCT_SAMPLER_3D;
    case GL_SAMPLER_CUBE:
        return ESCT_SAMPLER_CUBE;
    case GL_SAMPLER_1D_SHADOW:
        return ESCT_SAMPLER_1D_SHADOW;
    case GL_SAMPLER_2D_SHADOW:
        return ESCT_SAMPLER_2D_SHADOW;
    case GL_SAMPLER_1D_ARRAY:
        return ESCT_SAMPLER_1D_ARRAY;
    case GL_SAMPLER_2D_ARRAY:
        return ESCT_SAMPLER_2D_ARRAY;
    case GL_SAMPLER_1D_ARRAY_SHADOW:
        return ESCT_SAMPLER_1D_ARRAY_SHADOW;
    case GL_SAMPLER_2D_ARRAY_SHADOW:
        return ESCT_SAMPLER_2D_ARRAY_SHADOW;
    case GL_SAMPLER_2D_MULTISAMPLE:
        return ESCT_SAMPLER_2D_MULTISAMPLE;
    case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        return ESCT_SAMPLER_2D_MULTISAMPLE_ARRAY;
    case GL_SAMPLER_CUBE_SHADOW:
        return ESCT_SAMPLER_CUBE_SHADOW;
    case GL_SAMPLER_BUFFER:
        return ESCT_SAMPLER_BUFFER;
    case GL_SAMPLER_2D_RECT:
        return ESCT_SAMPLER_2D_RECT;
    case GL_SAMPLER_2D_RECT_SHADOW:
        return ESCT_SAMPLER_2D_RECT_SHADOW;
    case GL_INT_SAMPLER_1D:
        return ESCT_INT_SAMPLER_1D;
    case GL_INT_SAMPLER_2D:
        return ESCT_INT_SAMPLER_2D;
    case GL_INT_SAMPLER_3D:
        return ESCT_INT_SAMPLER_3D;
    case GL_INT_SAMPLER_CUBE:
        return ESCT_INT_SAMPLER_CUBE;
    case GL_INT_SAMPLER_1D_ARRAY:
        return ESCT_INT_SAMPLER_1D_ARRAY;
    case GL_INT_SAMPLER_2D_ARRAY:
        return ESCT_INT_SAMPLER_2D_ARRAY;
    case GL_INT_SAMPLER_2D_MULTISAMPLE:
        return ESCT_INT_SAMPLER_2D_MULTISAMPLE;
    case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        return ESCT_INT_SAMPLER_2D_MULTISAMPLE_ARRAY;
    case GL_INT_SAMPLER_BUFFER:
        return ESCT_INT_SAMPLER_BUFFER;
    case GL_UNSIGNED_INT_SAMPLER_1D:
        return ESCT_UINT_SAMPLER_1D;
    case GL_UNSIGNED_INT_SAMPLER_2D:
        return ESCT_UINT_SAMPLER_2D;
    case GL_UNSIGNED_INT_SAMPLER_3D:
        return ESCT_UINT_SAMPLER_3D;
    case GL_UNSIGNED_INT_SAMPLER_CUBE:
        return ESCT_UINT_SAMPLER_CUBE;
    case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
        return ESCT_UINT_SAMPLER_1D_ARRAY;
    case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
        return ESCT_UINT_SAMPLER_2D_ARRAY;
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
        return ESCT_UINT_SAMPLER_2D_MULTISAMPLE;
    case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
        return ESCT_UINT_SAMPLER_2D_MULTISAMPLE_ARRAY;
    case GL_UNSIGNED_INT_SAMPLER_BUFFER:
        return ESCT_UINT_SAMPLER_BUFFER;
    default:
        return ESCT_INVALID_COUNT;
    }

    return ESCT_INVALID_COUNT;
}


uint16_t COpenGLExtensionHandler::Version = 0;
bool COpenGLExtensionHandler::functionsAlreadyLoaded = false;
int32_t COpenGLExtensionHandler::pixelUnpackAlignment = 2;
bool COpenGLExtensionHandler::FeatureAvailable[] = {false};

uint32_t COpenGLExtensionHandler::MaxArrayTextureLayers = 2048;
uint32_t COpenGLExtensionHandler::MaxIndices = 65535;
uint32_t COpenGLExtensionHandler::MaxVertices = 0xffffffffu;
uint32_t COpenGLExtensionHandler::MaxVertexStreams = 1;
uint32_t COpenGLExtensionHandler::MaxXFormFeedbackComponents = 64;
uint32_t COpenGLExtensionHandler::MaxGPUWaitTimeout = 0;
uint32_t COpenGLExtensionHandler::MaxGeometryVerticesOut = 65535;

//uint32_t COpenGLExtensionHandler::MaxXFormFeedbackInterleavedAttributes = GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS;
//uint32_t COpenGLExtensionHandler::MaxXFormFeedbackSeparateAttributes = GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS;

bool COpenGLExtensionHandler::IsIntelGPU = false;

#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
//
PFNGLGETSTRINGIPROC COpenGLExtensionHandler::pGlGetStringi = NULL;

//fences
PFNGLFENCESYNCPROC COpenGLExtensionHandler::pGlFenceSync = NULL;
PFNGLDELETESYNCPROC COpenGLExtensionHandler::pGlDeleteSync = NULL;
PFNGLCLIENTWAITSYNCPROC COpenGLExtensionHandler::pGlClientWaitSync = NULL;
PFNGLWAITSYNCPROC COpenGLExtensionHandler::pGlWaitSync = NULL;

        //textures
PFNGLACTIVETEXTUREPROC COpenGLExtensionHandler::pGlActiveTexture = NULL;
PFNGLBINDTEXTUREUNITPROC COpenGLExtensionHandler::pGlBindTextureUnit = NULL;
PFNGLBINDMULTITEXTUREEXTPROC COpenGLExtensionHandler::pGlBindMultiTextureEXT = NULL;
PFNGLCREATETEXTURESPROC COpenGLExtensionHandler::pGlCreateTextures = NULL;
PFNGLTEXSTORAGE1DPROC COpenGLExtensionHandler::pGlTexStorage1D = NULL;
PFNGLTEXSTORAGE2DPROC COpenGLExtensionHandler::pGlTexStorage2D = NULL;
PFNGLTEXSTORAGE3DPROC COpenGLExtensionHandler::pGlTexStorage3D = NULL;
PFNGLTEXBUFFERPROC COpenGLExtensionHandler::pGlTexBuffer = NULL;
PFNGLTEXBUFFERRANGEPROC COpenGLExtensionHandler::pGlTexBufferRange = NULL;
PFNGLTEXTURESTORAGE1DPROC COpenGLExtensionHandler::pGlTextureStorage1D = NULL;
PFNGLTEXTURESTORAGE2DPROC COpenGLExtensionHandler::pGlTextureStorage2D = NULL;
PFNGLTEXTURESTORAGE3DPROC COpenGLExtensionHandler::pGlTextureStorage3D = NULL;
PFNGLTEXTUREBUFFERPROC COpenGLExtensionHandler::pGlTextureBuffer = NULL;
PFNGLTEXTUREBUFFERRANGEPROC COpenGLExtensionHandler::pGlTextureBufferRange = NULL;
PFNGLTEXTURESTORAGE1DEXTPROC COpenGLExtensionHandler::pGlTextureStorage1DEXT = NULL;
PFNGLTEXTURESTORAGE2DEXTPROC COpenGLExtensionHandler::pGlTextureStorage2DEXT = NULL;
PFNGLTEXTURESTORAGE3DEXTPROC COpenGLExtensionHandler::pGlTextureStorage3DEXT = NULL;
PFNGLTEXTUREBUFFEREXTPROC COpenGLExtensionHandler::pGlTextureBufferEXT = NULL;
PFNGLTEXTUREBUFFERRANGEEXTPROC COpenGLExtensionHandler::pGlTextureBufferRangeEXT = NULL;
        ///static PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC COpenGLExtensionHandler::pGlTextureStorage2DMultisample = NULL;
        ///static PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC COpenGLExtensionHandler::pGlTextureStorage3DMultisample = NULL;
PFNGLTEXSUBIMAGE3DPROC COpenGLExtensionHandler::pGlTexSubImage3D = NULL;
PFNGLTEXTURESUBIMAGE1DPROC COpenGLExtensionHandler::pGlTextureSubImage1D = NULL;
PFNGLTEXTURESUBIMAGE2DPROC COpenGLExtensionHandler::pGlTextureSubImage2D = NULL;
PFNGLTEXTURESUBIMAGE3DPROC COpenGLExtensionHandler::pGlTextureSubImage3D = NULL;
PFNGLTEXTURESUBIMAGE1DEXTPROC COpenGLExtensionHandler::pGlTextureSubImage1DEXT = NULL;
PFNGLTEXTURESUBIMAGE2DEXTPROC COpenGLExtensionHandler::pGlTextureSubImage2DEXT = NULL;
PFNGLTEXTURESUBIMAGE3DEXTPROC COpenGLExtensionHandler::pGlTextureSubImage3DEXT = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC COpenGLExtensionHandler::pGlCompressedTexSubImage1D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC COpenGLExtensionHandler::pGlCompressedTexSubImage2D = NULL;
PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC COpenGLExtensionHandler::pGlCompressedTexSubImage3D = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE1DPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage1D = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE2DPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage2D = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE3DPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage3D = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage1DEXT = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage2DEXT = NULL;
PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC COpenGLExtensionHandler::pGlCompressedTextureSubImage3DEXT = NULL;
PFNGLGENERATEMIPMAPPROC COpenGLExtensionHandler::pGlGenerateMipmap = NULL;
PFNGLGENERATETEXTUREMIPMAPPROC COpenGLExtensionHandler::pGlGenerateTextureMipmap = NULL;
PFNGLGENERATETEXTUREMIPMAPEXTPROC COpenGLExtensionHandler::pGlGenerateTextureMipmapEXT = NULL;

        //samplers
PFNGLGENSAMPLERSPROC COpenGLExtensionHandler::pGlGenSamplers = NULL;
PFNGLCREATESAMPLERSPROC COpenGLExtensionHandler::pGlCreateSamplers = NULL;
PFNGLDELETESAMPLERSPROC COpenGLExtensionHandler::pGlDeleteSamplers = NULL;
PFNGLBINDSAMPLERPROC COpenGLExtensionHandler::pGlBindSampler = NULL;
PFNGLSAMPLERPARAMETERIPROC COpenGLExtensionHandler::pGlSamplerParameteri = NULL;
PFNGLSAMPLERPARAMETERFPROC COpenGLExtensionHandler::pGlSamplerParameterf = NULL;

        //stuff
PFNGLBINDBUFFERBASEPROC COpenGLExtensionHandler::pGlBindBufferBase = NULL;
PFNGLBINDBUFFERRANGEPROC COpenGLExtensionHandler::pGlBindBufferRange = NULL;


        //shaders
PFNGLBINDATTRIBLOCATIONPROC COpenGLExtensionHandler::pGlBindAttribLocation = NULL;
PFNGLCREATEPROGRAMPROC COpenGLExtensionHandler::pGlCreateProgram = NULL;
PFNGLUSEPROGRAMPROC COpenGLExtensionHandler::pGlUseProgram = NULL;
PFNGLDELETEPROGRAMPROC COpenGLExtensionHandler::pGlDeleteProgram = NULL;
PFNGLDELETESHADERPROC COpenGLExtensionHandler::pGlDeleteShader = NULL;
PFNGLGETATTACHEDSHADERSPROC COpenGLExtensionHandler::pGlGetAttachedShaders = NULL;
PFNGLCREATESHADERPROC COpenGLExtensionHandler::pGlCreateShader = NULL;
PFNGLSHADERSOURCEPROC COpenGLExtensionHandler::pGlShaderSource = NULL;
PFNGLCOMPILESHADERPROC COpenGLExtensionHandler::pGlCompileShader = NULL;
PFNGLATTACHSHADERPROC COpenGLExtensionHandler::pGlAttachShader = NULL;
PFNGLTRANSFORMFEEDBACKVARYINGSPROC COpenGLExtensionHandler::pGlTransformFeedbackVaryings = NULL;
PFNGLLINKPROGRAMPROC COpenGLExtensionHandler::pGlLinkProgram = NULL;
PFNGLGETSHADERINFOLOGPROC COpenGLExtensionHandler::pGlGetShaderInfoLog = NULL;
PFNGLGETPROGRAMINFOLOGPROC COpenGLExtensionHandler::pGlGetProgramInfoLog = NULL;
PFNGLGETSHADERIVPROC COpenGLExtensionHandler::pGlGetShaderiv = NULL;
PFNGLGETSHADERIVPROC COpenGLExtensionHandler::pGlGetProgramiv = NULL;
PFNGLGETUNIFORMLOCATIONPROC COpenGLExtensionHandler::pGlGetUniformLocation = NULL;
PFNGLUNIFORM1FVPROC COpenGLExtensionHandler::pGlUniform1fv = NULL;
PFNGLUNIFORM2FVPROC COpenGLExtensionHandler::pGlUniform2fv = NULL;
PFNGLUNIFORM3FVPROC COpenGLExtensionHandler::pGlUniform3fv = NULL;
PFNGLUNIFORM4FVPROC COpenGLExtensionHandler::pGlUniform4fv = NULL;
PFNGLUNIFORM1IVPROC COpenGLExtensionHandler::pGlUniform1iv = NULL;
PFNGLUNIFORM2IVPROC COpenGLExtensionHandler::pGlUniform2iv = NULL;
PFNGLUNIFORM3IVPROC COpenGLExtensionHandler::pGlUniform3iv = NULL;
PFNGLUNIFORM4IVPROC COpenGLExtensionHandler::pGlUniform4iv = NULL;
PFNGLUNIFORM1UIVPROC COpenGLExtensionHandler::pGlUniform1uiv = NULL;
PFNGLUNIFORM2UIVPROC COpenGLExtensionHandler::pGlUniform2uiv = NULL;
PFNGLUNIFORM3UIVPROC COpenGLExtensionHandler::pGlUniform3uiv = NULL;
PFNGLUNIFORM4UIVPROC COpenGLExtensionHandler::pGlUniform4uiv = NULL;
PFNGLUNIFORMMATRIX2FVPROC COpenGLExtensionHandler::pGlUniformMatrix2fv = NULL;
PFNGLUNIFORMMATRIX3FVPROC COpenGLExtensionHandler::pGlUniformMatrix3fv = NULL;
PFNGLUNIFORMMATRIX4FVPROC COpenGLExtensionHandler::pGlUniformMatrix4fv = NULL;
PFNGLUNIFORMMATRIX2X3FVPROC COpenGLExtensionHandler::pGlUniformMatrix2x3fv = NULL;
PFNGLUNIFORMMATRIX2X4FVPROC COpenGLExtensionHandler::pGlUniformMatrix2x4fv = NULL;
PFNGLUNIFORMMATRIX3X2FVPROC COpenGLExtensionHandler::pGlUniformMatrix3x2fv = NULL;
PFNGLUNIFORMMATRIX3X4FVPROC COpenGLExtensionHandler::pGlUniformMatrix3x4fv = NULL;
PFNGLUNIFORMMATRIX4X2FVPROC COpenGLExtensionHandler::pGlUniformMatrix4x2fv = NULL;
PFNGLUNIFORMMATRIX4X3FVPROC COpenGLExtensionHandler::pGlUniformMatrix4x3fv = NULL;
PFNGLGETACTIVEUNIFORMPROC COpenGLExtensionHandler::pGlGetActiveUniform = NULL;
PFNGLPOINTPARAMETERFPROC COpenGLExtensionHandler:: pGlPointParameterf = NULL;
PFNGLPOINTPARAMETERFVPROC COpenGLExtensionHandler::pGlPointParameterfv = NULL;
PFNGLSTENCILFUNCSEPARATEPROC COpenGLExtensionHandler::pGlStencilFuncSeparate = NULL;
PFNGLSTENCILOPSEPARATEPROC COpenGLExtensionHandler::pGlStencilOpSeparate = NULL;
PFNGLSTENCILFUNCSEPARATEATIPROC COpenGLExtensionHandler::pGlStencilFuncSeparateATI = NULL;
PFNGLSTENCILOPSEPARATEATIPROC COpenGLExtensionHandler::pGlStencilOpSeparateATI = NULL;
		// ARB framebuffer object
PFNGLBLITNAMEDFRAMEBUFFERPROC COpenGLExtensionHandler::pGlBlitNamedFramebuffer = NULL;
PFNGLBLITFRAMEBUFFERPROC COpenGLExtensionHandler::pGlBlitFramebuffer = NULL;
PFNGLDELETEFRAMEBUFFERSPROC COpenGLExtensionHandler::pGlDeleteFramebuffers = NULL;
PFNGLCREATEFRAMEBUFFERSPROC COpenGLExtensionHandler::pGlCreateFramebuffers = NULL;
PFNGLGENFRAMEBUFFERSPROC COpenGLExtensionHandler::pGlGenFramebuffers = NULL;
PFNGLBINDFRAMEBUFFERPROC COpenGLExtensionHandler::pGlBindFramebuffer = NULL;
PFNGLCHECKFRAMEBUFFERSTATUSPROC COpenGLExtensionHandler::pGlCheckFramebufferStatus = NULL;
PFNGLCHECKNAMEDFRAMEBUFFERSTATUSPROC COpenGLExtensionHandler::pGlCheckNamedFramebufferStatus = NULL;
PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC COpenGLExtensionHandler::pGlCheckNamedFramebufferStatusEXT = NULL;
PFNGLFRAMEBUFFERTEXTUREPROC COpenGLExtensionHandler::pGlFramebufferTexture = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTUREPROC COpenGLExtensionHandler::pGlNamedFramebufferTexture = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC COpenGLExtensionHandler::pGlNamedFramebufferTextureEXT = NULL;
PFNGLFRAMEBUFFERTEXTURELAYERPROC COpenGLExtensionHandler::pGlFramebufferTextureLayer = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURELAYERPROC COpenGLExtensionHandler::pGlNamedFramebufferTextureLayer = NULL;
PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC COpenGLExtensionHandler::pGlNamedFramebufferTextureLayerEXT = NULL;
PFNGLDELETERENDERBUFFERSPROC COpenGLExtensionHandler::pGlDeleteRenderbuffers = NULL;
PFNGLGENRENDERBUFFERSPROC COpenGLExtensionHandler::pGlGenRenderbuffers = NULL;
PFNGLCREATERENDERBUFFERSPROC COpenGLExtensionHandler::pGlCreateRenderbuffers = NULL;
PFNGLBINDRENDERBUFFERPROC COpenGLExtensionHandler::pGlBindRenderbuffer = NULL;
PFNGLRENDERBUFFERSTORAGEPROC COpenGLExtensionHandler::pGlRenderbufferStorage = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEPROC COpenGLExtensionHandler::pGlNamedRenderbufferStorage = NULL;
PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC COpenGLExtensionHandler::pGlNamedRenderbufferStorageEXT = NULL;
PFNGLFRAMEBUFFERRENDERBUFFERPROC COpenGLExtensionHandler::pGlFramebufferRenderbuffer = NULL;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC COpenGLExtensionHandler::pGlNamedFramebufferRenderbuffer = NULL;
PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC COpenGLExtensionHandler::pGlNamedFramebufferRenderbufferEXT = NULL;
		// EXT framebuffer object
PFNGLACTIVESTENCILFACEEXTPROC COpenGLExtensionHandler::pGlActiveStencilFaceEXT = NULL;
PFNGLNAMEDFRAMEBUFFERREADBUFFERPROC COpenGLExtensionHandler::pGlNamedFramebufferReadBuffer = NULL;
PFNGLFRAMEBUFFERREADBUFFEREXTPROC COpenGLExtensionHandler::pGlFramebufferReadBufferEXT = NULL;
PFNGLNAMEDFRAMEBUFFERDRAWBUFFERPROC COpenGLExtensionHandler::pGlNamedFramebufferDrawBuffer = NULL;
PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC COpenGLExtensionHandler::pGlFramebufferDrawBufferEXT = NULL;
PFNGLDRAWBUFFERSPROC COpenGLExtensionHandler::pGlDrawBuffers = NULL;
PFNGLNAMEDFRAMEBUFFERDRAWBUFFERSPROC COpenGLExtensionHandler::pGlNamedFramebufferDrawBuffers = NULL;
PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC COpenGLExtensionHandler::pGlFramebufferDrawBuffersEXT = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERIVPROC COpenGLExtensionHandler::pGlClearNamedFramebufferiv = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERUIVPROC COpenGLExtensionHandler::pGlClearNamedFramebufferuiv = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERFVPROC COpenGLExtensionHandler::pGlClearNamedFramebufferfv = NULL;
PFNGLCLEARNAMEDFRAMEBUFFERFIPROC COpenGLExtensionHandler::pGlClearNamedFramebufferfi = NULL;
PFNGLCLEARBUFFERIVPROC COpenGLExtensionHandler::pGlClearBufferiv = NULL;
PFNGLCLEARBUFFERUIVPROC COpenGLExtensionHandler::pGlClearBufferuiv = NULL;
PFNGLCLEARBUFFERFVPROC COpenGLExtensionHandler::pGlClearBufferfv = NULL;
PFNGLCLEARBUFFERFIPROC COpenGLExtensionHandler::pGlClearBufferfi = NULL;

//
PFNGLGENBUFFERSPROC COpenGLExtensionHandler::pGlGenBuffers = NULL;
PFNGLCREATEBUFFERSPROC COpenGLExtensionHandler::pGlCreateBuffers = NULL;
PFNGLBINDBUFFERPROC COpenGLExtensionHandler::pGlBindBuffer = NULL;
PFNGLDELETEBUFFERSPROC COpenGLExtensionHandler::pGlDeleteBuffers = NULL;
PFNGLBUFFERSTORAGEPROC COpenGLExtensionHandler::pGlBufferStorage = NULL;
PFNGLNAMEDBUFFERSTORAGEPROC COpenGLExtensionHandler::pGlNamedBufferStorage = NULL;
PFNGLNAMEDBUFFERSTORAGEEXTPROC COpenGLExtensionHandler::pGlNamedBufferStorageEXT = NULL;
PFNGLBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlBufferSubData = NULL;
PFNGLNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlNamedBufferSubData = NULL;
PFNGLNAMEDBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlNamedBufferSubDataEXT = NULL;
PFNGLGETBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlGetBufferSubData = NULL;
PFNGLGETNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlGetNamedBufferSubData = NULL;
PFNGLGETNAMEDBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlGetNamedBufferSubDataEXT = NULL;
PFNGLMAPBUFFERPROC COpenGLExtensionHandler::pGlMapBuffer = NULL;
PFNGLMAPNAMEDBUFFERPROC COpenGLExtensionHandler::pGlMapNamedBuffer = NULL;
PFNGLMAPNAMEDBUFFEREXTPROC COpenGLExtensionHandler::pGlMapNamedBufferEXT = NULL;
PFNGLMAPBUFFERRANGEPROC COpenGLExtensionHandler::pGlMapBufferRange = NULL;
PFNGLMAPNAMEDBUFFERRANGEPROC COpenGLExtensionHandler::pGlMapNamedBufferRange = NULL;
PFNGLMAPNAMEDBUFFERRANGEEXTPROC COpenGLExtensionHandler::pGlMapNamedBufferRangeEXT = NULL;
PFNGLFLUSHMAPPEDBUFFERRANGEPROC COpenGLExtensionHandler::pGlFlushMappedBufferRange = NULL;
PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEPROC COpenGLExtensionHandler::pGlFlushMappedNamedBufferRange = NULL;
PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC COpenGLExtensionHandler::pGlFlushMappedNamedBufferRangeEXT = NULL;
PFNGLUNMAPBUFFERPROC COpenGLExtensionHandler::pGlUnmapBuffer = NULL;
PFNGLUNMAPNAMEDBUFFERPROC COpenGLExtensionHandler::pGlUnmapNamedBuffer = NULL;
PFNGLUNMAPNAMEDBUFFEREXTPROC COpenGLExtensionHandler::pGlUnmapNamedBufferEXT = NULL;
PFNGLCLEARBUFFERDATAPROC COpenGLExtensionHandler::pGlClearBufferData = NULL;
PFNGLCLEARNAMEDBUFFERDATAPROC COpenGLExtensionHandler::pGlClearNamedBufferData = NULL;
PFNGLCLEARNAMEDBUFFERDATAEXTPROC COpenGLExtensionHandler::pGlClearNamedBufferDataEXT = NULL;
PFNGLCLEARBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlClearBufferSubData = NULL;
PFNGLCLEARNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlClearNamedBufferSubData = NULL;
PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlClearNamedBufferSubDataEXT = NULL;
PFNGLCOPYBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlCopyBufferSubData = NULL;
PFNGLCOPYNAMEDBUFFERSUBDATAPROC COpenGLExtensionHandler::pGlCopyNamedBufferSubData = NULL;
PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC COpenGLExtensionHandler::pGlNamedCopyBufferSubDataEXT = NULL;
PFNGLISBUFFERPROC COpenGLExtensionHandler::pGlIsBuffer = NULL;
//vao
PFNGLGENVERTEXARRAYSPROC COpenGLExtensionHandler::pGlGenVertexArrays = NULL;
PFNGLCREATEVERTEXARRAYSPROC COpenGLExtensionHandler::pGlCreateVertexArrays = NULL;
PFNGLDELETEVERTEXARRAYSPROC COpenGLExtensionHandler::pGlDeleteVertexArrays = NULL;
PFNGLBINDVERTEXARRAYPROC COpenGLExtensionHandler::pGlBindVertexArray = NULL;
PFNGLVERTEXARRAYELEMENTBUFFERPROC COpenGLExtensionHandler::pGlVertexArrayElementBuffer = NULL;
PFNGLBINDVERTEXBUFFERPROC COpenGLExtensionHandler::pGlBindVertexBuffer = NULL;
PFNGLVERTEXARRAYVERTEXBUFFERPROC COpenGLExtensionHandler::pGlVertexArrayVertexBuffer = NULL;
PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC COpenGLExtensionHandler::pGlVertexArrayBindVertexBufferEXT = NULL;
PFNGLVERTEXATTRIBBINDINGPROC COpenGLExtensionHandler::pGlVertexAttribBinding = NULL;
PFNGLVERTEXARRAYATTRIBBINDINGPROC COpenGLExtensionHandler::pGlVertexArrayAttribBinding = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribBindingEXT = NULL;
PFNGLENABLEVERTEXATTRIBARRAYPROC COpenGLExtensionHandler::pGlEnableVertexAttribArray = NULL;
PFNGLENABLEVERTEXARRAYATTRIBPROC COpenGLExtensionHandler::pGlEnableVertexArrayAttrib = NULL;
PFNGLENABLEVERTEXARRAYATTRIBEXTPROC COpenGLExtensionHandler::pGlEnableVertexArrayAttribEXT = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYPROC COpenGLExtensionHandler::pGlDisableVertexAttribArray = NULL;
PFNGLDISABLEVERTEXARRAYATTRIBPROC COpenGLExtensionHandler::pGlDisableVertexArrayAttrib = NULL;
PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC COpenGLExtensionHandler::pGlDisableVertexArrayAttribEXT = NULL;
PFNGLVERTEXATTRIBFORMATPROC COpenGLExtensionHandler::pGlVertexAttribFormat = NULL;
PFNGLVERTEXATTRIBIFORMATPROC COpenGLExtensionHandler::pGlVertexAttribIFormat = NULL;
PFNGLVERTEXATTRIBLFORMATPROC COpenGLExtensionHandler::pGlVertexAttribLFormat = NULL;
PFNGLVERTEXARRAYATTRIBFORMATPROC COpenGLExtensionHandler::pGlVertexArrayAttribFormat = NULL;
PFNGLVERTEXARRAYATTRIBIFORMATPROC COpenGLExtensionHandler::pGlVertexArrayAttribIFormat = NULL;
PFNGLVERTEXARRAYATTRIBLFORMATPROC COpenGLExtensionHandler::pGlVertexArrayAttribLFormat = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribFormatEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribIFormatEXT = NULL;
PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexAttribLFormatEXT = NULL;
PFNGLVERTEXARRAYBINDINGDIVISORPROC COpenGLExtensionHandler::pGlVertexArrayBindingDivisor = NULL;
PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC COpenGLExtensionHandler::pGlVertexArrayVertexBindingDivisorEXT = NULL;
PFNGLVERTEXBINDINGDIVISORPROC COpenGLExtensionHandler::pGlVertexBindingDivisor = NULL;
//
PFNGLDRAWARRAYSINSTANCEDPROC COpenGLExtensionHandler::pGlDrawArraysInstanced = NULL;
PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC COpenGLExtensionHandler::pGlDrawArraysInstancedBaseInstance = NULL;
PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC COpenGLExtensionHandler::pGlDrawElementsInstancedBaseVertex = NULL;
PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC COpenGLExtensionHandler::pGlDrawElementsInstancedBaseVertexBaseInstance = NULL;
PFNGLDRAWTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlDrawTransformFeedback = NULL;
PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC COpenGLExtensionHandler::pGlDrawTransformFeedbackInstanced = NULL;
PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC COpenGLExtensionHandler::pGlDrawTransformFeedbackStream = NULL;
PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC COpenGLExtensionHandler::pGlDrawTransformFeedbackStreamInstanced = NULL;
PFNGLDRAWARRAYSINDIRECTPROC COpenGLExtensionHandler::pGlDrawArraysIndirect = NULL;
PFNGLDRAWELEMENTSINDIRECTPROC COpenGLExtensionHandler::pGlDrawElementsIndirect = NULL;
PFNGLMULTIDRAWARRAYSINDIRECTPROC COpenGLExtensionHandler::pGlMultiDrawArraysIndirect = NULL;
PFNGLMULTIDRAWELEMENTSINDIRECTPROC COpenGLExtensionHandler::pGlMultiDrawElementsIndirect = NULL;
//
PFNGLCREATETRANSFORMFEEDBACKSPROC COpenGLExtensionHandler::pGlCreateTransformFeedbacks = NULL;
PFNGLGENTRANSFORMFEEDBACKSPROC COpenGLExtensionHandler::pGlGenTransformFeedbacks = NULL;
PFNGLDELETETRANSFORMFEEDBACKSPROC COpenGLExtensionHandler::pGlDeleteTransformFeedbacks = NULL;
PFNGLBINDTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlBindTransformFeedback = NULL;
PFNGLBEGINTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlBeginTransformFeedback = NULL;
PFNGLPAUSETRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlPauseTransformFeedback = NULL;
PFNGLRESUMETRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlResumeTransformFeedback = NULL;
PFNGLENDTRANSFORMFEEDBACKPROC COpenGLExtensionHandler::pGlEndTransformFeedback = NULL;
PFNGLTRANSFORMFEEDBACKBUFFERBASEPROC COpenGLExtensionHandler::pGlTransformFeedbackBufferBase = NULL;
PFNGLTRANSFORMFEEDBACKBUFFERRANGEPROC COpenGLExtensionHandler::pGlTransformFeedbackBufferRange = NULL;
//
PFNGLBLENDFUNCSEPARATEPROC COpenGLExtensionHandler::pGlBlendFuncSeparate = NULL;
PFNGLPROVOKINGVERTEXPROC COpenGLExtensionHandler::pGlProvokingVertex = NULL;
PFNGLCOLORMASKIPROC COpenGLExtensionHandler::pGlColorMaski = NULL;
PFNGLENABLEIPROC COpenGLExtensionHandler::pGlEnablei = NULL;
PFNGLDISABLEIPROC COpenGLExtensionHandler::pGlDisablei = NULL;
PFNGLBLENDFUNCINDEXEDAMDPROC COpenGLExtensionHandler::pGlBlendFuncIndexedAMD = NULL;
PFNGLBLENDFUNCIPROC COpenGLExtensionHandler::pGlBlendFunciARB = NULL;
PFNGLBLENDEQUATIONINDEXEDAMDPROC COpenGLExtensionHandler::pGlBlendEquationIndexedAMD = NULL;
PFNGLBLENDEQUATIONIPROC COpenGLExtensionHandler::pGlBlendEquationiARB = NULL;
PFNGLPROGRAMPARAMETERIARBPROC COpenGLExtensionHandler::pGlProgramParameteriARB = NULL;
PFNGLPROGRAMPARAMETERIEXTPROC COpenGLExtensionHandler::pGlProgramParameteriEXT = NULL;
PFNGLPATCHPARAMETERIPROC COpenGLExtensionHandler::pGlPatchParameteri = NULL;
PFNGLPATCHPARAMETERFVPROC COpenGLExtensionHandler::pGlPatchParameterfv = NULL;
//
PFNGLCREATEQUERIESPROC COpenGLExtensionHandler::pGlCreateQueries = NULL;
PFNGLGENQUERIESPROC COpenGLExtensionHandler::pGlGenQueries = NULL;
PFNGLDELETEQUERIESPROC COpenGLExtensionHandler::pGlDeleteQueries = NULL;
PFNGLISQUERYPROC COpenGLExtensionHandler::pGlIsQuery = NULL;
PFNGLBEGINQUERYPROC COpenGLExtensionHandler::pGlBeginQuery = NULL;
PFNGLENDQUERYPROC COpenGLExtensionHandler::pGlEndQuery = NULL;
PFNGLBEGINQUERYINDEXEDPROC COpenGLExtensionHandler::pGlBeginQueryIndexed = NULL;
PFNGLENDQUERYINDEXEDPROC COpenGLExtensionHandler::pGlEndQueryIndexed = NULL;
PFNGLGETQUERYIVPROC COpenGLExtensionHandler::pGlGetQueryiv = NULL;
PFNGLGETQUERYOBJECTUIVPROC COpenGLExtensionHandler::pGlGetQueryObjectuiv = NULL;
PFNGLGETQUERYOBJECTUI64VPROC COpenGLExtensionHandler::pGlGetQueryObjectui64v = NULL;
PFNGLGETQUERYBUFFEROBJECTUIVPROC COpenGLExtensionHandler::pGlGetQueryBufferObjectuiv = NULL;
PFNGLGETQUERYBUFFEROBJECTUI64VPROC COpenGLExtensionHandler::pGlGetQueryBufferObjectui64v = NULL;
PFNGLQUERYCOUNTERPROC COpenGLExtensionHandler::pGlQueryCounter = NULL;
PFNGLBEGINCONDITIONALRENDERPROC COpenGLExtensionHandler::pGlBeginConditionalRender = NULL;
PFNGLENDCONDITIONALRENDERPROC COpenGLExtensionHandler::pGlEndConditionalRender = NULL;
//
PFNGLBLENDEQUATIONEXTPROC COpenGLExtensionHandler::pGlBlendEquationEXT = NULL;
PFNGLBLENDEQUATIONPROC COpenGLExtensionHandler::pGlBlendEquation = NULL;

PFNGLDEBUGMESSAGECONTROLPROC COpenGLExtensionHandler::pGlDebugMessageControl = NULL;
PFNGLDEBUGMESSAGECONTROLARBPROC COpenGLExtensionHandler::pGlDebugMessageControlARB = NULL;
PFNGLDEBUGMESSAGECALLBACKPROC COpenGLExtensionHandler::pGlDebugMessageCallback = NULL;
PFNGLDEBUGMESSAGECALLBACKARBPROC COpenGLExtensionHandler::pGlDebugMessageCallbackARB = NULL;

    #if defined(WGL_EXT_swap_control)
        PFNWGLSWAPINTERVALEXTPROC COpenGLExtensionHandler::pWglSwapIntervalEXT = NULL;
    #endif
    #if defined(GLX_SGI_swap_control)
        PFNGLXSWAPINTERVALSGIPROC COpenGLExtensionHandler::pGlxSwapIntervalSGI = NULL;
    #endif
    #if defined(GLX_EXT_swap_control)
        PFNGLXSWAPINTERVALEXTPROC COpenGLExtensionHandler::pGlxSwapIntervalEXT = NULL;
    #endif
    #if defined(GLX_MESA_swap_control)
        PFNGLXSWAPINTERVALMESAPROC COpenGLExtensionHandler::pGlxSwapIntervalMESA = NULL;
    #endif
#endif


core::LeakDebugger COpenGLExtensionHandler::bufferLeaker("GLBuffer");
core::LeakDebugger COpenGLExtensionHandler::textureLeaker("GLTex");



COpenGLExtensionHandler::COpenGLExtensionHandler() :
		StencilBuffer(false),
		TextureCompressionExtension(false),
		MaxTextureUnits(1), MaxLights(1),
		MaxAnisotropy(1), MaxUserClipPlanes(0), MaxAuxBuffers(0),
		MaxMultipleRenderTargets(1),
		MaxTextureLODBias(0.f), ShaderLanguageVersion(0)
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
#ifdef _DEBUG
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
#define tmplog(x,y) os::Printer::log(x, core::longlongtoa(y))
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
	StencilBuffer=stencilBuffer;


	GLint num=0;

	MaxLights=0;

	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS,&num);
	MaxArrayTextureLayers = num;

	if (FeatureAvailable[IRR_EXT_texture_filter_anisotropic])
	{
		glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &num);
		MaxAnisotropy=static_cast<uint8_t>(num);
	}


	if (queryFeature(EVDF_GEOMETRY_SHADER))
	{
		if (FeatureAvailable[IRR_ARB_geometry_shader4]||FeatureAvailable[IRR_EXT_geometry_shader4])
        {
            glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &num);
            MaxGeometryVerticesOut=static_cast<uint32_t>(num);
        }
	}
	if (FeatureAvailable[IRR_EXT_texture_lod_bias])
		glGetFloatv(GL_MAX_TEXTURE_LOD_BIAS_EXT, &MaxTextureLODBias);


	glGetIntegerv(GL_MAX_CLIP_DISTANCES, &num);
	MaxUserClipPlanes=static_cast<uint8_t>(num);
	glGetIntegerv(GL_AUX_BUFFERS, &num);
	MaxAuxBuffers=static_cast<uint8_t>(num);
    glGetIntegerv(GL_MAX_DRAW_BUFFERS_ARB, &num);
    MaxMultipleRenderTargets = static_cast<uint8_t>(num);

	glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, DimAliasedLine);
	glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE, DimAliasedPoint);
	glGetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE, DimSmoothedLine);
	glGetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, DimSmoothedPoint);

    const GLubyte* shaderVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    float sl_ver;
    sscanf(reinterpret_cast<const char*>(shaderVersion),"%f",&sl_ver);
    ShaderLanguageVersion = static_cast<uint16_t>(core::round32(sl_ver*100.0f));

/*
    //! For EXT-DSA testing
    Version = 440;
    FeatureAvailable[IRR_ARB_direct_state_access] = false;
    pGlBindTextureUnit = NULL;
    pGlCreateTextures = NULL;
    pGlTextureStorage1D = NULL;
    pGlTextureStorage2D = NULL;
    pGlTextureStorage3D = NULL;
    pGlTextureSubImage1D = NULL;
    pGlTextureSubImage2D = NULL;
    pGlTextureSubImage3D = NULL;
    pGlCompressedTextureSubImage1D = NULL;
    pGlCompressedTextureSubImage2D = NULL;
    pGlCompressedTextureSubImage3D = NULL;
    pGlGenerateTextureMipmap = NULL;
    pGlCreateSamplers = NULL;
    pGlBindAttribLocation = NULL;
    pGlBlitNamedFramebuffer = NULL;
    pGlCreateFramebuffers = NULL;
    pGlCheckNamedFramebufferStatus = NULL;
    pGlNamedFramebufferTexture = NULL;
    pGlNamedFramebufferTextureLayer = NULL;
    pGlCreateRenderbuffers = NULL;
    pGlNamedRenderbufferStorage = NULL;
    pGlNamedFramebufferRenderbuffer = NULL;
    pGlActiveStencilFaceEXT = NULL;
    pGlNamedFramebufferReadBuffer = NULL;
    pGlNamedFramebufferDrawBuffer = NULL;
    pGlNamedFramebufferDrawBuffers = NULL;
    pGlClearNamedFramebufferiv = NULL;
    pGlClearNamedFramebufferuiv = NULL;
    pGlClearNamedFramebufferfv = NULL;
    pGlClearNamedFramebufferfi = NULL;
    pGlCreateBuffers = NULL;
    pGlNamedBufferStorage = NULL;
    pGlNamedBufferSubData = NULL;
    pGlGetNamedBufferSubData = NULL;
    pGlMapNamedBuffer = NULL;
    pGlMapNamedBufferRange = NULL;
    pGlFlushMappedNamedBufferRange = NULL;
    pGlUnmapNamedBuffer = NULL;
    pGlClearNamedBufferData = NULL;
    pGlClearNamedBufferSubData = NULL;
    pGlCopyNamedBufferSubData = NULL;
    pGlCreateVertexArrays = NULL;
    pGlVertexArrayElementBuffer = NULL;
    pGlVertexArrayVertexBuffer = NULL;
    pGlVertexArrayAttribBinding = NULL;
    pGlEnableVertexArrayAttrib = NULL;
    pGlDisableVertexArrayAttrib = NULL;
    pGlVertexArrayAttribFormat = NULL;
    pGlVertexArrayAttribIFormat = NULL;
    pGlVertexArrayAttribLFormat = NULL;
    pGlVertexArrayBindingDivisor = NULL;
    pGlBlendFuncIndexedAMD = NULL;
    pGlBlendEquationIndexedAMD = NULL;
    pGlBlendEquationiARB = NULL;
    //! Non-DSA testing
    Version = 430;
    FeatureAvailable[IRR_EXT_direct_state_access] = FeatureAvailable[IRR_ARB_direct_state_access] = false;
    pGlBindMultiTextureEXT = NULL;
    pGlTextureStorage1DEXT = NULL;
    pGlTextureStorage2DEXT = NULL;
    pGlTextureStorage3DEXT = NULL;
    pGlTextureSubImage1DEXT = NULL;
    pGlTextureSubImage2DEXT = NULL;
    pGlTextureSubImage3DEXT = NULL;
    pGlCompressedTextureSubImage1DEXT = NULL;
    pGlCompressedTextureSubImage2DEXT = NULL;
    pGlCompressedTextureSubImage3DEXT = NULL;
    pGlGenerateTextureMipmapEXT = NULL;
    pGlCheckNamedFramebufferStatusEXT = NULL;
    pGlNamedFramebufferTextureEXT = NULL;
    pGlNamedFramebufferTextureLayerEXT = NULL;
    pGlNamedRenderbufferStorageEXT = NULL;
    pGlNamedFramebufferRenderbufferEXT = NULL;
    pGlFramebufferReadBufferEXT = NULL;
    pGlFramebufferDrawBufferEXT = NULL;
    pGlFramebufferDrawBuffersEXT = NULL;
    pGlNamedBufferStorageEXT = NULL;
    pGlNamedBufferSubDataEXT = NULL;
    pGlGetNamedBufferSubDataEXT = NULL;
    pGlMapNamedBufferEXT = NULL;
    pGlMapNamedBufferRangeEXT = NULL;
    pGlFlushMappedNamedBufferRangeEXT = NULL;
    pGlUnmapNamedBufferEXT = NULL;
    pGlClearNamedBufferDataEXT = NULL;
    pGlClearNamedBufferSubDataEXT = NULL;
    pGlNamedCopyBufferSubDataEXT = NULL;
    pGlVertexArrayBindVertexBufferEXT = NULL;
    pGlVertexArrayVertexAttribBindingEXT = NULL;
    pGlEnableVertexArrayAttribEXT = NULL;
    pGlDisableVertexArrayAttribEXT = NULL;
    pGlVertexArrayVertexAttribFormatEXT = NULL;
    pGlVertexArrayVertexAttribIFormatEXT = NULL;
    pGlVertexArrayVertexAttribLFormatEXT = NULL;
    pGlVertexArrayVertexBindingDivisorEXT = NULL;
**/

    num=0;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &num);
	MaxTextureUnits = core::min_(static_cast<uint8_t>(num), static_cast<uint8_t>(MATERIAL_MAX_TEXTURES));

    //num=100000000u;
	//glGetIntegerv(GL_MAX_ELEMENTS_INDICES,&num);

#ifdef _DEBUG
	if (FeatureAvailable[IRR_NVX_gpu_memory_info])
	{
		// undocumented flags, so use the RAW values
		GLint val;
		glGetIntegerv(0x9047, &val);
		os::Printer::log("Dedicated video memory (kB)", core::longlongtoa(val));
		glGetIntegerv(0x9048, &val);
		os::Printer::log("Total video memory (kB)", core::longlongtoa(val));
		glGetIntegerv(0x9049, &val);
		os::Printer::log("Available video memory (kB)", core::longlongtoa(val));
	}
#ifdef GL_ATI_meminfo
	if (FeatureAvailable[IRR_ATI_meminfo])
	{
		GLint val[4];
		glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, val);
		os::Printer::log("Free texture memory (kB)", core::longlongtoa(val[0]));
		glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, val);
		os::Printer::log("Free VBO memory (kB)", core::longlongtoa(val[0]));
		glGetIntegerv(GL_RENDERBUFFER_FREE_MEMORY_ATI, val);
		os::Printer::log("Free render buffer memory (kB)", core::longlongtoa(val[0]));
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


#ifdef _IRR_OPENGL_USE_EXTPOINTER_

#ifdef _IRR_WINDOWS_API_
	#define IRR_OGL_LOAD_EXTENSION(x) wglGetProcAddress(reinterpret_cast<const char*>(x))
#elif defined(_IRR_COMPILE_WITH_SDL_DEVICE_) && !defined(_IRR_COMPILE_WITH_X11_DEVICE_)
	#define IRR_OGL_LOAD_EXTENSION(x) SDL_GL_GetProcAddress(reinterpret_cast<const char*>(x))
#else
    #define IRR_OGL_LOAD_EXTENSION(X) glXGetProcAddress(reinterpret_cast<const GLubyte*>(X))
#endif // Windows, SDL, or Linux

    pGlGetStringi = (PFNGLGETSTRINGIPROC) IRR_OGL_LOAD_EXTENSION("glGetStringi");

#else

    pGlGetStringi = &glGetStringi;

#endif // _IRR_OPENGL_USE_EXTPOINTER_

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


#ifdef _IRR_OPENGL_USE_EXTPOINTER_
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

    //fences
    pGlFenceSync = (PFNGLFENCESYNCPROC) IRR_OGL_LOAD_EXTENSION("glFenceSync");
    pGlDeleteSync = (PFNGLDELETESYNCPROC) IRR_OGL_LOAD_EXTENSION("glDeleteSync");
    pGlClientWaitSync = (PFNGLCLIENTWAITSYNCPROC) IRR_OGL_LOAD_EXTENSION("glClientWaitSync");
    pGlWaitSync = (PFNGLWAITSYNCPROC) IRR_OGL_LOAD_EXTENSION("glWaitSync");

	// get multitexturing function pointers
    pGlActiveTexture = (PFNGLACTIVETEXTUREPROC) IRR_OGL_LOAD_EXTENSION("glActiveTexture");
	pGlBindTextureUnit = (PFNGLBINDTEXTUREUNITPROC) IRR_OGL_LOAD_EXTENSION("glBindTextureUnit");
	pGlBindMultiTextureEXT = (PFNGLBINDMULTITEXTUREEXTPROC) IRR_OGL_LOAD_EXTENSION("glBindMultiTextureEXT");
    pGlCreateTextures = (PFNGLCREATETEXTURESPROC) IRR_OGL_LOAD_EXTENSION("glCreateTextures");
    pGlTexStorage1D = (PFNGLTEXSTORAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage1D");
    pGlTexStorage2D = (PFNGLTEXSTORAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage2D");
    pGlTexStorage3D = (PFNGLTEXSTORAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTexStorage3D");
    pGlTexBuffer = (PFNGLTEXBUFFERPROC) IRR_OGL_LOAD_EXTENSION( "glTexBuffer");
    pGlTexBufferRange = (PFNGLTEXBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION( "glTexBufferRange");
    pGlTextureStorage1D = (PFNGLTEXTURESTORAGE1DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage1D");
    pGlTextureStorage2D = (PFNGLTEXTURESTORAGE2DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2D");
    pGlTextureStorage3D = (PFNGLTEXTURESTORAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3D");
    pGlTextureBuffer = (PFNGLTEXTUREBUFFERPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBuffer");
    pGlTextureBufferRange = (PFNGLTEXTUREBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBufferRange");
    pGlTextureStorage1DEXT = (PFNGLTEXTURESTORAGE1DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage1DEXT");
    pGlTextureStorage2DEXT = (PFNGLTEXTURESTORAGE2DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2DEXT");
    pGlTextureStorage3DEXT = (PFNGLTEXTURESTORAGE3DEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3DEXT");
    pGlTextureBufferEXT = (PFNGLTEXTUREBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBufferEXT");
    pGlTextureBufferRangeEXT = (PFNGLTEXTUREBUFFERRANGEEXTPROC) IRR_OGL_LOAD_EXTENSION( "glTextureBufferRangeEXT");
    ///PFNGLTEXTURESTORAGE2DMULTISAMPLEPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage2DMultisample");
    ///PFNGLTEXTURESTORAGE3DMULTISAMPLEPROC) IRR_OGL_LOAD_EXTENSION( "glTextureStorage3DMultisample");
    pGlTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC) IRR_OGL_LOAD_EXTENSION( "glTexSubImage3D");
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
    pGlGenerateMipmap = (PFNGLGENERATEMIPMAPPROC) IRR_OGL_LOAD_EXTENSION( "glGenerateMipmap");
    pGlGenerateTextureMipmap = (PFNGLGENERATETEXTUREMIPMAPPROC) IRR_OGL_LOAD_EXTENSION( "glGenerateTextureMipmap");
    pGlGenerateTextureMipmapEXT = (PFNGLGENERATETEXTUREMIPMAPEXTPROC) IRR_OGL_LOAD_EXTENSION( "glGenerateTextureMipmapEXT");

    //samplers
    pGlGenSamplers = (PFNGLGENSAMPLERSPROC) IRR_OGL_LOAD_EXTENSION( "glGenSamplers");
    pGlDeleteSamplers = (PFNGLDELETESAMPLERSPROC) IRR_OGL_LOAD_EXTENSION( "glDeleteSamplers");
    pGlBindSampler = (PFNGLBINDSAMPLERPROC) IRR_OGL_LOAD_EXTENSION( "glBindSampler");
    pGlSamplerParameteri = (PFNGLSAMPLERPARAMETERIPROC) IRR_OGL_LOAD_EXTENSION( "glSamplerParameteri");
    pGlSamplerParameterf = (PFNGLSAMPLERPARAMETERFPROC) IRR_OGL_LOAD_EXTENSION( "glSamplerParameterf");

    //
    pGlBindBufferBase = (PFNGLBINDBUFFERBASEPROC) IRR_OGL_LOAD_EXTENSION("glBindBufferBase");
    pGlBindBufferRange = (PFNGLBINDBUFFERRANGEPROC) IRR_OGL_LOAD_EXTENSION("glBindBufferRange");

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
	pGlUniform1fv = (PFNGLUNIFORM1FVPROC) IRR_OGL_LOAD_EXTENSION("glUniform1fv");
	pGlUniform2fv = (PFNGLUNIFORM2FVPROC) IRR_OGL_LOAD_EXTENSION("glUniform2fv");
	pGlUniform3fv = (PFNGLUNIFORM3FVPROC) IRR_OGL_LOAD_EXTENSION("glUniform3fv");
	pGlUniform4fv = (PFNGLUNIFORM4FVPROC) IRR_OGL_LOAD_EXTENSION("glUniform4fv");
	pGlUniform1iv = (PFNGLUNIFORM1IVPROC) IRR_OGL_LOAD_EXTENSION("glUniform1iv");
	pGlUniform2iv = (PFNGLUNIFORM2IVPROC) IRR_OGL_LOAD_EXTENSION("glUniform2iv");
	pGlUniform3iv = (PFNGLUNIFORM3IVPROC) IRR_OGL_LOAD_EXTENSION("glUniform3iv");
	pGlUniform4iv = (PFNGLUNIFORM4IVPROC) IRR_OGL_LOAD_EXTENSION("glUniform4iv");
	pGlUniform1uiv = (PFNGLUNIFORM1UIVPROC) IRR_OGL_LOAD_EXTENSION("glUniform1uiv");
	pGlUniform2uiv = (PFNGLUNIFORM2UIVPROC) IRR_OGL_LOAD_EXTENSION("glUniform2uiv");
	pGlUniform3uiv = (PFNGLUNIFORM3UIVPROC) IRR_OGL_LOAD_EXTENSION("glUniform3uiv");
	pGlUniform4uiv = (PFNGLUNIFORM4UIVPROC) IRR_OGL_LOAD_EXTENSION("glUniform4uiv");
	pGlUniformMatrix2fv = (PFNGLUNIFORMMATRIX2FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix2fv");
	pGlUniformMatrix3fv = (PFNGLUNIFORMMATRIX3FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix3fv");
	pGlUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix4fv");
	pGlUniformMatrix2x3fv = (PFNGLUNIFORMMATRIX2X3FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix2x3fv");
	pGlUniformMatrix3x2fv = (PFNGLUNIFORMMATRIX3X2FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix3x2fv");
	pGlUniformMatrix4x2fv = (PFNGLUNIFORMMATRIX4X2FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix4x2fv");
	pGlUniformMatrix2x4fv = (PFNGLUNIFORMMATRIX2X4FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix2x4fv");
	pGlUniformMatrix3x4fv = (PFNGLUNIFORMMATRIX3X4FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix3x4fv");
	pGlUniformMatrix4x3fv = (PFNGLUNIFORMMATRIX4X3FVPROC) IRR_OGL_LOAD_EXTENSION("glUniformMatrix4x3fv");
	pGlGetActiveUniform = (PFNGLGETACTIVEUNIFORMPROC) IRR_OGL_LOAD_EXTENSION("glGetActiveUniform");

	// get point parameter extension
	pGlPointParameterf = (PFNGLPOINTPARAMETERFARBPROC) IRR_OGL_LOAD_EXTENSION("glPointParameterf");
	pGlPointParameterfv = (PFNGLPOINTPARAMETERFVARBPROC) IRR_OGL_LOAD_EXTENSION("glPointParameterfv");

	// get stencil extension
	pGlStencilFuncSeparate = (PFNGLSTENCILFUNCSEPARATEPROC) IRR_OGL_LOAD_EXTENSION("glStencilFuncSeparate");
	pGlStencilOpSeparate = (PFNGLSTENCILOPSEPARATEPROC) IRR_OGL_LOAD_EXTENSION("glStencilOpSeparate");
	pGlStencilFuncSeparateATI = (PFNGLSTENCILFUNCSEPARATEATIPROC) IRR_OGL_LOAD_EXTENSION("glStencilFuncSeparateATI");
	pGlStencilOpSeparateATI = (PFNGLSTENCILOPSEPARATEATIPROC) IRR_OGL_LOAD_EXTENSION("glStencilOpSeparateATI");

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
	pGlDeleteRenderbuffers = (PFNGLDELETERENDERBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glDeleteRenderbuffers");
	pGlGenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glGenRenderbuffers");
	pGlCreateRenderbuffers = (PFNGLCREATERENDERBUFFERSPROC) IRR_OGL_LOAD_EXTENSION("glCreateRenderbuffers");
	pGlBindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glBindRenderbuffer");
	pGlRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC) IRR_OGL_LOAD_EXTENSION("glRenderbufferStorage");
	pGlNamedRenderbufferStorage = (PFNGLNAMEDRENDERBUFFERSTORAGEPROC) IRR_OGL_LOAD_EXTENSION("glNamedRenderbufferStorage");
	pGlNamedRenderbufferStorageEXT = (PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedRenderbufferStorageEXT");
	pGlFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glFramebufferRenderbuffer");
	pGlNamedFramebufferRenderbuffer = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferRenderbuffer");
	pGlNamedFramebufferRenderbufferEXT = (PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC) IRR_OGL_LOAD_EXTENSION("glNamedFramebufferRenderbufferEXT");
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
	pGlIsBuffer= (PFNGLISBUFFERPROC) IRR_OGL_LOAD_EXTENSION("glIsBuffer");
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
	pGlProvokingVertex = (PFNGLPROVOKINGVERTEXPROC) IRR_OGL_LOAD_EXTENSION("glProvokingVertex");
	pGlColorMaski= (PFNGLCOLORMASKIPROC) IRR_OGL_LOAD_EXTENSION("glColorMaski");
	pGlEnablei = (PFNGLENABLEIPROC) IRR_OGL_LOAD_EXTENSION("glEnablei");
	pGlDisablei = (PFNGLDISABLEIPROC) IRR_OGL_LOAD_EXTENSION("glDisablei");
	pGlBlendFuncIndexedAMD= (PFNGLBLENDFUNCINDEXEDAMDPROC) IRR_OGL_LOAD_EXTENSION("glBlendFuncIndexedAMD");
	pGlBlendFunciARB= (PFNGLBLENDFUNCIPROC) IRR_OGL_LOAD_EXTENSION("glBlendFunciARB");
	pGlBlendEquationIndexedAMD= (PFNGLBLENDEQUATIONINDEXEDAMDPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationIndexedAMD");
	pGlBlendEquationiARB= (PFNGLBLENDEQUATIONIPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationiARB");
	pGlProgramParameteriARB= (PFNGLPROGRAMPARAMETERIARBPROC) IRR_OGL_LOAD_EXTENSION("glProgramParameteriARB");
	pGlProgramParameteriEXT= (PFNGLPROGRAMPARAMETERIEXTPROC) IRR_OGL_LOAD_EXTENSION("glProgramParameteriEXT");
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



    pGlDebugMessageControl = (PFNGLDEBUGMESSAGECONTROLPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageControl");
    pGlDebugMessageControlARB = (PFNGLDEBUGMESSAGECONTROLARBPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageControlARB");
    pGlDebugMessageCallback = (PFNGLDEBUGMESSAGECALLBACKPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageCallback");
    pGlDebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC) IRR_OGL_LOAD_EXTENSION("glDebugMessageCallbackARB");

	// blend equation
	pGlBlendEquationEXT = (PFNGLBLENDEQUATIONEXTPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquationEXT");
	pGlBlendEquation = (PFNGLBLENDEQUATIONPROC) IRR_OGL_LOAD_EXTENSION("glBlendEquation");

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
#endif // use extension pointer
}

bool COpenGLExtensionHandler::queryFeature(const E_VIDEO_DRIVER_FEATURE &feature)
{
	switch (feature)
	{
	case EVDF_STENCIL_BUFFER:
	case EVDF_ARB_GLSL:
	case EVDF_COLOR_MASK:
		return true;
	case EVDF_ALPHA_TO_COVERAGE:
		return FeatureAvailable[IRR_ARB_multisample];
	case EVDF_GEOMETRY_SHADER:
		return true;
	case EVDF_MRT_BLEND:
	case EVDF_MRT_COLOR_MASK:
		return FeatureAvailable[IRR_EXT_draw_buffers2] || FeatureAvailable[IRR_ARB_draw_buffers_blend];
	case EVDF_MRT_BLEND_FUNC:
		return FeatureAvailable[IRR_ARB_draw_buffers_blend] || FeatureAvailable[IRR_AMD_draw_buffers_blend];
	case EVDF_OCCLUSION_QUERY:
	case EVDF_POLYGON_OFFSET:
	case EVDF_BLEND_OPERATIONS:
		return true;
	default:
		return false;
	};
}


bool COpenGLExtensionHandler::isDeviceCompatibile(core::array<std::string>* failedExtensions)
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

#ifdef _IRR_WINDOWS_API_
/*
    if (!(FeatureAvailable[WGL_ARB_pixel_format] || FeatureAvailable[WGL_ARB_framebuffer_sRGB]))
    {
        retval =  false;
        std::string error = "WGL_ARB_framebuffer_sRGB missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }*/
#else
    if (!(FeatureAvailable[IRR_EXT_framebuffer_sRGB]))
    {
        retval =  false;
        std::string error = "GL_EXT_framebuffer_sRGB missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }
#endif // _IRR_WINDOWS_API_

    if (!(FeatureAvailable[IRR_EXT_texture_filter_anisotropic]))
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

/*
    if (!(FeatureAvailable[IRR_NV_texture_barrier]||FeatureAvailable[IRR_ARB_texture_barrier]||Version>=450))
    {
        retval =  false;
        std::string error = "GL_NV_texture_barrier missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }*/

    if (!(FeatureAvailable[IRR_ARB_direct_state_access] || FeatureAvailable[IRR_EXT_direct_state_access] || Version>=450))
    {
        retval =  false;
        std::string error = "Direct State Access Extension missing\n";
        if (failedExtensions)
            failedExtensions->push_back(error);
        else
            os::Printer::log(error.c_str(), ELL_ERROR);
    }

    if (!FeatureAvailable[IRR_ARB_compute_shader]) //&&Version<430
    {
        retval =  false;
        std::string error = "GL_ARB_compute_shader missing\n";
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

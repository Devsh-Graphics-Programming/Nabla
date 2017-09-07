// Copyright (C) 2017- Mateusz Kielan
// This file is part of the "IrrlichtBAW".

#ifndef __C_OPENGL_STATE_MANAGER_H_INCLUDED__
#define __C_OPENGL_STATE_MANAGER_H_INCLUDED__

#include <limits>       // std::numeric_limits

#if defined(_IRR_WINDOWS_API_)
	// include windows headers for HWND
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
		#define GL_GLEXT_LEGACY 1
	#endif
	#include <GL/gl.h>
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
		#include "../Irrlicht/glext.h"
	#endif
	#include "wglext.h"

	#ifdef _MSC_VER
		#pragma comment(lib, "OpenGL32.lib")
//		#pragma comment(lib, "OpenCL.lib")
	#endif

#elif defined(_IRR_COMPILE_WITH_OSX_DEVICE_)
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
		#define GL_GLEXT_LEGACY 1
	#endif
	#include <OpenGL/gl.h>
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
		#include "../Irrlicht/glext.h"
	#endif
#elif defined(_IRR_COMPILE_WITH_SDL_DEVICE_) && !defined(_IRR_COMPILE_WITH_X11_DEVICE_)
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
		#define GL_GLEXT_LEGACY 1
		#define GLX_GLXEXT_LEGACY 1
	#else
		#define GL_GLEXT_PROTOTYPES 1
		#define GLX_GLXEXT_PROTOTYPES 1
	#endif
	#define NO_SDL_GLEXT
	#include <SDL/SDL_video.h>
	#include <SDL/SDL_opengl.h>
	#include "../source/Irrlicht/glext.h"
#else
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
		#define GL_GLEXT_LEGACY 1
		#define GLX_GLXEXT_LEGACY 1
	#else
		#define GL_GLEXT_PROTOTYPES 1
		#define GLX_GLXEXT_PROTOTYPES 1
	#endif
	#include <GL/gl.h>
	#include <GL/glx.h>
	#if defined(_IRR_OPENGL_USE_EXTPOINTER_)
        #include "../source/Irrlicht/glext.h"
	#endif
#endif

namespace irr
{
namespace video
{

    //! To be updated later as time moves on
	const uint32_t OGL_MAX_ENDISABLE_FLAGS = 36;
	const uint32_t OGL_MAX_ENDISABLEI_FLAGS = 2;

	const uint32_t OGL_STATE_MAX_VIEWPORTS = 16;
	const uint32_t OGL_STATE_MAX_DRAW_BUFFERS = 16;

	const uint8_t OGL_MAX_ENDISABLEI_INDICES = 16;//std::max(OGL_STATE_MAX_VIEWPORTS,OGL_STATE_MAX_DRAW_BUFFERS);
	const uint32_t OGL_STATE_MAX_TEXTURES = 192;
	//!

            enum E_GL_HINT_BIT
            {
                EGHB_FRAGMENT_SHADER_DERIVATIVE_HINT=0,
                EGHB_LINE_SMOOTH_HINT,
                EGHB_POLYGON_SMOOTH_HINT,
                EGHB_TEXTURE_COMPRESSION_HINT,
                EGHB_COUNT
            };
            enum E_GL_ENABLE_BIT
            {
                EGEB_DITHER=0,
                EGEB_FRAMEBUFFER_SRGB,
                EGEB_TEXTURE_CUBE_MAP_SEAMLESS,
                ///====>ALWAYS FORCE TO FALSE<====
                EGEB_POLYGON_OFFSET_FILL,
                EGEB_POLYGON_OFFSET_LINE,
                EGEB_POLYGON_OFFSET_POINT,
                ///====>ALWAYS FORCE TO FALSE<====
                EGEB_POLYGON_SMOOTH,
                EGEB_LINE_SMOOTH,
                EGEB_POINT_SMOOTH,
                EGEB_MULTISAMPLE,/*
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,*/
                EGEB_DEPTH_CLAMP,
                //EGEB_,
                EGEB_CLIP_DISTANCE0,
                EGEB_CLIP_DISTANCE1,
                EGEB_CLIP_DISTANCE2,
                EGEB_CLIP_DISTANCE3,
                EGEB_CLIP_DISTANCE4,
                EGEB_CLIP_DISTANCE5,
                EGEB_CLIP_DISTANCE6,
                EGEB_CLIP_DISTANCE7,/*
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,
                EGEB_,*/
                EGEB_RASTERIZER_DISCARD,
                EGEB_COUNT
            };

            enum E_GL_ENABLE_INDEX_BIT
            {
                EGEIB_BLEND=0,
                EGEIB_SCISSOR_TEST,
                EGEIB_COUNT
            };


	class COpenGLStateDiff
	{
        public:
            COpenGLStateDiff() : hintsToSet(0), glProvokingVertex_val(GL_INVALID_ENUM), glEnableCount(0), glDisableCount(0), glDisableiCount(0), glEnableiCount(0),
                                changeGlProgram(false), changeGlProgramPipeline(false), resetPolygonOffset(false),
                            setDepthRange(0), setViewportArray(0), setScissorBox(0),
                        glLogicOp_val(GL_INVALID_ENUM), setBlendColor(false), setBlendEquation(0), setBlendFunc(0),
                    texturesToBind(0), samplersToBind(0)
            {
                glPrimitiveSize[0] = std::numeric_limits<float>::quiet_NaN();
                glPrimitiveSize[1] = std::numeric_limits<float>::quiet_NaN();
            }


            uint8_t hintsToSet;
            GLenum glHint_pair[EGHB_COUNT][2];

            GLenum glProvokingVertex_val;

            //!
            GLenum glDisables[OGL_MAX_ENDISABLE_FLAGS];
            GLenum glEnables[OGL_MAX_ENDISABLE_FLAGS];

            //! these are sorted!
            struct EnDisAbleIndexedStatus
            {
                EnDisAbleIndexedStatus()
                {
                    flag = GL_INVALID_ENUM;
                    indices = 0;
                }

                GLenum flag;
                uint8_t indices;
            };
            EnDisAbleIndexedStatus glDisableis[OGL_MAX_ENDISABLEI_FLAGS];
            EnDisAbleIndexedStatus glEnableis[OGL_MAX_ENDISABLEI_FLAGS];

            uint8_t glDisableCount;
            uint8_t glEnableCount;

            uint8_t glDisableiCount;
            uint8_t glEnableiCount;

            class IndexedObjectToBind
            {
                public:
                    IndexedObjectToBind() : index(0xdeadbeefu), obj(0)
                    {
                    }
                    IndexedObjectToBind(const uint32_t& ix, const GLuint& object) : index(ix), obj(object)
                    {
                    }

                    uint32_t index;
                    GLuint obj;
            };

            bool resetPolygonOffset;
            float glPolygonOffset_factor,glPolygonOffset_units;
            float glPrimitiveSize[2]; //glPointSize, glLineWidth


            bool changeGlProgram,changeGlProgramPipeline;
            GLuint glUseProgram_val,glBindProgramPipeline_val;

            uint8_t setDepthRange;
            uint8_t setViewportArray;
            uint8_t setScissorBox;
            float glDepthRangeArray_vals[OGL_STATE_MAX_VIEWPORTS][2];
            GLint glViewportArray_vals[OGL_STATE_MAX_VIEWPORTS][4];
            GLint glScissorArray_vals[OGL_STATE_MAX_VIEWPORTS][4];

            GLenum glLogicOp_val;

            bool setBlendColor;
            uint8_t setBlendEquation;
            uint8_t setBlendFunc;
            float glBlendColor_vals[4];
            GLenum glBlendEquationSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][2];
            GLenum glBlendFuncSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][4];

            class TextureToBind : public IndexedObjectToBind
            {
                public:
                    TextureToBind() : IndexedObjectToBind(), target(GL_INVALID_ENUM)
                    {
                    }
                    TextureToBind(const uint32_t& ix, const GLuint& object, const GLenum& tgt) : IndexedObjectToBind(ix,object), target(tgt)
                    {
                    }

                    GLenum target;
            };
            uint32_t texturesToBind;
            uint32_t samplersToBind;
            TextureToBind bindTextures[OGL_STATE_MAX_TEXTURES];
            IndexedObjectToBind bindSamplers[OGL_STATE_MAX_TEXTURES];
        private:
	};


	class COpenGLState
	{
        public:
            inline GLenum glEnableBitToGLenum(const uint64_t &bit) const
            {
                switch (bit)
                {
                    case EGEB_DITHER:
                        return GL_DITHER;
                        break;
                    case EGEB_FRAMEBUFFER_SRGB:
                        return GL_FRAMEBUFFER_SRGB;
                        break;
                    case EGEB_TEXTURE_CUBE_MAP_SEAMLESS:
                        return GL_TEXTURE_CUBE_MAP_SEAMLESS;
                        break;
                    case EGEB_POLYGON_OFFSET_FILL:
                        return GL_POLYGON_OFFSET_FILL;
                        break;
                    case EGEB_POLYGON_OFFSET_LINE:
                        return GL_POLYGON_OFFSET_LINE;
                        break;
                    case EGEB_POLYGON_OFFSET_POINT:
                        return GL_POLYGON_OFFSET_POINT;
                        break;
                    case EGEB_POLYGON_SMOOTH:
                        return GL_POLYGON_SMOOTH;
                        break;
                    case EGEB_LINE_SMOOTH:
                        return GL_LINE_SMOOTH;
                        break;
                    case EGEB_POINT_SMOOTH:
                        return GL_POINT_SMOOTH;
                        break;
                    case EGEB_MULTISAMPLE:
                        return GL_MULTISAMPLE;
                        break;/*
                    case EGEB_:
                        return GL_;
                        break;*/
                    case EGEB_DEPTH_CLAMP:
                        return GL_DEPTH_CLAMP;
                        break;/*
                    case EGEB_:
                        return GL_;
                        break;*/
                    case EGEB_CLIP_DISTANCE0:
                        return GL_CLIP_DISTANCE0;
                        break;
                    case EGEB_CLIP_DISTANCE1:
                        return GL_CLIP_DISTANCE1;
                        break;
                    case EGEB_CLIP_DISTANCE2:
                        return GL_CLIP_DISTANCE2;
                        break;
                    case EGEB_CLIP_DISTANCE3:
                        return GL_CLIP_DISTANCE3;
                        break;
                    case EGEB_CLIP_DISTANCE4:
                        return GL_CLIP_DISTANCE4;
                        break;
                    case EGEB_CLIP_DISTANCE5:
                        return GL_CLIP_DISTANCE5;
                        break;
                    case EGEB_CLIP_DISTANCE6:
                        return GL_CLIP_DISTANCE6;
                        break;
                    case EGEB_CLIP_DISTANCE7:
                        return GL_CLIP_DISTANCE7;
                        break;
                    case EGEB_RASTERIZER_DISCARD:
                        return GL_RASTERIZER_DISCARD;
                        break;
                    default:
                        return GL_INVALID_ENUM;
                        break;
                }
            }

            inline GLenum glEnableiBitToGLenum(uint8_t& indexedBitOut, const uint64_t &bit) const
            {
                uint64_t ix = bit%OGL_MAX_ENDISABLEI_INDICES;
                indexedBitOut = 0x1u<<ix;
                switch (bit/OGL_MAX_ENDISABLEI_INDICES)
                {
                    case EGEIB_BLEND:
                        return GL_BLEND;
                        break;
                    case EGEIB_SCISSOR_TEST:
                        return GL_SCISSOR_TEST;
                        break;
                    default:
                        return GL_INVALID_ENUM;
                        break;
                }
            }

            //default OGL state at start of context as per the spec
            COpenGLState(const uint32_t& windowSizeX=0, const uint32_t& windowSizeY=0)
            {
                for (size_t i=0; i<EGHB_COUNT; i++)
                    glHint_vals[i] = GL_DONT_CARE;

                glProvokingVertex_val = GL_LAST_VERTEX_CONVENTION;

                size_t glEnableBitfieldByteSize = (EGEB_COUNT+63)/64;
                memset(glEnableBitfield,0,glEnableBitfieldByteSize);
                setGlEnableBit(EGEB_DITHER,true);
                setGlEnableBit(EGEB_MULTISAMPLE,true);

                size_t glEnableiBitfieldByteSize = (EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES+63)/64;
                memset(glEnableiBitfield,0,glEnableiBitfieldByteSize);

                glPolygonOffset_factor = 0.f;
                glPolygonOffset_units = 0.f;


                glUseProgram_val = 0;
                glBindProgramPipeline_val = 0;


                glPrimitiveSize[0] = 1.f;
                glPrimitiveSize[1] = 1.f;

                for (uint32_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
                {
                    glDepthRangeArray_vals[i][0] = 0.f;
                    glDepthRangeArray_vals[i][1] = 1.f;

                    glViewportArray_vals[i][0] = 0;
                    glViewportArray_vals[i][1] = 0;
                    glViewportArray_vals[i][2] = windowSizeX;
                    glViewportArray_vals[i][3] = windowSizeY;

                    glScissorArray_vals[i][0] = 0;
                    glScissorArray_vals[i][1] = 0;
                    glScissorArray_vals[i][2] = windowSizeX;
                    glScissorArray_vals[i][3] = windowSizeY;
                }

                glLogicOp_val = GL_COPY;

                memset(glBlendColor_vals,0,4*sizeof(float));

                for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
                {
                    glBlendEquationSeparatei_vals[i][0] = GL_FUNC_ADD;
                    glBlendEquationSeparatei_vals[i][1] = GL_FUNC_ADD;

                    glBlendFuncSeparatei_vals[i][0] = GL_ONE;
                    glBlendFuncSeparatei_vals[i][1] = GL_ZERO;
                    glBlendFuncSeparatei_vals[i][2] = GL_ONE;
                    glBlendFuncSeparatei_vals[i][3] = GL_ZERO;
                }

                for (size_t i=0; i<OGL_STATE_MAX_TEXTURES; i++)
                    boundTextureTargets[i] = GL_INVALID_ENUM;
                memset(boundTextures,0,sizeof(GLuint)*OGL_STATE_MAX_TEXTURES);
                memset(boundSamplers,0,sizeof(GLuint)*OGL_STATE_MAX_TEXTURES);
            }

            //! THIS IS SLOW AND FOR DEBUG ONLY!
            inline static COpenGLState collectGLState()
            {
                COpenGLState retval;

                //

                return retval;
            }

            inline void correctTheState()
            {
                setGlEnableBit(EGEB_POLYGON_OFFSET_POINT,false);
                setGlEnableBit(EGEB_POLYGON_OFFSET_LINE,false);
                setGlEnableBit(EGEB_POLYGON_OFFSET_FILL,false);

                setGlEnableBit(EGEB_POLYGON_SMOOTH,false);


                glPolygonOffset_factor = 0.f;
                glPolygonOffset_units = 0.f;

                //glPrimitiveSize[0] = 1.f;
                //glPrimitiveSize[1] = 1.f;
            }

            inline COpenGLStateDiff getStateDiff(const COpenGLState &previousState,
                                              const bool& careAboutHints=true, //should be default false
                                              const bool& careAboutPolygonOffset=true, //should be default false
                                              const bool& careAboutFBOs=true,
                                              const bool& careAboutProgram=true,
                                              const bool& careAboutPipeline=true,
                                              const bool& careAboutViewports=true,
                                              const bool& careAboutPointSize=true,
                                              const bool& careAboutLineWidth=true,
                                              const bool& careAboutLogicOp=true,
                                              const bool& careAboutBlending=true,
                                              const bool& careAboutTextures=true) const
            {
                COpenGLStateDiff diff;

                if (careAboutHints)
                {
                    for (uint64_t i=0; i<EGHB_COUNT; i++)
                    {
                        if (glHint_vals[i]!=previousState.glHint_vals[i])
                        {
                            switch (i)
                            {
                                case EGHB_FRAGMENT_SHADER_DERIVATIVE_HINT:
                                    diff.glHint_pair[i][0] = GL_FRAGMENT_SHADER_DERIVATIVE_HINT;
                                    break;
                                case EGHB_LINE_SMOOTH_HINT:
                                    diff.glHint_pair[i][0] = GL_LINE_SMOOTH_HINT;
                                    break;
                                case EGHB_POLYGON_SMOOTH_HINT:
                                    diff.glHint_pair[i][0] = GL_POLYGON_SMOOTH_HINT;
                                    break;
                                case EGHB_TEXTURE_COMPRESSION_HINT:
                                    diff.glHint_pair[i][0] = GL_TEXTURE_COMPRESSION_HINT;
                                    break;
                                default:
                                    diff.glHint_pair[i][0] = GL_INVALID_ENUM;
                                    break;
                            }
                            diff.glHint_pair[i][1] = glHint_vals[i];
                            diff.hintsToSet++;
                        }
                    }
                }

                if (glProvokingVertex_val!=previousState.glProvokingVertex_val)
                    diff.glProvokingVertex_val = glProvokingVertex_val;


                {
                    for (uint64_t i=0; i<EGEB_COUNT/64u; i++)
                    {
                        uint64_t bitdiff = glEnableBitfield[i]^previousState.glEnableBitfield[i];

                        for (uint64_t j=0; j<64u; j++)
                            setEnableDiffBits(diff,bitdiff,i,j);
                    }
                    uint64_t leftOvers = EGEB_COUNT%64u;
                    if (leftOvers)
                    {
                        uint64_t bitdiff = glEnableBitfield[EGEB_COUNT/64u]^previousState.glEnableBitfield[EGEB_COUNT/64u];
                        for (uint64_t j=0; j<leftOvers; j++)
                            setEnableDiffBits(diff,bitdiff,EGEB_COUNT/64u,j);
                    }
                }
                {
                    for (uint64_t i=0; i<(EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES)/64u; i++)
                    {
                        uint64_t bitdiff = glEnableiBitfield[i]^previousState.glEnableiBitfield[i];

                        for (uint64_t j=0; j<64u; j++)
                            setEnableiDiffBits(diff,bitdiff,i,j);
                    }
                    uint64_t leftOvers = (EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES)%64u;
                    if (leftOvers)
                    {
                        uint64_t i = (EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES)/64u;
                        uint64_t bitdiff = glEnableiBitfield[i]^previousState.glEnableiBitfield[i];
                        for (uint64_t j=0; j<leftOvers; j++)
                            setEnableiDiffBits(diff,bitdiff,i,j);
                    }
                }


                if (careAboutPolygonOffset&&(glPolygonOffset_factor!=previousState.glPolygonOffset_factor||glPolygonOffset_units!=previousState.glPolygonOffset_units))
                {
                    diff.resetPolygonOffset = false;
                    diff.glPolygonOffset_factor = glPolygonOffset_factor;
                    diff.glPolygonOffset_units = glPolygonOffset_units;
                }


                if (careAboutProgram&&glUseProgram_val!=previousState.glUseProgram_val)
                {
                    diff.changeGlProgram = true;
                    diff.glUseProgram_val = glUseProgram_val;
                }
                if (careAboutPipeline&&glBindProgramPipeline_val!=previousState.glBindProgramPipeline_val)
                {
                    diff.changeGlProgramPipeline = true;
                    diff.glBindProgramPipeline_val = glBindProgramPipeline_val;
                }


                if (careAboutFBOs)
                {
                    //;
                }

                if (careAboutViewports)
                {
                    size_t j=0;
                    for (uint32_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
                    {
                        if (glDepthRangeArray_vals[i][0]!=previousState.glDepthRangeArray_vals[i][0]||
                            glDepthRangeArray_vals[i][1]!=previousState.glDepthRangeArray_vals[i][1])
                        {
                            diff.setDepthRange |= 0x1u<<i;
                            memcpy(diff.glDepthRangeArray_vals[j],glDepthRangeArray_vals[i],4*sizeof(float));
                            j++;
                        }
                    }
                    j=0;
                    for (uint32_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
                    {
                        if (glViewportArray_vals[i][0]!=previousState.glViewportArray_vals[i][0]||
                            glViewportArray_vals[i][1]!=previousState.glViewportArray_vals[i][1]||
                            glViewportArray_vals[i][2]!=previousState.glViewportArray_vals[i][2]||
                            glViewportArray_vals[i][3]!=previousState.glViewportArray_vals[i][3])
                        {
                            diff.setViewportArray |= 0x1u<<i;
                            memcpy(diff.glViewportArray_vals[j],glViewportArray_vals[i],4*sizeof(float));
                            j++;
                        }
                    }
                    j=0;
                    for (uint32_t i=0; i<OGL_STATE_MAX_VIEWPORTS; i++)
                    {
                        if (glScissorArray_vals[i][0]!=previousState.glScissorArray_vals[i][0]||
                            glScissorArray_vals[i][1]!=previousState.glScissorArray_vals[i][1]||
                            glScissorArray_vals[i][2]!=previousState.glScissorArray_vals[i][2]||
                            glScissorArray_vals[i][3]!=previousState.glScissorArray_vals[i][3])
                        {
                            diff.setScissorBox |= 0x1u<<i;
                            memcpy(diff.glScissorArray_vals[j],glScissorArray_vals[i],4*sizeof(float));
                            j++;
                        }
                    }
                }

                if (careAboutPointSize&&glPrimitiveSize[0]!=previousState.glPrimitiveSize[0])
                    diff.glPrimitiveSize[0] = glPrimitiveSize[0];
                if (careAboutLineWidth&&glPrimitiveSize[1]!=previousState.glPrimitiveSize[1])
                    diff.glPrimitiveSize[1] = glPrimitiveSize[1];


                if (careAboutLogicOp&&glLogicOp_val!=previousState.glLogicOp_val)
                    diff.glLogicOp_val = glLogicOp_val;


                if (careAboutBlending)
                {
                    if (glBlendColor_vals[0]!=previousState.glBlendColor_vals[0]||
                        glBlendColor_vals[1]!=previousState.glBlendColor_vals[1]||
                        glBlendColor_vals[2]!=previousState.glBlendColor_vals[2]||
                        glBlendColor_vals[3]!=previousState.glBlendColor_vals[3])
                    {
                        diff.setBlendColor = true;
                        memcpy(diff.glBlendColor_vals,glBlendColor_vals,4*sizeof(float));
                    }
                    size_t j=0;
                    for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
                    {
                        if (glBlendEquationSeparatei_vals[i][0]!=previousState.glBlendEquationSeparatei_vals[i][0]||
                            glBlendEquationSeparatei_vals[i][1]!=previousState.glBlendEquationSeparatei_vals[i][1])
                        {
                            diff.glBlendEquationSeparatei_vals[j][0] = glBlendEquationSeparatei_vals[i][0];
                            diff.glBlendEquationSeparatei_vals[j][1] = glBlendEquationSeparatei_vals[i][1];
                            j++;
                        }
                    }
                    j=0;
                    for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
                    {
                        if (glBlendFuncSeparatei_vals[i][0]!=previousState.glBlendFuncSeparatei_vals[i][0]||
                            glBlendFuncSeparatei_vals[i][1]!=previousState.glBlendFuncSeparatei_vals[i][1]||
                            glBlendFuncSeparatei_vals[i][2]!=previousState.glBlendFuncSeparatei_vals[i][2]||
                            glBlendFuncSeparatei_vals[i][3]!=previousState.glBlendFuncSeparatei_vals[i][3])
                        {
                            diff.glBlendFuncSeparatei_vals[j][0] = glBlendFuncSeparatei_vals[i][0];
                            diff.glBlendFuncSeparatei_vals[j][1] = glBlendFuncSeparatei_vals[i][1];
                            diff.glBlendFuncSeparatei_vals[j][2] = glBlendFuncSeparatei_vals[i][2];
                            diff.glBlendFuncSeparatei_vals[j][3] = glBlendFuncSeparatei_vals[i][3];
                            j++;
                        }
                    }
                }

                if (careAboutTextures)
                {
                    for (uint32_t i=0; i<OGL_STATE_MAX_TEXTURES; i++)
                    {
                        if (boundTextures[i]!=previousState.boundTextures[i]||boundTextureTargets[i]!=previousState.boundTextureTargets[i])
                            diff.bindTextures[diff.texturesToBind++] = COpenGLStateDiff::TextureToBind(i,boundTextures[i],boundTextureTargets[i]);

                        if (boundSamplers[i]!=previousState.boundSamplers[i])
                            diff.bindSamplers[diff.samplersToBind++] = COpenGLStateDiff::IndexedObjectToBind(i,boundSamplers[i]);
                    }
                }

                return diff;
            }

            inline COpenGLStateDiff operator^(const COpenGLState &previousState) const {return getStateDiff(previousState);}


        private:
            inline void setEnableDiffBits(COpenGLStateDiff& diff, const uint64_t& bitdiff, const uint64_t& i, const uint64_t& j) const
            {
                const uint64_t bitFlag = uint64_t(0x1ull)<<j;
                if (bitdiff&bitFlag)
                {
                    if (glEnableBitfield[i]&bitFlag)
                        diff.glEnables[diff.glEnableCount++] = glEnableBitToGLenum(i*64+j);
                    else
                        diff.glDisables[diff.glDisableCount++] = glEnableBitToGLenum(i*64+j);
                }
            }
            inline void setEnableiDiffBits(COpenGLStateDiff& diff, const uint64_t& bitdiff, const uint64_t& i, const uint64_t& j) const
            {
                const uint64_t bitFlag = uint64_t(0x1ull)<<j;
                if (bitdiff&bitFlag)
                {
                    uint8_t ix;
                    GLenum combo = glEnableiBitToGLenum(ix,i*64+j);
                    if (glEnableBitfield[i]&bitFlag)
                    {
                        diff.glEnableis[diff.glEnableiCount].indices |= ix;

                        GLenum currVal = diff.glEnableis[diff.glEnableiCount].flag;
                        if (currVal!=combo)
                            diff.glEnableis[diff.glEnableiCount++].flag = combo;
                    }
                    else
                    {
                        diff.glDisableis[diff.glDisableiCount].indices |= ix;

                        GLenum currVal = diff.glDisableis[diff.glDisableiCount].flag;
                        if (currVal!=combo)
                            diff.glDisableis[diff.glDisableiCount++].flag = combo;
                    }
                }
            }


            GLenum glHint_vals[EGHB_COUNT];
            GLenum glProvokingVertex_val;

            uint64_t glEnableBitfield[(EGEB_COUNT+63)/64];
            inline bool getGlEnableBit(const E_GL_ENABLE_BIT& bit) const
            {
                return glEnableBitfield[bit/64]&(uint64_t(0x1ull)<<(bit%64));
            }
            inline void setGlEnableBit(const E_GL_ENABLE_BIT& bit, bool value)
            {
                if (value)
                    glEnableBitfield[bit/64] |= uint64_t(0x1ull)<<(bit%64);
                else
                    glEnableBitfield[bit/64] &= ~(uint64_t(0x1ull)<<(bit%64));
            }

            uint64_t glEnableiBitfield[(EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES+63)/64];
            inline bool getGlEnableiBit(uint64_t bit, const uint32_t& index) const
            {
                bit *= OGL_MAX_ENDISABLEI_INDICES;
                bit += index;
                return glEnableiBitfield[bit/64]&(uint64_t(0x1ull)<<(bit%64));
            }
            inline void setGlEnableiBit(uint64_t bit, const uint32_t& index, bool value)
            {
                bit *= OGL_MAX_ENDISABLEI_INDICES;
                bit += index;
                if (value)
                    glEnableiBitfield[bit/64] |= uint64_t(0x1ull)<<(bit%64);
                else
                    glEnableiBitfield[bit/64] &= ~(uint64_t(0x1ull)<<(bit%64));
            }

            float glPolygonOffset_factor,glPolygonOffset_units;
            float glPrimitiveSize[2]; //glPointSize, glLineWidth


            GLuint glUseProgram_val,glBindProgramPipeline_val;

            float glDepthRangeArray_vals[OGL_STATE_MAX_VIEWPORTS][2];
            GLint glViewportArray_vals[OGL_STATE_MAX_VIEWPORTS][4];
            GLint glScissorArray_vals[OGL_STATE_MAX_VIEWPORTS][4];

            //BUFFER BINDING POINTS

            GLenum glLogicOp_val;

            float glBlendColor_vals[4];
            GLenum glBlendEquationSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][2];
            GLenum glBlendFuncSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][4];

            GLenum boundTextureTargets[OGL_STATE_MAX_TEXTURES];
            GLuint boundTextures[OGL_STATE_MAX_TEXTURES];
            GLuint boundSamplers[OGL_STATE_MAX_TEXTURES];
	};


} // end namespace video
} // end namespace irr

#endif

/**
OpenGLState
{
GL_CULL_FACE : 1
glCullFace
glFrontFace
GL_DEPTH_TEST : 1
glDepthFunc : 3
glDepthMask : 1 //write to Z
glDepthRangeIndexed(viewport,zNear = 1, zFar = 1)
GL_STENCIL_TEST : 1
glStencilOpSeparate
glStencilMaskSeparate
glStencilFuncSeparate
GL_COLOR_LOGIC_OP
glLogicOp
GL_LINE_SMOOTH : 1
///GL_MULTISAMPLE : 1
glSampleCoverage


GL_POLYGON_SMOOTH : 1
GL_PRIMITIVE_RESTART : 1
glPrimitiveRestartIndex
GL_PRIMITIVE_RESTART_FIXED_INDEX : 1
GL_SAMPLE_ALPHA_TO_COVERAGE : 1
GL_SAMPLE_ALPHA_TO_ONE : 1
GL_SAMPLE_COVERAGE : 1
glSampleCoverage
GL_SAMPLE_SHADING
glMinSampleShading
GL_SAMPLE_MASK
glSampleMaski



glUniformBlockBinding
glBindBuffer
glBindBufferBase
glBindBufferRange
{
GL_PIXEL_PACK_BUFFER_BINDING
GL_PIXEL_UNPACK_BUFFER_BINDING
GL_UNIFORM_BUFFER_BINDING

}

glShaderStorageBlockBinding

GL_VERTEX_ARRAY_BINDING

GL_POINT_SIZE
GL_PROGRAM_POINT_SIZE
GL_POINT_FADE_THRESHOLD_SIZE
glPointSize
glPointParameter*

glColorMaski : 4 bool x MRT

glPolygonMode
GL_POLYGON_SMOOTH

BoundQueries

ActiveXFormFeedback



glPixelStorei //12 vals

GL_STENCIL_CLEAR_VALUE
glClearStencil


ActiveFBOs


glReadPixels
GL_READ_BUFFER

glClampColor : 1 bool

GL_PROVOKING_VERTEX
}
**/

// Copyright (C) 2017- Mateusz Kielan
// This file is part of the "IrrlichtBAW".

#ifndef __C_OPENGL_STATE_MANAGER_H_INCLUDED__
#define __C_OPENGL_STATE_MANAGER_H_INCLUDED__

#include <limits>       // std::numeric_limits
#include <utility>

#ifndef _IRR_OPENGL_USE_EXTPOINTER_
#   define _IRR_OPENGL_USE_EXTPOINTER_
#endif

#if defined(_IRR_WINDOWS_API_)
	// include windows headers for HWND
	#define WIN32_LEAN_AND_MEAN
	#define NOMINMAX
	#include <windows.h>
    #define GL_GLEXT_LEGACY 1
	#include <GL/gl.h>
    #undef GL_GLEXT_LEGACY
    #include "../source/Irrlicht/glext.h"
	#include "../source/Irrlicht/wglext.h"
#elif defined(_IRR_COMPILE_WITH_SDL_DEVICE_) && !defined(_IRR_COMPILE_WITH_X11_DEVICE_)
	#include <SDL/SDL_video.h>
    #define GL_GLEXT_LEGACY 1
	#include <SDL/SDL_opengl.h>
    #undef GL_GLEXT_LEGACY
	#include "../source/Irrlicht/glext.h"
#else
    #define GL_GLEXT_LEGACY 1
	#include <GL/gl.h>
    #undef GL_GLEXT_LEGACY
    #include "../source/Irrlicht/glext.h"
#endif

namespace irr
{
namespace video
{

/**

New IDEA EVERY DAY

COpenGLHandler:: static calls
    report to a tracker state

Normal calls take immediate effect they change the tracker state,
but only call OpenGL if there would actually be any change.


Could do deferred calls, then a ghost state gets updated (or first time, created).
When ExecuteDeferredStateChange gets called, all the ghost state
updates the tracker state.

APPENDIX: If ghost state is present when immediate state changes are about to run,
ExecuteDeferredStateChange first.

::extGlSomeCall(std::tuple<COpenGLState,COpenGLState,bool>* stateOfContext, ...)
{
    const bool hasGhostState = stateOfContext->third;

    if (hasGhostState) // essentially   ::ExecuteDeferredStateChange()
    {
        execGLDiff(stateOfContext->second.stateDiff(stateOfContext->first)); //get diff between actual state and ghost then execute
        stateOfContext->first = stateOfContext->second; //update the current state to be fully flushed
        stateOfContext->third = false; //ghost state is clear
    }

    glSomeCall(..);
    CHANGE_STATE_FOR_SomeCall(stateOfContext->first) //change current state to include this call too
}

::Deferred_extGlSomeCall(std::tuple<COpenGLState,COpenGLState,bool>* stateOfContext, ...)
{
    const bool hasGhostState = stateOfContext->third;

    if (!hasGhostState)
    {
        stateOfContext->second = stateOfContext->first;
        stateOfContext->third = true;
    }

    CHANGE_STATE_FOR_SomeCall(stateOfContext->second) //change current state to include this call too
}


P.S. Maybe Ghost == Pending

**/

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
                EGEB_COLOR_LOGIC_OP,
                EGEB_DEPTH_CLAMP,
                EGEB_PROGRAM_POINT_SIZE,
                EGEB_MULTISAMPLE,
                EGEB_SAMPLE_COVERAGE,
                EGEB_SAMPLE_ALPHA_TO_COVERAGE,
                EGEB_SAMPLE_ALPHA_TO_ONE,
                EGEB_SAMPLE_MASK,
                EGEB_SAMPLE_SHADING,
                EGEB_CLIP_DISTANCE0,
                EGEB_CLIP_DISTANCE1,
                EGEB_CLIP_DISTANCE2,
                EGEB_CLIP_DISTANCE3,
                EGEB_CLIP_DISTANCE4,
                EGEB_CLIP_DISTANCE5,
                EGEB_CLIP_DISTANCE6,
                EGEB_CLIP_DISTANCE7,
                EGEB_STENCIL_TEST,
                EGEB_DEPTH_TEST,
                EGEB_CULL_FACE,
                EGEB_PRIMITIVE_RESTART,
                EGEB_PRIMITIVE_RESTART_FIXED_INDEX,
                EGEB_RASTERIZER_DISCARD,
                EGEB_COUNT
            };

            enum E_GL_ENABLE_INDEX_BIT
            {
                EGEIB_BLEND=0,
                EGEIB_SCISSOR_TEST,
                EGEIB_COUNT
            };


            enum E_GL_PACK_PARAM
            {
                EGPP_ALIGNMENT=0,
                EGPP_COMPRESSED_BLOCK_WIDTH,
                EGPP_COMPRESSED_BLOCK_HEIGHT,
                EGPP_COMPRESSED_BLOCK_DEPTH,
                EGPP_COMPRESSED_BLOCK_SIZE,
                EGPP_ROW_LENGTH,
                EGPP_IMAGE_HEIGHT,
                EGPP_SWAP_BYTES,
                EGPP_SKIP_PIXELS,
                EGPP_SKIP_ROWS,
                EGPP_SKIP_IMAGES,
                EGPP_COUNT
            };
/*
            enum E_GL_UNPACK_ATTR
            {
                EGUPA_ALIGNMENT=0,
                EGUPA_COMPRESSED_BLOCK_WIDTH,
                EGUPA_COMPRESSED_BLOCK_HEIGHT,
                EGUPA_COMPRESSED_BLOCK_DEPTH,
                EGUPA_COMPRESSED_BLOCK_SIZE,
                EGUPA_ROW_LENGTH,
                EGUPA_IMAGE_HEIGHT,
                EGUPA_SWAP_BYTES,
                EGUPA_SKIP_PIXELS,
                EGUPA_SKIP_ROWS,
                EGUPA_SKIP_IMAGES,
                EGUPA_COUNT
            };
*/
            enum E_GL_BUFFER_TYPE
            {
                EGBT_PIXEL_PACK_BUFFER=0,
                EGBT_PIXEL_UNPACK_BUFFER,
                EGBT_DRAW_INDIRECT_BUFFER,
                //EGBT_QUERY_BUFFER,
                EGBT_DISPATCH_INDIRECT_BUFFER,
                EGBT_COUNT,
            };

            enum E_GL_RANGED_BUFFER_TYPE
            {
                EGRBT_UNIFORM_BUFFER=0,
                EGRBT_SHADER_STORAGE_BUFFER,
                EGRBT_COUNT,
            };

            enum E_GL_TEXTURE_TYPE
            {
                EGTT_1D=0,
                EGTT_1D_ARRAY,
                EGTT_2D,
                EGTT_2D_ARRAY,
                EGTT_2D_MULTISAMPLE,
                EGTT_2D_MULTISAMPLE_ARRAY,
                EGTT_3D,
                EGTT_BUFFER,
                EGTT_CUBE_MAP,
                EGTT_CUBE_MAP_ARRAY,
                EGTT_RECTANGLE,
                EGTT_COUNT
            };

    //! To be updated later as time moves on
	const uint32_t OGL_STATE_MAX_VIEWPORTS = 16;
	const uint32_t OGL_STATE_MAX_DRAW_BUFFERS = 16;

	const uint32_t OGL_STATE_MAX_SAMPLE_MASK_WORDS = 4;

	const uint32_t OGL_MAX_BUFFER_BINDINGS = 16; //! should be 96

	const uint8_t OGL_MAX_ENDISABLEI_INDICES = 16;//std::max(OGL_STATE_MAX_VIEWPORTS,OGL_STATE_MAX_DRAW_BUFFERS);
	const uint32_t OGL_STATE_MAX_IMAGES = 8; //! Should be 192
	const uint32_t OGL_STATE_MAX_TEXTURES = 8; //! Should be 192
	//!



	class COpenGLStateDiff
	{
        public:
            COpenGLStateDiff() : hintsToSet(0), glProvokingVertex_val(GL_INVALID_ENUM), glEnableCount(0), glDisableCount(0), glDisableiCount(0), glEnableiCount(0),
                                bindFramebuffers(0), resetPolygonOffset(false), glClampColor_val(GL_INVALID_ENUM), glPixelStoreiCount(0), setPrimitiveRestartIndex(false),
                            changeXFormFeedback(false), changeGlProgram(false), changeGlProgramPipeline(false), glPatchParameteri_val(0), setDepthRange(0), setViewportArray(0), setScissorBox(0),
                        glLogicOp_val(GL_INVALID_ENUM), setSampleMask(0), setBlendColor(false), setBlendEquation(0), setBlendFunc(0), setColorMask(0),
                    setStencilFunc(0), setStencilOp(0), setStencilMask(0), setDepthMask(0), glDepthFunc_val(GL_INVALID_ENUM), setBufferRanges(0),
                glPolygonMode_mode(GL_INVALID_ENUM), glFrontFace_val(GL_INVALID_ENUM), glCullFace_val(GL_INVALID_ENUM), setVAO(false), setImageBindings(0), texturesToBind(0), samplersToBind(0)
            {
                glPrimitiveSize[0] = -FLT_MAX;
                glPrimitiveSize[1] = -FLT_MAX;

                glPatchParameterfv_inner[0] = -FLT_MAX;
                glPatchParameterfv_outer[0] = -FLT_MAX;

                memset(setBuffers,0,sizeof(bool)*EGBT_COUNT);

                glSampleCoverage_val = -FLT_MAX;
                glSampleCoverage_invert = false;
                glMinSampleShading_val = -FLT_MAX;

                for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS/16; i++)
                    glColorMaski_vals[i] = ~uint64_t(0);
            }


            uint8_t hintsToSet;
            GLenum glHint_pair[EGHB_COUNT][2];

            GLenum glProvokingVertex_val;

            //!
            GLenum glDisables[EGEB_COUNT];
            GLenum glEnables[EGEB_COUNT];

            //! these are sorted!
            struct EnDisAbleIndexedStatus
            {
                EnDisAbleIndexedStatus()
                {
                    flag = GL_INVALID_ENUM;
                    indices = 0;
                }

                GLenum flag;
                uint16_t indices;
            };
            EnDisAbleIndexedStatus glDisableis[EGEIB_COUNT];
            EnDisAbleIndexedStatus glEnableis[EGEIB_COUNT];

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
                    IndexedObjectToBind(const uint32_t ix, const GLuint object) : index(ix), obj(object)
                    {
                    }

                    uint32_t index;
                    GLuint obj;
            };

            uint8_t bindFramebuffers;
            GLuint glBindFramebuffer_vals[2];

            bool resetPolygonOffset;
            float glPolygonOffset_factor,glPolygonOffset_units;
            float glPrimitiveSize[2]; //glPointSize, glLineWidth

            GLenum glClampColor_val;
            uint8_t glPixelStoreiCount;
            std::pair<GLenum,int32_t> glPixelStorei_vals[2*EGPP_COUNT];

            bool setPrimitiveRestartIndex;
            GLuint glPrimitiveRestartIndex_val;

            bool changeXFormFeedback,changeGlProgram,changeGlProgramPipeline;
            GLuint glBindTransformFeedback_val,glUseProgram_val,glBindProgramPipeline_val;

            GLint glPatchParameteri_val;
            float glPatchParameterfv_inner[2];
            float glPatchParameterfv_outer[4];

            uint16_t setDepthRange;
            uint16_t setViewportArray;
            uint16_t setScissorBox;
            float glDepthRangeArray_vals[OGL_STATE_MAX_VIEWPORTS][2];
            float glViewportArray_vals[OGL_STATE_MAX_VIEWPORTS][4];
            GLint glScissorArray_vals[OGL_STATE_MAX_VIEWPORTS][4];


            class RangedBufferBinding
            {
                public:
                    RangedBufferBinding() : object(0), offset(0), size(0)
                    {
                    }
                    RangedBufferBinding(const GLuint obj, const GLintptr off, const GLsizeiptr sz)
                                        : object(obj), offset(off), size(sz)
                    {
                    }

                    inline bool operator!=(const RangedBufferBinding &other) const
                    {
                        return object!=other.object||offset!=other.offset||size!=other.size;
                    }
                    inline bool operator==(const RangedBufferBinding &other) const
                    {
                        return !((*this)!=other);
                    }

                    GLuint object;
                    GLintptr offset;
                    GLsizeiptr size;
            };
            class RangedBufferBindingDiff : public RangedBufferBinding
            {
                public:
                    RangedBufferBindingDiff() : RangedBufferBinding(), index(0xdeadbeefu), bindPoint(GL_INVALID_ENUM)
                    {
                    }
                    RangedBufferBindingDiff(const GLenum bndPt, const uint32_t ix, const RangedBufferBinding& buff)
                                            : RangedBufferBinding(buff), bindPoint(bndPt), index(ix)
                    {
                    }

                    GLenum bindPoint;
                    uint32_t index;
            };
            bool setBuffers[EGBT_COUNT];
            GLuint bindBuffer[EGBT_COUNT];

            GLenum glLogicOp_val;
            uint8_t setSampleMask;
            bool setBlendColor;
            uint16_t setBlendEquation;
            uint16_t setBlendFunc;
            uint16_t setColorMask;
            float glSampleCoverage_val;
            bool glSampleCoverage_invert;
            uint32_t glSampleMaski_vals[OGL_STATE_MAX_SAMPLE_MASK_WORDS];
            float glMinSampleShading_val;
            float glBlendColor_vals[4];
            GLenum glBlendEquationSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][2];
            GLenum glBlendFuncSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][4];
            uint64_t glColorMaski_vals[OGL_STATE_MAX_DRAW_BUFFERS/16];

            uint8_t  setStencilFunc,setStencilOp,setStencilMask;
            GLenum  glStencilFuncSeparate_func[2];
            GLint   glStencilFuncSeparate_ref[2];
            GLuint  glStencilFuncSeparate_mask[2];
            GLenum  glStencilOpSeparate_sfail[2];
            GLenum  glStencilOpSeparate_dpfail[2];
            GLenum  glStencilOpSeparate_dppass[2];
            GLuint  glStencilMaskSeparate_mask[2];

            uint8_t  setDepthMask;
            GLenum  glDepthFunc_val;


            uint32_t setBufferRanges;
            RangedBufferBindingDiff bindBufferRange[EGRBT_COUNT*OGL_MAX_BUFFER_BINDINGS];

            class ImageToBind : public IndexedObjectToBind
            {
                public:
                    ImageToBind() : IndexedObjectToBind(), level(0),
                                layered(GL_FALSE), layer(0), access(GL_READ_ONLY), format(GL_R8)
                    {
                    }
                    ImageToBind(const uint32_t unit, const GLuint texture, const GLint level_in,
                                  const GLboolean layered_in, const GLint layer_in,
                                  const GLenum access_in, const GLenum format_in)
                                : IndexedObjectToBind(unit,texture), level(level_in),
                                    layered(layered_in), layer(layer_in),
                                    access(access_in), format(format_in)
                    {
                    }

                    GLint level;
                    GLboolean layered;
                    GLint layer;
                    GLenum access;
                    GLenum format;
            };
            uint32_t setImageBindings;
            ImageToBind bindImages[OGL_STATE_MAX_IMAGES];

            class TextureToBind : public IndexedObjectToBind
            {
                public:
                    TextureToBind() : IndexedObjectToBind(), target(GL_INVALID_ENUM)
                    {
                    }
                    TextureToBind(const uint32_t ix, const GLuint object, const GLenum tgt) : IndexedObjectToBind(ix,object), target(tgt)
                    {
                    }

                    GLenum target;
            };
            uint32_t texturesToBind;
            uint32_t samplersToBind;
            TextureToBind bindTextures[OGL_STATE_MAX_TEXTURES*EGTT_COUNT];
            IndexedObjectToBind bindSamplers[OGL_STATE_MAX_TEXTURES];

            GLenum glPolygonMode_mode;
            GLenum glFrontFace_val;
            GLenum glCullFace_val;

            bool setVAO;
            GLuint bindVAO;
        private:
	};

    void executeGLDiff(const COpenGLStateDiff& diff);


	class COpenGLState
	{
        public:
            inline static GLenum glEnableBitToGLenum(const uint64_t &bit)
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
                    case EGEB_COLOR_LOGIC_OP:
                        return GL_COLOR_LOGIC_OP;
                        break;
                    case EGEB_DEPTH_CLAMP:
                        return GL_DEPTH_CLAMP;
                        break;
                    case EGEB_PROGRAM_POINT_SIZE:
                        return GL_PROGRAM_POINT_SIZE;
                        break;
                    case EGEB_MULTISAMPLE:
                        return GL_MULTISAMPLE;
                        break;
                    case EGEB_SAMPLE_COVERAGE:
                        return GL_SAMPLE_COVERAGE;
                        break;
                    case EGEB_SAMPLE_ALPHA_TO_COVERAGE:
                        return GL_SAMPLE_ALPHA_TO_COVERAGE;
                        break;
                    case EGEB_SAMPLE_ALPHA_TO_ONE:
                        return GL_SAMPLE_ALPHA_TO_ONE;
                        break;
                    case EGEB_SAMPLE_MASK:
                        return GL_SAMPLE_MASK;
                        break;
                    case EGEB_SAMPLE_SHADING:
                        return GL_SAMPLE_SHADING;
                        break;
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
                    case EGEB_STENCIL_TEST:
                        return GL_STENCIL_TEST;
                        break;
                    case EGEB_DEPTH_TEST:
                        return GL_DEPTH_TEST;
                        break;
                    case EGEB_CULL_FACE:
                        return GL_CULL_FACE;
                        break;
                    case EGEB_PRIMITIVE_RESTART:
                        return GL_PRIMITIVE_RESTART;
                        break;
                    case EGEB_PRIMITIVE_RESTART_FIXED_INDEX:
                        return GL_PRIMITIVE_RESTART_FIXED_INDEX;
                        break;
                    case EGEB_RASTERIZER_DISCARD:
                        return GL_RASTERIZER_DISCARD;
                        break;
                    default:
                        return GL_INVALID_ENUM;
                        break;
                }
            }

            inline GLenum glEnableiBitToGLenum(uint16_t& indexedBitOut, const uint64_t bit) const
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

            inline static GLenum glPackParamToGLenum(const uint32_t packParam)
            {
                if (packParam<EGPP_COUNT)
                {
                    switch (packParam)
                    {
                        case (EGPP_ALIGNMENT):
                            return GL_PACK_ALIGNMENT;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_WIDTH):
                            return GL_PACK_COMPRESSED_BLOCK_WIDTH;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_HEIGHT):
                            return GL_PACK_COMPRESSED_BLOCK_HEIGHT;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_DEPTH):
                            return GL_PACK_COMPRESSED_BLOCK_DEPTH;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_SIZE):
                            return GL_PACK_COMPRESSED_BLOCK_SIZE;
                            break;
                        case (EGPP_ROW_LENGTH):
                            return GL_PACK_ROW_LENGTH;
                            break;
                        case (EGPP_IMAGE_HEIGHT):
                            return GL_PACK_IMAGE_HEIGHT;
                            break;
                        case (EGPP_SWAP_BYTES):
                            return GL_PACK_SWAP_BYTES;
                            break;
                        case (EGPP_SKIP_PIXELS):
                            return GL_PACK_SKIP_PIXELS;
                            break;
                        case (EGPP_SKIP_ROWS):
                            return GL_PACK_SKIP_PIXELS;
                            break;
                        case (EGPP_SKIP_IMAGES):
                            return GL_PACK_SKIP_IMAGES;
                            break;
                        default:
                            return GL_INVALID_ENUM;
                            break;
                    }
                }
                else
                {
                    switch (packParam)
                    {
                        case (EGPP_ALIGNMENT+EGPP_COUNT):
                            return GL_UNPACK_ALIGNMENT;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_WIDTH+EGPP_COUNT):
                            return GL_UNPACK_COMPRESSED_BLOCK_WIDTH;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_HEIGHT+EGPP_COUNT):
                            return GL_UNPACK_COMPRESSED_BLOCK_HEIGHT;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_DEPTH+EGPP_COUNT):
                            return GL_UNPACK_COMPRESSED_BLOCK_DEPTH;
                            break;
                        case (EGPP_COMPRESSED_BLOCK_SIZE+EGPP_COUNT):
                            return GL_UNPACK_COMPRESSED_BLOCK_SIZE;
                            break;
                        case (EGPP_ROW_LENGTH+EGPP_COUNT):
                            return GL_UNPACK_ROW_LENGTH;
                            break;
                        case (EGPP_IMAGE_HEIGHT+EGPP_COUNT):
                            return GL_UNPACK_IMAGE_HEIGHT;
                            break;
                        case (EGPP_SWAP_BYTES+EGPP_COUNT):
                            return GL_UNPACK_SWAP_BYTES;
                            break;
                        case (EGPP_SKIP_PIXELS+EGPP_COUNT):
                            return GL_UNPACK_SKIP_PIXELS;
                            break;
                        case (EGPP_SKIP_ROWS+EGPP_COUNT):
                            return GL_UNPACK_SKIP_PIXELS;
                            break;
                        case (EGPP_SKIP_IMAGES+EGPP_COUNT):
                            return GL_UNPACK_SKIP_IMAGES;
                            break;
                        default:
                            return GL_INVALID_ENUM;
                            break;
                    }
                }
            }

            inline static GLenum glTextureTypeToGLenum(const uint32_t texType)
            {
                switch (texType)
                {
                    case EGTT_1D:
                        return GL_TEXTURE_1D;
                        break;
                    case EGTT_1D_ARRAY:
                        return GL_TEXTURE_1D_ARRAY;
                        break;
                    case EGTT_2D:
                        return GL_TEXTURE_2D;
                        break;
                    case EGTT_2D_ARRAY:
                        return GL_TEXTURE_2D_ARRAY;
                        break;
                    case EGTT_2D_MULTISAMPLE:
                        return GL_TEXTURE_2D_MULTISAMPLE;
                        break;
                    case EGTT_2D_MULTISAMPLE_ARRAY:
                        return GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
                        break;
                    case EGTT_3D:
                        return GL_TEXTURE_3D;
                        break;
                    case EGTT_BUFFER:
                        return GL_TEXTURE_BUFFER;
                        break;
                    case EGTT_CUBE_MAP:
                        return GL_TEXTURE_CUBE_MAP;
                        break;
                    case EGTT_CUBE_MAP_ARRAY:
                        return GL_TEXTURE_CUBE_MAP_ARRAY;
                        break;
                    case EGTT_RECTANGLE:
                        return GL_TEXTURE_RECTANGLE;
                        break;
                    default:
                        return GL_INVALID_ENUM;
                        break;
                }
            }

            inline static GLenum glTextureTypeToBindingGLenum(const uint32_t texType)
            {
                switch (texType)
                {
                    case EGTT_1D:
                        return GL_TEXTURE_BINDING_1D;
                        break;
                    case EGTT_1D_ARRAY:
                        return GL_TEXTURE_BINDING_1D_ARRAY;
                        break;
                    case EGTT_2D:
                        return GL_TEXTURE_BINDING_2D;
                        break;
                    case EGTT_2D_ARRAY:
                        return GL_TEXTURE_BINDING_2D_ARRAY;
                        break;
                    case EGTT_2D_MULTISAMPLE:
                        return GL_TEXTURE_BINDING_2D_MULTISAMPLE;
                        break;
                    case EGTT_2D_MULTISAMPLE_ARRAY:
                        return GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
                        break;
                    case EGTT_3D:
                        return GL_TEXTURE_BINDING_3D;
                        break;
                    case EGTT_BUFFER:
                        return GL_TEXTURE_BINDING_BUFFER;
                        break;
                    case EGTT_CUBE_MAP:
                        return GL_TEXTURE_BINDING_CUBE_MAP;
                        break;
                    case EGTT_CUBE_MAP_ARRAY:
                        return GL_TEXTURE_BINDING_CUBE_MAP_ARRAY;
                        break;
                    case EGTT_RECTANGLE:
                        return GL_TEXTURE_BINDING_RECTANGLE;
                        break;
                    default:
                        return GL_INVALID_ENUM;
                        break;
                }
            }

            inline static GLenum glRangedBufferTypeToGLenum(const uint32_t rangedBufferType)
            {
                switch (rangedBufferType)
                {
                    case EGRBT_UNIFORM_BUFFER:
                        return GL_UNIFORM_BUFFER;
                        break;
                    case EGRBT_SHADER_STORAGE_BUFFER:
                        return GL_SHADER_STORAGE_BUFFER;
                        break;
                    default:
                        return GL_INVALID_ENUM;
                        break;
                }
            }

            //default OGL state at start of context as per the spec
            COpenGLState(const uint32_t windowSizeX=0, const uint32_t windowSizeY=0)
            {
                for (size_t i=0; i<EGHB_COUNT; i++)
                    glHint_vals[i] = GL_DONT_CARE;

                glProvokingVertex_val = GL_LAST_VERTEX_CONVENTION;

                size_t glEnableBitfieldByteSize = (EGEB_COUNT+63)/64;
                memset(glEnableBitfield,0,sizeof(uint64_t)*glEnableBitfieldByteSize);
                setGlEnableBit(EGEB_DITHER,true);
                setGlEnableBit(EGEB_MULTISAMPLE,true);

                size_t glEnableiBitfieldByteSize = (EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES+63)/64;
                memset(glEnableiBitfield,0,sizeof(uint64_t)*glEnableiBitfieldByteSize);

                glBindFramebuffer_vals[0] = 0;
                glBindFramebuffer_vals[1] = 0;


                glPolygonOffset_factor = 0.f;
                glPolygonOffset_units = 0.f;

                glClampColor_val = GL_FIXED_ONLY;
                memset(glPixelStorei_vals[0],0,sizeof(int32_t)*EGPP_COUNT); //pack
                memset(glPixelStorei_vals[1],0,sizeof(int32_t)*EGPP_COUNT); //unpack
                glPixelStorei_vals[0][EGPP_ALIGNMENT] = 4;
                glPixelStorei_vals[1][EGPP_ALIGNMENT] = 4;

                glPrimitiveRestartIndex_val = 0;

                glBindTransformFeedback_val = 0;

                glUseProgram_val = 0;
                glBindProgramPipeline_val = 0;

                glPatchParameteri_val = 3;
                glPatchParameterfv_inner[0] = 1.f;
                glPatchParameterfv_inner[1] = 1.f;
                glPatchParameterfv_outer[0] = 1.f;
                glPatchParameterfv_outer[1] = 1.f;
                glPatchParameterfv_outer[2] = 1.f;
                glPatchParameterfv_outer[3] = 1.f;

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

                glSampleCoverage_val = 1.f;
                glSampleCoverage_invert = false;

                for (uint32_t i=0; i<OGL_STATE_MAX_SAMPLE_MASK_WORDS; i++)
                    glSampleMaski_vals[i] = ~0x0u;

                glMinSampleShading_val = 0.f;


                for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
                {
                    glBlendEquationSeparatei_vals[i][0] = GL_FUNC_ADD;
                    glBlendEquationSeparatei_vals[i][1] = GL_FUNC_ADD;

                    glBlendFuncSeparatei_vals[i][0] = GL_ONE;
                    glBlendFuncSeparatei_vals[i][1] = GL_ZERO;
                    glBlendFuncSeparatei_vals[i][2] = GL_ONE;
                    glBlendFuncSeparatei_vals[i][3] = GL_ZERO;
                }
                for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS/16; i++)
                    glColorMaski_vals[i] = ~uint64_t(0);

                glStencilFuncSeparate_func[0] = GL_ALWAYS;
                glStencilFuncSeparate_func[1] = GL_ALWAYS;
                glStencilFuncSeparate_ref[0] = 0;
                glStencilFuncSeparate_ref[1] = 0;
                glStencilFuncSeparate_mask[0] = ~GLuint(0);
                glStencilFuncSeparate_mask[1] = ~GLuint(0);

                glStencilOpSeparate_sfail[0] = GL_KEEP;
                glStencilOpSeparate_sfail[1] = GL_KEEP;
                glStencilOpSeparate_dpfail[0] = GL_KEEP;
                glStencilOpSeparate_dpfail[1] = GL_KEEP;
                glStencilOpSeparate_dppass[0] = GL_KEEP;
                glStencilOpSeparate_dppass[1] = GL_KEEP;

                glStencilMaskSeparate_mask[0] = ~GLuint(0);
                glStencilMaskSeparate_mask[1] = ~GLuint(0);

                glDepthFunc_val = GL_LESS;
                glDepthMask_val = false;


                memset(boundBuffers,0,sizeof(GLuint)*EGBT_COUNT);

                for (size_t i=0; i<OGL_STATE_MAX_IMAGES; i++)
                {
                    glBindImageTexture_texture[i] = 0;
                    glBindImageTexture_level[i] = 0;
                    glBindImageTexture_layered[i] = GL_FALSE;
                    glBindImageTexture_layer[i] = 0;
                    glBindImageTexture_access[i] = GL_READ_ONLY;
                    glBindImageTexture_format[i] = GL_R8;
                }

                for (size_t i=0; i<OGL_STATE_MAX_TEXTURES; i++)
                    memset(boundTextures[i],0,sizeof(GLuint)*EGTT_COUNT);
                memset(boundSamplers,0,sizeof(GLuint)*OGL_STATE_MAX_TEXTURES);

                glPolygonMode_mode = GL_FILL;
                glFrontFace_val = GL_CCW;
                glCullFace_val = GL_BACK;

                boundVAO = 0;
            }

            //! THIS IS SLOW AND FOR DEBUG ONLY!
            static COpenGLState collectGLState(bool careAboutHints=true, //should be default false
                                              bool careAboutFBOs=true,
                                              bool careAboutPolygonOffset=true, //should be default false
                                              bool careAboutPixelXferOps=true,
                                              bool careAboutSSBOAndAtomicCounters=true,
                                              bool careAboutXFormFeedback=true,
                                              bool careAboutProgram=true,
                                              bool careAboutPipeline=true,
                                              bool careAboutTesellationParams=true,
                                              bool careAboutViewports=true,
                                              bool careAboutDrawIndirectBuffers=true,
                                              bool careAboutPointSize=true,
                                              bool careAboutLineWidth=true,
                                              bool careAboutLogicOp=true,
                                              bool careAboutMultisampling=true,
                                              bool careAboutBlending=true,
                                              bool careAboutColorWriteMasks=true,
                                              bool careAboutStencilFunc=true,
                                              bool careAboutStencilOp=true,
                                              bool careAboutStencilMask=true,
                                              bool careAboutDepthFunc=true,
                                              bool careAboutDepthMask=true,
                                              bool careAboutImages=true,
                                              bool careAboutTextures=true,
                                              bool careAboutFaceOrientOrCull=true,
                                              bool careAboutVAO=true,
                                              bool careAboutUBO=true);

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
                                              bool careAboutHints=true, //should be default false
                                              bool careAboutFBOs=true,
                                              bool careAboutPolygonOffset=true, //should be default false
                                              bool careAboutPixelXferOps=true,
                                              bool careAboutSSBOAndAtomicCounters=true,
                                              bool careAboutXFormFeedback=true,
                                              bool careAboutProgram=true,
                                              bool careAboutPipeline=true,
                                              bool careAboutTesellationParams=true,
                                              bool careAboutViewports=true,
                                              bool careAboutDrawIndirectBuffers=true,
                                              bool careAboutPointSize=true,
                                              bool careAboutLineWidth=true,
                                              bool careAboutLogicOp=true,
                                              bool careAboutMultisampling=true,
                                              bool careAboutBlending=true,
                                              bool careAboutColorWriteMasks=true,
                                              bool careAboutStencilFunc=true,
                                              bool careAboutStencilOp=true,
                                              bool careAboutStencilMask=true,
                                              bool careAboutDepthFunc=true,
                                              bool careAboutDepthMask=true,
                                              bool careAboutImages=true,
                                              bool careAboutTextures=true,
                                              bool careAboutFaceOrientOrCull=true,
                                              bool careAboutVAO=true,
                                              bool careAboutUBO=true) const
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
                                    diff.glHint_pair[diff.hintsToSet][0] = GL_FRAGMENT_SHADER_DERIVATIVE_HINT;
                                    break;
                                case EGHB_LINE_SMOOTH_HINT:
                                    diff.glHint_pair[diff.hintsToSet][0] = GL_LINE_SMOOTH_HINT;
                                    break;
                                case EGHB_POLYGON_SMOOTH_HINT:
                                    diff.glHint_pair[diff.hintsToSet][0] = GL_POLYGON_SMOOTH_HINT;
                                    break;
                                case EGHB_TEXTURE_COMPRESSION_HINT:
                                    diff.glHint_pair[diff.hintsToSet][0] = GL_TEXTURE_COMPRESSION_HINT;
                                    break;
                                default:
                                    diff.glHint_pair[diff.hintsToSet][0] = GL_INVALID_ENUM;
                                    break;
                            }
                            diff.glHint_pair[diff.hintsToSet][1] = glHint_vals[i];
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

                    if (diff.glDisableis[diff.glDisableiCount].flag!=GL_INVALID_ENUM)
                        diff.glDisableiCount++;
                    if (diff.glEnableis[diff.glEnableiCount].flag!=GL_INVALID_ENUM)
                        diff.glEnableiCount++;
                }


                if (careAboutFBOs)
                {
                    if (glBindFramebuffer_vals[0]!=previousState.glBindFramebuffer_vals[0])
                    {
                        diff.glBindFramebuffer_vals[0] = glBindFramebuffer_vals[0];
                        diff.bindFramebuffers |= 0x1u;
                    }
                    if (glBindFramebuffer_vals[1]!=previousState.glBindFramebuffer_vals[1])
                    {
                        diff.glBindFramebuffer_vals[1] = glBindFramebuffer_vals[1];
                        diff.bindFramebuffers |= 0x2u;
                    }
                }


                if (careAboutPolygonOffset&&(glPolygonOffset_factor!=previousState.glPolygonOffset_factor||glPolygonOffset_units!=previousState.glPolygonOffset_units))
                {
                    diff.resetPolygonOffset = false;
                    diff.glPolygonOffset_factor = glPolygonOffset_factor;
                    diff.glPolygonOffset_units = glPolygonOffset_units;
                }


                if (careAboutPixelXferOps)
                {
                    if (glClampColor_val!=previousState.glClampColor_val)
                        diff.glClampColor_val = glClampColor_val;

                    for (size_t i=0; i<EGPP_COUNT; i++)
                    {
                        if (glPixelStorei_vals[0][i]==previousState.glPixelStorei_vals[0][i])
                            continue;

                        diff.glPixelStorei_vals[diff.glPixelStoreiCount++] = std::pair<GLenum,int32_t>(glPackParamToGLenum(i),glPixelStorei_vals[0][i]);
                    }
                    for (size_t i=EGPP_COUNT; i<2*EGPP_COUNT; i++)
                    {
                        if (glPixelStorei_vals[1][i-EGPP_COUNT]==previousState.glPixelStorei_vals[1][i-EGPP_COUNT])
                            continue;

                        diff.glPixelStorei_vals[diff.glPixelStoreiCount++] = std::pair<GLenum,int32_t>(glPackParamToGLenum(i),glPixelStorei_vals[1][i-EGPP_COUNT]);
                    }

                    if (boundBuffers[EGBT_PIXEL_PACK_BUFFER]!=previousState.boundBuffers[EGBT_PIXEL_PACK_BUFFER])
                    {
                        diff.setBuffers[EGBT_PIXEL_PACK_BUFFER] = true;
                        diff.bindBuffer[EGBT_PIXEL_PACK_BUFFER] = boundBuffers[EGBT_PIXEL_PACK_BUFFER];
                    }
                    if (boundBuffers[EGBT_PIXEL_UNPACK_BUFFER]!=previousState.boundBuffers[EGBT_PIXEL_UNPACK_BUFFER])
                    {
                        diff.setBuffers[EGBT_PIXEL_UNPACK_BUFFER] = true;
                        diff.bindBuffer[EGBT_PIXEL_UNPACK_BUFFER] = boundBuffers[EGBT_PIXEL_UNPACK_BUFFER];
                    }
                }

                if (glPrimitiveRestartIndex_val!=previousState.glPrimitiveRestartIndex_val)
                {
                    diff.glPrimitiveRestartIndex_val = glPrimitiveRestartIndex_val;
                    diff.setPrimitiveRestartIndex = true;
                }

                if (careAboutXFormFeedback&&glBindTransformFeedback_val!=previousState.glBindTransformFeedback_val)
                {
                    diff.changeXFormFeedback = true;
                    diff.glBindTransformFeedback_val = glBindTransformFeedback_val;
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

                if (careAboutTesellationParams)
                {
                    if (glPatchParameteri_val!=previousState.glPatchParameteri_val)
                        diff.glPatchParameteri_val = glPatchParameteri_val;

                    if (glPatchParameterfv_inner[0]!=previousState.glPatchParameterfv_inner[0]||
                        glPatchParameterfv_inner[1]!=previousState.glPatchParameterfv_inner[1])
                    {
                        memcpy(diff.glPatchParameterfv_inner,glPatchParameterfv_inner,sizeof(float)*2);
                    }

                    if (glPatchParameterfv_outer[0]!=previousState.glPatchParameterfv_outer[0]||
                        glPatchParameterfv_outer[1]!=previousState.glPatchParameterfv_outer[1]||
                        glPatchParameterfv_outer[2]!=previousState.glPatchParameterfv_outer[2]||
                        glPatchParameterfv_outer[3]!=previousState.glPatchParameterfv_outer[3])
                    {
                        memcpy(diff.glPatchParameterfv_outer,glPatchParameterfv_outer,sizeof(float)*4);
                    }
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
                            memcpy(diff.glDepthRangeArray_vals[j],glDepthRangeArray_vals[i],2*sizeof(float));
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
                            memcpy(diff.glScissorArray_vals[j],glScissorArray_vals[i],4*sizeof(GLint));
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

                if (careAboutMultisampling)
                {
                    if (glSampleCoverage_val!=previousState.glSampleCoverage_val||glSampleCoverage_invert!=previousState.glSampleCoverage_invert)
                    {
                        diff.glSampleCoverage_val = glSampleCoverage_val;
                        diff.glSampleCoverage_invert = glSampleCoverage_val;
                    }

                    for (uint32_t i=0; i<OGL_STATE_MAX_SAMPLE_MASK_WORDS; i++)
                    {
                        if (glSampleMaski_vals[i]==previousState.glSampleMaski_vals[i])
                            continue;

                        diff.glSampleMaski_vals[i] = glSampleMaski_vals[i];
                        diff.setSampleMask = i;
                    }

                    if (glMinSampleShading_val!=previousState.glMinSampleShading_val)
                        diff.glMinSampleShading_val = glMinSampleShading_val;
                }

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
                            diff.setBlendEquation |= 0x1u<<i;
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
                            diff.setBlendFunc |= 0x1u<<i;
                            diff.glBlendFuncSeparatei_vals[j][0] = glBlendFuncSeparatei_vals[i][0];
                            diff.glBlendFuncSeparatei_vals[j][1] = glBlendFuncSeparatei_vals[i][1];
                            diff.glBlendFuncSeparatei_vals[j][2] = glBlendFuncSeparatei_vals[i][2];
                            diff.glBlendFuncSeparatei_vals[j][3] = glBlendFuncSeparatei_vals[i][3];
                            j++;
                        }
                    }
                }

                if (careAboutColorWriteMasks)
                {
                    for (uint32_t i=0; i<OGL_STATE_MAX_DRAW_BUFFERS; i++)
                    {
                        const uint64_t compareMask = 0xfull<<((i%16)*4);
                        if ((glColorMaski_vals[i/16]&compareMask) != (previousState.glColorMaski_vals[i/16]&compareMask))
                        {
                            diff.setColorMask |= 0x1u<<i;
                            diff.glColorMaski_vals[i/16] &= glColorMaski_vals[i/16] | (~compareMask);
                        }
                    }
                }

                if (careAboutStencilFunc)
                {
                    //front
                    if (glStencilFuncSeparate_func[0]!=previousState.glStencilFuncSeparate_func[0]||
                        glStencilFuncSeparate_ref[0]!=previousState.glStencilFuncSeparate_ref[0]||
                        glStencilFuncSeparate_mask[0]!=previousState.glStencilFuncSeparate_mask[0])
                    {
                        diff.glStencilFuncSeparate_func[0] = glStencilFuncSeparate_func[0];
                        diff.glStencilFuncSeparate_ref[0]  = glStencilFuncSeparate_ref[0];
                        diff.glStencilFuncSeparate_mask[0] = glStencilFuncSeparate_mask[0];
                        diff.setStencilFunc |= 0x1u;
                    }
                    //back
                    if (glStencilFuncSeparate_func[1]!=previousState.glStencilFuncSeparate_func[1]||
                        glStencilFuncSeparate_ref[1]!=previousState.glStencilFuncSeparate_ref[1]||
                        glStencilFuncSeparate_mask[1]!=previousState.glStencilFuncSeparate_mask[1])
                    {
                        diff.glStencilFuncSeparate_func[1] = glStencilFuncSeparate_func[1];
                        diff.glStencilFuncSeparate_ref[1]  = glStencilFuncSeparate_ref[1];
                        diff.glStencilFuncSeparate_mask[1] = glStencilFuncSeparate_mask[1];
                        diff.setStencilFunc |= 0x2u;
                    }
                }

                if (careAboutStencilOp)
                {
                    //front
                    if (glStencilOpSeparate_sfail[0]!=previousState.glStencilOpSeparate_sfail[0]||
                        glStencilOpSeparate_dpfail[0]!=previousState.glStencilOpSeparate_dpfail[0]||
                        glStencilOpSeparate_dppass[0]!=previousState.glStencilOpSeparate_dppass[0])
                    {
                        diff.glStencilOpSeparate_sfail[0]   = glStencilOpSeparate_sfail[0];
                        diff.glStencilOpSeparate_dpfail[0]  = glStencilOpSeparate_dpfail[0];
                        diff.glStencilOpSeparate_dppass[0]  = glStencilOpSeparate_dppass[0];
                        diff.setStencilOp |= 0x1u;
                    }
                    //back
                    if (glStencilOpSeparate_sfail[1]!=previousState.glStencilOpSeparate_sfail[1]||
                        glStencilOpSeparate_dpfail[1]!=previousState.glStencilOpSeparate_dpfail[1]||
                        glStencilOpSeparate_dppass[1]!=previousState.glStencilOpSeparate_dppass[1])
                    {
                        diff.glStencilOpSeparate_sfail[1]   = glStencilOpSeparate_sfail[1];
                        diff.glStencilOpSeparate_dpfail[1]  = glStencilOpSeparate_dpfail[1];
                        diff.glStencilOpSeparate_dppass[1]  = glStencilOpSeparate_dppass[1];
                        diff.setStencilOp |= 0x2u;
                    }
                }

                if (careAboutStencilMask)
                {
                    //front
                    if (glStencilMaskSeparate_mask[0]!=previousState.glStencilMaskSeparate_mask[0])
                    {
                        diff.glStencilMaskSeparate_mask[0]   = glStencilMaskSeparate_mask[0];
                        diff.setStencilMask |= 0x1u;
                    }
                    //back
                    if (glStencilMaskSeparate_mask[1]!=previousState.glStencilMaskSeparate_mask[1])
                    {
                        diff.glStencilMaskSeparate_mask[1]   = glStencilMaskSeparate_mask[1];
                        diff.setStencilMask |= 0x2u;
                    }
                }

                if (careAboutDepthFunc&&glDepthFunc_val!=previousState.glDepthFunc_val)
                    diff.glDepthFunc_val = glDepthFunc_val;

                if (careAboutDepthMask&&glDepthMask_val!=previousState.glDepthMask_val)
                {
                    diff.setDepthMask = glDepthMask_val ? 0x81u:0x80u;
                }


                if (careAboutDrawIndirectBuffers)
                {
                    if (boundBuffers[EGBT_DRAW_INDIRECT_BUFFER]!=previousState.boundBuffers[EGBT_DRAW_INDIRECT_BUFFER])
                    {
                        diff.setBuffers[EGBT_DRAW_INDIRECT_BUFFER] = true;
                        diff.bindBuffer[EGBT_DRAW_INDIRECT_BUFFER] = boundBuffers[EGBT_DRAW_INDIRECT_BUFFER];
                    }
                    if (boundBuffers[EGBT_DISPATCH_INDIRECT_BUFFER]!=previousState.boundBuffers[EGBT_DISPATCH_INDIRECT_BUFFER])
                    {
                        diff.setBuffers[EGBT_DISPATCH_INDIRECT_BUFFER] = true;
                        diff.bindBuffer[EGBT_DISPATCH_INDIRECT_BUFFER] = boundBuffers[EGBT_DISPATCH_INDIRECT_BUFFER];
                    }
                }


                if (careAboutSSBOAndAtomicCounters)
                {
                    for (uint32_t j=0; j<OGL_MAX_BUFFER_BINDINGS; j++)
                    {
                        if (boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][j]!=previousState.boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][j])
                            diff.bindBufferRange[diff.setBufferRanges++] = COpenGLStateDiff::RangedBufferBindingDiff(GL_SHADER_STORAGE_BUFFER,j,boundBufferRanges[EGRBT_SHADER_STORAGE_BUFFER][j]);
                    }
                }

                if (careAboutImages)
                {
                    for (uint32_t i=0; i<OGL_STATE_MAX_IMAGES; i++)
                    {
                        if (glBindImageTexture_texture[i]!=previousState.glBindImageTexture_texture[i]||
                            glBindImageTexture_level[i]!=previousState.glBindImageTexture_level[i]||
                            glBindImageTexture_layered[i]!=previousState.glBindImageTexture_layered[i]||
                            glBindImageTexture_layer[i]!=previousState.glBindImageTexture_layer[i]||
                            glBindImageTexture_access[i]!=previousState.glBindImageTexture_access[i]||
                            glBindImageTexture_format[i]!=previousState.glBindImageTexture_format[i])
                        {
                            diff.bindImages[diff.setImageBindings++] = COpenGLStateDiff::ImageToBind(i,
                                                                        glBindImageTexture_texture[i],
                                                                        glBindImageTexture_level[i],
                                                                        glBindImageTexture_layered[i],
                                                                        glBindImageTexture_layer[i],
                                                                        glBindImageTexture_access[i],
                                                                        glBindImageTexture_format[i]);
                        }
                    }
                }

                if (careAboutTextures)
                {
                    for (uint32_t i=0; i<OGL_STATE_MAX_TEXTURES; i++)
                    {
                        for (uint32_t j=0; j<EGTT_COUNT; j++)
                        {
                            if (boundTextures[i][j]!=previousState.boundTextures[i][j])
                                diff.bindTextures[diff.texturesToBind++] = COpenGLStateDiff::TextureToBind(i,boundTextures[i][j],glTextureTypeToGLenum(j));
                        }

                        if (boundSamplers[i]!=previousState.boundSamplers[i])
                            diff.bindSamplers[diff.samplersToBind++] = COpenGLStateDiff::IndexedObjectToBind(i,boundSamplers[i]);
                    }
                }

                if (careAboutFaceOrientOrCull)
                {
                    if (glPolygonMode_mode!=previousState.glPolygonMode_mode)
                        diff.glPolygonMode_mode = glPolygonMode_mode;

                    if (glFrontFace_val!=previousState.glFrontFace_val)
                        diff.glFrontFace_val = glFrontFace_val;

                    if (glCullFace_val!=previousState.glCullFace_val)
                        diff.glCullFace_val = glCullFace_val;
                }

                if (careAboutVAO&&boundVAO!=previousState.boundVAO)
                {
                    diff.setVAO = true;
                    diff.bindVAO = boundVAO;
                }

                if (careAboutUBO)
                {
                    for (uint32_t j=0; j<OGL_MAX_BUFFER_BINDINGS; j++)
                    {
                        if (boundBufferRanges[EGRBT_UNIFORM_BUFFER][j]!=previousState.boundBufferRanges[EGRBT_UNIFORM_BUFFER][j])
                            diff.bindBufferRange[diff.setBufferRanges++] = COpenGLStateDiff::RangedBufferBindingDiff(GL_UNIFORM_BUFFER,j,boundBufferRanges[EGRBT_UNIFORM_BUFFER][j]);
                    }
                }

                return diff;
            }

            inline COpenGLStateDiff operator^(const COpenGLState &previousState) const {return getStateDiff(previousState);}

            inline bool operator!=(const COpenGLState &previousState) const
            {
                COpenGLStateDiff diff = (*this)^previousState;

                if (diff.hintsToSet)
                    return true;
                if (diff.glProvokingVertex_val!=GL_INVALID_ENUM)
                    return true;
                if (diff.glDisableCount)
                    return true;
                if (diff.glEnableCount)
                    return true;
                if (diff.glDisableiCount)
                    return true;
                if (diff.glEnableiCount)
                    return true;
                if (diff.bindFramebuffers)
                    return true;
                if (diff.resetPolygonOffset)
                    return true;
                if (diff.glPrimitiveSize[0]>-FLT_MAX)
                    return true;
                if (diff.glPrimitiveSize[1]>-FLT_MAX)
                    return true;
                if (diff.glClampColor_val!=GL_INVALID_ENUM)
                    return true;
                if (diff.glPixelStoreiCount)
                    return true;
                if (diff.setBuffers[EGBT_PIXEL_UNPACK_BUFFER])
                    return true;
                if (diff.setBuffers[EGBT_PIXEL_PACK_BUFFER])
                    return true;
                if (diff.setPrimitiveRestartIndex)
                    return true;
                if (diff.changeXFormFeedback)
                    return true;
                if (diff.changeGlProgram)
                    return true;
                if (diff.changeGlProgramPipeline)
                    return true;
                if (diff.glPatchParameteri_val)
                    return true;
                if (diff.glPatchParameterfv_inner[0]>-FLT_MAX)
                    return true;
                if (diff.glPatchParameterfv_outer[0]>-FLT_MAX)
                    return true;
                if (diff.setBufferRanges)
                    return true;
                if (diff.setDepthRange)
                    return true;
                if (diff.setViewportArray)
                    return true;
                if (diff.setScissorBox)
                    return true;
                if (diff.setBuffers[EGBT_DRAW_INDIRECT_BUFFER]||diff.setBuffers[EGBT_DISPATCH_INDIRECT_BUFFER])
                    return true;
                if (diff.glLogicOp_val!=GL_INVALID_ENUM)
                    return true;
                if (diff.glSampleCoverage_val>-FLT_MAX)
                    return true;
                if (diff.setSampleMask)
                    return true;
                if (diff.glMinSampleShading_val>-FLT_MAX)
                    return true;
                if (diff.setBlendColor)
                    return true;
                if (diff.setBlendEquation)
                    return true;
                if (diff.setBlendFunc)
                    return true;
                if (diff.setColorMask)
                    return true;
                if (diff.setStencilFunc)
                    return true;
                if (diff.setStencilOp)
                    return true;
                if (diff.setStencilMask)
                    return true;
                if (diff.setDepthMask)
                    return true;
                if (diff.glDepthFunc_val!=GL_INVALID_ENUM)
                    return true;
                if (diff.setImageBindings)
                    return true;
                if (diff.texturesToBind||diff.samplersToBind)
                    return true;
                if (diff.glPolygonMode_mode!=GL_INVALID_ENUM||
                    diff.glFrontFace_val!=GL_INVALID_ENUM||
                    diff.glCullFace_val!=GL_INVALID_ENUM)
                    return true;
                if (diff.bindVAO)
                    return true;

                return false;
            }




            GLenum glHint_vals[EGHB_COUNT];
            GLenum glProvokingVertex_val;

            uint64_t glEnableBitfield[(EGEB_COUNT+63)/64];
            inline bool getGlEnableBit(const E_GL_ENABLE_BIT bit) const
            {
                return glEnableBitfield[bit/64]&(uint64_t(0x1ull)<<(bit%64));
            }
            inline void setGlEnableBit(const E_GL_ENABLE_BIT bit, bool value)
            {
                if (value)
                    glEnableBitfield[bit/64] |= uint64_t(0x1ull)<<(bit%64);
                else
                    glEnableBitfield[bit/64] &= ~(uint64_t(0x1ull)<<(bit%64));
            }

            uint64_t glEnableiBitfield[(EGEIB_COUNT*OGL_MAX_ENDISABLEI_INDICES+63)/64];
            inline bool getGlEnableiBit(uint64_t bit, const uint32_t index) const
            {
                bit *= OGL_MAX_ENDISABLEI_INDICES;
                bit += index;
                return glEnableiBitfield[bit/64]&(uint64_t(0x1ull)<<(bit%64));
            }
            inline void setGlEnableiBit(uint64_t bit, const uint32_t index, bool value)
            {
                bit *= OGL_MAX_ENDISABLEI_INDICES;
                bit += index;
                if (value)
                    glEnableiBitfield[bit/64] |= uint64_t(0x1ull)<<(bit%64);
                else
                    glEnableiBitfield[bit/64] &= ~(uint64_t(0x1ull)<<(bit%64));
            }

            //FBO
            GLuint glBindFramebuffer_vals[2];


            float glPolygonOffset_factor,glPolygonOffset_units;
            float glPrimitiveSize[2]; //glPointSize, glLineWidth

            GLenum glClampColor_val;
            int32_t glPixelStorei_vals[2][EGPP_COUNT];

            GLuint glPrimitiveRestartIndex_val;

            GLuint glBindTransformFeedback_val;

            GLuint glUseProgram_val,glBindProgramPipeline_val;

            GLint glPatchParameteri_val;
            float glPatchParameterfv_inner[2];
            float glPatchParameterfv_outer[4];

            float glDepthRangeArray_vals[OGL_STATE_MAX_VIEWPORTS][2];
            float glViewportArray_vals[OGL_STATE_MAX_VIEWPORTS][4];
            GLint glScissorArray_vals[OGL_STATE_MAX_VIEWPORTS][4];

            GLenum glLogicOp_val;

            float glSampleCoverage_val;
            bool glSampleCoverage_invert;
            uint32_t glSampleMaski_vals[OGL_STATE_MAX_SAMPLE_MASK_WORDS];
            float glMinSampleShading_val;

            float glBlendColor_vals[4];
            GLenum glBlendEquationSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][2];
            GLenum glBlendFuncSeparatei_vals[OGL_STATE_MAX_DRAW_BUFFERS][4];
            uint64_t glColorMaski_vals[OGL_STATE_MAX_DRAW_BUFFERS/16];

            GLenum  glStencilFuncSeparate_func[2];
            GLint   glStencilFuncSeparate_ref[2];
            GLuint  glStencilFuncSeparate_mask[2];
            GLenum  glStencilOpSeparate_sfail[2];
            GLenum  glStencilOpSeparate_dpfail[2];
            GLenum  glStencilOpSeparate_dppass[2];
            GLuint  glStencilMaskSeparate_mask[2];

            GLenum  glDepthFunc_val;
            bool    glDepthMask_val;

            //BUFFER BINDING POINTS
            GLuint boundBuffers[EGBT_COUNT];
            COpenGLStateDiff::RangedBufferBinding boundBufferRanges[EGRBT_COUNT][OGL_MAX_BUFFER_BINDINGS];


            GLuint glBindImageTexture_texture[OGL_STATE_MAX_IMAGES];
            GLint glBindImageTexture_level[OGL_STATE_MAX_IMAGES];
            GLboolean glBindImageTexture_layered[OGL_STATE_MAX_IMAGES];
            GLint glBindImageTexture_layer[OGL_STATE_MAX_IMAGES];
            GLenum glBindImageTexture_access[OGL_STATE_MAX_IMAGES];
            GLenum glBindImageTexture_format[OGL_STATE_MAX_IMAGES];

            GLuint boundTextures[OGL_STATE_MAX_TEXTURES][EGTT_COUNT];
            GLuint boundSamplers[OGL_STATE_MAX_TEXTURES];

            GLenum glPolygonMode_mode;
            GLenum glFrontFace_val;
            GLenum glCullFace_val;
            //VAO
            GLuint boundVAO;
        private:
            inline void setEnableDiffBits(COpenGLStateDiff& diff, const uint64_t bitdiff, const uint64_t i, const uint64_t j) const
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
            inline void setEnableiDiffBits(COpenGLStateDiff& diff, const uint64_t bitdiff, const uint64_t i, const uint64_t j) const
            {
                const uint64_t bitFlag = uint64_t(0x1ull)<<j;
                if (bitdiff&bitFlag)
                {
                    uint16_t ix;
                    GLenum combo = glEnableiBitToGLenum(ix,i*64+j);
                    if (glEnableiBitfield[i]&bitFlag)
                    {
                        diff.glEnableis[diff.glEnableiCount].indices |= ix;

                        GLenum currVal = diff.glEnableis[diff.glEnableiCount].flag;
                        if (diff.glEnableis[diff.glEnableiCount].flag==GL_INVALID_ENUM)
                            diff.glEnableis[diff.glEnableiCount].flag = combo;
                        else if (currVal!=combo)
                            diff.glEnableiCount++;
                    }
                    else
                    {
                        diff.glDisableis[diff.glDisableiCount].indices |= ix;

                        GLenum currVal = diff.glDisableis[diff.glDisableiCount].flag;
                        if (diff.glDisableis[diff.glDisableiCount].flag==GL_INVALID_ENUM)
                            diff.glDisableis[diff.glDisableiCount].flag = combo;
                        else if (currVal!=combo)
                            diff.glDisableiCount++;
                    }
                }
            }
	};


} // end namespace video
} // end namespace irr

#endif

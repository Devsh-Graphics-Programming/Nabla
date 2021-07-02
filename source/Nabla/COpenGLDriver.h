// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_VIDEO_OPEN_GL_H_INCLUDED__
#define __NBL_C_VIDEO_OPEN_GL_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/compile_config.h"

#include "SIrrCreationParameters.h"

namespace nbl
{
	class CIrrDeviceWin32;
	class CIrrDeviceLinux;
	class CIrrDeviceSDL;
	class CIrrDeviceMacOSX;
}

#ifdef _NBL_COMPILE_WITH_OPENGL_

#include "IDriverMemoryAllocation.h"
#include "nbl/video/COpenGLSpecializedShader.h"
#include "nbl/video/COpenGLRenderpassIndependentPipeline.h"
#include "nbl/video/COpenGLDescriptorSet.h"
#include "nbl/video/COpenGLPipelineLayout.h"
#include "nbl/video/COpenGLComputePipeline.h"

#include "CNullDriver.h"
// also includes the OpenGL stuff
#include "COpenGLFrameBuffer.h"
#include "COpenGLDriverFence.h"
#include "nbl/video/CCUDAHandler.h"
#include "COpenCLHandler.h"

#include <map>
//#include <atomic>
//#include <thread>
#include <mutex>
//#include <condition_variable>

namespace nbl
{
namespace video
{

    enum GL_STATE_BITS : uint32_t
    {
        // has to be flushed before constants are pushed (before `extGlProgramUniform*`)
        GSB_PIPELINE = 1u << 0,
        GSB_RASTER_PARAMETERS = 1u << 1,
        // we want the two to happen together and just before a draw (set VAO first, then binding)
        GSB_VAO_AND_VERTEX_INPUT = 1u << 2,
        // flush just before (indirect)dispatch or (multi)(indirect)draw, textures and samplers first, then storage image, then SSBO, finally UBO
        GSB_DESCRIPTOR_SETS = 1u << 3,
        // GL_DISPATCH_INDIRECT_BUFFER 
        GSB_DISPATCH_INDIRECT = 1u << 4,
        GSB_PUSH_CONSTANTS = 1u << 5,
        GSB_PIXEL_PACK_UNPACK = 1u << 6,
        // flush everything
        GSB_ALL = ~0x0u
    };


struct SOpenGLState
{
    struct SVAO {
        GLuint GLname;
        uint64_t lastUsed;
    };
    struct HashVAOPair
    {
		COpenGLRenderpassIndependentPipeline::SVAOHash first = {};
		SVAO second = { 0u,0ull };
		//extra vao state being cached
		std::array<asset::SBufferBinding<const COpenGLBuffer>, IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT> vtxBindings;
		core::smart_refctd_ptr<const COpenGLBuffer> idxBinding;

        inline bool operator<(const HashVAOPair& rhs) const { return first < rhs.first; }
    };
    struct SDescSetBnd {
        core::smart_refctd_ptr<const COpenGLPipelineLayout> pplnLayout;
        core::smart_refctd_ptr<const COpenGLDescriptorSet> set;
        core::smart_refctd_dynamic_array<uint32_t> dynamicOffsets;
    };

    using SGraphicsPipelineHash = std::array<GLuint, COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT>;

    struct {
        struct {
            core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline> pipeline;
            SGraphicsPipelineHash usedShadersHash = { 0u, 0u, 0u, 0u, 0u };
			GLuint usedPipeline = 0u;
        } graphics;
        struct {
            core::smart_refctd_ptr<const COpenGLComputePipeline> pipeline;
            GLuint usedShader = 0u;
        } compute;
    } pipeline;

    struct {
        core::smart_refctd_ptr<const COpenGLBuffer> buffer;
    } dispatchIndirect;

    struct {
        //in GL it is possible to set polygon mode separately for back- and front-faces, but in VK it's one setting for both
        GLenum polygonMode = GL_FILL;
        GLenum faceCullingEnable = 0;
        GLenum cullFace = GL_BACK;
        //in VK stencil params (both: stencilOp and stencilFunc) are 2 distinct for back- and front-faces, but in GL it's one for both
        struct SStencilOp {
            GLenum sfail = GL_KEEP;
            GLenum dpfail = GL_KEEP;
            GLenum dppass = GL_KEEP;
            bool operator!=(const SStencilOp& rhs) const { return sfail!=rhs.sfail || dpfail!=rhs.dpfail || dppass!=rhs.dppass; }
        };
        SStencilOp stencilOp_front, stencilOp_back;
        struct SStencilFunc {
            GLenum func = GL_ALWAYS;
            GLint ref = 0;
            GLuint mask = ~static_cast<GLuint>(0u);
            bool operator!=(const SStencilFunc& rhs) const { return func!=rhs.func || ref!=rhs.ref || mask!=rhs.mask; }
        };
        SStencilFunc stencilFunc_front, stencilFunc_back;
        GLenum depthFunc = GL_LESS;
        GLenum frontFace = GL_CCW;
        GLboolean depthClampEnable = 0;
        GLboolean rasterizerDiscardEnable = 0;
        GLboolean polygonOffsetEnable = 0;
        struct SPolyOffset {
            GLfloat factor = 0.f;//depthBiasSlopeFactor 
            GLfloat units = 0.f;//depthBiasConstantFactor 
            bool operator!=(const SPolyOffset& rhs) const { return factor!=rhs.factor || units!=rhs.units; }
        } polygonOffset;
        GLfloat lineWidth = 1.f;
        GLboolean sampleShadingEnable = 0;
        GLfloat minSampleShading = 0.f;
        GLboolean sampleMaskEnable = 0;
        GLbitfield sampleMask[2]{~static_cast<GLbitfield>(0), ~static_cast<GLbitfield>(0)};
        GLboolean sampleAlphaToCoverageEnable = 0;
        GLboolean sampleAlphaToOneEnable = 0;
        GLboolean depthTestEnable = 0;
        GLboolean depthWriteEnable = 1;
        //GLboolean depthBoundsTestEnable;
        GLboolean stencilTestEnable = 0;
        GLboolean multisampleEnable = 1;
        GLboolean primitiveRestartEnable = 0;

        GLboolean logicOpEnable = 0;
        GLenum logicOp = GL_COPY;
        struct SDrawbufferBlending
        {
            GLboolean blendEnable = 0;
            struct SBlendFunc {
                GLenum srcRGB = GL_ONE;
                GLenum dstRGB = GL_ZERO;
                GLenum srcAlpha = GL_ONE;
                GLenum dstAlpha = GL_ZERO;
                bool operator!=(const SBlendFunc& rhs) const { return srcRGB!=rhs.srcRGB || dstRGB!=rhs.dstRGB || srcAlpha!=rhs.srcAlpha || dstAlpha!=rhs.dstAlpha; }
            } blendFunc;
            struct SBlendEq {
                GLenum modeRGB = GL_FUNC_ADD;
                GLenum modeAlpha = GL_FUNC_ADD;
                bool operator!=(const SBlendEq& rhs) const { return modeRGB!=rhs.modeRGB || modeAlpha!=rhs.modeAlpha; }
            } blendEquation;
            struct SColorWritemask {
                GLboolean colorWritemask[4]{ 1,1,1,1 };
                bool operator!=(const SColorWritemask& rhs) const { return memcmp(colorWritemask, rhs.colorWritemask, 4); }
            } colorMask;
        } drawbufferBlend[asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT];
    } rasterParams;

    struct {
		HashVAOPair vao = {};

        //putting it here because idk where else
        core::smart_refctd_ptr<const COpenGLBuffer> indirectDrawBuf;
        core::smart_refctd_ptr<const COpenGLBuffer> parameterBuf;//GL>=4.6
    } vertexInputParams;

    struct {
        SDescSetBnd descSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
    } descriptorsParams[E_PIPELINE_BIND_POINT::EPBP_COUNT];

    struct SPixelPackUnpack {
        core::smart_refctd_ptr<const COpenGLBuffer> buffer;
        GLint alignment = 4;
        GLint rowLength = 0;
        GLint imgHeight = 0;
        GLint BCwidth = 0;
        GLint BCheight = 0;
        GLint BCdepth = 0;
    };
    SPixelPackUnpack pixelPack;
    SPixelPackUnpack pixelUnpack;
};

// GCC is special
template<E_PIPELINE_BIND_POINT>
struct pipeline_for_bindpoint;
template<> struct pipeline_for_bindpoint<EPBP_COMPUTE > { using type = COpenGLComputePipeline; };
template<> struct pipeline_for_bindpoint<EPBP_GRAPHICS> { using type = COpenGLRenderpassIndependentPipeline; };

template<E_PIPELINE_BIND_POINT PBP>
using pipeline_for_bindpoint_t = typename pipeline_for_bindpoint<PBP>::type;


class COpenGLDriver final : public CNullDriver, public COpenGLExtensionHandler
{
    protected:
		//! destructor
		virtual ~COpenGLDriver();

		//! inits the parts of the open gl driver used on all platforms
		bool genericDriverInit(asset::IAssetManager* assMgr) override;

	public:
        struct SAuxContext;

		#ifdef _NBL_COMPILE_WITH_WINDOWS_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceWin32* device, const asset::IGLSLCompiler* glslcomp);
		//! inits the windows specific parts of the open gl driver
		bool initDriver(CIrrDeviceWin32* device);
		bool changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceWin32* device);
		#endif

		#ifdef _NBL_COMPILE_WITH_X11_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceLinux* device, const asset::IGLSLCompiler* glslcomp);
		//! inits the GLX specific parts of the open gl driver
		bool initDriver(CIrrDeviceLinux* device, SAuxContext* auxCtxts);
		bool changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceLinux* device);
		#endif

		#ifdef _NBL_COMPILE_WITH_SDL_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceSDL* device, const asset::IGLSLCompiler* glslcomp);
		#endif


        inline bool isAllowedBufferViewFormat(asset::E_FORMAT _fmt) const override
        {
            using namespace asset;
            switch (_fmt)
            {
				case EF_R8_UNORM: [[fallthrough]];
				case EF_R16_UNORM: [[fallthrough]];
				case EF_R16_SFLOAT: [[fallthrough]];
				case EF_R32_SFLOAT: [[fallthrough]];
				case EF_R8_SINT: [[fallthrough]];
				case EF_R16_SINT: [[fallthrough]];
				case EF_R32_SINT: [[fallthrough]];
				case EF_R8_UINT: [[fallthrough]];
				case EF_R16_UINT: [[fallthrough]];
				case EF_R32_UINT: [[fallthrough]];
				case EF_R8G8_UNORM: [[fallthrough]];
				case EF_R16G16_UNORM: [[fallthrough]];
				case EF_R16G16_SFLOAT: [[fallthrough]];
				case EF_R32G32_SFLOAT: [[fallthrough]];
				case EF_R8G8_SINT: [[fallthrough]];
				case EF_R16G16_SINT: [[fallthrough]];
				case EF_R32G32_SINT: [[fallthrough]];
				case EF_R8G8_UINT: [[fallthrough]];
				case EF_R16G16_UINT: [[fallthrough]];
				case EF_R32G32_UINT: [[fallthrough]];
				case EF_R32G32B32_SFLOAT: [[fallthrough]];
				case EF_R32G32B32_SINT: [[fallthrough]];
				case EF_R32G32B32_UINT: [[fallthrough]];
				case EF_R8G8B8A8_UNORM: [[fallthrough]];
				case EF_R16G16B16A16_UNORM: [[fallthrough]];
				case EF_R16G16B16A16_SFLOAT: [[fallthrough]];
				case EF_R32G32B32A32_SFLOAT: [[fallthrough]];
				case EF_R8G8B8A8_SINT: [[fallthrough]];
				case EF_R16G16B16A16_SINT: [[fallthrough]];
				case EF_R32G32B32A32_SINT: [[fallthrough]];
				case EF_R8G8B8A8_UINT: [[fallthrough]];
				case EF_R16G16B16A16_UINT: [[fallthrough]];
				case EF_R32G32B32A32_UINT:
					return true;
					break;
				default:
					return false;
					break;
            }
        }

        inline bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const override
        {
            using namespace asset;
            switch (_fmt)
            {
				// signed/unsigned byte
				case EF_R8_UNORM:
				case EF_R8_SNORM:
				case EF_R8_UINT:
				case EF_R8_SINT:
				case EF_R8G8_UNORM:
				case EF_R8G8_SNORM:
				case EF_R8G8_UINT:
				case EF_R8G8_SINT:
				case EF_R8G8B8_UNORM:
				case EF_R8G8B8_SNORM:
				case EF_R8G8B8_UINT:
				case EF_R8G8B8_SINT:
				case EF_R8G8B8A8_UNORM:
				case EF_R8G8B8A8_SNORM:
				case EF_R8G8B8A8_UINT:
				case EF_R8G8B8A8_SINT:
				case EF_R8_USCALED:
				case EF_R8_SSCALED:
				case EF_R8G8_USCALED:
				case EF_R8G8_SSCALED:
				case EF_R8G8B8_USCALED:
				case EF_R8G8B8_SSCALED:
				case EF_R8G8B8A8_USCALED:
				case EF_R8G8B8A8_SSCALED:
				// unsigned byte BGRA (normalized only)
				case EF_B8G8R8A8_UNORM:
				// unsigned/signed short
				case EF_R16_UNORM:
				case EF_R16_SNORM:
				case EF_R16_UINT:
				case EF_R16_SINT:
				case EF_R16G16_UNORM:
				case EF_R16G16_SNORM:
				case EF_R16G16_UINT:
				case EF_R16G16_SINT:
				case EF_R16G16B16_UNORM:
				case EF_R16G16B16_SNORM:
				case EF_R16G16B16_UINT:
				case EF_R16G16B16_SINT:
				case EF_R16G16B16A16_UNORM:
				case EF_R16G16B16A16_SNORM:
				case EF_R16G16B16A16_UINT:
				case EF_R16G16B16A16_SINT:
				case EF_R16_USCALED:
				case EF_R16_SSCALED:
				case EF_R16G16_USCALED:
				case EF_R16G16_SSCALED:
				case EF_R16G16B16_USCALED:
				case EF_R16G16B16_SSCALED:
				case EF_R16G16B16A16_USCALED:
				case EF_R16G16B16A16_SSCALED:
				// unsigned/signed int
				case EF_R32_UINT:
				case EF_R32_SINT:
				case EF_R32G32_UINT:
				case EF_R32G32_SINT:
				case EF_R32G32B32_UINT:
				case EF_R32G32B32_SINT:
				case EF_R32G32B32A32_UINT:
				case EF_R32G32B32A32_SINT:
				// unsigned/signed rgb10a2 BGRA (normalized only)
				case EF_A2R10G10B10_UNORM_PACK32:
				case EF_A2R10G10B10_SNORM_PACK32:
				// unsigned/signed rgb10a2
				case EF_A2B10G10R10_UNORM_PACK32:
				case EF_A2B10G10R10_SNORM_PACK32:
				case EF_A2B10G10R10_UINT_PACK32:
				case EF_A2B10G10R10_SINT_PACK32:
				case EF_A2B10G10R10_SSCALED_PACK32:
				case EF_A2B10G10R10_USCALED_PACK32:
				// GL_UNSIGNED_INT_10F_11F_11F_REV
				case EF_B10G11R11_UFLOAT_PACK32:
				// half float
				case EF_R16_SFLOAT:
				case EF_R16G16_SFLOAT:
				case EF_R16G16B16_SFLOAT:
				case EF_R16G16B16A16_SFLOAT:
				// float
				case EF_R32_SFLOAT:
				case EF_R32G32_SFLOAT:
				case EF_R32G32B32_SFLOAT:
				case EF_R32G32B32A32_SFLOAT:
				// double
				case EF_R64_SFLOAT:
				case EF_R64G64_SFLOAT:
				case EF_R64G64B64_SFLOAT:
				case EF_R64G64B64A64_SFLOAT:
					return true;
				default: return false;
            }
        }
        inline bool isColorRenderableFormat(asset::E_FORMAT _fmt) const override
        {
            using namespace asset;
            switch (_fmt)
            {
				case EF_A1R5G5B5_UNORM_PACK16:
				case EF_B5G6R5_UNORM_PACK16:
				case EF_R5G6B5_UNORM_PACK16:
				case EF_R4G4_UNORM_PACK8:
				case EF_R4G4B4A4_UNORM_PACK16:
				case EF_B4G4R4A4_UNORM_PACK16:
				case EF_R8_UNORM:
				case EF_R8_SNORM:
				case EF_R8_UINT:
				case EF_R8_SINT:
				case EF_R8G8_UNORM:
				case EF_R8G8_SNORM:
				case EF_R8G8_UINT:
				case EF_R8G8_SINT:
				case EF_R8G8B8_UNORM:
				case EF_R8G8B8_SNORM:
				case EF_R8G8B8_UINT:
				case EF_R8G8B8_SINT:
				case EF_R8G8B8_SRGB:
				case EF_R8G8B8A8_UNORM:
				case EF_R8G8B8A8_SNORM:
				case EF_R8G8B8A8_UINT:
				case EF_R8G8B8A8_SINT:
				case EF_R8G8B8A8_SRGB:
				case EF_A8B8G8R8_UNORM_PACK32:
				case EF_A8B8G8R8_SNORM_PACK32:
				case EF_A8B8G8R8_UINT_PACK32:
				case EF_A8B8G8R8_SINT_PACK32:
				case EF_A8B8G8R8_SRGB_PACK32:
				case EF_A2B10G10R10_UNORM_PACK32:
				case EF_A2B10G10R10_UINT_PACK32:
				case EF_R16_UNORM:
				case EF_R16_SNORM:
				case EF_R16_UINT:
				case EF_R16_SINT:
				case EF_R16_SFLOAT:
				case EF_R16G16_UNORM:
				case EF_R16G16_SNORM:
				case EF_R16G16_UINT:
				case EF_R16G16_SINT:
				case EF_R16G16_SFLOAT:
				case EF_R16G16B16_UNORM:
				case EF_R16G16B16_SNORM:
				case EF_R16G16B16_UINT:
				case EF_R16G16B16_SINT:
				case EF_R16G16B16_SFLOAT:
				case EF_R16G16B16A16_UNORM:
				case EF_R16G16B16A16_SNORM:
				case EF_R16G16B16A16_UINT:
				case EF_R16G16B16A16_SINT:
				case EF_R16G16B16A16_SFLOAT:
				case EF_R32_UINT:
				case EF_R32_SINT:
				case EF_R32_SFLOAT:
				case EF_R32G32_UINT:
				case EF_R32G32_SINT:
				case EF_R32G32_SFLOAT:
				case EF_R32G32B32_UINT:
				case EF_R32G32B32_SINT:
				case EF_R32G32B32_SFLOAT:
				case EF_R32G32B32A32_UINT:
				case EF_R32G32B32A32_SINT:
				case EF_R32G32B32A32_SFLOAT:
					return true;
				default:
				{
					GLint res = GL_FALSE;
					extGlGetInternalformativ(GL_TEXTURE_2D, getSizedOpenGLFormatFromOurFormat(_fmt), GL_COLOR_RENDERABLE, 1, &res);
					return res==GL_TRUE;
				}
            }
        }
        inline bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const override
        {
            using namespace asset;
            switch (_fmt)
            {
				case EF_R32G32B32A32_SFLOAT:
				case EF_R16G16B16A16_SFLOAT:
				case EF_R32G32_SFLOAT:
				case EF_R16G16_SFLOAT:
				case EF_B10G11R11_UFLOAT_PACK32:
				case EF_R32_SFLOAT:
				case EF_R16_SFLOAT:
				case EF_R16G16B16A16_UNORM:
				case EF_A2B10G10R10_UNORM_PACK32:
				case EF_R8G8B8A8_UNORM:
				case EF_R16G16_UNORM:
				case EF_R8G8_UNORM:
				case EF_R16_UNORM:
				case EF_R8_UNORM:
				case EF_R16G16B16A16_SNORM:
				case EF_R8G8B8A8_SNORM:
				case EF_R16G16_SNORM:
				case EF_R8G8_SNORM:
				case EF_R16_SNORM:
				case EF_R32G32B32A32_UINT:
				case EF_R16G16B16A16_UINT:
				case EF_A2B10G10R10_UINT_PACK32:
				case EF_R8G8B8A8_UINT:
				case EF_R32G32_UINT:
				case EF_R16G16_UINT:
				case EF_R8G8_UINT:
				case EF_R32_UINT:
				case EF_R16_UINT:
				case EF_R8_UINT:
				case EF_R32G32B32A32_SINT:
				case EF_R16G16B16A16_SINT:
				case EF_R8G8B8A8_SINT:
				case EF_R32G32_SINT:
				case EF_R16G16_SINT:
				case EF_R8G8_SINT:
				case EF_R32_SINT:
				case EF_R16_SINT:
				case EF_R8_SINT:
					return true;
				default: return false;
            }
        }
        inline bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const override
        {
            using namespace asset;
            // opengl spec section 8.5.1
            switch (_fmt)
            {
				// formats checked as "Req. tex"
				case EF_R8_UNORM:
				case EF_R8_SNORM:
				case EF_R16_UNORM:
				case EF_R16_SNORM:
				case EF_R8G8_UNORM:
				case EF_R8G8_SNORM:
				case EF_R16G16_UNORM:
				case EF_R16G16_SNORM:
				case EF_R8G8B8_UNORM:
				case EF_R8G8B8_SNORM:
				case EF_A1R5G5B5_UNORM_PACK16:
				case EF_R8G8B8A8_SRGB:
				case EF_A8B8G8R8_UNORM_PACK32:
				case EF_A8B8G8R8_SNORM_PACK32:
				case EF_A8B8G8R8_SRGB_PACK32:
				case EF_R16_SFLOAT:
				case EF_R16G16_SFLOAT:
				case EF_R16G16B16_SFLOAT:
				case EF_R16G16B16A16_SFLOAT:
				case EF_R32_SFLOAT:
				case EF_R32G32_SFLOAT:
				case EF_R32G32B32_SFLOAT:
				case EF_R32G32B32A32_SFLOAT:
				case EF_B10G11R11_UFLOAT_PACK32:
				case EF_E5B9G9R9_UFLOAT_PACK32:
				case EF_A2B10G10R10_UNORM_PACK32:
				case EF_A2B10G10R10_UINT_PACK32:
				case EF_R16G16B16A16_UNORM:
				case EF_R8_UINT:
				case EF_R8_SINT:
				case EF_R8G8_UINT:
				case EF_R8G8_SINT:
				case EF_R8G8B8_UINT:
				case EF_R8G8B8_SINT:
				case EF_R8G8B8A8_UNORM:
				case EF_R8G8B8A8_SNORM:
				case EF_R8G8B8A8_UINT:
				case EF_R8G8B8A8_SINT:
				case EF_B8G8R8A8_UINT:
				case EF_R16_UINT:
				case EF_R16_SINT:
				case EF_R16G16_UINT:
				case EF_R16G16_SINT:
				case EF_R16G16B16_UINT:
				case EF_R16G16B16_SINT:
				case EF_R16G16B16A16_UINT:
				case EF_R16G16B16A16_SINT:
				case EF_R32_UINT:
				case EF_R32_SINT:
				case EF_R32G32_UINT:
				case EF_R32G32_SINT:
				case EF_R32G32B32_UINT:
				case EF_R32G32B32_SINT:
				case EF_R32G32B32A32_UINT:
				case EF_R32G32B32A32_SINT:

				// depth/stencil/depth+stencil formats checked as "Req. format"
				case EF_D16_UNORM:
				case EF_X8_D24_UNORM_PACK32:
				case EF_D32_SFLOAT:
				case EF_D24_UNORM_S8_UINT:
				case EF_S8_UINT:

				// specific compressed formats
				case EF_BC6H_UFLOAT_BLOCK:
				case EF_BC6H_SFLOAT_BLOCK:
				case EF_BC7_UNORM_BLOCK:
				case EF_BC7_SRGB_BLOCK:
				case EF_ETC2_R8G8B8_UNORM_BLOCK:
				case EF_ETC2_R8G8B8_SRGB_BLOCK:
				case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
				case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
				case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
				case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
				case EF_EAC_R11_UNORM_BLOCK:
				case EF_EAC_R11_SNORM_BLOCK:
				case EF_EAC_R11G11_UNORM_BLOCK:
				case EF_EAC_R11G11_SNORM_BLOCK:
					return true;

				// astc
				case EF_ASTC_4x4_UNORM_BLOCK:
				case EF_ASTC_5x4_UNORM_BLOCK:
				case EF_ASTC_5x5_UNORM_BLOCK:
				case EF_ASTC_6x5_UNORM_BLOCK:
				case EF_ASTC_6x6_UNORM_BLOCK:
				case EF_ASTC_8x5_UNORM_BLOCK:
				case EF_ASTC_8x6_UNORM_BLOCK:
				case EF_ASTC_8x8_UNORM_BLOCK:
				case EF_ASTC_10x5_UNORM_BLOCK:
				case EF_ASTC_10x6_UNORM_BLOCK:
				case EF_ASTC_10x8_UNORM_BLOCK:
				case EF_ASTC_10x10_UNORM_BLOCK:
				case EF_ASTC_12x10_UNORM_BLOCK:
				case EF_ASTC_12x12_UNORM_BLOCK:
				case EF_ASTC_4x4_SRGB_BLOCK:
				case EF_ASTC_5x4_SRGB_BLOCK:
				case EF_ASTC_5x5_SRGB_BLOCK:
				case EF_ASTC_6x5_SRGB_BLOCK:
				case EF_ASTC_6x6_SRGB_BLOCK:
				case EF_ASTC_8x5_SRGB_BLOCK:
				case EF_ASTC_8x6_SRGB_BLOCK:
				case EF_ASTC_8x8_SRGB_BLOCK:
				case EF_ASTC_10x5_SRGB_BLOCK:
				case EF_ASTC_10x6_SRGB_BLOCK:
				case EF_ASTC_10x8_SRGB_BLOCK:
				case EF_ASTC_10x10_SRGB_BLOCK:
				case EF_ASTC_12x10_SRGB_BLOCK:
				case EF_ASTC_12x12_SRGB_BLOCK:
					return queryOpenGLFeature(NBL_KHR_texture_compression_astc_ldr);

				default: return false;
            }
        }
        inline bool isHardwareBlendableFormat(asset::E_FORMAT _fmt) const override
        {
            return isColorRenderableFormat(_fmt) && (asset::isNormalizedFormat(_fmt) || asset::isFloatingPointFormat(_fmt));
        }


        const core::smart_refctd_dynamic_array<std::string> getSupportedGLSLExtensions() const override;


        bool bindGraphicsPipeline(const video::IGPURenderpassIndependentPipeline* _gpipeline) override;

        bool bindComputePipeline(const video::IGPUComputePipeline* _cpipeline) override;

        bool bindDescriptorSets(E_PIPELINE_BIND_POINT _pipelineType, const IGPUPipelineLayout* _layout,
            uint32_t _first, uint32_t _count, const IGPUDescriptorSet* const* _descSets, core::smart_refctd_dynamic_array<uint32_t>* _dynamicOffsets) override;


		core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false) override;

		core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset, size_t _size) override;

		core::smart_refctd_ptr<IGPUSampler> createGPUSampler(const IGPUSampler::SParams& _params) override;

		core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) override;

        core::smart_refctd_ptr<IGPUImageView> createGPUImageView(IGPUImageView::SCreationParams&& params) override;

        core::smart_refctd_ptr<IGPUShader> createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>&& _cpushader) override;
        core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt) override;

        core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override;

        core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout(
            const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
        ) override;

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createGPURenderpassIndependentPipeline(
			IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            IGPUSpecializedShader* const* _shadersBegin, IGPUSpecializedShader* const* _shadersEnd,
            const asset::SVertexInputParams& _vertexInputParams,
            const asset::SBlendParams& _blendParams,
            const asset::SPrimitiveAssemblyParams& _primAsmParams,
            const asset::SRasterizationParams& _rasterParams
        ) override;

        virtual core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline(
			IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
        ) override;

		core::smart_refctd_ptr<IGPUPipelineCache> createGPUPipelineCache() override;

        core::smart_refctd_ptr<IGPUDescriptorSet> createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& _layout) override;

		void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) override;


		bool pushConstants(const IGPUPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values) override;

		bool dispatch(uint32_t _groupCountX, uint32_t _groupCountY, uint32_t _groupCountZ) override;
		bool dispatchIndirect(const IGPUBuffer* _indirectBuf, size_t _offset) override;


		//! generic version which overloads the unimplemented versions
		bool changeRenderContext(const SExposedVideoData& videoData, void* device) {return false;}

        bool initAuxContext();
        const SAuxContext* getThreadContext(const std::thread::id& tid=std::this_thread::get_id());
        bool deinitAuxContext();

        uint16_t retrieveDisplayRefreshRate() const override;


        void flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) override;

        void invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) override;

		void fillBuffer(IGPUBuffer* buffer, size_t offset, size_t length, uint32_t value) override;

        void copyBuffer(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, size_t readOffset, size_t writeOffset, size_t length) override;

		void copyImage(IGPUImage* srcImage, IGPUImage* dstImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions) override;

		void copyBufferToImage(IGPUBuffer* srcBuffer, IGPUImage* dstImage, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions) override;

		void copyImageToBuffer(IGPUImage* srcImage, IGPUBuffer* dstBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions) override;

		//! clears the zbuffer
		bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0);

		//! presents the rendered scene on the screen, returns false if failed
		bool endScene();


		void beginQuery(IQueryObject* query);
		void endQuery(IQueryObject* query);

        IQueryObject* createPrimitivesGeneratedQuery() override;
        IQueryObject* createElapsedTimeQuery() override;
        IGPUTimestampQuery* createTimestampQuery() override;


        virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb);

		virtual void drawArraysIndirect(const asset::SBufferBinding<const IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) override;
		virtual void drawIndexedIndirect(const asset::SBufferBinding<const IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        asset::E_INDEX_TYPE indexType, const IGPUBuffer* indexBuff,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) override;


		//! queries the features of the driver, returns true if feature is available
		virtual bool queryFeature(const E_DRIVER_FEATURE& feature) const;

		//!
		virtual void issueGPUTextureBarrier() {COpenGLExtensionHandler::extGlTextureBarrier();}

        //! needs to be "deleted" since its not refcounted
        virtual core::smart_refctd_ptr<IDriverFence> placeFence(const bool& implicitFlushWaitSameThread=false) override final
        {
            return core::make_smart_refctd_ptr<COpenGLDriverFence>(implicitFlushWaitSameThread);
        }

		//! \return Returns the name of the video driver. Example: In case of the Direct3D8
		//! driver, it would return "Direct3D8.1".
		virtual const wchar_t* getName() const;

		//! sets a viewport
		virtual void setViewPort(const core::rect<int32_t>& area);

		//! Returns type of video driver
		inline E_DRIVER_TYPE getDriverType() const override { return EDT_OPENGL; }

		//! get color format of the current color buffer
		inline asset::E_FORMAT getColorFormat() const override { return ColorFormat; }


        virtual IFrameBuffer* addFrameBuffer();

        //! Remove
        virtual void removeFrameBuffer(IFrameBuffer* framebuf);

        virtual void removeAllFrameBuffers();

		virtual bool setRenderTarget(IFrameBuffer* frameBuffer, bool setNewViewport=true);

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth=true, bool copyStencil=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false);


    private:
        void clearColor_gatherAndOverrideState(SAuxContext* found, uint32_t _attIx, GLboolean* _rasterDiscard, GLboolean* _colorWmask);
        void clearColor_bringbackState(SAuxContext* found, uint32_t _attIx, GLboolean _rasterDiscard, const GLboolean* _colorWmask);

    public:
		//! Clears the ZBuffer.
		virtual void clearZBuffer(const float &depth=0.0);

		virtual void clearStencilBuffer(const int32_t &stencil);

		virtual void clearZStencilBuffers(const float &depth, const int32_t &stencil);

		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals);

		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals) override;
		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals) override;

		//! Enable/disable a clipping plane.
		//! There are at least 6 clipping planes available for the user to set at will.
		//! \param index: The plane index. Must be between 0 and MaxUserClipPlanes.
		//! \param enable: If true, enable the clipping plane else disable it.
		virtual void enableClipPlane(uint32_t index, bool enable);

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo() {return VendorName;}

		//!
		const size_t& getMaxConcurrentShaderInvocations() const {return maxConcurrentShaderInvocations;}

		//!
		const uint32_t& getMaxShaderComputeUnits() const {return maxShaderComputeUnits;}

		//!
		const size_t& getMaxShaderInvocationsPerALU() const {return maxALUShaderInvocations;}

#ifdef _NBL_COMPILE_WITH_OPENCL_
        const cl_device_id& getOpenCLAssociatedDevice() const {return clDevice;}
		const cl_context_properties* getOpenCLAssociatedContextProperties() const { return clProperties; }

        size_t getOpenCLAssociatedDeviceID() const {return clDeviceIx;}
        size_t getOpenCLAssociatedPlatformID() const {return clPlatformIx;}
#endif // _NBL_COMPILE_WITH_OPENCL_

        struct SAuxContext
        {
        //public:
            struct SPipelineCacheVal
            {
                GLuint GLname;
                core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline> object;//so that it holds shaders which concerns hash
                uint64_t lastUsed;
            };

            _NBL_STATIC_INLINE_CONSTEXPR size_t maxVAOCacheSize = 0x1u<<10; //make this cache configurable
            _NBL_STATIC_INLINE_CONSTEXPR size_t maxPipelineCacheSize = 0x1u<<13;//8k

            SAuxContext() : threadId(std::thread::id()), ctx(NULL),
                            CurrentFBO(0), CurrentRendertargetSize(0,0)
            {
                VAOMap.reserve(maxVAOCacheSize);
            }

            void flushState_descriptors(E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout);
            void flushStateGraphics(uint32_t stateBits);
            void flushStateCompute(uint32_t stateBits);

            SOpenGLState currentState;
            SOpenGLState nextState;
			// represents descriptors currently flushed into GL state,
			// layout is needed to disambiguate descriptor sets due to translation into OpenGL descriptor indices
            struct {
                SOpenGLState::SDescSetBnd descSets[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT];
				core::smart_refctd_ptr<const COpenGLPipelineLayout> layout;
            } effectivelyBoundDescriptors;

            //push constants are tracked outside of next/currentState because there can be multiple pushConstants() calls and each of them kinda depends on the pervious one (layout compatibility)
			pipeline_for_bindpoint_t<EPBP_COMPUTE>::PushConstantsState pushConstantsStateCompute;
			pipeline_for_bindpoint_t<EPBP_GRAPHICS>::PushConstantsState pushConstantsStateGraphics;
			template<E_PIPELINE_BIND_POINT PBP>
			typename pipeline_for_bindpoint_t<PBP>::PushConstantsState* pushConstantsState()
			{
				if constexpr(PBP==EPBP_COMPUTE)
					return &pushConstantsStateCompute;
				else if (PBP== EPBP_GRAPHICS)
					return &pushConstantsStateGraphics;
				else
					return nullptr;
			}

        //private:
            std::thread::id threadId;
            uint8_t ID; //index in array of contexts, just to be easier in use
            #ifdef _NBL_WINDOWS_API_
                HGLRC ctx;
            #endif
            #ifdef _NBL_COMPILE_WITH_X11_DEVICE_
                GLXContext ctx;
                GLXPbuffer pbuff;
            #endif
            #ifdef _NBL_COMPILE_WITH_OSX_DEVICE_
                AppleMakesAUselessOSWhichHoldsBackTheGamingIndustryAndSabotagesOpenStandards ctx;
            #endif

            //! FBOs
            core::vector<IFrameBuffer*>  FrameBuffers;
            COpenGLFrameBuffer*         CurrentFBO;
            core::dimension2d<uint32_t> CurrentRendertargetSize; // @Crisspl TODO: Fold this into SOpenGLState, as well as the Vulkan dynamic state (scissor rect, viewport, etc.)

            //!
            core::vector<SOpenGLState::HashVAOPair> VAOMap;
            struct HashPipelinePair
            {
                SOpenGLState::SGraphicsPipelineHash first;
                SPipelineCacheVal second;

                inline bool operator<(const HashPipelinePair& rhs) const { return first < rhs.first; }
            };
            core::vector<HashPipelinePair> GraphicsPipelineMap;

            GLuint createGraphicsPipeline(const SOpenGLState::SGraphicsPipelineHash& _hash);

            void updateNextState_pipelineAndRaster(const IGPURenderpassIndependentPipeline* _pipeline);
            //! Must be called AFTER updateNextState_pipelineAndRaster() if pipeline and raster params have to be modified at all in this pass
            void updateNextState_vertexInput(
                const asset::SBufferBinding<const IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                const IGPUBuffer* _indexBuffer,
                const IGPUBuffer* _indirectDrawBuffer,
                const IGPUBuffer* _paramBuffer
            );
			template<E_PIPELINE_BIND_POINT PBP>
            inline void pushConstants(const COpenGLPipelineLayout* _layout, uint32_t _stages, uint32_t _offset, uint32_t _size, const void* _values)
			{
				//validation is done in CNullDriver::pushConstants()
				//if arguments were invalid (dont comply Valid Usage section of vkCmdPushConstants docs), execution should not even get to this point

				if (pushConstantsState<PBP>()->layout && !pushConstantsState<PBP>()->layout->isCompatibleForPushConstants(_layout))
				{
				//#ifdef _NBL_DEBUG
					constexpr size_t toFill = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE / sizeof(uint64_t);
					constexpr size_t bytesLeft = IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE - (toFill * sizeof(uint64_t));
					constexpr uint64_t pattern = 0xdeadbeefDEADBEEFull;
					std::fill(reinterpret_cast<uint64_t*>(pushConstantsState<PBP>()->data), reinterpret_cast<uint64_t*>(pushConstantsState<PBP>()->data)+toFill, pattern);
					if constexpr (bytesLeft > 0ull)
						memcpy(reinterpret_cast<uint64_t*>(pushConstantsState<PBP>()->data) + toFill, &pattern, bytesLeft);
				//#endif

					_stages |= IGPUSpecializedShader::ESS_ALL;
				}
				pushConstantsState<PBP>()->incrementStamps(_stages);

				pushConstantsState<PBP>()->layout = core::smart_refctd_ptr<const COpenGLPipelineLayout>(_layout);
				memcpy(pushConstantsState<PBP>()->data+_offset, _values, _size);
			}

            inline size_t getVAOCacheSize() const
            {
                return VAOMap.size();
            }

            inline void freeUpVAOCache(bool exitOnFirstDelete)
            {
                for(auto it = VAOMap.begin(); VAOMap.size()>maxVAOCacheSize&&it!=VAOMap.end();)
                {
                    if (it->first==currentState.vertexInputParams.vao.first)
                        continue;

                    if (CNullDriver::ReallocationCounter-it->second.lastUsed>1000) //maybe make this configurable
                    {
                        COpenGLExtensionHandler::extGlDeleteVertexArrays(1, &it->second.GLname);
                        it = VAOMap.erase(it);
                        if (exitOnFirstDelete)
                            return;
                    }
                    else
                        it++;
                }
            }
            //TODO DRY
            inline void freeUpGraphicsPipelineCache(bool exitOnFirstDelete)
            {
                for (auto it = GraphicsPipelineMap.begin(); GraphicsPipelineMap.size() > maxPipelineCacheSize&&it != GraphicsPipelineMap.end();)
                {
                    if (it->first == currentState.pipeline.graphics.usedShadersHash)
                        continue;

                    if (CNullDriver::ReallocationCounter-it->second.lastUsed > 1000) //maybe make this configurable
                    {
                        COpenGLExtensionHandler::extGlDeleteProgramPipelines(1, &it->second.GLname);
                        it = GraphicsPipelineMap.erase(it);
                        if (exitOnFirstDelete)
                            return;
                    }
                    else
                        it++;
                }
            }
        };


		//!
		virtual uint32_t getRequiredUBOAlignment() const {return COpenGLExtensionHandler::reqUBOAlignment;}

		//!
		virtual uint32_t getRequiredSSBOAlignment() const {return COpenGLExtensionHandler::reqSSBOAlignment;}

		//!
		virtual uint32_t getRequiredTBOAlignment() const {return COpenGLExtensionHandler::reqTBOAlignment;}

		//!
		virtual uint32_t getMinimumMemoryMapAlignment() const {return COpenGLExtensionHandler::minMemoryMapAlignment;}

        //!
        virtual uint32_t getMaxComputeWorkGroupSize(uint32_t _dimension) const { return COpenGLExtensionHandler::MaxComputeWGSize[_dimension]; }

        //!
        virtual uint64_t getMaxUBOSize() const override { return COpenGLExtensionHandler::maxUBOSize; }

        //!
        virtual uint64_t getMaxSSBOSize() const override { return COpenGLExtensionHandler::maxSSBOSize; }

        //!
        virtual uint64_t getMaxTBOSizeInTexels() const override { return COpenGLExtensionHandler::maxTBOSizeInTexels; }

        //!
        virtual uint64_t getMaxBufferSize() const override { return COpenGLExtensionHandler::maxBufferSize; }

        uint32_t getMaxUBOBindings() const override { return COpenGLExtensionHandler::maxUBOBindings; }
        uint32_t getMaxSSBOBindings() const override { return COpenGLExtensionHandler::maxSSBOBindings; }
        uint32_t getMaxTextureBindings() const override { return COpenGLExtensionHandler::maxTextureBindings; }
        uint32_t getMaxTextureBindingsCompute() const override { return COpenGLExtensionHandler::maxTextureBindingsCompute; }
        uint32_t getMaxImageBindings() const override { return COpenGLExtensionHandler::maxImageBindings; }

		//!
		bool runningInRenderdoc() const { return runningInRenderDoc; }

    private:
        SAuxContext* getThreadContext_helper(const bool& alreadyLockedMutex, const std::thread::id& tid = std::this_thread::get_id());

        void cleanUpContextBeforeDelete();


        //COpenGLDriver::CGPUObjectFromAssetConverter
        class CGPUObjectFromAssetConverter;
        friend class CGPUObjectFromAssetConverter;

        using PipelineMapKeyT = std::pair<std::array<core::smart_refctd_ptr<IGPUSpecializedShader>, 5u>, std::thread::id>;
        core::map<PipelineMapKeyT, GLuint> Pipelines;

        bool runningInRenderDoc;

		// returns the current size of the screen or rendertarget
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize();

		void createMaterialRenderers();

		core::stringw Name;

		std::string VendorName;

		//! Color buffer format
		asset::E_FORMAT ColorFormat; //FIXME

        mutable core::smart_refctd_dynamic_array<std::string> m_supportedGLSLExtsNames;

		#ifdef _NBL_WINDOWS_API_
			HDC HDc; // Private GDI Device Context
			HWND Window;
		#ifdef _NBL_COMPILE_WITH_WINDOWS_DEVICE_
			CIrrDeviceWin32 *Win32Device;
		#endif
		#endif
		#ifdef _NBL_COMPILE_WITH_X11_DEVICE_
			GLXDrawable Drawable;
			Display* X11Display;
			CIrrDeviceLinux *X11Device;
		#endif
		#ifdef _NBL_COMPILE_WITH_OSX_DEVICE_
			CIrrDeviceMacOSX *OSXDevice;
		#endif
		#ifdef _NBL_COMPILE_WITH_SDL_DEVICE_
			CIrrDeviceSDL *SDLDevice;
		#endif

        size_t maxALUShaderInvocations;
        size_t maxConcurrentShaderInvocations;
        uint32_t maxShaderComputeUnits;
#ifdef _NBL_COMPILE_WITH_OPENCL_
        cl_device_id clDevice;
		cl_context_properties clProperties[7];
        size_t clPlatformIx, clDeviceIx;
#endif // _NBL_COMPILE_WITH_OPENCL_

        std::mutex glContextMutex;
		SAuxContext* AuxContexts;
        core::smart_refctd_ptr<const asset::IGLSLCompiler> GLSLCompiler;

		E_DEVICE_TYPE DeviceType;
	};

} // end namespace video
} // end namespace nbl


#endif // _NBL_COMPILE_WITH_OPENGL_
#endif


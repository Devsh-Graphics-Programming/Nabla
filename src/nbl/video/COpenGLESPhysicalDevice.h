#ifndef __NBL_C_OPENGLES_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGLES_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/CEGL.h"
#include "nbl/video/IOpenGL_PhysicalDeviceBase.h"
#include "nbl/video/COpenGL_LogicalDevice.h"

namespace nbl::video
{

class COpenGLESPhysicalDevice final : public IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>
{
    using base_t = IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>;

public:
	static COpenGLESPhysicalDevice* create(IAPIConnection* api, renderdoc_api_t* rdoc, core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb)
	{
		constexpr EGLint OPENGL_ES_MAJOR = 3;
		constexpr EGLint OPENGL_ES_MINOR_BEST  = 2;
		constexpr EGLint OPENGL_ES_MINOR_WORST = 1;

		auto initRes = createContext(&_egl, EGL_OPENGL_ES_API, { OPENGL_ES_MAJOR, OPENGL_ES_MINOR_BEST }, OPENGL_ES_MINOR_WORST);
		if (initRes.ctx==EGL_NO_CONTEXT)
			return nullptr;
		if (initRes.minor < OPENGL_ES_MINOR_WORST)
		{
			_egl.call.peglDestroyContext(_egl.display, initRes.ctx);
			return nullptr;
		}

		return new COpenGLESPhysicalDevice(api,rdoc,std::move(s),std::move(_egl),std::move(dbgCb),initRes.config,initRes.ctx,initRes.major,initRes.minor);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL_ES; }

	const SFormatImageUsage& getImageFormatUsagesLinear(const asset::E_FORMAT format) override
	{
		if (m_linearTilingUsages[format].isInitialized)
			return m_linearTilingUsages[format];

		_NBL_DEBUG_BREAK_IF("We don't support linear tiling at the moment!");
		return SFormatImageUsage();
	}

	const SFormatImageUsage& getImageFormatUsagesOptimal(const asset::E_FORMAT format) override
	{
		if (m_optimalTilingUsages[format].isInitialized)
			return m_optimalTilingUsages[format];

		m_optimalTilingUsages[format].sampledImage = isAllowedTextureFormat(format) ? 1 : 0;
		m_optimalTilingUsages[format].storageImage = isAllowedImageStoreFormat(format) ? 1 : 0;
		m_optimalTilingUsages[format].storageImageAtomic = isAllowedImageStoreAtomicFormat(format) ? 1 : 0;
		m_optimalTilingUsages[format].attachment = isRenderableFormat(format) ? 1 : 0;
		m_optimalTilingUsages[format].attachmentBlend = isRenderableFormat(format) ? 1 : 0;
		m_optimalTilingUsages[format].blitSrc = isRenderableFormat(format) ? 1 : 0;
		m_optimalTilingUsages[format].blitDst = isRenderableFormat(format) ? 1 : 0;
		const bool anyUsageFlagSet =
			m_optimalTilingUsages[format].sampledImage |
			m_optimalTilingUsages[format].storageImage |
			m_optimalTilingUsages[format].storageImageAtomic |
			m_optimalTilingUsages[format].attachment |
			m_optimalTilingUsages[format].attachmentBlend |
			m_optimalTilingUsages[format].blitSrc |
			m_optimalTilingUsages[format].blitDst;
		m_optimalTilingUsages[format].transferSrc = anyUsageFlagSet ? 1 : 0;
		m_optimalTilingUsages[format].transferDst = anyUsageFlagSet ? 1 : 0;
#if 0
		{
			auto GetInternalFormativ = reinterpret_cast<PFNGLGETINTERNALFORMATIVPROC>(m_egl.call.peglGetProcAddress("glGetInternalformativ"));

			GLint maxSamples;
			GetInternalFormativ(
				GL_TEXTURE_2D_MULTISAMPLE,
				getSizedOpenGLFormatFromOurFormat(nullptr, format),
				GL_SAMPLES, 1, &maxSamples); // probably should take function table from outside?
			assert(maxSamples <= 8);

			m_optimalTilingUsages[format].log2MaxSamples = maxSamples ? std::log2(maxSamples) : 0;
		}
#endif

		m_optimalTilingUsages[format].isInitialized = 1;

		return m_optimalTilingUsages[format];
	}

	const SFormatBufferUsage& getBufferFormatUsages(const asset::E_FORMAT format) override
	{
		if (m_bufferUsages[format].isInitialized)
			return m_bufferUsages[format];

		m_bufferUsages[format].vertexAttribute = isAllowedVertexAttribFormat(format) ? 1 : 0;
		m_bufferUsages[format].bufferView = isAllowedBufferViewFormat(format) ? 1 : 0;
		m_bufferUsages[format].storageBufferView = isAllowedBufferViewFormat(format) ? 1 : 0;
		m_bufferUsages[format].storageBufferViewAtomic = isAllowedBufferViewFormat(format) ? 1 : 0;
		m_bufferUsages[format].accelerationStructureVertex = isAllowedVertexAttribFormat(format);

		m_bufferUsages[format].isInitialized = 1;

		return m_bufferUsages[format];
	}

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLESLogicalDevice>(core::smart_refctd_ptr<IAPIConnection>(m_api),this,m_rdoc_api,params,&m_egl,&m_glfeatures,m_config,m_gl_major,m_gl_minor);
	}

private:
	inline bool isAllowedTextureFormat(const asset::E_FORMAT _fmt) const
	{
		using namespace asset;
		// OpenGLES 3.1 Spec Section 8.5.1
		switch (_fmt)
		{
		// formats checked as "Req. tex" in Table 8.1.3
		case EF_R8_UNORM:
		case EF_R8_SNORM:
		case EF_R8G8_UNORM:
		case EF_R8G8_SNORM:
		case EF_R8G8B8_UNORM:
		case EF_R8G8B8_SNORM:
		case EF_R5G6B5_UNORM_PACK16: // RGB565?
		case EF_R4G4B4A4_UNORM_PACK16: // RGBA4?
		case EF_R5G5B5A1_UNORM_PACK16: // RGB5_A1?
		case EF_R8G8B8A8_UNORM:
		case EF_R8G8B8A8_SNORM:
		case EF_A2R10G10B10_UNORM_PACK32: // RGB10_A2?
		case EF_A2R10G10B10_UINT_PACK32: // RGB10_A2UI?
		case EF_R8G8B8_SRGB: // SRGB8?
		case EF_R8G8B8A8_SRGB: // SRGB8_ALPHA8?
		case EF_R16_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_R16G16B16_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
		case EF_R32_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R32G32B32_SFLOAT:
		case EF_R32G32B32A32_SFLOAT:
		case EF_B10G11R11_UFLOAT_PACK32: // R11F_G11F_B10F?
		case EF_E5B9G9R9_UFLOAT_PACK32: // RGB9_E5?
		case EF_R8_SINT:
		case EF_R8_UINT:
		case EF_R16_SINT:
		case EF_R16_UINT:
		case EF_R32_SINT:
		case EF_R32_UINT:
		case EF_R8G8_SINT:
		case EF_R8G8_UINT:
		case EF_R16G16_SINT:
		case EF_R16G16_UINT:
		case EF_R32G32_SINT:
		case EF_R32G32_UINT:
		case EF_R8G8B8_SINT:
		case EF_R8G8B8_UINT:
		case EF_R16G16B16_SINT:
		case EF_R16G16B16_UINT:
		case EF_R32G32B32_SINT:
		case EF_R32G32B32_UINT:
		case EF_R8G8B8A8_SINT:
		case EF_R8G8B8A8_UINT:
		case EF_R16G16B16A16_SINT:
		case EF_R16G16B16A16_UINT:
		case EF_R32G32B32A32_SINT:
		case EF_R32G32B32A32_UINT:
		
		// specific compressed formats
		case EF_EAC_R11_UNORM_BLOCK:
		case EF_EAC_R11_SNORM_BLOCK:
		case EF_EAC_R11G11_UNORM_BLOCK:
		case EF_EAC_R11G11_SNORM_BLOCK:
		case EF_ETC2_R8G8B8_UNORM_BLOCK:
		case EF_ETC2_R8G8B8_SRGB_BLOCK:
		case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
		case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
		case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
		case EF_ETC2_R8G8B8A8_SRGB_BLOCK:

		// depth/stencil/depth+stencil formats checked as "Req. format"
		case EF_D16_UNORM:
		case EF_X8_D24_UNORM_PACK32:
		case EF_D32_SFLOAT:
		case EF_D24_UNORM_S8_UINT:
		case EF_D32_SFLOAT_S8_UINT:
		case EF_S8_UINT:
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
			return m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_texture_compression_astc_ldr);

		case EF_BC1_RGB_UNORM_BLOCK: // COMPRESSED_RGB_S3TC_DXT1_EXT?
		case EF_BC1_RGBA_UNORM_BLOCK: // COMPRESSED_RGBA_S3TC_DXT1_EXT?
		case EF_BC2_UNORM_BLOCK: // COMPRESSED_RGBA_S3TC_DXT3_EXT?                
		case EF_BC3_UNORM_BLOCK: // COMPRESSED_RGBA_S3TC_DXT5_EXT?                 
			return m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_compression_s3tc);

		case EF_BC4_UNORM_BLOCK: // COMPRESSED_RED_RGTC1_EXT?
		case EF_BC4_SNORM_BLOCK: // COMPRESSED_SIGNED_RED_RGTC1_EXT?
		case EF_BC5_UNORM_BLOCK: // COMPRESSED_RED_GREEN_RGTC2_EXT?
		case EF_BC5_SNORM_BLOCK: // COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT?
			return m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_compression_rgtc);

		case EF_BC6H_UFLOAT_BLOCK: // COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT?
		case EF_BC6H_SFLOAT_BLOCK: // COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT?
		case EF_BC7_UNORM_BLOCK: // COMPRESSED_RGBA_BPTC_UNORM_EXT?
		case EF_BC7_SRGB_BLOCK: // COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT?
			return m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_texture_compression_bptc);

		default: return false;
		}
	}

	inline bool isAllowedImageStoreFormat(const asset::E_FORMAT _fmt) const
	{
		// Table 8.2.8 in OpenGL ES 3.1 Spec
		using namespace asset;
		switch (_fmt)
		{
		case EF_R32G32B32A32_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
		case EF_R32_SFLOAT:
		case EF_R32G32B32A32_UINT:
		case EF_R16G16B16A16_UINT:
		case EF_R8G8B8A8_UINT:
		case EF_R32_UINT:
		case EF_R32G32B32A32_SINT:
		case EF_R16G16B16A16_SINT:
		case EF_R8G8B8A8_SINT:
		case EF_R32_SINT:
		case EF_R8G8B8A8_UNORM:
		case EF_R8G8B8A8_SNORM:
			return true;

		default: return false;
		}
	}

	inline bool isRenderableFormat(const asset::E_FORMAT format) const
	{
		using namespace asset;
		// Section 9.4 Framebuffer Completeness of OpenGL ES 3.1 spec
		switch (format)
		{	
		// CR column checked in Table 8.13 of OpenGL ES 3.1 Spec -- "color-renderable"
		case EF_R8_UNORM:
		case EF_R8G8_UNORM:
		case EF_R8G8B8_UNORM:
		case EF_R5G6B5_UNORM_PACK16:
		case EF_R4G4B4A4_UNORM_PACK16:
		case EF_R5G5B5A1_UNORM_PACK16:
		case EF_R8G8B8A8_UNORM:
		case EF_A2R10G10B10_UNORM_PACK32: // RGB10_A2?
		case EF_A2R10G10B10_UINT_PACK32: // RGB10_A2UI?
		case EF_R8G8B8A8_SRGB:
		case EF_R8_SINT:
		case EF_R8_UINT:
		case EF_R16_SINT:
		case EF_R16_UINT:
		case EF_R32_SINT:
		case EF_R32_UINT:
		case EF_R8G8_SINT:
		case EF_R8G8_UINT:
		case EF_R16G16_SINT:
		case EF_R16G16_UINT:
		case EF_R32G32_SINT:
		case EF_R32G32_UINT:
		case EF_R8G8B8A8_SINT:
		case EF_R8G8B8A8_UINT:
		case EF_R16G16B16A16_SINT:
		case EF_R16G16B16A16_UINT:
		case EF_R32G32B32A32_SINT:
		case EF_R32G32B32A32_UINT:

		// Table 8.14, Base Internal Format should be STENCIL_INDEX or DEPTH_STENCIL or DEPTH_COMPONENT -- "depth-renderable"/"stencil-renderable"
		case EF_D16_UNORM:
		case EF_X8_D24_UNORM_PACK32:
		case EF_D32_SFLOAT:
		case EF_D24_UNORM_S8_UINT:
		case EF_D32_SFLOAT_S8_UINT:
		case EF_S8_UINT:
			return true;

		// EXT_color_buffer_float here

		}
		return false;
	}

	inline bool isAllowedVertexAttribFormat(const asset::E_FORMAT _fmt) const
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
			// unsigned/signed rgb10a2
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_A2B10G10R10_UINT_PACK32:
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
			return true;

		default: return false;
		}
	}

	inline bool isAllowedBufferViewFormat(const asset::E_FORMAT _fmt) const
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
		default:
			return false;
		}
	}

	inline bool isAllowedImageStoreAtomicFormat(const asset::E_FORMAT format) const
	{
		switch (format)
		{
		case asset::EF_R32_SINT:
		case asset::EF_R32_UINT:
		// r32f also supports, but only imageAtomicExchange
			return true;
		default:
			return false;
		}
	}


	COpenGLESPhysicalDevice(
		IAPIConnection* api, renderdoc_api_t* rdoc, core::smart_refctd_ptr<system::ISystem>&& s,
		egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb,
		EGLConfig config, EGLContext ctx, EGLint major, EGLint minor
	) : base_t(api,rdoc,std::move(s),std::move(_egl),std::move(dbgCb), config,ctx,major,minor)
    {
    }
};

}

#endif
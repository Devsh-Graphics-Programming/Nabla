// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in Irrlicht.h

#ifndef __C_VIDEO_OPEN_GL_H_INCLUDED__
#define __C_VIDEO_OPEN_GL_H_INCLUDED__

#include "IrrCompileConfig.h"

#include "SIrrCreationParameters.h"

namespace irr
{
	class CIrrDeviceWin32;
	class CIrrDeviceLinux;
	class CIrrDeviceSDL;
	class CIrrDeviceMacOSX;
}

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "CNullDriver.h"
#include "IMaterialRendererServices.h"
// also includes the OpenGL stuff
#include "COpenGLExtensionHandler.h"
#include "COpenGLDriverFence.h"
#include "COpenGLTransformFeedback.h"
#include "COpenGLVAOSpec.h"
#include "COpenCLHandler.h"

#include <map>
#include "FW_Mutex.h"

namespace irr
{

namespace video
{
	class COpenGLTexture;
	class COpenGLFrameBuffer;

	class COpenGLDriver : public CNullDriver, public IMaterialRendererServices, public COpenGLExtensionHandler
	{
    protected:
		//! destructor
		virtual ~COpenGLDriver();

	public:
        struct SAuxContext;

		#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceWin32* device);
		//! inits the windows specific parts of the open gl driver
		bool initDriver(CIrrDeviceWin32* device);
		bool changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceWin32* device);
		#endif

		#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceLinux* device);
		//! inits the GLX specific parts of the open gl driver
		bool initDriver(CIrrDeviceLinux* device, SAuxContext* auxCtxts);
		bool changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceLinux* device);
		#endif

		#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceSDL* device);
		#endif

		#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
		COpenGLDriver(const SIrrlichtCreationParameters& params, io::IFileSystem* io, CIrrDeviceMacOSX *device);
		#endif

        inline virtual bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const override
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
        inline virtual bool isColorRenderableFormat(asset::E_FORMAT _fmt) const override
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
                extGlGetInternalformativ(GL_TEXTURE_2D, COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(_fmt), GL_COLOR_RENDERABLE, sizeof(res), &res);
                return res==GL_TRUE;
            }
            }
        }
        inline virtual bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const override
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
        inline virtual bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const override
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
                return queryOpenGLFeature(IRR_KHR_texture_compression_astc_ldr);

            default: return false;
            }
        }
        inline virtual bool isHardwareBlendableFormat(asset::E_FORMAT _fmt) const override
        {
            return isColorRenderableFormat(_fmt) && (asset::isNormalizedFormat(_fmt) || asset::isFloatingPointFormat(_fmt));
        }

		//! generic version which overloads the unimplemented versions
		bool changeRenderContext(const SExposedVideoData& videoData, void* device) {return false;}

        bool initAuxContext();
        const SAuxContext* getThreadContext(const std::thread::id& tid=std::this_thread::get_id()) const;
        bool deinitAuxContext();

	    virtual video::IGPUMeshDataFormatDesc* createGPUMeshDataFormatDesc(core::LeakDebugger* dbgr=NULL);

        virtual uint16_t retrieveDisplayRefreshRate() const override;

		virtual IGPUBuffer* createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false);

        void flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) override final;

        void invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) override final;

        void copyBuffer(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, size_t readOffset, size_t writeOffset, size_t length) override final;

		//! clears the zbuffer
		virtual bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0);

		//! presents the rendered scene on the screen, returns false if failed
		virtual bool endScene();


		virtual void beginQuery(IQueryObject* query);
		virtual void endQuery(IQueryObject* query);
		virtual void beginQuery(IQueryObject* query, const size_t& index);
		virtual void endQuery(IQueryObject* query, const size_t& index);

        virtual IQueryObject* createPrimitivesGeneratedQuery();
        virtual IQueryObject* createXFormFeedbackPrimitiveQuery();
        virtual IQueryObject* createElapsedTimeQuery();
        virtual IGPUTimestampQuery* createTimestampQuery();


        virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb);

		virtual void drawArraysIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const asset::E_PRIMITIVE_TYPE& mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride);
		virtual void drawIndexedIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                            const asset::E_PRIMITIVE_TYPE& mode,
                                            const asset::E_INDEX_TYPE& type, const IGPUBuffer* indirectDrawBuff,
                                            const size_t& offset, const size_t& count, const size_t& stride);


		//! queries the features of the driver, returns true if feature is available
		virtual bool queryFeature(const E_DRIVER_FEATURE& feature) const;

		//!
		virtual void issueGPUTextureBarrier() {COpenGLExtensionHandler::extGlTextureBarrier();}


		virtual const video::SGPUMaterial& getCurrentMaterial() const {return Material;}

		//! Sets a material. All 3d drawing functions draw geometry now
		//! using this material.
		//! \param material: Material to be used from now on.
		virtual void setMaterial(const SGPUMaterial& material);

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
		virtual E_DRIVER_TYPE getDriverType() const;

		//! get color format of the current color buffer
		virtual asset::E_FORMAT getColorFormat() const;

		//! Can be called by an IMaterialRenderer to make its work easier.
		virtual void setBasicRenderStates(const SGPUMaterial& material, const SGPUMaterial& lastmaterial,
			bool resetAllRenderstates);


        virtual void setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number=1);


        virtual int32_t addHighLevelShaderMaterial(
            const char* vertexShaderProgram,
            const char* controlShaderProgram,
            const char* evaluationShaderProgram,
            const char* geometryShaderProgram,
            const char* pixelShaderProgram,
            uint32_t patchVertices=3,
            E_MATERIAL_TYPE baseMaterial=video::EMT_SOLID,
            IShaderConstantSetCallBack* callback=0,
            const char** xformFeedbackOutputs = NULL,
            const uint32_t& xformFeedbackOutputCount = 0,
            int32_t userData=0,
            const char* vertexShaderEntryPointName="main",
            const char* controlShaderEntryPointName="main",
            const char* evaluationShaderEntryPointName="main",
            const char* geometryShaderEntryPointName="main",
            const char* pixelShaderEntryPointName="main");

		//! Returns a pointer to the IVideoDriver interface. (Implementation for
		//! IMaterialRendererServices)
		virtual IVideoDriver* getVideoDriver();

		//! Returns the maximum amount of primitives (mostly vertices) which
		//! the device is able to render with one drawIndexedTriangleList
		//! call.
		virtual uint32_t getMaximalIndicesCount() const;

        ITexture* createGPUTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, asset::E_FORMAT format = asset::EF_B8G8R8A8_UNORM) override;

        //!
        virtual IMultisampleTexture* addMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size, asset::E_FORMAT format = asset::EF_B8G8R8A8_UNORM, const bool& fixedSampleLocations = false);

		//! A.
        virtual ITextureBufferObject* addTextureBufferObject(IGPUBuffer* buf, const ITextureBufferObject::E_TEXURE_BUFFER_OBJECT_FORMAT& format = ITextureBufferObject::ETBOF_RGBA8, const size_t& offset=0, const size_t& length=0);

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


		//! Clears the ZBuffer.
		virtual void clearZBuffer(const float &depth=0.0);

		virtual void clearStencilBuffer(const int32_t &stencil);

		virtual void clearZStencilBuffers(const float &depth, const int32_t &stencil);

		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals);
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals);

		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals);
		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals);


		virtual ITransformFeedback* createTransformFeedback();

		//!
		virtual void bindTransformFeedback(ITransformFeedback* xformFeedback);

		virtual ITransformFeedback* getBoundTransformFeedback() {return getThreadContext_helper(false,std::this_thread::get_id())->CurrentXFormFeedback;}

        /** Only POINTS, LINES, and TRIANGLES are allowed as capture types.. no strips or fans!
        This issues an implicit call to bindTransformFeedback()
        **/
		virtual void beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const asset::E_PRIMITIVE_TYPE& primType= asset::EPT_POINTS);

		//! A redundant wrapper call to ITransformFeedback::pauseTransformFeedback(), made just for clarity
		virtual void pauseTransformFeedback();

		//! A redundant wrapper call to ITransformFeedback::pauseTransformFeedback(), made just for clarity
		virtual void resumeTransformFeedback();

		virtual void endTransformFeedback();


		//! Enable/disable a clipping plane.
		//! There are at least 6 clipping planes available for the user to set at will.
		//! \param index: The plane index. Must be between 0 and MaxUserClipPlanes.
		//! \param enable: If true, enable the clipping plane else disable it.
		virtual void enableClipPlane(uint32_t index, bool enable);

		//! Enable the 2d override material
		virtual void enableMaterial2D(bool enable=true);

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo() {return VendorName;}

		//! sets the needed renderstates
		void setRenderStates3DMode();

		//!
		const size_t& getMaxConcurrentShaderInvocations() const {return maxConcurrentShaderInvocations;}

		//!
		const uint32_t& getMaxShaderComputeUnits() const {return maxShaderComputeUnits;}

		//!
		const size_t& getMaxShaderInvocationsPerALU() const {return maxALUShaderInvocations;}

#ifdef _IRR_COMPILE_WITH_OPENCL_
        const cl_device_id& getOpenCLAssociatedDevice() const {return clDevice;}

        const size_t& getOpenCLAssociatedDeviceID() const {return clDeviceIx;}
        const size_t& getOpenCLAssociatedPlatformID() const {return clPlatformIx;}
#endif // _IRR_COMPILE_WITH_OPENCL_

        struct SAuxContext
        {
        //public:
            constexpr static size_t maxVAOCacheSize = 0x1u<<14; //make this cache configurable

            SAuxContext() : threadId(std::thread::id()), ctx(NULL), XFormFeedbackRunning(false), CurrentXFormFeedback(NULL),
                            CurrentFBO(0), CurrentRendertargetSize(0,0)
            {
                VAOMap.reserve(maxVAOCacheSize);
                CurrentVAO = HashVAOPair(COpenGLVAOSpec::HashAttribs(),NULL);

                for (size_t i=0; i<MATERIAL_MAX_TEXTURES; i++)
                {
                    CurrentSamplerHash[i] = 0xffffffffffffffffuLL;
                }
            }

            inline void setActiveSSBO(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes)
            {
                shaderStorageBufferObjects.set(first,count,buffers,offsets,sizes);
            }

            inline void setActiveUBO(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes)
            {
                uniformBufferObjects.set(first,count,buffers,offsets,sizes);
            }

            inline void setActiveIndirectDrawBuffer(const COpenGLBuffer* const buff)
            {
                indirectDraw.set(buff);
            }

            bool setActiveVAO(const COpenGLVAOSpec* const spec, const video::IGPUMeshBuffer* correctOffsetsForXFormDraw=NULL);

            //! sets the current Texture
            //! Returns whether setting was a success or not.
            bool setActiveTexture(uint32_t stage, video::IVirtualTexture* texture, const video::STextureSamplingParams &sampleParams);

            const GLuint& constructSamplerInCache(const uint64_t &hashVal);

        //private:
            std::thread::id threadId;
            #ifdef _IRR_WINDOWS_API_
                HGLRC ctx;
            #endif
            #ifdef _IRR_COMPILE_WITH_X11_DEVICE_
                GLXContext ctx;
                GLXPbuffer pbuff;
            #endif
            #ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
                AppleMakesAUselessOSWhichHoldsBackTheGamingIndustryAndSabotagesOpenStandards ctx;
            #endif

            bool                                         XFormFeedbackRunning; // TODO: delete
            COpenGLTransformFeedback* CurrentXFormFeedback; //TODO: delete


            //! FBOs
            core::vector<IFrameBuffer*>  FrameBuffers;
            COpenGLFrameBuffer*         CurrentFBO;
            core::dimension2d<uint32_t> CurrentRendertargetSize;


            //! Buffers
            template<GLenum BIND_POINT,size_t BIND_POINTS>
            class BoundIndexedBuffer : public core::AllocationOverrideDefault
            {
                    const COpenGLBuffer* boundBuffer[BIND_POINTS];
                    ptrdiff_t boundOffsets[BIND_POINTS];
                    ptrdiff_t boundSizes[BIND_POINTS];
                    uint64_t lastValidatedBuffer[BIND_POINTS];
                public:
                    BoundIndexedBuffer()
                    {
                        memset(boundBuffer,0,sizeof(boundBuffer));
                        memset(boundOffsets,0,sizeof(boundOffsets));
                        memset(boundSizes,0,sizeof(boundSizes));
                        memset(lastValidatedBuffer,0,sizeof(boundBuffer));
                    }

                    ~BoundIndexedBuffer()
                    {
                        set(0,BIND_POINTS,nullptr,nullptr,nullptr);
                    }

                    void set(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes);
            };

            //! SSBO
            BoundIndexedBuffer<GL_SHADER_STORAGE_BUFFER,OGL_MAX_BUFFER_BINDINGS>    shaderStorageBufferObjects;
            //! UBO
            BoundIndexedBuffer<GL_UNIFORM_BUFFER,OGL_MAX_BUFFER_BINDINGS>           uniformBufferObjects;

            //!
            template<GLenum BIND_POINT>
            class BoundBuffer : public core::AllocationOverrideDefault
            {
                    const COpenGLBuffer* boundBuffer;
                    uint64_t lastValidatedBuffer;
                public:
                    BoundBuffer() : lastValidatedBuffer(0)
                    {
                        boundBuffer = NULL;
                    }

                    ~BoundBuffer()
                    {
                        set(NULL);
                    }

                    void set(const COpenGLBuffer* buff);
            };

            //! Indirect
            BoundBuffer<GL_DRAW_INDIRECT_BUFFER> indirectDraw;


            /** We will operate on some assumptions here:

            1) On all GPU's known to me  GPUs MAX_VERTEX_ATTRIB_BINDINGS <= MAX_VERTEX_ATTRIBS,
            so it makes absolutely no sense to support buffer binding mix'n'match as it wouldn't
            get us anything (however if MVAB>MVA then we could have more inputs into a vertex shader).
            Also the VAO Attrib Binding is a VAO state so more VAOs would have to be created in the cache.

            2) Relative byte offset on VAO Attribute spec is capped to 2047 across all GPUs, which makes it
            useful only for specifying the offset from a single interleaved buffer, since we have to specify
            absolute (unbounded) offset and stride when binding a buffer to a VAO bind-point, it makes absolutely
            no sense to use this feature as its redundant.

            So the only things worth tracking for the VAO are:
            1) Element Buffer Binding
            2) Per Attribute (x16)
                A) Enabled (1 bit)
                B) Format (5 bits)
                C) Component Count (3 bits)
                D) Divisors (32bits - no limit)

            Total 16*4+16+16/8+4 = 11 uint64_t

            If we limit divisors artificially to 1 bit

            16/8+16/8+16+4 = 3 uint64_t
            **/
            class COpenGLVAO : public core::AllocationOverrideDefault
            {
                    size_t                      attrOffset[asset::EVAI_COUNT];
                    uint32_t                    attrStride[asset::EVAI_COUNT];
                    //vertices
                    const COpenGLBuffer*        mappedAttrBuf[asset::EVAI_COUNT];
                    //indices
                    const COpenGLBuffer*        mappedIndexBuf;

                    GLuint                      vao;
                    uint64_t                    lastValidated;
                #ifdef _IRR_DEBUG
                    COpenGLVAOSpec::HashAttribs debugHash;
                #endif // _IRR_DEBUG
                public:
                    _IRR_NO_DEFAULT_FINAL(COpenGLVAO);
                    _IRR_NO_COPY_FINAL(COpenGLVAO);

                    COpenGLVAO(const COpenGLVAOSpec* spec);
                    inline COpenGLVAO(COpenGLVAO&& other)
                    {
                        memcpy(this,&other,sizeof(COpenGLVAO));
                        memset(other.attrOffset,0,sizeof(mappedAttrBuf));
                        memset(other.attrStride,0,sizeof(mappedAttrBuf));
                        memset(other.mappedAttrBuf,0,sizeof(mappedAttrBuf));
                        other.mappedIndexBuf = NULL;
                        other.vao = 0;
                        other.lastValidated = 0;
                    }
                    ~COpenGLVAO();

                    inline const GLuint& getOpenGLName() const {return vao;}


                    inline COpenGLVAO& operator=(COpenGLVAO&& other)
                    {
                        this->~COpenGLVAO();
                        memcpy(this,&other,sizeof(COpenGLVAO));
                        memset(other.mappedAttrBuf,0,sizeof(mappedAttrBuf));
                        memset(other.attrStride,0,sizeof(mappedAttrBuf));
                        memset(other.mappedAttrBuf,0,sizeof(mappedAttrBuf));
                        other.mappedIndexBuf = NULL;
                        other.vao = 0;
                        other.lastValidated = 0;
                        return *this;
                    }


                    void bindBuffers(   const COpenGLBuffer* indexBuf,
                                        const COpenGLBuffer* const* attribBufs,
                                        const size_t offsets[asset::EVAI_COUNT],
                                        const uint32_t strides[asset::EVAI_COUNT]);

                    inline const uint64_t& getLastBoundStamp() const {return lastValidated;}

                #ifdef _IRR_DEBUG
                    inline const COpenGLVAOSpec::HashAttribs& getDebugHash() const {return debugHash;}
                #endif // _IRR_DEBUG
            };

            //!
            typedef std::pair<COpenGLVAOSpec::HashAttribs,COpenGLVAO*> HashVAOPair;
            HashVAOPair                 CurrentVAO;
            core::vector<HashVAOPair>    VAOMap;

            inline size_t getVAOCacheSize() const
            {
                return VAOMap.size();
            }

            inline void freeUpVAOCache(bool exitOnFirstDelete)
            {
                for(auto it = VAOMap.begin(); VAOMap.size()>maxVAOCacheSize&&it!=VAOMap.end();)
                {
                    if (it->first==CurrentVAO.first)
                        continue;

                    if (CNullDriver::ReallocationCounter-it->second->getLastBoundStamp()>1000) //maybe make this configurable
                    {
                        delete it->second;
                        it = VAOMap.erase(it);
                        if (exitOnFirstDelete)
                            return;
                    }
                    else
                        it++;
                }
            }

            //! Textures and Samplers
            class STextureStageCache : public core::AllocationOverrideDefault
            {
                const IVirtualTexture* CurrentTexture[MATERIAL_MAX_TEXTURES];
            public:
                STextureStageCache()
                {
                    for (uint32_t i=0; i<MATERIAL_MAX_TEXTURES; ++i)
                    {
                        CurrentTexture[i] = 0;
                    }
                }

                ~STextureStageCache()
                {
                    clear();
                }

                void set(uint32_t stage, const IVirtualTexture* tex)
                {
                    if (stage<MATERIAL_MAX_TEXTURES)
                    {
                        const IVirtualTexture* oldTexture=CurrentTexture[stage];
                        if (tex)
                            tex->grab();
                        CurrentTexture[stage]=tex;
                        if (oldTexture)
                            oldTexture->drop();
                    }
                }

                const IVirtualTexture* operator[](int stage) const
                {
                    if ((uint32_t)stage<MATERIAL_MAX_TEXTURES)
                        return CurrentTexture[stage];
                    else
                        return 0;
                }

                void remove(const IVirtualTexture* tex);

                void clear();
            };

            //!
            STextureStageCache                  CurrentTexture;

            //! Samplers
            uint64_t                            CurrentSamplerHash[MATERIAL_MAX_TEXTURES];
            core::unordered_map<uint64_t,GLuint> SamplerMap;
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
        virtual uint64_t getMaxTBOSize() const override { return COpenGLExtensionHandler::maxTBOSize; }

        //!
        virtual uint64_t getMaxBufferSize() const override { return COpenGLExtensionHandler::maxBufferSize; }

    private:
        SAuxContext* getThreadContext_helper(const bool& alreadyLockedMutex, const std::thread::id& tid = std::this_thread::get_id());

        void cleanUpContextBeforeDelete();


        void bindTransformFeedback(ITransformFeedback* xformFeedback, SAuxContext* toContext);


        //COpenGLDriver::CGPUObjectFromAssetConverter
        class CGPUObjectFromAssetConverter;
        friend class CGPUObjectFromAssetConverter;


        bool runningInRenderDoc;

		//! enumeration for rendering modes such as 2d and 3d for minizing the switching of renderStates.
		enum E_RENDER_MODE
		{
			ERM_NONE = 0,	// no render state has been set yet.
			ERM_2D,		// 2d drawing rendermode
			ERM_3D		// 3d rendering mode
		};

		E_RENDER_MODE CurrentRenderMode;
		//! bool to make all renderstates reset if set to true.
		bool ResetRenderStates;

		SGPUMaterial Material, LastMaterial;



		//! inits the parts of the open gl driver used on all platforms
		bool genericDriverInit();

		//! returns a device dependent texture from a software surface (IImage)
		virtual video::ITexture* createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, const io::path& name, asset::E_FORMAT format = asset::EF_B8G8R8A8_UNORM);

		// returns the current size of the screen or rendertarget
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const;

		void createMaterialRenderers();

		core::stringw Name;

		std::string VendorName;

		//! Color buffer format
		asset::E_FORMAT ColorFormat; //FIXME

		SIrrlichtCreationParameters Params;

		#ifdef _IRR_WINDOWS_API_
			HDC HDc; // Private GDI Device Context
			HWND Window;
		#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
			CIrrDeviceWin32 *Win32Device;
		#endif
		#endif
		#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
			GLXDrawable Drawable;
			Display* X11Display;
			CIrrDeviceLinux *X11Device;
		#endif
		#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
			CIrrDeviceMacOSX *OSXDevice;
		#endif
		#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
			CIrrDeviceSDL *SDLDevice;
		#endif

        size_t maxALUShaderInvocations;
        size_t maxConcurrentShaderInvocations;
        uint32_t maxShaderComputeUnits;
#ifdef _IRR_COMPILE_WITH_OPENCL_
        cl_device_id clDevice;
        size_t clPlatformIx, clDeviceIx;
#endif // _IRR_COMPILE_WITH_OPENCL_

        FW_Mutex* glContextMutex;
		SAuxContext* AuxContexts;

		E_DEVICE_TYPE DeviceType;
	};

} // end namespace video
} // end namespace irr


#endif // _IRR_COMPILE_WITH_OPENGL_
#endif


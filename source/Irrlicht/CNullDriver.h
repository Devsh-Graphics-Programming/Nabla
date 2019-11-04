// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_VIDEO_NULL_H_INCLUDED__
#define __C_VIDEO_NULL_H_INCLUDED__

#include "IVideoDriver.h"
#include "IFileSystem.h"
#include "irr/asset/IMesh.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "IMeshSceneNode.h"
#include "CFPSCounter.h"
#include "SExposedVideoData.h"
#include "FW_Mutex.h"


#ifdef _MSC_VER
#pragma warning( disable: 4996)
#endif

namespace irr
{
namespace video
{

class CNullDriver : public IVideoDriver
{
    protected:
		//! destructor
		virtual ~CNullDriver();

	public:
        static FW_AtomicCounter ReallocationCounter;
        static int32_t incrementAndFetchReallocCounter();

		//! constructor
		CNullDriver(IrrlichtDevice* dev, io::IFileSystem* io, const core::dimension2d<uint32_t>& screenSize);

        inline bool isAllowedBufferViewFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline bool isColorRenderableFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline bool isHardwareBlendableFormat(asset::E_FORMAT _fmt) const override { return false; }

        bool bindGraphicsPipeline(video::IGPURenderpassIndependentPipeline* _gpipeline) override { return false; }

        bool bindDescriptorSets(E_PIPELINE_BIND_POINT _pipelineType, const IGPUPipelineLayout* _layout,
            uint32_t _first, uint32_t _count, const IGPUDescriptorSet** _descSets, core::smart_refctd_dynamic_array<uint32_t>* _dynamicOffsets) override 
        { 
            return false; 
        }

		//!
        virtual bool initAuxContext() {return false;}
        virtual bool deinitAuxContext() {return false;}

		virtual bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0) override;

		virtual bool endScene() override;

		//!
		virtual void issueGPUTextureBarrier() override {}

        //! GPU fence, is signalled when preceeding GPU work is completed
        virtual core::smart_refctd_ptr<IDriverFence> placeFence(const bool& implicitFlushWaitSameThread=false) override {return nullptr;}

		//! Sets new multiple render targets.
		virtual bool setRenderTarget(IFrameBuffer* frameBuffer, bool setNewViewport=true) override;

		//! Clears the ZBuffer.
		/** Note that you usually need not to call this method, as it
		is automatically done in IVideoDriver::beginScene() or
		IVideoDriver::setRenderTarget() if you enable zBuffer. But if
		you have to render some special things, you can clear the
		zbuffer during the rendering process with this method any time.
		*/
		virtual void clearZBuffer(const float &depth=0.0) override;

		virtual void clearStencilBuffer(const int32_t &stencil) override;

		virtual void clearZStencilBuffers(const float &depth, const int32_t &stencil) override;

		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals) override;
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals) override;
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals) override;

		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals) override;
		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals) override;


		//! sets a viewport
		virtual void setViewPort(const core::rect<int32_t>& area) override;

		//! gets the area of the current viewport
		virtual const core::rect<int32_t>& getViewPort() const override;

        virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb) override;

		//! Indirect Draw
		virtual void drawArraysIndirect(const asset::SBufferBinding<IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) override
        {
        }
		virtual void drawIndexedIndirect(const asset::SBufferBinding<IGPUBuffer> _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        asset::E_INDEX_TYPE indexType, const IGPUBuffer* indexBuff,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer = nullptr, size_t countOffset = 0u) override
        {
        }

		//! get color format of the current color buffer
		virtual asset::E_FORMAT getColorFormat() const;

		//! get screen size
		virtual const core::dimension2d<uint32_t>& getScreenSize() const;

		//! get render target size
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const;

		// get current frames per second value
		virtual int32_t getFPS() const;

		//! returns amount of primitives (mostly triangles) were drawn in the last frame.
		//! very useful method for statistics.
		virtual uint32_t getPrimitiveCountDrawn( uint32_t param = 0 ) const;

		//! \return Returns the name of the video driver. Example: In case of the DIRECT3D8
		//! driver, it would return "Direct3D8.1".
		virtual const wchar_t* getName() const;

		virtual void removeFrameBuffer(IFrameBuffer* framebuf);

		virtual void removeAllFrameBuffers();

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth=true, bool copyStencil=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false);

		//! Returns the maximum amount of primitives (mostly vertices) which
		//! the device is able to render with one drawIndexedTriangleList
		//! call.
		virtual uint32_t getMaximalIndicesCount() const;


	public:
		virtual void beginQuery(IQueryObject* query);
		virtual void endQuery(IQueryObject* query);

		//! Only used by the engine internally.
		/** Used to notify the driver that the window was resized. */
		virtual void OnResize(const core::dimension2d<uint32_t>& size);

		//! Returns driver and operating system specific data about the IVideoDriver.
		virtual const SExposedVideoData& getExposedVideoData();

		//! Returns type of video driver
		virtual E_DRIVER_TYPE getDriverType() const override;

		//! Enable/disable a clipping plane.
		//! There are at least 6 clipping planes available for the user to set at will.
		//! \param index: The plane index. Must be between 0 and MaxUserClipPlanes.
		//! \param enable: If true, enable the clipping plane else disable it.
		virtual void enableClipPlane(uint32_t index, bool enable);

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo() {return "Not available on this driver.";}

		//! Returns the maximum texture size supported.
		virtual const uint32_t* getMaxTextureSize(const IGPUImageView::E_TYPE& type) const override;

		//!
		virtual uint32_t getRequiredUBOAlignment() const override {return 0u;}

		//!
		virtual uint32_t getRequiredSSBOAlignment() const override {return 0u;}

		//!
		virtual uint32_t getRequiredTBOAlignment() const override {return 0u;}

        virtual uint32_t getMaxComputeWorkGroupSize(uint32_t) const override { return 0u; }

        virtual uint64_t getMaxUBOSize() const override { return 0ull; }

        virtual uint64_t getMaxSSBOSize() const override { return 0ull; }

        virtual uint64_t getMaxTBOSizeInTexels() const override { return 0ull; }

        virtual uint64_t getMaxBufferSize() const override { return 0ull; }

        uint32_t getMaxUBOBindings() const override { return 0u; }
        uint32_t getMaxSSBOBindings() const override { return 0u; }
        uint32_t getMaxTextureBindings() const override { return 0u; }
        uint32_t getMaxTextureBindingsCompute() const override { return 0u; }
        uint32_t getMaxImageBindings() const override { return 0u; }

	protected:
        void bindDescriptorSets_generic(const IGPUPipelineLayout* _newLayout, uint32_t _first, uint32_t _count, const IGPUDescriptorSet** _descSets,
            const IGPUPipelineLayout** _destPplnLayouts);

		// prints renderer version
		void printVersion();

    protected:
        IQueryObject* currentQuery[EQOT_COUNT];

		io::IFileSystem* FileSystem;

		core::rect<int32_t> ViewPort;
		core::dimension2d<uint32_t> ScreenSize;

		CFPSCounter FPSCounter;

		uint32_t PrimitivesDrawn;

		SExposedVideoData ExposedData;

		uint32_t MaxTextureSizes[IGPUImageView::ET_COUNT][3];
	};

} // end namespace video
} // end namespace irr


#endif

// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_VIDEO_DRIVER_H_INCLUDED__
#define __IRR_I_VIDEO_DRIVER_H_INCLUDED__

#include "rect.h"
#include "SColor.h"
#include "matrixutil.h"
#include "dimension2d.h"
#include "position2d.h"
#include "IDriverFence.h"
#include "ITransformFeedback.h"
#include "SExposedVideoData.h"
#include "IDriver.h"
#include "irr/video/CDerivativeMapCreator.h"
#include "irr/video/IGPUBufferView.h"
#include "irr/video/IGPURenderpassIndependentPipeline.h"

namespace irr
{
namespace io
{
	class IReadFile;
	class IWriteFile;
} // end namespace io

namespace video
{
	class CImageData;
    class IGPUShader;
    class IGPUSpecializedShader;

	//! enumeration for geometry transformation states
	enum E_4X3_TRANSFORMATION_STATE
	{
		//! View transformation
		E4X3TS_VIEW = 0,
		//! World transformation
		E4X3TS_WORLD,
		//!
		E4X3TS_WORLD_VIEW,
		//!
		E4X3TS_VIEW_INVERSE,
		//!
		E4X3TS_WORLD_INVERSE,
		//!
		E4X3TS_WORLD_VIEW_INVERSE,
		//!
		E4X3TS_NORMAL_MATRIX,
		//! Not used
		E4X3TS_COUNT
	};
	enum E_PROJECTION_TRANSFORMATION_STATE
	{
		//! Projection transformation
		EPTS_PROJ,
		//!
		EPTS_PROJ_VIEW,
		//!
		EPTS_PROJ_VIEW_WORLD,
		//!
		EPTS_PROJ_INVERSE,
		//!
		EPTS_PROJ_VIEW_INVERSE,
		//!
		EPTS_PROJ_VIEW_WORLD_INVERSE,
		//! Not used
		EPTS_COUNT
	};

	enum E_SCREEN_BUFFERS
	{
		ESB_FRONT_LEFT=0,
		ESB_FRONT_RIGHT,
		ESB_BACK_LEFT,
		ESB_BACK_RIGHT
	};

	enum E_MIP_CHAIN_ERROR
	{
	    EMCE_NO_ERR=0,
	    EMCE_SUB_IMAGE_OUT_OF_BOUNDS,
	    EMCE_MIP_LEVEL_OUT_OF_BOUND,
	    EMCE_INVALID_IMAGE,
	    EMCE_OTHER_ERR
	};

    enum E_PIPELINE_BIND_POINT
    {
        EPBP_GRAPHICS = 0,
        EPBP_COMPUTE = 1
    };

	//! Interface to driver which is able to perform 2d and 3d graphics functions.
	/** This interface is one of the most important interfaces of
	the Irrlicht Engine: All rendering and texture manipulation is done with
	this interface. You are able to use the Irrlicht Engine by only
	invoking methods of this interface if you like to, although the
	irr::scene::ISceneManager interface provides a lot of powerful classes
	and methods to make the programmer's life easier.
	*/
	class IVideoDriver : public IDriver
	{
	public:
        IVideoDriver(IrrlichtDevice* _dev) : IDriver(_dev) {}



        virtual bool initAuxContext() = 0;
        virtual bool deinitAuxContext() = 0;


        virtual bool isAllowedBufferViewFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isColorRenderableFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isHardwareBlendableFormat(asset::E_FORMAT _fmt) const = 0;


        virtual bool bindGraphicsPipeline(video::IGPURenderpassIndependentPipeline* _gpipeline) = 0;

        virtual bool bindDescriptorSets(E_PIPELINE_BIND_POINT _pipelineType, const IGPUPipelineLayout* _layout,
            uint32_t _first, uint32_t _count, const IGPUDescriptorSet** _descSets, core::smart_refctd_dynamic_array<uint32_t>* _dynamicOffsets) = 0;

		//! Applications must call this method before performing any rendering.
		/** This method can clear the back- and the z-buffer.
		\param backBuffer Specifies if the back buffer should be
		cleared, which means that the screen is filled with the color
		specified. If this parameter is false, the back buffer will
		not be cleared and the color parameter is ignored.
		\param zBuffer Specifies if the depth buffer (z buffer) should
		be cleared. It is not nesesarry to do so if only 2d drawing is
		used.
		\param color The color used for back buffer clearing
		\param videoData Handle of another window, if you want the
		bitmap to be displayed on another window. If this is an empty
		element, everything will be displayed in the default window.
		Note: This feature is not fully implemented for all devices.
		\param sourceRect Pointer to a rectangle defining the source
		rectangle of the area to be presented. Set to null to present
		everything. Note: not implemented in all devices.
		\return False if failed. */
		virtual bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0) =0;

		//! Presents the rendered image to the screen.
		/** Applications must call this method after performing any
		rendering.
		\return False if failed and true if succeeded. */
		virtual bool endScene() =0;

		//!
		virtual void issueGPUTextureBarrier() =0;

		//! Sets transformation matrices.
		/** \param state Transformation type to be set, e.g. view,
		world, or projection.
		\param mat Matrix describing the transformation. */
		virtual void setTransform(const E_4X3_TRANSFORMATION_STATE& state, const core::matrix4x3& mat) =0;

		virtual void setTransform(const E_PROJECTION_TRANSFORMATION_STATE& state, const core::matrix4SIMD& mat) =0;

		//! Returns the transformation set by setTransform
		/** \param state Transformation type to query
		\return Matrix describing the transformation. */
		virtual const core::matrix4x3& getTransform(const E_4X3_TRANSFORMATION_STATE& state) =0;

		virtual const core::matrix4SIMD& getTransform(const E_PROJECTION_TRANSFORMATION_STATE& state) =0;

        //! A.
        virtual IMultisampleTexture* createMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size,
                                                           asset::E_FORMAT format = asset::EF_B8G8R8A8_UNORM, const bool& fixedSampleLocations = false) {return nullptr;}

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth=true, bool copyStencil=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false) = 0;

        virtual void removeFrameBuffer(IFrameBuffer* framebuf) = 0;

		//! This only removes all IFrameBuffers created in the calling thread.
		virtual void removeAllFrameBuffers() =0;


		//! Queries
		virtual void beginQuery(IQueryObject* query) = 0;
		virtual void endQuery(IQueryObject* query) = 0;

		//! Sets new multiple render targets.
		virtual bool setRenderTarget(IFrameBuffer* frameBuffer, bool setNewViewport=true) = 0;

		//! Clears the ZBuffer.
		/** Note that you usually need not to call this method, as it
		is automatically done in IVideoDriver::beginScene() or
		IVideoDriver::setRenderTarget() if you enable zBuffer. But if
		you have to render some special things, you can clear the
		zbuffer during the rendering process with this method any time.
		*/
		virtual void clearZBuffer(const float &depth=0.0) =0;

		virtual void clearStencilBuffer(const int32_t &stencil) =0;

		virtual void clearZStencilBuffers(const float &depth, const int32_t &stencil) =0;

		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals) =0;
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals) =0;
		virtual void clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals) =0;

		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals) =0;
		virtual void clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals) =0;


		//! Sets a new viewport.
		/** Every rendering operation is done into this new area.
		\param area: Rectangle defining the new area of rendering
		operations. */
		virtual void setViewPort(const core::rect<int32_t>& area) =0;

		//! Gets the area of the current viewport.
		/** \return Rectangle of the current viewport. */
		virtual const core::rect<int32_t>& getViewPort() const =0;

		//! Draws a mesh buffer
		/** \param mb Buffer to draw */
		virtual void drawMeshBuffer(const video::IGPUMeshBuffer* mb) =0;

		//! Indirect Draw
		virtual void drawArraysIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const asset::E_PRIMITIVE_TYPE& mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride) =0;
		virtual void drawIndexedIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const asset::E_PRIMITIVE_TYPE& mode,
                                        const asset::E_INDEX_TYPE& type,
                                        const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride) =0;

		//! Get the size of the screen or render window.
		/** \return Size of screen or render window. */
		virtual const core::dimension2d<uint32_t>& getScreenSize() const =0;

		//! Get the size of the current render target
		/** This method will return the screen size if the driver
		doesn't support render to texture, or if the current render
		target is the screen.
		\return Size of render target or screen/window */
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const =0;

		//! Returns current frames per second value.
		/** This value is updated approximately every 1.5 seconds and
		is only intended to provide a rough guide to the average frame
		rate. It is not suitable for use in performing timing
		calculations or framerate independent movement.
		\return Approximate amount of frames per second drawn. */
		virtual int32_t getFPS() const =0;

		//! Returns amount of primitives (mostly triangles) which were drawn in the last frame.
		/** Together with getFPS() very useful method for statistics.
		\param mode Defines if the primitives drawn are accumulated or
		counted per frame.
		\return Amount of primitives drawn in the last frame. */
		virtual uint32_t getPrimitiveCountDrawn( uint32_t mode =0 ) const =0;

		//! Returns the maximum amount of primitives
		/** (mostly vertices) which the device is able to render.
		\return Maximum amount of primitives. */
		virtual uint32_t getMaximalIndicesCount() const =0;

		//! Enables or disables a texture creation flag.
		/** These flags define how textures should be created. By
		changing this value, you can influence for example the speed of
		rendering a lot. But please note that the video drivers take
		this value only as recommendation. It could happen that you
		enable the ETCF_ALWAYS_16_BIT mode, but the driver still creates
		32 bit textures.
		\param flag Texture creation flag.
		\param enabled Specifies if the given flag should be enabled or
		disabled. */
		virtual void setTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag, bool enabled=true) =0;

		//! Returns if a texture creation flag is enabled or disabled.
		/** You can change this value using setTextureCreationFlag().
		\param flag Texture creation flag.
		\return The current texture creation flag enabled mode. */
		virtual bool getTextureCreationFlag(E_TEXTURE_CREATION_FLAG flag) const =0;

		//! Event handler for resize events. Only used by the engine internally.
		/** Used to notify the driver that the window was resized.
		Usually, there is no need to call this method. */
		virtual void OnResize(const core::dimension2d<uint32_t>& size) =0;

		//! Returns driver and operating system specific data about the IVideoDriver.
		/** This method should only be used if the engine should be
		extended without having to modify the source of the engine.
		\return Collection of device dependent pointers. */
		virtual const SExposedVideoData& getExposedVideoData() =0;

		//! Gets the IGPUProgrammingServices interface.
		/** \return Pointer to the IGPUProgrammingServices. Returns 0
		if the video driver does not support this. For example the
		Software driver and the Null driver will always return 0. */
		//virtual IGPUProgrammingServices* getGPUProgrammingServices() =0;

		//! Enable or disable a clipping plane.
		/** There are at least 6 clipping planes available for the user
		to set at will.
		\param index The plane index. Must be between 0 and
		MaxUserClipPlanes.
		\param enable If true, enable the clipping plane else disable
		it. */
		virtual void enableClipPlane(uint32_t index, bool enable) =0;

        virtual const CDerivativeMapCreator* getDerivativeMapCreator() const { return nullptr; }
	};

} // end namespace video
} // end namespace irr


#endif

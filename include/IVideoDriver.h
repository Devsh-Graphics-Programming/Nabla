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
#include "SMaterial.h"
#include "IDriverFence.h"
#include "irr/video/SGPUMesh.h"
#include "SExposedVideoData.h"
#include "IDriver.h"
#include "irr/asset/EFormat.h"

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
	class IImage;
	class IImageLoader;
	class IImageWriter;
	class IMaterialRenderer;
	class IGPUProgrammingServices;

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


        virtual bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isColorRenderableFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const = 0;
        virtual bool isHardwareBlendableFormat(asset::E_FORMAT _fmt) const = 0;


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

		//! Retrieve the number of image loaders
		/** \return Number of image loaders */
		virtual uint32_t getImageLoaderCount() const = 0;

		//! Retrieve the given image loader
		/** \param n The index of the loader to retrieve. This parameter is an 0-based
		array index.
		\return A pointer to the specified loader, 0 if the index is incorrect. */
		virtual IImageLoader* getImageLoader(uint32_t n) = 0;

		//! Retrieve the number of image writers
		/** \return Number of image writers */
		virtual uint32_t getImageWriterCount() const = 0;

		//! Retrieve the given image writer
		/** \param n The index of the writer to retrieve. This parameter is an 0-based
		array index.
		\return A pointer to the specified writer, 0 if the index is incorrect. */
		virtual IImageWriter* getImageWriter(uint32_t n) = 0;

		//! Sets a material.
		/** All 3d drawing functions will draw geometry using this material thereafter.
		\param material: Material to be used from now on. */
		virtual void setMaterial(const SGPUMaterial& material) =0;

		//! A.
		/** \param B
		\param C
		\return D. */
        virtual E_MIP_CHAIN_ERROR validateMipChain(const ITexture* tex, const core::vector<asset::CImageData*>& mipChain) = 0;

        //! A.
        virtual IMultisampleTexture* addMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size,
                                                           asset::E_FORMAT format = asset::EF_B8G8R8A8_UNORM, const bool& fixedSampleLocations = false) {return nullptr;}

        //! A.
        virtual ITextureBufferObject* addTextureBufferObject(IGPUBuffer* buf, const ITextureBufferObject::E_TEXURE_BUFFER_OBJECT_FORMAT& format = ITextureBufferObject::ETBOF_RGBA8,
                                                             const size_t& offset=0, const size_t& length=0) {return nullptr;}

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth=true, bool copyStencil=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false) = 0;

		virtual void removeMultisampleTexture(IMultisampleTexture* tex) =0;

		virtual void removeTextureBufferObject(ITextureBufferObject* tbo) =0;

        virtual void removeFrameBuffer(IFrameBuffer* framebuf) = 0;

		virtual void removeAllMultisampleTextures() =0;

		virtual void removeAllTextureBufferObjects() =0;

		//! This only removes all IFrameBuffers created in the calling thread.
		virtual void removeAllFrameBuffers() =0;


		//! Queries
		virtual void beginQuery(IQueryObject* query) = 0;
		virtual void endQuery(IQueryObject* query) = 0;
		virtual void beginQuery(IQueryObject* query, const size_t& index) = 0;
		virtual void endQuery(IQueryObject* query, const size_t& index) = 0;


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



		virtual ITransformFeedback* createTransformFeedback() = 0;

		//!
		virtual void bindTransformFeedback(ITransformFeedback* xformFeedback) = 0;

		virtual ITransformFeedback* getBoundTransformFeedback() = 0;

        /** Only POINTS, LINES, and TRIANGLES are allowed as capture types.. no strips or fans!
        This issues an implicit call to bindTransformFeedback()
        **/
		virtual void beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const asset::E_PRIMITIVE_TYPE& primType=asset::EPT_POINTS) = 0;

		//! A redundant wrapper call to ITransformFeedback::pauseTransformFeedback(), made just for clarity
		virtual void pauseTransformFeedback() = 0;

		//! A redundant wrapper call to ITransformFeedback::pauseTransformFeedback(), made just for clarity
		virtual void resumeTransformFeedback() = 0;

        //! This issues an implicit call to bindTransformFeedback(NULL)
		virtual void endTransformFeedback() = 0;


		//! Sets a new viewport.
		/** Every rendering operation is done into this new area.
		\param area: Rectangle defining the new area of rendering
		operations. */
		virtual void setViewPort(const core::rect<int32_t>& area) =0;

		//! Gets the area of the current viewport.
		/** \return Rectangle of the current viewport. */
		virtual const core::rect<int32_t>& getViewPort() const =0;

		//! Draws a 2d image without any special effects
		/** \param texture Pointer to texture to use.
		\param destPos Upper left 2d destination position where the
		image will be drawn. */
		virtual void draw2DImage(const video::ITexture* texture,
			const core::position2d<int32_t>& destPos) =0;

		//! Draws a 2d image using a color
		/** (if color is other than
		Color(255,255,255,255)) and the alpha channel of the texture.
		\param texture Texture to be drawn.
		\param destPos Upper left 2d destination position where the
		image will be drawn.
		\param sourceRect Source rectangle in the image.
		\param clipRect Pointer to rectangle on the screen where the
		image is clipped to.
		If this pointer is NULL the image is not clipped.
		\param color Color with which the image is drawn. If the color
		equals Color(255,255,255,255) it is ignored. Note that the
		alpha component is used: If alpha is other than 255, the image
		will be transparent.
		\param useAlphaChannelOfTexture: If true, the alpha channel of
		the texture is used to draw the image.*/
		virtual void draw2DImage(const video::ITexture* texture, const core::position2d<int32_t>& destPos,
			const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect =0,
			SColor color=SColor(255,255,255,255), bool useAlphaChannelOfTexture=false) =0;

		//! Draws a set of 2d images, using a color and the alpha channel of the texture.
		/** The images are drawn beginning at pos and concatenated in
		one line. All drawings are clipped against clipRect (if != 0).
		The subtextures are defined by the array of sourceRects and are
		chosen by the indices given.
		\param texture Texture to be drawn.
		\param pos Upper left 2d destination position where the image
		will be drawn.
		\param sourceRects Source rectangles of the image.
		\param indices List of indices which choose the actual
		rectangle used each time.
		\param kerningWidth Offset to Position on X
		\param clipRect Pointer to rectangle on the screen where the
		image is clipped to.
		If this pointer is 0 then the image is not clipped.
		\param color Color with which the image is drawn.
		Note that the alpha component is used. If alpha is other than
		255, the image will be transparent.
		\param useAlphaChannelOfTexture: If true, the alpha channel of
		the texture is used to draw the image. */
		virtual void draw2DImageBatch(const video::ITexture* texture,
				const core::position2d<int32_t>& pos,
				const core::vector<core::rect<int32_t> >& sourceRects,
				const core::vector<int32_t>& indices,
				int32_t kerningWidth=0,
				const core::rect<int32_t>* clipRect=0,
				SColor color=SColor(255,255,255,255),
				bool useAlphaChannelOfTexture=false) =0;

		//! Draws a set of 2d images, using a color and the alpha channel of the texture.
		/** All drawings are clipped against clipRect (if != 0).
		The subtextures are defined by the array of sourceRects and are
		positioned using the array of positions.
		\param texture Texture to be drawn.
		\param positions Array of upper left 2d destinations where the
		images will be drawn.
		\param sourceRects Source rectangles of the image.
		\param clipRect Pointer to rectangle on the screen where the
		images are clipped to.
		If this pointer is 0 then the image is not clipped.
		\param color Color with which the image is drawn.
		Note that the alpha component is used. If alpha is other than
		255, the image will be transparent.
		\param useAlphaChannelOfTexture: If true, the alpha channel of
		the texture is used to draw the image. */
		virtual void draw2DImageBatch(const video::ITexture* texture,
				const core::vector<core::position2d<int32_t> >& positions,
				const core::vector<core::rect<int32_t> >& sourceRects,
				const core::rect<int32_t>* clipRect=0,
				SColor color=SColor(255,255,255,255),
				bool useAlphaChannelOfTexture=false) =0;

		//! Draws a part of the texture into the rectangle. Note that colors must be an array of 4 colors if used.
		/** Suggested and first implemented by zola.
		\param texture The texture to draw from
		\param destRect The rectangle to draw into
		\param sourceRect The rectangle denoting a part of the texture
		\param clipRect Clips the destination rectangle (may be 0)
		\param colors Array of 4 colors denoting the color values of
		the corners of the destRect
		\param useAlphaChannelOfTexture True if alpha channel will be
		blended. */
		virtual void draw2DImage(const video::ITexture* texture, const core::rect<int32_t>& destRect,
			const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect =0,
			const video::SColor * const colors=0, bool useAlphaChannelOfTexture=false) =0;

		//! Draws a 2d rectangle.
		/** \param color Color of the rectangle to draw. The alpha
		component will not be ignored and specifies how transparent the
		rectangle will be.
		\param pos Position of the rectangle.
		\param clip Pointer to rectangle against which the rectangle
		will be clipped. If the pointer is null, no clipping will be
		performed. */
		virtual void draw2DRectangle(SColor color, const core::rect<int32_t>& pos,
			const core::rect<int32_t>* clip =0) =0;

		//! Draws a 2d rectangle with a gradient.
		/** \param colorLeftUp Color of the upper left corner to draw.
		The alpha component will not be ignored and specifies how
		transparent the rectangle will be.
		\param colorRightUp Color of the upper right corner to draw.
		The alpha component will not be ignored and specifies how
		transparent the rectangle will be.
		\param colorLeftDown Color of the lower left corner to draw.
		The alpha component will not be ignored and specifies how
		transparent the rectangle will be.
		\param colorRightDown Color of the lower right corner to draw.
		The alpha component will not be ignored and specifies how
		transparent the rectangle will be.
		\param pos Position of the rectangle.
		\param clip Pointer to rectangle against which the rectangle
		will be clipped. If the pointer is null, no clipping will be
		performed. */
		virtual void draw2DRectangle(const core::rect<int32_t>& pos,
				SColor colorLeftUp, SColor colorRightUp,
				SColor colorLeftDown, SColor colorRightDown,
				const core::rect<int32_t>* clip =0) =0;

		//! Draws the outline of a 2D rectangle.
		/** \param pos Position of the rectangle.
		\param color Color of the rectangle to draw. The alpha component
		specifies how transparent the rectangle outline will be. */
		virtual void draw2DRectangleOutline(const core::recti& pos,
				SColor color=SColor(255,255,255,255)) =0;

		//! Draws a 2d line. Both start and end will be included in coloring.
		/** \param start Screen coordinates of the start of the line
		in pixels.
		\param end Screen coordinates of the start of the line in
		pixels.
		\param color Color of the line to draw. */
		virtual void draw2DLine(const core::position2d<int32_t>& start,
					const core::position2d<int32_t>& end,
					SColor color=SColor(255,255,255,255)) =0;


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

		//! Adds an external image loader to the engine.
		/** This is useful if the Irrlicht Engine should be able to load
		textures of currently unsupported file formats (e.g. gif). The
		IImageLoader only needs to be implemented for loading this file
		format. A pointer to the implementation can be passed to the
		engine using this method.
		\param loader Pointer to the external loader created. */
		virtual void addExternalImageLoader(IImageLoader* loader) =0;

		//! Adds an external image writer to the engine.
		/** This is useful if the Irrlicht Engine should be able to
		write textures of currently unsupported file formats (e.g
		.gif). The IImageWriter only needs to be implemented for
		writing this file format. A pointer to the implementation can
		be passed to the engine using this method.
		\param writer: Pointer to the external writer created. */
		virtual void addExternalImageWriter(IImageWriter* writer) =0;

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

		//! Creates a ... from a file.
		/**
		\param filename .
		\return .
		If you no longer need the image data, you should call CImageData::drop().
		See IReferenceCounted::drop() for more information. */
		virtual core::vector<asset::CImageData*> createImageDataFromFile(const io::path& filename) = 0;

		//! Creates a ... from a file.
		/**
		\param file .
		\return .
		If you no longer need the image data, you should call CImageData::drop()
		on all vector members.
		See IReferenceCounted::drop() for more information. */
		virtual core::vector<asset::CImageData*> createImageDataFromFile(io::IReadFile* file) =0;

		//! Writes the provided image to a file.
		/** Requires that there is a suitable image writer registered
		for writing the image.
		\param image Image to write.
		\param filename Name of the file to write.
		\param param Control parameter for the backend (e.g. compression
		level).
		\return True on successful write. */
		virtual bool writeImageToFile(IImage* image, const io::path& filename, uint32_t param = 0) = 0;

		//! Writes the provided image to a file.
		/** Requires that there is a suitable image writer registered
		for writing the image.
		\param image Image to write.
		\param file  An already open io::IWriteFile object. The name
		will be used to determine the appropriate image writer to use.
		\param param Control parameter for the backend (e.g. compression
		level).
		\return True on successful write. */
		virtual bool writeImageToFile(IImage* image, io::IWriteFile* file, uint32_t param =0) =0;

		//! Creates a software image from a byte array.
		/** No hardware texture will be created for this image.
		\param imageData
		\param ownForeignMemory If true, the image will use the data
		pointer directly and own it afterwards. If false, the memory
		will by copied internally.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		virtual IImage* createImageFromData(asset::CImageData* imageData, bool ownForeignMemory=true) =0;

		//!
		virtual IImage* createImage(const asset::E_FORMAT& format, const core::dimension2d<uint32_t>& size) =0;

		//! Event handler for resize events. Only used by the engine internally.
		/** Used to notify the driver that the window was resized.
		Usually, there is no need to call this method. */
		virtual void OnResize(const core::dimension2d<uint32_t>& size) =0;

		//! Adds a new material renderer to the video device.
		/** Use this method to extend the VideoDriver with new material
		types. To extend the engine using this method do the following:
		Derive a class from IMaterialRenderer and override the methods
		you need. For setting the right renderstates, you can try to
		get a pointer to the real rendering device using
		IVideoDriver::getExposedVideoData(). Add your class with
		IVideoDriver::addMaterialRenderer(). To use an object being
		displayed with your new material, set the MaterialType member of
		the SGPUMaterial struct to the value returned by this method.
		If you simply want to create a new material using vertex and/or
		pixel shaders it would be easier to use the
		video::IGPUProgrammingServices interface which you can get
		using the getGPUProgrammingServices() method.
		\param renderer A pointer to the new renderer.
		\param name Optional name for the material renderer entry.
		\return The number of the material type which can be set in
		SGPUMaterial::MaterialType to use the renderer. -1 is returned if
		an error occured. For example if you tried to add an material
		renderer to the software renderer or the null device, which do
		not accept material renderers. */
		virtual int32_t addMaterialRenderer(IMaterialRenderer* renderer, const char* name =0) =0;

		//! Get access to a material renderer by index.
		/** \param idx Id of the material renderer. Can be a value of
		the E_MATERIAL_TYPE enum or a value which was returned by
		addMaterialRenderer().
		\return Pointer to material renderer or null if not existing. */
		virtual IMaterialRenderer* getMaterialRenderer(uint32_t idx) =0;

		//! Get amount of currently available material renderers.
		/** \return Amount of currently available material renderers. */
		virtual uint32_t getMaterialRendererCount() const =0;

		//! Get name of a material renderer
		/** This string can, e.g., be used to test if a specific
		renderer already has been registered/created, or use this
		string to store data about materials: This returned name will
		be also used when serializing materials.
		\param idx Id of the material renderer. Can be a value of the
		E_MATERIAL_TYPE enum or a value which was returned by
		addMaterialRenderer().
		\return String with the name of the renderer, or 0 if not
		exisiting */
		virtual const char* getMaterialRendererName(uint32_t idx) const =0;

		//! Sets the name of a material renderer.
		/** Will have no effect on built-in material renderers.
		\param idx: Id of the material renderer. Can be a value of the
		E_MATERIAL_TYPE enum or a value which was returned by
		addMaterialRenderer().
		\param name: New name of the material renderer. */
		virtual void setMaterialRendererName(int32_t idx, const char* name) =0;

		//! Returns driver and operating system specific data about the IVideoDriver.
		/** This method should only be used if the engine should be
		extended without having to modify the source of the engine.
		\return Collection of device dependent pointers. */
		virtual const SExposedVideoData& getExposedVideoData() =0;

		//! Gets the IGPUProgrammingServices interface.
		/** \return Pointer to the IGPUProgrammingServices. Returns 0
		if the video driver does not support this. For example the
		Software driver and the Null driver will always return 0. */
		virtual IGPUProgrammingServices* getGPUProgrammingServices() =0;

		//! Enable or disable a clipping plane.
		/** There are at least 6 clipping planes available for the user
		to set at will.
		\param index The plane index. Must be between 0 and
		MaxUserClipPlanes.
		\param enable If true, enable the clipping plane else disable
		it. */
		virtual void enableClipPlane(uint32_t index, bool enable) =0;


		//! Get the 2d override material for altering its values
		/** The 2d override materual allows to alter certain render
		states of the 2d methods. Not all members of SGPUMaterial are
		honored, especially not MaterialType and Textures. Moreover,
		the zbuffer is always ignored, and lighting is always off. All
		other flags can be changed, though some might have to effect
		in most cases.
		Please note that you have to enable/disable this effect with
		enableInitMaterial2D(). This effect is costly, as it increases
		the number of state changes considerably. Always reset the
		values when done.
		\return Material reference which should be altered to reflect
		the new settings.
		*/
		virtual SGPUMaterial& getMaterial2D() =0;

		//! Enable the 2d override material
		/** \param enable Flag which tells whether the material shall be
		enabled or disabled. */
		virtual void enableMaterial2D(bool enable=true) =0;

		//! Only used by the engine internally.
		/** Passes the global material flag AllowZWriteOnTransparent.
		Use the SceneManager attribute to set this value from your app.
		\param flag Default behavior is to disable ZWrite, i.e. false. */
		virtual void setAllowZWriteOnTransparent(bool flag) =0;
	};

} // end namespace video
} // end namespace irr


#endif

// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_VIDEO_DRIVER_H_INCLUDED__
#define __IRR_I_VIDEO_DRIVER_H_INCLUDED__

#include "rect.h"
#include "SColor.h"
#include "IGPUMappedBuffer.h"
#include "ITexture.h"
#include "IRenderBuffer.h"
#include "IFrameBuffer.h"
#include "irrArray.h"
#include "matrix4x3.h"
#include "plane3d.h"
#include "dimension2d.h"
#include "position2d.h"
#include "SMaterial.h"
#include "IDriverFence.h"
#include "SMesh.h"
#include "IGPUTimestampQuery.h"
#include "IOcclusionQuery.h"
#include "triangle3d.h"
#include "EDriverTypes.h"
#include "EDriverFeatures.h"
#include "SExposedVideoData.h"

namespace irr
{
namespace io
{
	class IReadFile;
	class IWriteFile;
} // end namespace io

namespace video
{
	struct SLight;
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

	//! enumeration for signaling resources which were lost after the last render cycle
	/** These values can be signaled by the driver, telling the app that some resources
	were lost and need to be recreated. Irrlicht will sometimes recreate the actual objects,
	but the content needs to be recreated by the application. */
	enum E_LOST_RESOURCE
	{
		//! The whole device/driver is lost
		ELR_DEVICE = 1,
		//! All texture are lost, rare problem
		ELR_TEXTURES = 2,
		//! The Render Target Textures are lost, typical problem for D3D
		ELR_RTTS = 4,
		//! The HW buffers are lost, will be recreated automatically, but might require some more time this frame
		ELR_HW_BUFFERS = 8
	};

	enum E_SCREEN_BUFFERS
	{
		ESB_FRONT_LEFT=0,
		ESB_FRONT_RIGHT,
		ESB_BACK_LEFT,
		ESB_BACK_RIGHT
	};

	enum E_MESH_DESC_CONVERT_BEHAVIOUR
	{
	    EMDCB_CLONE_AND_MIRROR_LAYOUT = 0,
	    EMDCB_PACK_ATTRIBUTES_SINGLE_BUFFER,
	    EMDCB_PACK_ALL_SINGLE_BUFFER,
	    EMDCB_INTERLEAVED_PACK_ATTRIBUTES_SINGLE_BUFFER,
	    EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER
	};


	struct SOverrideMaterial
	{
		//! The Material values
		SMaterial Material;
		//! Which values are taken for override
		/** OR'ed values from E_MATERIAL_FLAGS. */
		u32 EnableFlags;
		//! Set in which render passes the material override is active.
		/** OR'ed values from E_SCENE_NODE_RENDER_PASS. */
		u16 EnablePasses;
		//! Global enable flag, overwritten by the SceneManager in each pass
		/** The Scenemanager uses the EnablePass array and sets Enabled to
		true if the Override material is enabled in the current pass. */
		bool Enabled;

		//! Default constructor
		SOverrideMaterial() : EnableFlags(0), EnablePasses(0), Enabled(false) {}

		//! Apply the enabled overrides
		void apply(SMaterial& material)
		{
			if (Enabled)
			{
				for (u32 i=0; i<32; ++i)
				{
					const u32 num=(1<<i);
					if (EnableFlags & num)
					{
						switch (num)
						{
						case EMF_WIREFRAME: material.Wireframe = Material.Wireframe; break;
						case EMF_POINTCLOUD: material.PointCloud = Material.PointCloud; break;
						case EMF_ZBUFFER: material.ZBuffer = Material.ZBuffer; break;
						case EMF_ZWRITE_ENABLE: material.ZWriteEnable = Material.ZWriteEnable; break;
						case EMF_BACK_FACE_CULLING: material.BackfaceCulling = Material.BackfaceCulling; break;
						case EMF_FRONT_FACE_CULLING: material.FrontfaceCulling = Material.FrontfaceCulling; break;
						case EMF_ANTI_ALIASING: material.AntiAliasing = Material.AntiAliasing; break;
						case EMF_COLOR_MASK: material.ColorMask = Material.ColorMask; break;
						case EMF_BLEND_OPERATION: material.BlendOperation = Material.BlendOperation; break;
						}
					}
				}
			}
		}

	};

	//! Interface to driver which is able to perform 2d and 3d graphics functions.
	/** This interface is one of the most important interfaces of
	the Irrlicht Engine: All rendering and texture manipulation is done with
	this interface. You are able to use the Irrlicht Engine by only
	invoking methods of this interface if you like to, although the
	irr::scene::ISceneManager interface provides a lot of powerful classes
	and methods to make the programmer's life easier.
	*/
	class IVideoDriver : public virtual IReferenceCounted
	{
	public:
        virtual IGPUBuffer* createGPUBuffer(const size_t &size, const void* data, const bool canModifySubData=false, const bool &inCPUMem=false, const E_GPU_BUFFER_ACCESS &usagePattern=EGBA_NONE) = 0;

	    virtual IGPUMappedBuffer* createPersistentlyMappedBuffer(const size_t &size, const void* data, const E_GPU_BUFFER_ACCESS &usagePattern, const bool &assumedCoherent, const bool &inCPUMem=true) = 0;

	    virtual scene::IGPUMeshDataFormatDesc* createGPUMeshDataFormatDesc() = 0;

	    virtual scene::IGPUMesh* createGPUMeshFromCPU(scene::ICPUMesh* mesh, const E_MESH_DESC_CONVERT_BEHAVIOUR& bufferOptions=EMDCB_CLONE_AND_MIRROR_LAYOUT) = 0;

        virtual void bufferCopy(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, const size_t& readOffset, const size_t& writeOffset, const size_t& length) = 0;


        virtual bool initAuxContext(const size_t& ctxIx) = 0;
        virtual bool deinitAuxContext() = 0;


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
				core::rect<s32>* sourceRect=0) =0;

		//! Presents the rendered image to the screen.
		/** Applications must call this method after performing any
		rendering.
		\return False if failed and true if succeeded. */
		virtual bool endScene() =0;

		//! Queries the features of the driver.
		/** Returns true if a feature is available
		\param feature Feature to query.
		\return True if the feature is available, false if not. */
		virtual bool queryFeature(E_VIDEO_DRIVER_FEATURE feature) const =0;

		//! Disable a feature of the driver.
		/** Can also be used to enable the features again. It is not
		possible to enable unsupported features this way, though.
		\param feature Feature to disable.
		\param flag When true the feature is disabled, otherwise it is enabled. */
		virtual void disableFeature(E_VIDEO_DRIVER_FEATURE feature, bool flag=true) =0;

		//! Check if the driver was recently reset.
		/** For d3d devices you will need to recreate the RTTs if the
		driver was reset. Should be queried right after beginScene().
		*/
		virtual bool checkDriverReset() =0;

		//! Sets transformation matrices.
		/** \param state Transformation type to be set, e.g. view,
		world, or projection.
		\param mat Matrix describing the transformation. */
		virtual void setTransform(const E_4X3_TRANSFORMATION_STATE& state, const core::matrix4x3& mat) =0;

		virtual void setTransform(const E_PROJECTION_TRANSFORMATION_STATE& state, const core::matrix4& mat) =0;

		//! Returns the transformation set by setTransform
		/** \param state Transformation type to query
		\return Matrix describing the transformation. */
		virtual const core::matrix4x3& getTransform(const E_4X3_TRANSFORMATION_STATE& state) =0;

		virtual const core::matrix4& getTransform(const E_PROJECTION_TRANSFORMATION_STATE& state) =0;

		//! Retrieve the number of image loaders
		/** \return Number of image loaders */
		virtual u32 getImageLoaderCount() const = 0;

		//! Retrieve the given image loader
		/** \param n The index of the loader to retrieve. This parameter is an 0-based
		array index.
		\return A pointer to the specified loader, 0 if the index is incorrect. */
		virtual IImageLoader* getImageLoader(u32 n) = 0;

		//! Retrieve the number of image writers
		/** \return Number of image writers */
		virtual u32 getImageWriterCount() const = 0;

		//! Retrieve the given image writer
		/** \param n The index of the writer to retrieve. This parameter is an 0-based
		array index.
		\return A pointer to the specified writer, 0 if the index is incorrect. */
		virtual IImageWriter* getImageWriter(u32 n) = 0;

		//! Sets a material.
		/** All 3d drawing functions will draw geometry using this material thereafter.
		\param material: Material to be used from now on. */
		virtual void setMaterial(const SMaterial& material) =0;

        //! needs to be "deleted" since its not refcounted
        /** Since not owned by any openGL context and hence not owned by driver
        **/
        virtual IDriverFence* placeFence() = 0;

		//! Get access to a named texture.
		/** Loads the texture from disk if it is not
		already loaded and generates mipmap levels if desired.
		Texture loading can be influenced using the
		setTextureCreationFlag() method. The texture can be in several
		imageformats, such as BMP, JPG, TGA, PCX, PNG, and PSD.
		\param filename Filename of the texture to be loaded.
		\return Pointer to the texture, or 0 if the texture
		could not be loaded. This pointer should not be dropped. See
		IReferenceCounted::drop() for more information. */
		virtual ITexture* getTexture(const io::path& filename) = 0;

		//! Get access to a named texture.
		/** Loads the texture from disk if it is not
		already loaded and generates mipmap levels if desired.
		Texture loading can be influenced using the
		setTextureCreationFlag() method. The texture can be in several
		imageformats, such as BMP, JPG, TGA, PCX, PNG, and PSD.
		\param file Pointer to an already opened file.
		\return Pointer to the texture, or 0 if the texture
		could not be loaded. This pointer should not be dropped. See
		IReferenceCounted::drop() for more information. */
		virtual ITexture* getTexture(io::IReadFile* file) =0;

		//! Returns a texture by index
		/** \param index: Index of the texture, must be smaller than
		getTextureCount() Please note that this index might change when
		adding or removing textures
		\return Pointer to the texture, or 0 if the texture was not
		set or index is out of bounds. This pointer should not be
		dropped. See IReferenceCounted::drop() for more information. */
		virtual ITexture* getTextureByIndex(u32 index) =0;

		//! Returns amount of textures currently loaded
		/** \return Amount of textures currently loaded */
		virtual u32 getTextureCount() const = 0;

		//! Renames a texture
		/** \param texture Pointer to the texture to rename.
		\param newName New name for the texture. This should be a unique name. */
		virtual void renameTexture(ITexture* texture, const io::path& newName) = 0;

		//! Creates an empty texture of specified size.
		/** \param size: Size of the texture.
		\param name A name for the texture. Later calls to
		getTexture() with this name will return this texture
		\param format Desired color format of the texture. Please note
		that the driver may choose to create the texture in another
		color format.
		\return Pointer to the newly created texture. This pointer
		should not be dropped. See IReferenceCounted::drop() for more
		information. */
		virtual ITexture* addTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels,
			const io::path& name, ECOLOR_FORMAT format = ECF_A8R8G8B8) = 0;

		//! Creates a texture from an IImage.
		/** \param name A name for the texture. Later calls of
		getTexture() with this name will return this texture
		\param image Image the texture is created from.
		\param mipmapData Optional pointer to a set of images which
		build up the whole mipmap set. Must be images of the same color
		type as image. If this parameter is not given, the mipmaps are
		derived from image.
		\return Pointer to the newly created texture. This pointer
		should not be dropped. See IReferenceCounted::drop() for more
		information. */
		virtual ITexture* addTexture(const io::path& name, IImage* image, void* mipmapData=0) = 0;

		virtual IRenderBuffer* addRenderBuffer(const core::dimension2d<u32>& size, ECOLOR_FORMAT format = ECF_A8R8G8B8) = 0;

        virtual IFrameBuffer* addFrameBuffer() = 0;

		virtual void blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out, bool copyDepth=true,
										core::recti srcRect=core::recti(0,0,0,0),
										core::recti dstRect=core::recti(0,0,0,0),
										bool bilinearFilter=false) = 0;

		//! Removes a texture from the texture cache and deletes it.
		/** This method can free a lot of memory!
		Please note that after calling this, the pointer to the
		ITexture may no longer be valid, if it was not grabbed before
		by other parts of the engine for storing it longer. So it is a
		good idea to set all materials which are using this texture to
		0 or another texture first.
		\param texture Texture to delete from the engine cache. */
		virtual void removeTexture(ITexture* texture) =0;

		virtual void removeRenderBuffer(IRenderBuffer* renderbuf) =0;

		virtual void removeFrameBuffer(IFrameBuffer* framebuf) =0;

		//! Removes all textures from the texture cache and deletes them.
		/** This method can free a lot of memory!
		Please note that after calling this, the pointer to the
		ITexture may no longer be valid, if it was not grabbed before
		by other parts of the engine for storing it longer. So it is a
		good idea to set all materials which are using this texture to
		0 or another texture first. */
		virtual void removeAllTextures() =0;

		virtual void removeAllRenderBuffers() =0;

		virtual void removeAllFrameBuffers() =0;


		//! Queries
		virtual void beginQuery(IQueryObject* query) = 0;
		virtual void endQuery(IQueryObject* query) = 0;
		virtual void beginQuery(IQueryObject* query, const size_t& index) = 0;
		virtual void endQuery(IQueryObject* query, const size_t& index) = 0;

        virtual IOcclusionQuery* createOcclusionQuery(const E_OCCLUSION_QUERY_TYPE& heuristic) = 0;

        virtual IQueryObject* createPrimitivesGeneratedQuery() = 0;
        virtual IQueryObject* createXFormFeedbackPrimitiveQuery() = 0;
        virtual IQueryObject* createElapsedTimeQuery() = 0;
        virtual IGPUTimestampQuery* createTimestampQuery() = 0;


		//! Creates a normal map from a height map texture.
		/** If the target texture has 32 bit, the height value is
		stored in the alpha component of the texture as addition. This
		value is used by the video::EMT_PARALLAX_MAP_SOLID material and
		similar materials.
		\param texture Texture whose alpha channel is modified.
		\param amplitude Constant value by which the height
		information is multiplied.*/
		virtual void makeNormalMapTexture(video::ITexture* texture, f32 amplitude=1.0f) const =0;


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
		virtual void beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const scene::E_PRIMITIVE_TYPE& primType=scene::EPT_POINTS) = 0;

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
		virtual void setViewPort(const core::rect<s32>& area) =0;

		//! Gets the area of the current viewport.
		/** \return Rectangle of the current viewport. */
		virtual const core::rect<s32>& getViewPort() const =0;

		//! Draws a 3d line.
		/** Note that the line is drawn using the current transformation
		matrix and material. So if you need to draw the 3D line
		independently of the current transformation, use
		\code
		driver->setMaterial(someMaterial);
		driver->setTransform(video::E4X3TS_WORLD, core::IdentityMatrix);
		\endcode
		for some properly set up material before drawing the line.
		Some drivers support line thickness set in the material.
		\param start Start of the 3d line.
		\param end End of the 3d line.
		\param color Color of the line. */
		virtual void draw3DLine(const core::vector3df& start,
			const core::vector3df& end, SColor color = SColor(255,255,255,255)) =0;

		//! Draws a 3d axis aligned box.
		/** This method simply calls draw3DLine for the edges of the
		box. Note that the box is drawn using the current transformation
		matrix and material. So if you need to draw it independently of
		the current transformation, use
		\code
		driver->setMaterial(someMaterial);
		driver->setTransform(video::E4X3TS_WORLD, core::IdentityMatrix);
		\endcode
		for some properly set up material before drawing the box.
		\param box The axis aligned box to draw
		\param color Color to use while drawing the box. */
		virtual void draw3DBox(const core::aabbox3d<f32>& box,
			SColor color = SColor(255,255,255,255)) =0;

		//! Draws a 2d image without any special effects
		/** \param texture Pointer to texture to use.
		\param destPos Upper left 2d destination position where the
		image will be drawn. */
		virtual void draw2DImage(const video::ITexture* texture,
			const core::position2d<s32>& destPos) =0;

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
		virtual void draw2DImage(const video::ITexture* texture, const core::position2d<s32>& destPos,
			const core::rect<s32>& sourceRect, const core::rect<s32>* clipRect =0,
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
				const core::position2d<s32>& pos,
				const core::array<core::rect<s32> >& sourceRects,
				const core::array<s32>& indices,
				s32 kerningWidth=0,
				const core::rect<s32>* clipRect=0,
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
				const core::array<core::position2d<s32> >& positions,
				const core::array<core::rect<s32> >& sourceRects,
				const core::rect<s32>* clipRect=0,
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
		virtual void draw2DImage(const video::ITexture* texture, const core::rect<s32>& destRect,
			const core::rect<s32>& sourceRect, const core::rect<s32>* clipRect =0,
			const video::SColor * const colors=0, bool useAlphaChannelOfTexture=false) =0;

		//! Draws a 2d rectangle.
		/** \param color Color of the rectangle to draw. The alpha
		component will not be ignored and specifies how transparent the
		rectangle will be.
		\param pos Position of the rectangle.
		\param clip Pointer to rectangle against which the rectangle
		will be clipped. If the pointer is null, no clipping will be
		performed. */
		virtual void draw2DRectangle(SColor color, const core::rect<s32>& pos,
			const core::rect<s32>* clip =0) =0;

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
		virtual void draw2DRectangle(const core::rect<s32>& pos,
				SColor colorLeftUp, SColor colorRightUp,
				SColor colorLeftDown, SColor colorRightDown,
				const core::rect<s32>* clip =0) =0;

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
		virtual void draw2DLine(const core::position2d<s32>& start,
					const core::position2d<s32>& end,
					SColor color=SColor(255,255,255,255)) =0;

		//! Draws a pixel.
		/** \param x The x-position of the pixel.
		\param y The y-position of the pixel.
		\param color Color of the pixel to draw. */
		virtual void drawPixel(u32 x, u32 y, const SColor& color) =0;

		//! Draws a non filled concyclic regular 2d polyon.
		/** This method can be used to draw circles, but also
		triangles, tetragons, pentagons, hexagons, heptagons, octagons,
		enneagons, decagons, hendecagons, dodecagon, triskaidecagons,
		etc. I think you'll got it now. And all this by simply
		specifying the vertex count. Welcome to the wonders of
		geometry.
		\param center Position of center of circle (pixels).
		\param radius Radius of circle in pixels.
		\param color Color of the circle.
		\param vertexCount Amount of vertices of the polygon. Specify 2
		to draw a line, 3 to draw a triangle, 4 for tetragons and a lot
		(>10) for nearly a circle. */
		virtual void draw2DPolygon(core::position2d<s32> center,
				f32 radius,
				video::SColor color=SColor(100,255,255,255),
				s32 vertexCount=10) =0;

		//! Draws a mesh buffer
		/** \param mb Buffer to draw */
		virtual void drawMeshBuffer(scene::IGPUMeshBuffer* mb, IOcclusionQuery* query = NULL) =0;

		//! Get the current color format of the color buffer
		/** \return Color format of the color buffer. */
		virtual ECOLOR_FORMAT getColorFormat() const =0;

		//! Get the size of the screen or render window.
		/** \return Size of screen or render window. */
		virtual const core::dimension2d<u32>& getScreenSize() const =0;

		//! Get the size of the current render target
		/** This method will return the screen size if the driver
		doesn't support render to texture, or if the current render
		target is the screen.
		\return Size of render target or screen/window */
		virtual const core::dimension2d<u32>& getCurrentRenderTargetSize() const =0;

		//! Returns current frames per second value.
		/** This value is updated approximately every 1.5 seconds and
		is only intended to provide a rough guide to the average frame
		rate. It is not suitable for use in performing timing
		calculations or framerate independent movement.
		\return Approximate amount of frames per second drawn. */
		virtual s32 getFPS() const =0;

		//! Returns amount of primitives (mostly triangles) which were drawn in the last frame.
		/** Together with getFPS() very useful method for statistics.
		\param mode Defines if the primitives drawn are accumulated or
		counted per frame.
		\return Amount of primitives drawn in the last frame. */
		virtual u32 getPrimitiveCountDrawn( u32 mode =0 ) const =0;

		//! Deletes all dynamic lights which were previously added with addDynamicLight().
		virtual void deleteAllDynamicLights() =0;

		//! adds a dynamic light, returning an index to the light
		//! \param light: the light data to use to create the light
		//! \return An index to the light, or -1 if an error occurs
		virtual s32 addDynamicLight(const SLight& light) =0;

		//! Returns the maximal amount of dynamic lights the device can handle
		/** \return Maximal amount of dynamic lights. */
		virtual u32 getMaximalDynamicLightAmount() const =0;

		//! Returns amount of dynamic lights currently set
		/** \return Amount of dynamic lights currently set */
		virtual u32 getDynamicLightCount() const =0;

		//! Returns light data which was previously set by IVideoDriver::addDynamicLight().
		/** \param idx Zero based index of the light. Must be 0 or
		greater and smaller than IVideoDriver::getDynamicLightCount.
		\return Light data. */
		virtual const SLight& getDynamicLight(u32 idx) const =0;

		//! Turns a dynamic light on or off
		//! \param lightIndex: the index returned by addDynamicLight
		//! \param turnOn: true to turn the light on, false to turn it off
		virtual void turnLightOn(s32 lightIndex, bool turnOn) =0;

		//! Gets name of this video driver.
		/** \return Returns the name of the video driver, e.g. in case
		of the Direct3D8 driver, it would return "Direct3D 8.1". */
		virtual const wchar_t* getName() const =0;

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
		virtual u32 getMaximalIndicesCount() const =0;

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

		//! Creates a software image from a file.
		/** No hardware texture will be created for this image. This
		method is useful for example if you want to read a heightmap
		for a terrain renderer.
		\param filename Name of the file from which the image is
		created.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		virtual IImage* createImageFromFile(const io::path& filename) = 0;

		//! Creates a software image from a file.
		/** No hardware texture will be created for this image. This
		method is useful for example if you want to read a heightmap
		for a terrain renderer.
		\param file File from which the image is created.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		virtual IImage* createImageFromFile(io::IReadFile* file) =0;

		//! Writes the provided image to a file.
		/** Requires that there is a suitable image writer registered
		for writing the image.
		\param image Image to write.
		\param filename Name of the file to write.
		\param param Control parameter for the backend (e.g. compression
		level).
		\return True on successful write. */
		virtual bool writeImageToFile(IImage* image, const io::path& filename, u32 param = 0) = 0;

		//! Writes the provided image to a file.
		/** Requires that there is a suitable image writer registered
		for writing the image.
		\param image Image to write.
		\param file  An already open io::IWriteFile object. The name
		will be used to determine the appropriate image writer to use.
		\param param Control parameter for the backend (e.g. compression
		level).
		\return True on successful write. */
		virtual bool writeImageToFile(IImage* image, io::IWriteFile* file, u32 param =0) =0;

		//! Creates a software image from a byte array.
		/** No hardware texture will be created for this image. This
		method is useful for example if you want to read a heightmap
		for a terrain renderer.
		\param format Desired color format of the texture
		\param size Desired size of the image
		\param data A byte array with pixel color information
		\param ownForeignMemory If true, the image will use the data
		pointer directly and own it afterwards. If false, the memory
		will by copied internally.
		\param deleteMemory Whether the memory is deallocated upon
		destruction.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		virtual IImage* createImageFromData(ECOLOR_FORMAT format,
			const core::dimension2d<u32>& size, void *data,
			bool ownForeignMemory=false,
			bool deleteMemory = true) =0;

		//! Creates an empty software image.
		/**
		\param format Desired color format of the image.
		\param size Size of the image to create.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		virtual IImage* createImage(ECOLOR_FORMAT format, const core::dimension2d<u32>& size) =0;

		//! Creates a software image by converting it to given format from another image.
		/** \deprecated Create an empty image and use copyTo(). This method may be removed by Irrlicht 1.9.
		\param format Desired color format of the image.
		\param imageToCopy Image to copy to the new image.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		_IRR_DEPRECATED_ virtual IImage* createImage(ECOLOR_FORMAT format, IImage *imageToCopy) =0;

		//! Creates a software image from a part of another image.
		/** \deprecated Create an empty image and use copyTo(). This method may be removed by Irrlicht 1.9.
		\param imageToCopy Image to copy to the new image in part.
		\param pos Position of rectangle to copy.
		\param size Extents of rectangle to copy.
		\return The created image.
		If you no longer need the image, you should call IImage::drop().
		See IReferenceCounted::drop() for more information. */
		_IRR_DEPRECATED_ virtual IImage* createImage(IImage* imageToCopy,
				const core::position2d<s32>& pos,
				const core::dimension2d<u32>& size) =0;

		//! Event handler for resize events. Only used by the engine internally.
		/** Used to notify the driver that the window was resized.
		Usually, there is no need to call this method. */
		virtual void OnResize(const core::dimension2d<u32>& size) =0;

		//! Adds a new material renderer to the video device.
		/** Use this method to extend the VideoDriver with new material
		types. To extend the engine using this method do the following:
		Derive a class from IMaterialRenderer and override the methods
		you need. For setting the right renderstates, you can try to
		get a pointer to the real rendering device using
		IVideoDriver::getExposedVideoData(). Add your class with
		IVideoDriver::addMaterialRenderer(). To use an object being
		displayed with your new material, set the MaterialType member of
		the SMaterial struct to the value returned by this method.
		If you simply want to create a new material using vertex and/or
		pixel shaders it would be easier to use the
		video::IGPUProgrammingServices interface which you can get
		using the getGPUProgrammingServices() method.
		\param renderer A pointer to the new renderer.
		\param name Optional name for the material renderer entry.
		\return The number of the material type which can be set in
		SMaterial::MaterialType to use the renderer. -1 is returned if
		an error occured. For example if you tried to add an material
		renderer to the software renderer or the null device, which do
		not accept material renderers. */
		virtual s32 addMaterialRenderer(IMaterialRenderer* renderer, const c8* name =0) =0;

		//! Get access to a material renderer by index.
		/** \param idx Id of the material renderer. Can be a value of
		the E_MATERIAL_TYPE enum or a value which was returned by
		addMaterialRenderer().
		\return Pointer to material renderer or null if not existing. */
		virtual IMaterialRenderer* getMaterialRenderer(u32 idx) =0;

		//! Get amount of currently available material renderers.
		/** \return Amount of currently available material renderers. */
		virtual u32 getMaterialRendererCount() const =0;

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
		virtual const c8* getMaterialRendererName(u32 idx) const =0;

		//! Sets the name of a material renderer.
		/** Will have no effect on built-in material renderers.
		\param idx: Id of the material renderer. Can be a value of the
		E_MATERIAL_TYPE enum or a value which was returned by
		addMaterialRenderer().
		\param name: New name of the material renderer. */
		virtual void setMaterialRendererName(s32 idx, const c8* name) =0;

		//! Returns driver and operating system specific data about the IVideoDriver.
		/** This method should only be used if the engine should be
		extended without having to modify the source of the engine.
		\return Collection of device dependent pointers. */
		virtual const SExposedVideoData& getExposedVideoData() =0;

		//! Get type of video driver
		/** \return Type of driver. */
		virtual E_DRIVER_TYPE getDriverType() const =0;

		//! Gets the IGPUProgrammingServices interface.
		/** \return Pointer to the IGPUProgrammingServices. Returns 0
		if the video driver does not support this. For example the
		Software driver and the Null driver will always return 0. */
		virtual IGPUProgrammingServices* getGPUProgrammingServices() =0;

		//! Check if the image is already loaded.
		/** Works similar to getTexture(), but does not load the texture
		if it is not currently loaded.
		\param filename Name of the texture.
		\return Pointer to loaded texture, or 0 if not found. */
		virtual video::ITexture* findTexture(const io::path& filename) = 0;

		//! Enable or disable a clipping plane.
		/** There are at least 6 clipping planes available for the user
		to set at will.
		\param index The plane index. Must be between 0 and
		MaxUserClipPlanes.
		\param enable If true, enable the clipping plane else disable
		it. */
		virtual void enableClipPlane(u32 index, bool enable) =0;

		//! Set the minimum number of vertices for which a hw buffer will be created
		/** \param count Number of vertices to set as minimum. */
		virtual void setMinHardwareBufferVertexCount(u32 count) =0;

		//! Get the global Material, which might override local materials.
		/** Depending on the enable flags, values from this Material
		are used to override those of local materials of some
		meshbuffer being rendered.
		\return Reference to the Override Material. */
		virtual SOverrideMaterial& getOverrideMaterial() =0;

		//! Get the 2d override material for altering its values
		/** The 2d override materual allows to alter certain render
		states of the 2d methods. Not all members of SMaterial are
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
		virtual SMaterial& getMaterial2D() =0;

		//! Enable the 2d override material
		/** \param enable Flag which tells whether the material shall be
		enabled or disabled. */
		virtual void enableMaterial2D(bool enable=true) =0;

		//! Get the graphics card vendor name.
		virtual core::stringc getVendorInfo() =0;

		//! Only used by the engine internally.
		/** Passes the global material flag AllowZWriteOnTransparent.
		Use the SceneManager attribute to set this value from your app.
		\param flag Default behavior is to disable ZWrite, i.e. false. */
		virtual void setAllowZWriteOnTransparent(bool flag) =0;

		//! Get the maximum texture size supported.
		virtual const uint32_t* getMaxTextureSize(const ITexture::E_TEXTURE_TYPE& type) const =0;

		//! Color conversion convenience function
		/** Convert an image (as array of pixels) from source to destination
		array, thereby converting the color format. The pixel size is
		determined by the color formats.
		\param sP Pointer to source
		\param sF Color format of source
		\param sN Number of pixels to convert, both array must be large enough
		\param dP Pointer to destination
		\param dF Color format of destination
		*/
		virtual void convertColor(const void* sP, ECOLOR_FORMAT sF, s32 sN,
				void* dP, ECOLOR_FORMAT dF) const =0;
	};

} // end namespace video
} // end namespace irr


#endif

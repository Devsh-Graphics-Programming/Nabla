// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_VIDEO_2_SOFTWARE_H_INCLUDED__
#define __C_VIDEO_2_SOFTWARE_H_INCLUDED__

#include "SoftwareDriver2_compile_config.h"
#include "IBurningShader.h"
#include "CNullDriver.h"
#include "CImage.h"
#include "os.h"
#include "irr/core/irrString.h"
#include "SIrrCreationParameters.h"

namespace irr
{
namespace video
{
	class CBurningVideoDriver : public CNullDriver
	{
    protected:
		//! destructor
		virtual ~CBurningVideoDriver();

	public:
		//! constructor
		CBurningVideoDriver(IrrlichtDevice* dev, const irr::SIrrlichtCreationParameters& params, io::IFileSystem* io, video::IImagePresenter* presenter);

        inline virtual bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline virtual bool isColorRenderableFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline virtual bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const override { return false; }
        inline virtual bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const override { return false; }

		//! sets a material
		virtual void setMaterial(const SGPUMaterial& material);

		virtual bool setRenderTarget(video::ITexture* texture, bool clearBackBuffer,
						bool clearZBuffer, SColor color);

		//! sets a viewport
		virtual void setViewPort(const core::rect<int32_t>& area);

		//! clears the zbuffer
		virtual bool beginScene(bool backBuffer=true, bool zBuffer=true,
				SColor color=SColor(255,0,0,0),
				const SExposedVideoData& videoData=SExposedVideoData(),
				core::rect<int32_t>* sourceRect=0);

		//! presents the rendered scene on the screen, returns false if failed
		virtual bool endScene();

		//! Only used by the internal engine. Used to notify the driver that
		//! the window was resized.
		virtual void OnResize(const core::dimension2d<uint32_t>& size);

		//! returns size of the current render target
		virtual const core::dimension2d<uint32_t>& getCurrentRenderTargetSize() const;


		//! draws an 2d image, using a color (if color is other then Color(255,255,255,255)) and the alpha channel of the texture if wanted.
		virtual void draw2DImage(const video::ITexture* texture, const core::position2d<int32_t>& destPos,
			const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect = 0,
			SColor color=SColor(255,255,255,255), bool useAlphaChannelOfTexture=false);

	//! Draws a part of the texture into the rectangle.
		virtual void draw2DImage(const video::ITexture* texture, const core::rect<int32_t>& destRect,
				const core::rect<int32_t>& sourceRect, const core::rect<int32_t>* clipRect = 0,
				const video::SColor* const colors=0, bool useAlphaChannelOfTexture=false);

		//! draw an 2d rectangle
		virtual void draw2DRectangle(SColor color, const core::rect<int32_t>& pos,
			const core::rect<int32_t>* clip = 0);

		//!Draws an 2d rectangle with a gradient.
		virtual void draw2DRectangle(const core::rect<int32_t>& pos,
			SColor colorLeftUp, SColor colorRightUp, SColor colorLeftDown, SColor colorRightDown,
			const core::rect<int32_t>* clip = 0);

		//! Draws a 2d line.
		virtual void draw2DLine(const core::position2d<int32_t>& start,
					const core::position2d<int32_t>& end,
					SColor color=SColor(255,255,255,255));

		//! Draws a single pixel
		virtual void drawPixel(uint32_t x, uint32_t y, const SColor & color);

		//! \return Returns the name of the video driver. Example: In case of the DirectX8
		//! driver, it would return "Direct3D8.1".
		virtual const wchar_t* getName() const;

		//! Returns type of video driver
		virtual E_DRIVER_TYPE getDriverType() const;

		//! get color format of the current color buffer
		virtual asset::E_FORMAT getColorFormat() const;

		//! Clears the DepthBuffer.
		virtual void clearZBuffer();

		//! Returns the maximum amount of primitives (mostly vertices) which
		//! the device is able to render with one drawIndexedTriangleList
		//! call.
		virtual uint32_t getMaximalIndicesCount() const;

		//! Returns the graphics card vendor name.
		virtual std::string getVendorInfo();

		virtual IDepthBuffer * getDepthBuffer () { return DepthBuffer; }
		virtual IStencilBuffer * getStencilBuffer () { return StencilBuffer; }

	protected:
		//! sets a render target
		void setRenderTarget(video::CImage* image);

		//! sets the current Texture
		//bool setTexture(uint32_t stage, video::ITexture* texture);

		video::CImage* BackBuffer;
		video::IImagePresenter* Presenter;

		void* WindowId;
		core::rect<int32_t>* SceneSourceRect;

		video::ITexture* RenderTargetTexture;
		video::IImage* RenderTargetSurface;
		core::dimension2d<uint32_t> RenderTargetSize;

		//! selects the right triangle renderer based on the render states.
		void setCurrentShader();

		IBurningShader* CurrentShader;
		IBurningShader* BurningShader[ETR2_COUNT];

		IDepthBuffer* DepthBuffer;
		IStencilBuffer* StencilBuffer;


		/*
			extend Matrix Stack
			-> combined CameraProjection
			-> combined CameraProjectionWorld
			-> ClipScale from NDC to DC Space
		*
		enum E_TRANSFORMATION_STATE_BURNING_VIDEO
		{
			E4X3TS_VIEW_PROJECTION = ETS_COUNT,
			ETS_CURRENT,
			ETS_CLIPSCALE,
			E4X3TS_VIEW_INVERSE,
			E4X3TS_WORLD_INVERSE,

			ETS_COUNT_BURNING
		};*/

		enum E_TRANSFORMATION_FLAG
		{
			ETF_IDENTITY = 1,
			ETF_TEXGEN_CAMERA_NORMAL = 2,
			ETF_TEXGEN_CAMERA_REFLECTION = 4,
		};

		void getCameraPosWorldSpace ();

		core::matrix4SIMD ClipscaleTransformation;


		// Vertex Cache
#ifndef NEW_MESHES
		static const SVSize vSize[];

		SVertexCache VertexCache;

		void VertexCache_reset (const void* vertices, uint32_t vertexCount,
					const void* indices, uint32_t indexCount,
					E_VERTEX_TYPE vType,scene::E_PRIMITIVE_TYPE pType,
					scene::E_INDEX_TYPE iType);
		void VertexCache_get ( const s4DVertex ** face );
		void VertexCache_getbypass ( s4DVertex ** face );

		void VertexCache_fill ( const uint32_t sourceIndex,const uint32_t destIndex );
		s4DVertex * VertexCache_getVertex ( const uint32_t sourceIndex );
#endif // NEW_MESHES

		// culling & clipping
		uint32_t clipToHyperPlane ( s4DVertex * dest, const s4DVertex * source, uint32_t inCount, const sVec4 &plane );
		uint32_t clipToFrustumTest ( const s4DVertex * v  ) const;
		uint32_t clipToFrustum ( s4DVertex *source, s4DVertex * temp, const uint32_t vIn );



		// holds transformed, clipped vertices
		SAlignedVertex CurrentOut;
		SAlignedVertex Temp;

		void ndc_2_dc_and_project ( s4DVertex *dest,s4DVertex *source, uint32_t vIn ) const;
		float screenarea ( const s4DVertex *v0 ) const;
		void select_polygon_mipmap ( s4DVertex *source, uint32_t vIn, uint32_t tex, const core::dimension2du& texSize ) const;
		float texelarea ( const s4DVertex *v0, int tex ) const;


		void ndc_2_dc_and_project2 ( const s4DVertex **v, const uint32_t size ) const;
		float screenarea2 ( const s4DVertex **v ) const;
		float texelarea2 ( const s4DVertex **v, int tex ) const;
		void select_polygon_mipmap2 ( s4DVertex **source, uint32_t tex, const core::dimension2du& texSize ) const;


		SBurningShaderLightSpace LightSpace;
		SBurningShaderMaterial Material;

		static const sVec4 NDCPlane[6];
	};

} // end namespace video
} // end namespace irr


#endif


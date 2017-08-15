// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_TRIANGLE_RENDERER_TEXTURE_GOURAUD_H_INCLUDED__
#define __C_TRIANGLE_RENDERER_TEXTURE_GOURAUD_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifndef _IRR_COMPILE_WITH_SOFTWARE_
// forward declarations for create methods
namespace irr
{
namespace video
{
	class ITriangleRenderer;
	class IZBuffer;
} // end namespace video
} // end namespace irr

#else

#include "ITriangleRenderer.h"
#include "IImage.h"

namespace irr
{
namespace video
{
	//! CTRTextureGouraud class
	class CTRTextureGouraud : public ITriangleRenderer
	{
	public:

		//! constructor
		CTRTextureGouraud(IZBuffer* zbuffer);

		//! destructor
		virtual ~CTRTextureGouraud();

		//! sets a render target
		virtual void setRenderTarget(video::IImage* surface, const core::rect<int32_t>& viewPort);

		//! draws an indexed triangle list
		virtual void drawIndexedTriangleList(S2DVertex* vertices, int32_t vertexCount, const uint16_t* indexList, int32_t triangleCount);

		//! en or disables the backface culling
		virtual void setBackfaceCulling(bool enabled = true);

		//! sets the Texture
		virtual void setTexture(video::IImage* texture);

	protected:

		//! vertauscht zwei vertizen
		inline void swapVertices(const S2DVertex** v1, const S2DVertex** v2)
		{
			const S2DVertex* b = *v1;
			*v1 = *v2;
			*v2 = b;
		}

		video::IImage* RenderTarget;
		core::rect<int32_t> ViewPortRect;

		IZBuffer* ZBuffer;

		int32_t SurfaceWidth;
		int32_t SurfaceHeight;
		bool BackFaceCullingEnabled;
		TZBufferType* lockedZBuffer;
		uint16_t* lockedSurface;
		uint16_t* lockedTexture;
		int32_t lockedTextureWidth;
		int32_t textureXMask, textureYMask;
		video::IImage* Texture;
	};

} // end namespace video
} // end namespace irr

#endif

#endif


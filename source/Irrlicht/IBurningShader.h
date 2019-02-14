// Copyright (C) 2002-2012 Nikolaus Gebhardt / Thomas Alten
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_BURNING_SHADER_H_INCLUDED__
#define __I_BURNING_SHADER_H_INCLUDED__

#include "SoftwareDriver2_compile_config.h"
#include "irr/core/IReferenceCounted.h"
#include "irr/core/math/irrMath.h"
#include "IImage.h"
#include "S2DVertex.h"
#include "rect.h"
#include "CDepthBuffer.h"
#include "S4DVertex.h"
#include "SMaterial.h"
#include "os.h"


namespace irr
{

namespace video
{


	enum eLightFlags
	{
		ENABLED		= 0x01,
		POINTLIGHT	= 0x02,
		SPECULAR	= 0x04,
		FOG			= 0x08,
		NORMALIZE	= 0x10,
		VERTEXTRANSFORM	= 0x20,
	};

	struct SBurningShaderLightSpace
	{
		void reset ()
		{
			Global_AmbientLight.set ( 0.f, 0.f, 0.f );
			Flags = 0;
		}
		sVec3 Global_AmbientLight;
		sVec4 FogColor;
		sVec4 campos;
		sVec4 vertex;
		sVec4 normal;
		uint32_t Flags;
	};

	struct SBurningShaderMaterial
	{
		SGPUMaterial org;

		sVec3 AmbientColor;
		sVec3 DiffuseColor;
		sVec3 SpecularColor;
		sVec3 EmissiveColor;

	};

	enum EBurningFFShader
	{
		ETR_GOURAUD=0,
		ETR_TEXTURE_GOURAUD,
		ETR_TEXTURE_GOURAUD_NOZ,
		ETR_TEXTURE_GOURAUD_ADD,
		ETR_TEXTURE_GOURAUD_ADD_NO_Z,

		ETR_TEXTURE_GOURAUD_VERTEX_ALPHA,

		ETR_TEXTURE_GOURAUD_ALPHA,
		ETR_TEXTURE_GOURAUD_ALPHA_NOZ,

		ETR_REFERENCE,
		ETR_INVALID,

		ETR2_COUNT
	};


	class CBurningVideoDriver;
	class IBurningShader : public virtual core::IReferenceCounted
	{
	public:
		IBurningShader(CBurningVideoDriver* driver);

		//! sets a render target
		virtual void setRenderTarget(video::IImage* surface, const core::rect<int32_t>& viewPort);

		//! sets the Texture
		virtual void setTextureParam( uint32_t stage, video::CSoftwareTexture2* texture, int32_t lodLevel);
		virtual void drawTriangle ( const s4DVertex *a,const s4DVertex *b,const s4DVertex *c ) = 0;
		virtual void drawLine ( const s4DVertex *a,const s4DVertex *b) {};

		virtual void setParam ( uint32_t index, float value) {};
		virtual void setZCompareFunc ( uint32_t func) {};

		virtual void setMaterial ( const SBurningShaderMaterial &material ) {};

	protected:
		//! destructor
		virtual ~IBurningShader();

		CBurningVideoDriver *Driver;

		video::CImage* RenderTarget;
		CDepthBuffer* DepthBuffer;
		CStencilBuffer * Stencil;
		tVideoSample ColorMask;

		sInternalTexture IT[ BURNING_MATERIAL_MAX_TEXTURES ];

		static const tFixPointu dithermask[ 4 * 4];
	};


	IBurningShader* createTriangleRendererTextureGouraud2(CBurningVideoDriver* driver);
	IBurningShader* createTriangleRendererTextureVertexAlpha2(CBurningVideoDriver* driver);

	IBurningShader* createTriangleRendererGouraud2(CBurningVideoDriver* driver);
	IBurningShader* createTRTextureGouraudNoZ2(CBurningVideoDriver* driver);
	IBurningShader* createTRTextureGouraudAdd2(CBurningVideoDriver* driver);
	IBurningShader* createTRTextureGouraudAddNoZ2(CBurningVideoDriver* driver);

	IBurningShader* createTRTextureGouraudAlpha(CBurningVideoDriver* driver);
	IBurningShader* createTRTextureGouraudAlphaNoZ(CBurningVideoDriver* driver);

	IBurningShader* createTriangleRendererReference(CBurningVideoDriver* driver);



} // end namespace video
} // end namespace irr

#endif


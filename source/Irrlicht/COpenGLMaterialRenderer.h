// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPENGL_MATERIAL_RENDERER_H_INCLUDED__
#define __C_OPENGL_MATERIAL_RENDERER_H_INCLUDED__

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGLDriver.h"
#include "IMaterialRenderer.h"

namespace irr
{
namespace video
{

extern const core::stringc std_vert;
extern const core::stringc std_solid_frag;
extern const core::stringc std_trans_add_frag;
extern const core::stringc std_trans_alpha_frag;
extern const core::stringc std_trans_vertex_frag;


//! Base class for all internal OpenGL material renderers
class COpenGLMaterialRenderer : public IMaterialRenderer
{
public:

	//! Constructor
	COpenGLMaterialRenderer(video::COpenGLDriver* driver) : Driver(driver)
	{
	}

protected:

	video::COpenGLDriver* Driver;
};


//! Solid material renderer
class COpenGLMaterialRenderer_SOLID : public COpenGLMaterialRenderer
{
public:

	COpenGLMaterialRenderer_SOLID(video::COpenGLDriver* d)
		: COpenGLMaterialRenderer(d) {}

	virtual void OnSetMaterial(const SGPUMaterial& material, const SGPUMaterial& lastMaterial,
		bool resetAllRenderstates, IMaterialRendererServices* services)
	{
		Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);
	}
};



//! Transparent add color material renderer
class COpenGLMaterialRenderer_TRANSPARENT_ADD_COLOR : public COpenGLMaterialRenderer
{
public:

	COpenGLMaterialRenderer_TRANSPARENT_ADD_COLOR(video::COpenGLDriver* d)
		: COpenGLMaterialRenderer(d) {}

	virtual void OnSetMaterial(const SGPUMaterial& material, const SGPUMaterial& lastMaterial,
		bool resetAllRenderstates, IMaterialRendererServices* services)
	{
		Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);

		if ((material.MaterialType != lastMaterial.MaterialType) || resetAllRenderstates)
		{
			glBlendFunc(GL_ONE, GL_ONE);
			glEnable(GL_BLEND);
		}
	}

	virtual void OnUnsetMaterial()
	{
		glDisable(GL_BLEND);
	}

	//! Returns if the material is transparent.
	virtual bool isTransparent() const
	{
		return true;
	}
};


//! Transparent alpha channel material renderer
class COpenGLMaterialRenderer_TRANSPARENT_ALPHA_CHANNEL : public COpenGLMaterialRenderer
{
public:

	COpenGLMaterialRenderer_TRANSPARENT_ALPHA_CHANNEL(video::COpenGLDriver* d)
		: COpenGLMaterialRenderer(d) {}

	virtual void OnSetMaterial(const SGPUMaterial& material, const SGPUMaterial& lastMaterial,
		bool resetAllRenderstates, IMaterialRendererServices* services)
	{
		Driver->setBasicRenderStates(material, lastMaterial, resetAllRenderstates);

		if (material.MaterialType != lastMaterial.MaterialType || resetAllRenderstates)
		{
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_BLEND);
		}
	}

	virtual void OnUnsetMaterial()
	{
		glDisable(GL_BLEND);
	}

	//! Returns if the material is transparent.
	virtual bool isTransparent() const
	{
		return true;
	}
};


//! Transparent vertex alpha material renderer
class COpenGLMaterialRenderer_TRANSPARENT_VERTEX_ALPHA : public COpenGLMaterialRenderer_TRANSPARENT_ALPHA_CHANNEL
{
public:

	COpenGLMaterialRenderer_TRANSPARENT_VERTEX_ALPHA(video::COpenGLDriver* d)
		: COpenGLMaterialRenderer_TRANSPARENT_ALPHA_CHANNEL(d) {}
};

} // end namespace video
} // end namespace irr

#endif
#endif


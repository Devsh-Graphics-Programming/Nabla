// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_MATERIAL_RENDERER_H_INCLUDED__
#define __I_MATERIAL_RENDERER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "SMaterial.h"

namespace irr
{
namespace video
{

class IVideoDriver;
class IMaterialRendererServices;

//! Interface for material rendering.
/** Can be used to extend the engine with new materials. Refer to
IVideoDriver::addMaterialRenderer() for more informations on how to extend the
engine with new materials. */
class IMaterialRenderer : public virtual core::IReferenceCounted
{
public:

	//! Called by the IVideoDriver implementation the let the renderer set its needed render states.
	/** This is called during the IVideoDriver::setMaterial() call.
	When overriding this, you can set some renderstates or for example a
	vertex or pixel shader if you like.
	\param material: The new material parameters to be set. The renderer
	may change the material flags in this material. For example if this
	material does not accept the zbuffer = true, it can set it to false.
	This is useful, because in the next lastMaterial will be just the
	material in this call.
	\param lastMaterial: The material parameters which have been set before
	this material.
	\param resetAllRenderstates: True if all renderstates should really be
	reset. This is usually true if the last rendering mode was not a usual
	3d rendering mode, but for example a 2d rendering mode.
	You should reset really all renderstates if this is true, no matter if
	the lastMaterial had some similar settings. This is used because in
	most cases, some common renderstates are not changed if they are
	already there, for example bilinear filtering, wireframe,
	gouraudshading, lighting, zbuffer, zwriteenable, backfaceculling and
	fogenable.
	\param services: Interface providing some methods for changing
	advanced, internal states of a IVideoDriver. */
	virtual void OnSetMaterial(const SGPUMaterial& material, const SGPUMaterial& lastMaterial,
		bool resetAllRenderstates, IMaterialRendererServices* services) {}

	//! Called every time before a new bunch of geometry is being drawn using this material with for example drawIndexedTriangleList() call.
	/** OnSetMaterial should normally only be called if the renderer decides
	that the renderstates should be changed, it won't be called if for
	example two drawIndexedTriangleList() will be called with the same
	material set. This method will be called every time. This is useful for
	example for materials with shaders, which don't only set new
	renderstates but also shader constants.
	\param service: Pointer to interface providing methos for setting
	constants and other things.
	\param vtxtype: Vertex type with which the next rendering will be done.
	This can be used by the material renderer to set some specific
	optimized shaders or if this is an incompatible vertex type for this
	renderer, to refuse rendering for example.
	\return Returns true if everything is ok, and false if nothing should
	be rendered. The material renderer can choose to return false for
	example if he doesn't support the specified vertex type. This is
	actually done in D3D8 and D3D9 when using a normal mapped material with
	a vertex type other than EVT_TANGENTS. */
	virtual bool OnRender(IMaterialRendererServices* service) { return true; }

	//! Called by the IVideoDriver to unset this material.
	/** Called during the IVideoDriver::setMaterial() call before the new
	material will get the OnSetMaterial() call. */
	virtual void OnUnsetMaterial() {}



	virtual bool isTransparent() const { return false; } //depr
};


} // end namespace video
} // end namespace irr

#endif


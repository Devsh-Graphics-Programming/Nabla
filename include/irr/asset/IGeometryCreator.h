// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_GEOMETRY_CREATOR_H_INCLUDED__
#define __I_GEOMETRY_CREATOR_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/IMesh.h"
#include "irr/asset/ICPUMesh.h"
#include "irr/video/IGPUMesh.h"

namespace irr
{
namespace video
{
	class IVideoDriver;
}

namespace asset
{

//! Helper class for creating geometry on the fly.
/** You can get an instance of this class through ISceneManager::getGeometryCreator() */
class IGeometryCreator : public core::IReferenceCounted
{
    _IRR_INTERFACE_CHILD(IGeometryCreator) {}
public:

	//! Creates a simple cube mesh.
	/**
	\param size Dimensions of the cube.
	\return Generated mesh.
	*/
	virtual asset::ICPUMesh* createCubeMesh(const core::vector3df& size=core::vector3df(5.f,5.f,5.f)) const =0;


	//! Create an arrow mesh, composed of a cylinder and a cone.
	/**
	\param tesselationCylinder Number of quads composing the cylinder.
	\param tesselationCone Number of triangles composing the cone's roof.
	\param height Total height of the arrow
	\param cylinderHeight Total height of the cylinder, should be lesser
	than total height
	\param widthCylinder Diameter of the cylinder
	\param widthCone Diameter of the cone's base, should be not smaller
	than the cylinder's diameter
	\param colorCylinder color of the cylinder
	\param colorCone color of the cone
	\return Generated mesh.
	*/
	virtual asset::ICPUMesh* createArrowMesh(const uint32_t tesselationCylinder = 4,
			const uint32_t tesselationCone = 8, const float height = 1.f,
			const float cylinderHeight = 0.6f, const float widthCylinder = 0.05f,
			const float widthCone = 0.3f, const video::SColor colorCylinder = 0xFFFFFFFF,
			const video::SColor colorCone = 0xFFFFFFFF) const =0;


	//! Create a sphere mesh.
	/**
	\param radius Radius of the sphere
	\param polyCountX Number of quads used for the horizontal tiling
	\param polyCountY Number of quads used for the vertical tiling
	\return Generated mesh.
	*/
	virtual asset::ICPUMesh* createSphereMesh(float radius = 5.f,
			uint32_t polyCountX = 16, uint32_t polyCountY = 16) const =0;

	//! Create a cylinder mesh.
	/**
	\param radius Radius of the cylinder.
	\param length Length of the cylinder.
	\param tesselation Number of quads around the circumference of the cylinder.
	\param color The color of the cylinder.
	\param closeTop If true, close the ends of the cylinder, otherwise leave them open.
	\param oblique (to be documented)
	\return Generated mesh.
	*/
	virtual asset::ICPUMesh* createCylinderMesh(float radius, float length,
			uint32_t tesselation,
			const video::SColor& color=video::SColor(0xffffffff),
			bool closeTop=true, float oblique=0.f) const =0;

	//! Create a cone mesh.
	/**
	\param radius Radius of the cone.
	\param length Length of the cone.
	\param tesselation Number of quads around the circumference of the cone.
	\param colorTop The color of the top of the cone.
	\param colorBottom The color of the bottom of the cone.
	\param oblique (to be documented)
	\return Generated mesh.
	*/
	virtual asset::ICPUMesh* createConeMesh(float radius, float length, uint32_t tesselation,
			const video::SColor& colorTop=video::SColor(0xffffffff),
			const video::SColor& colorBottom=video::SColor(0xffffffff),
			float oblique=0.f) const =0;

};

} // end namespace asset
} // end namespace irr

#endif // __I_GEOMETRY_CREATOR_H_INCLUDED__


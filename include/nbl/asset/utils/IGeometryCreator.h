// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_GEOMETRY_CREATOR_H_INCLUDED__
#define __NBL_ASSET_I_GEOMETRY_CREATOR_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUMesh.h"
#include "nbl/asset/utils/IMeshManipulator.h"

#include "SColor.h"

namespace nbl
{
namespace asset
{

//! Helper class for creating geometry on the fly.
/** You can get an instance of this class through ISceneManager::getGeometryCreator() */
class IGeometryCreator : public core::IReferenceCounted
{
		_NBL_INTERFACE_CHILD(IGeometryCreator) {}
	public:
		struct return_type
		{
			SVertexInputParams inputParams;
			SPrimitiveAssemblyParams assemblyParams;
			SBufferBinding<ICPUBuffer> bindings[ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
			SBufferBinding<ICPUBuffer> indexBuffer;
			E_INDEX_TYPE indexType;
			uint32_t indexCount;
			core::aabbox3df bbox;
		};

		//! Creates a simple cube mesh.
		/**
		\param size Dimensions of the cube.
		\return Generated mesh.
		*/
		virtual return_type createCubeMesh(const core::vector3df& size=core::vector3df(5.f,5.f,5.f)) const =0;


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
		virtual return_type createArrowMesh(const uint32_t tesselationCylinder = 4,
				const uint32_t tesselationCone = 8, const float height = 1.f,
				const float cylinderHeight = 0.6f, const float widthCylinder = 0.05f,
				const float widthCone = 0.3f, const video::SColor colorCylinder = 0xFFFFFFFF,
				const video::SColor colorCone = 0xFFFFFFFF, IMeshManipulator* const meshManipulatorOverride = nullptr) const =0;


		//! Create a sphere mesh.
		/**
		\param radius Radius of the sphere
		\param polyCountX Number of quads used for the horizontal tiling
		\param polyCountY Number of quads used for the vertical tiling
		\return Generated mesh.
		*/
		virtual return_type createSphereMesh(float radius = 5.f,
				uint32_t polyCountX = 16, uint32_t polyCountY = 16, IMeshManipulator* const meshManipulatorOverride = nullptr) const =0;

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
		virtual return_type createCylinderMesh(float radius, float length,
				uint32_t tesselation,
				const video::SColor& color=video::SColor(0xffffffff), IMeshManipulator* const meshManipulatorOverride = nullptr) const =0;

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
		virtual return_type createConeMesh(float radius, float length, uint32_t tesselation,
				const video::SColor& colorTop=video::SColor(0xffffffff),
				const video::SColor& colorBottom=video::SColor(0xffffffff),
				float oblique=0.f, IMeshManipulator* const meshManipulatorOverride = nullptr) const =0;

		virtual return_type createRectangleMesh(const core::vector2df_SIMD& size = core::vector2df_SIMD(0.5f, 0.5f)) const = 0;

		virtual return_type createDiskMesh(float radius, uint32_t tesselation) const = 0;

		//! Create a icosphere geometry
		/**
			\param radius Radius of the icosphere.
			\param subdivision Specifies subdivision level of the icosphere.
			\param smooth Specifies whether vertecies should be built for smooth or flat shading.
		*/

		virtual return_type createIcoSphere(float radius = 1.0f, uint32_t subdivision = 1, bool smooth = false) const = 0;

};

} // end namespace asset
} // end namespace nbl

#endif


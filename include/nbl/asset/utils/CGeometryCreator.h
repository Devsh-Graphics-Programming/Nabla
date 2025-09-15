// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_GEOMETRY_CREATOR_H_INCLUDED_
#define _NBL_ASSET_C_GEOMETRY_CREATOR_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
// legacy, needs to be removed
#include "SColor.h"

#include "nbl/asset/ICPUGeometryCollection.h"


namespace nbl::asset
{

//! Helper class for creating geometry on the fly.
/** You can get an instance of this class through ISceneManager::getGeometryCreator() */
class NBL_API2 CGeometryCreator final : public core::IReferenceCounted
{
	public:
		struct SCreationParams
		{
			core::smart_refctd_ptr<CQuantNormalCache> normalCache = nullptr;
			core::smart_refctd_ptr<CQuantQuaternionCache> quaternionCache = nullptr;
		};
		inline CGeometryCreator(SCreationParams&& params={}) : m_params(std::move(params))
		{
			if (!m_params.normalCache)
				m_params.normalCache = core::make_smart_refctd_ptr<CQuantNormalCache>();
			if (!m_params.quaternionCache)
				m_params.quaternionCache = core::make_smart_refctd_ptr<CQuantQuaternionCache>();
		}

		//
		const SCreationParams& getCreationParams() const {return m_params;}

		//! Creates a simple cube mesh.
		/**
		\param size Dimensions of the cube.
		\return Generated mesh.
		*/
		core::smart_refctd_ptr<ICPUPolygonGeometry> createCube(const hlsl::float32_t3 size={5.f,5.f,5.f}) const;


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
		core::smart_refctd_ptr<ICPUGeometryCollection> createArrow(const uint16_t tesselationCylinder = 4,
				const uint16_t tesselationCone = 8, const float height = 1.f,
				const float cylinderHeight = 0.6f, const float widthCylinder = 0.05f,
				const float widthCone = 0.3f) const;


		//! Create a sphere mesh.
		/**
		\param radius Radius of the sphere
		\param polyCountX Number of quads used for the horizontal tiling
		\param polyCountY Number of quads used for the vertical tiling
		\return Generated mesh.
		*/
		core::smart_refctd_ptr<ICPUPolygonGeometry> createSphere(float radius = 5.f,
				uint32_t polyCountX = 16, uint32_t polyCountY = 16, CQuantNormalCache* const quantNormalCacheOverride=nullptr) const;

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
		core::smart_refctd_ptr<ICPUPolygonGeometry> createCylinder(float radius, float length,
				uint16_t tesselation,
				CQuantNormalCache* const quantNormalCacheOverride=nullptr) const;

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
		core::smart_refctd_ptr<ICPUPolygonGeometry> createCone(float radius, float length, uint16_t tesselation,
				float oblique=0.f, CQuantNormalCache* const quantNormalCacheOverride=nullptr) const;

		core::smart_refctd_ptr<ICPUPolygonGeometry> createPrism(float radius, float length, uint16_t sideCount) const;

		core::smart_refctd_ptr<ICPUPolygonGeometry> createRectangle(const hlsl::float32_t2 size={0.5f,0.5f}) const;

		core::smart_refctd_ptr<ICPUPolygonGeometry> createDisk(const float radius, const uint32_t tesselation) const;

		//! Create a icosphere geometry
		/**
			\param radius Radius of the icosphere.
			\param subdivision Specifies subdivision level of the icosphere.
			\param smooth Specifies whether vertecies should be built for smooth or flat shading.
		*/

		core::smart_refctd_ptr<ICPUPolygonGeometry> createIcoSphere(float radius=1.f, uint32_t subdivision=1, bool smooth=false) const;

	private:
		SCreationParams m_params;
};

} // end namespace nbl::asset
#endif


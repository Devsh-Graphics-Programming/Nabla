// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_GEOMETRY_CREATOR_H_INCLUDED__
#define __C_GEOMETRY_CREATOR_H_INCLUDED__

#include "IGeometryCreator.h"

namespace irr
{

namespace scene
{

//! class for creating geometry on the fly
class CGeometryCreator : public IGeometryCreator
{
public:
	ICPUMesh* createCubeMeshCPU(const core::vector3df& size) const;
	IGPUMesh* createCubeMeshGPU(video::IVideoDriver* driver, const core::vector3df& size) const;

	ICPUMesh* createTerrainMeshCPU(video::IImage* texture,
		video::IImage* heightmap, const core::dimension2d<f32>& stretchSize,
		f32 maxHeight, video::IVideoDriver* driver,
		const core::dimension2d<u32>& defaultVertexBlockSize,
		bool debugBorders=false) const;
	IGPUMesh* createTerrainMeshGPU(video::IImage* texture,
		video::IImage* heightmap, const core::dimension2d<f32>& stretchSize,
		f32 maxHeight, video::IVideoDriver* driver,
		const core::dimension2d<u32>& defaultVertexBlockSize,
		bool debugBorders=false) const;

	ICPUMesh* createArrowMeshCPU(const u32 tesselationCylinder,
			const u32 tesselationCone, const f32 height,
			const f32 cylinderHeight, const f32 width0,
			const f32 width1, const video::SColor vtxColor0,
			const video::SColor vtxColor1) const;
	IGPUMesh* createArrowMeshGPU(video::IVideoDriver* driver, const u32 tesselationCylinder,
			const u32 tesselationCone, const f32 height,
			const f32 cylinderHeight, const f32 width0,
			const f32 width1, const video::SColor vtxColor0,
			const video::SColor vtxColor1) const;

	ICPUMesh* createSphereMeshCPU(f32 radius, u32 polyCountX, u32 polyCountY) const;
	IGPUMesh* createSphereMeshGPU(video::IVideoDriver* driver, f32 radius, u32 polyCountX, u32 polyCountY) const;

	ICPUMesh* createCylinderMeshCPU(f32 radius, f32 length, u32 tesselation,
				const video::SColor& color=0xffffffff,
				bool closeTop=true, f32 oblique=0.f) const;
	IGPUMesh* createCylinderMeshGPU(video::IVideoDriver* driver, f32 radius, f32 length, u32 tesselation,
				const video::SColor& color=0xffffffff,
				bool closeTop=true, f32 oblique=0.f) const;

	ICPUMesh* createConeMeshCPU(f32 radius, f32 length, u32 tesselation,
				const video::SColor& colorTop=0xffffffff,
				const video::SColor& colorBottom=0xffffffff,
				f32 oblique=0.f) const;
	IGPUMesh* createConeMeshGPU(video::IVideoDriver* driver, f32 radius, f32 length, u32 tesselation,
				const video::SColor& colorTop=0xffffffff,
				const video::SColor& colorBottom=0xffffffff,
				f32 oblique=0.f) const;

};

} // end namespace scene
} // end namespace irr

#endif


// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_MESH_MANIPULATOR_H_INCLUDED__
#define __C_MESH_MANIPULATOR_H_INCLUDED__

#include "IMeshManipulator.h"

namespace irr
{
namespace scene
{

//! An interface for easy manipulation of meshes.
/** Scale, set alpha value, flip surfaces, and so on. This exists for fixing
problems with wrong imported or exported meshes quickly after loading. It is
not intended for doing mesh modifications and/or animations during runtime.
*/
class CMeshManipulator : public IMeshManipulator
{
	struct Attrib
	{
		E_COMPONENT_TYPE type;
		size_t size;
		E_VERTEX_ATTRIBUTE_ID vaid;
		E_COMPONENTS_PER_ATTRIBUTE cpa;
		size_t offset;

		Attrib() : type(ECT_COUNT), size(0), vaid(EVAI_COUNT) {}

		friend bool operator>(const Attrib& _a, const Attrib& _b) { return _a.size > _b.size; }
	};

public:
#ifndef NEW_MESHES
    virtual void isolateAndExtractMeshBuffer(ICPUMeshBuffer* inbuffer, const bool& interleaved=true) const = 0;
#endif // NEW_MESHES

	//! Flips the direction of surfaces.
	/** Changes backfacing triangles to frontfacing triangles and vice versa.
	\param mesh: Mesh on which the operation is performed. */
	virtual void flipSurfaces(ICPUMeshBuffer* inbuffer) const;

#ifndef NEW_MESHES
	//! Recalculates all normals of the mesh.
	/** \param mesh: Mesh on which the operation is performed.
	    \param smooth: Whether to use smoothed normals. */
	virtual void recalculateNormals(scene::IMesh* mesh, bool smooth = false, bool angleWeighted = false) const;

	//! Recalculates all normals of the mesh buffer.
	/** \param buffer: Mesh buffer on which the operation is performed.
	    \param smooth: Whether to use smoothed normals. */
	virtual void recalculateNormals(IMeshBuffer* buffer, bool smooth = false, bool angleWeighted = false) const;

	//! Creates a planar texture mapping on the mesh
	/** \param mesh: Mesh on which the operation is performed.
	\param resolution: resolution of the planar mapping. This is the value
	specifying which is the relation between world space and
	texture coordinate space. */
	virtual void makePlanarTextureMapping(scene::IMesh* mesh, float resolution=0.001f) const;

	//! Creates a planar texture mapping on the meshbuffer
	virtual void makePlanarTextureMapping(scene::IMeshBuffer* meshbuffer, float resolution=0.001f) const;

	//! Creates a planar texture mapping on the meshbuffer
	void makePlanarTextureMapping(scene::IMeshBuffer* buffer, float resolutionS, float resolutionT, uint8_t axis, const core::vector3df& offset) const;

	//! Creates a planar texture mapping on the mesh
	void makePlanarTextureMapping(scene::IMesh* mesh, float resolutionS, float resolutionT, uint8_t axis, const core::vector3df& offset) const;

	//! Recalculates tangents, requires a tangent mesh buffer
	virtual void recalculateTangents(IMeshBuffer* buffer, bool recalculateNormals=false, bool smooth=false, bool angleWeighted=false) const;

	//! Recalculates tangents, requires a tangent mesh
	virtual void recalculateTangents(IMesh* mesh, bool recalculateNormals=false, bool smooth=false, bool angleWeighted=false) const;
#endif // NEW_MESHES

	//! Creates a copy of the mesh, which will only consist of unique triangles, i.e. no vertices are shared.
	virtual ICPUMeshBuffer* createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer) const;

	//! Creates a copy of the mesh, which will have all duplicated vertices removed, i.e. maximal amount of vertices are shared via indexing.
	virtual ICPUMeshBuffer* createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const bool& makeNewMesh=false, float tolerance=core::ROUNDING_ERROR_f32) const;

	virtual ICPUMeshBuffer* createOptimizedMeshBuffer(ICPUMeshBuffer* inbuffer) const;

#ifndef NEW_MESHES
	//! create a mesh optimized for the vertex cache
	virtual ICPUMeshBuffer* createForsythOptimizedMeshBuffer(const ICPUMeshBuffer *meshbuffer) const;
#endif // NEW_MESHES

private:
	std::vector<core::vectorSIMDf> findBetterFormatF(E_COMPONENT_TYPE* _outType, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId) const;
	
	struct SIntegerAttr
	{
		uint32_t pointer[4];
	};
	std::vector<SIntegerAttr> findBetterFormatI(E_COMPONENT_TYPE* _outType, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId) const;

	E_COMPONENT_TYPE getBestTypeF(bool _normalized, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const float* _min, const float* _max) const;
	E_COMPONENT_TYPE getBestTypeI(bool _nativeInt, bool _unsigned, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const uint32_t* _min, const uint32_t* _max) const;
};


} // end namespace scene
} // end namespace irr


#endif

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
	struct SAttrib
	{
		E_COMPONENT_TYPE type;
		E_COMPONENT_TYPE prevType;
		size_t size;
		E_VERTEX_ATTRIBUTE_ID vaid;
		E_COMPONENTS_PER_ATTRIBUTE cpa;
		size_t offset;

		SAttrib() : type(ECT_COUNT), size(0), vaid(EVAI_COUNT) {}

		friend bool operator>(const SAttrib& _a, const SAttrib& _b) { return _a.size > _b.size; }
	};
	struct SAttribTypeChoice
	{
		E_COMPONENT_TYPE type;
		E_COMPONENTS_PER_ATTRIBUTE cpa;
	};

public:
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

	//! Recalculates tangents, requires a tangent mesh buffer
	virtual void recalculateTangents(IMeshBuffer* buffer, bool recalculateNormals=false, bool smooth=false, bool angleWeighted=false) const;

	//! Recalculates tangents, requires a tangent mesh
	virtual void recalculateTangents(IMesh* mesh, bool recalculateNormals=false, bool smooth=false, bool angleWeighted=false) const;
#endif // NEW_MESHES

	virtual ICPUMeshBuffer* createMeshBufferFetchOptimized(const ICPUMeshBuffer* _inbuffer) const;

	//! Creates a copy of the mesh, which will only consist of unique triangles, i.e. no vertices are shared.
	virtual ICPUMeshBuffer* createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer) const;

	//! Creates a copy of the mesh, which will have all duplicated vertices removed, i.e. maximal amount of vertices are shared via indexing.
	virtual ICPUMeshBuffer* createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const bool& reduceIdxBufSize = true, const bool& makeNewMesh=false, float tolerance=core::ROUNDING_ERROR_f32) const;

	virtual ICPUMeshBuffer* createOptimizedMeshBuffer(const ICPUMeshBuffer* inbuffer, const SErrorMetric* _requantErrMetric) const;

	virtual void requantizeMeshBuffer(ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric) const;

	virtual ICPUMeshBuffer* createMeshBufferDuplicate(const ICPUMeshBuffer* _src) const;

    virtual void filterInvalidTriangles(ICPUMeshBuffer* _input) const;

    virtual core::ICPUBuffer* idxBufferFromTriangleStripsToTriangles(const void* _input, size_t _idxCount, video::E_INDEX_TYPE _idxType) const;

    virtual core::ICPUBuffer* idxBufferFromTrianglesFanToTriangles(const void* _input, size_t _idxCount, video::E_INDEX_TYPE _idxType) const;

private:
    template<typename IdxT>
    void priv_filterInvalidTriangles(ICPUMeshBuffer* _input) const;

	//! Meant to create 32bit index buffer from subrange of index buffer containing 16bit indices. Remember to set to index buffer offset to 0 after mapping buffer resulting from this function.
	core::ICPUBuffer* create32BitFrom16BitIdxBufferSubrange(const uint16_t* _in, size_t _idxCount) const;

	std::vector<core::vectorSIMDf> findBetterFormatF(E_COMPONENT_TYPE* _outType, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, E_COMPONENT_TYPE* _outPrevType, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric) const;
	
	struct SIntegerAttr
	{
		uint32_t pointer[4];
	};
	std::vector<SIntegerAttr> findBetterFormatI(E_COMPONENT_TYPE* _outType, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, E_COMPONENT_TYPE* _outPrevType, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric) const;

	//E_COMPONENT_TYPE getBestTypeF(bool _normalized, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const float* _min, const float* _max) const;
	E_COMPONENT_TYPE getBestTypeI(bool _nativeInt, bool _unsigned, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const uint32_t* _min, const uint32_t* _max) const;
	std::vector<SAttribTypeChoice> findTypesOfProperRangeF(E_COMPONENT_TYPE _type, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t _sizeThreshold, const float* _min, const float* _max, const SErrorMetric& _errMetric) const;

	//! Calculates quantization errors and compares them with given epsilon.
	/** @returns false when first of calculated errors goes above epsilon or true if reached end without such. */
	bool calcMaxQuantizationError(const SAttribTypeChoice& _srcType, const SAttribTypeChoice& _dstType, const std::vector<core::vectorSIMDf>& _data, const SErrorMetric& _errMetric) const;

	template<typename T>
	core::ICPUBuffer* triangleStripsToTriangles(const void* _input, size_t _idxCount) const;

	template<typename T>
	core::ICPUBuffer* trianglesFanToTriangles(const void* _input, size_t _idxCount) const;
};

} // end namespace scene
} // end namespace irr


#endif

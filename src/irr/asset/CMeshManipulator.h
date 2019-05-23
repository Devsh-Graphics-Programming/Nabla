// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_MESH_MANIPULATOR_H_INCLUDED__
#define __C_MESH_MANIPULATOR_H_INCLUDED__

#include "irr/asset/IMeshManipulator.h"

namespace irr
{
namespace asset
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
		asset::E_FORMAT type;
		asset::E_FORMAT prevType;
		size_t size;
        asset::E_VERTEX_ATTRIBUTE_ID vaid;
		size_t offset;

		SAttrib() : type(asset::EF_UNKNOWN), size(0), vaid(asset::EVAI_COUNT) {}

		friend bool operator>(const SAttrib& _a, const SAttrib& _b) { return _a.size > _b.size; }
	};
	struct SAttribTypeChoice
	{
		asset::E_FORMAT type;
	};

public:
	//! Flips the direction of surfaces.
	/** Changes backfacing triangles to frontfacing triangles and vice versa.
	\param mesh: Mesh on which the operation is performed. */
	virtual void flipSurfaces(asset::ICPUMeshBuffer* inbuffer) const;

	virtual asset::ICPUMeshBuffer* createMeshBufferFetchOptimized(const asset::ICPUMeshBuffer* _inbuffer) const;

	//! Creates a copy of the mesh, which will only consist of unique triangles, i.e. no vertices are shared.
	virtual asset::ICPUMeshBuffer* createMeshBufferUniquePrimitives(asset::ICPUMeshBuffer* inbuffer) const;

	//
	virtual asset::ICPUMeshBuffer* calculateSmoothNormals(asset::ICPUMeshBuffer* inbuffer, bool makeNewMesh, float epsilon,
		asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp) const override;

	//! Creates a copy of the mesh, which will have all duplicated vertices removed, i.e. maximal amount of vertices are shared via indexing.
	virtual asset::ICPUMeshBuffer* createMeshBufferWelded(asset::ICPUMeshBuffer *inbuffer, const SErrorMetric* _errMetrics, const bool& optimIndexType = true, const bool& makeNewMesh=false) const;

	virtual asset::ICPUMeshBuffer* createOptimizedMeshBuffer(const asset::ICPUMeshBuffer* inbuffer, const SErrorMetric* _errMetric) const;

	virtual void requantizeMeshBuffer(asset::ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric) const;

	virtual asset::ICPUMeshBuffer* createMeshBufferDuplicate(const asset::ICPUMeshBuffer* _src) const;

    virtual void filterInvalidTriangles(asset::ICPUMeshBuffer* _input) const;

    virtual asset::ICPUBuffer* idxBufferFromTriangleStripsToTriangles(const void* _input, size_t _idxCount, asset::E_INDEX_TYPE _idxType) const;

    virtual asset::ICPUBuffer* idxBufferFromTrianglesFanToTriangles(const void* _input, size_t _idxCount, asset::E_INDEX_TYPE _idxType) const;

    virtual bool compareFloatingPointAttribute(const core::vectorSIMDf& _a, const core::vectorSIMDf& _b, size_t _cpa, const SErrorMetric& _errMetric) const;

private:
    //! Copies only member variables not being pointers to another dynamically allocated irr::IReferenceCounted derivatives.
    //! Purely helper function. Not really meant to be used outside createMeshBufferDuplicate().
    template<typename T>
    void copyMeshBufferMemberVars(T* _dst, const T* _src) const;

    template<typename IdxT>
    void priv_filterInvalidTriangles(asset::ICPUMeshBuffer* _input) const;

	//! Meant to create 32bit index buffer from subrange of index buffer containing 16bit indices. Remember to set to index buffer offset to 0 after mapping buffer resulting from this function.
	asset::ICPUBuffer* create32BitFrom16BitIdxBufferSubrange(const uint16_t* _in, size_t _idxCount) const;

	core::vector<core::vectorSIMDf> findBetterFormatF(asset::E_FORMAT* _outType, size_t* _outSize, asset::E_FORMAT* _outPrevType, const asset::ICPUMeshBuffer* _meshbuffer, asset::E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric) const;

	struct SIntegerAttr
	{
		uint32_t pointer[4];
	};
	core::vector<SIntegerAttr> findBetterFormatI(asset::E_FORMAT* _outType, size_t* _outSize, asset::E_FORMAT* _outPrevType, const asset::ICPUMeshBuffer* _meshbuffer, asset::E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric) const;

	//E_COMPONENT_TYPE getBestTypeF(bool _normalized, E_COMPONENTS_PER_ATTRIBUTE _cpa, size_t* _outSize, E_COMPONENTS_PER_ATTRIBUTE* _outCpa, const float* _min, const float* _max) const;
	asset::E_FORMAT getBestTypeI(asset::E_FORMAT _originalType, size_t* _outSize, const uint32_t* _min, const uint32_t* _max) const;
	core::vector<SAttribTypeChoice> findTypesOfProperRangeF(asset::E_FORMAT _type, size_t _sizeThreshold, const float* _min, const float* _max, const SErrorMetric& _errMetric) const;

	//! Calculates quantization errors and compares them with given epsilon.
	/** @returns false when first of calculated errors goes above epsilon or true if reached end without such. */
	bool calcMaxQuantizationError(const SAttribTypeChoice& _srcType, const SAttribTypeChoice& _dstType, const core::vector<core::vectorSIMDf>& _data, const SErrorMetric& _errMetric) const;

	template<typename T>
	asset::ICPUBuffer* triangleStripsToTriangles(const void* _input, size_t _idxCount) const;

	template<typename T>
	asset::ICPUBuffer* trianglesFanToTriangles(const void* _input, size_t _idxCount) const;
};

} // end namespace scene
} // end namespace irr


#endif

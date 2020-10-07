// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "os.h"
#include "irr/asset/asset.h"
#include "irr/asset/CMeshManipulator.h"
#include "irr/asset/CSmoothNormalGenerator.h"
#include "irr/asset/CForsythVertexCacheOptimizer.h"
#include "irr/asset/COverdrawMeshOptimizer.h"

namespace irr
{
namespace asset
{

//! Flips the direction of surfaces. Changes backfacing triangles to frontfacing
//! triangles and vice versa.
//! \param mesh: Mesh on which the operation is performed.
void IMeshManipulator::flipSurfaces(ICPUMeshBuffer* inbuffer)
{
}

core::smart_refctd_ptr<ICPUMeshBuffer> CMeshManipulator::createMeshBufferFetchOptimized(const ICPUMeshBuffer* _inbuffer)
{
    return nullptr;
}

//! Creates a copy of the mesh, which will only consist of unique primitives
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer, bool _makeIndexBuf)
{
    return nullptr;
}

//
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::calculateSmoothNormals(ICPUMeshBuffer* inbuffer, bool makeNewMesh, float epsilon, uint32_t normalAttrID, VxCmpFunction vxcmp)
{
	if (inbuffer == nullptr)
	{
		_NBL_DEBUG_BREAK_IF(true);
		return nullptr;
	}

	//Mesh has to have unique primitives
	if (inbuffer->getIndexType() != E_INDEX_TYPE::EIT_UNKNOWN)
	{
		_NBL_DEBUG_BREAK_IF(true);
		return nullptr;
	}

	auto outbuffer = makeNewMesh ? createMeshBufferDuplicate(inbuffer) : core::smart_refctd_ptr<ICPUMeshBuffer>(inbuffer);
	CSmoothNormalGenerator::calculateNormals(outbuffer.get(), epsilon, normalAttrID, vxcmp);

	return outbuffer;
}

// Used by createMeshBufferWelded only
static bool cmpVertices(ICPUMeshBuffer* _inbuf, const void* _va, const void* _vb, size_t _vsize, const IMeshManipulator::SErrorMetric* _errMetrics)
{
    return false;
}

//! Creates a copy of a mesh, which will have identical vertices welded together
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const SErrorMetric* _errMetrics, const bool& optimIndexType, const bool& makeNewMesh)
{
    return nullptr;
}

core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createOptimizedMeshBuffer(const ICPUMeshBuffer* _inbuffer, const SErrorMetric* _errMetric)
{
    return nullptr;
}

void IMeshManipulator::requantizeMeshBuffer(ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric)
{
}


template<>
void CMeshManipulator::copyMeshBufferMemberVars<ICPUMeshBuffer>(ICPUMeshBuffer* _dst, const ICPUMeshBuffer* _src)
{
	_dst->setBoundingBox(
		_src->getBoundingBox()
	);
	for (uint32_t i = 0u; i < ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
		_dst->setVertexBufferBinding(
			SBufferBinding(_src->getVertexBufferBindings()[i]), i
		);
	_dst->setIndexBufferBinding(
		SBufferBinding(_src->getIndexBufferBinding())
	);/*
	_dst->setAttachedDescriptorSet(
		core::smart_refctd_ptr<ICPUDescriptorSet>(_src->getAttachedDescriptorSet())
	);
	_dst->setPipeline(
		core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>(_src->getPipeline())
	);*/
	_dst->setIndexType(
		_src->getIndexType()
	);
	_dst->setBaseVertex(
		_src->getBaseVertex()
	);
	memcpy(_dst->getPushConstantsDataPtr(),_src->getPushConstantsDataPtr(),ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
	_dst->setIndexCount(
		_src->getIndexCount()
	);
    _dst->setInstanceCount(
        _src->getInstanceCount()
    );
	_dst->setBaseInstance(
		_src->getBaseInstance()
	);
    _dst->setPositionAttributeIx(
        _src->getPositionAttributeIx()
    );
	_dst->setNormalnAttributeIx(
		_src->getNormalAttributeIx()
	);
    //_dst->getMaterial() = _src->getMaterial();
}
template<>
void CMeshManipulator::copyMeshBufferMemberVars<ICPUSkinnedMeshBuffer>(ICPUSkinnedMeshBuffer* _dst, const ICPUSkinnedMeshBuffer* _src)
{
    copyMeshBufferMemberVars<ICPUMeshBuffer>(_dst, _src);
    _dst->setIndexRange(
        _src->getIndexMinBound(),
        _src->getIndexMaxBound()
    );
    _dst->setMaxVertexBoneInfluences(
        _src->getMaxVertexBoneInfluences()
    );
}

core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createMeshBufferDuplicate(const ICPUMeshBuffer* _src)
{
    return nullptr;
}

void IMeshManipulator::filterInvalidTriangles(ICPUMeshBuffer* _input)
{
    if (!_input || !_input->getPipeline() || !_input->getIndices())
        return;

    switch (_input->getIndexType())
    {
    case EIT_16BIT:
        return CMeshManipulator::_filterInvalidTriangles<uint16_t>(_input);
    case EIT_32BIT:
        return CMeshManipulator::_filterInvalidTriangles<uint32_t>(_input);
    default: return;
    }
}

template<typename IdxT>
void CMeshManipulator::_filterInvalidTriangles(ICPUMeshBuffer* _input)
{
    const size_t size = _input->getIndexCount() * sizeof(IdxT);
    void* const copy = _NBL_ALIGNED_MALLOC(size,_NBL_SIMD_ALIGNMENT);
    memcpy(copy, _input->getIndices(), size);

    struct Triangle
    {
        IdxT i[3];
    } *const begin = (Triangle*)copy, *const end = (Triangle*)((uint8_t*)copy + size);

    Triangle* const newEnd = std::remove_if(begin, end,
        [&_input](const Triangle& _t) {
            core::vectorSIMDf p0, p1, p2;
            const uint32_t pvaid = _input->getPositionAttributeIx();
            _input->getAttribute(p0, pvaid, _t.i[0]);
            _input->getAttribute(p1, pvaid, _t.i[1]);
            _input->getAttribute(p2, pvaid, _t.i[2]);
			return core::length(core::cross(p1 - p0, p2 - p0)).x<=1.0e-19F;
    });
    const size_t newSize = std::distance(begin, newEnd) * sizeof(Triangle);

    auto newBuf = core::make_smart_refctd_ptr<ICPUBuffer>(newSize);
    memcpy(newBuf->getPointer(), copy, newSize);
    _NBL_ALIGNED_FREE(copy);

    SBufferBinding<ICPUBuffer> idxBufBinding;
    idxBufBinding.offset = 0ull;
    idxBufBinding.buffer = std::move(newBuf);
    _input->setIndexBufferBinding(std::move(idxBufBinding));
    _input->setIndexCount(newSize/sizeof(IdxT));
}
template void CMeshManipulator::_filterInvalidTriangles<uint16_t>(ICPUMeshBuffer* _input);
template void CMeshManipulator::_filterInvalidTriangles<uint32_t>(ICPUMeshBuffer* _input);

core::vector<core::vectorSIMDf> CMeshManipulator::findBetterFormatF(E_FORMAT* _outType, size_t* _outSize, E_FORMAT* _outPrevType, const ICPUMeshBuffer* _meshbuffer, uint32_t _attrId, const SErrorMetric& _errMetric, CQuantNormalCache& _cache)
{
	if (!_meshbuffer->getPipeline())
        return {};

	const E_FORMAT thisType = _meshbuffer->getAttribFormat(_attrId);

    if (!isFloatingPointFormat(thisType) && !isNormalizedFormat(thisType) && !isScaledFormat(thisType))
        return {};

	core::vector<core::vectorSIMDf> attribs;


    const uint32_t cpa = getFormatChannelCount(thisType);

	float min[4]{ FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	float max[4]{ -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

	core::vectorSIMDf attr;
    const size_t cnt = _meshbuffer->calcVertexCount();
    for (size_t idx = 0u; idx < cnt; ++idx)
	{
        _meshbuffer->getAttribute(attr, _attrId, idx);
		attribs.push_back(attr);
		for (uint32_t i = 0; i < cpa ; ++i)
		{
			if (attr.pointer[i] < min[i])
				min[i] = attr.pointer[i];
			if (attr.pointer[i] > max[i])
				max[i] = attr.pointer[i];
		}
	}

	core::vector<SAttribTypeChoice> possibleTypes = findTypesOfProperRangeF(thisType, getTexelOrBlockBytesize(thisType), min, max, _errMetric);
	std::sort(possibleTypes.begin(), possibleTypes.end(), [](const SAttribTypeChoice& t1, const SAttribTypeChoice& t2) { return getTexelOrBlockBytesize(t1.type) < getTexelOrBlockBytesize(t2.type); });

	*_outPrevType = thisType;
    *_outType = thisType;
    *_outSize = getTexelOrBlockBytesize(*_outType);

	for (const SAttribTypeChoice& t : possibleTypes)
	{
		if (calcMaxQuantizationError({ thisType }, t, attribs, _errMetric, _cache))
		{
            if (getTexelOrBlockBytesize(t.type) < getTexelOrBlockBytesize(thisType))
            {
                *_outType = t.type;
                *_outSize = getTexelOrBlockBytesize(*_outType);
            }

			return attribs;
		}
	}

	return attribs;
}

core::vector<CMeshManipulator::SIntegerAttr> CMeshManipulator::findBetterFormatI(E_FORMAT* _outType, size_t* _outSize, E_FORMAT* _outPrevType, const ICPUMeshBuffer* _meshbuffer, uint32_t _attrId, const SErrorMetric& _errMetric)
{
	if (!_meshbuffer->getPipeline())
        return {};

    const E_FORMAT thisType = _meshbuffer->getAttribFormat(_attrId);

    if (!isIntegerFormat(thisType))
        return {};

    if (isBGRALayoutFormat(thisType))
        return {}; // BGRA is supported only by a few normalized types (this is function for integer types)

	core::vector<SIntegerAttr> attribs;


    const uint32_t cpa = getFormatChannelCount(thisType);

	uint32_t min[4];
	uint32_t max[4];
	if (!isSignedFormat(thisType))
		for (size_t i = 0; i < 4; ++i)
			min[i] = UINT_MAX;
	else
		for (size_t i = 0; i < 4; ++i)
			min[i] = INT_MAX;
	if (!isSignedFormat(thisType))
		for (size_t i = 0; i < 4; ++i)
			max[i] = 0;
	else
		for (size_t i = 0; i < 4; ++i)
			max[i] = INT_MIN;


	SIntegerAttr attr;
    const size_t cnt = _meshbuffer->calcVertexCount();
    for (size_t idx = 0u; idx < cnt; ++idx)
	{
        _meshbuffer->getAttribute(attr.pointer, _attrId, idx);
		attribs.push_back(attr);
		for (size_t i = 0; i < cpa; ++i)
		{
			if (!isSignedFormat(thisType))
			{
				if (attr.pointer[i] < min[i])
					min[i] = attr.pointer[i];
				if (attr.pointer[i] > max[i])
					max[i] = attr.pointer[i];
			}
			else
			{
				if (((int32_t*)attr.pointer + i)[0] < ((int32_t*)min + i)[0])
					min[i] = attr.pointer[i];
				if (((int32_t*)attr.pointer + i)[0] > ((int32_t*)max + i)[0])
					max[i] = attr.pointer[i];
			}
		}
	}

	*_outPrevType = *_outType = thisType;
	*_outSize = getTexelOrBlockBytesize(thisType);
	*_outPrevType = thisType;

	if (_errMetric.method == EEM_ANGLES) // native integers normals does not change
		return attribs;

	*_outType = getBestTypeI(thisType, _outSize, min, max);
    if (getTexelOrBlockBytesize(*_outType) >= getTexelOrBlockBytesize(thisType))
    {
        *_outType = thisType;
        *_outSize = getTexelOrBlockBytesize(thisType);
    }
	return attribs;
}

E_FORMAT CMeshManipulator::getBestTypeI(E_FORMAT _originalType, size_t* _outSize, const uint32_t* _min, const uint32_t* _max)
{
    using namespace video;

    const bool isNativeInteger = isIntegerFormat(_originalType);
    const bool isUnsigned = !isSignedFormat(_originalType);

    const uint32_t originalCpa = getFormatChannelCount(_originalType);

    core::vector<E_FORMAT> nativeInts{
        EF_R8G8_UINT,
        EF_R8G8_SINT,
        EF_R8G8B8_UINT,
        EF_R8G8B8_SINT,
        EF_R8G8B8A8_UINT,
        EF_R8G8B8A8_SINT,
        EF_A2B10G10R10_UINT_PACK32,
        EF_A2B10G10R10_SINT_PACK32,
        EF_R16_UINT,
        EF_R16_SINT,
        EF_R16G16_UINT,
        EF_R16G16_SINT,
        EF_R16G16B16_UINT,
        EF_R16G16B16_SINT,
        EF_R16G16B16A16_UINT,
        EF_R16G16B16A16_SINT,
        EF_R32_UINT,
        EF_R32_SINT,
        EF_R32G32_UINT,
        EF_R32G32_SINT,
        EF_R32G32B32_UINT,
        EF_R32G32B32_SINT,
        EF_R32G32B32A32_UINT,
        EF_R32G32B32A32_SINT
    };
    core::vector<E_FORMAT> scaledInts{
        EF_R8G8_USCALED,
        EF_R8G8_SSCALED,
        EF_R8G8B8_USCALED,
        EF_R8G8B8_SSCALED,
        EF_R8G8B8A8_USCALED,
        EF_R8G8B8A8_SSCALED,
        EF_A2B10G10R10_USCALED_PACK32,
        EF_A2B10G10R10_SSCALED_PACK32,
        EF_R16_USCALED,
        EF_R16_SSCALED,
        EF_R16G16_USCALED,
        EF_R16G16_SSCALED,
        EF_R16G16B16_USCALED,
        EF_R16G16B16_SSCALED,
        EF_R16G16B16A16_USCALED,
        EF_R16G16B16A16_SSCALED
    };

    core::vector<E_FORMAT>& all = isNativeInteger ? nativeInts : scaledInts;
    if (originalCpa > 1u)
    {
        all.erase(
            std::remove_if(all.begin(), all.end(),
                [originalCpa](E_FORMAT fmt) { return getFormatChannelCount(fmt) < originalCpa; }
            ),
            all.end()
        );
    }

    auto minValueOfTypeINT = [](E_FORMAT _fmt, uint32_t _cmpntNum) -> int32_t {
        if (!isSignedFormat(_fmt))
            return 0;

        switch (_fmt)
        {
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2B10G10R10_SSCALED_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
            if (_cmpntNum < 3u)
                return -512;
            else return -2;
            break;
        default:
        {
        const uint32_t bitsPerCh = getTexelOrBlockBytesize(_fmt)*8u/getFormatChannelCount(_fmt);
        return int32_t(-uint64_t(1ull<<(bitsPerCh-1u)));
        }
        }
    };
    auto maxValueOfTypeINT = [](E_FORMAT _fmt, uint32_t _cmpntNum) -> uint32_t {
        switch (_fmt)
        {
        case EF_A2R10G10B10_USCALED_PACK32:
        case EF_A2R10G10B10_UINT_PACK32:
        case EF_A2B10G10R10_USCALED_PACK32:
        case EF_A2B10G10R10_UINT_PACK32:
            if (_cmpntNum < 3u)
                return 1023u;
            else return 3u;
            break;
        case EF_A2R10G10B10_SSCALED_PACK32:
        case EF_A2R10G10B10_SINT_PACK32:
        case EF_A2B10G10R10_SSCALED_PACK32:
        case EF_A2B10G10R10_SINT_PACK32:
            if (_cmpntNum < 3u)
                return 511u;
            else return 1u;
            break;
        default:
        {
            const uint32_t bitsPerCh = getTexelOrBlockBytesize(_fmt)*8u/getFormatChannelCount(_fmt);
            const uint64_t r = (1ull<<bitsPerCh)-1ull;
            if (!isSignedFormat(_fmt))
                return (uint32_t)r;
            return (uint32_t)(r>>1);
        }
        }
    };

    E_FORMAT bestType = _originalType;
    for (auto it = all.begin(); it != all.end(); ++it)
    {
        bool ok = true;
        for (uint32_t cmpntNum = 0; cmpntNum < originalCpa; ++cmpntNum) // check only `_cpa` components because even if (chosenCpa > _cpa), we don't care about extra components
        {
            if (isUnsigned)
            {
                if (!(_min[cmpntNum] >= minValueOfTypeINT(*it, cmpntNum) && _max[cmpntNum] <= maxValueOfTypeINT(*it, cmpntNum))) //! TODO: FIX signed vs. unsigned comparison
                {
                    ok = false;
                    break;
                }
            }
            else
            {
                if (!(((int32_t*)(_min + cmpntNum))[0] >= minValueOfTypeINT(*it, cmpntNum) && ((int32_t*)(_max + cmpntNum))[0] <= maxValueOfTypeINT(*it, cmpntNum))) //! TODO: FIX signed vs. unsigned comparison
                {
                    ok = false;
                    break;
                }
            }
        }
        if (ok && getTexelOrBlockBytesize(*it) < getTexelOrBlockBytesize(bestType)) // vertexAttrSize array defined in IMeshBuffer.h
        {
            bestType = *it;
            *_outSize = getTexelOrBlockBytesize(bestType);
        }
    }

    return bestType;
}

core::vector<CMeshManipulator::SAttribTypeChoice> CMeshManipulator::findTypesOfProperRangeF(E_FORMAT _type, size_t _sizeThreshold, const float * _min, const float * _max, const SErrorMetric& _errMetric)
{
    using namespace video;

    core::vector<E_FORMAT> all{
        EF_B10G11R11_UFLOAT_PACK32,
        EF_R16_SFLOAT,
        EF_R16G16_SFLOAT,
        EF_R16G16B16_SFLOAT,
        EF_R16G16B16A16_SFLOAT,
        EF_R32_SFLOAT,
        EF_R32G32_SFLOAT,
        EF_R32G32B32_SFLOAT,
        EF_R32G32B32A32_SFLOAT,
        EF_R8G8_UNORM,
        EF_R8G8_SNORM,
        EF_R8G8B8_UNORM,
        EF_R8G8B8_SNORM,
        EF_B8G8R8A8_UNORM, //bgra
        EF_R8G8B8A8_UNORM,
        EF_R8G8B8A8_SNORM,
        EF_A2B10G10R10_UNORM_PACK32,
        EF_A2B10G10R10_SNORM_PACK32,
        EF_A2R10G10B10_UNORM_PACK32, //bgra
        EF_A2R10G10B10_SNORM_PACK32, //bgra
        EF_R16_UNORM,
        EF_R16_SNORM,
        EF_R16G16_UNORM,
        EF_R16G16_SNORM,
        EF_R16G16B16_UNORM,
        EF_R16G16B16_SNORM,
        EF_R16G16B16A16_UNORM,
        EF_R16G16B16A16_SNORM
    };
    core::vector<E_FORMAT> normalized{
        EF_B8G8R8A8_UNORM, //bgra
        EF_R8G8B8A8_UNORM,
        EF_R8G8B8A8_SNORM,
        EF_A2B10G10R10_UNORM_PACK32,
        EF_A2B10G10R10_SNORM_PACK32,
        EF_A2R10G10B10_UNORM_PACK32, //bgra
        EF_A2R10G10B10_SNORM_PACK32, //bgra
        EF_R16_UNORM,
        EF_R16_SNORM,
        EF_R16G16_UNORM,
        EF_R16G16_SNORM,
        EF_R16G16B16_UNORM,
        EF_R16G16B16_SNORM,
        EF_R16G16B16A16_UNORM,
        EF_R16G16B16A16_SNORM
    };
    core::vector<E_FORMAT> bgra{
        EF_B8G8R8A8_UNORM, //bgra
        EF_A2R10G10B10_UNORM_PACK32, //bgra
        EF_A2R10G10B10_SNORM_PACK32, //bgra
    };
    core::vector<E_FORMAT> normals{
        EF_R8_SNORM,
        EF_R8G8_SNORM,
        EF_R8G8B8_SNORM,
        EF_R8G8B8A8_SNORM,
        EF_R16_SNORM,
        EF_R16G16_SNORM,
        EF_R16G16B16_SNORM,
        EF_R16G16B16A16_SNORM,
        EF_A2B10G10R10_SNORM_PACK32,
        EF_A2R10G10B10_SNORM_PACK32, //bgra
        EF_R16_SFLOAT,
        EF_R16G16_SFLOAT,
        EF_R16G16B16_SFLOAT,
        EF_R16G16B16A16_SFLOAT
    };

    auto minValueOfTypeFP = [](E_FORMAT _fmt, uint32_t _cmpntNum) -> float {
        if (isNormalizedFormat(_fmt))
        {
            return isSignedFormat(_fmt) ? -1.f : 0.f;
        }
        switch (_fmt)
        {
        case EF_R16_SFLOAT:
        case EF_R16G16_SFLOAT:
        case EF_R16G16B16_SFLOAT:
        case EF_R16G16B16A16_SFLOAT:
            return -65504.f;
        case EF_R32_SFLOAT:
        case EF_R32G32_SFLOAT:
        case EF_R32G32B32_SFLOAT:
        case EF_R32G32B32A32_SFLOAT:
            return -FLT_MAX;
        case EF_B10G11R11_UFLOAT_PACK32:
            return 0.f;
        default:
            return 1.f;
        }
    };
    auto maxValueOfTypeFP = [](E_FORMAT _fmt, uint32_t _cmpntNum) -> float {
        if (isNormalizedFormat(_fmt))
        {
            return 1.f;
        }
        switch (_fmt)
        {
        case EF_R16_SFLOAT:
        case EF_R16G16_SFLOAT:
        case EF_R16G16B16_SFLOAT:
        case EF_R16G16B16A16_SFLOAT:
            return 65504.f;
        case EF_R32_SFLOAT:
        case EF_R32G32_SFLOAT:
        case EF_R32G32B32_SFLOAT:
        case EF_R32G32B32A32_SFLOAT:
            return FLT_MAX;
        case EF_B10G11R11_UFLOAT_PACK32:
            if (_cmpntNum < 2u)
                return 65024.f;
            else return 64512.f;
        default:
            return 0.f;
        }
    };

	if (isNormalizedFormat(_type) || _errMetric.method == EEM_ANGLES)
	{
		if (_errMetric.method == EEM_ANGLES)
		{
            if (isBGRALayoutFormat(_type))
            {
                all = core::vector<E_FORMAT>(1u, EF_A2R10G10B10_SNORM_PACK32);
            }
			else all = std::move(normals);
		}
		else if (isBGRALayoutFormat(_type))
			all = std::move(bgra);
		else
			all = std::move(normalized);
	}

	if (isNormalizedFormat(_type) && !isSignedFormat(_type))
		all.erase(std::remove_if(all.begin(), all.end(), [](E_FORMAT _t) { return isSignedFormat(_t); }), all.end());
	else if (isNormalizedFormat(_type) && isSignedFormat(_type))
		all.erase(std::remove_if(all.begin(), all.end(), [](E_FORMAT _t) { return !isSignedFormat(_t); }), all.end());

    const uint32_t originalCpa = getFormatChannelCount(_type);
    all.erase(
        std::remove_if(all.begin(), all.end(),
            [originalCpa](E_FORMAT fmt) { return getFormatChannelCount(fmt) < originalCpa; }
        ),
        all.end()
    );

	core::vector<SAttribTypeChoice> possibleTypes;
	core::vectorSIMDf min(_min), max(_max);

	for (auto it = all.begin(); it != all.end(); ++it)
	{
		bool ok = true;
		for (uint32_t cmpntNum = 0; cmpntNum < originalCpa; ++cmpntNum) // check only `_cpa` components because even if (chosenCpa > _cpa), we don't care about extra components
		{
			if (!(min.pointer[cmpntNum] >= minValueOfTypeFP(*it, cmpntNum) && max.pointer[cmpntNum] <= maxValueOfTypeFP(*it, cmpntNum)))
			{
				ok = false;
				break; // break loop comparing (*it)'s range component by component
			}
		}
		if (ok && getTexelOrBlockBytesize(*it) <= _sizeThreshold)
			possibleTypes.push_back({*it});
	}
	return possibleTypes;
}

bool CMeshManipulator::calcMaxQuantizationError(const SAttribTypeChoice& _srcType, const SAttribTypeChoice& _dstType, const core::vector<core::vectorSIMDf>& _srcData, const SErrorMetric& _errMetric, CQuantNormalCache& _cache)
{
    using namespace video;

	using QuantF_t = core::vectorSIMDf(*)(const core::vectorSIMDf&, E_FORMAT, E_FORMAT, CQuantNormalCache & _cache);

	QuantF_t quantFunc = nullptr;

	if (_errMetric.method == EEM_ANGLES)
	{
		switch (_dstType.type)
		{
		case EF_R8_SNORM:
        case EF_R8G8_SNORM:
        case EF_R8G8B8_SNORM:
        case EF_R8G8B8A8_SNORM:
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT, CQuantNormalCache& _cache) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint32_t*)buf)[0] = _cache.quantizeNormal<CQuantNormalCache::E_CACHE_TYPE::ECT_8_8_8>(_in);

				core::vectorSIMDf retval;
				ICPUMeshBuffer::getAttribute(retval, buf, EF_R8G8B8A8_SNORM);
				retval.w = 1.f;
				return retval;
			};
			break;
		case EF_A2R10G10B10_SNORM_PACK32:
		case EF_A2B10G10R10_SNORM_PACK32: // bgra
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT, CQuantNormalCache& _cache) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint32_t*)buf)[0] = _cache.quantizeNormal<CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10>(_in);

				core::vectorSIMDf retval;
				ICPUMeshBuffer::getAttribute(retval, buf, EF_A2R10G10B10_SNORM_PACK32);
				retval.w = 1.f;
				return retval;
			};
			break;
        case EF_R16_SNORM:
        case EF_R16G16_SNORM:
        case EF_R16G16B16_SNORM:
        case EF_R16G16B16A16_SNORM:
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT, CQuantNormalCache& _cache) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint64_t*)buf)[0] = _cache.quantizeNormal<CQuantNormalCache::E_CACHE_TYPE::ECT_16_16_16>(_in);

				core::vectorSIMDf retval;
				ICPUMeshBuffer::getAttribute(retval, buf, EF_R16G16B16A16_SNORM);
				retval.w = 1.f;
				return retval;
			};
			break;
        default: 
            quantFunc = nullptr;
            break;
		}
	}
	else
	{
		quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT _inType, E_FORMAT _outType, CQuantNormalCache& _cache) -> core::vectorSIMDf {
			uint8_t buf[32];
			ICPUMeshBuffer::setAttribute(_in, buf, _outType);
			core::vectorSIMDf out(0.f, 0.f, 0.f, 1.f);
			ICPUMeshBuffer::getAttribute(out, buf, _outType);
			return out;
		};
	}

	_NBL_DEBUG_BREAK_IF(!quantFunc)
	if (!quantFunc)
		return false;

	for (const core::vectorSIMDf& d : _srcData)
	{
		const core::vectorSIMDf quantized = quantFunc(d, _srcType.type, _dstType.type, _cache);
        if (!compareFloatingPointAttribute(d, quantized, getFormatChannelCount(_srcType.type), _errMetric))
            return false;
	}

	return true;
}

core::smart_refctd_ptr<ICPUBuffer> IMeshManipulator::idxBufferFromTriangleStripsToTriangles(const void* _input, size_t _idxCount, E_INDEX_TYPE _idxType)
{
	if (_idxType == EIT_16BIT)
		return CMeshManipulator::triangleStripsToTriangles<uint16_t>(_input, _idxCount);
	else if (_idxType == EIT_32BIT)
		return CMeshManipulator::triangleStripsToTriangles<uint32_t>(_input, _idxCount);
	return nullptr;
}

core::smart_refctd_ptr<ICPUBuffer> IMeshManipulator::idxBufferFromTrianglesFanToTriangles(const void* _input, size_t _idxCount, E_INDEX_TYPE _idxType)
{
	if (_idxType == EIT_16BIT)
		return CMeshManipulator::trianglesFanToTriangles<uint16_t>(_input, _idxCount);
	else if (_idxType == EIT_32BIT)
		return CMeshManipulator::trianglesFanToTriangles<uint32_t>(_input, _idxCount);
	return nullptr;
}

} // end namespace scene
} // end namespace irr


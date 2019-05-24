// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CMeshManipulator.h"

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "irr/video/SGPUMesh.h"
#include "irr/video/IGPUMeshBuffer.h"
#include "os.h"
#include "irr/asset/CForsythVertexCacheOptimizer.h"
#include "irr/asset/COverdrawMeshOptimizer.h"
#include "irr/asset/ICPUSkinnedMeshBuffer.h"
#include "irr/asset/CSmoothNormalGenerator.h"

namespace irr
{
namespace asset
{

// declared as extern in SVertexManipulator.h
core::vector<QuantizationCacheEntry2_10_10_10> normalCacheFor2_10_10_10Quant;
core::vector<QuantizationCacheEntry8_8_8> normalCacheFor8_8_8Quant;
core::vector<QuantizationCacheEntry16_16_16> normalCacheFor16_16_16Quant;
core::vector<QuantizationCacheEntryHalfFloat> normalCacheForHalfFloatQuant;


//! Flips the direction of surfaces. Changes backfacing triangles to frontfacing
//! triangles and vice versa.
//! \param mesh: Mesh on which the operation is performed.
void CMeshManipulator::flipSurfaces(asset::ICPUMeshBuffer* inbuffer) const
{
	if (!inbuffer)
		return;

    const uint32_t idxcnt = inbuffer->getIndexCount();
    if (!inbuffer->getIndices())
        return;


    if (inbuffer->getIndexType() == asset::EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case asset::EPT_TRIANGLE_FAN:
            for (uint32_t i=1; i<idxcnt; i+=2)
            {
                const uint16_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case asset::EPT_TRIANGLE_STRIP:
            if (idxcnt%2) //odd
            {
                for (uint32_t i=0; i<(idxcnt>>1); i++)
                {
                    const uint16_t tmp = idx[i];
                    idx[i] = idx[idxcnt-1-i];
                    idx[idxcnt-1-i] = tmp;
                }
            }
            else //even
            {
                asset::ICPUBuffer* newIndexBuffer = new asset::ICPUBuffer(idxcnt*2+2);
                ((uint16_t*)newIndexBuffer->getPointer())[0] = idx[0];
                memcpy(((uint16_t*)newIndexBuffer->getPointer())+1,idx,idxcnt*2);
                inbuffer->setIndexCount(idxcnt+1);
                inbuffer->setIndexBufferOffset(0);
                inbuffer->getMeshDataAndFormat()->setIndexBuffer(newIndexBuffer);
                newIndexBuffer->drop();
            }
            break;
        case asset::EPT_TRIANGLES:
            for (uint32_t i=0; i<idxcnt; i+=3)
            {
                const uint16_t tmp = idx[i+1];
                idx[i+1] = idx[i+2];
                idx[i+2] = tmp;
            }
            break;
        default: break;
        }
    }
    else if (inbuffer->getIndexType() == asset::EIT_32BIT)
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case asset::EPT_TRIANGLE_FAN:
            for (uint32_t i=1; i<idxcnt; i+=2)
            {
                const uint32_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case asset::EPT_TRIANGLE_STRIP:
            if (idxcnt%2) //odd
            {
                for (uint32_t i=0; i<(idxcnt>>1); i++)
                {
                    const uint32_t tmp = idx[i];
                    idx[i] = idx[idxcnt-1-i];
                    idx[idxcnt-1-i] = tmp;
                }
            }
            else //even
            {
                asset::ICPUBuffer* newIndexBuffer = new asset::ICPUBuffer(idxcnt*4+4);
                ((uint32_t*)newIndexBuffer->getPointer())[0] = idx[0];
                memcpy(((uint32_t*)newIndexBuffer->getPointer())+1,idx,idxcnt*4);
                inbuffer->setIndexCount(idxcnt+1);
                inbuffer->setIndexBufferOffset(0);
                inbuffer->getMeshDataAndFormat()->setIndexBuffer(newIndexBuffer);
                newIndexBuffer->drop();
            }
            break;
        case asset::EPT_TRIANGLES:
            for (uint32_t i=0; i<idxcnt; i+=3)
            {
                const uint32_t tmp = idx[i+1];
                idx[i+1] = idx[i+2];
                idx[i+2] = tmp;
            }
            break;
        default: break;
        }
    }
}

asset::ICPUMeshBuffer* CMeshManipulator::createMeshBufferFetchOptimized(const asset::ICPUMeshBuffer* _inbuffer) const
{
	if (!_inbuffer || !_inbuffer->getMeshDataAndFormat() || !_inbuffer->getIndices())
		return NULL;

	asset::ICPUMeshBuffer* outbuffer = createMeshBufferDuplicate(_inbuffer);
    asset::IMeshDataFormatDesc<asset::ICPUBuffer>* outDesc = outbuffer->getMeshDataAndFormat();

	// Find vertex count
	size_t vertexCount = _inbuffer->calcVertexCount();
	const void* ind = _inbuffer->getIndices();

	core::unordered_set<const asset::ICPUBuffer*> buffers;
	for (size_t i = 0; i < asset::EVAI_COUNT; ++i)
		buffers.insert(outDesc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i));

	size_t offsets[asset::EVAI_COUNT];
	memset(offsets, -1, sizeof(offsets));
	asset::E_FORMAT types[asset::EVAI_COUNT];
	if (buffers.size() != 1)
	{
		size_t lastOffset = 0u;
		size_t lastSize = 0u;
		for (size_t i = 0; i < asset::EVAI_COUNT; ++i)
		{
			if (outDesc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i))
			{
				types[i] = outDesc->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)i);

                const uint32_t typeSz = asset::getTexelOrBlockSize(types[i]);
                const size_t alignment = (typeSz/asset::getFormatChannelCount(types[i]) == 8u) ? 8ull : 4ull; // if format 64bit per channel, than align to 8

				offsets[i] = lastOffset + lastSize;
				const size_t mod = offsets[i] % alignment;
				offsets[i] += mod;

				lastOffset = offsets[i];
                lastSize = typeSz;
			}
		}
		const size_t vertexSize = lastOffset + lastSize;

		asset::ICPUBuffer* newVertBuffer = new asset::ICPUBuffer(vertexCount*vertexSize);
		for (size_t i = 0; i < asset::EVAI_COUNT; ++i)
		{
			if (offsets[i] < 0xffffffff)
			{
				outDesc->setVertexAttrBuffer(newVertBuffer, (asset::E_VERTEX_ATTRIBUTE_ID)i, types[i], vertexSize, offsets[i]);
			}
		}
	}
	outbuffer->setBaseVertex(0);

	core::vector<asset::E_VERTEX_ATTRIBUTE_ID> activeAttribs;
	for (size_t i = 0; i < asset::EVAI_COUNT; ++i)
		if (outDesc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i))
			activeAttribs.push_back((asset::E_VERTEX_ATTRIBUTE_ID)i);

	uint32_t* remapBuffer = (uint32_t*)_IRR_ALIGNED_MALLOC(vertexCount*4,_IRR_SIMD_ALIGNMENT);
	memset(remapBuffer, 0xffffffffu, vertexCount*4);

	const asset::E_INDEX_TYPE idxType = outbuffer->getIndexType();
	void* indices = outbuffer->getIndices();
	size_t nextVert = 0u;

	for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
	{
		const uint32_t index = idxType == asset::EIT_32BIT ? ((uint32_t*)indices)[i] : ((uint16_t*)indices)[i];

		uint32_t& remap = remapBuffer[index];

		if (remap == 0xffffffffu)
		{
			for (size_t j = 0; j < activeAttribs.size(); ++j)
			{
				asset::E_FORMAT type = types[activeAttribs[j]];

                if (!asset::isNormalizedFormat(type) && (asset::isIntegerFormat(type) || asset::isScaledFormat(type)))
				{
					uint32_t dst[4];
					_inbuffer->getAttribute(dst, (asset::E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], index);
					outbuffer->setAttribute(dst, (asset::E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], nextVert);
				}
				else
				{
					core::vectorSIMDf dst;
					_inbuffer->getAttribute(dst, (asset::E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], index);
					outbuffer->setAttribute(dst, (asset::E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], nextVert);
				}
			}

			remap = nextVert++;
		}

		if (idxType == asset::EIT_32BIT)
			((uint32_t*)indices)[i] = remap;
		else
			((uint16_t*)indices)[i] = remap;
	}

	_IRR_ALIGNED_FREE(remapBuffer);

	_IRR_DEBUG_BREAK_IF(nextVert > vertexCount)

	return outbuffer;
}

//! Creates a copy of the mesh, which will only consist of unique primitives
asset::ICPUMeshBuffer* CMeshManipulator::createMeshBufferUniquePrimitives(asset::ICPUMeshBuffer* inbuffer) const
{
	if (!inbuffer)
		return 0;
    asset::IMeshDataFormatDesc<asset::ICPUBuffer>* oldDesc = inbuffer->getMeshDataAndFormat();
    if (!oldDesc)
        return 0;

    if (!inbuffer->getIndices())
    {
        inbuffer->grab();
        return inbuffer;
    }
    const uint32_t idxCnt = inbuffer->getIndexCount();
    asset::ICPUMeshBuffer* clone = new asset::ICPUMeshBuffer();
    clone->setBoundingBox(inbuffer->getBoundingBox());
    clone->setIndexCount(idxCnt);
    const asset::E_PRIMITIVE_TYPE ept = inbuffer->getPrimitiveType();
    clone->setPrimitiveType(ept);
    clone->getMaterial() = inbuffer->getMaterial();

    asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();
    clone->setMeshDataAndFormat(desc);
    desc->drop();

    size_t stride = 0;
    int32_t offset[asset::EVAI_COUNT];
    size_t newAttribSizes[asset::EVAI_COUNT];
    uint8_t* sourceBuffers[asset::EVAI_COUNT] = {NULL};
    size_t sourceBufferStrides[asset::EVAI_COUNT];
    for (size_t i=0; i< asset::EVAI_COUNT; i++)
    {
        const asset::ICPUBuffer* vbuf = oldDesc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i);
        if (vbuf)
        {
            offset[i] = stride;
            newAttribSizes[i] = asset::getTexelOrBlockSize(oldDesc->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)i));
            stride += newAttribSizes[i];
            if (stride>=0xdeadbeefu)
            {
                clone->drop();
                return 0;
            }
            sourceBuffers[i] = (uint8_t*)vbuf->getPointer();
            sourceBuffers[i] += oldDesc->getMappedBufferOffset((asset::E_VERTEX_ATTRIBUTE_ID)i);
            sourceBufferStrides[i] = oldDesc->getMappedBufferStride((asset::E_VERTEX_ATTRIBUTE_ID)i);
        }
        else
            offset[i] = -1;
    }

    asset::ICPUBuffer* vertexBuffer = new asset::ICPUBuffer(stride*idxCnt);
    for (size_t i=0; i<asset::EVAI_COUNT; i++)
    {
        if (offset[i]>=0)
            desc->setVertexAttrBuffer(vertexBuffer,(asset::E_VERTEX_ATTRIBUTE_ID)i,oldDesc->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)i),stride,offset[i]);
    }
    vertexBuffer->drop();

    uint8_t* destPointer = (uint8_t*)vertexBuffer->getPointer();
    if (inbuffer->getIndexType()==asset::EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        for (uint64_t i=0; i<idxCnt; i++,idx++)
        for (size_t j=0; j<asset::EVAI_COUNT; j++)
        {
            if (offset[j]<0)
                continue;

            memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
            destPointer += newAttribSizes[j];
        }
    }
    else if (inbuffer->getIndexType()==asset::EIT_32BIT)
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
        for (uint64_t i=0; i<idxCnt; i++,idx++)
        for (size_t j=0; j<asset::EVAI_COUNT; j++)
        {
            if (offset[j]<0)
                continue;

            memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
            destPointer += newAttribSizes[j];
        }
    }

	return clone;
}

//
asset::ICPUMeshBuffer* CMeshManipulator::calculateSmoothNormals(asset::ICPUMeshBuffer* inbuffer, bool makeNewMesh, float epsilon,
	asset::E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp) const
{
	if (inbuffer == nullptr)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;
	}

	//Mesh has to have unique primitives
	if (inbuffer->getIndexType() != asset::E_INDEX_TYPE::EIT_UNKNOWN)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;
	}

	asset::ICPUMeshBuffer* outbuffer = (makeNewMesh == true) ? createMeshBufferDuplicate(inbuffer) : inbuffer;
	CSmoothNormalGenerator::calculateNormals(outbuffer, epsilon, normalAttrID, vxcmp);

	return outbuffer;
}

// Used by createMeshBufferWelded only
static bool cmpVertices(asset::ICPUMeshBuffer* _inbuf, const void* _va, const void* _vb, size_t _vsize, const IMeshManipulator::SErrorMetric* _errMetrics, const IMeshManipulator* _meshManip)
{
    auto cmpInteger = [](uint32_t* _a, uint32_t* _b, size_t _n) -> bool {
        return !memcmp(_a, _b, _n*4);
    };

    const uint8_t* va = (uint8_t*)_va, *vb = (uint8_t*)_vb;
    auto desc = _inbuf->getMeshDataAndFormat();
    for (size_t i = 0u; i < asset::EVAI_COUNT; ++i)
    {
        if (!desc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i))
            continue;

        const auto atype = desc->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)i);
        const auto cpa = asset::getFormatChannelCount(atype);

        if (asset::isIntegerFormat(atype) || asset::isScaledFormat(atype))
        {
            uint32_t attr[8];
            asset::ICPUMeshBuffer::getAttribute(attr, va, atype);
            asset::ICPUMeshBuffer::getAttribute(attr+4, vb, atype);
            if (!cmpInteger(attr, attr+4, cpa))
                return false;
        }
        else
        {
            core::vectorSIMDf attr[2];
            asset::ICPUMeshBuffer::getAttribute(attr[0], va, atype);
            asset::ICPUMeshBuffer::getAttribute(attr[1], vb, atype);
            if (!_meshManip->compareFloatingPointAttribute(attr[0], attr[1], cpa, _errMetrics[i]))
                return false;
        }

        const uint32_t sz = asset::getTexelOrBlockSize(atype);
        va += sz;
        vb += sz;
    }

    return true;
}

//! Creates a copy of a mesh, which will have identical vertices welded together
asset::ICPUMeshBuffer* CMeshManipulator::createMeshBufferWelded(asset::ICPUMeshBuffer *inbuffer, const SErrorMetric* _errMetrics, const bool& optimIndexType, const bool& makeNewMesh) const
{
    if (!inbuffer)
        return nullptr;
    asset::IMeshDataFormatDesc<asset::ICPUBuffer>* oldDesc = inbuffer->getMeshDataAndFormat();
    if (!oldDesc)
        return nullptr;

    bool bufferPresent[asset::EVAI_COUNT];

    size_t vertexAttrSize[asset::EVAI_COUNT];
    size_t vertexSize = 0;
    for (size_t i=0; i<asset::EVAI_COUNT; i++)
    {
        const asset::ICPUBuffer* buf = oldDesc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i);
        bufferPresent[i] = buf;
        if (buf)
        {
            const asset::E_FORMAT componentType = oldDesc->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)i);
            vertexAttrSize[i] = asset::getTexelOrBlockSize(componentType);
            vertexSize += vertexAttrSize[i];
        }
    }

    auto cmpfunc = [&, inbuffer, this, vertexSize, _errMetrics](const void* _va, const void* _vb) {
        return cmpVertices(inbuffer, _va, _vb, vertexSize, _errMetrics, this);
    };

    size_t vertexCount = inbuffer->calcVertexCount();
    asset::E_INDEX_TYPE oldIndexType = inbuffer->getIndexType();

    if (vertexCount==0)
    {
        return nullptr;
    }

    // reset redirect list
    uint32_t* redirects = new uint32_t[vertexCount];

    uint32_t maxRedirect = 0;

    uint8_t* epicData = (uint8_t*)_IRR_ALIGNED_MALLOC(vertexSize*vertexCount,_IRR_SIMD_ALIGNMENT);
    for (size_t i=0; i < vertexCount; i++)
    {
        uint8_t* currentVertexPtr = epicData+i*vertexSize;
        for (size_t k=0; k<asset::EVAI_COUNT; k++)
        {
            if (!bufferPresent[k])
                continue;

            size_t stride = oldDesc->getMappedBufferStride((asset::E_VERTEX_ATTRIBUTE_ID)k);
            void* sourcePtr = inbuffer->getAttribPointer((asset::E_VERTEX_ATTRIBUTE_ID)k)+i*stride;
            memcpy(currentVertexPtr,sourcePtr,vertexAttrSize[k]);
            currentVertexPtr += vertexAttrSize[k];
        }
    }

    for (size_t i=0; i<vertexCount; i++)
    {
        uint32_t redir = i;

        for (size_t j = 0u; j < vertexCount; ++j)
        {
            if (i == j)
                continue;
            if (cmpfunc(epicData+vertexSize*i, epicData+vertexSize*j))
            {
                redir = j;
                break;
            }
        }

        redirects[i] = redir;
        if (redir>maxRedirect)
            maxRedirect = redir;
    }
    _IRR_ALIGNED_FREE(epicData);

    void* oldIndices = inbuffer->getIndices();
    asset::ICPUMeshBuffer* clone = nullptr;
    if (makeNewMesh)
    {
        clone = createMeshBufferDuplicate(inbuffer);
    }
    else
    {
        if (!oldDesc->getIndexBuffer())
        {
            asset::ICPUBuffer* indexCpy = new asset::ICPUBuffer((maxRedirect>=0x10000u ? 4:2)*inbuffer->getIndexCount());
            oldDesc->setIndexBuffer(indexCpy);
            indexCpy->drop();
        }
    }


    if (oldIndexType==asset::EIT_16BIT)
    {
        uint16_t* indicesIn = reinterpret_cast<uint16_t*>(oldIndices);
        if ((makeNewMesh ? clone:inbuffer)->getIndexType()==asset::EIT_32BIT)
        {
            uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
        else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==asset::EIT_16BIT)
        {
            uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
    }
    else if (oldIndexType==asset::EIT_32BIT)
    {
        uint32_t* indicesIn = reinterpret_cast<uint32_t*>(oldIndices);
        if ((makeNewMesh ? clone:inbuffer)->getIndexType()==asset::EIT_32BIT)
        {
            uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
        else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==asset::EIT_16BIT)
        {
            uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
    }
    else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==asset::EIT_32BIT)
    {
        uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
        for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            indicesOut[i] = redirects[i];
    }
    else if ((makeNewMesh ? clone:inbuffer)->getIndexType()==asset::EIT_16BIT)
    {
        uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone:inbuffer)->getIndices());
        for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            indicesOut[i] = redirects[i];
    }
    delete [] redirects;

    if (makeNewMesh)
        return clone;
    else
        return inbuffer;
}

asset::ICPUMeshBuffer* CMeshManipulator::createOptimizedMeshBuffer(const asset::ICPUMeshBuffer* _inbuffer, const SErrorMetric* _errMetric) const
{
	if (!_inbuffer)
		return NULL;
	asset::ICPUMeshBuffer* outbuffer = createMeshBufferDuplicate(_inbuffer);
	if (!outbuffer->getMeshDataAndFormat())
		return outbuffer;

	// Find vertex count
	size_t vertexCount = outbuffer->calcVertexCount();

	// make index buffer 0,1,2,3,4,... if nothing's mapped
	if (!outbuffer->getIndices())
	{
		asset::ICPUBuffer* ib = new asset::ICPUBuffer(vertexCount * 4);
		asset::IMeshDataFormatDesc<asset::ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		uint32_t* indices = (uint32_t*)ib->getPointer();
		for (uint32_t i = 0; i < vertexCount; ++i)
			indices[i] = i;
		newDesc->setIndexBuffer(ib);
		ib->drop();
		outbuffer->setIndexCount(vertexCount);
		outbuffer->setIndexType(asset::EIT_32BIT);
	}

	// make 32bit index buffer if 16bit one is present
	if (outbuffer->getIndexType() == asset::EIT_16BIT)
	{
        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		asset::ICPUBuffer* newIb = create32BitFrom16BitIdxBufferSubrange((uint16_t*)outbuffer->getIndices(), outbuffer->getIndexCount());
		newDesc->setIndexBuffer(newIb);
		// no need to set index buffer offset to 0 because it already is
		outbuffer->setIndexType(asset::EIT_32BIT);
	}

	// convert index buffer for triangle primitives
	if (outbuffer->getPrimitiveType() == asset::EPT_TRIANGLE_FAN)
	{
        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		const asset::ICPUBuffer* ib = newDesc->getIndexBuffer();
		asset::ICPUBuffer* newIb = idxBufferFromTrianglesFanToTriangles(outbuffer->getIndices(), outbuffer->getIndexCount(), asset::EIT_32BIT);
		newDesc->setIndexBuffer(newIb);
		outbuffer->setPrimitiveType(asset::EPT_TRIANGLES);
		outbuffer->setIndexCount(newIb->getSize() / 4);
	}
	else if (outbuffer->getPrimitiveType() == asset::EPT_TRIANGLE_STRIP)
	{
        asset::IMeshDataFormatDesc<asset::ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		asset::ICPUBuffer* newIb = idxBufferFromTriangleStripsToTriangles(outbuffer->getIndices(), outbuffer->getIndexCount(), asset::EIT_32BIT);
		newDesc->setIndexBuffer(newIb);
		outbuffer->setPrimitiveType(asset::EPT_TRIANGLES);
		outbuffer->setIndexCount(newIb->getSize() / 4);
	}
	else if (outbuffer->getPrimitiveType() != asset::EPT_TRIANGLES)
	{
		outbuffer->drop();
		return NULL;
	}

	// STEP: weld
    createMeshBufferWelded(outbuffer, _errMetric, false, false);

    // STEP: filter invalid triangles
    filterInvalidTriangles(outbuffer);

	// STEP: overdraw optimization
	COverdrawMeshOptimizer::createOptimized(outbuffer, false);

	// STEP: Forsyth
	{
		uint32_t* indices = (uint32_t*)outbuffer->getIndices();
		CForsythVertexCacheOptimizer forsyth;
		forsyth.optimizeTriangleOrdering(vertexCount, outbuffer->getIndexCount(), indices, indices);
	}

	// STEP: prefetch optimization
	{
		asset::ICPUMeshBuffer* old = outbuffer;
		outbuffer = createMeshBufferFetchOptimized(outbuffer); // here we also get interleaved attributes (single vertex buffer)
		old->drop();
	}
	// STEP: requantization
	requantizeMeshBuffer(outbuffer, _errMetric);

	// STEP: reduce index buffer to 16bit or completely get rid of it
	{
		const void* const indices = outbuffer->getIndices();
		uint32_t* indicesCopy = (uint32_t*)_IRR_ALIGNED_MALLOC(outbuffer->getIndexCount()*4,_IRR_SIMD_ALIGNMENT);
		memcpy(indicesCopy, indices, outbuffer->getIndexCount()*4);
		std::sort(indicesCopy, indicesCopy + outbuffer->getIndexCount());

		bool continuous = true; // indices are i.e. 0,1,2,3,4,5,... (also implies indices being unique)
		bool unique = true; // indices are unique (but not necessarily continuos)

		for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
		{
			uint32_t idx = indicesCopy[i], prevIdx = 0xffffffffu;
			if (i)
			{
				prevIdx = indicesCopy[i-1];

				if (idx == prevIdx)
				{
					unique = false;
					continuous = false;
					break;
				}
				if (idx != prevIdx + 1)
					continuous = false;
			}
		}

		const uint32_t minIdx = indicesCopy[0];
		const uint32_t maxIdx = indicesCopy[outbuffer->getIndexCount() - 1];

		_IRR_ALIGNED_FREE(indicesCopy);

		asset::ICPUBuffer* newIdxBuffer = NULL;
		bool verticesMustBeReordered = false;
        asset::E_INDEX_TYPE newIdxType = asset::EIT_32BIT;

		if (!continuous)
		{
			if (unique)
			{
				// no index buffer
				// vertices have to be reordered
				verticesMustBeReordered = true;
			}
			else
			{
				if (maxIdx - minIdx <= USHRT_MAX)
					newIdxType = asset::EIT_16BIT;

				outbuffer->setIndexType(newIdxType);
				outbuffer->setBaseVertex(outbuffer->getBaseVertex() + minIdx);

				if (newIdxType == asset::EIT_16BIT)
				{
					newIdxBuffer = new asset::ICPUBuffer(outbuffer->getIndexCount()*2);
					// no need to change index buffer offset because it's always 0 (after duplicating original mesh)
					for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
						((uint16_t*)newIdxBuffer->getPointer())[i] = ((uint32_t*)indices)[i] - minIdx;
				}
			}
		}
		else
		{
			outbuffer->setBaseVertex(outbuffer->getBaseVertex()+minIdx);
		}

		if (newIdxBuffer)
		{
			outbuffer->getMeshDataAndFormat()->setIndexBuffer(newIdxBuffer);
			newIdxBuffer->drop();
		}


		if (verticesMustBeReordered)
		{
			// reorder vertices according to index buffer
#define _ACCESS_IDX(n) ((newIdxType == asset::EIT_32BIT) ? *((uint32_t*)(indices)+(n)) : *((uint16_t*)(indices)+(n)))

			const size_t vertexSize = outbuffer->getMeshDataAndFormat()->getMappedBufferStride(outbuffer->getPositionAttributeIx());
			uint8_t* const v = (uint8_t*)(outbuffer->getMeshDataAndFormat()->getMappedBuffer(outbuffer->getPositionAttributeIx())->getPointer()); // after prefetch optim. we have guarantee of single vertex buffer so we can do like this
			uint8_t* const vCopy = (uint8_t*)_IRR_ALIGNED_MALLOC(outbuffer->getMeshDataAndFormat()->getMappedBuffer(outbuffer->getPositionAttributeIx())->getSize(),_IRR_SIMD_ALIGNMENT);
			memcpy(vCopy, v, outbuffer->getMeshDataAndFormat()->getMappedBuffer(outbuffer->getPositionAttributeIx())->getSize());

			size_t baseVtx = outbuffer->getBaseVertex();
			for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
			{
				const uint32_t idx = _ACCESS_IDX(i+baseVtx);
				if (idx != i+baseVtx)
					memcpy(v + (vertexSize*(i + baseVtx)), vCopy + (vertexSize*idx), vertexSize);
			}
#undef _ACCESS_IDX
			_IRR_ALIGNED_FREE(vCopy);
		}
	}

	return outbuffer;
}

void CMeshManipulator::requantizeMeshBuffer(asset::ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric) const
{
	SAttrib newAttribs[asset::EVAI_COUNT];
	for (size_t i = 0u; i < asset::EVAI_COUNT; ++i)
		newAttribs[i].vaid = (asset::E_VERTEX_ATTRIBUTE_ID)i;

	core::unordered_map<asset::E_VERTEX_ATTRIBUTE_ID, core::vector<SIntegerAttr>> attribsI;
	core::unordered_map<asset::E_VERTEX_ATTRIBUTE_ID, core::vector<core::vectorSIMDf>> attribsF;
	for (size_t vaid = asset::EVAI_ATTR0; vaid < (size_t)asset::EVAI_COUNT; ++vaid)
	{
		const asset::E_FORMAT type = _meshbuffer->getMeshDataAndFormat()->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)vaid);

		if (_meshbuffer->getMeshDataAndFormat()->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)vaid))
		{
			if (!asset::isNormalizedFormat(type) && asset::isIntegerFormat(type))
				attribsI[(asset::E_VERTEX_ATTRIBUTE_ID)vaid] = findBetterFormatI(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].prevType, _meshbuffer, (asset::E_VERTEX_ATTRIBUTE_ID)vaid, _errMetric[vaid]);
			else
				attribsF[(asset::E_VERTEX_ATTRIBUTE_ID)vaid] = findBetterFormatF(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].prevType, _meshbuffer, (asset::E_VERTEX_ATTRIBUTE_ID)vaid, _errMetric[vaid]);
		}
	}

	const size_t activeAttributeCount = attribsI.size() + attribsF.size();

#ifdef _IRR_DEBUG
	{
		core::unordered_set<size_t> sizesSet;
		for (core::unordered_map<asset::E_VERTEX_ATTRIBUTE_ID, core::vector<SIntegerAttr>>::iterator it = attribsI.begin(); it != attribsI.end(); ++it)
			sizesSet.insert(it->second.size());
		for (core::unordered_map<asset::E_VERTEX_ATTRIBUTE_ID, core::vector<core::vectorSIMDf>>::iterator it = attribsF.begin(); it != attribsF.end(); ++it)
			sizesSet.insert(it->second.size());
		_IRR_DEBUG_BREAK_IF(sizesSet.size() != 1);
	}
#endif
	const size_t vertexCnt = (!attribsI.empty() ? attribsI.begin()->second.size() : (!attribsF.empty() ? attribsF.begin()->second.size() : 0));

	std::sort(newAttribs, newAttribs + asset::EVAI_COUNT, std::greater<SAttrib>()); // sort decreasing by size

	for (size_t i = 0u; i < activeAttributeCount; ++i)
	{
        const uint32_t typeSz = asset::getTexelOrBlockSize(newAttribs[i].type);
        const size_t alignment = (typeSz / asset::getFormatChannelCount(newAttribs[i].type) == 8u) ? 8ull : 4ull; // if format 64bit per channel, than align to 8

		newAttribs[i].offset = (i ? newAttribs[i - 1].offset + newAttribs[i - 1].size : 0u);
		const size_t mod = newAttribs[i].offset % alignment;
		newAttribs[i].offset += mod;
	}

	const size_t vertexSize = newAttribs[activeAttributeCount - 1].offset + newAttribs[activeAttributeCount - 1].size;

    asset::IMeshDataFormatDesc<asset::ICPUBuffer>* desc = _meshbuffer->getMeshDataAndFormat();
	asset::ICPUBuffer* newVertexBuffer = new asset::ICPUBuffer(vertexCnt * vertexSize);

	for (size_t i = 0u; i < activeAttributeCount; ++i)
	{
		desc->setVertexAttrBuffer(newVertexBuffer, newAttribs[i].vaid, newAttribs[i].type, vertexSize, newAttribs[i].offset);

		core::unordered_map<asset::E_VERTEX_ATTRIBUTE_ID, core::vector<SIntegerAttr>>::iterator iti = attribsI.find(newAttribs[i].vaid);
		if (iti != attribsI.end())
		{
			const core::vector<SIntegerAttr>& attrVec = iti->second;
			for (size_t ai = 0u; ai < attrVec.size(); ++ai)
			{
				const bool check = _meshbuffer->setAttribute(attrVec[ai].pointer, newAttribs[i].vaid, ai);
				_IRR_DEBUG_BREAK_IF(!check)
			}
			continue;
		}

		core::unordered_map<asset::E_VERTEX_ATTRIBUTE_ID, core::vector<core::vectorSIMDf>>::iterator itf = attribsF.find(newAttribs[i].vaid);
		if (itf != attribsF.end())
		{
			const core::vector<core::vectorSIMDf>& attrVec = itf->second;
			for (size_t ai = 0u; ai < attrVec.size(); ++ai)
			{
				const bool check = _meshbuffer->setAttribute(attrVec[ai], newAttribs[i].vaid, ai);
				_IRR_DEBUG_BREAK_IF(!check)
			}
		}
	}

	newVertexBuffer->drop();
}


template<>
void CMeshManipulator::copyMeshBufferMemberVars<asset::ICPUMeshBuffer>(asset::ICPUMeshBuffer* _dst, const asset::ICPUMeshBuffer* _src) const
{
    _dst->setBaseInstance(
        _src->getBaseInstance()
    );
    _dst->setBaseVertex(
        _src->getBaseVertex()
    );
    _dst->setIndexBufferOffset(
        _src->getIndexBufferOffset()
    );
    _dst->setBoundingBox(
        _src->getBoundingBox()
    );
    _dst->setIndexCount(
        _src->getIndexCount()
    );
    _dst->setIndexType(
        _src->getIndexType()
    );
    _dst->setInstanceCount(
        _src->getInstanceCount()
    );
    _dst->setPrimitiveType(
        _src->getPrimitiveType()
    );
    _dst->setPositionAttributeIx(
        _src->getPositionAttributeIx()
    );
    _dst->getMaterial() = _src->getMaterial();
}
template<>
void CMeshManipulator::copyMeshBufferMemberVars<asset::ICPUSkinnedMeshBuffer>(asset::ICPUSkinnedMeshBuffer* _dst, const asset::ICPUSkinnedMeshBuffer* _src) const
{
    copyMeshBufferMemberVars<asset::ICPUMeshBuffer>(_dst, _src);
    _dst->setIndexRange(
        _src->getIndexMinBound(),
        _src->getIndexMaxBound()
    );
    _dst->setMaxVertexBoneInfluences(
        _src->getMaxVertexBoneInfluences()
    );
}

asset::ICPUMeshBuffer* CMeshManipulator::createMeshBufferDuplicate(const asset::ICPUMeshBuffer* _src) const
{
	if (!_src)
		return NULL;

	asset::ICPUMeshBuffer* dst = NULL;
    if (const asset::ICPUSkinnedMeshBuffer* smb = dynamic_cast<const asset::ICPUSkinnedMeshBuffer*>(_src)) // we can do other checks for meshbuffer type than dynamic_cast // how then?
    {
        dst = new asset::ICPUSkinnedMeshBuffer();
        copyMeshBufferMemberVars(static_cast<asset::ICPUSkinnedMeshBuffer*>(dst), smb);
    }
    else
    {
        dst = new asset::ICPUMeshBuffer();
        copyMeshBufferMemberVars(dst, _src);
    }

	if (!_src->getMeshDataAndFormat())
		return dst;

	asset::ICPUBuffer* idxBuffer = NULL;
	if (_src->getIndices())
	{
		idxBuffer = new asset::ICPUBuffer((_src->getIndexType() == asset::EIT_16BIT ? 2 : 4) * _src->getIndexCount());
		memcpy(idxBuffer->getPointer(), _src->getIndices(), idxBuffer->getSize());
		dst->setIndexBufferOffset(0);
	}

    asset::ICPUMeshDataFormatDesc* newDesc = new asset::ICPUMeshDataFormatDesc();
	const asset::IMeshDataFormatDesc<asset::ICPUBuffer>* oldDesc = _src->getMeshDataAndFormat();

	core::unordered_map<const asset::ICPUBuffer*, asset::E_VERTEX_ATTRIBUTE_ID> oldBuffers;
	core::vector<asset::ICPUBuffer*> newBuffers;
	for (size_t i = 0; i < asset::EVAI_COUNT; ++i)
	{
		const asset::ICPUBuffer* oldBuf = oldDesc->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i);
		if (!oldBuf)
			continue;
		asset::ICPUBuffer* newBuf = NULL;

		core::unordered_map<const asset::ICPUBuffer*, asset::E_VERTEX_ATTRIBUTE_ID>::iterator itr = oldBuffers.find(oldBuf);
		if (itr == oldBuffers.end())
		{
			oldBuffers[oldBuf] = (asset::E_VERTEX_ATTRIBUTE_ID)i;
			newBuf = new asset::ICPUBuffer(oldBuf->getSize());
			memcpy(newBuf->getPointer(), oldBuf->getPointer(), newBuf->getSize());
			newBuffers.push_back(newBuf);
		}
		else
		{
			newBuf = const_cast<asset::ICPUBuffer*>(newDesc->getMappedBuffer(itr->second));
		}

		newDesc->setVertexAttrBuffer(newBuf, (asset::E_VERTEX_ATTRIBUTE_ID)i, oldDesc->getAttribFormat((asset::E_VERTEX_ATTRIBUTE_ID)i),
			oldDesc->getMappedBufferStride((asset::E_VERTEX_ATTRIBUTE_ID)i), oldDesc->getMappedBufferOffset((asset::E_VERTEX_ATTRIBUTE_ID)i), oldDesc->getAttribDivisor((asset::E_VERTEX_ATTRIBUTE_ID)i));
	}
	if (idxBuffer)
	{
		newDesc->setIndexBuffer(idxBuffer);
		idxBuffer->drop();
	}
	for (size_t i = 0; i < newBuffers.size(); ++i)
		newBuffers[i]->drop();

	dst->setMeshDataAndFormat(newDesc);
    newDesc->drop();

	return dst;
}

void CMeshManipulator::filterInvalidTriangles(asset::ICPUMeshBuffer* _input) const
{
    if (!_input || !_input->getMeshDataAndFormat() || !_input->getIndices())
        return;

    switch (_input->getIndexType())
    {
    case asset::EIT_16BIT:
        return priv_filterInvalidTriangles<uint16_t>(_input);
    case asset::EIT_32BIT:
        return priv_filterInvalidTriangles<uint32_t>(_input);
    default: return;
    }
}

template<typename IdxT>
void CMeshManipulator::priv_filterInvalidTriangles(asset::ICPUMeshBuffer* _input) const
{
    const size_t size = _input->getIndexCount() * sizeof(IdxT);
    void* const copy = _IRR_ALIGNED_MALLOC(size,_IRR_SIMD_ALIGNMENT);
    memcpy(copy, _input->getIndices(), size);

    struct Triangle
    {
        IdxT i[3];
    } *const begin = (Triangle*)copy, *const end = (Triangle*)((uint8_t*)copy + size);

    Triangle* const newEnd = std::remove_if(begin, end,
        [&_input](const Triangle& _t) {
            core::vectorSIMDf p0, p1, p2;
            const asset::E_VERTEX_ATTRIBUTE_ID pvaid = _input->getPositionAttributeIx();
            uint32_t m = 0xffffffff;
            const core::vectorSIMDu32 mask(m, m, m, 0);
            _input->getAttribute(p0, pvaid, _t.i[0]);
            _input->getAttribute(p1, pvaid, _t.i[1]);
            _input->getAttribute(p2, pvaid, _t.i[2]);
            p0 &= mask; p1 &= mask; p2 &= mask;
            return (p0 == p1).all() || (p0 == p2).all() || (p1 == p2).all();
    });
    const size_t newSize = std::distance(begin, newEnd) * sizeof(Triangle);

    auto newBuf = new asset::ICPUBuffer(newSize);
    memcpy(newBuf->getPointer(), copy, newSize);
    _IRR_ALIGNED_FREE(copy);
    _input->getMeshDataAndFormat()->setIndexBuffer(newBuf);
    _input->setIndexBufferOffset(0);
    _input->setIndexCount(newSize/sizeof(IdxT));
    newBuf->drop();
}
template void CMeshManipulator::priv_filterInvalidTriangles<uint16_t>(asset::ICPUMeshBuffer* _input) const;
template void CMeshManipulator::priv_filterInvalidTriangles<uint32_t>(asset::ICPUMeshBuffer* _input) const;

asset::ICPUBuffer* CMeshManipulator::create32BitFrom16BitIdxBufferSubrange(const uint16_t* _in, size_t _idxCount) const
{
	if (!_in)
		return NULL;

	asset::ICPUBuffer* out = new asset::ICPUBuffer(_idxCount * 4);

	uint32_t* outPtr = (uint32_t*)out->getPointer();

	for (size_t i = 0; i < _idxCount; ++i)
		outPtr[i] = _in[i];

	return out;
}

core::vector<core::vectorSIMDf> CMeshManipulator::findBetterFormatF(asset::E_FORMAT* _outType, size_t* _outSize, asset::E_FORMAT* _outPrevType, const asset::ICPUMeshBuffer* _meshbuffer, asset::E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric) const
{
	const asset::E_FORMAT thisType = _meshbuffer->getMeshDataAndFormat()->getAttribFormat(_attrId);

    if (!asset::isFloatingPointFormat(thisType) && !asset::isNormalizedFormat(thisType) && !asset::isScaledFormat(thisType))
        return {};

	core::vector<core::vectorSIMDf> attribs;

	if (!_meshbuffer->getMeshDataAndFormat())
		return attribs;

    const uint32_t cpa = asset::getFormatChannelCount(thisType);

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

	core::vector<SAttribTypeChoice> possibleTypes = findTypesOfProperRangeF(thisType, asset::getTexelOrBlockSize(thisType), min, max, _errMetric);
	std::sort(possibleTypes.begin(), possibleTypes.end(), [](const SAttribTypeChoice& t1, const SAttribTypeChoice& t2) { return asset::getTexelOrBlockSize(t1.type) < asset::getTexelOrBlockSize(t2.type); });

	*_outPrevType = thisType;
    *_outType = thisType;
    *_outSize = asset::getTexelOrBlockSize(*_outType);

	for (const SAttribTypeChoice& t : possibleTypes)
	{
		if (calcMaxQuantizationError({ thisType }, t, attribs, _errMetric))
		{
            if (asset::getTexelOrBlockSize(t.type) < asset::getTexelOrBlockSize(thisType))
            {
                *_outType = t.type;
                *_outSize = asset::getTexelOrBlockSize(*_outType);
            }

			return attribs;
		}
	}

	return attribs;
}

core::vector<CMeshManipulator::SIntegerAttr> CMeshManipulator::findBetterFormatI(asset::E_FORMAT* _outType, size_t* _outSize, asset::E_FORMAT* _outPrevType, const asset::ICPUMeshBuffer* _meshbuffer, asset::E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric) const
{
    const asset::E_FORMAT thisType = _meshbuffer->getMeshDataAndFormat()->getAttribFormat(_attrId);

    if (!asset::isIntegerFormat(thisType))
        return {};

    if (asset::isBGRALayoutFormat(thisType))
        return {}; // BGRA is supported only by a few normalized types (this is function for integer types)

	core::vector<SIntegerAttr> attribs;

	if (!_meshbuffer->getMeshDataAndFormat())
		return attribs;

    const uint32_t cpa = asset::getFormatChannelCount(thisType);

	uint32_t min[4];
	uint32_t max[4];
	if (!asset::isSignedFormat(thisType))
		for (size_t i = 0; i < 4; ++i)
			min[i] = UINT_MAX;
	else
		for (size_t i = 0; i < 4; ++i)
			min[i] = INT_MAX;
	if (!asset::isSignedFormat(thisType))
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
			if (!asset::isSignedFormat(thisType))
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
	*_outSize = asset::getTexelOrBlockSize(thisType);
	*_outPrevType = thisType;

	if (_errMetric.method == EEM_ANGLES) // native integers normals does not change
		return attribs;

	*_outType = getBestTypeI(thisType, _outSize, min, max);
    if (asset::getTexelOrBlockSize(*_outType) >= asset::getTexelOrBlockSize(thisType))
    {
        *_outType = thisType;
        *_outSize = asset::getTexelOrBlockSize(thisType);
    }
	return attribs;
}

asset::E_FORMAT CMeshManipulator::getBestTypeI(asset::E_FORMAT _originalType, size_t* _outSize, const uint32_t* _min, const uint32_t* _max) const
{
    using namespace video;

    const bool isNativeInteger = isIntegerFormat(_originalType);
    const bool isUnsigned = !isSignedFormat(_originalType);

    const uint32_t originalCpa = getFormatChannelCount(_originalType);

    core::vector<asset::E_FORMAT> nativeInts{
        asset::EF_R8G8_UINT,
        asset::EF_R8G8_SINT,
        asset::EF_R8G8B8_UINT,
        asset::EF_R8G8B8_SINT,
        asset::EF_R8G8B8A8_UINT,
        asset::EF_R8G8B8A8_SINT,
        asset::EF_A2B10G10R10_UINT_PACK32,
        asset::EF_A2B10G10R10_SINT_PACK32,
        asset::EF_R16_UINT,
        asset::EF_R16_SINT,
        asset::EF_R16G16_UINT,
        asset::EF_R16G16_SINT,
        asset::EF_R16G16B16_UINT,
        asset::EF_R16G16B16_SINT,
        asset::EF_R16G16B16A16_UINT,
        asset::EF_R16G16B16A16_SINT,
        asset::EF_R32_UINT,
        asset::EF_R32_SINT,
        asset::EF_R32G32_UINT,
        asset::EF_R32G32_SINT,
        asset::EF_R32G32B32_UINT,
        asset::EF_R32G32B32_SINT,
        asset::EF_R32G32B32A32_UINT,
        asset::EF_R32G32B32A32_SINT
    };
    core::vector<asset::E_FORMAT> scaledInts{
        asset::EF_R8G8_USCALED,
        asset::EF_R8G8_SSCALED,
        asset::EF_R8G8B8_USCALED,
        asset::EF_R8G8B8_SSCALED,
        asset::EF_R8G8B8A8_USCALED,
        asset::EF_R8G8B8A8_SSCALED,
        asset::EF_A2B10G10R10_USCALED_PACK32,
        asset::EF_A2B10G10R10_SSCALED_PACK32,
        asset::EF_R16_USCALED,
        asset::EF_R16_SSCALED,
        asset::EF_R16G16_USCALED,
        asset::EF_R16G16_SSCALED,
        asset::EF_R16G16B16_USCALED,
        asset::EF_R16G16B16_SSCALED,
        asset::EF_R16G16B16A16_USCALED,
        asset::EF_R16G16B16A16_SSCALED
    };

    core::vector<asset::E_FORMAT>& all = isNativeInteger ? nativeInts : scaledInts;
    if (originalCpa > 1u)
    {
        all.erase(
            std::remove_if(all.begin(), all.end(),
                [originalCpa](asset::E_FORMAT fmt) { return getFormatChannelCount(fmt) < originalCpa; }
            ),
            all.end()
        );
    }

    auto minValueOfTypeINT = [](asset::E_FORMAT _fmt, uint32_t _cmpntNum) -> int32_t {
        if (!isSignedFormat(_fmt))
            return 0;

        switch (_fmt)
        {
        case asset::EF_A2B10G10R10_SSCALED_PACK32:
        case asset::EF_A2B10G10R10_SINT_PACK32:
            if (_cmpntNum < 3u)
                return -512;
            else return -2;
            break;
        default:
        {
        const uint32_t bitsPerCh = getTexelOrBlockSize(_fmt)/getFormatChannelCount(_fmt);
        return int32_t(-uint64_t(1ull<<(bitsPerCh-1u)));
        }
        }
    };
    auto maxValueOfTypeINT = [](asset::E_FORMAT _fmt, uint32_t _cmpntNum) -> uint32_t {
        switch (_fmt)
        {
        case asset::EF_A2B10G10R10_USCALED_PACK32:
        case asset::EF_A2B10G10R10_UINT_PACK32:
            if (_cmpntNum < 3u)
                return 1023u;
            else return 3u;
            break;
        case asset::EF_A2B10G10R10_SSCALED_PACK32:
        case asset::EF_A2B10G10R10_SINT_PACK32:
            if (_cmpntNum < 3u)
                return 511u;
            else return 1u;
            break;
        default:
        {
            const uint32_t bitsPerCh = getTexelOrBlockSize(_fmt)/getFormatChannelCount(_fmt);
            const uint64_t r = (1ull<<bitsPerCh)-1ull;
            if (!isSignedFormat(_fmt))
                return (uint32_t)r;
            return (uint32_t)(r>>1);
        }
        }
    };

    asset::E_FORMAT bestType = _originalType;
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
        if (ok && getTexelOrBlockSize(*it) < getTexelOrBlockSize(bestType)) // vertexAttrSize array defined in IMeshBuffer.h
        {
            bestType = *it;
            *_outSize = getTexelOrBlockSize(bestType);
        }
    }

    return bestType;
}

core::vector<CMeshManipulator::SAttribTypeChoice> CMeshManipulator::findTypesOfProperRangeF(asset::E_FORMAT _type, size_t _sizeThreshold, const float * _min, const float * _max, const SErrorMetric& _errMetric) const
{
    using namespace video;

    core::vector<asset::E_FORMAT> all{
        asset::EF_B10G11R11_UFLOAT_PACK32,
        asset::EF_R16_SFLOAT,
        asset::EF_R16G16_SFLOAT,
        asset::EF_R16G16B16_SFLOAT,
        asset::EF_R16G16B16A16_SFLOAT,
        asset::EF_R32_SFLOAT,
        asset::EF_R32G32_SFLOAT,
        asset::EF_R32G32B32_SFLOAT,
        asset::EF_R32G32B32A32_SFLOAT,
        asset::EF_R8G8_UNORM,
        asset::EF_R8G8_SNORM,
        asset::EF_R8G8B8_UNORM,
        asset::EF_R8G8B8_SNORM,
        asset::EF_B8G8R8A8_UNORM, //bgra
        asset::EF_R8G8B8A8_UNORM,
        asset::EF_R8G8B8A8_SNORM,
        asset::EF_A2B10G10R10_UNORM_PACK32,
        asset::EF_A2B10G10R10_SNORM_PACK32,
        asset::EF_A2R10G10B10_UNORM_PACK32, //bgra
        asset::EF_A2R10G10B10_SNORM_PACK32, //bgra
        asset::EF_R16_UNORM,
        asset::EF_R16_SNORM,
        asset::EF_R16G16_UNORM,
        asset::EF_R16G16_SNORM,
        asset::EF_R16G16B16_UNORM,
        asset::EF_R16G16B16_SNORM,
        asset::EF_R16G16B16A16_UNORM,
        asset::EF_R16G16B16A16_SNORM
    };
    core::vector<asset::E_FORMAT> normalized{
        asset::EF_B8G8R8A8_UNORM, //bgra
        asset::EF_R8G8B8A8_UNORM,
        asset::EF_R8G8B8A8_SNORM,
        asset::EF_A2B10G10R10_UNORM_PACK32,
        asset::EF_A2B10G10R10_SNORM_PACK32,
        asset::EF_A2R10G10B10_UNORM_PACK32, //bgra
        asset::EF_A2R10G10B10_SNORM_PACK32, //bgra
        asset::EF_R16_UNORM,
        asset::EF_R16_SNORM,
        asset::EF_R16G16_UNORM,
        asset::EF_R16G16_SNORM,
        asset::EF_R16G16B16_UNORM,
        asset::EF_R16G16B16_SNORM,
        asset::EF_R16G16B16A16_UNORM,
        asset::EF_R16G16B16A16_SNORM
    };
    core::vector<asset::E_FORMAT> bgra{
        asset::EF_B8G8R8A8_UNORM, //bgra
        asset::EF_A2R10G10B10_UNORM_PACK32, //bgra
        asset::EF_A2R10G10B10_SNORM_PACK32, //bgra
    };
    core::vector<asset::E_FORMAT> normals{
        asset::EF_R8_SNORM,
        asset::EF_R8G8_SNORM,
        asset::EF_R8G8B8_SNORM,
        asset::EF_R8G8B8A8_SNORM,
        asset::EF_R16_SNORM,
        asset::EF_R16G16_SNORM,
        asset::EF_R16G16B16_SNORM,
        asset::EF_R16G16B16A16_SNORM,
        asset::EF_A2B10G10R10_SNORM_PACK32,
        asset::EF_A2R10G10B10_SNORM_PACK32, //bgra
        asset::EF_R16_SFLOAT,
        asset::EF_R16G16_SFLOAT,
        asset::EF_R16G16B16_SFLOAT,
        asset::EF_R16G16B16A16_SFLOAT
    };

    auto minValueOfTypeFP = [](asset::E_FORMAT _fmt, uint32_t _cmpntNum) -> float {
        if (isNormalizedFormat(_fmt))
        {
            return isSignedFormat(_fmt) ? -1.f : 0.f;
        }
        switch (_fmt)
        {
        case asset::EF_R16_SFLOAT:
        case asset::EF_R16G16_SFLOAT:
        case asset::EF_R16G16B16_SFLOAT:
        case asset::EF_R16G16B16A16_SFLOAT:
            return -65504.f;
        case asset::EF_R32_SFLOAT:
        case asset::EF_R32G32_SFLOAT:
        case asset::EF_R32G32B32_SFLOAT:
        case asset::EF_R32G32B32A32_SFLOAT:
            return -FLT_MAX;
        case asset::EF_B10G11R11_UFLOAT_PACK32:
            return 0.f;
        default:
            return 1.f;
        }
    };
    auto maxValueOfTypeFP = [](asset::E_FORMAT _fmt, uint32_t _cmpntNum) -> float {
        if (isNormalizedFormat(_fmt))
        {
            return 1.f;
        }
        switch (_fmt)
        {
        case asset::EF_R16_SFLOAT:
        case asset::EF_R16G16_SFLOAT:
        case asset::EF_R16G16B16_SFLOAT:
        case asset::EF_R16G16B16A16_SFLOAT:
            return 65504.f;
        case asset::EF_R32_SFLOAT:
        case asset::EF_R32G32_SFLOAT:
        case asset::EF_R32G32B32_SFLOAT:
        case asset::EF_R32G32B32A32_SFLOAT:
            return FLT_MAX;
        case asset::EF_B10G11R11_UFLOAT_PACK32:
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
                all = core::vector<asset::E_FORMAT>(1u, asset::EF_A2R10G10B10_SNORM_PACK32);
            }
			else all = std::move(normals);
		}
		else if (isBGRALayoutFormat(_type))
			all = std::move(bgra);
		else
			all = std::move(normalized);
	}

	if (isNormalizedFormat(_type) && !isSignedFormat(_type))
		all.erase(std::remove_if(all.begin(), all.end(), [](asset::E_FORMAT _t) { return isSignedFormat(_t); }), all.end());
	else if (isNormalizedFormat(_type) && isSignedFormat(_type))
		all.erase(std::remove_if(all.begin(), all.end(), [](asset::E_FORMAT _t) { return !isSignedFormat(_t); }), all.end());

    const uint32_t originalCpa = getFormatChannelCount(_type);
    all.erase(
        std::remove_if(all.begin(), all.end(),
            [originalCpa](asset::E_FORMAT fmt) { return getFormatChannelCount(fmt) < originalCpa; }
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
		if (ok && getTexelOrBlockSize(*it) <= _sizeThreshold)
			possibleTypes.push_back({*it});
	}
	return possibleTypes;
}

bool CMeshManipulator::calcMaxQuantizationError(const SAttribTypeChoice& _srcType, const SAttribTypeChoice& _dstType, const core::vector<core::vectorSIMDf>& _srcData, const SErrorMetric& _errMetric) const
{
    using namespace video;

	using QuantF_t = core::vectorSIMDf(*)(const core::vectorSIMDf&, asset::E_FORMAT, asset::E_FORMAT);

	QuantF_t quantFunc = nullptr;

	if (_errMetric.method == EEM_ANGLES)
	{
		switch (_dstType.type)
		{
		case asset::EF_R8_SNORM:
        case asset::EF_R8G8_SNORM:
        case asset::EF_R8G8B8_SNORM:
        case asset::EF_R8G8B8A8_SNORM:
			quantFunc = [](const core::vectorSIMDf& _in, asset::E_FORMAT, asset::E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint32_t*)buf)[0] = quantizeNormal888(_in);

				core::vectorSIMDf retval;
				asset::ICPUMeshBuffer::getAttribute(retval, buf, asset::EF_R8G8B8A8_SNORM);
				retval.w = 1.f;
				return retval;
			};
			break;
		case asset::EF_A2B10G10R10_SINT_PACK32: // RGB10_A2
			quantFunc = [](const core::vectorSIMDf& _in, asset::E_FORMAT, asset::E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint32_t*)buf)[0] = quantizeNormal2_10_10_10(_in);

				core::vectorSIMDf retval;
				asset::ICPUMeshBuffer::getAttribute(retval, buf, asset::EF_A2B10G10R10_SINT_PACK32);
				retval.w = 1.f;
				return retval;
			};
			break;
        case asset::EF_R16_SNORM:
        case asset::EF_R16G16_SNORM:
        case asset::EF_R16G16B16_SNORM:
        case asset::EF_R16G16B16A16_SNORM:
			quantFunc = [](const core::vectorSIMDf& _in, asset::E_FORMAT, asset::E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint64_t*)buf)[0] = quantizeNormal16_16_16(_in);

				core::vectorSIMDf retval;
				asset::ICPUMeshBuffer::getAttribute(retval, buf, asset::EF_R16G16B16A16_SNORM);
				retval.w = 1.f;
				return retval;
			};
			break;
        case asset::EF_R16_SFLOAT:
        case asset::EF_R16G16_SFLOAT:
        case asset::EF_R16G16B16_SFLOAT:
        case asset::EF_R16G16B16A16_SFLOAT:
			quantFunc = [](const core::vectorSIMDf& _in, asset::E_FORMAT, asset::E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint64_t*)buf)[0] = quantizeNormalHalfFloat(_in);

				core::vectorSIMDf retval;
				asset::ICPUMeshBuffer::getAttribute(retval, buf, asset::EF_R16G16B16A16_SFLOAT);
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
		quantFunc = [](const core::vectorSIMDf& _in, asset::E_FORMAT _inType, asset::E_FORMAT _outType) -> core::vectorSIMDf {
			uint8_t buf[32];
			asset::ICPUMeshBuffer::setAttribute(_in, buf, _outType);
			core::vectorSIMDf out(0.f, 0.f, 0.f, 1.f);
			asset::ICPUMeshBuffer::getAttribute(out, buf, _outType);
			return out;
		};
	}

	_IRR_DEBUG_BREAK_IF(!quantFunc)
	if (!quantFunc)
		return false;

	for (const core::vectorSIMDf& d : _srcData)
	{
		const core::vectorSIMDf quantized = quantFunc(d, _srcType.type, _dstType.type);

        if (!compareFloatingPointAttribute(d, quantized, asset::getFormatChannelCount(_srcType.type), _errMetric))
            return false;
	}

	return true;
}

asset::ICPUBuffer* CMeshManipulator::idxBufferFromTriangleStripsToTriangles(const void* _input, size_t _idxCount, asset::E_INDEX_TYPE _idxType) const
{
	if (_idxType == asset::EIT_16BIT)
		return triangleStripsToTriangles<uint16_t>(_input, _idxCount);
	else if (_idxType == asset::EIT_32BIT)
		return triangleStripsToTriangles<uint32_t>(_input, _idxCount);
	return NULL;
}

template<typename T>
asset::ICPUBuffer* CMeshManipulator::triangleStripsToTriangles(const void* _input, size_t _idxCount) const
{
	const size_t outputSize = (_idxCount - 2)*3;

	asset::ICPUBuffer* output = new asset::ICPUBuffer(outputSize * sizeof(T));
	T* iptr = (T*)_input;
	T* optr = (T*)output->getPointer();
	for (size_t i = 0, j = 0; i < outputSize; j+=2)
	{
		optr[i++] = iptr[j+0];
		optr[i++] = iptr[j+1];
		optr[i++] = iptr[j+2];
		if (i == outputSize)
			break;
		optr[i++] = iptr[j+2];
		optr[i++] = iptr[j+1];
		optr[i++] = iptr[j+3];
	}
	return output;
}
template asset::ICPUBuffer* CMeshManipulator::triangleStripsToTriangles<uint16_t>(const void* _input, size_t _idxCount) const;
template asset::ICPUBuffer* CMeshManipulator::triangleStripsToTriangles<uint32_t>(const void* _input, size_t _idxCount) const;

asset::ICPUBuffer* CMeshManipulator::idxBufferFromTrianglesFanToTriangles(const void* _input, size_t _idxCount, asset::E_INDEX_TYPE _idxType) const
{
	if (_idxType == asset::EIT_16BIT)
		return trianglesFanToTriangles<uint16_t>(_input, _idxCount);
	else if (_idxType == asset::EIT_32BIT)
		return trianglesFanToTriangles<uint32_t>(_input, _idxCount);
	return NULL;
}

template<typename T>
inline asset::ICPUBuffer* CMeshManipulator::trianglesFanToTriangles(const void* _input, size_t _idxCount) const
{
	const size_t outputSize = ((_idxCount-1)/2) * 3;

	asset::ICPUBuffer* output = new asset::ICPUBuffer(outputSize*sizeof(T));
	T* iptr = (T*)_input;
	T* optr = (T*)output->getPointer();
	for (size_t i = 0, j = 1; i < outputSize; j+=2)
	{
		optr[i++] = iptr[0];
		optr[i++] = iptr[j];
		optr[i++] = iptr[j+1];
	}
	return output;
}
template asset::ICPUBuffer* CMeshManipulator::trianglesFanToTriangles<uint16_t>(const void* _input, size_t _idxCount) const;
template asset::ICPUBuffer* CMeshManipulator::trianglesFanToTriangles<uint32_t>(const void* _input, size_t _idxCount) const;

bool CMeshManipulator::compareFloatingPointAttribute(const core::vectorSIMDf& _a, const core::vectorSIMDf& _b, size_t _cpa, const SErrorMetric& _errMetric) const
{
	using ErrorF_t = core::vectorSIMDf(*)(core::vectorSIMDf, core::vectorSIMDf);

	ErrorF_t errorFunc = nullptr;

	switch (_errMetric.method)
	{
	case EEM_POSITIONS:
		errorFunc = [](core::vectorSIMDf _d1, core::vectorSIMDf _d2) -> core::vectorSIMDf {
			return core::abs(_d1 - _d2);
		};
		break;
	case EEM_ANGLES:
		errorFunc = [](core::vectorSIMDf _d1, core::vectorSIMDf _d2)->core::vectorSIMDf {
			_d1.w = _d2.w = 0.f;
			return core::dot(_d1, _d2) / (core::length(_d1) * core::length(_d2));
		};
		break;
	case EEM_QUATERNION:
		errorFunc = [](core::vectorSIMDf _d1, core::vectorSIMDf _d2)->core::vectorSIMDf {
			return core::dot(_d1, _d2) / (core::length(_d1) * core::length(_d2));
		};
		break;
    default:
        errorFunc = nullptr;
        break;
	}

	using CmpF_t = bool(*)(const core::vectorSIMDf&, const core::vectorSIMDf&, size_t);

	CmpF_t cmpFunc = nullptr;

	switch (_errMetric.method)
	{
	case EEM_POSITIONS:
		cmpFunc = [](const core::vectorSIMDf& _err, const core::vectorSIMDf& _epsilon, size_t _cpa) -> bool {
			for (size_t i = 0u; i < _cpa; ++i)
				if (_err.pointer[i] > _epsilon.pointer[i])
					return false;
			return true;
		};
		break;
	case EEM_ANGLES:
	case EEM_QUATERNION:
		cmpFunc = [](const core::vectorSIMDf& _err, const core::vectorSIMDf& _epsilon, size_t _cpa) -> bool {
			return _err.x > (1.f - _epsilon.x);
		};
		break;
    default:
        cmpFunc = nullptr;
        break;
	}

	_IRR_DEBUG_BREAK_IF(!errorFunc)
	_IRR_DEBUG_BREAK_IF(!cmpFunc)
	if (!errorFunc || !cmpFunc)
		return false;

    const core::vectorSIMDf err = errorFunc(_a, _b);
    return cmpFunc(err, _errMetric.epsilon, _cpa);
}

template<>
bool IMeshManipulator::getPolyCount<asset::ICPUBuffer>(uint32_t& outCount, asset::IMeshBuffer<asset::ICPUBuffer>* meshbuffer)
{
    outCount= 0;
    if (meshbuffer)
        return false;

    uint32_t trianglecount;

    switch (meshbuffer->getPrimitiveType())
    {
        case asset::EPT_POINTS:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case asset::EPT_LINE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-1;
            break;
        case asset::EPT_LINE_LOOP:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case asset::EPT_LINES:
            trianglecount = meshbuffer->getIndexCount()/2;
            break;
        case asset::EPT_TRIANGLE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_FAN:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLES:
            trianglecount = meshbuffer->getIndexCount()/3;
            break;
    }

    outCount = trianglecount;
    return true;
}
template<>
bool IMeshManipulator::getPolyCount<video::IGPUBuffer>(uint32_t& outCount, asset::IMeshBuffer<video::IGPUBuffer>* meshbuffer)
{
    outCount = 0;
    if (meshbuffer)
        return false;

    if (static_cast<video::IGPUMeshBuffer*>(meshbuffer)->isIndexCountGivenByXFormFeedback())
        return false;

    uint32_t trianglecount;

    switch (meshbuffer->getPrimitiveType())
    {
        case asset::EPT_POINTS:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case asset::EPT_LINE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-1;
            break;
        case asset::EPT_LINE_LOOP:
            trianglecount = meshbuffer->getIndexCount();
            break;
        case asset::EPT_LINES:
            trianglecount = meshbuffer->getIndexCount()/2;
            break;
        case asset::EPT_TRIANGLE_STRIP:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLE_FAN:
            trianglecount = meshbuffer->getIndexCount()-2;
            break;
        case asset::EPT_TRIANGLES:
            trianglecount = meshbuffer->getIndexCount()/3;
            break;
    }

    outCount = trianglecount;
    return true;
}


//! Returns amount of polygons in mesh.
template<typename T>
bool IMeshManipulator::getPolyCount(uint32_t& outCount, asset::IMesh<T>* mesh)
{
    outCount = 0;
	if (!mesh)
		return false;

    bool retval = true;
	for (uint32_t g=0; g<mesh->getMeshBufferCount(); ++g)
    {
        uint32_t trianglecount;
        retval = retval&&getPolyCount(trianglecount,mesh->getMeshBuffer(g));
    }

	return retval;
}

template bool IMeshManipulator::getPolyCount<asset::ICPUMeshBuffer>(uint32_t& outCount, asset::IMesh<asset::ICPUMeshBuffer>* mesh);
template bool IMeshManipulator::getPolyCount<video::IGPUMeshBuffer>(uint32_t& outCount, asset::IMesh<video::IGPUMeshBuffer>* mesh);

#ifndef NEW_MESHES
//! Returns amount of polygons in mesh.
uint32_t IMeshManipulator::getPolyCount(scene::IAnimatedMesh* mesh)
{
	if (mesh && mesh->getFrameCount() != 0)
		return getPolyCount(mesh->getMesh(0));

	return 0;
}
#endif // NEW_MESHES

} // end namespace scene
} // end namespace irr


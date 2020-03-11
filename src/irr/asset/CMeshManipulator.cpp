// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h


#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "os.h"
#include "irr/asset/asset.h"
#include "irr/asset/normal_quantization.h"
#include "irr/asset/CMeshManipulator.h"
#include "irr/asset/CSmoothNormalGenerator.h"
#include "irr/asset/CForsythVertexCacheOptimizer.h"
#include "irr/asset/COverdrawMeshOptimizer.h"

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
void IMeshManipulator::flipSurfaces(ICPUMeshBuffer* inbuffer)
{
	if (!inbuffer)
		return;

    const uint32_t idxcnt = inbuffer->getIndexCount();
    if (!inbuffer->getIndices())
        return;


    if (inbuffer->getIndexType() == EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case EPT_TRIANGLE_FAN:
            for (uint32_t i=1; i<idxcnt; i+=2)
            {
                const uint16_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
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
                auto newIndexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>((idxcnt+1u)*sizeof(uint16_t));
				auto* destPtr = reinterpret_cast<uint16_t*>(newIndexBuffer->getPointer());
				destPtr[0] = idx[0];
                memcpy(destPtr+1u,idx,sizeof(uint16_t)*idxcnt);
                inbuffer->setIndexCount(idxcnt+1u);
                inbuffer->setIndexBufferOffset(0);
                inbuffer->getMeshDataAndFormat()->setIndexBuffer(std::move(newIndexBuffer));
            }
            break;
        case EPT_TRIANGLES:
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
    else if (inbuffer->getIndexType() == EIT_32BIT)
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
        switch (inbuffer->getPrimitiveType())
        {
        case EPT_TRIANGLE_FAN:
            for (uint32_t i=1; i<idxcnt; i+=2)
            {
                const uint32_t tmp = idx[i];
                idx[i] = idx[i+1];
                idx[i+1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
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
                auto newIndexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>((idxcnt+1u)*sizeof(uint32_t));
				auto* destPtr = reinterpret_cast<uint32_t*>(newIndexBuffer->getPointer());
				destPtr[0] = idx[0];
                memcpy(destPtr+1u,idx,sizeof(uint32_t)*idxcnt);
                inbuffer->setIndexCount(idxcnt+1);
                inbuffer->setIndexBufferOffset(0);
                inbuffer->getMeshDataAndFormat()->setIndexBuffer(std::move(newIndexBuffer));
            }
            break;
        case EPT_TRIANGLES:
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

core::smart_refctd_ptr<ICPUMeshBuffer> CMeshManipulator::createMeshBufferFetchOptimized(const ICPUMeshBuffer* _inbuffer)
{
	if (!_inbuffer || !_inbuffer->getMeshDataAndFormat() || !_inbuffer->getIndices())
		return NULL;

	auto outbuffer = createMeshBufferDuplicate(_inbuffer);
    IMeshDataFormatDesc<ICPUBuffer>* outDesc = outbuffer->getMeshDataAndFormat();

	// Find vertex count
	size_t vertexCount = _inbuffer->calcVertexCount();
	const void* ind = _inbuffer->getIndices();

	core::unordered_set<const ICPUBuffer*> buffers;
	for (size_t i = 0; i < EVAI_COUNT; ++i)
		buffers.insert(outDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i));

	size_t offsets[EVAI_COUNT];
	memset(offsets, -1, sizeof(offsets));
	E_FORMAT types[EVAI_COUNT];
	if (buffers.size() != 1)
	{
		size_t lastOffset = 0u;
		size_t lastSize = 0u;
		for (size_t i = 0; i < EVAI_COUNT; ++i)
		{
			if (outDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i))
			{
				types[i] = outDesc->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i);

                const uint32_t typeSz = getTexelOrBlockBytesize(types[i]);
                const size_t alignment = (typeSz/getFormatChannelCount(types[i]) == 8u) ? 8ull : 4ull; // if format 64bit per channel, than align to 8

				offsets[i] = lastOffset + lastSize;
				const size_t mod = offsets[i] % alignment;
				offsets[i] += mod;

				lastOffset = offsets[i];
                lastSize = typeSz;
			}
		}
		const size_t vertexSize = lastOffset + lastSize;

		auto newVertBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(vertexCount*vertexSize);
		for (size_t i = 0; i < EVAI_COUNT; ++i)
		{
			if (offsets[i] < 0xffffffff)
			{
				outDesc->setVertexAttrBuffer(core::smart_refctd_ptr(newVertBuffer), (E_VERTEX_ATTRIBUTE_ID)i, types[i], vertexSize, offsets[i]);
			}
		}
	}
	outbuffer->setBaseVertex(0);

	core::vector<E_VERTEX_ATTRIBUTE_ID> activeAttribs;
	for (size_t i = 0; i < EVAI_COUNT; ++i)
		if (outDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i))
			activeAttribs.push_back((E_VERTEX_ATTRIBUTE_ID)i);

	uint32_t* remapBuffer = (uint32_t*)_IRR_ALIGNED_MALLOC(vertexCount*4,_IRR_SIMD_ALIGNMENT);
	memset(remapBuffer, 0xffffffffu, vertexCount*4);

	const E_INDEX_TYPE idxType = outbuffer->getIndexType();
	void* indices = outbuffer->getIndices();
	size_t nextVert = 0u;

	for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
	{
		const uint32_t index = idxType == EIT_32BIT ? ((uint32_t*)indices)[i] : ((uint16_t*)indices)[i];

		uint32_t& remap = remapBuffer[index];

		if (remap == 0xffffffffu)
		{
			for (size_t j = 0; j < activeAttribs.size(); ++j)
			{
				E_FORMAT type = types[activeAttribs[j]];

                if (!isNormalizedFormat(type) && (isIntegerFormat(type) || isScaledFormat(type)))
				{
					uint32_t dst[4];
					_inbuffer->getAttribute(dst, (E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], index);
					outbuffer->setAttribute(dst, (E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], nextVert);
				}
				else
				{
					core::vectorSIMDf dst;
					_inbuffer->getAttribute(dst, (E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], index);
					outbuffer->setAttribute(dst, (E_VERTEX_ATTRIBUTE_ID)activeAttribs[j], nextVert);
				}
			}

			remap = nextVert++;
		}

		if (idxType == EIT_32BIT)
			((uint32_t*)indices)[i] = remap;
		else
			((uint16_t*)indices)[i] = remap;
	}

	_IRR_ALIGNED_FREE(remapBuffer);

	_IRR_DEBUG_BREAK_IF(nextVert > vertexCount)

	return outbuffer;
}

//! Creates a copy of the mesh, which will only consist of unique primitives
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer, bool _makeIndexBuf)
{
	if (!inbuffer)
		return 0;
    IMeshDataFormatDesc<ICPUBuffer>* oldDesc = inbuffer->getMeshDataAndFormat();
    if (!oldDesc)
        return 0;

    if (!inbuffer->getIndices())
        return core::smart_refctd_ptr<ICPUMeshBuffer>(inbuffer); // yes we want an extra grab
    
    const uint32_t idxCnt = inbuffer->getIndexCount();
    auto clone = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
    clone->setBoundingBox(inbuffer->getBoundingBox());
    clone->setIndexCount(idxCnt);
    const E_PRIMITIVE_TYPE ept = inbuffer->getPrimitiveType();
    clone->setPrimitiveType(ept);
    clone->getMaterial() = inbuffer->getMaterial();

	{
		auto desc = core::make_smart_refctd_ptr<ICPUMeshDataFormatDesc>();

		size_t stride = 0;
		int32_t offset[EVAI_COUNT];
		size_t newAttribSizes[EVAI_COUNT];
		uint8_t* sourceBuffers[EVAI_COUNT] = {NULL};
		size_t sourceBufferStrides[EVAI_COUNT];
		for (size_t i=0; i< EVAI_COUNT; i++)
		{
			const ICPUBuffer* vbuf = oldDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i);
			if (vbuf)
			{
				offset[i] = stride;
				newAttribSizes[i] = getTexelOrBlockBytesize(oldDesc->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i));
				stride += newAttribSizes[i];
				if (stride>=0xdeadbeefu)
					return nullptr;

				sourceBuffers[i] = (uint8_t*)vbuf->getPointer();
				sourceBuffers[i] += oldDesc->getMappedBufferOffset((E_VERTEX_ATTRIBUTE_ID)i);
				sourceBufferStrides[i] = oldDesc->getMappedBufferStride((E_VERTEX_ATTRIBUTE_ID)i);
			}
			else
				offset[i] = -1;
		}

		auto vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(stride*idxCnt);
		for (size_t i=0; i<EVAI_COUNT; i++)
		{
			if (offset[i]>=0)
				desc->setVertexAttrBuffer(core::smart_refctd_ptr(vertexBuffer),(E_VERTEX_ATTRIBUTE_ID)i,oldDesc->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i),stride,offset[i]);
		}

		uint8_t* destPointer = (uint8_t*)vertexBuffer->getPointer();
		if (inbuffer->getIndexType()==EIT_16BIT)
		{
			uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
			for (uint64_t i=0; i<idxCnt; i++,idx++)
			for (size_t j=0; j<EVAI_COUNT; j++)
			{
				if (offset[j]<0)
					continue;

				memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
				destPointer += newAttribSizes[j];
			}
		}
		else if (inbuffer->getIndexType()==EIT_32BIT)
		{
			uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
			for (uint64_t i=0; i<idxCnt; i++,idx++)
			for (size_t j=0; j<EVAI_COUNT; j++)
			{
				if (offset[j]<0)
					continue;

				memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
				destPointer += newAttribSizes[j];
			}
		}
        
        if (_makeIndexBuf)
        {
            auto idxbuf = core::make_smart_refctd_ptr<ICPUBuffer>(idxCnt*(idxCnt<0x10000 ? 2u : 4u));
            if (idxCnt<0x10000u)
            {
                for (uint32_t i = 0u; i < idxCnt; ++i)
                    reinterpret_cast<uint16_t*>(idxbuf->getPointer())[i] = i;
                clone->setIndexType(EIT_16BIT);
                clone->setIndexBufferOffset(0);
            }
            else
            {
                for (uint32_t i = 0u; i < idxCnt; ++i)
                    reinterpret_cast<uint32_t*>(idxbuf->getPointer())[i] = i;
                clone->setIndexType(EIT_32BIT);
                clone->setIndexBufferOffset(0);
            }
            desc->setIndexBuffer(std::move(idxbuf));
        }

		clone->setMeshDataAndFormat(std::move(desc));
	}

	return clone;
}

//
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::calculateSmoothNormals(ICPUMeshBuffer* inbuffer, bool makeNewMesh, float epsilon, E_VERTEX_ATTRIBUTE_ID normalAttrID, VxCmpFunction vxcmp)
{
	if (inbuffer == nullptr)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;
	}

	//Mesh has to have unique primitives
	if (inbuffer->getIndexType() != E_INDEX_TYPE::EIT_UNKNOWN)
	{
		_IRR_DEBUG_BREAK_IF(true);
		return nullptr;
	}

	auto outbuffer = makeNewMesh ? createMeshBufferDuplicate(inbuffer) : core::smart_refctd_ptr<ICPUMeshBuffer>(inbuffer);
	CSmoothNormalGenerator::calculateNormals(outbuffer.get(), epsilon, normalAttrID, vxcmp);

	return outbuffer;
}

// Used by createMeshBufferWelded only
static bool cmpVertices(ICPUMeshBuffer* _inbuf, const void* _va, const void* _vb, size_t _vsize, const IMeshManipulator::SErrorMetric* _errMetrics)
{
    auto cmpInteger = [](uint32_t* _a, uint32_t* _b, size_t _n) -> bool {
        return !memcmp(_a, _b, _n*4);
    };

    const uint8_t* va = (uint8_t*)_va, *vb = (uint8_t*)_vb;
    auto desc = _inbuf->getMeshDataAndFormat();
    for (size_t i = 0u; i < EVAI_COUNT; ++i)
    {
        if (!desc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i))
            continue;

        const auto atype = desc->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i);
        const auto cpa = getFormatChannelCount(atype);

        if (isIntegerFormat(atype) || isScaledFormat(atype))
        {
            uint32_t attr[8];
            ICPUMeshBuffer::getAttribute(attr, va, atype);
            ICPUMeshBuffer::getAttribute(attr+4, vb, atype);
            if (!cmpInteger(attr, attr+4, cpa))
                return false;
        }
        else
        {
            core::vectorSIMDf attr[2];
            ICPUMeshBuffer::getAttribute(attr[0], va, atype);
            ICPUMeshBuffer::getAttribute(attr[1], vb, atype);
            if (!IMeshManipulator::compareFloatingPointAttribute(attr[0], attr[1], cpa, _errMetrics[i]))
                return false;
        }

        const uint32_t sz = getTexelOrBlockBytesize(atype);
        va += sz;
        vb += sz;
    }

    return true;
}

//! Creates a copy of a mesh, which will have identical vertices welded together
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createMeshBufferWelded(ICPUMeshBuffer *inbuffer, const SErrorMetric* _errMetrics, const bool& optimIndexType, const bool& makeNewMesh)
{
    if (!inbuffer)
        return nullptr;
    IMeshDataFormatDesc<ICPUBuffer>* oldDesc = inbuffer->getMeshDataAndFormat();
    if (!oldDesc)
        return nullptr;

    bool bufferPresent[EVAI_COUNT];

    size_t vertexAttrSize[EVAI_COUNT];
    size_t vertexSize = 0;
    for (size_t i=0; i<EVAI_COUNT; i++)
    {
        const ICPUBuffer* buf = oldDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i);
        bufferPresent[i] = buf;
        if (buf)
        {
            const E_FORMAT componentType = oldDesc->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i);
            vertexAttrSize[i] = getTexelOrBlockBytesize(componentType);
            vertexSize += vertexAttrSize[i];
        }
    }

    auto cmpfunc = [&, inbuffer, vertexSize, _errMetrics](const void* _va, const void* _vb) {
        return cmpVertices(inbuffer, _va, _vb, vertexSize, _errMetrics);
    };

    size_t vertexCount = inbuffer->calcVertexCount();
    E_INDEX_TYPE oldIndexType = inbuffer->getIndexType();

    if (vertexCount==0)
        return nullptr;

    // reset redirect list
    uint32_t* redirects = new uint32_t[vertexCount];

    uint32_t maxRedirect = 0;

    uint8_t* epicData = (uint8_t*)_IRR_ALIGNED_MALLOC(vertexSize*vertexCount,_IRR_SIMD_ALIGNMENT);
    for (size_t i=0; i < vertexCount; i++)
    {
        uint8_t* currentVertexPtr = epicData+i*vertexSize;
        for (size_t k=0; k<EVAI_COUNT; k++)
        {
            if (!bufferPresent[k])
                continue;

            size_t stride = oldDesc->getMappedBufferStride((E_VERTEX_ATTRIBUTE_ID)k);
            uint8_t* sourcePtr = inbuffer->getAttribPointer((E_VERTEX_ATTRIBUTE_ID)k)+i*stride;
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
    core::smart_refctd_ptr<ICPUMeshBuffer> clone;
    if (makeNewMesh)
        clone = createMeshBufferDuplicate(inbuffer);
    else
    {
        if (!oldDesc->getIndexBuffer())
        {
            oldDesc->setIndexBuffer(core::make_smart_refctd_ptr<ICPUBuffer>((maxRedirect >= 0x10000u ? sizeof(uint32_t) : sizeof(uint16_t)) * inbuffer->getIndexCount()));
            inbuffer->setIndexType(maxRedirect>=0x10000u ? EIT_32BIT:EIT_16BIT);
        }
    }


    if (oldIndexType==EIT_16BIT)
    {
        uint16_t* indicesIn = reinterpret_cast<uint16_t*>(oldIndices);
        if ((makeNewMesh ? clone.get():inbuffer)->getIndexType()==EIT_32BIT)
        {
            uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone.get():inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
        else if ((makeNewMesh ? clone.get():inbuffer)->getIndexType()==EIT_16BIT)
        {
            uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone.get():inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
    }
    else if (oldIndexType==EIT_32BIT)
    {
        uint32_t* indicesIn = reinterpret_cast<uint32_t*>(oldIndices);
        if ((makeNewMesh ? clone.get():inbuffer)->getIndexType()==EIT_32BIT)
        {
            uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone.get():inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
        else if ((makeNewMesh ? clone.get():inbuffer)->getIndexType()==EIT_16BIT)
        {
            uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone.get():inbuffer)->getIndices());
            for (size_t i=0; i<inbuffer->getIndexCount(); i++)
                indicesOut[i] = redirects[indicesIn[i]];
        }
    }
    else if ((makeNewMesh ? clone.get():inbuffer)->getIndexType()==EIT_32BIT)
    {
        uint32_t* indicesOut = reinterpret_cast<uint32_t*>((makeNewMesh ? clone.get():inbuffer)->getIndices());
        for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            indicesOut[i] = redirects[i];
    }
    else if ((makeNewMesh ? clone.get():inbuffer)->getIndexType()==EIT_16BIT)
    {
        uint16_t* indicesOut = reinterpret_cast<uint16_t*>((makeNewMesh ? clone.get():inbuffer)->getIndices());
        for (size_t i=0; i<inbuffer->getIndexCount(); i++)
            indicesOut[i] = redirects[i];
    }
    delete [] redirects;

    if (makeNewMesh)
        return clone;
    else
        return core::smart_refctd_ptr<ICPUMeshBuffer>(inbuffer);
}

core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createOptimizedMeshBuffer(const ICPUMeshBuffer* _inbuffer, const SErrorMetric* _errMetric)
{
	if (!_inbuffer)
		return nullptr;
	auto outbuffer = createMeshBufferDuplicate(_inbuffer);
	if (!outbuffer->getMeshDataAndFormat())
		return outbuffer;

	// Find vertex count
	size_t vertexCount = outbuffer->calcVertexCount();

	// make index buffer 0,1,2,3,4,... if nothing's mapped
	if (!outbuffer->getIndices())
	{
		auto ib = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(uint32_t)*vertexCount);
		IMeshDataFormatDesc<ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		uint32_t* indices = (uint32_t*)ib->getPointer();
		for (uint32_t i = 0; i < vertexCount; ++i)
			indices[i] = i;
		newDesc->setIndexBuffer(std::move(ib));
		outbuffer->setIndexCount(vertexCount);
		outbuffer->setIndexType(EIT_32BIT);
	}

	// make 32bit index buffer if 16bit one is present
	if (outbuffer->getIndexType() == EIT_16BIT)
	{
        IMeshDataFormatDesc<ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		newDesc->setIndexBuffer(CMeshManipulator::create32BitFrom16BitIdxBufferSubrange(reinterpret_cast<uint16_t*>(outbuffer->getIndices()), outbuffer->getIndexCount()));
		// no need to set index buffer offset to 0 because it already is
		outbuffer->setIndexType(EIT_32BIT);
	}

	// convert index buffer for triangle primitives
	if (outbuffer->getPrimitiveType() == EPT_TRIANGLE_FAN)
	{
		IMeshDataFormatDesc<ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		const ICPUBuffer* ib = newDesc->getIndexBuffer();
		outbuffer->setPrimitiveType(EPT_TRIANGLES);
		auto newIb = idxBufferFromTrianglesFanToTriangles(outbuffer->getIndices(), outbuffer->getIndexCount(), EIT_32BIT);
		outbuffer->setIndexCount(newIb->getSize() / sizeof(uint32_t));
		newDesc->setIndexBuffer(std::move(newIb));
	}
	else if (outbuffer->getPrimitiveType() == EPT_TRIANGLE_STRIP)
	{
		IMeshDataFormatDesc<ICPUBuffer>* newDesc = outbuffer->getMeshDataAndFormat();
		outbuffer->setPrimitiveType(EPT_TRIANGLES);
		auto newIb = idxBufferFromTriangleStripsToTriangles(outbuffer->getIndices(), outbuffer->getIndexCount(), EIT_32BIT);
		outbuffer->setIndexCount(newIb->getSize() / sizeof(uint32_t));
		newDesc->setIndexBuffer(std::move(newIb));
	}
	else if (outbuffer->getPrimitiveType() != EPT_TRIANGLES)
		return nullptr;

	// STEP: weld
    createMeshBufferWelded(outbuffer.get(), _errMetric, false, false);

    // STEP: filter invalid triangles
    filterInvalidTriangles(outbuffer.get());

	// STEP: overdraw optimization
	COverdrawMeshOptimizer::createOptimized(outbuffer.get(), false);

	// STEP: Forsyth
	{
		uint32_t* indices = reinterpret_cast<uint32_t*>(outbuffer->getIndices());
		CForsythVertexCacheOptimizer forsyth;
		forsyth.optimizeTriangleOrdering(vertexCount, outbuffer->getIndexCount(), indices, indices);
	}

	// STEP: prefetch optimization
	outbuffer = CMeshManipulator::createMeshBufferFetchOptimized(outbuffer.get()); // here we also get interleaved attributes (single vertex buffer)
	
	// STEP: requantization
	requantizeMeshBuffer(outbuffer.get(), _errMetric);

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

		core::smart_refctd_ptr<ICPUBuffer> newIdxBuffer;
		bool verticesMustBeReordered = false;
        E_INDEX_TYPE newIdxType = EIT_UNKNOWN;

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
					newIdxType = EIT_16BIT;
				else
					newIdxType = EIT_32BIT;

				outbuffer->setBaseVertex(outbuffer->getBaseVertex() + minIdx);

				if (newIdxType == EIT_16BIT)
				{
					newIdxBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(uint16_t)*outbuffer->getIndexCount());
					// no need to change index buffer offset because it's always 0 (after duplicating original mesh)
					for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
						reinterpret_cast<uint16_t*>(newIdxBuffer->getPointer())[i] = reinterpret_cast<const uint32_t*>(indices)[i] - minIdx;
				}
			}
		}
		else
		{
			outbuffer->setBaseVertex(outbuffer->getBaseVertex()+minIdx);
		}

		outbuffer->setIndexType(newIdxType);
		outbuffer->getMeshDataAndFormat()->setIndexBuffer(std::move(newIdxBuffer));

		if (verticesMustBeReordered)
		{
			// reorder vertices according to index buffer
#define _ACCESS_IDX(n) ((newIdxType == EIT_32BIT) ? *(reinterpret_cast<const uint32_t*>(indices)+(n)) : *(reinterpret_cast<const uint16_t*>(indices)+(n)))

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

void IMeshManipulator::requantizeMeshBuffer(ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric)
{
	CMeshManipulator::SAttrib newAttribs[EVAI_COUNT];
	for (size_t i = 0u; i < EVAI_COUNT; ++i)
		newAttribs[i].vaid = (E_VERTEX_ATTRIBUTE_ID)i;

	core::unordered_map<E_VERTEX_ATTRIBUTE_ID, core::vector<CMeshManipulator::SIntegerAttr>> attribsI;
	core::unordered_map<E_VERTEX_ATTRIBUTE_ID, core::vector<core::vectorSIMDf>> attribsF;
	for (size_t vaid = EVAI_ATTR0; vaid < (size_t)EVAI_COUNT; ++vaid)
	{
		const E_FORMAT type = _meshbuffer->getMeshDataAndFormat()->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)vaid);

		if (_meshbuffer->getMeshDataAndFormat()->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)vaid))
		{
			if (!isNormalizedFormat(type) && isIntegerFormat(type))
				attribsI[(E_VERTEX_ATTRIBUTE_ID)vaid] = CMeshManipulator::findBetterFormatI(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].prevType, _meshbuffer, (E_VERTEX_ATTRIBUTE_ID)vaid, _errMetric[vaid]);
			else
				attribsF[(E_VERTEX_ATTRIBUTE_ID)vaid] = CMeshManipulator::findBetterFormatF(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].prevType, _meshbuffer, (E_VERTEX_ATTRIBUTE_ID)vaid, _errMetric[vaid]);
		}
	}

	const size_t activeAttributeCount = attribsI.size() + attribsF.size();

#ifdef _IRR_DEBUG
	{
		core::unordered_set<size_t> sizesSet;
		for (core::unordered_map<E_VERTEX_ATTRIBUTE_ID, core::vector<CMeshManipulator::SIntegerAttr>>::iterator it = attribsI.begin(); it != attribsI.end(); ++it)
			sizesSet.insert(it->second.size());
		for (core::unordered_map<E_VERTEX_ATTRIBUTE_ID, core::vector<core::vectorSIMDf>>::iterator it = attribsF.begin(); it != attribsF.end(); ++it)
			sizesSet.insert(it->second.size());
		_IRR_DEBUG_BREAK_IF(sizesSet.size() != 1);
	}
#endif
	const size_t vertexCnt = (!attribsI.empty() ? attribsI.begin()->second.size() : (!attribsF.empty() ? attribsF.begin()->second.size() : 0));

	std::sort(newAttribs, newAttribs + EVAI_COUNT, std::greater<CMeshManipulator::SAttrib>()); // sort decreasing by size

	for (size_t i = 0u; i < activeAttributeCount; ++i)
	{
        const uint32_t typeSz = getTexelOrBlockBytesize(newAttribs[i].type);
        const size_t alignment = (typeSz / getFormatChannelCount(newAttribs[i].type) == 8u) ? 8ull : 4ull; // if format 64bit per channel, than align to 8

		newAttribs[i].offset = (i ? newAttribs[i - 1].offset + newAttribs[i - 1].size : 0u);
		const size_t mod = newAttribs[i].offset % alignment;
		newAttribs[i].offset += mod;
	}

	const size_t vertexSize = newAttribs[activeAttributeCount - 1].offset + newAttribs[activeAttributeCount - 1].size;

    IMeshDataFormatDesc<ICPUBuffer>* desc = _meshbuffer->getMeshDataAndFormat();
	auto newVertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(vertexCnt * vertexSize);

	for (size_t i = 0u; i < activeAttributeCount; ++i)
	{
		desc->setVertexAttrBuffer(core::smart_refctd_ptr(newVertexBuffer), newAttribs[i].vaid, newAttribs[i].type, vertexSize, newAttribs[i].offset);

		core::unordered_map<E_VERTEX_ATTRIBUTE_ID, core::vector<CMeshManipulator::SIntegerAttr>>::iterator iti = attribsI.find(newAttribs[i].vaid);
		if (iti != attribsI.end())
		{
			const core::vector<CMeshManipulator::SIntegerAttr>& attrVec = iti->second;
			for (size_t ai = 0u; ai < attrVec.size(); ++ai)
			{
				const bool check = _meshbuffer->setAttribute(attrVec[ai].pointer, newAttribs[i].vaid, ai);
				_IRR_DEBUG_BREAK_IF(!check)
			}
			continue;
		}

		core::unordered_map<E_VERTEX_ATTRIBUTE_ID, core::vector<core::vectorSIMDf>>::iterator itf = attribsF.find(newAttribs[i].vaid);
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
}


template<>
void CMeshManipulator::copyMeshBufferMemberVars<ICPUMeshBuffer>(ICPUMeshBuffer* _dst, const ICPUMeshBuffer* _src)
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
	_dst->setNormalnAttributeIx(
		_src->getNormalAttributeIx()
	);
    _dst->getMaterial() = _src->getMaterial();
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
	if (!_src)
		return nullptr;

	core::smart_refctd_ptr<ICPUMeshBuffer> dst;
    if (_src->getMeshBufferType() == asset::EMT_ANIMATED_SKINNED)
    {
        dst = core::make_smart_refctd_ptr<ICPUSkinnedMeshBuffer>();
		CMeshManipulator::copyMeshBufferMemberVars(static_cast<ICPUSkinnedMeshBuffer*>(dst.get()), static_cast<const ICPUSkinnedMeshBuffer*>(_src));
    }
    else
    {
        dst = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
		CMeshManipulator::copyMeshBufferMemberVars(dst.get(), _src);
    }

	if (!_src->getMeshDataAndFormat())
		return dst;

	core::smart_refctd_ptr<ICPUBuffer> idxBuffer;
	if (_src->getIndices())
	{
		idxBuffer = core::make_smart_refctd_ptr<ICPUBuffer>((_src->getIndexType() == EIT_16BIT ? 2 : 4) * _src->getIndexCount());
		memcpy(idxBuffer->getPointer(), _src->getIndices(), idxBuffer->getSize());
		dst->setIndexBufferOffset(0);
	}

    auto newDesc = core::make_smart_refctd_ptr<ICPUMeshDataFormatDesc>();
	const IMeshDataFormatDesc<ICPUBuffer>* oldDesc = _src->getMeshDataAndFormat();

	core::unordered_map<const ICPUBuffer*, E_VERTEX_ATTRIBUTE_ID> oldBuffers;
	for (size_t i = 0; i < EVAI_COUNT; ++i)
	{
		const ICPUBuffer* oldBuf = oldDesc->getMappedBuffer((E_VERTEX_ATTRIBUTE_ID)i);
		if (!oldBuf)
			continue;
		core::smart_refctd_ptr<ICPUBuffer> newBuf;

		core::unordered_map<const ICPUBuffer*, E_VERTEX_ATTRIBUTE_ID>::iterator itr = oldBuffers.find(oldBuf);
		if (itr == oldBuffers.end())
		{
			oldBuffers[oldBuf] = (E_VERTEX_ATTRIBUTE_ID)i;
			newBuf = core::make_smart_refctd_ptr<ICPUBuffer>(oldBuf->getSize());
			memcpy(newBuf->getPointer(), oldBuf->getPointer(), newBuf->getSize());
		}
		else
		{
			const ICPUBuffer* dupBuff = const_cast<const ICPUMeshDataFormatDesc*>(newDesc.get())->getMappedBuffer(itr->second);
			newBuf = core::smart_refctd_ptr<ICPUBuffer>(const_cast<ICPUBuffer*>(dupBuff));
		}

		newDesc->setVertexAttrBuffer(std::move(newBuf), (E_VERTEX_ATTRIBUTE_ID)i, oldDesc->getAttribFormat((E_VERTEX_ATTRIBUTE_ID)i),
			oldDesc->getMappedBufferStride((E_VERTEX_ATTRIBUTE_ID)i), oldDesc->getMappedBufferOffset((E_VERTEX_ATTRIBUTE_ID)i), oldDesc->getAttribDivisor((E_VERTEX_ATTRIBUTE_ID)i));
	}
	if (idxBuffer)
		newDesc->setIndexBuffer(std::move(idxBuffer));

	dst->setMeshDataAndFormat(std::move(newDesc));

	return dst;
}

void IMeshManipulator::filterInvalidTriangles(ICPUMeshBuffer* _input)
{
    if (!_input || !_input->getMeshDataAndFormat() || !_input->getIndices())
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
    void* const copy = _IRR_ALIGNED_MALLOC(size,_IRR_SIMD_ALIGNMENT);
    memcpy(copy, _input->getIndices(), size);

    struct Triangle
    {
        IdxT i[3];
    } *const begin = (Triangle*)copy, *const end = (Triangle*)((uint8_t*)copy + size);

    Triangle* const newEnd = std::remove_if(begin, end,
        [&_input](const Triangle& _t) {
            core::vectorSIMDf p0, p1, p2;
            const E_VERTEX_ATTRIBUTE_ID pvaid = _input->getPositionAttributeIx();
            _input->getAttribute(p0, pvaid, _t.i[0]);
            _input->getAttribute(p1, pvaid, _t.i[1]);
            _input->getAttribute(p2, pvaid, _t.i[2]);
			return core::length(core::cross(p1 - p0, p2 - p0)).x<=1.0e-19F;
    });
    const size_t newSize = std::distance(begin, newEnd) * sizeof(Triangle);

    auto newBuf = core::make_smart_refctd_ptr<ICPUBuffer>(newSize);
    memcpy(newBuf->getPointer(), copy, newSize);
    _IRR_ALIGNED_FREE(copy);
    _input->getMeshDataAndFormat()->setIndexBuffer(std::move(newBuf));
    _input->setIndexBufferOffset(0);
    _input->setIndexCount(newSize/sizeof(IdxT));
}
template void CMeshManipulator::_filterInvalidTriangles<uint16_t>(ICPUMeshBuffer* _input);
template void CMeshManipulator::_filterInvalidTriangles<uint32_t>(ICPUMeshBuffer* _input);

core::vector<core::vectorSIMDf> CMeshManipulator::findBetterFormatF(E_FORMAT* _outType, size_t* _outSize, E_FORMAT* _outPrevType, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric)
{
	const E_FORMAT thisType = _meshbuffer->getMeshDataAndFormat()->getAttribFormat(_attrId);

    if (!isFloatingPointFormat(thisType) && !isNormalizedFormat(thisType) && !isScaledFormat(thisType))
        return {};

	core::vector<core::vectorSIMDf> attribs;

	if (!_meshbuffer->getMeshDataAndFormat())
		return attribs;

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
		if (calcMaxQuantizationError({ thisType }, t, attribs, _errMetric))
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

core::vector<CMeshManipulator::SIntegerAttr> CMeshManipulator::findBetterFormatI(E_FORMAT* _outType, size_t* _outSize, E_FORMAT* _outPrevType, const ICPUMeshBuffer* _meshbuffer, E_VERTEX_ATTRIBUTE_ID _attrId, const SErrorMetric& _errMetric)
{
    const E_FORMAT thisType = _meshbuffer->getMeshDataAndFormat()->getAttribFormat(_attrId);

    if (!isIntegerFormat(thisType))
        return {};

    if (isBGRALayoutFormat(thisType))
        return {}; // BGRA is supported only by a few normalized types (this is function for integer types)

	core::vector<SIntegerAttr> attribs;

	if (!_meshbuffer->getMeshDataAndFormat())
		return attribs;

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

bool CMeshManipulator::calcMaxQuantizationError(const SAttribTypeChoice& _srcType, const SAttribTypeChoice& _dstType, const core::vector<core::vectorSIMDf>& _srcData, const SErrorMetric& _errMetric)
{
    using namespace video;

	using QuantF_t = core::vectorSIMDf(*)(const core::vectorSIMDf&, E_FORMAT, E_FORMAT);

	QuantF_t quantFunc = nullptr;

	if (_errMetric.method == EEM_ANGLES)
	{
		switch (_dstType.type)
		{
		case EF_R8_SNORM:
        case EF_R8G8_SNORM:
        case EF_R8G8B8_SNORM:
        case EF_R8G8B8A8_SNORM:
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint32_t*)buf)[0] = quantizeNormal888(_in);

				core::vectorSIMDf retval;
				ICPUMeshBuffer::getAttribute(retval, buf, EF_R8G8B8A8_SNORM);
				retval.w = 1.f;
				return retval;
			};
			break;
		case EF_A2R10G10B10_SNORM_PACK32:
		case EF_A2B10G10R10_SNORM_PACK32: // bgra
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint32_t*)buf)[0] = quantizeNormal2_10_10_10(_in);

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
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint64_t*)buf)[0] = quantizeNormal16_16_16(_in);

				core::vectorSIMDf retval;
				ICPUMeshBuffer::getAttribute(retval, buf, EF_R16G16B16A16_SNORM);
				retval.w = 1.f;
				return retval;
			};
			break;
        case EF_R16_SFLOAT:
        case EF_R16G16_SFLOAT:
        case EF_R16G16B16_SFLOAT:
        case EF_R16G16B16A16_SFLOAT:
			quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT, E_FORMAT) -> core::vectorSIMDf {
				uint8_t buf[32];
				((uint64_t*)buf)[0] = quantizeNormalHalfFloat(_in);

				core::vectorSIMDf retval;
				ICPUMeshBuffer::getAttribute(retval, buf, EF_R16G16B16A16_SFLOAT);
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
		quantFunc = [](const core::vectorSIMDf& _in, E_FORMAT _inType, E_FORMAT _outType) -> core::vectorSIMDf {
			uint8_t buf[32];
			ICPUMeshBuffer::setAttribute(_in, buf, _outType);
			core::vectorSIMDf out(0.f, 0.f, 0.f, 1.f);
			ICPUMeshBuffer::getAttribute(out, buf, _outType);
			return out;
		};
	}

	_IRR_DEBUG_BREAK_IF(!quantFunc)
	if (!quantFunc)
		return false;

	for (const core::vectorSIMDf& d : _srcData)
	{
		const core::vectorSIMDf quantized = quantFunc(d, _srcType.type, _dstType.type);
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


// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>


#include "nbl/asset/asset.h"
#include "nbl/asset/IRenderpassIndependentPipeline.h"
#include "nbl/asset/utils/CMeshManipulator.h"
#include "nbl/asset/utils/CSmoothNormalGenerator.h"
#include "nbl/asset/utils/CForsythVertexCacheOptimizer.h"
#include "nbl/asset/utils/COverdrawMeshOptimizer.h"

namespace nbl::asset
{

//! Flips the direction of surfaces. Changes backfacing triangles to frontfacing
//! triangles and vice versa.
//! \param mesh: Mesh on which the operation is performed.
void IMeshManipulator::flipSurfaces(ICPUMeshBuffer* inbuffer) 
{
	if (!inbuffer)
		return;
    auto* pipeline = inbuffer->getPipeline();
    const E_PRIMITIVE_TOPOLOGY primType = pipeline->getCachedCreationParams().primitiveAssembly.primitiveType;

    const uint32_t idxcnt = inbuffer->getIndexCount();
    if (!inbuffer->getIndices())
        return;

    if (inbuffer->getIndexType() == EIT_16BIT)
    {
        uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
        switch (primType)
        {
        case EPT_TRIANGLE_FAN:
            for (uint32_t i = 1; i < idxcnt; i += 2)
            {
                const uint16_t tmp = idx[i];
                idx[i] = idx[i + 1];
                idx[i + 1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
            if (idxcnt % 2) //odd
            {
                for (uint32_t i = 0; i < (idxcnt >> 1); i++)
                {
                    const uint16_t tmp = idx[i];
                    idx[i] = idx[idxcnt - 1 - i];
                    idx[idxcnt - 1 - i] = tmp;
                }
            }
            else //even
            {
                auto newIndexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>((idxcnt + 1u) * sizeof(uint16_t));
                auto* destPtr = reinterpret_cast<uint16_t*>(newIndexBuffer->getPointer());
                destPtr[0] = idx[0];
                memcpy(destPtr + 1u, idx, sizeof(uint16_t) * idxcnt);
                inbuffer->setIndexCount(idxcnt + 1u);
                SBufferBinding<ICPUBuffer> ixBufBinding{ 0u, std::move(newIndexBuffer) };
                inbuffer->setIndexBufferBinding(std::move(ixBufBinding));
            }
            break;
        case EPT_TRIANGLE_LIST:
            for (uint32_t i = 0; i < idxcnt; i += 3)
            {
                const uint16_t tmp = idx[i + 1];
                idx[i + 1] = idx[i + 2];
                idx[i + 2] = tmp;
            }
            break;
        default: break;
        }
    }
    else if (inbuffer->getIndexType() == EIT_32BIT)
    {
        uint32_t* idx = reinterpret_cast<uint32_t*>(inbuffer->getIndices());
        switch (primType)
        {
        case EPT_TRIANGLE_FAN:
            for (uint32_t i = 1; i < idxcnt; i += 2)
            {
                const uint32_t tmp = idx[i];
                idx[i] = idx[i + 1];
                idx[i + 1] = tmp;
            }
            break;
        case EPT_TRIANGLE_STRIP:
            if (idxcnt % 2) //odd
            {
                for (uint32_t i = 0; i < (idxcnt >> 1); i++)
                {
                    const uint32_t tmp = idx[i];
                    idx[i] = idx[idxcnt - 1 - i];
                    idx[idxcnt - 1 - i] = tmp;
                }
            }
            else //even
            {
                auto newIndexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>((idxcnt + 1u) * sizeof(uint32_t));
                auto* destPtr = reinterpret_cast<uint32_t*>(newIndexBuffer->getPointer());
                destPtr[0] = idx[0];
                memcpy(destPtr + 1u, idx, sizeof(uint32_t) * idxcnt);
                inbuffer->setIndexCount(idxcnt + 1);
                SBufferBinding<ICPUBuffer> ixBufBinding{ 0u, std::move(newIndexBuffer) };
                inbuffer->setIndexBufferBinding(std::move(ixBufBinding));
            }
            break;
        case EPT_TRIANGLE_LIST:
            for (uint32_t i = 0; i < idxcnt; i += 3)
            {
                const uint32_t tmp = idx[i + 1];
                idx[i + 1] = idx[i + 2];
                idx[i + 2] = tmp;
            }
            break;
        default: break;
        }
    }
}

core::smart_refctd_ptr<ICPUMeshBuffer> CMeshManipulator::createMeshBufferFetchOptimized(const ICPUMeshBuffer* _inbuffer)
{
	if (!_inbuffer)
		return nullptr;

    const auto* pipeline = _inbuffer->getPipeline();
    const void* ind = _inbuffer->getIndices();
	if (!pipeline || !ind)
		return nullptr;

	auto outbuffer = core::move_and_static_cast<ICPUMeshBuffer>(_inbuffer->clone(1u));
    outbuffer->setAttachedDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSet>(const_cast<ICPUDescriptorSet*>(_inbuffer->getAttachedDescriptorSet())));
    outbuffer->setSkin(
        SBufferBinding<ICPUBuffer>(reinterpret_cast<const SBufferBinding<ICPUBuffer>&>(_inbuffer->getInverseBindPoseBufferBinding())),
        SBufferBinding<ICPUBuffer>(reinterpret_cast<const SBufferBinding<ICPUBuffer>&>(_inbuffer->getJointAABBBufferBinding())),
        _inbuffer->getJointCount(),_inbuffer->getMaxJointsPerVertex()
    );

    constexpr uint32_t MAX_ATTRIBS = asset::ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT;

	// Find vertex count
	size_t vertexCount = IMeshManipulator::upperBoundVertexID(_inbuffer);

	core::unordered_set<const ICPUBuffer*> buffers;
	for (size_t i = 0; i < MAX_ATTRIBS; ++i)
        if (auto* buf = _inbuffer->getAttribBoundBuffer(i).buffer.get())
		    buffers.insert(buf);

	size_t offsets[MAX_ATTRIBS];
	memset(offsets, -1, sizeof(offsets));
	E_FORMAT types[MAX_ATTRIBS];
	if (buffers.size() != 1)
	{
		size_t lastOffset = 0u;
		size_t lastSize = 0u;
		for (size_t i = 0; i < MAX_ATTRIBS; ++i)
		{
			if (_inbuffer->isAttributeEnabled(i))
			{
				types[i] = _inbuffer->getAttribFormat(i);

                const uint32_t typeSz = getTexelOrBlockBytesize(types[i]);
                const size_t alignment = (typeSz/getFormatChannelCount(types[i]) == 8u) ? 8ull : 4ull; // if format 64bit per channel, then align to 8

				offsets[i] = lastOffset + lastSize;
				const size_t mod = offsets[i] % alignment;
				offsets[i] += mod;

				lastOffset = offsets[i];
                lastSize = typeSz;
			}
		}
		const size_t vertexSize = lastOffset + lastSize;

        constexpr uint32_t NEW_VTX_BUF_BINDING = 0u;
        auto& vtxParams = outbuffer->getPipeline()->getCachedCreationParams().vertexInput;
        vtxParams = SVertexInputParams();
        vtxParams.enabledAttribFlags = _inbuffer->getPipeline()->getCachedCreationParams().vertexInput.enabledAttribFlags;
        vtxParams.enabledBindingFlags = 1u << NEW_VTX_BUF_BINDING;
        vtxParams.bindings[NEW_VTX_BUF_BINDING].stride = vertexSize;
        vtxParams.bindings[NEW_VTX_BUF_BINDING].inputRate = SVertexInputBindingParams::EVIR_PER_VERTEX;

		auto newVertBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(vertexCount*vertexSize);
        outbuffer->setVertexBufferBinding({ 0u, core::smart_refctd_ptr(newVertBuffer) }, NEW_VTX_BUF_BINDING);
		for (size_t i = 0; i < MAX_ATTRIBS; ++i)
		{
			if (offsets[i] < 0xffffffff)
			{
                vtxParams.attributes[i].binding = NEW_VTX_BUF_BINDING;
                vtxParams.attributes[i].format = types[i];
                vtxParams.attributes[i].relativeOffset = offsets[i];
			}
		}
	}
	outbuffer->setBaseVertex(0);

	core::vector<uint32_t> activeAttribs;
	for (size_t i = 0; i < MAX_ATTRIBS; ++i)
		if (outbuffer->isAttributeEnabled(i))
			activeAttribs.push_back(i);

	uint32_t* remapBuffer = _NBL_NEW_ARRAY(uint32_t,vertexCount);
	memset(remapBuffer, 0xffffffffu, vertexCount*sizeof(uint32_t));

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
					_inbuffer->getAttribute(dst, activeAttribs[j], index);
					outbuffer->setAttribute(dst, activeAttribs[j], nextVert);
				}
				else
				{
					core::vectorSIMDf dst;
					_inbuffer->getAttribute(dst, activeAttribs[j], index);
					outbuffer->setAttribute(dst, activeAttribs[j], nextVert);
				}
			}

			remap = nextVert++;
		}

		if (idxType == EIT_32BIT)
			((uint32_t*)indices)[i] = remap;
		else
			((uint16_t*)indices)[i] = remap;
	}

    _NBL_DELETE_ARRAY(remapBuffer,vertexCount);

	_NBL_DEBUG_BREAK_IF(nextVert > vertexCount)

	return outbuffer;
}

//! Creates a copy of the mesh, which will only consist of unique primitives
core::smart_refctd_ptr<ICPUMeshBuffer> IMeshManipulator::createMeshBufferUniquePrimitives(ICPUMeshBuffer* inbuffer, bool _makeIndexBuf)
{
	if (!inbuffer)
		return nullptr;
    const ICPURenderpassIndependentPipeline* oldPipeline = inbuffer->getPipeline();
    if (!oldPipeline)
        return nullptr;

    const uint32_t idxCnt = inbuffer->getIndexCount();
    if (idxCnt<2u || !inbuffer->getIndices())
        return core::smart_refctd_ptr<ICPUMeshBuffer>(inbuffer); // yes we want an extra grab
    
    const auto& oldVtxParams = oldPipeline->getCachedCreationParams().vertexInput;
    
	auto clone = core::move_and_static_cast<ICPUMeshBuffer>(inbuffer->clone(0u));

    constexpr uint32_t NEW_VTX_BUF_BINDING = 0u;

    auto pipeline = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(oldPipeline->clone(0u));
    auto& vtxParams = pipeline->getCachedCreationParams().vertexInput;
    vtxParams = SVertexInputParams();

    vtxParams.enabledBindingFlags = (1u<<NEW_VTX_BUF_BINDING);
    vtxParams.enabledAttribFlags = oldVtxParams.enabledAttribFlags;

    constexpr size_t MAX_ATTRIBS = asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
	{
		size_t stride = 0;
		int32_t offset[MAX_ATTRIBS];
		size_t newAttribSizes[MAX_ATTRIBS];
		const uint8_t* sourceBuffers[MAX_ATTRIBS] = {NULL};
		size_t sourceBufferStrides[MAX_ATTRIBS];
		for (size_t i=0; i< MAX_ATTRIBS; i++)
		{
			const auto& vbuf = inbuffer->getAttribBoundBuffer(i);
			if (inbuffer->isAttributeEnabled(i) && vbuf.buffer)
			{
				offset[i] = stride;
				newAttribSizes[i] = getTexelOrBlockBytesize(inbuffer->getAttribFormat(i));
				stride += newAttribSizes[i];
				if (stride>=0xdeadbeefu)
					return nullptr;

				sourceBuffers[i] = reinterpret_cast<const uint8_t*>(vbuf.buffer->getPointer());
				sourceBuffers[i] += inbuffer->getAttribCombinedOffset(i);
				sourceBufferStrides[i] = inbuffer->getAttribStride(i);
			}
			else
				offset[i] = -1;
		}

        vtxParams.bindings[NEW_VTX_BUF_BINDING].inputRate = SVertexInputBindingParams::EVIR_PER_VERTEX;
        vtxParams.bindings[NEW_VTX_BUF_BINDING].stride = stride;

		auto vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(stride*idxCnt);
        clone->setVertexBufferBinding({0u, vertexBuffer}, 0u);
		for (size_t i=0; i<MAX_ATTRIBS; i++)
		{
            if (offset[i] >= 0)
            {
                vtxParams.attributes[i].binding = NEW_VTX_BUF_BINDING;
                vtxParams.attributes[i].format = inbuffer->getAttribFormat(i);
                vtxParams.attributes[i].relativeOffset = offset[i];
            }
		}

		uint8_t* destPointer = reinterpret_cast<uint8_t*>(vertexBuffer->getPointer());
		if (inbuffer->getIndexType()==EIT_16BIT)
		{
			uint16_t* idx = reinterpret_cast<uint16_t*>(inbuffer->getIndices());
			for (uint64_t i=0; i<idxCnt; i++,idx++)
			for (size_t j=0; j<MAX_ATTRIBS; j++)
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
			for (size_t j=0; j<MAX_ATTRIBS; j++)
			{
				if (offset[j]<0)
					continue;

				memcpy(destPointer,sourceBuffers[j]+(int64_t(*idx)+inbuffer->getBaseVertex())*sourceBufferStrides[j],newAttribSizes[j]);
				destPointer += newAttribSizes[j];
			}
		}

        clone->setPipeline(std::move(pipeline));

        if (_makeIndexBuf)
        {
            auto idxbuf = core::make_smart_refctd_ptr<ICPUBuffer>(idxCnt*(idxCnt<0x10000 ? 2u : 4u));
            if (idxCnt<0x10000u)
            {
                for (uint32_t i = 0u; i < idxCnt; ++i)
                    reinterpret_cast<uint16_t*>(idxbuf->getPointer())[i] = i;
                clone->setIndexType(EIT_16BIT);
            }
            else
            {
                for (uint32_t i = 0u; i < idxCnt; ++i)
                    reinterpret_cast<uint32_t*>(idxbuf->getPointer())[i] = i;
                clone->setIndexType(EIT_32BIT);
            }
            clone->setIndexBufferBinding({ 0u, std::move(idxbuf) });
        }
        else
        {
            clone->setIndexType(EIT_UNKNOWN);
        }
	}

	return clone;
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

    core::smart_refctd_ptr<ICPUMeshBuffer> outbuffer;
    if (makeNewMesh)
    {
        outbuffer = core::move_and_static_cast<ICPUMeshBuffer>(inbuffer->clone(0u));

        const auto normalAttr = inbuffer->getNormalAttributeIx();
        auto normalBinding = inbuffer->getBindingNumForAttribute(normalAttr);
        const auto oldPipeline = inbuffer->getPipeline();
        auto vertexParams = oldPipeline->getCachedCreationParams().vertexInput;
        bool notUniqueBinding = false;
        for (uint16_t attr=0u; attr<SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; attr++)
        if (attr!=normalAttr && (vertexParams.enabledAttribFlags&(0x1u<<attr))!=0u && vertexParams.attributes[attr].binding==normalBinding)
            notUniqueBinding = true;
        if (notUniqueBinding)
        {
            int32_t firstBindingNotUsed = hlsl::findLSB(vertexParams.enabledBindingFlags^0xffffu);
            assert(firstBindingNotUsed>0 && firstBindingNotUsed<SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT);
            normalBinding = static_cast<uint32_t>(firstBindingNotUsed);

            vertexParams.attributes[normalAttr].binding = normalBinding;
            vertexParams.enabledBindingFlags |= 0x1u<<normalBinding;
        }

        const auto normalFormatBytesize = asset::getTexelOrBlockBytesize(inbuffer->getAttribFormat(normalAttr));
        auto normalBuf = core::make_smart_refctd_ptr<ICPUBuffer>(normalFormatBytesize*IMeshManipulator::upperBoundVertexID(inbuffer));
        outbuffer->setVertexBufferBinding({0ull,std::move(normalBuf)},normalBinding);

        auto pipeline = core::move_and_static_cast<ICPURenderpassIndependentPipeline>(oldPipeline->clone(0u));
        vertexParams.bindings[normalBinding].stride = normalFormatBytesize;
        vertexParams.attributes[normalAttr].relativeOffset = 0u;
        pipeline->getCachedCreationParams().vertexInput = vertexParams;
        outbuffer->setPipeline(std::move(pipeline));
    }
    else
        outbuffer = core::smart_refctd_ptr<ICPUMeshBuffer>(inbuffer);
	CSmoothNormalGenerator::calculateNormals(outbuffer.get(), epsilon, normalAttrID, vxcmp);

	return outbuffer;
}

// Used by createMeshBufferWelded only
static bool cmpVertices(ICPUMeshBuffer* _inbuf, const void* _va, const void* _vb, size_t _vsize, const IMeshManipulator::SErrorMetric* _errMetrics)
{
    auto cmpInteger = [](uint32_t* _a, uint32_t* _b, size_t _n) -> bool {
        return !memcmp(_a, _b, _n*4);
    };

    constexpr uint32_t MAX_ATTRIBS = ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT;

    const uint8_t* va = (uint8_t*)_va, *vb = (uint8_t*)_vb;
    for (size_t i = 0u; i < MAX_ATTRIBS; ++i)
    {
        if (!_inbuf->isAttributeEnabled(i))
            continue;

        const auto atype = _inbuf->getAttribFormat(i);
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
    if (!inbuffer || !inbuffer->getPipeline())
        return nullptr;

    constexpr uint32_t MAX_ATTRIBS = ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT;

    bool bufferPresent[MAX_ATTRIBS];

    size_t vertexAttrSize[MAX_ATTRIBS];
    size_t vertexSize = 0;
    for (size_t i=0; i<MAX_ATTRIBS; i++)
    {
        const auto& buf = inbuffer->getAttribBoundBuffer(i).buffer;
        bufferPresent[i] = inbuffer->isAttributeEnabled(i);
        if (bufferPresent[i] && buf)
        {
            const E_FORMAT componentType = inbuffer->getAttribFormat(i);
            vertexAttrSize[i] = getTexelOrBlockBytesize(componentType);
            vertexSize += vertexAttrSize[i];
        }
    }

    auto cmpfunc = [&, inbuffer, vertexSize, _errMetrics](const void* _va, const void* _vb) {
        return cmpVertices(inbuffer, _va, _vb, vertexSize, _errMetrics);
    };

    const uint32_t vertexCount = IMeshManipulator::upperBoundVertexID(inbuffer);
    const E_INDEX_TYPE oldIndexType = inbuffer->getIndexType();

    if (!vertexCount)
        return nullptr;

    // reset redirect list
    uint32_t* redirects = new uint32_t[vertexCount];

    uint32_t maxRedirect = 0;

    uint8_t* epicData = (uint8_t*)_NBL_ALIGNED_MALLOC(vertexSize*vertexCount,_NBL_SIMD_ALIGNMENT);
    for (auto i=0u; i<vertexCount; i++)
    {
        uint8_t* currentVertexPtr = epicData+i*vertexSize;
        for (size_t k=0; k<MAX_ATTRIBS; k++)
        {
            if (!bufferPresent[k])
                continue;

            size_t stride = inbuffer->getAttribStride(k);
            uint8_t* sourcePtr = inbuffer->getAttribPointer(k) + i*stride;
            memcpy(currentVertexPtr,sourcePtr,vertexAttrSize[k]);
            currentVertexPtr += vertexAttrSize[k];
        }
    }

    for (auto i=0u; i<vertexCount; i++)
    {
        uint32_t redir = i;
        for (auto j=0u; j<vertexCount; ++j) // TODO: Use spatial hash for this like in the smooth normal computation @Przemog
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
    _NBL_ALIGNED_FREE(epicData);

    void* oldIndices = inbuffer->getIndices();
    core::smart_refctd_ptr<ICPUMeshBuffer> clone;
    if (makeNewMesh)
        clone = core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(inbuffer->clone(0u));
    else
    {
        if (!oldIndices)
        {
            inbuffer->setIndexBufferBinding({ 0u, core::make_smart_refctd_ptr<ICPUBuffer>((maxRedirect >= 0x10000u ? sizeof(uint32_t) : sizeof(uint16_t)) * inbuffer->getIndexCount()) });
            inbuffer->setIndexType(maxRedirect>=0x10000u ? EIT_32BIT:EIT_16BIT);
        }
    }

    // TODO: reduce the code duplication via the use of a generic lambda (with a `auto*`)
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
    const auto oldPipeline = _inbuffer->getPipeline();
	auto outbuffer = core::move_and_static_cast<ICPUMeshBuffer>(_inbuffer->clone(oldPipeline ? 1u:0u));
	if (!oldPipeline)
		return outbuffer;

    // restore shared skeleton and descriptor set
    outbuffer->setAttachedDescriptorSet(core::smart_refctd_ptr<ICPUDescriptorSet>(const_cast<ICPUDescriptorSet*>(_inbuffer->getAttachedDescriptorSet())));
    outbuffer->setSkin(
        SBufferBinding<ICPUBuffer>(reinterpret_cast<const SBufferBinding<ICPUBuffer>&>(_inbuffer->getInverseBindPoseBufferBinding())),
        SBufferBinding<ICPUBuffer>(reinterpret_cast<const SBufferBinding<ICPUBuffer>&>(_inbuffer->getJointAABBBufferBinding())),
        _inbuffer->getJointCount(),_inbuffer->getMaxJointsPerVertex()
    );

    // make index buffer 0,1,2,3,4,... if nothing's mapped
	// make 32bit index buffer if 16bit one is present
	// convert index buffer for triangle primitives
    constexpr auto canonicalMeshBufferIndexType = EIT_32BIT;
    IMeshManipulator::homogenizePrimitiveTypeAndIndices(&outbuffer.get(),&outbuffer.get()+1,EPT_TRIANGLE_LIST,canonicalMeshBufferIndexType);
    if (outbuffer->getPipeline()->getCachedCreationParams().primitiveAssembly.primitiveType != EPT_TRIANGLE_LIST)
		return nullptr;

	// STEP: weld
    createMeshBufferWelded(outbuffer.get(), _errMetric, false, false);

    // STEP: filter invalid triangles
    if (!_inbuffer->isSkinned())
        filterInvalidTriangles(outbuffer.get());

	// STEP: overdraw optimization
	COverdrawMeshOptimizer::createOptimized(outbuffer.get(),outbuffer.get());

	// STEP: Forsyth
	{
		uint32_t* indices = reinterpret_cast<uint32_t*>(outbuffer->getIndices());
		CForsythVertexCacheOptimizer forsyth;
        const uint32_t vertexCount = IMeshManipulator::upperBoundVertexID(_inbuffer);
		forsyth.optimizeTriangleOrdering(vertexCount, outbuffer->getIndexCount(), indices, indices);
	}

	// STEP: prefetch optimization
	outbuffer = CMeshManipulator::createMeshBufferFetchOptimized(outbuffer.get()); // here we also get interleaved attributes (single vertex buffer)
	
	// STEP: requantization
	requantizeMeshBuffer(outbuffer.get(), _errMetric);

	// STEP: reduce index buffer to 16bit or completely get rid of it
	{
		const void* const indices = outbuffer->getIndices();
		uint32_t* indicesCopy = (uint32_t*)_NBL_ALIGNED_MALLOC(outbuffer->getIndexCount()*4,_NBL_SIMD_ALIGNMENT);
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

		_NBL_ALIGNED_FREE(indicesCopy);

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
        outbuffer->setIndexBufferBinding({ 0u, std::move(newIdxBuffer) });

		if (verticesMustBeReordered)
		{
            auto* pipeline = outbuffer->getPipeline();

			// reorder vertices according to index buffer
#define _ACCESS_IDX(n) ((newIdxType == EIT_32BIT) ? *(reinterpret_cast<const uint32_t*>(indices)+(n)) : *(reinterpret_cast<const uint16_t*>(indices)+(n)))

            const uint32_t posId = outbuffer->getPositionAttributeIx();
            const size_t bufsz = outbuffer->getAttribBoundBuffer(posId).buffer->getSize();

			const size_t vertexSize = pipeline->getCachedCreationParams().vertexInput.bindings[0].stride;
			uint8_t* const v = (uint8_t*)(outbuffer->getAttribBoundBuffer(posId).buffer->getPointer()); // after prefetch optim. we have guarantee of single vertex buffer so we can do like this
			uint8_t* const vCopy = (uint8_t*)_NBL_ALIGNED_MALLOC(bufsz, _NBL_SIMD_ALIGNMENT);
			memcpy(vCopy, v, bufsz);

			size_t baseVtx = outbuffer->getBaseVertex();
			for (size_t i = 0; i < outbuffer->getIndexCount(); ++i)
			{
				const uint32_t idx = _ACCESS_IDX(i+baseVtx);
				if (idx != i+baseVtx)
					memcpy(v + (vertexSize*(i + baseVtx)), vCopy + (vertexSize*idx), vertexSize);
			}
#undef _ACCESS_IDX
			_NBL_ALIGNED_FREE(vCopy);
		}
	}

	return outbuffer;
}

void IMeshManipulator::requantizeMeshBuffer(ICPUMeshBuffer* _meshbuffer, const SErrorMetric* _errMetric)
{
    constexpr uint32_t MAX_ATTRIBS = ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT;

	CMeshManipulator::SAttrib newAttribs[MAX_ATTRIBS];
	for (size_t i = 0u; i < MAX_ATTRIBS; ++i)
		newAttribs[i].vaid = i;

    CQuantNormalCache quantizationCache;

	core::unordered_map<uint32_t, core::vector<CMeshManipulator::SIntegerAttr>> attribsI;
	core::unordered_map<uint32_t, core::vector<core::vectorSIMDf>> attribsF;
	for (size_t vaid = 0u; vaid < MAX_ATTRIBS; ++vaid)
	{
        const E_FORMAT type = _meshbuffer->getAttribFormat(vaid);

        const auto& vbuf = _meshbuffer->getAttribBoundBuffer(vaid).buffer;
		if (_meshbuffer->isAttributeEnabled(vaid) && vbuf)
		{
			if (!isNormalizedFormat(type) && isIntegerFormat(type))
				attribsI[vaid] = CMeshManipulator::findBetterFormatI(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].prevType, _meshbuffer, vaid, _errMetric[vaid]);
			else
				attribsF[vaid] = CMeshManipulator::findBetterFormatF(&newAttribs[vaid].type, &newAttribs[vaid].size, &newAttribs[vaid].prevType, _meshbuffer, vaid, _errMetric[vaid], quantizationCache);
		}
	}

	const size_t activeAttributeCount = attribsI.size() + attribsF.size();

#ifdef _NBL_DEBUG
	{
		core::unordered_set<size_t> sizesSet;
		for (core::unordered_map<uint32_t, core::vector<CMeshManipulator::SIntegerAttr>>::iterator it = attribsI.begin(); it != attribsI.end(); ++it)
			sizesSet.insert(it->second.size());
		for (core::unordered_map<uint32_t, core::vector<core::vectorSIMDf>>::iterator it = attribsF.begin(); it != attribsF.end(); ++it)
			sizesSet.insert(it->second.size());
		_NBL_DEBUG_BREAK_IF(sizesSet.size() != 1);
	}
#endif
	const size_t vertexCnt = (!attribsI.empty() ? attribsI.begin()->second.size() : (!attribsF.empty() ? attribsF.begin()->second.size() : 0));

	std::sort(newAttribs, newAttribs + MAX_ATTRIBS, std::greater<CMeshManipulator::SAttrib>()); // sort decreasing by size

	for (size_t i = 0u; i < activeAttributeCount; ++i)
	{
        const uint32_t typeSz = getTexelOrBlockBytesize(newAttribs[i].type);
        const size_t alignment = (typeSz / getFormatChannelCount(newAttribs[i].type) == 8u) ? 8ull : 4ull; // if format 64bit per channel, than align to 8

		newAttribs[i].offset = (i ? newAttribs[i - 1].offset + newAttribs[i - 1].size : 0u);
		const size_t mod = newAttribs[i].offset % alignment;
		newAttribs[i].offset += mod;
	}

	const size_t vertexSize = newAttribs[activeAttributeCount - 1].offset + newAttribs[activeAttributeCount - 1].size;

	auto newVertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(vertexCnt * vertexSize);

    constexpr uint32_t VTX_BUF_BINDING = 0u;
    assert(_meshbuffer->getVertexBufferBindings()[0].buffer);
    assert(_meshbuffer->isVertexAttribBufferBindingEnabled(VTX_BUF_BINDING));
    _meshbuffer->setVertexBufferBinding({ 0u, core::smart_refctd_ptr(newVertexBuffer) }, VTX_BUF_BINDING);

    auto* pipeline = _meshbuffer->getPipeline();
    auto& vtxParams = pipeline->getCachedCreationParams().vertexInput;

    vtxParams.bindings[VTX_BUF_BINDING].stride = vertexSize;
    vtxParams.bindings[VTX_BUF_BINDING].inputRate = SVertexInputBindingParams::EVIR_PER_VERTEX;
	for (size_t i = 0u; i < activeAttributeCount; ++i)
	{
        const uint32_t vaid = newAttribs[i].vaid;
        vtxParams.attributes[vaid].binding = VTX_BUF_BINDING;
        vtxParams.attributes[vaid].format = newAttribs[i].type;
        vtxParams.attributes[vaid].relativeOffset = newAttribs[i].offset;

		core::unordered_map<uint32_t, core::vector<CMeshManipulator::SIntegerAttr>>::iterator iti = attribsI.find(newAttribs[i].vaid);
		if (iti != attribsI.end())
		{
			const core::vector<CMeshManipulator::SIntegerAttr>& attrVec = iti->second;
			for (size_t ai = 0u; ai < attrVec.size(); ++ai)
			{
				const bool check = _meshbuffer->setAttribute(attrVec[ai].pointer, newAttribs[i].vaid, ai);
				_NBL_DEBUG_BREAK_IF(!check)
			}
			continue;
		}

		core::unordered_map<uint32_t, core::vector<core::vectorSIMDf>>::iterator itf = attribsF.find(newAttribs[i].vaid);
		if (itf != attribsF.end())
		{
			const core::vector<core::vectorSIMDf>& attrVec = itf->second;
			for (size_t ai = 0u; ai < attrVec.size(); ++ai)
			{
				const bool check = _meshbuffer->setAttribute(attrVec[ai], newAttribs[i].vaid, ai);
				_NBL_DEBUG_BREAK_IF(!check)
			}
		}
	}
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
    const uint32_t cnt = IMeshManipulator::upperBoundVertexID(_meshbuffer);
    for (uint32_t idx = 0u; idx < cnt; ++idx)
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
    const uint32_t cnt = IMeshManipulator::upperBoundVertexID(_meshbuffer);
    for (uint32_t idx = 0u; idx < cnt; ++idx)
	{
        _meshbuffer->getAttribute(attr.pointer, _attrId, idx);
		attribs.push_back(attr);
		for (uint32_t i = 0; i < cpa; ++i)
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
				((CQuantNormalCache::value_type_t<EF_R8G8B8_SNORM>*)buf)[0] = _cache.quantize<EF_R8G8B8_SNORM>(_in);

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
				((CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>*)buf)[0] = _cache.quantize<EF_A2B10G10R10_SNORM_PACK32>(_in);

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
				((CQuantNormalCache::value_type_t<EF_R16G16B16_SNORM>*)buf)[0] = _cache.quantize<EF_R16G16B16_SNORM>(_in);

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

core::smart_refctd_ptr<ICPUBuffer> IMeshManipulator::idxBufferFromLineStripsToLines(const void* _input, uint32_t& _idxCount, E_INDEX_TYPE _inIndexType, E_INDEX_TYPE _outIndexType)
{
    if (_inIndexType == EIT_16BIT)
    {
        if (_outIndexType == EIT_16BIT)
            return CMeshManipulator::lineStripsToLines<uint16_t,uint16_t>(_input, _idxCount);
        else
            return CMeshManipulator::lineStripsToLines<uint16_t,uint32_t>(_input, _idxCount);
    }
	else if (_inIndexType == EIT_32BIT)
    {
        if (_outIndexType == EIT_16BIT)
            return CMeshManipulator::lineStripsToLines<uint32_t,uint16_t>(_input, _idxCount);
        else
            return CMeshManipulator::lineStripsToLines<uint32_t,uint32_t>(_input, _idxCount);
    }
	return nullptr;
}

core::smart_refctd_ptr<ICPUBuffer> IMeshManipulator::idxBufferFromTriangleStripsToTriangles(const void* _input, uint32_t& _idxCount, E_INDEX_TYPE _inIndexType, E_INDEX_TYPE _outIndexType)
{
	if (_inIndexType == EIT_16BIT)
    {
        if (_outIndexType == EIT_16BIT)
            return CMeshManipulator::triangleStripsToTriangles<uint16_t,uint16_t>(_input, _idxCount);
        else
            return CMeshManipulator::triangleStripsToTriangles<uint16_t,uint32_t>(_input, _idxCount);
    }
	else if (_inIndexType == EIT_32BIT)
    {
        if (_outIndexType == EIT_16BIT)
            return CMeshManipulator::triangleStripsToTriangles<uint32_t,uint16_t>(_input, _idxCount);
        else
            return CMeshManipulator::triangleStripsToTriangles<uint32_t,uint32_t>(_input, _idxCount);
    }
	return nullptr;
}

core::smart_refctd_ptr<ICPUBuffer> IMeshManipulator::idxBufferFromTrianglesFanToTriangles(const void* _input, uint32_t& _idxCount, E_INDEX_TYPE _inIndexType, E_INDEX_TYPE _outIndexType)
{
	if (_inIndexType == EIT_16BIT)
    {
        if (_outIndexType == EIT_16BIT)
            return CMeshManipulator::trianglesFanToTriangles<uint16_t,uint16_t>(_input, _idxCount);
        else
            return CMeshManipulator::trianglesFanToTriangles<uint16_t,uint32_t>(_input, _idxCount);
    }
	else if (_inIndexType == EIT_32BIT)
    {
        if (_outIndexType == EIT_16BIT)
            return CMeshManipulator::trianglesFanToTriangles<uint32_t,uint16_t>(_input, _idxCount);
        else
            return CMeshManipulator::trianglesFanToTriangles<uint32_t,uint32_t>(_input, _idxCount);
    }
	return nullptr;
}

float IMeshManipulator::DistanceToLine(core::vectorSIMDf P0, core::vectorSIMDf P1, core::vectorSIMDf InPoint) 
{
    core::vectorSIMDf PointToStart = InPoint - P0;
    core::vectorSIMDf Diff = core::cross(P0 - P1, PointToStart);

    return core::dot(Diff, Diff).x;
}

float IMeshManipulator::DistanceToPlane(core::vectorSIMDf InPoint, core::vectorSIMDf PlanePoint, core::vectorSIMDf PlaneNormal) 
{
    core::vectorSIMDf PointToPlane = InPoint - PlanePoint;

    return (core::dot(PointToPlane, PlaneNormal).x >= 0) ? core::abs(core::dot(PointToPlane, PlaneNormal).x) : 0;
}

core::matrix3x4SIMD IMeshManipulator::calculateOBB(const nbl::asset::ICPUMeshBuffer* meshbuffer) 
{
    auto FindMinMaxProj = [&](const core::vectorSIMDf& Dir, const core::vectorSIMDf Extrema[]) -> core::vectorSIMDf
    {
        float MinPoint, MaxPoint;
        MinPoint = MaxPoint = core::dot(Dir, Extrema[0]).x;

        for (int i = 1; i < 12; i++) {
            float Proj = core::dot(Dir, Extrema[i]).x;
            if (MinPoint > Proj) MinPoint = Proj;
            if (MaxPoint < Proj) MaxPoint = Proj;
        }

        return core::vectorSIMDf(MaxPoint, MinPoint, 0);
    };

    auto ComputeAxis = [&](const core::vectorSIMDf& P0, const core::vectorSIMDf& P1, const core::vectorSIMDf& P2, core::vectorSIMDf* AxesEdge, float& PrevQuality, const core::vectorSIMDf Extrema[]) -> void
    {
        core::vectorSIMDf e0 = P1 - P0;
        core::vectorSIMDf Edges[3];
        Edges[0] = e0 / core::length(e0);
        Edges[1] = core::cross(P2 - P1, P1 - P0);
        Edges[1] = Edges[1] / core::length(Edges[1]);
        Edges[2] = core::cross(Edges[0], Edges[1]);

        core::vectorSIMDf Edge10Proj = FindMinMaxProj(Edges[0], Extrema);
        core::vectorSIMDf Edge20Proj = FindMinMaxProj(Edges[1], Extrema);
        core::vectorSIMDf Edge30Proj = FindMinMaxProj(Edges[2], Extrema);
        core::vectorSIMDf Max2 = core::vectorSIMDf(Edge10Proj.x, Edge20Proj.x, Edge30Proj.x);
        core::vectorSIMDf Min2 = core::vectorSIMDf(Edge10Proj.y, Edge20Proj.y, Edge30Proj.y);
        core::vectorSIMDf Diff = Max2 - Min2;
        float Quality = Diff.x * Diff.y + Diff.x * Diff.z + Diff.y * Diff.z;

        if (Quality < PrevQuality) {
            PrevQuality = Quality;
            for (int i = 0; i < 3; i++) {
                AxesEdge[i] = Edges[i];
            }
        }
    };

    core::vectorSIMDf Extrema[12];
    float A = (core::sqrt(5.0f) - 1.0f) / 2.0f;
    core::vectorSIMDf N[6];
    N[0] = core::vectorSIMDf(0, 1, A);
    N[1] = core::vectorSIMDf(0, 1, -A);
    N[2] = core::vectorSIMDf(1, A, 0);
    N[3] = core::vectorSIMDf(1, -A, 0);
    N[4] = core::vectorSIMDf(A, 0, 1);
    N[5] = core::vectorSIMDf(A, 0, -1);
    float Bs[12];
    float B;
    int indexcount = meshbuffer->getIndexCount();
    core::vectorSIMDf CachedVertex = meshbuffer->getPosition(meshbuffer->getIndexValue(0));
    core::vectorSIMDf AABBMax = CachedVertex;
    core::vectorSIMDf AABBMin = CachedVertex;
    for (int k = 0; k < 12; k += 2) {
        B = core::dot(N[k / 2], CachedVertex).x;
        Extrema[k] = core::vectorSIMDf(CachedVertex.x, CachedVertex.y, CachedVertex.z); Bs[k] = B;
        Extrema[k + 1] = core::vectorSIMDf(CachedVertex.x, CachedVertex.y, CachedVertex.z); Bs[k + 1] = B;
    }
    for (uint32_t j = 1u; j < indexcount; j += 1u) {
        CachedVertex = meshbuffer->getPosition(meshbuffer->getIndexValue(j));
        for (int k = 0; k < 12; k += 2) {
            B = core::dot(N[k / 2], CachedVertex).x;
            if (B > Bs[k] || j == 0) { Extrema[k] = core::vectorSIMDf(CachedVertex.x, CachedVertex.y, CachedVertex.z); Bs[k] = B; }
            if (B < Bs[k + 1] || j == 0) { Extrema[k + 1] = core::vectorSIMDf(CachedVertex.x, CachedVertex.y, CachedVertex.z); Bs[k + 1] = B; }
        }
        AABBMax = core::max(AABBMax, CachedVertex);
        AABBMin = core::min(AABBMin, CachedVertex);
    }

    int LBTE1 = -1;
    float MaxDiff = 0;
    for (int i = 0; i < 12; i += 2) {
        core::vectorSIMDf C = (Extrema[i]) - (Extrema[i + 1]); float TempDiff = core::dot(C, C).x; if (TempDiff > MaxDiff) { MaxDiff = TempDiff; LBTE1 = i; }
    }
    assert(LBTE1 != -1);

    core::vectorSIMDf P0 = Extrema[LBTE1];
    core::vectorSIMDf P1 = Extrema[LBTE1 + 1];

    int LBTE3 = 0;
    float MaxDist = 0;
    int RemoveAt = 0;

    for (int i = 0; i < 10; i++) {
        int index = i;
        if (index >= LBTE1) index += 2;
        float TempDist = DistanceToLine(P0, P1, core::vectorSIMDf(Extrema[index].x, Extrema[index].y, Extrema[index].z));
        if (TempDist > MaxDist || i == 0) {
            MaxDist = TempDist;
            LBTE3 = index;
            RemoveAt = i;
        }
    }

    core::vectorSIMDf P2 = Extrema[LBTE3];
    core::vectorSIMDf ExtremaRemainingTemp[9];
    for (int i = 0; i < 9; i++) {
        int index = i;
        if (index >= RemoveAt) index += 1;
        if (index >= LBTE1) index += 2;
        ExtremaRemainingTemp[i] = core::vectorSIMDf(Extrema[index].x, Extrema[index].y, Extrema[index].z, index);
    }

    float MaxDistPlane = -9999999.0f;
    float MinDistPlane = -9999999.0f;
    float TempDistPlane = 0;
    core::vectorSIMDf Q0 = core::vectorSIMDf(0, 0, 0);
    core::vectorSIMDf Q1 = core::vectorSIMDf(0, 0, 0);
    core::vectorSIMDf Norm = core::cross(P2 - P1, P2 - P0);
    Norm /= core::length(Norm);
    for (int i = 0; i < 9; i++) {
        TempDistPlane = DistanceToPlane(core::vectorSIMDf(ExtremaRemainingTemp[i].x, ExtremaRemainingTemp[i].y, ExtremaRemainingTemp[i].z), P0, Norm);
        if (TempDistPlane > MaxDistPlane || i == 0) {
            MaxDistPlane = TempDistPlane;
            Q0 = Extrema[(int)ExtremaRemainingTemp[i].w];
        }
        TempDistPlane = DistanceToPlane(core::vectorSIMDf(ExtremaRemainingTemp[i].x, ExtremaRemainingTemp[i].y, ExtremaRemainingTemp[i].z), P0, -Norm);
        if (TempDistPlane > MinDistPlane || i == 0) {
            MinDistPlane = TempDistPlane;
            Q1 = Extrema[(int)ExtremaRemainingTemp[i].w];
        }
    }

    float BestQuality = 99999999999999.0f;
    core::vectorSIMDf BestAxis[3];
    ComputeAxis(P0, P1, P2, BestAxis, BestQuality, Extrema);
    ComputeAxis(P2, P0, P1, BestAxis, BestQuality, Extrema);
    ComputeAxis(P1, P2, P0, BestAxis, BestQuality, Extrema);

    ComputeAxis(P1, Q0, P0, BestAxis, BestQuality, Extrema);
    ComputeAxis(P0, P1, Q0, BestAxis, BestQuality, Extrema);
    ComputeAxis(Q0, P0, P1, BestAxis, BestQuality, Extrema);

    ComputeAxis(P2, Q0, P0, BestAxis, BestQuality, Extrema);
    ComputeAxis(P0, P2, Q0, BestAxis, BestQuality, Extrema);
    ComputeAxis(Q0, P0, P2, BestAxis, BestQuality, Extrema);

    ComputeAxis(P1, Q0, P2, BestAxis, BestQuality, Extrema);
    ComputeAxis(P2, P1, Q0, BestAxis, BestQuality, Extrema);
    ComputeAxis(Q0, P2, P1, BestAxis, BestQuality, Extrema);

    ComputeAxis(P1, Q1, P0, BestAxis, BestQuality, Extrema);
    ComputeAxis(P0, P1, Q1, BestAxis, BestQuality, Extrema);
    ComputeAxis(Q1, P0, P1, BestAxis, BestQuality, Extrema);

    ComputeAxis(P2, Q1, P0, BestAxis, BestQuality, Extrema);
    ComputeAxis(P0, P2, Q1, BestAxis, BestQuality, Extrema);
    ComputeAxis(Q1, P0, P2, BestAxis, BestQuality, Extrema);

    ComputeAxis(P1, Q1, P2, BestAxis, BestQuality, Extrema);
    ComputeAxis(P2, P1, Q1, BestAxis, BestQuality, Extrema);
    ComputeAxis(Q1, P2, P1, BestAxis, BestQuality, Extrema);

    core::matrix3x4SIMD TransMat = core::matrix3x4SIMD(
        BestAxis[0].x, BestAxis[1].x, BestAxis[2].x, 0,
        BestAxis[0].y, BestAxis[1].y, BestAxis[2].y, 0,
        BestAxis[0].z, BestAxis[1].z, BestAxis[2].z, 0);

    core::vectorSIMDf MinPoint;
    core::vectorSIMDf MaxPoint;
    CachedVertex = meshbuffer->getPosition(meshbuffer->getIndexValue(0));
    MinPoint = core::vectorSIMDf(core::dot(BestAxis[0], CachedVertex).x, core::dot(BestAxis[1], CachedVertex).x, core::dot(BestAxis[2], CachedVertex).x);
    MaxPoint = MinPoint;
    for (uint32_t j = 1u; j < indexcount; j += 1u)
    {
        CachedVertex = meshbuffer->getPosition(meshbuffer->getIndexValue(j));
        core::vectorSIMDf Proj = core::vectorSIMDf(core::dot(BestAxis[0], CachedVertex).x, core::dot(BestAxis[1], CachedVertex).x, core::dot(BestAxis[2], CachedVertex).x);
        MinPoint = core::min(MinPoint, Proj);
        MaxPoint = core::max(MaxPoint, Proj);
    }

    core::vectorSIMDf OBBDiff = MaxPoint - MinPoint;
    float OBBQuality = OBBDiff.x * OBBDiff.y + OBBDiff.y * OBBDiff.z + OBBDiff.z * OBBDiff.x;

    core::vectorSIMDf ABBDiff = AABBMax - AABBMin;
    float ABBQuality = ABBDiff.x * ABBDiff.y + ABBDiff.y * ABBDiff.z + ABBDiff.z * ABBDiff.x;
    core::matrix3x4SIMD scaleMat;
    core::matrix3x4SIMD translationMat;
    translationMat.setTranslation(-(MinPoint) / OBBDiff);
    scaleMat.setScale(OBBDiff);
    TransMat = core::concatenateBFollowedByA(TransMat, scaleMat);
    TransMat = core::concatenateBFollowedByA(TransMat, translationMat);
    if (ABBQuality < OBBQuality) {
        translationMat.setTranslation(-(AABBMin) / ABBDiff);
        scaleMat.setScale(ABBDiff);
        TransMat = core::matrix3x4SIMD(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
        TransMat = core::concatenateBFollowedByA(TransMat, scaleMat);
        TransMat = core::concatenateBFollowedByA(TransMat, translationMat);
    }

    return TransMat;
}

} // end namespace nbl::asset


// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/asset/asset.h"

#include <functional>
#include <algorithm>

#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/asset/utils/CVertexWelder.h"
#include "nbl/asset/utils/CSmoothNormalGenerator.h"


namespace nbl::asset
{

core::smart_refctd_ptr<ICPUPolygonGeometry> CPolygonGeometryManipulator::createUnweldedList(const ICPUPolygonGeometry* inGeo)
{
    const auto* indexing = inGeo->getIndexingCallback();
    if (!indexing)
        return nullptr;

    const auto indexView = inGeo->getIndexView();
    const auto primCount = inGeo->getPrimitiveCount();
    const uint8_t degree = indexing->degree();
    const auto outIndexCount = primCount*degree;
    if (outIndexCount<primCount)
        return nullptr;

    const auto outGeometry = core::move_and_static_cast<ICPUPolygonGeometry>(inGeo->clone(0u));

    auto* outGeo = outGeometry.get();
    outGeo->setIndexing(IPolygonGeometryBase::NGonList(degree));

    auto createOutView = [&](const ICPUPolygonGeometry::SDataView& inView) -> ICPUPolygonGeometry::SDataView
    {
        if (!inView)
            return {};
        auto buffer = ICPUBuffer::create({ outIndexCount*inView.composed.stride , inView.src.buffer->getUsageFlags() });
        return {
            .composed = inView.composed,
            .src = {.offset = 0, .size = buffer->getSize(), .buffer = std::move(buffer)}
        };
    };

    const auto inIndexView = inGeo->getIndexView();
    auto outIndexView = createOutView(inIndexView);
    auto indexBuffer = outIndexView.src.buffer;
    const auto indexSize = inIndexView.composed.stride;
    std::byte* outIndices = reinterpret_cast<std::byte*>(outIndexView.getPointer());
    outGeo->setIndexView({});

    const auto inVertexView = inGeo->getPositionView();
    auto outVertexView = createOutView(inVertexView);
    auto vertexBuffer = outVertexView.src.buffer;
    const auto vertexSize = inVertexView.composed.stride;
    const std::byte* inVertices = reinterpret_cast<const std::byte*>(inVertexView.getPointer());
    std::byte* const outVertices = reinterpret_cast<std::byte*>(vertexBuffer->getPointer());
    outGeo->setPositionView(std::move(outVertexView));

    const auto inNormalView = inGeo->getNormalView();
    const std::byte* const inNormals = reinterpret_cast<const std::byte*>(inNormalView.getPointer());
    auto outNormalView = createOutView(inNormalView);
    auto outNormalBuffer = outNormalView.src.buffer;
    outGeo->setNormalView(std::move(outNormalView));

    outGeometry->getJointWeightViews()->resize(inGeo->getJointWeightViews().size());
    for (uint64_t jointView_i = 0u; jointView_i < inGeo->getJointWeightViews().size(); jointView_i++)
    {
        auto& inJointWeightView = inGeo->getJointWeightViews()[jointView_i];
        auto& outJointWeightView = outGeometry->getJointWeightViews()->operator[](jointView_i);
        outJointWeightView.indices = createOutView(inJointWeightView.indices);
        outJointWeightView.weights = createOutView(inJointWeightView.weights);
    }

    outGeometry->getAuxAttributeViews()->resize(inGeo->getAuxAttributeViews().size());
    for (uint64_t auxView_i = 0u; auxView_i < inGeo->getAuxAttributeViews().size(); auxView_i++)
        outGeo->getAuxAttributeViews()->operator[](auxView_i) = createOutView(inGeo->getAuxAttributeViews()[auxView_i]);

    std::array<uint32_t,255> indices;
    for (uint64_t prim_i = 0u; prim_i < primCount; prim_i++)
    {
        IPolygonGeometryBase::IIndexingCallback::SContext<uint32_t> context{
            .indexBuffer = indexView.getPointer(),
            .indexSize = indexView.composed.stride,
            .beginPrimitive = prim_i,
            .endPrimitive = prim_i + 1,
            .out = indices.data()
        };
        indexing->operator()(context);
        for (uint8_t primIndex_i=0; primIndex_i<degree; primIndex_i++)
        {
            const auto outIndex = prim_i * degree + primIndex_i;
            const auto inIndex = indices[primIndex_i];
            // TODO: these memcpys from view to view could really be DRY-ed and lambdified
            memcpy(outIndices + outIndex * indexSize, &outIndex, indexSize);
            memcpy(outVertices + outIndex * vertexSize, inVertices + inIndex * vertexSize, vertexSize);
            if (inNormalView)
            {
                std::byte* const outNormals = reinterpret_cast<std::byte*>(outNormalBuffer->getPointer());
                const auto normalSize = inNormalView.composed.stride;
                memcpy(outNormals + outIndex * normalSize, inNormals + inIndex * normalSize, normalSize);
            }

            for (uint64_t jointView_i = 0u; jointView_i < inGeo->getJointWeightViews().size(); jointView_i++)
            {
                auto& inView = inGeo->getJointWeightViews()[jointView_i];
                auto& outView = outGeometry->getJointWeightViews()->operator[](jointView_i);

                const std::byte* const inJointIndices = reinterpret_cast<const std::byte*>(inView.indices.getPointer());
                const auto jointIndexSize = inView.indices.composed.stride;
                std::byte* const outJointIndices = reinterpret_cast<std::byte*>(outView.indices.getPointer());
                memcpy(outJointIndices + outIndex * jointIndexSize, inJointIndices + inIndex * jointIndexSize, jointIndexSize);

                const std::byte* const inWeights = reinterpret_cast<const std::byte*>(inView.weights.getPointer());
                const auto jointWeightSize = inView.weights.composed.stride;
                std::byte* const outWeights = reinterpret_cast<std::byte*>(outView.weights.getPointer());
                memcpy(outWeights + outIndex * jointWeightSize, outWeights + inIndex * jointWeightSize, jointWeightSize);
            }

            for (uint64_t auxView_i = 0u; auxView_i < inGeo->getAuxAttributeViews().size(); auxView_i++)
            {
                auto& inView = inGeo->getAuxAttributeViews()[auxView_i];
                auto& outView = outGeometry->getAuxAttributeViews()->operator[](auxView_i);
                const auto attrSize = inView.composed.stride;
                const std::byte* const inAuxs = reinterpret_cast<const std::byte*>(inView.getPointer());
                std::byte* const outAuxs = reinterpret_cast<std::byte*>(outView.getPointer());
                memcpy(outAuxs + outIndex * attrSize, inAuxs + inIndex * attrSize, attrSize);
            }
        }
    }

    recomputeContentHashes(outGeo);
    return outGeometry;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CPolygonGeometryManipulator::createSmoothVertexNormal(const ICPUPolygonGeometry* inPolygon, bool enableWelding, float epsilon, VxCmpFunction vxcmp)
{
    if (!inPolygon)
    {
        _NBL_DEBUG_BREAK_IF(true);
        return nullptr;
    }

    // Mesh need to be unwelded (TODO: why? the output only need to be unwelded, really should be checking `inPolygon->getIndexingCallback()->count()!=3`)
    if (inPolygon->getIndexView() && inPolygon->getIndexingCallback()!=IPolygonGeometryBase::TriangleList())
    {
      _NBL_DEBUG_BREAK_IF(true);
      return nullptr;
    }

    auto result = CSmoothNormalGenerator::calculateNormals(inPolygon, epsilon, vxcmp);
    if (enableWelding)
    {
      return CVertexWelder::weldVertices(result.geom.get(), result.vertexHashGrid, CVertexWelder::DefaultWeldPredicate(epsilon));
    }
    return result.geom;
}

#if 0
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

		auto newVertBuffer = ICPUBuffer::create({ vertexCount*vertexSize });
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
#endif
} // end namespace nbl::asset


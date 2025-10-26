// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CSmoothNormalGenerator.h"

#include "nbl/core/declarations.h"
#include "nbl/builtin/hlsl/shapes/triangle.hlsl"

#include <algorithm>

namespace nbl
{
namespace asset
{
static bool operator<(uint32_t lhs, const CPolygonGeometryManipulator::SSNGVertexData& rhs)
{
	return lhs < rhs.hash;
}

static bool operator<(const CPolygonGeometryManipulator::SSNGVertexData& lhs, uint32_t rhs)
{
	return lhs.hash < rhs;
}

static bool compareVertexPosition(const hlsl::float32_t3& a, const hlsl::float32_t3& b, float epsilon)
{
	const hlsl::float32_t3 difference = abs(b - a);
	return (difference.x <= epsilon && difference.y <= epsilon && difference.z <= epsilon);
}

CSmoothNormalGenerator::Result CSmoothNormalGenerator::calculateNormals(const asset::ICPUPolygonGeometry* polygon, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp)
{
	VertexHashMap vertexHashMap = setupData(polygon, epsilon);
	const auto smoothPolygon = processConnectedVertices(polygon, vertexHashMap, epsilon,vxcmp);
	return { vertexHashMap, smoothPolygon };
}


CSmoothNormalGenerator::VertexHashMap CSmoothNormalGenerator::setupData(const asset::ICPUPolygonGeometry* polygon, float epsilon)
{
	const size_t idxCount = polygon->getPrimitiveCount() * 3;

	const auto cellCount = std::max<uint32_t>(core::roundUpToPoT<uint32_t>((idxCount + 31) >> 5), 4);
	VertexHashMap vertices(idxCount, std::min(16u * 1024u, cellCount), epsilon == 0.0f ? 0.00001f : epsilon * 2.f);

	for (uint64_t i = 0; i < idxCount; i += 3)
	{
		//calculate face normal of parent triangle
		hlsl::float32_t3 v1, v2, v3;
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i, v1);
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i + 1, v2);
		polygon->getPositionView().decodeElement<hlsl::float32_t3>(i + 2, v3);

		const auto faceNormal = normalize(cross(v2 - v1, v3 - v1));

		//set data for m_vertices
		const auto angleWages = hlsl::shapes::util::compInternalAngle(v2 - v3, v1 - v3, v1 - v2);

		vertices.add({ i,	0,	faceNormal * angleWages.x, v1});
		vertices.add({ i + 1,	0,	faceNormal * angleWages.y,v2});
		vertices.add({ i + 2,	0,	faceNormal * angleWages.z, v3});
	}

	vertices.validate();

	return vertices;
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CSmoothNormalGenerator::processConnectedVertices(const asset::ICPUPolygonGeometry* polygon, VertexHashMap& vertexHashMap, float epsilon, CPolygonGeometryManipulator::VxCmpFunction vxcmp)
{
	auto outPolygon = core::move_and_static_cast<ICPUPolygonGeometry>(polygon->clone(0u));
	static constexpr auto NormalFormat = EF_R32G32B32_SFLOAT;
	const auto normalFormatBytesize = asset::getTexelOrBlockBytesize(NormalFormat);
	auto normalBuf = ICPUBuffer::create({ normalFormatBytesize * outPolygon->getPositionView().getElementCount()});
	auto normalView = polygon->getNormalView();

	hlsl::shapes::AABB<4,hlsl::float32_t> aabb;
	aabb.maxVx = hlsl::float32_t4(1, 1, 1, 0.f);
	aabb.minVx = -aabb.maxVx;
	outPolygon->setNormalView({
		.composed = {
			.encodedDataRange = {.f32 = aabb},
			.stride = sizeof(hlsl::float32_t3),
			.format = NormalFormat,
			.rangeFormat = IGeometryBase::EAABBFormat::F32
		},
		.src = { .offset = 0, .size = normalBuf->getSize(), .buffer = std::move(normalBuf) }
	});

	auto* normalPtr = reinterpret_cast<std::byte*>(outPolygon->getNormalAccessor().getPointer());
	constexpr auto normalStride = sizeof(hlsl::float32_t3);
	assert(outPolygon->getNormalView().composed.stride==normalStride);

	for (auto& processedVertex : vertexHashMap.vertices())
	{
		auto normal = processedVertex.weightedNormal;

		vertexHashMap.iterateBroadphaseCandidates(processedVertex, [&](const VertexHashMap::vertex_data_t& candidate)
			{
				if (compareVertexPosition(processedVertex.position, candidate.position, epsilon) &&
					vxcmp(processedVertex, candidate, polygon))
				{
					//TODO: better mean calculation algorithm
					normal += candidate.weightedNormal;
				}
				return true;
			});

		normal = normalize(normal);
		memcpy(normalPtr + (normalStride * processedVertex.index), &normal, sizeof(normal));
	}

	CPolygonGeometryManipulator::recomputeContentHashes(outPolygon.get());

	return outPolygon;
}

}
}
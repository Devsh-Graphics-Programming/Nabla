#include "CSmoothNormalGenerator.h"

#include <iostream>

namespace irr
{
namespace asset
{

static inline core::vector3df_SIMD getAngleWeight(const core::vector3df_SIMD & v1,
	const core::vector3df_SIMD & v2,
	const core::vector3df_SIMD & v3)
{
		// Calculate this triangle's weight for each of its three vertices
		// start by calculating the lengths of its sides
	const float a = v2.getDistanceFromSQAsFloat(v3);
	const float asqrt = sqrtf(a);
	const float b = v1.getDistanceFromSQAsFloat(v3);
	const float bsqrt = sqrtf(b);
	const float c = v1.getDistanceFromSQAsFloat(v2);
	const float csqrt = sqrtf(c);

		// use them to find the angle at each vertex
	return core::vector3df_SIMD(
		acosf((b + c - a) / (2.f * bsqrt * csqrt)),
		acosf((-b + c + a) / (2.f * asqrt * csqrt)),
		acosf((b - c + a) / (2.f * bsqrt * asqrt)));
}


asset::ICPUMeshBuffer* irr::asset::CSmoothNormalGenerator::calculateNormals(asset::ICPUMeshBuffer* buffer, float creaseAngle)
{
	core::vector<Vertex> vertexArray = setupData(buffer, creaseAngle);
	processConnectedVertices(buffer, vertexArray, creaseAngle);

	return buffer;
}

core::vector<CSmoothNormalGenerator::Vertex> CSmoothNormalGenerator::setupData(asset::ICPUMeshBuffer* buffer, float creaseAngle)
{
	core::vector<Vertex> vertices;

	size_t idxCount = buffer->getIndexCount();
	size_t vxCount = buffer->calcVertexCount();
	_IRR_DEBUG_BREAK_IF((idxCount % 3));

	vertices.reserve(idxCount);

	core::vector3df_SIMD faceNormal;

	for (uint32_t i = 0; i < idxCount; i += 3)
	{
			//calculate face normal of parent triangle
		core::vectorSIMDf v1 = buffer->getPosition(i);
		core::vectorSIMDf v2 = buffer->getPosition(i + 1);
		core::vectorSIMDf v3 = buffer->getPosition(i + 2);

		faceNormal = core::cross(v2 - v1, v3 - v1);
		faceNormal = core::normalize(faceNormal);
		
			//set data for vertices
		core::vector3df_SIMD angleWages = getAngleWeight(buffer->getPosition(i), buffer->getPosition(i + 1), buffer->getPosition(i + 2));
		vertices.push_back({ i,		faceNormal * angleWages.x,	faceNormal, angleWages.x });
		vertices.push_back({ i + 1,	faceNormal * angleWages.y,	faceNormal, angleWages.y });
		vertices.push_back({ i + 2,	faceNormal * angleWages.z,	faceNormal, angleWages.z });
	}

	return vertices;
}
void CSmoothNormalGenerator::processConnectedVertices(asset::ICPUMeshBuffer* buffer, core::vector<Vertex>& vertices, float creaseAngle)
{
	for (size_t i = 0; i < vertices.size(); i++)
	{
		float cosOfCreaseAngle = std::cos(creaseAngle);
		Vertex processedVertex = vertices[i];

		for (size_t j = i+1; j < vertices.size(); j++)
		{
			if (buffer->getPosition(processedVertex.indexOffset).x == buffer->getPosition(vertices[j].indexOffset).x &&
				buffer->getPosition(processedVertex.indexOffset).y == buffer->getPosition(vertices[j].indexOffset).y &&
				buffer->getPosition(processedVertex.indexOffset).z == buffer->getPosition(vertices[j].indexOffset).z)
			if (processedVertex.parentTriangleFaceNormal.dotProductAsFloat(vertices[j].parentTriangleFaceNormal) > cosOfCreaseAngle)
			{
				processedVertex.normal += vertices[j].parentTriangleFaceNormal * vertices[j].wage;
				vertices[j].normal += processedVertex.parentTriangleFaceNormal * processedVertex.wage;
			}
		}
		
		processedVertex.normal = core::normalize(processedVertex.normal);
		if (!buffer->setAttribute(processedVertex.normal, asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR3, processedVertex.indexOffset))
			_IRR_DEBUG_BREAK_IF(true);
	}
}

}
}


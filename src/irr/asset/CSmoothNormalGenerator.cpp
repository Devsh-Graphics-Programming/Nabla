#include "CSmoothNormalGenerator.h"

#include <iostream>

namespace irr
{
namespace asset
{

static inline bool compareVertexPosition(const core::vectorSIMDf& a, const core::vectorSIMDf& b, float epsilon)
{
	const core::vectorSIMDf difference = core::abs(b - a);
	return (difference.x <= epsilon && difference.y <= epsilon && difference.z <= epsilon);
}

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

#pragma region linearSearch

asset::ICPUMeshBuffer* irr::asset::CSmoothNormalGenerator::calculateNormals(asset::ICPUMeshBuffer* buffer, float creaseAngle, float epsilon)
{
	core::vector<Vertex> vertexArray = setupData(buffer, creaseAngle);
	processConnectedVertices(buffer, vertexArray, creaseAngle, epsilon);

	return buffer;
}

core::vector<CSmoothNormalGenerator::Vertex> CSmoothNormalGenerator::setupData(asset::ICPUMeshBuffer* buffer, float creaseAngle)
{
	core::vector<Vertex> vertices;

	const size_t idxCount = buffer->getIndexCount();
	const size_t vxCount = buffer->calcVertexCount();
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
		core::vector3df_SIMD angleWages = getAngleWeight(v1, v2, v3);
		vertices.push_back({ i,		angleWages.x,  v1,  faceNormal * angleWages.x,	faceNormal });
		vertices.push_back({ i + 1,	angleWages.y,  v2,	faceNormal * angleWages.y,	faceNormal });
		vertices.push_back({ i + 2,	angleWages.z,  v3,  faceNormal * angleWages.z,	faceNormal });
	}

	return vertices;
}

void CSmoothNormalGenerator::processConnectedVertices(asset::ICPUMeshBuffer* buffer, core::vector<Vertex>& vertices, float creaseAngle, float epsilon)
{
	for (size_t i = 0; i < vertices.size(); i++)
	{
		float cosOfCreaseAngle = std::cos(creaseAngle);
		Vertex processedVertex = vertices[i];

		for (size_t j = i+1; j < vertices.size(); j++)
		{
			if (compareVertexPosition(processedVertex.position, vertices[j].position, epsilon))
			if (processedVertex.parentTriangleFaceNormal.dotProductAsFloat(vertices[j].parentTriangleFaceNormal) > cosOfCreaseAngle)
			{
				processedVertex.normal += vertices[j].parentTriangleFaceNormal * vertices[j].wage;
				vertices[j].normal += processedVertex.parentTriangleFaceNormal * processedVertex.wage;
			}
			
		}
		
		processedVertex.normal = core::normalize(processedVertex.normal);
		buffer->setAttribute(processedVertex.normal, asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR3, processedVertex.indexOffset);
	}

}

#pragma endregion


#pragma region hashing

CSmoothNormalGenerator::VertexHashMap::VertexHashMap(size_t _hashTableSize, float _cellSize)
	:hashTableSize(_hashTableSize), cellSize(_cellSize)
{
	hashTable.reserve(hashTableSize);

	for (int i = 0; i < hashTableSize; i++)
		hashTable.emplace_back();

}

uint32_t CSmoothNormalGenerator::VertexHashMap::hash(const CSmoothNormalGenerator::Vertex& vertex) const
{
	static constexpr uint32_t primeNumber1 = 73856093;
	static constexpr uint32_t primeNumber2 = 19349663;
	static constexpr uint32_t primeNumber3 = 83492791;

	const core::vector3df_SIMD position = vertex.position / cellSize;

	return	(((uint32_t)position.x * primeNumber1) ^
			((uint32_t)position.y * primeNumber2) ^
			((uint32_t)position.z * primeNumber3)) % hashTableSize;
}

void CSmoothNormalGenerator::VertexHashMap::add(const Vertex& vertex)
{
	hashTable[hash(vertex)].push_back(vertex);
}

asset::ICPUMeshBuffer * irr::asset::CSmoothNormalGenerator::calculateNormals_hash(asset::ICPUMeshBuffer * buffer, float creaseAngle, float epsilon)
{
	VertexHashMap vertexArray = setupData_hash(buffer, creaseAngle);
	processConnectedVertices_hash(buffer, vertexArray, creaseAngle, epsilon);

	return buffer;
}

CSmoothNormalGenerator::VertexHashMap CSmoothNormalGenerator::setupData_hash(asset::ICPUMeshBuffer * buffer, float creaseAngle)
{
	//hash map and cell size is constant (that will be changed ofc) 
	VertexHashMap vertices(200, 0.001f);

	const size_t idxCount = buffer->getIndexCount();
	const size_t vxCount = buffer->calcVertexCount();
	_IRR_DEBUG_BREAK_IF((idxCount % 3));

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
		core::vector3df_SIMD angleWages = getAngleWeight(v1, v2, v3);
		
		vertices.add({ i,		angleWages.x,  v1,  faceNormal * angleWages.x,	faceNormal });
		vertices.add({ i + 1,	angleWages.y,  v2,	faceNormal * angleWages.y,	faceNormal });
		vertices.add({ i + 2,	angleWages.z,  v3,  faceNormal * angleWages.z,	faceNormal });
	}

	return vertices;
}

void CSmoothNormalGenerator::processConnectedVertices_hash(asset::ICPUMeshBuffer * buffer, CSmoothNormalGenerator::VertexHashMap& vertexHashMap, float creaseAngle, float epsilon)
{

	for (uint32_t cell = 0; cell < vertexHashMap.getTableSize(); cell++)
	{
		core::vector<Vertex> vertices = vertexHashMap.getBucket(cell);

		for (size_t i = 0; i < vertices.size(); i++)
		{
			float cosOfCreaseAngle = std::cos(creaseAngle);
			Vertex processedVertex = vertices[i];

			for (size_t j = i + 1; j < vertices.size(); j++)
			{
				if (compareVertexPosition(processedVertex.position, vertices[j].position, epsilon))
				{
					if (processedVertex.parentTriangleFaceNormal.dotProductAsFloat(vertices[j].parentTriangleFaceNormal) > cosOfCreaseAngle)
					{
						processedVertex.normal += vertices[j].parentTriangleFaceNormal * vertices[j].wage;
						vertices[j].normal += processedVertex.parentTriangleFaceNormal * processedVertex.wage;
					}
				}
			}

			processedVertex.normal = core::normalize(processedVertex.normal);
			buffer->setAttribute(processedVertex.normal, asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR3, processedVertex.indexOffset);
		}
	}
	

}

#pragma endregion

}
}
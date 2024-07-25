// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <iostream>
#include <iomanip>
#include <cmath>

#include "nbl/asset/utils/CGeometryCreator.h"
#include "nbl/asset/utils/CQuantNormalCache.h"

namespace nbl::asset
{

CGeometryCreator::CGeometryCreator(IMeshManipulator* const _defaultMeshManipulator) 
	: defaultMeshManipulator(_defaultMeshManipulator)
{
	if (defaultMeshManipulator == nullptr)
	{
		_NBL_DEBUG_BREAK_IF(true);
		assert(false);
	}
}

CGeometryCreator::return_type CGeometryCreator::createCubeMesh(const core::vector3df& size) const
{
	return_type retval;

	constexpr size_t vertexSize = sizeof(CGeometryCreator::CubeVertex);
	retval.inputParams = {0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(CubeVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(CubeVertex,color)},
											{0u,EF_R8G8_USCALED,offsetof(CubeVertex,uv)},
											{0u,EF_R8G8B8_SSCALED,offsetof(CubeVertex,normal)}
										},{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX}};

	// Create indices
	{
		retval.indexCount = 36u;
		auto indices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint16_t)*retval.indexCount);
		indices->addUsageFlags(asset::IBuffer::EUF_INDEX_BUFFER_BIT);
		auto u = reinterpret_cast<uint16_t*>(indices->getPointer());
		for (uint32_t i=0u; i<6u; ++i)
		{
			u[i*6+0] = 4*i+0;
			u[i*6+1] = 4*i+1;
			u[i*6+2] = 4*i+3;
			u[i*6+3] = 4*i+1;
			u[i*6+4] = 4*i+2;
			u[i*6+5] = 4*i+3;
		}
		retval.indexBuffer = {0ull,std::move(indices)};
	}

	// Create vertices
	auto vertices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(24u*vertexSize);
	vertices->addUsageFlags(IBuffer::EUF_VERTEX_BUFFER_BIT);
	CubeVertex* ptr = (CubeVertex*)vertices->getPointer();

	const core::vector3d<int8_t> normals[6] =
	{
		core::vector3d<int8_t>(0, 0, 1),
		core::vector3d<int8_t>(1, 0, 0),
		core::vector3d<int8_t>(0, 0, -1),
		core::vector3d<int8_t>(-1, 0, 0),
		core::vector3d<int8_t>(0, 1, 0),
		core::vector3d<int8_t>(0, -1, 0)
	};
	const core::vector3df pos[8] =
	{
		core::vector3df(-0.5f,-0.5f, 0.5f)*size,
		core::vector3df( 0.5f,-0.5f, 0.5f)*size,
		core::vector3df( 0.5f, 0.5f, 0.5f)*size,
		core::vector3df(-0.5f, 0.5f, 0.5f)*size,
		core::vector3df( 0.5f,-0.5f,-0.5f)*size,
		core::vector3df(-0.5f, 0.5f,-0.5f)*size,
		core::vector3df(-0.5f,-0.5f,-0.5f)*size,
		core::vector3df( 0.5f, 0.5f,-0.5f)*size
	};
	const core::vector2d<uint8_t> uvs[4] =
	{
		core::vector2d<uint8_t>(0, 1),
		core::vector2d<uint8_t>(1, 1),
		core::vector2d<uint8_t>(1, 0),
		core::vector2d<uint8_t>(0, 0)
	};

	for (size_t f=0ull; f<6ull; ++f)
	{
		const size_t v = f*4ull;

		for (size_t i=0ull; i<4ull; ++i)
		{
			const core::vector3d<int8_t>& n = normals[f];
			const core::vector2d<uint8_t>& uv = uvs[i];
			ptr[v+i].setColor(255, 255, 255, 255);
			ptr[v+i].setNormal(n.X, n.Y, n.Z);
			ptr[v+i].setUv(uv.X, uv.Y);
		}

		switch (f)
		{
			case 0:
				ptr[v+0].setPos(pos[0].X, pos[0].Y, pos[0].Z);
				ptr[v+1].setPos(pos[1].X, pos[1].Y, pos[1].Z);
				ptr[v+2].setPos(pos[2].X, pos[2].Y, pos[2].Z);
				ptr[v+3].setPos(pos[3].X, pos[3].Y, pos[3].Z);
				break;
			case 1:
				ptr[v+0].setPos(pos[1].X, pos[1].Y, pos[1].Z);
				ptr[v+1].setPos(pos[4].X, pos[4].Y, pos[4].Z);
				ptr[v+2].setPos(pos[7].X, pos[7].Y, pos[7].Z);
				ptr[v+3].setPos(pos[2].X, pos[2].Y, pos[2].Z);
				break;
			case 2:
				ptr[v+0].setPos(pos[4].X, pos[4].Y, pos[4].Z);
				ptr[v+1].setPos(pos[6].X, pos[6].Y, pos[6].Z);
				ptr[v+2].setPos(pos[5].X, pos[5].Y, pos[5].Z);
				ptr[v+3].setPos(pos[7].X, pos[7].Y, pos[7].Z);
				break;
			case 3:
				ptr[v+0].setPos(pos[6].X, pos[6].Y, pos[6].Z);
				ptr[v+2].setPos(pos[3].X, pos[3].Y, pos[3].Z);
				ptr[v+1].setPos(pos[0].X, pos[0].Y, pos[0].Z);
				ptr[v+3].setPos(pos[5].X, pos[5].Y, pos[5].Z);
				break;
			case 4:
				ptr[v+0].setPos(pos[3].X, pos[3].Y, pos[3].Z);
				ptr[v+1].setPos(pos[2].X, pos[2].Y, pos[2].Z);
				ptr[v+2].setPos(pos[7].X, pos[7].Y, pos[7].Z);
				ptr[v+3].setPos(pos[5].X, pos[5].Y, pos[5].Z);
				break;
			case 5:
				ptr[v+0].setPos(pos[0].X, pos[0].Y, pos[0].Z);
				ptr[v+1].setPos(pos[6].X, pos[6].Y, pos[6].Z);
				ptr[v+2].setPos(pos[4].X, pos[4].Y, pos[4].Z);
				ptr[v+3].setPos(pos[1].X, pos[1].Y, pos[1].Z);
				break;
		}
	}
	retval.bindings[0] = {0ull,std::move(vertices)};

	// Recalculate bounding box
	retval.indexType = asset::EIT_16BIT;
	retval.bbox = core::aabbox3df(-size*0.5f,size*0.5f);

	return retval;
}


/*
	a cylinder, a cone and a cross
	point up on (0,1.f, 0.f )
*/
CGeometryCreator::return_type CGeometryCreator::createArrowMesh(const uint32_t tesselationCylinder,
																const uint32_t tesselationCone,
																const float height,
																const float cylinderHeight,
																const float width0,
																const float width1,
																const video::SColor vtxColor0,
																const video::SColor vtxColor1,
																IMeshManipulator* const meshManipulatorOverride) const
{
    assert(height > cylinderHeight);

    auto cylinder = createCylinderMesh(width0, cylinderHeight, tesselationCylinder, vtxColor0);
    auto cone = createConeMesh(width1, height-cylinderHeight, tesselationCone, vtxColor1, vtxColor1);

	auto cylinderVertices = reinterpret_cast<CylinderVertex*>(cylinder.bindings[0].buffer->getPointer());
	auto coneVertices = reinterpret_cast<ConeVertex*>(cone.bindings[0].buffer->getPointer());

	auto cylinderIndecies = reinterpret_cast<uint16_t*>(cylinder.indexBuffer.buffer->getPointer());
	auto coneIndecies = reinterpret_cast<uint16_t*>(cone.indexBuffer.buffer->getPointer());

	const auto cylinderVertexCount = cylinder.bindings[0].buffer->getSize() / sizeof(CylinderVertex);
	const auto coneVertexCount = cone.bindings[0].buffer->getSize() / sizeof(ConeVertex);
	const auto newArrowVertexCount = cylinderVertexCount + coneVertexCount;

	const auto cylinderIndexCount = cylinder.indexBuffer.buffer->getSize() / sizeof(uint16_t);
	const auto coneIndexCount = cone.indexBuffer.buffer->getSize() / sizeof(uint16_t);
	const auto newArrowIndexCount = cylinderIndexCount + coneIndexCount;

	for (auto i = 0ull; i < coneVertexCount; ++i)
	{
		core::vector3df_SIMD newPos = coneVertices[i].pos;
		newPos.rotateYZByRAD(-1.5707963268);

		for (auto c = 0; c < 3; ++c)
			coneVertices[i].pos[c] = newPos[c];
	}

	auto newArrowVertexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(newArrowVertexCount * sizeof(ArrowVertex));
	newArrowVertexBuffer->setUsageFlags(newArrowVertexBuffer->getUsageFlags() | asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	auto newArrowIndexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(newArrowIndexCount * sizeof(uint16_t));
	newArrowIndexBuffer->setUsageFlags(newArrowIndexBuffer->getUsageFlags() | asset::IBuffer::EUF_INDEX_BUFFER_BIT);

	for (auto z = 0ull; z < newArrowVertexCount; ++z)
	{
		auto arrowVertex = reinterpret_cast<ArrowVertex*>(newArrowVertexBuffer->getPointer()) + z;

		if (z < cylinderVertexCount)
		{
			auto cylinderVertex = (cylinderVertices + z);
			memcpy(arrowVertex, cylinderVertex, sizeof(ArrowVertex));
		}
		else
		{
			auto coneVertex = (coneVertices + z - cylinderVertexCount);
			memcpy(arrowVertex, coneVertex, offsetof(ConeVertex, normal)); // copy position and color
			arrowVertex->uv[0] = 0;
			arrowVertex->uv[1] = 0;
			arrowVertex->normal = coneVertex->normal;
		}
	}

	{
		auto ArrowIndices = reinterpret_cast<uint16_t*>(newArrowIndexBuffer->getPointer());
		auto newConeIndices = (ArrowIndices + cylinderIndexCount);

		memcpy(ArrowIndices, cylinderIndecies, sizeof(uint16_t) * cylinderIndexCount);
		memcpy(newConeIndices, coneIndecies, sizeof(uint16_t) * coneIndexCount);

		for (auto i = 0ull; i < coneIndexCount; ++i)
			*(newConeIndices + i) += cylinderVertexCount;
	}

	return_type arrow;

	constexpr size_t vertexSize = sizeof(ArrowVertex);
	arrow.inputParams = 
	{ 0b1111u,0b1u,
		{
			{0u,EF_R32G32B32_SFLOAT,offsetof(ArrowVertex,pos)},
			{0u,EF_R8G8B8A8_UNORM,offsetof(ArrowVertex,color)},
			{0u,EF_R32G32_SFLOAT,offsetof(ArrowVertex,uv)},
			{0u,EF_A2B10G10R10_SNORM_PACK32,offsetof(ArrowVertex,normal)}
		},
		{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX} 
	};

	arrow.bindings[0] = { 0, std::move(newArrowVertexBuffer) }; 
	arrow.indexBuffer = { 0, std::move(newArrowIndexBuffer) };
	arrow.indexCount = newArrowIndexCount;
	arrow.indexType = EIT_16BIT;

    return arrow;
}

/* A sphere with proper normals and texture coords */
CGeometryCreator::return_type CGeometryCreator::createSphereMesh(float radius, uint32_t polyCountX, uint32_t polyCountY, IMeshManipulator* const meshManipulatorOverride) const
{
	// we are creating the sphere mesh here.
	return_type retval;
	constexpr size_t vertexSize = sizeof(CGeometryCreator::SphereVertex);
	CQuantNormalCache* const quantNormalCache = (meshManipulatorOverride == nullptr) ? defaultMeshManipulator->getQuantNormalCache() : meshManipulatorOverride->getQuantNormalCache();
	retval.inputParams = { 0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(SphereVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(SphereVertex,color)},
											{0u,EF_R32G32_SFLOAT,offsetof(SphereVertex,uv)},
											{0u,EF_A2B10G10R10_SNORM_PACK32,offsetof(SphereVertex,normal)}
										},{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX} };

	if (polyCountX < 2)
		polyCountX = 2;
	if (polyCountY < 2)
		polyCountY = 2;

	const uint32_t polyCountXPitch = polyCountX + 1; // get to same vertex on next level

	retval.indexCount = (polyCountX * polyCountY) * 6;
	auto indices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t) * retval.indexCount);

	// Create indices
	{
		uint32_t level = 0;
		size_t indexAddIx = 0;
		uint32_t* indexPtr = (uint32_t*)indices->getPointer();
		for (uint32_t p1 = 0; p1 < polyCountY - 1; ++p1)
		{
			//main quads, top to bottom
			for (uint32_t p2 = 0; p2 < polyCountX - 1; ++p2)
			{
				const uint32_t curr = level + p2;
				indexPtr[indexAddIx++] = curr + polyCountXPitch;
				indexPtr[indexAddIx++] = curr;
				indexPtr[indexAddIx++] = curr + 1;
				indexPtr[indexAddIx++] = curr + polyCountXPitch;
				indexPtr[indexAddIx++] = curr + 1;
				indexPtr[indexAddIx++] = curr + 1 + polyCountXPitch;
			}

			// the connectors from front to end
			indexPtr[indexAddIx++] = level + polyCountX - 1 + polyCountXPitch;
			indexPtr[indexAddIx++] = level + polyCountX - 1;
			indexPtr[indexAddIx++] = level + polyCountX;

			indexPtr[indexAddIx++] = level + polyCountX - 1 + polyCountXPitch;
			indexPtr[indexAddIx++] = level + polyCountX;
			indexPtr[indexAddIx++] = level + polyCountX + polyCountXPitch;
			level += polyCountXPitch;
		}

		const uint32_t polyCountSq = polyCountXPitch * polyCountY; // top point
		const uint32_t polyCountSq1 = polyCountSq + 1; // bottom point
		const uint32_t polyCountSqM1 = (polyCountY - 1) * polyCountXPitch; // last row's first vertex

		for (uint32_t p2 = 0; p2 < polyCountX - 1; ++p2)
		{
			// create triangles which are at the top of the sphere

			indexPtr[indexAddIx++] = polyCountSq;
			indexPtr[indexAddIx++] = p2 + 1;
			indexPtr[indexAddIx++] = p2;

			// create triangles which are at the bottom of the sphere

			indexPtr[indexAddIx++] = polyCountSqM1 + p2;
			indexPtr[indexAddIx++] = polyCountSqM1 + p2 + 1;
			indexPtr[indexAddIx++] = polyCountSq1;
		}

		// create final triangle which is at the top of the sphere

		indexPtr[indexAddIx++] = polyCountSq;
		indexPtr[indexAddIx++] = polyCountX;
		indexPtr[indexAddIx++] = polyCountX - 1;

		// create final triangle which is at the bottom of the sphere

		indexPtr[indexAddIx++] = polyCountSqM1 + polyCountX - 1;
		indexPtr[indexAddIx++] = polyCountSqM1;
		indexPtr[indexAddIx++] = polyCountSq1;
	}
	indices->setUsageFlags(indices->getUsageFlags() | asset::IBuffer::EUF_INDEX_BUFFER_BIT);
	retval.indexBuffer = {0ull, std::move(indices)};

	// handle vertices
	{
		size_t vertexSize = 3 * 4 + 4 + 2 * 4 + 4;
		size_t vertexCount = (polyCountXPitch * polyCountY) + 2;
		auto vtxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vertexCount * vertexSize);
		auto* tmpMem = reinterpret_cast<uint8_t*>(vtxBuf->getPointer());
		for (size_t i = 0; i < vertexCount; i++)
		{
			tmpMem[i * vertexSize + 3 * 4 + 0] = 255;
			tmpMem[i * vertexSize + 3 * 4 + 1] = 255;
			tmpMem[i * vertexSize + 3 * 4 + 2] = 255;
			tmpMem[i * vertexSize + 3 * 4 + 3] = 255;
		}
		// calculate the angle which separates all points in a circle
		const float AngleX = 2 * core::PI<float>() / polyCountX;
		const float AngleY = core::PI<float>() / polyCountY;

		double axz;

		// we don't start at 0.

		double ay = 0;//AngleY / 2;

		using quant_normal_t = CQuantNormalCache::value_type_t<EF_A2B10G10R10_SNORM_PACK32>;
		uint8_t* tmpMemPtr = tmpMem;
		for (uint32_t y = 0; y < polyCountY; ++y)
		{
			ay += AngleY;
			const double sinay = sin(ay);
			axz = 0;

			// calculate the necessary vertices without the doubled one
			uint8_t* oldTmpMemPtr = tmpMemPtr;
			for (uint32_t xz = 0; xz < polyCountX; ++xz)
			{
				// calculate points position

				core::vector3df pos(static_cast<float>(cos(axz) * sinay),
					static_cast<float>(cos(ay)),
					static_cast<float>(sin(axz) * sinay));
				// for spheres the normal is the position
				core::vectorSIMDf normal(&pos.X);
				normal.makeSafe3D();
				quant_normal_t quantizedNormal = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(normal);
				pos *= radius;

				// calculate texture coordinates via sphere mapping
				// tu is the same on each level, so only calculate once
				float tu = 0.5f;
				//if (y==0)
				//{
				if (normal.Y != -1.0f && normal.Y != 1.0f)
					tu = static_cast<float>(acos(core::clamp(normal.X / sinay, -1.0, 1.0)) * 0.5 * core::RECIPROCAL_PI<double>());
				if (normal.Z < 0.0f)
					tu = 1 - tu;
				//}
				//else
					//tu = ((float*)(tmpMem+(i-polyCountXPitch)*vertexSize))[4];

				((float*)tmpMemPtr)[0] = pos.X;
				((float*)tmpMemPtr)[1] = pos.Y;
				((float*)tmpMemPtr)[2] = pos.Z;
				((float*)tmpMemPtr)[4] = tu;
				((float*)tmpMemPtr)[5] = static_cast<float>(ay * core::RECIPROCAL_PI<double>());
				((quant_normal_t*)tmpMemPtr)[6] = quantizedNormal;
				static_assert(sizeof(quant_normal_t)==4u);

				tmpMemPtr += vertexSize;
				axz += AngleX;
			}
			// This is the doubled vertex on the initial position

			((float*)tmpMemPtr)[0] = ((float*)oldTmpMemPtr)[0];
			((float*)tmpMemPtr)[1] = ((float*)oldTmpMemPtr)[1];
			((float*)tmpMemPtr)[2] = ((float*)oldTmpMemPtr)[2];
			((float*)tmpMemPtr)[4] = 1.f;
			((float*)tmpMemPtr)[5] = ((float*)oldTmpMemPtr)[5];
			((uint32_t*)tmpMemPtr)[6] = ((uint32_t*)oldTmpMemPtr)[6];
			tmpMemPtr += vertexSize;
		}

		// the vertex at the top of the sphere
		((float*)tmpMemPtr)[0] = 0.f;
		((float*)tmpMemPtr)[1] = radius;
		((float*)tmpMemPtr)[2] = 0.f;
		((float*)tmpMemPtr)[4] = 0.5f;
		((float*)tmpMemPtr)[5] = 0.f;
		((quant_normal_t*)tmpMemPtr)[6] = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(core::vectorSIMDf(0.f, 1.f, 0.f));

		// the vertex at the bottom of the sphere
		tmpMemPtr += vertexSize;
		((float*)tmpMemPtr)[0] = 0.f;
		((float*)tmpMemPtr)[1] = -radius;
		((float*)tmpMemPtr)[2] = 0.f;
		((float*)tmpMemPtr)[4] = 0.5f;
		((float*)tmpMemPtr)[5] = 1.f;
		((quant_normal_t*)tmpMemPtr)[6] = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(core::vectorSIMDf(0.f, -1.f, 0.f));

		// recalculate bounding box
		core::aabbox3df BoundingBox;
		BoundingBox.reset(core::vector3df(radius));
		BoundingBox.addInternalPoint(-radius, -radius, -radius);

		// set vertex buffer
		vtxBuf->setUsageFlags(vtxBuf->getUsageFlags() | asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
		retval.bindings[0] = { 0ull,std::move(vtxBuf) };
		retval.indexType = asset::EIT_32BIT;
		retval.bbox = BoundingBox;
	}

	return retval;
}

/* A cylinder with proper normals and texture coords */
CGeometryCreator::return_type CGeometryCreator::createCylinderMesh(float radius, float length,
			uint32_t tesselation, const video::SColor& color, IMeshManipulator* const meshManipulatorOverride) const
{
	return_type retval;
	constexpr size_t vertexSize = sizeof(CGeometryCreator::CylinderVertex);
	CQuantNormalCache* const quantNormalCache = (meshManipulatorOverride == nullptr) ? defaultMeshManipulator->getQuantNormalCache() : meshManipulatorOverride->getQuantNormalCache();
	retval.inputParams = { 0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(CylinderVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(CylinderVertex,color)},
											{0u,EF_R32G32_SFLOAT,offsetof(CylinderVertex,uv)},
											{0u,EF_A2B10G10R10_SNORM_PACK32,offsetof(CylinderVertex,normal)}
										},{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX} };

    const size_t vtxCnt = 2u*tesselation;
    auto vtxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vtxCnt*sizeof(CylinderVertex));

    CylinderVertex* vertices = reinterpret_cast<CylinderVertex*>(vtxBuf->getPointer());
	for (auto i=0ull; i<vtxCnt; i++)
		vertices[i] = CylinderVertex();

    const uint32_t halfIx = tesselation;

    uint8_t glcolor[4];
    color.toOpenGLColor(glcolor);

    const float tesselationRec = core::reciprocal_approxim<float>(tesselation);
    const float step = 2.f*core::PI<float>()*tesselationRec;
    for (uint32_t i = 0u; i<tesselation; ++i)
    {
        core::vectorSIMDf p(std::cos(i*step), std::sin(i*step), 0.f);
        p *= radius;
        const auto n = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(core::normalize(p));

        memcpy(vertices[i].pos, p.pointer, 12u);
        vertices[i].normal = n;
        memcpy(vertices[i].color, glcolor, 4u);
        vertices[i].uv[0] = float(i) * tesselationRec;

        vertices[i+halfIx] = vertices[i];
        vertices[i+halfIx].pos[2] = length;
        vertices[i+halfIx].uv[1] = 1.f;
    }

    constexpr uint32_t rows = 2u;
	retval.indexCount = rows * 3u * tesselation;
    auto idxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(retval.indexCount *sizeof(uint16_t));
    uint16_t* indices = (uint16_t*)idxBuf->getPointer();

    for (uint32_t i = 0u, j = 0u; i < halfIx; ++i)
    {
        indices[j++] = i;
        indices[j++] = (i+1u)!=halfIx ? (i+1u):0u;
        indices[j++] = i+halfIx;
        indices[j++] = i+halfIx;
        indices[j++] = (i+1u)!=halfIx ? (i+1u):0u;
        indices[j++] = (i+1u)!=halfIx ? (i+1u+halfIx):halfIx;
    }

	// set vertex buffer
	idxBuf->setUsageFlags(idxBuf->getUsageFlags() | asset::IBuffer::EUF_INDEX_BUFFER_BIT);
	retval.indexBuffer = { 0ull, std::move(idxBuf) };
	vtxBuf->setUsageFlags(vtxBuf->getUsageFlags() | asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	retval.bindings[0] = { 0ull, std::move(vtxBuf) };
	retval.indexType = asset::EIT_16BIT;
	//retval.bbox = ?;

	return retval;
}

/* A cone with proper normals and texture coords */
CGeometryCreator::return_type CGeometryCreator::createConeMesh(	float radius, float length, uint32_t tesselation,
																const video::SColor& colorTop,
																const video::SColor& colorBottom,
																float oblique,
																IMeshManipulator* const meshManipulatorOverride) const
{
    const size_t vtxCnt = tesselation * 2;
    auto vtxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vtxCnt * sizeof(ConeVertex));
    ConeVertex* vertices = reinterpret_cast<ConeVertex*>(vtxBuf->getPointer());

	ConeVertex* baseVertices = vertices;
	ConeVertex* apexVertices = vertices + tesselation;

    std::fill(vertices,vertices+vtxCnt, ConeVertex(core::vectorSIMDf(0.f),{},colorBottom));
	CQuantNormalCache* const quantNormalCache = (meshManipulatorOverride == nullptr) ? defaultMeshManipulator->getQuantNormalCache() : meshManipulatorOverride->getQuantNormalCache();

    const float step = (2.f*core::PI<float>()) / tesselation;

	const core::vectorSIMDf apexVertexCoords(oblique, length, 0.0f);

	//vertex positions
	for (uint32_t i = 0u; i < tesselation; i++)
	{
		core::vectorSIMDf v(std::cos(i * step), 0.0f, std::sin(i * step), 0.0f);
		v *= radius;

		memcpy(baseVertices[i].pos, v.pointer, sizeof(float) * 3);
		memcpy(apexVertices[i].pos, apexVertexCoords.pointer, sizeof(float) * 3);
	}

	//vertex normals
	for (uint32_t i = 0; i < tesselation; i++)
	{
		const core::vectorSIMDf v0ToApex = apexVertexCoords - core::vectorSIMDf(vertices[i].pos[0], vertices[i].pos[1], vertices[i].pos[2]);

		uint32_t nextVertexIndex = i == (tesselation - 1) ? 0 : i + 1;
		core::vectorSIMDf u1 = core::vectorSIMDf(baseVertices[nextVertexIndex].pos[0], baseVertices[nextVertexIndex].pos[1], baseVertices[nextVertexIndex].pos[2]);
		u1 -= core::vectorSIMDf(baseVertices[i].pos[0], baseVertices[i].pos[1], baseVertices[i].pos[2]);
		float angleWeight = std::acos(core::dot(core::normalize(apexVertexCoords), core::normalize(u1)).x);
		u1 = core::normalize(core::cross(v0ToApex, u1)) * angleWeight;

		uint32_t prevVertexIndex = i == 0 ? (tesselation - 1) : i - 1;
		core::vectorSIMDf u2 = core::vectorSIMDf(baseVertices[prevVertexIndex].pos[0], baseVertices[prevVertexIndex].pos[1], baseVertices[prevVertexIndex].pos[2]);
		u2 -= core::vectorSIMDf(baseVertices[i].pos[0], baseVertices[i].pos[1], baseVertices[i].pos[2]);
		angleWeight = std::acos(core::dot(core::normalize(apexVertexCoords), core::normalize(u2)).x);
		u2 = core::normalize(core::cross(u2, v0ToApex)) * angleWeight;

		baseVertices[i].normal = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(core::normalize(u1 + u2));
		apexVertices[i].normal = quantNormalCache->quantize<EF_A2B10G10R10_SNORM_PACK32>(core::normalize(u1));
	}

	auto idxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(3u * tesselation * sizeof(uint16_t));
	uint16_t* indices = (uint16_t*)idxBuf->getPointer();

	const uint32_t firstIndexOfBaseVertices = 0;
	const uint32_t firstIndexOfApexVertices = tesselation;
	for (uint32_t i = 0; i < tesselation; i++)
	{
		indices[i * 3] = firstIndexOfApexVertices + i;
		indices[(i * 3) + 1] = firstIndexOfBaseVertices + i;
		indices[(i * 3) + 2] = i == (tesselation - 1) ? firstIndexOfBaseVertices : firstIndexOfBaseVertices + i + 1;
	}

	return_type cone;

	constexpr size_t vertexSize = sizeof(ConeVertex);
	cone.inputParams =
	{ 0b111u,0b1u,
		{
			{0u,EF_R32G32B32_SFLOAT,offsetof(ConeVertex,pos)},
			{0u,EF_R8G8B8A8_UNORM,offsetof(ConeVertex,color)},
			{0u,EF_A2B10G10R10_SNORM_PACK32,offsetof(ConeVertex,normal)}
		},
		{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX}
	};

	vtxBuf->addUsageFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	cone.bindings[0] = { 0, std::move(vtxBuf) };
	idxBuf->addUsageFlags(asset::IBuffer::EUF_INDEX_BUFFER_BIT);
	cone.indexBuffer = { 0, std::move(idxBuf) };
	cone.indexCount = cone.indexBuffer.buffer->getSize() / sizeof(uint16_t);
	cone.indexType = EIT_16BIT;

    return cone;
}


CGeometryCreator::return_type CGeometryCreator::createRectangleMesh(const core::vector2df_SIMD& _size) const
{
	return_type retval;
	constexpr size_t vertexSize = sizeof(CGeometryCreator::RectangleVertex);
	retval.inputParams = { 0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(RectangleVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(RectangleVertex,color)},
											{0u,EF_R8G8_USCALED,offsetof(RectangleVertex,uv)},
											{0u,EF_R32G32B32_SFLOAT,offsetof(RectangleVertex,normal)}
										},{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX} };
	// Create indices
	retval.indexCount = 6;
	retval.indexType = asset::EIT_16BIT;
	uint16_t u[6];

	/*
	0---1
	| / |
	3---2
	*/
	u[0] = 0;
	u[1] = 3;
	u[2] = 1;
	u[3] = 1;
	u[4] = 3;
	u[5] = 2;

	auto indices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(u));
	memcpy(indices->getPointer(), u, sizeof(u));
	indices->addUsageFlags(asset::IBuffer::EUF_INDEX_BUFFER_BIT);
	retval.indexBuffer = { 0ull, std::move(indices) };

	// Create vertices
	auto vertices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(4 * vertexSize);
	RectangleVertex* ptr = (RectangleVertex*)vertices->getPointer();

	ptr[0] = RectangleVertex(core::vector3df_SIMD(-1.0f,  1.0f, 0.0f) * _size, video::SColor(0xFFFFFFFFu), 
		core::vector2du32_SIMD(0u, 1u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));
	ptr[1] = RectangleVertex(core::vector3df_SIMD( 1.0f,  1.0f, 0.0f) * _size, video::SColor(0xFFFFFFFFu),
		core::vector2du32_SIMD(1u, 1u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));
	ptr[2] = RectangleVertex(core::vector3df_SIMD( 1.0f, -1.0f, 0.0f) * _size, video::SColor(0xFFFFFFFFu),
		core::vector2du32_SIMD(1u, 0u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));
	ptr[3] = RectangleVertex(core::vector3df_SIMD(-1.0f, -1.0f, 0.0f) * _size, video::SColor(0xFFFFFFFFu),
		core::vector2du32_SIMD(0u, 0u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));

	vertices->addUsageFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	retval.bindings[0] = {0ull, std::move(vertices)};

	return retval;
}

CGeometryCreator::return_type CGeometryCreator::createDiskMesh(float radius, uint32_t tesselation) const
{
	return_type retval;
	constexpr size_t vertexSize = sizeof(CGeometryCreator::DiskVertex);

	retval.inputParams = { 0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(DiskVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(DiskVertex,color)},
											{0u,EF_R8G8_USCALED,offsetof(DiskVertex,uv)},
											{0u,EF_R32G32B32_SFLOAT,offsetof(DiskVertex,normal)}
										},{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX} };
	retval.assemblyParams.primitiveType = EPT_TRIANGLE_FAN; // without indices
	retval.indexType = EIT_UNKNOWN;

	const size_t vertexCount = 2u + tesselation;
	retval.indexCount = vertexCount;

	const float angle = 360.0f / static_cast<float>(tesselation);
	
	auto vertices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vertexCount * vertexSize);
	DiskVertex* ptr = (DiskVertex*)vertices->getPointer();

	const core::vectorSIMDf v0(0.0f, radius, 0.0f, 1.0f);
	core::matrix3x4SIMD rotation;

	//center
	ptr[0] = DiskVertex(core::vector3df_SIMD(0.0f), video::SColor(0xFFFFFFFFu),
		core::vector2du32_SIMD(0u, 1u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));

	//v0
	ptr[1] = DiskVertex(v0, video::SColor(0xFFFFFFFFu),
		core::vector2du32_SIMD(0u, 1u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));

	//vn
	ptr[vertexCount - 1] = ptr[1];

	//v1, v2, ..., vn-1
	for (int i = 2; i < vertexCount-1; i++)
	{
		core::vectorSIMDf vn;
		core::matrix3x4SIMD rotMatrix;
		rotMatrix.setRotation(core::quaternion(0.0f, 0.0f, core::radians((i-1)*angle)));
		rotMatrix.transformVect(vn, v0);

		ptr[i] = DiskVertex(vn, video::SColor(0xFFFFFFFFu),
			core::vector2du32_SIMD(0u, 1u), core::vector3df_SIMD(0.0f, 0.0f, 1.0f));
	}

	vertices->addUsageFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	retval.bindings[0] = {0ull, std::move(vertices)};

	return retval;
}

/*
	Helpful Icosphere class implementation used to compute
	and create icopshere's vertices and indecies.

	Polyhedron subdividing icosahedron (20 tris) by N-times iteration
    The icosphere with N=1 (default) has 80 triangles by subdividing a triangle
    of icosahedron into 4 triangles. If N=0, it is identical to icosahedron.
*/

class Icosphere
{
public:
	Icosphere(float radius = 1.0f, int subdivision = 1, bool smooth = false) : radius(radius), subdivision(subdivision), smooth(smooth), interleavedStride(32)
	{
		if (smooth)
			buildVerticesSmooth();
		else
			buildVerticesFlat();
	}

	~Icosphere() {}

	unsigned int getVertexCount() const { return (unsigned int)vertices.size() / 3; }
	unsigned int getNormalCount() const { return (unsigned int)normals.size() / 3; }
	unsigned int getTexCoordCount() const { return (unsigned int)texCoords.size() / 2; }
	unsigned int getIndexCount() const { return (unsigned int)indices.size(); }
	unsigned int getLineIndexCount() const { return (unsigned int)lineIndices.size(); }
	unsigned int getTriangleCount() const { return getIndexCount() / 3; }

	unsigned int getVertexSize() const { return (unsigned int)vertices.size() * sizeof(float); }   // # of bytes
	unsigned int getNormalSize() const { return (unsigned int)normals.size() * sizeof(float); }
	unsigned int getTexCoordSize() const { return (unsigned int)texCoords.size() * sizeof(float); }
	unsigned int getIndexSize() const { return (unsigned int)indices.size() * sizeof(unsigned int); }
	unsigned int getLineIndexSize() const { return (unsigned int)lineIndices.size() * sizeof(unsigned int); }

	const float* getVertices() const { return vertices.data(); }
	const float* getNormals() const { return normals.data(); }
	const float* getTexCoords() const { return texCoords.data(); }
	const unsigned int* getIndices() const { return indices.data(); }
	const unsigned int* getLineIndices() const { return lineIndices.data(); }

	// for interleaved vertices: V/N/T
	unsigned int getInterleavedVertexCount() const { return getVertexCount(); }    // # of vertices
	unsigned int getInterleavedVertexSize() const { return (unsigned int)interleavedVertices.size() * sizeof(float); }    // # of bytes
	int getInterleavedStride() const { return interleavedStride; }   // should be 32 bytes
	const float* getInterleavedVertices() const { return interleavedVertices.data(); }

protected:

private:

	/*
		return face normal (4th param) of a triangle v1-v2-v3
		if a triangle has no surface (normal length = 0), then return a zero vector
	*/

	static inline void computeFaceNormal(const float v1[3], const float v2[3], const float v3[3], float normal[3])
	{
		constexpr float EPSILON = 0.000001f;

		// default return value (0, 0, 0)
		normal[0] = normal[1] = normal[2] = 0;

		// find 2 edge vectors: v1-v2, v1-v3
		float ex1 = v2[0] - v1[0];
		float ey1 = v2[1] - v1[1];
		float ez1 = v2[2] - v1[2];
		float ex2 = v3[0] - v1[0];
		float ey2 = v3[1] - v1[1];
		float ez2 = v3[2] - v1[2];

		// cross product: e1 x e2
		float nx, ny, nz;
		nx = ey1 * ez2 - ez1 * ey2;
		ny = ez1 * ex2 - ex1 * ez2;
		nz = ex1 * ey2 - ey1 * ex2;

		// normalize only if the length is > 0
		float length = sqrtf(nx * nx + ny * ny + nz * nz);
		if (length > EPSILON)
		{
			// normalize
			float lengthInv = 1.0f / length;
			normal[0] = nx * lengthInv;
			normal[1] = ny * lengthInv;
			normal[2] = nz * lengthInv;
		}
	}

	/*
		return vertex normal (2nd param) by mormalizing the vertex vector
	*/

	static inline void computeVertexNormal(const float v[3], float normal[3])
	{
		// normalize
		float scale = Icosphere::computeScaleForLength(v, 1);
		normal[0] = v[0] * scale;
		normal[1] = v[1] * scale;
		normal[2] = v[2] * scale;
	}

	/*
		get the scale factor for vector to resize to the given length of vector
	*/

	static inline float computeScaleForLength(const float v[3], float length)
	{
		// and normalize the vector then re-scale to new radius
		return length / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}

	/*
		find middle point of 2 vertices
		NOTE: new vertex must be resized, so the length is equal to the given length
	*/

	static inline void computeHalfVertex(const float v1[3], const float v2[3], float length, float newV[3])
	{
		newV[0] = v1[0] + v2[0];
		newV[1] = v1[1] + v2[1];
		newV[2] = v1[2] + v2[2];
		float scale = Icosphere::computeScaleForLength(newV, length);
		newV[0] *= scale;
		newV[1] *= scale;
		newV[2] *= scale;
	}

	/*
		find middle texcoords of 2 tex coords and return new coord (3rd param)
	*/

	static inline void computeHalfTexCoord(const float t1[2], const float t2[2], float newT[2])
	{
		newT[0] = (t1[0] + t2[0]) * 0.5f;
		newT[1] = (t1[1] + t2[1]) * 0.5f;
	}

	/*
		This function used 20 non-shared line segments to determine if the given
		texture coordinate is shared or no. If it is on the line segments, it is also
		non-shared point

		   00  01  02  03  04         
		   /\  /\  /\  /\  /\         
		  /  \/  \/  \/  \/  \        
		 05  06  07  08  09   \       
		   \   10  11  12  13  14     
			\  /\  /\  /\  /\  /      
			 \/  \/  \/  \/  \/       
			  15  16  17  18  19      
	*/

	static inline bool isSharedTexCoord(const float t[2])
	{
		// 20 non-shared line segments
		const float S = 1.0f / 11;  // texture steps
		const float T = 1.0f / 3;
		static float segments[] = { S, 0,       0, T,			// 00 - 05
									S, 0,       S * 2, T,		// 00 - 06
									S * 3, 0,     S * 2, T,     // 01 - 06
									S * 3, 0,     S * 4, T,     // 01 - 07
									S * 5, 0,     S * 4, T,     // 02 - 07
									S * 5, 0,     S * 6, T,     // 02 - 08
									S * 7, 0,     S * 6, T,     // 03 - 08
									S * 7, 0,     S * 8, T,     // 03 - 09
									S * 9, 0,     S * 8, T,     // 04 - 09
									S * 9, 0,     1, T * 2,     // 04 - 14
									0, T,       S * 2, 1,		// 05 - 15
									S * 3, T * 2,   S * 2, 1,   // 10 - 15
									S * 3, T * 2,   S * 4, 1,   // 10 - 16
									S * 5, T * 2,   S * 4, 1,   // 11 - 16
									S * 5, T * 2,   S * 6, 1,   // 11 - 17
									S * 7, T * 2,   S * 6, 1,   // 12 - 17
									S * 7, T * 2,   S * 8, 1,   // 12 - 18
									S * 9, T * 2,   S * 8, 1,   // 13 - 18
									S * 9, T * 2,   S * 10, 1,  // 13 - 19
									1, T * 2,     S * 10, 1 };  // 14 - 19

		// check the point with all 20 line segments
		// if it is on the line segment, it is non-shared
		int32_t count = (sizeof(segments) / sizeof(segments[0]));
		for (int32_t i = 0, j = 2; i < count; i += 4, j += 4)
		{
			if (Icosphere::isOnLineSegment(&segments[i], &segments[j], t))
				return false;   // not shared
		}

		return true;
	}

	/*
		determine a point c is on the line segment a-b
	*/

	static inline bool isOnLineSegment(const float a[2], const float b[2], const float c[2])
	{
		constexpr float EPSILON = 0.0001f;

		// cross product must be 0 if c is on the line
		float cross = ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]));
		if (cross > EPSILON || cross < -EPSILON)
			return false;

		// c must be within a-b
		if ((c[0] > a[0] && c[0] > b[0]) || (c[0] < a[0] && c[0] < b[0]))
			return false;
		if ((c[1] > a[1] && c[1] > b[1]) || (c[1] < a[1] && c[1] < b[1]))
			return false;

		return true;    // all passed, it is on the line segment
	}

	void updateRadius()
	{
		float scale = computeScaleForLength(&vertices[0], radius);

		std::size_t i, j;
		std::size_t count = vertices.size();
		for (i = 0, j = 0; i < count; i += 3, j += 8)
		{
			vertices[i] *= scale;
			vertices[i + 1] *= scale;
			vertices[i + 2] *= scale;

			// for interleaved array
			interleavedVertices[j] *= scale;
			interleavedVertices[j + 1] *= scale;
			interleavedVertices[j + 2] *= scale;
		}
	}

	/*
		compute 12 vertices of icosahedron using spherical coordinates
		The north pole is at (0, 0, r) and the south pole is at (0,0,-r).
		5 vertices are placed by rotating 72 deg at elevation 26.57 deg (=atan(1/2))
		5 vertices are placed by rotating 72 deg at elevation -26.57 deg
	*/

	core::vector<float> computeIcosahedronVertices()		
	{
		const float PI = acos(-1);
		const float H_ANGLE = PI / 180 * 72;    // 72 degree = 360 / 5
		const float V_ANGLE = atanf(1.0f / 2);  // elevation = 26.565 degree

		core::vector<float> vertices(12 * 3);    // 12 vertices
		int32_t i1, i2;                             // indices
		float z, xy;                            // coords
		float hAngle1 = -PI / 2 - H_ANGLE / 2;  // start from -126 deg at 2nd row
		float hAngle2 = -PI / 2;                // start from -90 deg at 3rd row

		// the first top vertex (0, 0, r)
		vertices[0] = 0;
		vertices[1] = 0;
		vertices[2] = radius;

		// 10 vertices at 2nd and 3rd rows
		for (auto i = 1; i <= 5; ++i)
		{
			i1 = i * 3;         // for 2nd row
			i2 = (i + 5) * 3;   // for 3rd row

			z = radius * sinf(V_ANGLE);             // elevaton
			xy = radius * cosf(V_ANGLE);

			vertices[i1] = xy * cosf(hAngle1);      // x
			vertices[i2] = xy * cosf(hAngle2);
			vertices[i1 + 1] = xy * sinf(hAngle1);  // x
			vertices[i2 + 1] = xy * sinf(hAngle2);
			vertices[i1 + 2] = z;                   // z
			vertices[i2 + 2] = -z;

			// next horizontal angles
			hAngle1 += H_ANGLE;
			hAngle2 += H_ANGLE;
		}

		// the last bottom vertex (0, 0, -r)
		i1 = 11 * 3;
		vertices[i1] = 0;
		vertices[i1 + 1] = 0;
		vertices[i1 + 2] = -radius;

		return vertices;
	}
	/*
		generate vertices with flat shading
		each triangle is independent (no shared vertices)
	*/

	void buildVerticesFlat()
	{
		//const float S_STEP = 1 / 11.0f;         // horizontal texture step
		//const float T_STEP = 1 / 3.0f;          // vertical texture step
		const float S_STEP = 186 / 2048.0f;		  // horizontal texture step
		const float T_STEP = 322 / 1024.0f;		  // vertical texture step

		// compute 12 vertices of icosahedron
		core::vector<float> tmpVertices = computeIcosahedronVertices();

		// clear memory of prev arrays
		core::vector<float>().swap(vertices);
		core::vector<float>().swap(normals);
		core::vector<float>().swap(texCoords);
		core::vector<uint32_t>().swap(indices);
		core::vector<uint32_t>().swap(lineIndices);

		const float* v0, * v1, * v2, * v3, * v4, * v11;          // vertex positions
		float n[3];												 // face normal
		float t0[2], t1[2], t2[2], t3[2], t4[2], t11[2];         // texCoords
		uint32_t index = 0;

		// compute and add 20 tiangles of icosahedron first
		v0 = &tmpVertices[0];       // 1st vertex
		v11 = &tmpVertices[11 * 3]; // 12th vertex
		for (auto i = 1; i <= 5; ++i)
		{
			// 4 vertices in the 2nd row
			v1 = &tmpVertices[i * 3];
			if (i < 5)
				v2 = &tmpVertices[(i + 1) * 3];
			else
				v2 = &tmpVertices[3];

			v3 = &tmpVertices[(i + 5) * 3];
			if ((i + 5) < 10)
				v4 = &tmpVertices[(i + 6) * 3];
			else
				v4 = &tmpVertices[6 * 3];

			// texture coords
			t0[0] = (2 * i - 1) * S_STEP;   t0[1] = 0;
			t1[0] = (2 * i - 2) * S_STEP;   t1[1] = T_STEP;
			t2[0] = (2 * i - 0) * S_STEP;   t2[1] = T_STEP;
			t3[0] = (2 * i - 1) * S_STEP;   t3[1] = T_STEP * 2;
			t4[0] = (2 * i + 1) * S_STEP;   t4[1] = T_STEP * 2;
			t11[0] = 2 * i * S_STEP;         t11[1] = T_STEP * 3;

			// add a triangle in 1st row
			Icosphere::computeFaceNormal(v0, v1, v2, n);
			addVertices(v0, v1, v2);
			addNormals(n, n, n);
			addTexCoords(t0, t1, t2);
			addIndices(index, index + 1, index + 2);

			// add 2 triangles in 2nd row
			Icosphere::computeFaceNormal(v1, v3, v2, n);
			addVertices(v1, v3, v2);
			addNormals(n, n, n);
			addTexCoords(t1, t3, t2);
			addIndices(index + 3, index + 4, index + 5);

			Icosphere::computeFaceNormal(v2, v3, v4, n);
			addVertices(v2, v3, v4);
			addNormals(n, n, n);
			addTexCoords(t2, t3, t4);
			addIndices(index + 6, index + 7, index + 8);

			// add a triangle in 3rd row
			Icosphere::computeFaceNormal(v3, v11, v4, n);
			addVertices(v3, v11, v4);
			addNormals(n, n, n);
			addTexCoords(t3, t11, t4);
			addIndices(index + 9, index + 10, index + 11);

			// add 6 edge lines per iteration
			//  i
			//  /   /   /   /   /       : (i, i+1)                              
			// /__ /__ /__ /__ /__                                              
			// \  /\  /\  /\  /\  /     : (i+3, i+4), (i+3, i+5), (i+4, i+5)    
			//  \/__\/__\/__\/__\/__                                            
			//   \   \   \   \   \      : (i+9,i+10), (i+9, i+11)               
			//    \   \   \   \   \                                             
			lineIndices.push_back(index);		  // (i, i+1)
			lineIndices.push_back(index + 1);     // (i, i+1)
			lineIndices.push_back(index + 3);     // (i+3, i+4)
			lineIndices.push_back(index + 4);
			lineIndices.push_back(index + 3);     // (i+3, i+5)
			lineIndices.push_back(index + 5);
			lineIndices.push_back(index + 4);     // (i+4, i+5)
			lineIndices.push_back(index + 5);
			lineIndices.push_back(index + 9);     // (i+9, i+10)
			lineIndices.push_back(index + 10);
			lineIndices.push_back(index + 9);     // (i+9, i+11)
			lineIndices.push_back(index + 11);

			// next index
			index += 12;
		}

		// subdivide icosahedron
		subdivideVerticesFlat();

		// generate interleaved vertex array as well
		buildInterleavedVertices();
	}

	/*
		 generate vertices with smooth shading
		 NOTE: The north and south pole vertices cannot be shared for smooth shading
		 because they have same position and normal, but different texcoords per face
		 And, the first vertex on each row is also not shared.
	*/

	void buildVerticesSmooth()
	{
		//const float S_STEP = 1 / 11.0f;            // horizontal texture step
		//const float T_STEP = 1 / 3.0f;		     // vertical texture step
		const float S_STEP = 186 / 2048.0f;			 // horizontal texture step
		const float T_STEP = 322 / 1024.0f;			 // vertical texture step

		// compute 12 vertices of icosahedron
		// NOTE: v0 (top), v11(bottom), v1, v6(first vert on each row) cannot be
		// shared for smooth shading (they have different texcoords)
		core::vector<float> tmpVertices = computeIcosahedronVertices();

		// clear memory of prev arrays
		core::vector<float>().swap(vertices);
		core::vector<float>().swap(normals);
		core::vector<float>().swap(texCoords);
		core::vector<uint32_t>().swap(indices);
		core::vector<uint32_t>().swap(lineIndices);
		std::map<std::pair<float, float>, uint32_t>().swap(sharedIndices);

		float v[3];                             // vertex
		float n[3];                             // normal
		float scale;                            // scale factor for normalization

		// smooth icosahedron has 14 non-shared (0 to 13) and
		// 8 shared vertices (14 to 21) (total 22 vertices)
		//  00  01  02  03  04          
		//  /\  /\  /\  /\  /\          
		// /  \/  \/  \/  \/  \         
		//10--14--15--16--17--11        
		// \  /\  /\  /\  /\  /\        
		//  \/  \/  \/  \/  \/  \       
		//  12--18--19--20--21--13      
		//   \  /\  /\  /\  /\  /       
		//    \/  \/  \/  \/  \/        
		//    05  06  07  08  09        
		// add 14 non-shared vertices first (index from 0 to 13)

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v0 (top)
		addNormal(0, 0, 1);
		addTexCoord(S_STEP, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v1
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 3, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v2
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 5, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v3
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 7, 0);

		addVertex(tmpVertices[0], tmpVertices[1], tmpVertices[2]);      // v4
		addNormal(0, 0, 1);
		addTexCoord(S_STEP * 9, 0);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v5 (bottom)
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 2, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v6
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 4, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v7
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 6, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v8
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 8, T_STEP * 3);

		addVertex(tmpVertices[33], tmpVertices[34], tmpVertices[35]);   // v9
		addNormal(0, 0, -1);
		addTexCoord(S_STEP * 10, T_STEP * 3);

		v[0] = tmpVertices[3];  v[1] = tmpVertices[4];  v[2] = tmpVertices[5];  // v10 (left)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(0, T_STEP);

		addVertex(v[0], v[1], v[2]);                                            // v11 (right)
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 10, T_STEP);

		v[0] = tmpVertices[18]; v[1] = tmpVertices[19]; v[2] = tmpVertices[20]; // v12 (left)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP, T_STEP * 2);

		addVertex(v[0], v[1], v[2]);                                            // v13 (right)
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 11, T_STEP * 2);

		// add 8 shared vertices to array (index from 14 to 21)
		v[0] = tmpVertices[6];  v[1] = tmpVertices[7];  v[2] = tmpVertices[8];  // v14 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 2, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 2, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[9];  v[1] = tmpVertices[10]; v[2] = tmpVertices[11]; // v15 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 4, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 4, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[12]; v[1] = tmpVertices[13]; v[2] = tmpVertices[14]; // v16 (shared)
		scale = Icosphere::computeScaleForLength(v, 1);
		n[0] = v[0] * scale;    n[1] = v[1] * scale;    n[2] = v[2] * scale;
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 6, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 6, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[15]; v[1] = tmpVertices[16]; v[2] = tmpVertices[17]; // v17 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 8, T_STEP);
		sharedIndices[std::make_pair(S_STEP * 8, T_STEP)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[21]; v[1] = tmpVertices[22]; v[2] = tmpVertices[23]; // v18 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 3, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 3, T_STEP * 2)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[24]; v[1] = tmpVertices[25]; v[2] = tmpVertices[26]; // v19 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 5, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 5, T_STEP * 2)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[27]; v[1] = tmpVertices[28]; v[2] = tmpVertices[29]; // v20 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 7, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 7, T_STEP * 2)] = texCoords.size() / 2 - 1;

		v[0] = tmpVertices[30]; v[1] = tmpVertices[31]; v[2] = tmpVertices[32]; // v21 (shared)
		Icosphere::computeVertexNormal(v, n);
		addVertex(v[0], v[1], v[2]);
		addNormal(n[0], n[1], n[2]);
		addTexCoord(S_STEP * 9, T_STEP * 2);
		sharedIndices[std::make_pair(S_STEP * 9, T_STEP * 2)] = texCoords.size() / 2 - 1;

		// build index list for icosahedron (20 triangles)
		addIndices(0, 10, 14);      // 1st row (5 tris)
		addIndices(1, 14, 15);
		addIndices(2, 15, 16);
		addIndices(3, 16, 17);
		addIndices(4, 17, 11);
		addIndices(10, 12, 14);      // 2nd row (10 tris)
		addIndices(12, 18, 14);
		addIndices(14, 18, 15);
		addIndices(18, 19, 15);
		addIndices(15, 19, 16);
		addIndices(19, 20, 16);
		addIndices(16, 20, 17);
		addIndices(20, 21, 17);
		addIndices(17, 21, 11);
		addIndices(21, 13, 11);
		addIndices(5, 18, 12);      // 3rd row (5 tris)
		addIndices(6, 19, 18);
		addIndices(7, 20, 19);
		addIndices(8, 21, 20);
		addIndices(9, 13, 21);

		// add edge lines of icosahedron
		lineIndices.push_back(0);   lineIndices.push_back(10);       // 00 - 10
		lineIndices.push_back(1);   lineIndices.push_back(14);       // 01 - 14
		lineIndices.push_back(2);   lineIndices.push_back(15);       // 02 - 15
		lineIndices.push_back(3);   lineIndices.push_back(16);       // 03 - 16
		lineIndices.push_back(4);   lineIndices.push_back(17);       // 04 - 17
		lineIndices.push_back(10);  lineIndices.push_back(14);       // 10 - 14
		lineIndices.push_back(14);  lineIndices.push_back(15);       // 14 - 15
		lineIndices.push_back(15);  lineIndices.push_back(16);       // 15 - 16
		lineIndices.push_back(16);  lineIndices.push_back(17);       // 10 - 14
		lineIndices.push_back(17);  lineIndices.push_back(11);       // 17 - 11
		lineIndices.push_back(10);  lineIndices.push_back(12);       // 10 - 12
		lineIndices.push_back(12);  lineIndices.push_back(14);       // 12 - 14
		lineIndices.push_back(14);  lineIndices.push_back(18);       // 14 - 18
		lineIndices.push_back(18);  lineIndices.push_back(15);       // 18 - 15
		lineIndices.push_back(15);  lineIndices.push_back(19);       // 15 - 19
		lineIndices.push_back(19);  lineIndices.push_back(16);       // 19 - 16
		lineIndices.push_back(16);  lineIndices.push_back(20);       // 16 - 20
		lineIndices.push_back(20);  lineIndices.push_back(17);       // 20 - 17
		lineIndices.push_back(17);  lineIndices.push_back(21);       // 17 - 21
		lineIndices.push_back(21);  lineIndices.push_back(11);       // 21 - 11
		lineIndices.push_back(12);  lineIndices.push_back(18);       // 12 - 18
		lineIndices.push_back(18);  lineIndices.push_back(19);       // 18 - 19
		lineIndices.push_back(19);  lineIndices.push_back(20);       // 19 - 20
		lineIndices.push_back(20);  lineIndices.push_back(21);       // 20 - 21
		lineIndices.push_back(21);  lineIndices.push_back(13);       // 21 - 13
		lineIndices.push_back(5);   lineIndices.push_back(12);       // 05 - 12
		lineIndices.push_back(6);   lineIndices.push_back(18);       // 06 - 18
		lineIndices.push_back(7);   lineIndices.push_back(19);       // 07 - 19
		lineIndices.push_back(8);   lineIndices.push_back(20);       // 08 - 20
		lineIndices.push_back(9);   lineIndices.push_back(21);       // 09 - 21

		// subdivide icosahedron
		subdivideVerticesSmooth();

		// generate interleaved vertex array as well
		buildInterleavedVertices();
	}
	/*
		divide a trinage into 4 sub triangles and repeat N times
		If subdivision=0, do nothing.
	*/

	void subdivideVerticesFlat()
	{
		core::vector<float> tmpVertices;
		core::vector<float> tmpTexCoords;
		core::vector<unsigned int> tmpIndices;
		int32_t indexCount;
		const float* v1, * v2, * v3;			// ptr to original vertices of a triangle
		const float* t1, * t2, * t3;			// ptr to original texcoords of a triangle
		float newV1[3], newV2[3], newV3[3];		// new vertex positions
		float newT1[2], newT2[2], newT3[2];		// new texture coords
		float normal[3];						// new face normal
		uint32_t index = 0;					// new index value
		int32_t i, j;

		// iteration
		for (i = 1; i <= subdivision; ++i)
		{
			// copy prev arrays
			tmpVertices = vertices;
			tmpTexCoords = texCoords;
			tmpIndices = indices;

			// clear prev arrays
			vertices.clear();
			normals.clear();
			texCoords.clear();
			indices.clear();
			lineIndices.clear();

			index = 0;
			indexCount = (int)tmpIndices.size();
			for (j = 0; j < indexCount; j += 3)
			{
				// get 3 vertice and texcoords of a triangle
				v1 = &tmpVertices[tmpIndices[j] * 3];
				v2 = &tmpVertices[tmpIndices[j + 1] * 3];
				v3 = &tmpVertices[tmpIndices[j + 2] * 3];
				t1 = &tmpTexCoords[tmpIndices[j] * 2];
				t2 = &tmpTexCoords[tmpIndices[j + 1] * 2];
				t3 = &tmpTexCoords[tmpIndices[j + 2] * 2];

				// get 3 new vertices by spliting half on each edge
				computeHalfVertex(v1, v2, radius, newV1);
				computeHalfVertex(v2, v3, radius, newV2);
				computeHalfVertex(v1, v3, radius, newV3);
				computeHalfTexCoord(t1, t2, newT1);
				computeHalfTexCoord(t2, t3, newT2);
				computeHalfTexCoord(t1, t3, newT3);

				// add 4 new triangles
				addVertices(v1, newV1, newV3);
				addTexCoords(t1, newT1, newT3);
				computeFaceNormal(v1, newV1, newV3, normal);
				addNormals(normal, normal, normal);
				addIndices(index, index + 1, index + 2);

				addVertices(newV1, v2, newV2);
				addTexCoords(newT1, t2, newT2);
				computeFaceNormal(newV1, v2, newV2, normal);
				addNormals(normal, normal, normal);
				addIndices(index + 3, index + 4, index + 5);

				addVertices(newV1, newV2, newV3);
				addTexCoords(newT1, newT2, newT3);
				computeFaceNormal(newV1, newV2, newV3, normal);
				addNormals(normal, normal, normal);
				addIndices(index + 6, index + 7, index + 8);

				addVertices(newV3, newV2, v3);
				addTexCoords(newT3, newT2, t3);
				computeFaceNormal(newV3, newV2, v3, normal);
				addNormals(normal, normal, normal);
				addIndices(index + 9, index + 10, index + 11);

				// add new line indices per iteration
				addSubLineIndices(index, index + 1, index + 4, index + 5, index + 11, index + 9); //CCW

				// next index
				index += 12;
			}
		}
	}

	/*
		divide a trianlge (v1-v2-v3) into 4 sub triangles by adding middle vertices
		(newV1, newV2, newV3) and repeat N times
		If subdivision=0, do nothing.

				 v1           
				/ \           
		 newV1 *---* newV3    
			  / \ / \         
			v2---*---v3       
				newV2         
	*/

	void subdivideVerticesSmooth()
	{
		core::vector<uint32_t> tmpIndices;
		int32_t indexCount;
		uint32_t i1, i2, i3;            // indices from original triangle
		const float* v1, * v2, * v3;          // ptr to original vertices of a triangle
		const float* t1, * t2, * t3;          // ptr to original texcoords of a triangle
		float newV1[3], newV2[3], newV3[3]; // new subdivided vertex positions
		float newN1[3], newN2[3], newN3[3]; // new subdivided normals
		float newT1[2], newT2[2], newT3[2]; // new subdivided texture coords
		uint32_t newI1, newI2, newI3;   // new subdivided indices
		int32_t i, j;

		// iteration for subdivision
		for (i = 1; i <= subdivision; ++i)
		{
			// copy prev indices
			tmpIndices = indices;

			// clear prev arrays
			indices.clear();
			lineIndices.clear();

			indexCount = tmpIndices.size();
			for (j = 0; j < indexCount; j += 3)
			{
				// get 3 indices of each triangle
				i1 = tmpIndices[j];
				i2 = tmpIndices[j + 1];
				i3 = tmpIndices[j + 2];

				// get 3 vertex attribs from prev triangle
				v1 = &vertices[i1 * 3];
				v2 = &vertices[i2 * 3];
				v3 = &vertices[i3 * 3];
				t1 = &texCoords[i1 * 2];
				t2 = &texCoords[i2 * 2];
				t3 = &texCoords[i3 * 2];

				// get 3 new vertex attribs by spliting half on each edge
				computeHalfVertex(v1, v2, radius, newV1);
				computeHalfVertex(v2, v3, radius, newV2);
				computeHalfVertex(v1, v3, radius, newV3);
				computeHalfTexCoord(t1, t2, newT1);
				computeHalfTexCoord(t2, t3, newT2);
				computeHalfTexCoord(t1, t3, newT3);
				computeVertexNormal(newV1, newN1);
				computeVertexNormal(newV2, newN2);
				computeVertexNormal(newV3, newN3);

				// add new vertices/normals/texcoords to arrays
				// It will check if it is shared/non-shared and return index
				newI1 = addSubVertexAttribs(newV1, newN1, newT1);
				newI2 = addSubVertexAttribs(newV2, newN2, newT2);
				newI3 = addSubVertexAttribs(newV3, newN3, newT3);

				// add 4 new triangle indices
				addIndices(i1, newI1, newI3);
				addIndices(newI1, i2, newI2);
				addIndices(newI1, newI2, newI3);
				addIndices(newI3, newI2, i3);

				// add new line indices
				addSubLineIndices(i1, newI1, i2, newI2, i3, newI3); //CCW
			}
		}
	}

	/*
		generate interleaved vertices: V/N/T
		stride must be 32 bytes
	*/

	void buildInterleavedVertices()
	{
		core::vector<float>().swap(interleavedVertices);

		std::size_t i, j;
		std::size_t count = vertices.size();
		for (i = 0, j = 0; i < count; i += 3, j += 2)
		{
			interleavedVertices.push_back(vertices[i]);
			interleavedVertices.push_back(vertices[i + 1]);
			interleavedVertices.push_back(vertices[i + 2]);

			interleavedVertices.push_back(normals[i]);
			interleavedVertices.push_back(normals[i + 1]);
			interleavedVertices.push_back(normals[i + 2]);

			interleavedVertices.push_back(texCoords[j]);
			interleavedVertices.push_back(texCoords[j + 1]);
		}
	}

	void addVertex(float x, float y, float z)
	{
		vertices.push_back(x);
		vertices.push_back(y);
		vertices.push_back(z);
	}

	void addVertices(const float v1[3], const float v2[3], const float v3[3])
	{
		vertices.push_back(v1[0]);  // x
		vertices.push_back(v1[1]);  // y
		vertices.push_back(v1[2]);  // z
		vertices.push_back(v2[0]);
		vertices.push_back(v2[1]);
		vertices.push_back(v2[2]);
		vertices.push_back(v3[0]);
		vertices.push_back(v3[1]);
		vertices.push_back(v3[2]);
	}

	void addNormal(float nx, float ny, float nz)
	{
		normals.push_back(nx);
		normals.push_back(ny);
		normals.push_back(nz);
	}

	void addNormals(const float n1[3], const float n2[3], const float n3[3])
	{
		normals.push_back(n1[0]);  // nx
		normals.push_back(n1[1]);  // ny
		normals.push_back(n1[2]);  // nz
		normals.push_back(n2[0]);
		normals.push_back(n2[1]);
		normals.push_back(n2[2]);
		normals.push_back(n3[0]);
		normals.push_back(n3[1]);
		normals.push_back(n3[2]);
	}

	void addTexCoord(float s, float t)
	{
		texCoords.push_back(s);
		texCoords.push_back(t);
	}

	void addTexCoords(const float t1[2], const float t2[2], const float t3[2])
	{
		texCoords.push_back(t1[0]); // s
		texCoords.push_back(t1[1]); // t
		texCoords.push_back(t2[0]);
		texCoords.push_back(t2[1]);
		texCoords.push_back(t3[0]);
		texCoords.push_back(t3[1]);
	}

	void addIndices(unsigned int i1, unsigned int i2, unsigned int i3)
	{
		indices.push_back(i1);
		indices.push_back(i2);
		indices.push_back(i3);
	}

	/*
		add 7 sub edge lines per triangle to array using 6 indices (CCW)           
			 i1                                                                     
			 /            : (i1, i2)                                                
		   i2---i6        : (i2, i6)												  
		   / \  /         : (i2, i3), (i2, i4), (i6, i4)							  
		 i3---i4---i5     : (i3, i4), (i4, i5)									  
	*/

	void addSubLineIndices(unsigned int i1, unsigned int i2, unsigned int i3, unsigned int i4, unsigned int i5, unsigned int i6)
	{
		lineIndices.push_back(i1);      // i1 - i2
		lineIndices.push_back(i2);
		lineIndices.push_back(i2);      // i2 - i6
		lineIndices.push_back(i6);
		lineIndices.push_back(i2);      // i2 - i3
		lineIndices.push_back(i3);
		lineIndices.push_back(i2);      // i2 - i4
		lineIndices.push_back(i4);
		lineIndices.push_back(i6);      // i6 - i4
		lineIndices.push_back(i4);
		lineIndices.push_back(i3);      // i3 - i4
		lineIndices.push_back(i4);
		lineIndices.push_back(i4);      // i4 - i5
		lineIndices.push_back(i5);
	}

	/*
		add a subdivided vertex attribs (vertex, normal, texCoord) to arrays, then
		return its index value
		If it is a shared vertex, remember its index, so it can be re-used
	*/

	unsigned int addSubVertexAttribs(const float v[3], const float n[3], const float t[2])
	{
		unsigned int index; // return value;

		// check if is shared vertex or not first
		if (Icosphere::isSharedTexCoord(t))
		{
			// find if it does already exist in sharedIndices map using (s,t) key
			// if not in the list, add the vertex attribs to arrays and return its index
			// if exists, return the current index
			std::pair<float, float> key = std::make_pair(t[0], t[1]);
			std::map<std::pair<float, float>, unsigned int>::iterator iter = sharedIndices.find(key);
			if (iter == sharedIndices.end())
			{
				addVertex(v[0], v[1], v[2]);
				addNormal(n[0], n[1], n[2]);
				addTexCoord(t[0], t[1]);
				index = texCoords.size() / 2 - 1;
				sharedIndices[key] = index;
			}
			else
			{
				index = iter->second;
			}
		}
		// not shared
		else
		{
			addVertex(v[0], v[1], v[2]);
			addNormal(n[0], n[1], n[2]);
			addTexCoord(t[0], t[1]);
			index = texCoords.size() / 2 - 1;
		}

		return index;
	}

	float radius;													// circumscribed radius
	uint32_t subdivision;
	bool smooth;
	core::vector<float> vertices;
	core::vector<float> normals;
	core::vector<float> texCoords;
	core::vector<uint32_t> indices;
	core::vector<uint32_t> lineIndices;
	std::map<std::pair<float, float>, uint32_t> sharedIndices;   // indices of shared vertices, key is tex coord (s,t)

	// interleaved
	core::vector<float> interleavedVertices;
	uint32_t interleavedStride;											// # of bytes to hop to the next vertex (should be 32 bytes)

};

CGeometryCreator::return_type CGeometryCreator::createIcoSphere(float radius, uint32_t subdivision, bool smooth) const
{
	Icosphere IcosphereData(radius, subdivision, smooth);
	
	return_type icosphereGeometry;

	constexpr size_t vertexSize = sizeof(IcosphereVertex);

	icosphereGeometry.inputParams =
	{ 0b111u,0b1u,
		{
			{0u, EF_R32G32B32_SFLOAT, offsetof(IcosphereVertex,pos)},
			{0u, EF_R32G32B32_SFLOAT, offsetof(IcosphereVertex,normals)},
			{0u, EF_R32G32_SFLOAT, offsetof(IcosphereVertex,uv)}
		},
		{vertexSize,SVertexInputBindingParams::EVIR_PER_VERTEX} 
	};

	auto vertexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(IcosphereData.getInterleavedVertexSize());
	auto indexBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(IcosphereData.getIndexSize());

	memcpy(vertexBuffer->getPointer(), IcosphereData.getInterleavedVertices(), vertexBuffer->getSize());
	memcpy(indexBuffer->getPointer(), IcosphereData.getIndices(), indexBuffer->getSize());

	vertexBuffer->addUsageFlags(asset::IBuffer::EUF_VERTEX_BUFFER_BIT);
	icosphereGeometry.bindings[0] = { 0, std::move(vertexBuffer) };
	indexBuffer->addUsageFlags(asset::IBuffer::EUF_INDEX_BUFFER_BIT);
	icosphereGeometry.indexBuffer = { 0, std::move(indexBuffer) };
	icosphereGeometry.indexCount = IcosphereData.getIndexCount();
	icosphereGeometry.indexType = EIT_32BIT;

	return icosphereGeometry;
}


} // end namespace nbl::asset


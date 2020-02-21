// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "os.h"

#include "irr/asset/CGeometryCreator.h"
#include "irr/asset/normal_quantization.h"
#include "irr/asset/CCPUMesh.h"

namespace irr
{
namespace asset
{

CGeometryCreator::return_type CGeometryCreator::createCubeMesh(const core::vector3df& size) const
{
	return_type retval;

	constexpr size_t vertexSize = sizeof(CGeometryCreator::CubeVertex);
	retval.inputParams = {0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(CubeVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(CubeVertex,color)},
											{0u,EF_R8G8_USCALED,offsetof(CubeVertex,uv)},
											{0u,EF_R8G8B8_SSCALED,offsetof(CubeVertex,normal)}
										},{vertexSize,EVIR_PER_VERTEX}};

	// Create indices
	{
		retval.indexCount = 36u;
		auto indices = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint16_t)*retval.indexCount);
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
																const video::SColor vtxColor1) const
{
#ifndef NEW_SHADERS
    assert(height > cylinderHeight);

    auto cylinder = createCylinderMesh(width0, cylinderHeight, tesselationCylinder, vtxColor0);
	// TODO: disk meshbuffer to close it
    auto cone = core::move_and_static_cast<asset::CCPUMesh>(createConeMesh(width1, height-cylinderHeight, tesselationCone, vtxColor1, vtxColor1));

    if (!cylinder || !cone)
        return nullptr;

    asset::ICPUMeshBuffer* coneMb = cone->getMeshBuffer(0u);

    asset::ICPUBuffer* coneVtxBuf = const_cast<asset::ICPUBuffer*>(coneMb->getMeshDataAndFormat()->getMappedBuffer(asset::EVAI_ATTR0));
    ConeVertex* coneVertices = reinterpret_cast<ConeVertex*>(coneVtxBuf->getPointer());
    for (uint32_t i = 0u; i < tesselationCone+2u; ++i)
        coneVertices[i].pos[2] += cylinderHeight;
    coneMb->recalculateBoundingBox();

    cone->addMeshBuffer(core::smart_refctd_ptr<asset::ICPUMeshBuffer>(cylinder->getMeshBuffer(0u)));
    cone->recalculateBoundingBox();

    return cone;
#else
	return {};
#endif
}


/* A sphere with proper normals and texture coords */
CGeometryCreator::return_type CGeometryCreator::createSphereMesh(float radius, uint32_t polyCountX, uint32_t polyCountY) const
{
	// we are creating the sphere mesh here.
	return_type retval;
	constexpr size_t vertexSize = sizeof(CGeometryCreator::SphereVertex);
	retval.inputParams = { 0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(SphereVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(SphereVertex,color)},
											{0u,EF_R32G32_SFLOAT,offsetof(SphereVertex,uv)},
											{0u,EF_A2B10G10R10_SNORM_PACK32,offsetof(SphereVertex,normal)}
										},{vertexSize,EVIR_PER_VERTEX} };

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
				uint32_t quantizedNormal = quantizeNormal2_10_10_10(normal);
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
				((uint32_t*)tmpMemPtr)[6] = quantizedNormal;

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
		((uint32_t*)tmpMemPtr)[6] = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f, 1.f, 0.f));

		// the vertex at the bottom of the sphere
		tmpMemPtr += vertexSize;
		((float*)tmpMemPtr)[0] = 0.f;
		((float*)tmpMemPtr)[1] = -radius;
		((float*)tmpMemPtr)[2] = 0.f;
		((float*)tmpMemPtr)[4] = 0.5f;
		((float*)tmpMemPtr)[5] = 1.f;
		((uint32_t*)tmpMemPtr)[6] = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f, -1.f, 0.f));

		// recalculate bounding box
		core::aabbox3df BoundingBox;
		BoundingBox.reset(core::vector3df(radius));
		BoundingBox.addInternalPoint(-radius, -radius, -radius);

		// set vertex buffer
		retval.bindings[0] = { 0ull,std::move(vtxBuf) };
		retval.indexType = asset::EIT_32BIT;
		retval.bbox = BoundingBox;
	}

	return retval;
}

/* A cylinder with proper normals and texture coords */
CGeometryCreator::return_type CGeometryCreator::createCylinderMesh(float radius, float length,
			uint32_t tesselation, const video::SColor& color) const
{
	return_type retval;
	constexpr size_t vertexSize = sizeof(CGeometryCreator::CylinderVertex);
	retval.inputParams = { 0b1111u,0b1u,{
											{0u,EF_R32G32B32_SFLOAT,offsetof(CylinderVertex,pos)},
											{0u,EF_R8G8B8A8_UNORM,offsetof(CylinderVertex,color)},
											{0u,EF_R32G32_SFLOAT,offsetof(CylinderVertex,uv)},
											{0u,EF_A2B10G10R10_SNORM_PACK32,offsetof(CylinderVertex,normal)}
										},{vertexSize,EVIR_PER_VERTEX} };

    const size_t vtxCnt = 2u*tesselation;
    auto vtxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vtxCnt*sizeof(CylinderVertex));

    CylinderVertex* vertices = reinterpret_cast<CylinderVertex*>(vtxBuf->getPointer());
    std::fill(vertices, vertices + vtxCnt, CylinderVertex());

    const uint32_t halfIx = tesselation;

    uint8_t glcolor[4];
    color.toOpenGLColor(glcolor);

    const float tesselationRec = core::reciprocal_approxim<float>(tesselation);
    const float step = 2.f*core::PI<float>()*tesselationRec;
    for (uint32_t i = 0u; i<tesselation; ++i)
    {
        core::vectorSIMDf p(std::cos(i*step), std::sin(i*step), 0.f);
        p *= radius;
        const uint32_t n = quantizeNormal2_10_10_10(core::normalize(p));

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
	retval.indexBuffer = { 0ull, std::move(idxBuf) };
	retval.bindings[0] = { 0ull, std::move(vtxBuf) };
	retval.indexType = asset::EIT_16BIT;
	//retval.bbox = ?;

	return retval;
}

/* A cone with proper normals and texture coords */
CGeometryCreator::return_type CGeometryCreator::createConeMesh(	float radius, float length, uint32_t tesselation,
																const video::SColor& colorTop,
																const video::SColor& colorBottom,
																float oblique) const
{
#ifndef NEW_SHADERS
    const size_t vtxCnt = tesselation+2u;
    auto vtxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(vtxCnt * sizeof(ConeVertex));
    ConeVertex* vertices = reinterpret_cast<ConeVertex*>(vtxBuf->getPointer());
    std::fill(vertices, vertices + vtxCnt, ConeVertex(core::vectorSIMDf(0.f), 0u, colorBottom));

    const float step = (2.f*core::PI<float>()) / tesselation;
    for (uint32_t i = 2u; i < vtxCnt; ++i)
    {
        const core::vectorSIMDf p(std::cos(i*step), 0.f, std::sin(i*step), 0.f);
        memcpy(vertices[i].pos, (p*radius).pointer, 12u);
        vertices[i].normal = quantizeNormal2_10_10_10(core::normalize(p));
    }
    const uint32_t peakIx = 0u;
    const uint32_t bottomCenterIx = 1u;

    const core::vectorSIMDf p(oblique, length, 0.f, 0.f);
    memcpy(vertices[peakIx].pos, p.pointer, 12u);
    vertices[peakIx].normal = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f, 1.f, 0.f, 0.f));
    colorTop.toOpenGLColor(vertices[peakIx].color);
    memset(vertices[bottomCenterIx].pos, 0, 12u);
    vertices[bottomCenterIx].normal = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f, -1.f, 0.f, 0.f));

    auto idxBuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(2u*3u*tesselation*sizeof(uint16_t));
    uint16_t* indices = (uint16_t*)idxBuf->getPointer();
    
    for (uint32_t i = 2u, j = 0u; i < vtxCnt; ++i)
    {
        indices[j++] = peakIx;
        indices[j++] = i+1u == vtxCnt ? 2u : i+1u;
        indices[j++] = i;
    }
    
    indices += idxBuf->getSize()/2u/sizeof(uint16_t);

    for (uint32_t i = 2u, j = 0u; i < vtxCnt; ++i)
    {
        indices[j++] = bottomCenterIx;
        indices[j++] = i;
        indices[j++] = i+1u == vtxCnt ? 2u : i+1u;
    }

    auto mesh = core::make_smart_refctd_ptr<asset::CCPUMesh>();
    auto meshbuf = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>();
    auto desc = core::make_smart_refctd_ptr<asset::ICPUMeshDataFormatDesc>();
    desc->setVertexAttrBuffer(core::smart_refctd_ptr(vtxBuf), asset::EVAI_ATTR0, asset::EF_R32G32B32_SFLOAT, sizeof(ConeVertex), offsetof(ConeVertex, pos));
    desc->setVertexAttrBuffer(core::smart_refctd_ptr(vtxBuf), asset::EVAI_ATTR1, asset::EF_R8G8B8A8_UNORM, sizeof(ConeVertex), offsetof(ConeVertex, color));
    desc->setVertexAttrBuffer(core::smart_refctd_ptr(vtxBuf), asset::EVAI_ATTR3, asset::EF_A2B10G10R10_SNORM_PACK32, sizeof(ConeVertex), offsetof(ConeVertex, normal));
    meshbuf->setIndexCount(idxBuf->getSize()/2u);
	desc->setIndexBuffer(std::move(idxBuf));
    meshbuf->setIndexType(asset::EIT_16BIT);
    meshbuf->setPrimitiveType(asset::EPT_TRIANGLES);
    meshbuf->setMeshDataAndFormat(std::move(desc));
	meshbuf->recalculateBoundingBox();

    mesh->addMeshBuffer(std::move(meshbuf));

    mesh->recalculateBoundingBox(true);

    return mesh;
#else
	return {};
#endif
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
										},{vertexSize,EVIR_PER_VERTEX} };
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
										},{vertexSize,EVIR_PER_VERTEX} };
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

	retval.bindings[0] = {0ull, std::move(vertices)};

	return retval;
}


} // end namespace asset
} // end namespace irr


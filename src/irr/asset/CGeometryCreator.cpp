// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CGeometryCreator.h"
#include "irr/video/SGPUMesh.h"
#include "irr/asset/SCPUMesh.h"
#include "irr/asset/IMesh.h"
#include "IVideoDriver.h"
#include "SVertexManipulator.h"
#include "os.h"
#include "irr/asset/ICPUMeshBuffer.h"

namespace irr
{
namespace asset
{

asset::ICPUMesh* CGeometryCreator::createCubeMesh(const core::vector3df& size) const
{
	asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();
	asset::ICPUMeshBuffer* buffer = new asset::ICPUMeshBuffer();
	buffer->setMeshDataAndFormat(desc);
	desc->drop();

	// Create indices
	uint16_t u[36];
	for (int i = 0; i < 6; ++i)
	{
		u[i*6+0] = 4*i+0;
		u[i*6+1] = 4*i+1;
		u[i*6+2] = 4*i+3;
		u[i*6+3] = 4*i+1;
		u[i*6+4] = 4*i+2;
		u[i*6+5] = 4*i+3;
	}

    asset::ICPUBuffer* indices = new asset::ICPUBuffer(sizeof(u));
    memcpy(indices->getPointer(),u,sizeof(u));
    desc->setIndexBuffer(indices);
    buffer->setIndexType(asset::EIT_16BIT);
    buffer->setIndexCount(sizeof(u)/sizeof(*u));
    indices->drop();

	// Create vertices
	const size_t vertexSize = sizeof(CGeometryCreator::CubeVertex);
	asset::ICPUBuffer* vertices = new asset::ICPUBuffer(24*vertexSize);
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
		core::vector3df(0, 0, 0),
		core::vector3df(1, 0, 0),
		core::vector3df(1, 1, 0),
		core::vector3df(0, 1, 0),
		core::vector3df(1, 0, -1),
		core::vector3df(0, 1, -1),
		core::vector3df(0, 0, -1),
		core::vector3df(1, 1, -1)
	};
	const core::vector2d<uint8_t> uvs[4] =
	{
		core::vector2d<uint8_t>(0, 1),
		core::vector2d<uint8_t>(1, 1),
		core::vector2d<uint8_t>(1, 0),
		core::vector2d<uint8_t>(0, 0)
	};

	for (size_t f = 0; f < 6; ++f)
	{
		const size_t v = f * 4;

		for (size_t i = 0; i < 4; ++i)
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

	// Recalculate bounding box
	buffer->setBoundingBox(core::aabbox3df(-size*0.5f,size*0.5f));

	for (uint32_t i = 0; i < 24; ++i)
	{
		ptr[i].translate(-0.5f, -0.5f, 0.5f);
		core::vector3df& pos = *((core::vector3df*)(ptr[i].pos));
		pos *= size;
	}
    //setVertexAttrBuffer(asset::ICPUBuffer* attrBuf, const E_VERTEX_ATTRIBUTE_ID& attrId, E_COMPONENTS_PER_ATTRIBUTE components, E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0)
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,vertexSize, offsetof(CubeVertex, pos));
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR1,asset::EF_R8G8B8A8_UNORM,vertexSize,offsetof(CubeVertex, color));
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR2,asset::EF_R8G8_UNORM,vertexSize,offsetof(CubeVertex, uv));
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR3,asset::EF_R8G8B8_SNORM,vertexSize,offsetof(CubeVertex, normal));
    vertices->drop();

	asset::SCPUMesh* mesh = new asset::SCPUMesh;
	mesh->addMeshBuffer(buffer);
	buffer->drop();

	mesh->recalculateBoundingBox();
	return mesh;
}


/*
	a cylinder, a cone and a cross
	point up on (0,1.f, 0.f )
*/
asset::ICPUMesh* CGeometryCreator::createArrowMesh(const uint32_t tesselationCylinder,
						const uint32_t tesselationCone,
						const float height,
						const float cylinderHeight,
						const float width0,
						const float width1,
						const video::SColor vtxColor0,
						const video::SColor vtxColor1) const
{
    assert(height > cylinderHeight);

    asset::ICPUMesh* cylinder = createCylinderMesh(width0, cylinderHeight, tesselationCylinder, vtxColor0, false);
    asset::SCPUMesh* cone = static_cast<asset::SCPUMesh*>(createConeMesh(width1, height-cylinderHeight, tesselationCone, vtxColor1, vtxColor1));

    if (!cylinder || !cone)
        return nullptr;

    asset::ICPUMeshBuffer* coneMb = cone->getMeshBuffer(0u);

    asset::ICPUBuffer* coneVtxBuf = const_cast<asset::ICPUBuffer*>(coneMb->getMeshDataAndFormat()->getMappedBuffer(asset::EVAI_ATTR0));
    ConeVertex* coneVertices = reinterpret_cast<ConeVertex*>(coneVtxBuf->getPointer());
    for (uint32_t i = 0u; i < tesselationCone+2u; ++i)
        coneVertices[i].pos[1] += cylinderHeight;
    coneMb->recalculateBoundingBox();

    cone->addMeshBuffer(cylinder->getMeshBuffer(0u));
    cone->recalculateBoundingBox();

    cylinder->drop();

    return cone;
}


/* A sphere with proper normals and texture coords */
asset::ICPUMesh* CGeometryCreator::createSphereMesh(float radius, uint32_t polyCountX, uint32_t polyCountY) const
{
	// thanks to Alfaz93 who made his code available for Irrlicht on which
	// this one is based!

	// we are creating the sphere mesh here.

	if (polyCountX < 2)
		polyCountX = 2;
	if (polyCountY < 2)
		polyCountY = 2;

	const uint32_t polyCountXPitch = polyCountX+1; // get to same vertex on next level

    asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();
	asset::ICPUMeshBuffer* buffer = new asset::ICPUMeshBuffer();
	buffer->setMeshDataAndFormat(desc);
	desc->drop();

    size_t indexCount = (polyCountX * polyCountY) * 6;
    asset::ICPUBuffer* indices = new asset::ICPUBuffer(indexCount * 4);
    desc->setIndexBuffer(indices);
    buffer->setIndexType(asset::EIT_32BIT);
    buffer->setIndexCount(indexCount);
    //buffer->setIndexRange(0,11);
    indices->drop();

	uint32_t level = 0;
	size_t indexAddIx = 0;
    uint32_t* indexPtr = (uint32_t*)indices->getPointer();
	for (uint32_t p1 = 0; p1 < polyCountY-1; ++p1)
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


	size_t vertexSize = 3*4+4+2*4+4;
	size_t vertexCount = (polyCountXPitch * polyCountY) + 2;
    asset::ICPUBuffer* vertices = new asset::ICPUBuffer(vertexCount*vertexSize);
	uint8_t* tmpMem = (uint8_t*)vertices->getPointer();
	for (size_t i=0; i<vertexCount; i++)
    {
        tmpMem[i*vertexSize+3*4+0] = 255;
        tmpMem[i*vertexSize+3*4+1] = 255;
        tmpMem[i*vertexSize+3*4+2] = 255;
        tmpMem[i*vertexSize+3*4+3] = 255;
    }
	// calculate the angle which separates all points in a circle
	const double AngleX = 2 * core::PI / polyCountX;
	const double AngleY = core::PI / polyCountY;

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
		for (uint32_t xz = 0;xz < polyCountX; ++xz)
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
					tu = static_cast<float>(acos(core::clamp(normal.X/sinay, -1.0, 1.0)) * 0.5 *core::RECIPROCAL_PI64);
				if (normal.Z < 0.0f)
					tu=1-tu;
			//}
			//else
				//tu = ((float*)(tmpMem+(i-polyCountXPitch)*vertexSize))[4];

            ((float*)tmpMemPtr)[0] = pos.X;
            ((float*)tmpMemPtr)[1] = pos.Y;
            ((float*)tmpMemPtr)[2] = pos.Z;
            ((float*)tmpMemPtr)[4] = tu;
            ((float*)tmpMemPtr)[5] = static_cast<float>(ay*core::RECIPROCAL_PI64);
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
    ((uint32_t*)tmpMemPtr)[6] = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f,1.f,0.f));

	// the vertex at the bottom of the sphere
    tmpMemPtr += vertexSize;
    ((float*)tmpMemPtr)[0] = 0.f;
    ((float*)tmpMemPtr)[1] = -radius;
    ((float*)tmpMemPtr)[2] = 0.f;
    ((float*)tmpMemPtr)[4] = 0.5f;
    ((float*)tmpMemPtr)[5] = 1.f;
    ((uint32_t*)tmpMemPtr)[6] = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f,-1.f,0.f));

    //setVertexAttrBuffer(asset::ICPUBuffer* attrBuf, const E_VERTEX_ATTRIBUTE_ID& attrId, E_COMPONENTS_PER_ATTRIBUTE components, E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0)
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,vertexSize);
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR1,asset::EF_R8G8B8A8_UNORM,vertexSize,4*3);
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR2,asset::EF_R32G32_SFLOAT,vertexSize,4*3+4);
    desc->setVertexAttrBuffer(vertices,asset::EVAI_ATTR3,asset::EF_A2B10G10R10_SNORM_PACK32,vertexSize,4*3+4+2*4);
    vertices->drop();

	// recalculate bounding box
	core::aabbox3df BoundingBox;
	BoundingBox.reset(core::vector3df(radius));
	BoundingBox.addInternalPoint(-radius,-radius,-radius);
	buffer->setBoundingBox(BoundingBox);

	asset::SCPUMesh* mesh = new asset::SCPUMesh;
	mesh->addMeshBuffer(buffer);
	buffer->drop();

	mesh->recalculateBoundingBox();
	return mesh;
}


/* A cylinder with proper normals and texture coords */
asset::ICPUMesh* CGeometryCreator::createCylinderMesh(float radius, float length,
			uint32_t tesselation, const video::SColor& color,
			bool closeTop, float oblique) const
{
    const size_t vtxCnt = 2u*tesselation + 2u;
    asset::ICPUBuffer* vtxBuf = new asset::ICPUBuffer(vtxCnt * sizeof(CylinderVertex));
    CylinderVertex* vertices = reinterpret_cast<CylinderVertex*>(vtxBuf->getPointer());
    std::fill(vertices, vertices + vtxCnt, CylinderVertex());

    const uint32_t bottomCenterIx = 0u;
    const uint32_t topCenterIx = vtxCnt/2u;

    uint8_t glcolor[4];
    color.toOpenGLColor(glcolor);

    const float tesselationRec = core::reciprocal((float)tesselation);
    const float step = (2.f*core::PI)/tesselation;
    for (uint32_t i = 1u; i < vtxCnt/2u; ++i)
    {
        core::vectorSIMDf p(std::cos(i*step), 0.f, std::sin(i*step), 0.f);
        p *= radius;
        const uint32_t n = quantizeNormal2_10_10_10(core::normalize(p));

        memcpy(vertices[i].pos, p.pointer, 12u);
        vertices[i].normal = n;
        memcpy(vertices[i].color, glcolor, 4u);
        vertices[i].uv[0] = (i-1u) * tesselationRec;

        p += core::vectorSIMDf(oblique, length, 0.f, 0.f);
        memcpy(vertices[i+ topCenterIx].pos, p.pointer, 12u);
        vertices[i + topCenterIx].normal = n;
        memcpy(vertices[i+ topCenterIx].color, glcolor, 4u);
        vertices[i + topCenterIx].uv[0] = (i-1u) * tesselationRec;
        vertices[i + topCenterIx].uv[1] = 1.f;
    }
    memset(vertices[bottomCenterIx].pos, 0, 12u);
    vertices[bottomCenterIx].normal = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f, -1.f, 0.f, 0.f));
    memcpy(vertices[bottomCenterIx].color, glcolor, 4u);
    core::vectorSIMDf p(oblique, length, 0.f, 0.f);
    memcpy(vertices[topCenterIx].pos, p.pointer, 12u);
    vertices[topCenterIx].normal = quantizeNormal2_10_10_10(core::vectorSIMDf(0.f, 1.f, 0.f, 0.f));
    memcpy(vertices[topCenterIx].color, glcolor, 4u);

    const uint32_t parts = closeTop ? 4u : 3u;
    asset::ICPUBuffer* idxBuf = new asset::ICPUBuffer(parts*3u*tesselation*sizeof(uint16_t));
    uint16_t* indices = (uint16_t*)idxBuf->getPointer();

    for (uint32_t i = 1u, j = 0u; i < vtxCnt/2u; ++i)
    {
        indices[j++] = bottomCenterIx;
        indices[j++] = i;
        indices[j++] = i+1u == vtxCnt/2u ? 1u : i + 1u;
    }

    if (closeTop)
    {
        indices += idxBuf->getSize()/parts/sizeof(uint16_t);

        for (uint32_t i = 1u, j = 0u; i < vtxCnt/2u; ++i)
        {
            indices[j++] = topCenterIx;
            indices[j++] = (i+1u == vtxCnt/2u ? 1u : i + 1u) + topCenterIx;
            indices[j++] = i + topCenterIx;
        }
    }

    indices += idxBuf->getSize()/parts/sizeof(uint16_t);

    for (uint32_t i = 1u, j = 0u; i < vtxCnt/2u; ++i)
    {
        indices[j++] = (i+1u == vtxCnt/2u ? 1u : i+1u) + topCenterIx;
        indices[j++] = (i+1u == vtxCnt/2u ? 1u : i+1u);
        indices[j++] = i;
        indices[j++] = i;
        indices[j++] = i + topCenterIx;
        indices[j++] = (i+1u == vtxCnt/2u ? 1u : i+1u) + topCenterIx;
    }

    asset::SCPUMesh* mesh = new asset::SCPUMesh();
    asset::ICPUMeshBuffer* meshbuf = new asset::ICPUMeshBuffer();
    asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR0, asset::EF_R32G32B32_SFLOAT, sizeof(CylinderVertex), offsetof(CylinderVertex, pos));
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR1, asset::EF_R8G8B8A8_UNORM, sizeof(CylinderVertex), offsetof(CylinderVertex, color));
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR2, asset::EF_R32G32_SFLOAT, sizeof(CylinderVertex), offsetof(CylinderVertex, uv));
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR3, asset::EF_A2B10G10R10_UNORM_PACK32, sizeof(CylinderVertex), offsetof(CylinderVertex, normal));
    vtxBuf->drop();
    desc->setIndexBuffer(idxBuf);
    meshbuf->setIndexCount(idxBuf->getSize()/2u);
    meshbuf->setIndexType(asset::EIT_16BIT);
    meshbuf->setPrimitiveType(asset::EPT_TRIANGLES);
    idxBuf->drop();
    meshbuf->setMeshDataAndFormat(desc);
    desc->drop();
    mesh->addMeshBuffer(meshbuf);
    meshbuf->drop();

    mesh->recalculateBoundingBox(true);

    return mesh;
}

/* A cone with proper normals and texture coords */
asset::ICPUMesh* CGeometryCreator::createConeMesh(float radius, float length, uint32_t tesselation,
					const video::SColor& colorTop,
					const video::SColor& colorBottom,
					float oblique) const
{
    const size_t vtxCnt = tesselation+2u;
    asset::ICPUBuffer* vtxBuf = new asset::ICPUBuffer(vtxCnt * sizeof(ConeVertex));
    ConeVertex* vertices = reinterpret_cast<ConeVertex*>(vtxBuf->getPointer());
    std::fill(vertices, vertices + vtxCnt, ConeVertex(core::vectorSIMDf(0.f), 0u, colorBottom));

    const float step = (2.f*core::PI) / tesselation;
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

    asset::ICPUBuffer* idxBuf = new asset::ICPUBuffer(2u*3u*tesselation*sizeof(uint16_t));
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

    asset::SCPUMesh* mesh = new asset::SCPUMesh();
    asset::ICPUMeshBuffer* meshbuf = new asset::ICPUMeshBuffer();
    asset::ICPUMeshDataFormatDesc* desc = new asset::ICPUMeshDataFormatDesc();
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR0, asset::EF_R32G32B32_SFLOAT, sizeof(ConeVertex), offsetof(ConeVertex, pos));
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR1, asset::EF_R8G8B8A8_UNORM, sizeof(ConeVertex), offsetof(ConeVertex, color));
    desc->setVertexAttrBuffer(vtxBuf, asset::EVAI_ATTR3, asset::EF_A2B10G10R10_UNORM_PACK32, sizeof(ConeVertex), offsetof(ConeVertex, normal));
    vtxBuf->drop();
    desc->setIndexBuffer(idxBuf);
    meshbuf->setIndexCount(idxBuf->getSize()/2u);
    meshbuf->setIndexType(asset::EIT_16BIT);
    meshbuf->setPrimitiveType(asset::EPT_TRIANGLES);
    idxBuf->drop();
    meshbuf->setMeshDataAndFormat(desc);
    desc->drop();
    mesh->addMeshBuffer(meshbuf);
    meshbuf->drop();

    mesh->recalculateBoundingBox(true);

    return mesh;
}


} // end namespace asset
} // end namespace irr


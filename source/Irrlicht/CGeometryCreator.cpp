// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CGeometryCreator.h"
#include "SAnimatedMesh.h"
#include "SMesh.h"
#include "IMesh.h"
#include "IVideoDriver.h"
#include "SVertexManipulator.h"
#include "os.h"

namespace irr
{
namespace scene
{

ICPUMesh* CGeometryCreator::createCubeMeshCPU(const core::vector3df& size) const
{
	ICPUMeshDataFormatDesc* desc = new ICPUMeshDataFormatDesc();
	ICPUMeshBuffer* buffer = new ICPUMeshBuffer();
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

    core::ICPUBuffer* indices = new core::ICPUBuffer(sizeof(u));
    memcpy(indices->getPointer(),u,sizeof(u));
    desc->mapIndexBuffer(indices);
    buffer->setIndexType(video::EIT_16BIT);
    buffer->setIndexCount(sizeof(u)/sizeof(*u));
    indices->drop();

	// Create vertices
	const size_t vertexSize = sizeof(CGeometryCreator::CubeVertex);
	core::ICPUBuffer* vertices = new core::ICPUBuffer(24*vertexSize);
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
    //mapVertexAttrBuffer(core::ICPUBuffer* attrBuf, const E_VERTEX_ATTRIBUTE_ID& attrId, E_COMPONENTS_PER_ATTRIBUTE components, E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0)
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,vertexSize, offsetof(CubeVertex, pos));
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR1,scene::ECPA_FOUR,scene::ECT_NORMALIZED_UNSIGNED_BYTE,vertexSize,offsetof(CubeVertex, color));
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR2,scene::ECPA_TWO,scene::ECT_UNSIGNED_BYTE,vertexSize,offsetof(CubeVertex, uv));
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR3,scene::ECPA_THREE,scene::ECT_BYTE,vertexSize,offsetof(CubeVertex, normal));
    vertices->drop();

	SCPUMesh* mesh = new SCPUMesh;
	mesh->addMeshBuffer(buffer);
	buffer->drop();

	mesh->recalculateBoundingBox();
	return mesh;
}

IGPUMesh* CGeometryCreator::createCubeMeshGPU(video::IVideoDriver* driver, const core::vector3df& size) const
{
    if (!driver)
        return NULL;

	ICPUMesh* cpumesh = createCubeMeshCPU(size);

	auto retval = driver->createGPUMeshesFromCPU(std::vector<ICPUMesh*>(1,cpumesh));
	IGPUMesh* mesh = nullptr;
	if (retval.size())
        mesh = retval[0];

	cpumesh->drop();

	return mesh;
}


/*
	a cylinder, a cone and a cross
	point up on (0,1.f, 0.f )
*/
ICPUMesh* CGeometryCreator::createArrowMeshCPU(const uint32_t tesselationCylinder,
						const uint32_t tesselationCone,
						const float height,
						const float cylinderHeight,
						const float width0,
						const float width1,
						const video::SColor vtxColor0,
						const video::SColor vtxColor1) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	SMesh* mesh = (SMesh*)createCylinderMesh(width0, cylinderHeight, tesselationCylinder, vtxColor0, false);

	IMesh* mesh2 = createConeMesh(width1, height-cylinderHeight, tesselationCone, vtxColor1, vtxColor0);
	for (uint32_t i=0; i<mesh2->getMeshBufferCount(); ++i)
	{
		scene::IMeshBuffer* buffer = mesh2->getMeshBuffer(i);
		for (uint32_t j=0; j<buffer->getVertexCount(); ++j)
			buffer->getPosition(j).Y += cylinderHeight;
		buffer->setDirty(EBT_VERTEX);
		buffer->recalculateBoundingBox();
		mesh->addMeshBuffer(buffer);
	}
	mesh2->drop();

	mesh->recalculateBoundingBox();
	return mesh;
#endif // NEW_MESHES
}

IGPUMesh* CGeometryCreator::createArrowMeshGPU(video::IVideoDriver* driver,
                        const uint32_t tesselationCylinder,
						const uint32_t tesselationCone,
						const float height,
						const float cylinderHeight,
						const float width0,
						const float width1,
						const video::SColor vtxColor0,
						const video::SColor vtxColor1) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createArrowMeshCPU(tesselationCylinder,tesselationCone,height,cylinderHeight,width0,width1,vtxColor0,vtxColor1));

	auto retval = driver->createGPUMeshesFromCPU(std::vector<ICPUMesh*>(1,cpumesh));
	IGPUMesh* mesh = nullptr;
	if (retval.size())
        mesh = retval[0];

	cpumesh->drop();

	return mesh;
}


/* A sphere with proper normals and texture coords */
ICPUMesh* CGeometryCreator::createSphereMeshCPU(float radius, uint32_t polyCountX, uint32_t polyCountY) const
{
	// thanks to Alfaz93 who made his code available for Irrlicht on which
	// this one is based!

	// we are creating the sphere mesh here.

	if (polyCountX < 2)
		polyCountX = 2;
	if (polyCountY < 2)
		polyCountY = 2;

	const uint32_t polyCountXPitch = polyCountX+1; // get to same vertex on next level

	ICPUMeshDataFormatDesc* desc = new ICPUMeshDataFormatDesc();
	ICPUMeshBuffer* buffer = new ICPUMeshBuffer();
	buffer->setMeshDataAndFormat(desc);
	desc->drop();

    size_t indexCount = (polyCountX * polyCountY) * 6;
    core::ICPUBuffer* indices = new core::ICPUBuffer(indexCount * 4);
    desc->mapIndexBuffer(indices);
    buffer->setIndexType(video::EIT_32BIT);
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
    core::ICPUBuffer* vertices = new core::ICPUBuffer(vertexCount*vertexSize);
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

    //mapVertexAttrBuffer(core::ICPUBuffer* attrBuf, const E_VERTEX_ATTRIBUTE_ID& attrId, E_COMPONENTS_PER_ATTRIBUTE components, E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0)
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,vertexSize);
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR1,scene::ECPA_FOUR,scene::ECT_NORMALIZED_UNSIGNED_BYTE,vertexSize,4*3);
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR2,scene::ECPA_TWO,scene::ECT_FLOAT,vertexSize,4*3+4);
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR3,scene::ECPA_FOUR,scene::ECT_INT_2_10_10_10_REV,vertexSize,4*3+4+2*4);
    vertices->drop();

	// recalculate bounding box
	core::aabbox3df BoundingBox;
	BoundingBox.reset(core::vector3df(radius));
	BoundingBox.addInternalPoint(-radius,-radius,-radius);
	buffer->setBoundingBox(BoundingBox);

	SCPUMesh* mesh = new SCPUMesh;
	mesh->addMeshBuffer(buffer);
	buffer->drop();

	mesh->recalculateBoundingBox();
	return mesh;
}

IGPUMesh* CGeometryCreator::createSphereMeshGPU(video::IVideoDriver* driver, float radius, uint32_t polyCountX, uint32_t polyCountY) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createSphereMeshCPU(radius,polyCountX,polyCountY));

	auto retval = driver->createGPUMeshesFromCPU(std::vector<ICPUMesh*>(1,cpumesh));
	IGPUMesh* mesh = nullptr;
	if (retval.size())
        mesh = retval[0];

	cpumesh->drop();

	return mesh;
}


/* A cylinder with proper normals and texture coords */
ICPUMesh* CGeometryCreator::createCylinderMeshCPU(float radius, float length,
			uint32_t tesselation, const video::SColor& color,
			bool closeTop, float oblique) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	SMeshBuffer* buffer = new SMeshBuffer();

	const float recTesselation = core::reciprocal((float)tesselation);
	const float recTesselationHalf = recTesselation * 0.5f;
	const float angleStep = (core::PI * 2.f ) * recTesselation;
	const float angleStepHalf = angleStep*0.5f;

	uint32_t i;
	video::S3DVertex v;
	v.Color = color;
	buffer->Vertices.reallocate(tesselation*4+4+(closeTop?2:1));
	buffer->Indices.reallocate((tesselation*2+1)*(closeTop?12:9));
	float tcx = 0.f;
	for ( i = 0; i <= tesselation; ++i )
	{
		const float angle = angleStep * i;
		v.Pos.X = radius * cosf(angle);
		v.Pos.Y = 0.f;
		v.Pos.Z = radius * sinf(angle);
		v.Normal = v.Pos;
		v.Normal.normalize();
		v.TCoords.X=tcx;
		v.TCoords.Y=0.f;
		buffer->Vertices.push_back(v);

		v.Pos.X += oblique;
		v.Pos.Y = length;
		v.Normal = v.Pos;
		v.Normal.normalize();
		v.TCoords.Y=1.f;
		buffer->Vertices.push_back(v);

		v.Pos.X = radius * cosf(angle + angleStepHalf);
		v.Pos.Y = 0.f;
		v.Pos.Z = radius * sinf(angle + angleStepHalf);
		v.Normal = v.Pos;
		v.Normal.normalize();
		v.TCoords.X=tcx+recTesselationHalf;
		v.TCoords.Y=0.f;
		buffer->Vertices.push_back(v);

		v.Pos.X += oblique;
		v.Pos.Y = length;
		v.Normal = v.Pos;
		v.Normal.normalize();
		v.TCoords.Y=1.f;
		buffer->Vertices.push_back(v);
		tcx += recTesselation;
	}

	// indices for the main hull part
	const uint32_t nonWrappedSize = tesselation* 4;
	for (i=0; i != nonWrappedSize; i += 2)
	{
		indexPtr[indexAddIx++] = i + 2);
		indexPtr[indexAddIx++] = i + 0);
		indexPtr[indexAddIx++] = i + 1);

		indexPtr[indexAddIx++] = i + 2);
		indexPtr[indexAddIx++] = i + 1);
		indexPtr[indexAddIx++] = i + 3);
	}

	// two closing quads between end and start
	indexPtr[indexAddIx++] = 0);
	indexPtr[indexAddIx++] = i + 0);
	indexPtr[indexAddIx++] = i + 1);

	indexPtr[indexAddIx++] = 0);
	indexPtr[indexAddIx++] = i + 1);
	indexPtr[indexAddIx++] = 1);

	// close down
	v.Pos.X = 0.f;
	v.Pos.Y = 0.f;
	v.Pos.Z = 0.f;
	v.Normal.X = 0.f;
	v.Normal.Y = -1.f;
	v.Normal.Z = 0.f;
	v.TCoords.X = 1.f;
	v.TCoords.Y = 1.f;
	buffer->Vertices.push_back(v);

	uint32_t index = buffer->Vertices.size() - 1;

	for ( i = 0; i != nonWrappedSize; i += 2 )
	{
		indexPtr[indexAddIx++] = index);
		indexPtr[indexAddIx++] = i + 0);
		indexPtr[indexAddIx++] = i + 2);
	}

	indexPtr[indexAddIx++] = index);
	indexPtr[indexAddIx++] = i + 0);
	indexPtr[indexAddIx++] = 0);

	if (closeTop)
	{
		// close top
		v.Pos.X = oblique;
		v.Pos.Y = length;
		v.Pos.Z = 0.f;
		v.Normal.X = 0.f;
		v.Normal.Y = 1.f;
		v.Normal.Z = 0.f;
		v.TCoords.X = 0.f;
		v.TCoords.Y = 0.f;
		buffer->Vertices.push_back(v);

		index = buffer->Vertices.size() - 1;

		for ( i = 0; i != nonWrappedSize; i += 2 )
		{
			indexPtr[indexAddIx++] = i + 1);
			indexPtr[indexAddIx++] = index);
			indexPtr[indexAddIx++] = i + 3);
		}

		indexPtr[indexAddIx++] = i + 1);
		indexPtr[indexAddIx++] = index);
		indexPtr[indexAddIx++] = 1);
	}

	buffer->recalculateBoundingBox();
	SMesh* mesh = new SMesh();
	mesh->addMeshBuffer(buffer);
	mesh->setHardwareMappingHint(EHM_STATIC);
	mesh->recalculateBoundingBox();
	buffer->drop();
	return mesh;
#endif // NEW_MESHES
}

IGPUMesh* CGeometryCreator::createCylinderMeshGPU(video::IVideoDriver* driver,
            float radius, float length,
			uint32_t tesselation, const video::SColor& color,
			bool closeTop, float oblique) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createCylinderMeshCPU(radius,length,tesselation,color,closeTop,oblique));

	auto retval = driver->createGPUMeshesFromCPU(std::vector<ICPUMesh*>(1,cpumesh));
	IGPUMesh* mesh = nullptr;
	if (retval.size())
        mesh = retval[0];

	cpumesh->drop();

	return mesh;
}



/* A cone with proper normals and texture coords */
ICPUMesh* CGeometryCreator::createConeMeshCPU(float radius, float length, uint32_t tesselation,
					const video::SColor& colorTop,
					const video::SColor& colorBottom,
					float oblique) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	SMeshBuffer* buffer = new SMeshBuffer();

	const float angleStep = (core::PI * 2.f ) / tesselation;
	const float angleStepHalf = angleStep*0.5f;

	video::S3DVertex v;
	uint32_t i;

	v.Color = colorTop;
	for ( i = 0; i != tesselation; ++i )
	{
		float angle = angleStep * float(i);

		v.Pos.X = radius * cosf(angle);
		v.Pos.Y = 0.f;
		v.Pos.Z = radius * sinf(angle);
		v.Normal = v.Pos;
		v.Normal.normalize();
		buffer->Vertices.push_back(v);

		angle += angleStepHalf;
		v.Pos.X = radius * cosf(angle);
		v.Pos.Y = 0.f;
		v.Pos.Z = radius * sinf(angle);
		v.Normal = v.Pos;
		v.Normal.normalize();
		buffer->Vertices.push_back(v);
	}
	const uint32_t nonWrappedSize = buffer->Vertices.size() - 1;

	// close top
	v.Pos.X = oblique;
	v.Pos.Y = length;
	v.Pos.Z = 0.f;
	v.Normal.X = 0.f;
	v.Normal.Y = 1.f;
	v.Normal.Z = 0.f;
	buffer->Vertices.push_back(v);

	uint32_t index = buffer->Vertices.size() - 1;

	for ( i = 0; i != nonWrappedSize; i += 1 )
	{
		indexPtr[indexAddIx++] = i + 0);
		indexPtr[indexAddIx++] = index);
		indexPtr[indexAddIx++] = i + 1);
	}

	indexPtr[indexAddIx++] = i + 0);
	indexPtr[indexAddIx++] = index);
	indexPtr[indexAddIx++] = 0);

	// close down
	v.Color = colorBottom;
	v.Pos.X = 0.f;
	v.Pos.Y = 0.f;
	v.Pos.Z = 0.f;
	v.Normal.X = 0.f;
	v.Normal.Y = -1.f;
	v.Normal.Z = 0.f;
	buffer->Vertices.push_back(v);

	index = buffer->Vertices.size() - 1;

	for ( i = 0; i != nonWrappedSize; i += 1 )
	{
		indexPtr[indexAddIx++] = index);
		indexPtr[indexAddIx++] = i + 0);
		indexPtr[indexAddIx++] = i + 1);
	}

	indexPtr[indexAddIx++] = index);
	indexPtr[indexAddIx++] = i + 0);
	indexPtr[indexAddIx++] = 0);

	buffer->recalculateBoundingBox();
	SMesh* mesh = new SMesh();
	mesh->addMeshBuffer(buffer);
	buffer->drop();

	mesh->recalculateBoundingBox();
	return mesh;
#endif // NEW_MESHES
}

IGPUMesh* CGeometryCreator::createConeMeshGPU(video::IVideoDriver* driver,
                    float radius, float length, uint32_t tesselation,
					const video::SColor& colorTop,
					const video::SColor& colorBottom,
					float oblique) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createConeMeshCPU(radius,length,tesselation,colorTop,colorBottom,oblique));

	auto retval = driver->createGPUMeshesFromCPU(std::vector<ICPUMesh*>(1,cpumesh));
	IGPUMesh* mesh = nullptr;
	if (retval.size())
        mesh = retval[0];

	cpumesh->drop();

	return mesh;
}


} // end namespace scene
} // end namespace irr


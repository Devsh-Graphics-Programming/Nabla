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

    //! This is a baaaad mesh, you dont share vertices in a cube across faces
	// Create indices
	const u16 u[36] = {   0,2,1,   0,3,2,   1,5,4,   1,2,5,   4,6,7,   4,5,6,
            7,3,0,   7,6,3,   9,5,2,   9,8,5,   0,11,10,   0,10,7};

    core::ICPUBuffer* indices = new core::ICPUBuffer(sizeof(u));
    memcpy(indices->getPointer(),u,sizeof(u));
    desc->mapIndexBuffer(indices);
    buffer->setIndexType(video::EIT_16BIT);
    buffer->setIndexCount(36);
    //buffer->setIndexRange(0,11);
    indices->drop();


	// Create vertices
	size_t vertexSize = 3*4+4+2+3;
    core::ICPUBuffer* vertices = new core::ICPUBuffer(12*vertexSize);
	uint8_t* tmpMem = (uint8_t*)vertices->getPointer();
	for (size_t i=0; i<12; i++)
    {
        tmpMem[i*vertexSize+3*4+0] = 255;
        tmpMem[i*vertexSize+3*4+1] = 255;
        tmpMem[i*vertexSize+3*4+2] = 255;
        tmpMem[i*vertexSize+3*4+3] = 255;
    }
    tmpMem[0*vertexSize+3*4+4+0] = 0;
    tmpMem[0*vertexSize+3*4+4+1] = 1;
    tmpMem[1*vertexSize+3*4+4+0] = 1;
    tmpMem[1*vertexSize+3*4+4+1] = 1;
    tmpMem[2*vertexSize+3*4+4+0] = 1;
    tmpMem[2*vertexSize+3*4+4+1] = 0;
    tmpMem[3*vertexSize+3*4+4+0] = 0;
    tmpMem[3*vertexSize+3*4+4+1] = 0;
    tmpMem[4*vertexSize+3*4+4+0] = 0;
    tmpMem[4*vertexSize+3*4+4+1] = 1;
    tmpMem[5*vertexSize+3*4+4+0] = 0;
    tmpMem[5*vertexSize+3*4+4+1] = 0;
    tmpMem[6*vertexSize+3*4+4+0] = 1;
    tmpMem[6*vertexSize+3*4+4+1] = 0;
    tmpMem[7*vertexSize+3*4+4+0] = 1;
    tmpMem[7*vertexSize+3*4+4+1] = 1;
    tmpMem[8*vertexSize+3*4+4+0] = 0;
    tmpMem[8*vertexSize+3*4+4+1] = 1;
    tmpMem[9*vertexSize+3*4+4+0] = 1;
    tmpMem[9*vertexSize+3*4+4+1] = 1;
    tmpMem[10*vertexSize+3*4+4+0] = 1;
    tmpMem[10*vertexSize+3*4+4+1] = 0;
    tmpMem[11*vertexSize+3*4+4+0] = 0;
    tmpMem[11*vertexSize+3*4+4+1] = 0;
    ((float*)(tmpMem+0*vertexSize))[0] = 0;
    ((float*)(tmpMem+0*vertexSize))[1] = 0;
    ((float*)(tmpMem+0*vertexSize))[2] = 0;
    ((float*)(tmpMem+1*vertexSize))[0] = 1;
    ((float*)(tmpMem+1*vertexSize))[1] = 0;
    ((float*)(tmpMem+1*vertexSize))[2] = 0;
    ((float*)(tmpMem+2*vertexSize))[0] = 1;
    ((float*)(tmpMem+2*vertexSize))[1] = 1;
    ((float*)(tmpMem+2*vertexSize))[2] = 0;
    ((float*)(tmpMem+3*vertexSize))[0] = 0;
    ((float*)(tmpMem+3*vertexSize))[1] = 1;
    ((float*)(tmpMem+3*vertexSize))[2] = 0;
    ((float*)(tmpMem+4*vertexSize))[0] = 1;
    ((float*)(tmpMem+4*vertexSize))[1] = 0;
    ((float*)(tmpMem+4*vertexSize))[2] = 1;
    ((float*)(tmpMem+5*vertexSize))[0] = 1;
    ((float*)(tmpMem+5*vertexSize))[1] = 1;
    ((float*)(tmpMem+5*vertexSize))[2] = 1;
    ((float*)(tmpMem+6*vertexSize))[0] = 0;
    ((float*)(tmpMem+6*vertexSize))[1] = 1;
    ((float*)(tmpMem+6*vertexSize))[2] = 1;
    ((float*)(tmpMem+7*vertexSize))[0] = 0;
    ((float*)(tmpMem+7*vertexSize))[1] = 0;
    ((float*)(tmpMem+7*vertexSize))[2] = 1;
    ((float*)(tmpMem+8*vertexSize))[0] = 0;
    ((float*)(tmpMem+8*vertexSize))[1] = 1;
    ((float*)(tmpMem+8*vertexSize))[2] = 1;
    ((float*)(tmpMem+9*vertexSize))[0] = 0;
    ((float*)(tmpMem+9*vertexSize))[1] = 1;
    ((float*)(tmpMem+9*vertexSize))[2] = 0;
    ((float*)(tmpMem+10*vertexSize))[0] = 1;
    ((float*)(tmpMem+10*vertexSize))[1] = 0;
    ((float*)(tmpMem+10*vertexSize))[2] = 1;
    ((float*)(tmpMem+11*vertexSize))[0] = 1;
    ((float*)(tmpMem+11*vertexSize))[1] = 0;
    ((float*)(tmpMem+11*vertexSize))[2] = 0;
    tmpMem[0*vertexSize+3*4+4+2+0] = 0;
    tmpMem[0*vertexSize+3*4+4+2+1] = 0;
    tmpMem[0*vertexSize+3*4+4+2+2] = -1;
    tmpMem[1*vertexSize+3*4+4+2+0] = 0;
    tmpMem[1*vertexSize+3*4+4+2+1] = 0;
    tmpMem[1*vertexSize+3*4+4+2+2] = -1;
    tmpMem[2*vertexSize+3*4+4+2+0] = 0;
    tmpMem[2*vertexSize+3*4+4+2+1] = 0;
    tmpMem[2*vertexSize+3*4+4+2+2] = -1;
    tmpMem[3*vertexSize+3*4+4+2+0] = 0;
    tmpMem[3*vertexSize+3*4+4+2+1] = 0;
    tmpMem[3*vertexSize+3*4+4+2+2] = -1;
    tmpMem[4*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[4*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[4*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[5*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[5*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[5*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[6*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[6*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[6*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[7*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[7*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[7*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[8*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[8*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[8*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[9*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[9*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[9*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[10*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[10*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[10*vertexSize+3*4+4+2+2] = 0xfu;
    tmpMem[11*vertexSize+3*4+4+2+0] = 0xfu;
    tmpMem[11*vertexSize+3*4+4+2+1] = 0xfu;
    tmpMem[11*vertexSize+3*4+4+2+2] = 0xfu;

	// Recalculate bounding box
	buffer->setBoundingBox(core::aabbox3df(-size*0.5f,size*0.5f));

	for (u32 i=0; i<12; ++i)
	{
	    core::vector3df& Pos = *((core::vector3df*)(tmpMem+i*vertexSize));
		Pos -= core::vector3df(0.5f, 0.5f, 0.5f);
		Pos *= size;
	}
    //mapVertexAttrBuffer(core::ICPUBuffer* attrBuf, const E_VERTEX_ATTRIBUTE_ID& attrId, E_COMPONENTS_PER_ATTRIBUTE components, E_COMPONENT_TYPE type, const size_t &stride=0, size_t offset=0)
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR0,scene::ECPA_THREE,scene::ECT_FLOAT,vertexSize);
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR1,scene::ECPA_FOUR,scene::ECT_NORMALIZED_UNSIGNED_BYTE,vertexSize,4*3);
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR2,scene::ECPA_TWO,scene::ECT_UNSIGNED_BYTE,vertexSize,4*3+4);
    desc->mapVertexAttrBuffer(vertices,scene::EVAI_ATTR3,scene::ECPA_THREE,scene::ECT_BYTE,vertexSize,4*3+4+2);
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

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createCubeMeshCPU(size));
	IGPUMesh* mesh = driver->createGPUMeshFromCPU(cpumesh,video::EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER);
	cpumesh->drop();

	return mesh;
}



ICPUMesh* CGeometryCreator::createTerrainMeshCPU(video::IImage* texture,
		video::IImage* heightmap, const core::dimension2d<f32>& stretchSize,
		f32 maxHeight, video::IVideoDriver* driver,
		const core::dimension2d<u32>& maxVtxBlockSize,
		bool debugBorders) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	if (!texture || !heightmap)
		return 0;

	// debug border
	const s32 borderSkip = debugBorders ? 0 : 1;

	video::S3DVertex vtx;
	vtx.Color.set(255,255,255,255);

	SMesh* mesh = new SMesh();

	const u32 tm = os::Timer::getRealTime()/1000;
	const core::dimension2d<u32> hMapSize= heightmap->getDimension();
	const core::dimension2d<u32> tMapSize= texture->getDimension();
	const core::position2d<f32> thRel(static_cast<f32>(tMapSize.Width) / hMapSize.Width, static_cast<f32>(tMapSize.Height) / hMapSize.Height);
	maxHeight /= 255.0f; // height step per color value

	core::position2d<u32> processed(0,0);
	while (processed.Y<hMapSize.Height)
	{
		while(processed.X<hMapSize.Width)
		{
			core::dimension2d<u32> blockSize = maxVtxBlockSize;
			if (processed.X + blockSize.Width > hMapSize.Width)
				blockSize.Width = hMapSize.Width - processed.X;
			if (processed.Y + blockSize.Height > hMapSize.Height)
				blockSize.Height = hMapSize.Height - processed.Y;

			SMeshBuffer* buffer = new SMeshBuffer();
			buffer->Vertices.reallocate(blockSize.getArea());
			// add vertices of vertex block
			u32 y;
			core::vector2df pos(0.f, processed.Y*stretchSize.Height);
			const core::vector2df bs(1.f/blockSize.Width, 1.f/blockSize.Height);
			core::vector2df tc(0.f, 0.5f*bs.Y);
			for (y=0; y<blockSize.Height; ++y)
			{
				pos.X=processed.X*stretchSize.Width;
				tc.X=0.5f*bs.X;
				for (u32 x=0; x<blockSize.Width; ++x)
				{
					const f32 height = heightmap->getPixel(x+processed.X, y+processed.Y).getAverage() * maxHeight;

					vtx.Pos.set(pos.X, height, pos.Y);
					vtx.TCoords.set(tc);
					buffer->Vertices.push_back(vtx);
					pos.X += stretchSize.Width;
					tc.X += bs.X;
				}
				pos.Y += stretchSize.Height;
				tc.Y += bs.Y;
			}

			buffer->Indices.reallocate((blockSize.Height-1)*(blockSize.Width-1)*6);
			// add indices of vertex block
			s32 c1 = 0;
			for (y=0; y<blockSize.Height-1; ++y)
			{
				for (u32 x=0; x<blockSize.Width-1; ++x)
				{
					const s32 c = c1 + x;

					indexPtr[indexAddIx++] = c);
					indexPtr[indexAddIx++] = c + blockSize.Width);
					indexPtr[indexAddIx++] = c + 1);

					indexPtr[indexAddIx++] = c + 1);
					indexPtr[indexAddIx++] = c + blockSize.Width);
					indexPtr[indexAddIx++] = c + 1 + blockSize.Width);
				}
				c1 += blockSize.Width;
			}

			// recalculate normals
			for (u32 i=0; i<buffer->Indices.size(); i+=3)
			{
				const core::vector3df normal = core::plane3d<f32>(
					buffer->Vertices[buffer->Indices[i+0]].Pos,
					buffer->Vertices[buffer->Indices[i+1]].Pos,
					buffer->Vertices[buffer->Indices[i+2]].Pos).Normal;

				buffer->Vertices[buffer->Indices[i+0]].Normal = normal;
				buffer->Vertices[buffer->Indices[i+1]].Normal = normal;
				buffer->Vertices[buffer->Indices[i+2]].Normal = normal;
			}

			if (buffer->Vertices.size())
			{
				c8 textureName[64];
				// create texture for this block
				video::IImage* img = driver->createImage(texture->getColorFormat(), core::dimension2d<u32>(core::floor32(blockSize.Width*thRel.X), core::floor32(blockSize.Height*thRel.Y)));
				texture->copyTo(img, core::position2di(0,0), core::recti(
					core::position2d<s32>(core::floor32(processed.X*thRel.X), core::floor32(processed.Y*thRel.Y)),
					core::dimension2d<u32>(core::floor32(blockSize.Width*thRel.X), core::floor32(blockSize.Height*thRel.Y))), 0);

				sprintf(textureName, "terrain%u_%u", tm, mesh->getMeshBufferCount());

				buffer->Material.setTexture(0, driver->addTexture(textureName, img));

				if (buffer->Material.getTexture(0))
				{
					c8 tmp[255];
					sprintf(tmp, "Generated terrain texture (%dx%d): %s",
						buffer->Material.getTexture(0)->getSize().Width,
						buffer->Material.getTexture(0)->getSize().Height,
						textureName);
					os::Printer::log(tmp);
				}
				else
					os::Printer::log("Could not create terrain texture.", textureName, ELL_ERROR);

				img->drop();
			}

			buffer->recalculateBoundingBox();
			mesh->addMeshBuffer(buffer);
			buffer->drop();

			// keep on processing
			processed.X += maxVtxBlockSize.Width - borderSkip;
		}

		// keep on processing
		processed.X = 0;
		processed.Y += maxVtxBlockSize.Height - borderSkip;
	}

	mesh->recalculateBoundingBox();
	return mesh;
#endif // NEW_MESHES
}


IGPUMesh* CGeometryCreator::createTerrainMeshGPU(video::IImage* texture,
		video::IImage* heightmap, const core::dimension2d<f32>& stretchSize,
		f32 maxHeight, video::IVideoDriver* driver,
		const core::dimension2d<u32>& maxVtxBlockSize,
		bool debugBorders) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createTerrainMeshCPU(texture,heightmap,stretchSize,maxHeight,driver,maxVtxBlockSize,debugBorders));
	IGPUMesh* mesh = driver->createGPUMeshFromCPU(cpumesh,video::EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER);
	cpumesh->drop();

	return mesh;
}


/*
	a cylinder, a cone and a cross
	point up on (0,1.f, 0.f )
*/
ICPUMesh* CGeometryCreator::createArrowMeshCPU(const u32 tesselationCylinder,
						const u32 tesselationCone,
						const f32 height,
						const f32 cylinderHeight,
						const f32 width0,
						const f32 width1,
						const video::SColor vtxColor0,
						const video::SColor vtxColor1) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	SMesh* mesh = (SMesh*)createCylinderMesh(width0, cylinderHeight, tesselationCylinder, vtxColor0, false);

	IMesh* mesh2 = createConeMesh(width1, height-cylinderHeight, tesselationCone, vtxColor1, vtxColor0);
	for (u32 i=0; i<mesh2->getMeshBufferCount(); ++i)
	{
		scene::IMeshBuffer* buffer = mesh2->getMeshBuffer(i);
		for (u32 j=0; j<buffer->getVertexCount(); ++j)
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
                        const u32 tesselationCylinder,
						const u32 tesselationCone,
						const f32 height,
						const f32 cylinderHeight,
						const f32 width0,
						const f32 width1,
						const video::SColor vtxColor0,
						const video::SColor vtxColor1) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createArrowMeshCPU(tesselationCylinder,tesselationCone,height,cylinderHeight,width0,width1,vtxColor0,vtxColor1));
	IGPUMesh* mesh = driver->createGPUMeshFromCPU(cpumesh,video::EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER);
	cpumesh->drop();

	return mesh;
}


/* A sphere with proper normals and texture coords */
ICPUMesh* CGeometryCreator::createSphereMeshCPU(f32 radius, u32 polyCountX, u32 polyCountY) const
{
	// thanks to Alfaz93 who made his code available for Irrlicht on which
	// this one is based!

	// we are creating the sphere mesh here.

	if (polyCountX < 2)
		polyCountX = 2;
	if (polyCountY < 2)
		polyCountY = 2;

	const u32 polyCountXPitch = polyCountX+1; // get to same vertex on next level

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

	u32 level = 0;
	size_t indexAddIx = 0;
    uint32_t* indexPtr = (uint32_t*)indices->getPointer();
	for (u32 p1 = 0; p1 < polyCountY-1; ++p1)
	{
		//main quads, top to bottom
		for (u32 p2 = 0; p2 < polyCountX - 1; ++p2)
		{
			const u32 curr = level + p2;
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

	const u32 polyCountSq = polyCountXPitch * polyCountY; // top point
	const u32 polyCountSq1 = polyCountSq + 1; // bottom point
	const u32 polyCountSqM1 = (polyCountY - 1) * polyCountXPitch; // last row's first vertex

	for (u32 p2 = 0; p2 < polyCountX - 1; ++p2)
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
	const f64 AngleX = 2 * core::PI / polyCountX;
	const f64 AngleY = core::PI / polyCountY;

	f64 axz;

	// we don't start at 0.

	f64 ay = 0;//AngleY / 2;

    uint8_t* tmpMemPtr = tmpMem;
	for (u32 y = 0; y < polyCountY; ++y)
	{
		ay += AngleY;
		const f64 sinay = sin(ay);
		axz = 0;

		// calculate the necessary vertices without the doubled one
		uint8_t* oldTmpMemPtr = tmpMemPtr;
		for (u32 xz = 0;xz < polyCountX; ++xz)
		{
			// calculate points position

			core::vector3df pos(static_cast<f32>(cos(axz) * sinay),
						static_cast<f32>(cos(ay)),
						static_cast<f32>(sin(axz) * sinay));
			// for spheres the normal is the position
			core::vectorSIMDf normal(&pos.X);
			normal.makeSafe3D();
			uint32_t quantizedNormal = quantizeNormal2_10_10_10(normal);
			pos *= radius;

			// calculate texture coordinates via sphere mapping
			// tu is the same on each level, so only calculate once
			f32 tu = 0.5f;
			//if (y==0)
			//{
				if (normal.Y != -1.0f && normal.Y != 1.0f)
					tu = static_cast<f32>(acos(core::clamp(normal.X/sinay, -1.0, 1.0)) * 0.5 *core::RECIPROCAL_PI64);
				if (normal.Z < 0.0f)
					tu=1-tu;
			//}
			//else
				//tu = ((float*)(tmpMem+(i-polyCountXPitch)*vertexSize))[4];

            ((float*)tmpMemPtr)[0] = pos.X;
            ((float*)tmpMemPtr)[1] = pos.Y;
            ((float*)tmpMemPtr)[2] = pos.Z;
            ((float*)tmpMemPtr)[4] = tu;
            ((float*)tmpMemPtr)[5] = static_cast<f32>(ay*core::RECIPROCAL_PI64);
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

IGPUMesh* CGeometryCreator::createSphereMeshGPU(video::IVideoDriver* driver, f32 radius, u32 polyCountX, u32 polyCountY) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createSphereMeshCPU(radius,polyCountX,polyCountY));
	IGPUMesh* mesh = driver->createGPUMeshFromCPU(cpumesh,video::EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER);
	cpumesh->drop();

	return mesh;
}


/* A cylinder with proper normals and texture coords */
ICPUMesh* CGeometryCreator::createCylinderMeshCPU(f32 radius, f32 length,
			u32 tesselation, const video::SColor& color,
			bool closeTop, f32 oblique) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	SMeshBuffer* buffer = new SMeshBuffer();

	const f32 recTesselation = core::reciprocal((f32)tesselation);
	const f32 recTesselationHalf = recTesselation * 0.5f;
	const f32 angleStep = (core::PI * 2.f ) * recTesselation;
	const f32 angleStepHalf = angleStep*0.5f;

	u32 i;
	video::S3DVertex v;
	v.Color = color;
	buffer->Vertices.reallocate(tesselation*4+4+(closeTop?2:1));
	buffer->Indices.reallocate((tesselation*2+1)*(closeTop?12:9));
	f32 tcx = 0.f;
	for ( i = 0; i <= tesselation; ++i )
	{
		const f32 angle = angleStep * i;
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
	const u32 nonWrappedSize = tesselation* 4;
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

	u32 index = buffer->Vertices.size() - 1;

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
            f32 radius, f32 length,
			u32 tesselation, const video::SColor& color,
			bool closeTop, f32 oblique) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createCylinderMeshCPU(radius,length,tesselation,color,closeTop,oblique));
	IGPUMesh* mesh = driver->createGPUMeshFromCPU(cpumesh,video::EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER);
	cpumesh->drop();

	return mesh;
}



/* A cone with proper normals and texture coords */
ICPUMesh* CGeometryCreator::createConeMeshCPU(f32 radius, f32 length, u32 tesselation,
					const video::SColor& colorTop,
					const video::SColor& colorBottom,
					f32 oblique) const
{
#ifdef NEW_MESHES
    return NULL;
#else
	SMeshBuffer* buffer = new SMeshBuffer();

	const f32 angleStep = (core::PI * 2.f ) / tesselation;
	const f32 angleStepHalf = angleStep*0.5f;

	video::S3DVertex v;
	u32 i;

	v.Color = colorTop;
	for ( i = 0; i != tesselation; ++i )
	{
		f32 angle = angleStep * f32(i);

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
	const u32 nonWrappedSize = buffer->Vertices.size() - 1;

	// close top
	v.Pos.X = oblique;
	v.Pos.Y = length;
	v.Pos.Z = 0.f;
	v.Normal.X = 0.f;
	v.Normal.Y = 1.f;
	v.Normal.Z = 0.f;
	buffer->Vertices.push_back(v);

	u32 index = buffer->Vertices.size() - 1;

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
                    f32 radius, f32 length, u32 tesselation,
					const video::SColor& colorTop,
					const video::SColor& colorBottom,
					f32 oblique) const
{
    if (!driver)
        return NULL;

	SCPUMesh* cpumesh = static_cast<SCPUMesh*>(createConeMeshCPU(radius,length,tesselation,colorTop,colorBottom,oblique));
	IGPUMesh* mesh = driver->createGPUMeshFromCPU(cpumesh,video::EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER);
	cpumesh->drop();

	return mesh;
}


} // end namespace scene
} // end namespace irr


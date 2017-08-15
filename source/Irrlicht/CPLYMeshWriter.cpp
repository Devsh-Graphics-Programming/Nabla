// Copyright (C) 2008-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_PLY_WRITER_

#include "CPLYMeshWriter.h"
#include "os.h"
#include "IMesh.h"
#include "IMeshBuffer.h"
#include "IWriteFile.h"

namespace irr
{
namespace scene
{

#ifndef NEW_MESHES
CPLYMeshWriter::CPLYMeshWriter()
{
	#ifdef _DEBUG
	setDebugName("CPLYMeshWriter");
	#endif
}


//! Returns the type of the mesh writer
EMESH_WRITER_TYPE CPLYMeshWriter::getType() const
{
	return EMWT_PLY;
}

//! writes a mesh
bool CPLYMeshWriter::writeMesh(io::IWriteFile* file, scene::IMesh* mesh, int32_t flags)
{
	if (!file || !mesh)
		return false;

	os::Printer::log("Writing mesh", file->getFileName());

	// write PLY header
	core::stringc header =
        	"ply\n"
		"format ascii 1.0\n"
		"comment Irrlicht Engine ";
	header +=  IRRLICHT_SDK_VERSION;

	// get vertex and triangle counts
	uint32_t VertexCount   = 0;
	uint32_t TriangleCount = 0;

	for (uint32_t i=0; i < mesh->getMeshBufferCount(); ++i)
	{
		VertexCount   += mesh->getMeshBuffer(i)->getVertexCount();
		TriangleCount += mesh->getMeshBuffer(i)->getIndexCount() / 3;
	}

	// vertex definition
	header += "\nelement vertex ";
	header += VertexCount;

	header += "\n"
		"property float x\n"
		"property float y\n"
		"property float z\n"
		"property float nx\n"
		"property float ny\n"
		"property float nz\n";
	// todo: writer flags for extended (r,g,b,u,v) and non-standard (alpha,u1,uv,tx,ty,tz) properties
	//	"property uchar red\n"
	//	"property uchar green\n"
	//	"property uchar blue\n"
	//	"property uchar alpha\n"
	//	"property float u\n"
	//	"property float v\n";
	//	"property float u1\n
	//	"property float v1\n"
	//	"property float tx\n"
	//	"property float ty\n"
	//	"property float tz\n"

	// face definition

	header += "element face ";
	header += TriangleCount;
	header += "\n"
		"property list uchar int vertex_indices\n"
		"end_header\n";

	// write header
	file->write(header.c_str(), header.size());

	// write vertices

	char outLine[1024];

	for (uint32_t i=0; i < mesh->getMeshBufferCount(); ++i)
	{
		scene::IMeshBuffer* mb = mesh->getMeshBuffer(i);
		for (uint32_t j=0; j < mb->getVertexCount(); ++j)
		{
			const core::vector3df& pos = mb->getPosition(j);
			const core::vector3df& n   = mb->getNormal(j);
//			const core::vector2df& tc  = mb->getTCoords(j);

			uint8_t *buf  = (uint8_t*)mb->getVertices();
			switch(mb->getVertexType())
			{
			case video::EVT_STANDARD:
				buf += sizeof(video::S3DVertex)*j;
				break;
			case video::EVT_2TCOORDS:
				buf += sizeof(video::S3DVertex2TCoords)*j;
				break;
			case video::EVT_TANGENTS:
				buf += sizeof(video::S3DVertexTangents)*j;
				break;
			}
//			video::SColor &col = ( (video::S3DVertex*)buf )->Color;

			// x y z nx ny nz red green blue alpha u v [u1 v1 | tx ty tz]\n
			snprintf(outLine, 1024,
				"%f %f %f %f %f %f\n",// %u %u %u %u %f %f\n",
				pos.X, pos.Z, pos.Y, // Y and Z are flipped
				n.X, n.Z, n.Y);
				/*col.getRed(), col.getGreen(), col.getBlue(), col.getAlpha(),
				tc.X, tc.Y);*/

			// write the line
			file->write(outLine, strlen(outLine));
		}
	}

	// index of the first vertex in the current mesh buffer
	uint32_t StartOffset = 0;

	// write triangles
	for (uint32_t i=0; i < mesh->getMeshBufferCount(); ++i)
	{
		scene::IMeshBuffer* mb = mesh->getMeshBuffer(i);
		for (uint32_t j=0; j < mb->getIndexCount(); j+=3)
		{
			// y and z are flipped so triangles are reversed
			uint32_t a=StartOffset,
			    b=StartOffset,
			    c=StartOffset;

			switch(mb->getIndexType())
			{
			case video::EIT_16BIT:
				a += ((uint16_t*)mb->getIndices())[j+0];
				c += ((uint16_t*)mb->getIndices())[j+1];
				b += ((uint16_t*)mb->getIndices())[j+2];
				break;
			case video::EIT_32BIT:
				a += ((uint32_t*)mb->getIndices()) [j+0];
				c += ((uint32_t*)mb->getIndices()) [j+0];
				b += ((uint32_t*)mb->getIndices()) [j+0];
				break;
			}

			// count a b c\n
			snprintf(outLine, 1024, "3 %u %u %u\n", a, b, c);
			// write the line
			file->write(outLine, strlen(outLine));
		}

		// increment offset
		StartOffset += mb->getVertexCount();
	}

	// all done!


	return true;
}
#endif // NEW_MESHES

} // end namespace
} // end namespace

#endif // _IRR_COMPILE_WITH_PLY_WRITER_


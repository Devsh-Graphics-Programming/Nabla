// Copyright (C) 2007-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_STL_LOADER_

#include "CSTLMeshFileLoader.h"
#include "SMesh.h"
#include "SAnimatedMesh.h"
#include "IReadFile.h"
#include "fast_atof.h"
#include "coreutil.h"
#include "os.h"
#include "SVertexManipulator.h"

#include <vector>

namespace irr
{
namespace scene
{

#include "irrpack.h"
struct STLVertex
{
    float pos[3];
    uint32_t normal32bit;
    uint32_t color;
} PACK_STRUCT;
#include "irrunpack.h"

//! returns true if the file maybe is able to be loaded by this class
//! based on the file extension (e.g. ".bsp")
bool CSTLMeshFileLoader::isALoadableFileExtension(const io::path& filename) const
{
	return core::hasFileExtension ( filename, "stl" );
}


inline void addTriangleToMesh(std::vector<STLVertex> &verticesOut, const core::vectorSIMDf* positions, const core::vectorSIMDf &normal, const video::SColor &color)
{
    STLVertex vertex;
    vertex.normal32bit = quantizeNormal2_10_10_10(normal);
    vertex.color = color.color;
    memcpy(vertex.pos,positions+2,12);
    verticesOut.push_back(vertex);
    memcpy(vertex.pos,positions+1,12);
    verticesOut.push_back(vertex);
    memcpy(vertex.pos,positions+0,12);
    verticesOut.push_back(vertex);
}

//! creates/loads an animated mesh from the file.
//! \return Pointer to the created mesh. Returns 0 if loading failed.
//! If you no longer need the mesh, you should call ICPUMesh::drop().
//! See IReferenceCounted::drop() for more information.
ICPUMesh* CSTLMeshFileLoader::createMesh(io::IReadFile* file)
{
	const long filesize = file->getSize();
	if (filesize < 6) // we need a header
		return 0;

	ICPUMeshDataFormatDesc* desc = new ICPUMeshDataFormatDesc();
	ICPUMeshBuffer* meshbuffer = new ICPUMeshBuffer();
	meshbuffer->setMeshDataAndFormat(desc);
	desc->drop();

	SCPUMesh* mesh = new SCPUMesh();
	mesh->addMeshBuffer(meshbuffer);
	meshbuffer->drop();

	core::vectorSIMDf vertex[3];
	core::vectorSIMDf normal;

	bool binary = false;
	core::stringc token;
	if (getNextToken(file, token) != "solid")
		binary = true;
	// read/skip header
	u32 binFaceCount = 0;
	std::vector<STLVertex> vertices;
	if (binary)
	{
		file->seek(80);
		file->read(&binFaceCount, 4);
#ifdef __BIG_ENDIAN__
		binFaceCount = os::Byteswap::byteswap(binFaceCount);
#endif
        vertices.reserve(binFaceCount);
	}
	else
		goNextLine(file);


	u16 attrib=0;
	token.reserve(32);
	while (file->getPos() < filesize)
	{
		if (!binary)
		{
			if (getNextToken(file, token) != "facet")
			{
				if (token=="endsolid")
					break;
				mesh->drop();
				return 0;
			}
			if (getNextToken(file, token) != "normal")
			{
				mesh->drop();
				return 0;
			}
		}
		getNextVector(file, normal, binary);
		if (!binary)
		{
			if (getNextToken(file, token) != "outer")
			{
				mesh->drop();
				return 0;
			}
			if (getNextToken(file, token) != "loop")
			{
				mesh->drop();
				return 0;
			}
		}
		for (u32 i=0; i<3; ++i)
		{
			if (!binary)
			{
				if (getNextToken(file, token) != "vertex")
				{
					mesh->drop();
					return 0;
				}
			}
			getNextVector(file, vertex[i], binary);
		}
		if (!binary)
		{
			if (getNextToken(file, token) != "endloop")
			{
				mesh->drop();
				return 0;
			}
			if (getNextToken(file, token) != "endfacet")
			{
				mesh->drop();
				return 0;
			}
		}
		else
		{
			file->read(&attrib, 2);
#ifdef __BIG_ENDIAN__
			attrib = os::Byteswap::byteswap(attrib);
#endif
		}

		video::SColor color(0xffffffff);
		if (attrib & 0x8000)
			color = video::A1R5G5B5toA8R8G8B8(attrib);
		if ((normal==core::vectorSIMDf()).all())
			normal.set(core::plane3df(vertex[2].getAsVector3df(),vertex[1].getAsVector3df(),vertex[0].getAsVector3df()).Normal);
        //
       addTriangleToMesh(vertices,vertex,normal,color);
	}	// end while (file->getPos() < filesize)
	core::ICPUBuffer* vertexBuf = new core::ICPUBuffer(sizeof(STLVertex)*vertices.size());
	std::copy( vertices.begin(), vertices.end(), (STLVertex*)vertexBuf->getPointer() );
	desc->mapVertexAttrBuffer(vertexBuf,EVAI_ATTR0,ECPA_THREE,ECT_FLOAT,sizeof(STLVertex),0);
	desc->mapVertexAttrBuffer(vertexBuf,EVAI_ATTR3,ECPA_FOUR,ECT_INT_2_10_10_10_REV,sizeof(STLVertex),12);
	desc->mapVertexAttrBuffer(vertexBuf,EVAI_ATTR1,ECPA_REVERSED_OR_BGRA,ECT_NORMALIZED_UNSIGNED_BYTE,sizeof(STLVertex),16);
	vertexBuf->drop();
	meshbuffer->setIndexCount(vertices.size());
	mesh->recalculateBoundingBox(true);

	return mesh;
}


//! Read 3d vector of floats
void CSTLMeshFileLoader::getNextVector(io::IReadFile* file, core::vectorSIMDf& vec, bool binary) const
{
	if (binary)
	{
		file->read(&vec.X, 4);
		file->read(&vec.Y, 4);
		file->read(&vec.Z, 4);
#ifdef __BIG_ENDIAN__
		vec.X = os::Byteswap::byteswap(vec.X);
		vec.Y = os::Byteswap::byteswap(vec.Y);
		vec.Z = os::Byteswap::byteswap(vec.Z);
#endif
	}
	else
	{
		goNextWord(file);
		core::stringc tmp;

		getNextToken(file, tmp);
		core::fast_atof_move(tmp.c_str(), vec.X);
		getNextToken(file, tmp);
		core::fast_atof_move(tmp.c_str(), vec.Y);
		getNextToken(file, tmp);
		core::fast_atof_move(tmp.c_str(), vec.Z);
	}
	vec.X=-vec.X;
}


//! Read next word
const core::stringc& CSTLMeshFileLoader::getNextToken(io::IReadFile* file, core::stringc& token) const
{
	goNextWord(file);
	u8 c;
	token = "";
	while(file->getPos() != file->getSize())
	{
		file->read(&c, 1);
		// found it, so leave
		if (core::isspace(c))
			break;
		token.append(c);
	}
	return token;
}


//! skip to next word
void CSTLMeshFileLoader::goNextWord(io::IReadFile* file) const
{
	u8 c;
	while(file->getPos() != file->getSize())
	{
		file->read(&c, 1);
		// found it, so leave
		if (!core::isspace(c))
		{
			file->seek(-1, true);
			break;
		}
	}
}


//! Read until line break is reached and stop at the next non-space character
void CSTLMeshFileLoader::goNextLine(io::IReadFile* file) const
{
	u8 c;
	// look for newline characters
	while(file->getPos() != file->getSize())
	{
		file->read(&c, 1);
		// found it, so leave
		if (c=='\n' || c=='\r')
			break;
	}
}

} // end namespace scene
} // end namespace irr


#endif // _IRR_COMPILE_WITH_STL_LOADER_


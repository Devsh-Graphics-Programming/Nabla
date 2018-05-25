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

#ifdef NEW_MESHES
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
bool CPLYMeshWriter::writeMesh(io::IWriteFile* file, scene::ICPUMesh* mesh, int32_t flags)
{
	if (!file || !mesh || mesh->getMeshBufferCount() > 1)
		return false;

	os::Printer::log("Writing mesh", file->getFileName().c_str());

	// write PLY header
    std::string header = "ply\n";
    header += (flags & EMWF_WRITE_BINARY) ? "format binary_little_endian 1.0" : "format ascii 1.0";
	header += "\ncomment Irrlicht Engine ";
	header +=  IRRLICHT_SDK_VERSION;

	// get vertex and triangle counts
	size_t vtxCount = mesh->getMeshBuffer(0)->calcVertexCount();
    size_t faceCount = mesh->getMeshBuffer(0)->getIndexCount() / 3;

	// vertex definition
	header += "\nelement vertex ";
	header += std::to_string(vtxCount) + '\n';

    bool vaidToWrite[4]{ 0, 0, 0, 0 };
    auto desc = mesh->getMeshBuffer(0)->getMeshDataAndFormat();
    if (desc->getMappedBuffer(EVAI_ATTR0))
    {
        vaidToWrite[0] = true;
        header +=
            "property float x\n"
            "property float y\n"
            "property float z\n";
    }
    if (desc->getMappedBuffer(EVAI_ATTR1))
    {
        vaidToWrite[1] = true;
        header +=
            "property float red\n"
            "property float blue\n"
            "property float green\n";
        if (desc->getAttribType(EVAI_ATTR1) == ECPA_FOUR || desc->getAttribType(EVAI_ATTR1) == ECPA_REVERSED_OR_BGRA)
            header += "property float alpha\n";
    }
    if (desc->getMappedBuffer(EVAI_ATTR2))
    {
        vaidToWrite[2] = true;
        header +=
            "property float u\n"
            "property float v\n";
    }
    if (desc->getMappedBuffer(EVAI_ATTR3))
    {
        vaidToWrite[3] = true;
        header +=
            "property float nx\n"
            "property float ny\n"
            "property float nz\n";
    }

    if (desc->getIndexBuffer() && mesh->getMeshBuffer(0)->getIndexType() != video::EIT_UNKNOWN)
    {
        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        const std::string idxType = mesh->getMeshBuffer(0)->getIndexType() == video::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxType + " vertex_indices\n";
    }
    else faceCount = 0u;
    header += "end_header\n";

    file->write(header.c_str(), header.size());

    if (flags & EMWF_WRITE_BINARY)
        writeBinary(file, mesh->getMeshBuffer(0), vtxCount, faceCount, vaidToWrite);
    else
        writeText(file, mesh->getMeshBuffer(0), vtxCount, faceCount, vaidToWrite);

	return true;
}

void CPLYMeshWriter::writeBinary(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, const bool _vaidToWrite[4]) const
{
    size_t colCpa = _mbuf->getMeshDataAndFormat()->getAttribComponentCount(EVAI_ATTR1);
    if (colCpa == ECPA_REVERSED_OR_BGRA)
        colCpa = 4u;

    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf a;
        if (_vaidToWrite[EVAI_ATTR0])
        {
            _mbuf->getAttribute(a, EVAI_ATTR0, i);
            _file->write(a.pointer, 3*sizeof(float));
        }
        if (_vaidToWrite[EVAI_ATTR1])
        {
            _mbuf->getAttribute(a, EVAI_ATTR1, i);
            _file->write(a.pointer, colCpa*sizeof(float));
        }
        if (_vaidToWrite[EVAI_ATTR2])
        {
            _mbuf->getAttribute(a, EVAI_ATTR2, i);
            _file->write(a.pointer, 2*sizeof(float));
        }
        if (_vaidToWrite[EVAI_ATTR3])
        {
            _mbuf->getAttribute(a, EVAI_ATTR3, i);
            _file->write(a.pointer, 3*sizeof(float));
        }
    }
    const uint8_t listSize = 3u;
    if (_mbuf->getIndexType() == video::EIT_32BIT)
    {
        uint32_t* indices = (uint32_t*)_mbuf->getIndices();
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(&listSize, 1);
            _file->write(indices, listSize * 4);
            indices += listSize;
        }
    }
    else
    {
        uint16_t* indices = (uint16_t*)_mbuf->getIndices();
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(&listSize, 1);
            _file->write(indices, listSize*2);
            indices += listSize;
        }
    }
}

void CPLYMeshWriter::writeText(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, const bool _vaidToWrite[4]) const
{
    size_t colCpa = _mbuf->getMeshDataAndFormat()->getAttribComponentCount(EVAI_ATTR1);
    if (colCpa == ECPA_REVERSED_OR_BGRA)
        colCpa = 4u;

    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf a;
        if (_vaidToWrite[EVAI_ATTR0])
        {
            _mbuf->getAttribute(a, EVAI_ATTR0, i);
            writeVectorAsText(_file, a.pointer, 3);
            _file->write("\n", 1);
        }
        if (_vaidToWrite[EVAI_ATTR1])
        {
            _mbuf->getAttribute(a, EVAI_ATTR1, i);
            writeVectorAsText(_file, a.pointer, colCpa);
            _file->write("\n", 1);
        }
        if (_vaidToWrite[EVAI_ATTR2])
        {
            _mbuf->getAttribute(a, EVAI_ATTR2, i);
            writeVectorAsText(_file, a.pointer, 2);
            _file->write("\n", 1);
        }
        if (_vaidToWrite[EVAI_ATTR3])
        {
            _mbuf->getAttribute(a, EVAI_ATTR3, i);
            writeVectorAsText(_file, a.pointer, 3);
            _file->write("\n", 1);
        }
    }
    const char* listSize = "3 ";
    if (_mbuf->getIndexType() == video::EIT_32BIT)
    {
        uint32_t* indices = (uint32_t*)_mbuf->getIndices();
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(listSize, 2);
            writeVectorAsText(_file, indices, 3);
            _file->write("\n", 1);
            indices += 3;
        }
    }
    else
    {
        uint16_t* indices = (uint16_t*)_mbuf->getIndices();
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(&listSize, 2);
            writeVectorAsText(_file, indices, 3);
            _file->write("\n", 1);
            indices += 3;
        }
    }
}
#endif // NEW_MESHES

} // end namespace
} // end namespace

#endif // _IRR_COMPILE_WITH_PLY_WRITER_


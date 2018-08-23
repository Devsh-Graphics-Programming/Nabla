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
#include "CMeshManipulator.h"


namespace irr
{
namespace scene
{

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
	header += "\ncomment IrrlichtBAW ";
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
        const E_COMPONENT_TYPE t = desc->getAttribType(EVAI_ATTR0);
        std::string typeStr = getTypeString(t);
        vaidToWrite[0] = true;
        header +=
            "property " + typeStr + " x\n" +
            "property " + typeStr + " y\n" +
            "property " + typeStr + " z\n";
    }
    if (desc->getMappedBuffer(EVAI_ATTR1))
    {
        const E_COMPONENT_TYPE t = desc->getAttribType(EVAI_ATTR1);
        std::string typeStr = getTypeString(t);
        vaidToWrite[1] = true;
        header +=
            "property " + typeStr + " red\n" +
            "property " + typeStr + " green\n" +
            "property " + typeStr + " blue\n";
        if (desc->getAttribComponentCount(EVAI_ATTR1) == ECPA_FOUR || desc->getAttribComponentCount(EVAI_ATTR1) == ECPA_REVERSED_OR_BGRA)
        {
            header += "property " + typeStr + " alpha\n";
        }
    }
    if (desc->getMappedBuffer(EVAI_ATTR2))
    {
        const E_COMPONENT_TYPE t = desc->getAttribType(EVAI_ATTR2);
        std::string typeStr = getTypeString(t);
        vaidToWrite[2] = true;
        header +=
            "property " + typeStr + " u\n" +
            "property " + typeStr + " v\n";
    }
    if (desc->getMappedBuffer(EVAI_ATTR3))
    {
        const E_COMPONENT_TYPE t = desc->getAttribType(EVAI_ATTR3);
        std::string typeStr = getTypeString(t);
        vaidToWrite[3] = true;
        header +=
            "property " + typeStr + " nx\n" +
            "property " + typeStr + " ny\n" +
            "property " + typeStr + " nz\n";
    }


    void* indices = nullptr;
    bool needToFreeIndices = false;
    if (mesh->getMeshBuffer(0)->getIndices() && mesh->getMeshBuffer(0)->getPrimitiveType() != EPT_TRIANGLES)
    {
        void* ind = mesh->getMeshBuffer(0)->getIndices();
        const size_t idxCnt = mesh->getMeshBuffer(0)->getIndexCount();
        const video::E_INDEX_TYPE idxtype = mesh->getMeshBuffer(0)->getIndexType();
        const E_PRIMITIVE_TYPE primitiveT = mesh->getMeshBuffer(0)->getPrimitiveType();
        if (primitiveT == EPT_TRIANGLE_FAN || primitiveT == EPT_TRIANGLE_STRIP)
        {
            core::ICPUBuffer* buf = nullptr;
            if (primitiveT == EPT_TRIANGLE_FAN)
            {
                buf = CMeshManipulator().idxBufferFromTrianglesFanToTriangles(ind, idxCnt, idxtype);
            }
            else if (primitiveT == EPT_TRIANGLE_STRIP)
            {
                buf = CMeshManipulator().idxBufferFromTriangleStripsToTriangles(ind, idxCnt, idxtype);
            }
            needToFreeIndices = true;
            faceCount = buf->getSize() / (idxtype == video::EIT_16BIT ? 2u : 4u) / 3u;
            indices = malloc(buf->getSize());
            memcpy(indices, buf->getPointer(), buf->getSize());
            buf->drop();
        }
    }
    else
    {
        indices = mesh->getMeshBuffer(0)->getIndices();
    }

    video::E_INDEX_TYPE idxT = video::EIT_UNKNOWN;
    bool forceFaces = false;
    if (mesh->getMeshBuffer(0)->getPrimitiveType() == EPT_POINTS)
    {
        faceCount = 0u;
    }
    else if (indices && mesh->getMeshBuffer(0)->getIndexType() != video::EIT_UNKNOWN)
    {
        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        idxT = mesh->getMeshBuffer(0)->getIndexType();
        const std::string idxTypeStr = idxT == video::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxTypeStr + " vertex_indices\n";
    }
    else if (mesh->getMeshBuffer(0)->getPrimitiveType() == EPT_TRIANGLES)
    {
        faceCount = vtxCount / 3;
        forceFaces = true;

        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        idxT = vtxCount <= (1u<<16 - 1) ? video::EIT_16BIT : video::EIT_32BIT;
        const std::string idxTypeStr = idxT == video::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxTypeStr + " vertex_indices\n";
    }
    else
    {
        faceCount = 0u;
    }
    header += "end_header\n";

    file->write(header.c_str(), header.size());

    if (flags & EMWF_WRITE_BINARY)
        writeBinary(file, mesh->getMeshBuffer(0), vtxCount, faceCount, idxT, indices, forceFaces, vaidToWrite);
    else
        writeText(file, mesh->getMeshBuffer(0), vtxCount, faceCount, idxT, indices, forceFaces, vaidToWrite);

    if (needToFreeIndices)
        free(indices);

	return true;
}

void CPLYMeshWriter::writeBinary(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, video::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4]) const
{
    size_t colCpa = _mbuf->getMeshDataAndFormat()->getAttribComponentCount(EVAI_ATTR1);
    if (colCpa == ECPA_REVERSED_OR_BGRA)
        colCpa = 4u;

    ICPUMeshBuffer* mbCopy = createCopyMBuffNormalizedReplacedWithTrueInt(_mbuf);
    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf f;
        uint32_t ui[4];
        if (_vaidToWrite[EVAI_ATTR0])
        {
            writeAttribBinary(_file, mbCopy, EVAI_ATTR0, i, 3u);
            /*_mbuf->getAttribute(f, EVAI_ATTR0, i);
            _file->write(f.pointer, 3*sizeof(float));*/
        }
        if (_vaidToWrite[EVAI_ATTR1])
        {
            writeAttribBinary(_file, mbCopy, EVAI_ATTR1, i, colCpa);
            //const E_COMPONENT_TYPE t = _mbuf->getMeshDataAndFormat()->getAttribType(EVAI_ATTR1);
            //if (scene::isWeakInteger(t) || scene::isNativeInteger(t))
            //{
            //    _mbuf->getAttribute(ui, EVAI_ATTR1, i);
            //    uint8_t ui8[4];
            //    for (uint32_t k = 0u; k < colCpa; ++k)
            //        ui8[k] = ui[k];
            //    _file->write(ui8, colCpa);
            //}
            //else
            //{
            //    _mbuf->getAttribute(f, EVAI_ATTR1, i);
            //    f *= 255.f;
            //    _file->write(f.pointer, colCpa * sizeof(float));
            //}
        }
        if (_vaidToWrite[EVAI_ATTR2])
        {
            writeAttribBinary(_file, mbCopy, EVAI_ATTR2, i, 2u);
            //_mbuf->getAttribute(f, EVAI_ATTR2, i);
            //_file->write(f.pointer, 2*sizeof(float));
        }
        if (_vaidToWrite[EVAI_ATTR3])
        {
            writeAttribBinary(_file, mbCopy, EVAI_ATTR3, i, 3u);
            //_mbuf->getAttribute(f, EVAI_ATTR3, i);
            //_file->write(f.pointer, 3*sizeof(float));
        }
    }
    mbCopy->drop();
    mbCopy = nullptr;

    const uint8_t listSize = 3u;
    void* indices = _indices;
    if (_forceFaces)
    {
        indices = malloc((_idxType == video::EIT_32BIT ? 4 : 2) * listSize * _fcCount);
        if (_idxType == video::EIT_16BIT)
        {
            for (uint16_t i = 0u; i < _fcCount; ++i)
                ((uint16_t*)indices)[i] = i;
        }
        else
        {
            for (uint32_t i = 0u; i < _fcCount; ++i)
                ((uint32_t*)indices)[i] = i;
        }
    }
    if (_idxType == video::EIT_32BIT)
    {
        uint32_t* ind = (uint32_t*)indices;
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(&listSize, 1);
            _file->write(ind, listSize * 4);
            ind += listSize;
        }
    }
    else
    {
        uint16_t* ind = (uint16_t*)indices;
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(&listSize, 1);
            _file->write(ind, listSize*2);
            ind += listSize;
        }
    }

    if (_forceFaces)
        free(indices);
}

void CPLYMeshWriter::writeText(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, video::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4]) const
{
    ICPUMeshBuffer* mbCopy = createCopyMBuffNormalizedReplacedWithTrueInt(_mbuf);

    auto writefunc = [&_file,&mbCopy,this](E_VERTEX_ATTRIBUTE_ID _vaid, size_t _ix, size_t _cpa)
    {
        uint32_t ui[4];
        core::vectorSIMDf f;
        const E_COMPONENT_TYPE t = mbCopy->getMeshDataAndFormat()->getAttribType(_vaid);
        if (scene::isWeakInteger(t) || scene::isNativeInteger(t))
        {
            mbCopy->getAttribute(ui, _vaid, _ix);
            if (scene::isUnsigned(t))
                writeVectorAsText(_file, ui, _cpa);
            else
            {
                int32_t ii[4];
                memcpy(ii, ui, 4*4);
                writeVectorAsText(_file, ii, _cpa);
            }
        }
        else
        {
            mbCopy->getAttribute(f, _vaid, _ix);
            writeVectorAsText(_file, f.pointer, _cpa);
        }
    };

    size_t colCpa = _mbuf->getMeshDataAndFormat()->getAttribComponentCount(EVAI_ATTR1);
    if (colCpa == ECPA_REVERSED_OR_BGRA)
        colCpa = 4u;

    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf f;
        uint32_t ui[4];
        if (_vaidToWrite[EVAI_ATTR0])
        {
            //_mbuf->getAttribute(f, EVAI_ATTR0, i);
            //writeVectorAsText(_file, f.pointer, 3);
            writefunc(EVAI_ATTR0, i, 3u);
            //_file->write("\n", 1);
        }
        if (_vaidToWrite[EVAI_ATTR1])
        {
            writefunc(EVAI_ATTR1, i, colCpa);
            //_file->write("\n", 1);
        }
        if (_vaidToWrite[EVAI_ATTR2])
        {
            //_mbuf->getAttribute(f, EVAI_ATTR2, i);
            //writeVectorAsText(_file, f.pointer, 2);
            writefunc(EVAI_ATTR2, i, 2u);
            //_file->write("\n", 1);
        }
        if (_vaidToWrite[EVAI_ATTR3])
        {
            //_mbuf->getAttribute(f, EVAI_ATTR3, i);
            //writeVectorAsText(_file, f.pointer, 3);
            writefunc(EVAI_ATTR3, i, 3u);
            //_file->write("\n", 1);
        }
        _file->write("\n", 1);
    }
    mbCopy->drop();
    mbCopy = nullptr;

    const char* listSize = "3 ";
    void* indices = _indices;
    if (_forceFaces)
    {
        indices = malloc((_idxType == video::EIT_32BIT ? 4 : 2) * 3 * _fcCount);
        if (_idxType == video::EIT_16BIT)
        {
            for (uint16_t i = 0u; i < _fcCount; ++i)
                ((uint16_t*)indices)[i] = i;
        }
        else
        {
            for (uint32_t i = 0u; i < _fcCount; ++i)
                ((uint32_t*)indices)[i] = i;
        }
    }
    if (_idxType == video::EIT_32BIT)
    {
        uint32_t* ind = (uint32_t*)indices;
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(listSize, 2);
            writeVectorAsText(_file, ind, 3);
            _file->write("\n", 1);
            ind += 3;
        }
    }
    else
    {
        uint16_t* ind = (uint16_t*)indices;
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            _file->write(&listSize, 2);
            writeVectorAsText(_file, ind, 3);
            _file->write("\n", 1);
            ind += 3;
        }
    }

    if (_forceFaces)
        free(indices);
}

void CPLYMeshWriter::writeAttribBinary(io::IWriteFile* _file, ICPUMeshBuffer* _mbuf, E_VERTEX_ATTRIBUTE_ID _vaid, size_t _ix, size_t _cpa) const
{
    uint32_t ui[4];
    core::vectorSIMDf f;
    E_COMPONENT_TYPE t = _mbuf->getMeshDataAndFormat()->getAttribType(_vaid);
    if (scene::isWeakInteger(t) || scene::isNativeInteger(t))
    {
        _mbuf->getAttribute(ui, _vaid, _ix);

        if (scene::vertexAttrSize[t][ECPA_ONE] == 1 || t == ECT_INT_2_10_10_10_REV || t == ECT_UNSIGNED_INT_2_10_10_10_REV)
        {
            uint8_t a[4];
            for (uint32_t k = 0u; k < _cpa; ++k)
                a[k] = ui[k];
            _file->write(a, _cpa);
        }
        else if (scene::vertexAttrSize[t][ECPA_ONE] == 2)
        {
            uint16_t a[4];
            for (uint32_t k = 0u; k < _cpa; ++k)
                a[k] = ui[k];
            _file->write(a, 2*_cpa);
        }
        else if (scene::vertexAttrSize[t][ECPA_ONE] == 4)
        {
            _file->write(ui, 4*_cpa);
        }
    }
    else
    {
        _mbuf->getAttribute(f, _vaid, _ix);
        _file->write(f.pointer, 4*_cpa);
    }
}

ICPUMeshBuffer* CPLYMeshWriter::createCopyMBuffNormalizedReplacedWithTrueInt(const ICPUMeshBuffer* _mbuf)
{
    ICPUMeshBuffer* mbCopy = new ICPUMeshBuffer();
    auto origDesc = _mbuf->getMeshDataAndFormat();
    ICPUMeshDataFormatDesc* desc = new ICPUMeshDataFormatDesc();
    for (size_t i = EVAI_ATTR0; i < EVAI_COUNT; ++i)
    {
        E_VERTEX_ATTRIBUTE_ID vaid = (E_VERTEX_ATTRIBUTE_ID)i;
        E_COMPONENT_TYPE t = origDesc->getAttribType(vaid);
        if (origDesc->getMappedBuffer(vaid))
            desc->mapVertexAttrBuffer(
                const_cast<core::ICPUBuffer*>(origDesc->getMappedBuffer(vaid)),
                vaid,
                origDesc->getAttribComponentCount(vaid),
                scene::isNormalized(t) ? scene::getCorrespondingNativeIntType(scene::getCorrespondingDenormalizedType(t)) : t,
                origDesc->getMappedBufferStride(vaid),
                origDesc->getMappedBufferOffset(vaid),
                origDesc->getAttribDivisor(vaid)
            );
    }
    mbCopy->setMeshDataAndFormat(desc);
    desc->drop();

    mbCopy->setBaseVertex(_mbuf->getBaseVertex());
    mbCopy->setIndexCount(_mbuf->getIndexCount());
    mbCopy->setBaseInstance(_mbuf->getBaseInstance());
    mbCopy->setInstanceCount(_mbuf->getInstanceCount());
    mbCopy->setIndexBufferOffset(_mbuf->getIndexBufferOffset());
    mbCopy->setIndexType(_mbuf->getIndexType());
    mbCopy->setPrimitiveType(_mbuf->getPrimitiveType());
    mbCopy->setPositionAttributeIx(_mbuf->getPositionAttributeIx());

    return mbCopy;
}

std::string CPLYMeshWriter::getTypeString(E_COMPONENT_TYPE _t)
{
    switch (_t)
    {
    case ECT_DOUBLE_IN_DOUBLE_OUT:
    case ECT_DOUBLE_IN_FLOAT_OUT:
    case ECT_FLOAT: 
    case ECT_HALF_FLOAT:
    case ECT_UNSIGNED_INT_10F_11F_11F_REV:
        return "float";
    case ECT_BYTE:
    case ECT_NORMALIZED_BYTE:
    case ECT_INT_2_10_10_10_REV:
    case ECT_NORMALIZED_INT_2_10_10_10_REV:
    case ECT_INTEGER_BYTE:
    case ECT_INTEGER_INT_2_10_10_10_REV:
        return "char";
    case ECT_UNSIGNED_BYTE:
    case ECT_NORMALIZED_UNSIGNED_BYTE:
    case ECT_UNSIGNED_INT_2_10_10_10_REV:
    case ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV:
    case ECT_INTEGER_UNSIGNED_BYTE:
    case ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV:
        return "uchar";
    case ECT_SHORT:
    case ECT_NORMALIZED_SHORT:
    case ECT_INTEGER_SHORT:
        return "short";
    case ECT_UNSIGNED_SHORT:
    case ECT_NORMALIZED_UNSIGNED_SHORT:
    case ECT_INTEGER_UNSIGNED_SHORT:
        return "ushort";
    case ECT_INT:
    case ECT_NORMALIZED_INT:
    case ECT_INTEGER_INT:
        return "int";
    case ECT_UNSIGNED_INT:
    case ECT_INTEGER_UNSIGNED_INT:
    case ECT_NORMALIZED_UNSIGNED_INT:
        return "uint";
    }
    return "";
}

} // end namespace
} // end namespace

#endif // _IRR_COMPILE_WITH_PLY_WRITER_


// Copyright (C) 2008-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "irr/core/core.h"

#ifdef _IRR_COMPILE_WITH_PLY_WRITER_

#include "CPLYMeshWriter.h"
#include "os.h"
#include "irr/asset/IMesh.h"
#include "IWriteFile.h"
#include "CMeshManipulator.h"


namespace irr
{
namespace asset
{

namespace impl
{
static asset::E_FORMAT getCorrespondingIntegerFormat(asset::E_FORMAT _fmt)
{
    using namespace asset;
    switch (_fmt)
    {
    case EF_R8_UNORM: return EF_R8_UINT;
    case EF_R8_SNORM: return EF_R8_SINT;
    case EF_R8G8_UNORM: return EF_R8G8_UINT;
    case EF_R8G8_SNORM: return EF_R8G8_SINT;
    case EF_R8G8B8_UNORM: return EF_R8G8B8_UINT;
    case EF_R8G8B8_SNORM: return EF_R8G8B8_SINT;
    case EF_R8G8B8A8_UNORM: return EF_R8G8B8A8_UINT;
    case EF_R8G8B8A8_SNORM: return EF_R8G8B8A8_SINT;
    case EF_R16_UNORM: return EF_R16_UINT;
    case EF_R16_SNORM: return EF_R16_SINT;
    case EF_R16G16_UNORM: return EF_R16G16_UINT;
    case EF_R16G16_SNORM: return EF_R16G16_SINT;
    case EF_R16G16B16_UNORM: return EF_R16G16B16_UINT;
    case EF_R16G16B16_SNORM: return EF_R16G16B16_SINT;
    case EF_R16G16B16A16_UNORM: return EF_R16G16B16A16_UINT;
    case EF_R16G16B16A16_SNORM: return EF_R16G16B16A16_SINT;
    case EF_A2B10G10R10_UNORM_PACK32: return EF_A2B10G10R10_UINT_PACK32;
    case EF_A2B10G10R10_SNORM_PACK32: return EF_A2B10G10R10_SINT_PACK32;
    case EF_B8G8R8A8_UNORM: return EF_R8G8B8A8_SINT;
    case EF_A2R10G10B10_UNORM_PACK32: return EF_A2B10G10R10_UINT_PACK32;
    case EF_A2R10G10B10_SNORM_PACK32: return EF_A2B10G10R10_SINT_PACK32;
    default: return EF_UNKNOWN;
    }
}
}

CPLYMeshWriter::CPLYMeshWriter()
{
	#ifdef _IRR_DEBUG
	setDebugName("CPLYMeshWriter");
	#endif
}

//! writes a mesh
bool CPLYMeshWriter::writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext ctx{_params, _file};

    const asset::ICPUMesh* mesh =
#   ifndef _IRR_DEBUG
        static_cast<const asset::ICPUMesh*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::ICPUMesh*>(_params.rootAsset);
#   endif
    assert(mesh);

    io::IWriteFile* file = _override->getOutputFile(_file, ctx, {mesh, 0u});

	if (!file || !mesh || mesh->getMeshBufferCount() > 1)
		return false;

	os::Printer::log("Writing mesh", file->getFileName().c_str());

    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(ctx, mesh, 0u);

	// write PLY header
    std::string header = "ply\n";
    header += (flags & asset::EWF_BINARY) ? "format binary_little_endian 1.0" : "format ascii 1.0";
	header += "\ncomment IrrlichtBAW ";
	header +=  IRRLICHTBAW_SDK_VERSION;

	// get vertex and triangle counts
	size_t vtxCount = mesh->getMeshBuffer(0)->calcVertexCount();
    size_t faceCount = mesh->getMeshBuffer(0)->getIndexCount() / 3;

	// vertex definition
	header += "\nelement vertex ";
	header += std::to_string(vtxCount) + '\n';

    bool vaidToWrite[4]{ 0, 0, 0, 0 };
    auto desc = mesh->getMeshBuffer(0)->getMeshDataAndFormat();
    if (desc->getMappedBuffer(asset::EVAI_ATTR0))
    {
        const asset::E_FORMAT t = desc->getAttribFormat(asset::EVAI_ATTR0);
        std::string typeStr = getTypeString(t);
        vaidToWrite[0] = true;
        header +=
            "property " + typeStr + " x\n" +
            "property " + typeStr + " y\n" +
            "property " + typeStr + " z\n";
    }
    if (desc->getMappedBuffer(asset::EVAI_ATTR1))
    {
        const asset::E_FORMAT t = desc->getAttribFormat(asset::EVAI_ATTR1);
        std::string typeStr = getTypeString(t);
        vaidToWrite[1] = true;
        header +=
            "property " + typeStr + " red\n" +
            "property " + typeStr + " green\n" +
            "property " + typeStr + " blue\n";
        if (asset::getFormatChannelCount(t) == 4u)
        {
            header += "property " + typeStr + " alpha\n";
        }
    }
    if (desc->getMappedBuffer(asset::EVAI_ATTR2))
    {
        const asset::E_FORMAT t = desc->getAttribFormat(asset::EVAI_ATTR1);
        std::string typeStr = getTypeString(t);
        vaidToWrite[2] = true;
        header +=
            "property " + typeStr + " u\n" +
            "property " + typeStr + " v\n";
    }
    if (desc->getMappedBuffer(asset::EVAI_ATTR3))
    {
        const asset::E_FORMAT t = desc->getAttribFormat(asset::EVAI_ATTR1);
        std::string typeStr = getTypeString(t);
        vaidToWrite[3] = true;
        header +=
            "property " + typeStr + " nx\n" +
            "property " + typeStr + " ny\n" +
            "property " + typeStr + " nz\n";
    }


    void* indices = nullptr;
    bool needToFreeIndices = false;
    if (mesh->getMeshBuffer(0)->getIndices() && mesh->getMeshBuffer(0)->getPrimitiveType() != asset::EPT_TRIANGLES)
    {
        void* ind = mesh->getMeshBuffer(0)->getIndices();
        const size_t idxCnt = mesh->getMeshBuffer(0)->getIndexCount();
        const asset::E_INDEX_TYPE idxtype = mesh->getMeshBuffer(0)->getIndexType();
        const asset::E_PRIMITIVE_TYPE primitiveT = mesh->getMeshBuffer(0)->getPrimitiveType();
        if (primitiveT == asset::EPT_TRIANGLE_FAN || primitiveT == asset::EPT_TRIANGLE_STRIP)
        {
			core::smart_refctd_ptr<ICPUBuffer> buf;
            if (primitiveT == asset::EPT_TRIANGLE_FAN)
            {
                buf = IMeshManipulator::idxBufferFromTrianglesFanToTriangles(ind, idxCnt, idxtype);
            }
            else if (primitiveT == asset::EPT_TRIANGLE_STRIP)
            {
                buf = IMeshManipulator::idxBufferFromTriangleStripsToTriangles(ind, idxCnt, idxtype);
            }
            needToFreeIndices = true;
            faceCount = buf->getSize() / (idxtype == asset::EIT_16BIT ? 2u : 4u) / 3u;
            indices = _IRR_ALIGNED_MALLOC(buf->getSize(),_IRR_SIMD_ALIGNMENT);
            memcpy(indices, buf->getPointer(), buf->getSize());
        }
    }
    else
    {
        indices = mesh->getMeshBuffer(0)->getIndices();
    }

    asset::E_INDEX_TYPE idxT = asset::EIT_UNKNOWN;
    bool forceFaces = false;
    if (mesh->getMeshBuffer(0)->getPrimitiveType() == asset::EPT_POINTS)
    {
        faceCount = 0u;
    }
    else if (indices && mesh->getMeshBuffer(0)->getIndexType() != asset::EIT_UNKNOWN)
    {
        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        idxT = mesh->getMeshBuffer(0)->getIndexType();
        const std::string idxTypeStr = idxT == asset::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxTypeStr + " vertex_indices\n";
    }
    else if (mesh->getMeshBuffer(0)->getPrimitiveType() == asset::EPT_TRIANGLES)
    {
        faceCount = vtxCount / 3;
        forceFaces = true;

        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        idxT = vtxCount <= (1u<<16 - 1) ? asset::EIT_16BIT : asset::EIT_32BIT;
        const std::string idxTypeStr = idxT == asset::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxTypeStr + " vertex_indices\n";
    }
    else
    {
        faceCount = 0u;
    }
    header += "end_header\n";

    file->write(header.c_str(), header.size());

    if (flags & asset::EWF_BINARY)
        writeBinary(file, mesh->getMeshBuffer(0), vtxCount, faceCount, idxT, indices, forceFaces, vaidToWrite, _params);
    else
        writeText(file, mesh->getMeshBuffer(0), vtxCount, faceCount, idxT, indices, forceFaces, vaidToWrite, _params);

    if (needToFreeIndices)
        _IRR_ALIGNED_FREE(indices);

	return true;
}

void CPLYMeshWriter::writeBinary(io::IWriteFile* _file, asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4], const SAssetWriteParams& _params) const
{
    const size_t colCpa = asset::getFormatChannelCount(_mbuf->getMeshDataAndFormat()->getAttribFormat(asset::EVAI_ATTR1));

	bool flipVectors = (!(_params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED)) ? true : false;

    asset::ICPUMeshBuffer* mbCopy = createCopyMBuffNormalizedReplacedWithTrueInt(_mbuf);
    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf f;
        uint32_t ui[4];
        if (_vaidToWrite[asset::EVAI_ATTR0])
        {
            writeAttribBinary(_file, mbCopy, asset::EVAI_ATTR0, i, 3u, flipVectors);
            /*_mbuf->getAttribute(f, EVAI_ATTR0, i);
            _file->write(f.pointer, 3*sizeof(float));*/
        }
        if (_vaidToWrite[asset::EVAI_ATTR1])
        {
            writeAttribBinary(_file, mbCopy, asset::EVAI_ATTR1, i, colCpa);
        }
        if (_vaidToWrite[asset::EVAI_ATTR2])
        {
            writeAttribBinary(_file, mbCopy, asset::EVAI_ATTR2, i, 2u);
        }
        if (_vaidToWrite[asset::EVAI_ATTR3])
        {
            writeAttribBinary(_file, mbCopy, asset::EVAI_ATTR3, i, 3u, flipVectors);
        }
    }
    mbCopy->drop();
    mbCopy = nullptr;

    const uint8_t listSize = 3u;
    void* indices = _indices;
    if (_forceFaces)
    {
        indices = _IRR_ALIGNED_MALLOC((_idxType == asset::EIT_32BIT ? 4 : 2) * listSize * _fcCount,_IRR_SIMD_ALIGNMENT);
        if (_idxType == asset::EIT_16BIT)
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
    if (_idxType == asset::EIT_32BIT)
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
        _IRR_ALIGNED_FREE(indices);
}

void CPLYMeshWriter::writeText(io::IWriteFile* _file, asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4], const SAssetWriteParams& _params) const
{
    asset::ICPUMeshBuffer* mbCopy = createCopyMBuffNormalizedReplacedWithTrueInt(_mbuf);

    auto writefunc = [&_file,&mbCopy, &_params, this](asset::E_VERTEX_ATTRIBUTE_ID _vaid, size_t _ix, size_t _cpa)
    {
		bool flipVerteciesAndNormals = false;
		if (!(_params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
			if(_vaid == asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR0 || _vaid == asset::E_VERTEX_ATTRIBUTE_ID::EVAI_ATTR3)
				flipVerteciesAndNormals = true;

        uint32_t ui[4];
        core::vectorSIMDf f;
        const asset::E_FORMAT t = mbCopy->getMeshDataAndFormat()->getAttribFormat(_vaid);
        if (asset::isScaledFormat(t) || asset::isIntegerFormat(t))
        {
            mbCopy->getAttribute(ui, _vaid, _ix);
            if (!asset::isSignedFormat(t))
                writeVectorAsText(_file, ui, _cpa, flipVerteciesAndNormals);
            else
            {
                int32_t ii[4];
                memcpy(ii, ui, 4*4);
                writeVectorAsText(_file, ii, _cpa, flipVerteciesAndNormals);
            }
        }
        else
        {
            mbCopy->getAttribute(f, _vaid, _ix);
            writeVectorAsText(_file, f.pointer, _cpa, flipVerteciesAndNormals);
        }
    };

    const size_t colCpa = asset::getFormatChannelCount(_mbuf->getMeshDataAndFormat()->getAttribFormat(asset::EVAI_ATTR1));

    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf f;
        uint32_t ui[4];
        if (_vaidToWrite[asset::EVAI_ATTR0])
        {
            writefunc(asset::EVAI_ATTR0, i, 3u);
        }
        if (_vaidToWrite[asset::EVAI_ATTR1])
        {
            writefunc(asset::EVAI_ATTR1, i, colCpa);
        }
        if (_vaidToWrite[asset::EVAI_ATTR2])
        {
            writefunc(asset::EVAI_ATTR2, i, 2u);
        }
        if (_vaidToWrite[asset::EVAI_ATTR3])
        {
            writefunc(asset::EVAI_ATTR3, i, 3u);
        }
        _file->write("\n", 1);
    }
    mbCopy->drop();
    mbCopy = nullptr;

    const char* listSize = "3 ";
    void* indices = _indices;
    if (_forceFaces)
    {
        indices = _IRR_ALIGNED_MALLOC((_idxType == asset::EIT_32BIT ? 4 : 2) * 3 * _fcCount,_IRR_SIMD_ALIGNMENT);
        if (_idxType == asset::EIT_16BIT)
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
    if (_idxType == asset::EIT_32BIT)
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
        _IRR_ALIGNED_FREE(indices);
}

void CPLYMeshWriter::writeAttribBinary(io::IWriteFile* _file, asset::ICPUMeshBuffer* _mbuf, asset::E_VERTEX_ATTRIBUTE_ID _vaid, size_t _ix, size_t _cpa, bool flipVectors) const
{
    uint32_t ui[4];
    core::vectorSIMDf f;
    asset::E_FORMAT t = _mbuf->getMeshDataAndFormat()->getAttribFormat(_vaid);
    if (asset::isScaledFormat(t) || asset::isIntegerFormat(t))
    {
		constexpr size_t xID = 0u;
        _mbuf->getAttribute(ui, _vaid, _ix);
		if (flipVectors)
			ui[xID] = -ui[xID];

        const uint32_t bytesPerCh = asset::getTexelOrBlockBytesize(t)/asset::getFormatChannelCount(t);
        if (bytesPerCh == 1u || t == asset::EF_A2B10G10R10_UINT_PACK32 || t == asset::EF_A2B10G10R10_SINT_PACK32 || t == asset::EF_A2B10G10R10_SSCALED_PACK32 || t == asset::EF_A2B10G10R10_USCALED_PACK32)
        {
            uint8_t a[4];
            for (uint32_t k = 0u; k < _cpa; ++k)
                a[k] = ui[k];
            _file->write(a, _cpa);
        }
        else if (bytesPerCh == 2u)
        {
            uint16_t a[4];
            for (uint32_t k = 0u; k < _cpa; ++k)
                a[k] = ui[k];
            _file->write(a, 2*_cpa);
        }
        else if (bytesPerCh == 4u)
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

asset::ICPUMeshBuffer* CPLYMeshWriter::createCopyMBuffNormalizedReplacedWithTrueInt(const asset::ICPUMeshBuffer* _mbuf)
{
    asset::ICPUMeshBuffer* mbCopy = new asset::ICPUMeshBuffer();
    auto origDesc = _mbuf->getMeshDataAndFormat();
    auto desc = core::make_smart_refctd_ptr<asset::ICPUMeshDataFormatDesc>();
    for (size_t i = asset::EVAI_ATTR0; i < asset::EVAI_COUNT; ++i)
    {
        asset::E_VERTEX_ATTRIBUTE_ID vaid = (asset::E_VERTEX_ATTRIBUTE_ID)i;
        asset::E_FORMAT t = origDesc->getAttribFormat(vaid);
        if (origDesc->getMappedBuffer(vaid))
        {
            desc->setVertexAttrBuffer(
                core::smart_refctd_ptr<asset::ICPUBuffer>(const_cast<asset::ICPUBuffer*>(origDesc->getMappedBuffer(vaid))),
                vaid,
                asset::isNormalizedFormat(t) ? impl::getCorrespondingIntegerFormat(t) : t,
                origDesc->getMappedBufferStride(vaid),
                origDesc->getMappedBufferOffset(vaid),
                origDesc->getAttribDivisor(vaid)
            );
        }
    }
    mbCopy->setMeshDataAndFormat(std::move(desc));

    mbCopy->setBaseVertex(_mbuf->getBaseVertex());
    mbCopy->setIndexCount(_mbuf->getIndexCount());
    mbCopy->setBaseInstance(_mbuf->getBaseInstance());
    mbCopy->setInstanceCount(_mbuf->getInstanceCount());
    mbCopy->setIndexBufferOffset(_mbuf->getIndexBufferOffset());
    mbCopy->setIndexType(_mbuf->getIndexType());
    mbCopy->setPrimitiveType(_mbuf->getPrimitiveType());
    mbCopy->setPositionAttributeIx(_mbuf->getPositionAttributeIx());
	mbCopy->setNormalnAttributeIx(_mbuf->getNormalAttributeIx());
    return mbCopy;
}

std::string CPLYMeshWriter::getTypeString(asset::E_FORMAT _t)
{
    using namespace asset;

    if (isFloatingPointFormat(_t))
        return "float";

    switch (_t)
    {
    case EF_R8_SNORM:
    case EF_R8_SINT:
    case EF_R8_SSCALED:
    case EF_R8G8_SNORM:
    case EF_R8G8_SINT:
    case EF_R8G8_SSCALED:
    case EF_R8G8B8_SNORM:
    case EF_R8G8B8_SINT:
    case EF_R8G8B8_SSCALED:
    case EF_R8G8B8A8_SNORM:
    case EF_R8G8B8A8_SINT:
    case EF_R8G8B8A8_SSCALED:
    case EF_B8G8R8A8_UNORM:
    case EF_A2B10G10R10_SNORM_PACK32:
    case EF_A2B10G10R10_SINT_PACK32:
    case EF_A2B10G10R10_SSCALED_PACK32:
    case EF_A2R10G10B10_SNORM_PACK32:
        return "char";

    case EF_R8_UNORM:
    case EF_R8_UINT:
    case EF_R8_USCALED:
    case EF_R8G8_UNORM:
    case EF_R8G8_UINT:
    case EF_R8G8_USCALED:
    case EF_R8G8B8_UNORM:
    case EF_R8G8B8_UINT:
    case EF_R8G8B8_USCALED:
    case EF_R8G8B8A8_UNORM:
    case EF_R8G8B8A8_UINT:
    case EF_R8G8B8A8_USCALED:
    case EF_A2R10G10B10_UNORM_PACK32:
    case EF_A2B10G10R10_UNORM_PACK32:
    case EF_A2B10G10R10_UINT_PACK32:
    case EF_A2B10G10R10_USCALED_PACK32:
        return "uchar";

    case EF_R16_UNORM:
    case EF_R16_UINT:
    case EF_R16_USCALED:
    case EF_R16G16_UNORM:
    case EF_R16G16_UINT:
    case EF_R16G16_USCALED:
    case EF_R16G16B16_UNORM:
    case EF_R16G16B16_UINT:
    case EF_R16G16B16_USCALED:
    case EF_R16G16B16A16_UNORM:
    case EF_R16G16B16A16_UINT:
    case EF_R16G16B16A16_USCALED:
        return "ushort";

    case EF_R16_SNORM:
    case EF_R16_SINT:
    case EF_R16_SSCALED:
    case EF_R16G16_SNORM:
    case EF_R16G16_SINT:
    case EF_R16G16_SSCALED:
    case EF_R16G16B16_SNORM:
    case EF_R16G16B16_SINT:
    case EF_R16G16B16_SSCALED:
    case EF_R16G16B16A16_SNORM:
    case EF_R16G16B16A16_SINT:
    case EF_R16G16B16A16_SSCALED:
        return "short";

    case EF_R32_UINT:
    case EF_R32G32_UINT:
    case EF_R32G32B32_UINT:
    case EF_R32G32B32A32_UINT:
        return "uint";

    case EF_R32_SINT:
    case EF_R32G32_SINT:
    case EF_R32G32B32_SINT:
    case EF_R32G32B32A32_SINT:
        return "int";

    default:
        return "";
    }
}

} // end namespace
} // end namespace

#endif // _IRR_COMPILE_WITH_PLY_WRITER_


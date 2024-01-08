// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"
#include "nbl/asset/utils/CMeshManipulator.h"

namespace nbl
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
	#ifdef _NBL_DEBUG
	setDebugName("CPLYMeshWriter");
	#endif
}

//! writes a mesh
bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext inCtx{ _params, _file };

    const asset::ICPUMesh* mesh = IAsset::castDown<const ICPUMesh>(_params.rootAsset);
    if (!mesh)
        return false;

    system::IFile* file = _override->getOutputFile(_file, inCtx, {mesh, 0u});

    auto meshbuffers = mesh->getMeshBuffers();
	if (!file || !mesh)
		return false;

    SContext context = { SAssetWriteContext{ inCtx.params, file} };
    
    if (meshbuffers.size() > 1)
    {
        #ifdef  _NBL_DEBUG
        context.writeContext.params.logger.log("PLY WRITER WARNING (" + std::to_string(__LINE__) + " line): Only one meshbuffer input is allowed for writing! Saving first one", system::ILogger::ELL_WARNING, file->getFileName().string().c_str());
        #endif // _NBL_DEBUG
    }

    context.writeContext.params.logger.log("Writing PLY mesh", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(context.writeContext, mesh, 0u);

    auto getConvertedCpuMeshBufferWithIndexBuffer = [&]() -> core::smart_refctd_ptr<asset::ICPUMeshBuffer>
    {
        auto inputMeshBuffer = *meshbuffers.begin();
        const bool doesItHaveIndexBuffer = inputMeshBuffer->getIndexBufferBinding().buffer.get();
        const bool isItNotTriangleListsPrimitive = inputMeshBuffer->getPipeline()->getCachedCreationParams().primitiveAssembly.primitiveType != asset::EPT_TRIANGLE_LIST;
        
        if (doesItHaveIndexBuffer && isItNotTriangleListsPrimitive)
        {
            auto cpuConvertedMeshBuffer = core::smart_refctd_ptr_static_cast<asset::ICPUMeshBuffer>(inputMeshBuffer->clone());
            IMeshManipulator::homogenizePrimitiveTypeAndIndices(&cpuConvertedMeshBuffer, &cpuConvertedMeshBuffer + 1, asset::EPT_TRIANGLE_LIST, asset::EIT_32BIT);
            return cpuConvertedMeshBuffer;
        }
        else
            return nullptr;
    };

    const auto cpuConvertedMeshBufferWithIndexBuffer = getConvertedCpuMeshBufferWithIndexBuffer();
    const asset::ICPUMeshBuffer* rawCopyMeshBuffer = cpuConvertedMeshBufferWithIndexBuffer.get() ? cpuConvertedMeshBufferWithIndexBuffer.get() : *meshbuffers.begin();
    const bool doesItUseIndexBufferBinding = (rawCopyMeshBuffer->getIndexBufferBinding().buffer.get() && rawCopyMeshBuffer->getIndexType() != asset::EIT_UNKNOWN);

    uint32_t faceCount = {}; 
    size_t vertexCount = {};

    void* indices = nullptr;
    {
        auto indexCount = rawCopyMeshBuffer->getIndexCount();

        indices = _NBL_ALIGNED_MALLOC(indexCount * sizeof(uint32_t), _NBL_SIMD_ALIGNMENT);
        memcpy(indices, rawCopyMeshBuffer->getIndices(), indexCount * sizeof(uint32_t));
        
        IMeshManipulator::getPolyCount(faceCount, rawCopyMeshBuffer);
        vertexCount = IMeshManipulator::upperBoundVertexID(rawCopyMeshBuffer);
    }

	// write PLY header
    std::string header = "ply\n";
    header += (flags & asset::EWF_BINARY) ? "format binary_little_endian 1.0" : "format ascii 1.0";
	header += "\ncomment IrrlichtBAW ";
	header +=  NABLA_SDK_VERSION;

	// vertex definition
	header += "\nelement vertex ";
	header += std::to_string(vertexCount) + '\n';

    bool vaidToWrite[4]{ 0, 0, 0, 0 };

    const uint32_t POSITION_ATTRIBUTE = rawCopyMeshBuffer->getPositionAttributeIx();
    constexpr uint32_t COLOR_ATTRIBUTE = 1;
    constexpr uint32_t UV_ATTRIBUTE = 2;
    const uint32_t NORMAL_ATTRIBUTE = rawCopyMeshBuffer->getNormalAttributeIx();

    if (rawCopyMeshBuffer->getAttribBoundBuffer(POSITION_ATTRIBUTE).buffer)
    {
        const asset::E_FORMAT t = rawCopyMeshBuffer->getAttribFormat(POSITION_ATTRIBUTE);
        std::string typeStr = getTypeString(t);
        vaidToWrite[0] = true;
        header +=
            "property " + typeStr + " x\n" +
            "property " + typeStr + " y\n" +
            "property " + typeStr + " z\n";
    }
    if (rawCopyMeshBuffer->getAttribBoundBuffer(COLOR_ATTRIBUTE).buffer)
    {
        const asset::E_FORMAT t = rawCopyMeshBuffer->getAttribFormat(COLOR_ATTRIBUTE);
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
    if (rawCopyMeshBuffer->getAttribBoundBuffer(UV_ATTRIBUTE).buffer)
    {
        const asset::E_FORMAT t = rawCopyMeshBuffer->getAttribFormat(UV_ATTRIBUTE);
        std::string typeStr = getTypeString(t);
        vaidToWrite[2] = true;
        header +=
            "property " + typeStr + " u\n" +
            "property " + typeStr + " v\n";
    }
    if (rawCopyMeshBuffer->getAttribBoundBuffer(NORMAL_ATTRIBUTE).buffer)
    {
        const asset::E_FORMAT t = rawCopyMeshBuffer->getAttribFormat(NORMAL_ATTRIBUTE);
        std::string typeStr = getTypeString(t);
        vaidToWrite[3] = true;
        header +=
            "property " + typeStr + " nx\n" +
            "property " + typeStr + " ny\n" +
            "property " + typeStr + " nz\n";
    }    

    asset::E_INDEX_TYPE idxT = asset::EIT_UNKNOWN;
    bool forceFaces = false;

    const auto primitiveType = rawCopyMeshBuffer->getPipeline()->getCachedCreationParams().primitiveAssembly.primitiveType;
    const auto indexType = rawCopyMeshBuffer->getIndexType();
  
    if (primitiveType == asset::EPT_POINT_LIST)
        faceCount = 0u;
    else if (doesItUseIndexBufferBinding)
    {
        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        idxT = indexType;
        const std::string idxTypeStr = idxT == asset::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxTypeStr + " vertex_indices\n";
    }
    else if (primitiveType == asset::EPT_TRIANGLE_LIST)
    {
        forceFaces = true;

        header += "element face ";
        header += std::to_string(faceCount) + '\n';
        idxT = vertexCount <= ((1u<<16) - 1) ? asset::EIT_16BIT : asset::EIT_32BIT;
        const std::string idxTypeStr = idxT == asset::EIT_32BIT ? "uint32" : "uint16";
        header += "property list uchar " + idxTypeStr + " vertex_indices\n";
    }
    else
        faceCount = 0u;
    header += "end_header\n";

    {
        system::IFile::success_t success;
        file->write(success, header.c_str(), context.fileOffset, header.size());
        context.fileOffset += success.getBytesProcessed();
    }
 
    if (flags & asset::EWF_BINARY)
        writeBinary(rawCopyMeshBuffer, vertexCount, faceCount, idxT, indices, forceFaces, vaidToWrite, context);
    else
        writeText(rawCopyMeshBuffer, vertexCount, faceCount, idxT, indices, forceFaces, vaidToWrite, context);

    _NBL_ALIGNED_FREE(const_cast<void*>(indices));

	return true;
}

void CPLYMeshWriter::writeBinary(const asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4], SContext& context) const
{
    const size_t colCpa = asset::getFormatChannelCount(_mbuf->getAttribFormat(1));

	bool flipVectors = (!(context.writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED)) ? true : false;

    auto mbCopy = createCopyMBuffNormalizedReplacedWithTrueInt(_mbuf);
    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf f;
        uint32_t ui[4];
        if (_vaidToWrite[0])
        {
            writeAttribBinary(context, mbCopy.get(), 0, i, 3u, flipVectors);
        }
        if (_vaidToWrite[1])
        {
            writeAttribBinary(context, mbCopy.get(), 1, i, colCpa);
        }
        if (_vaidToWrite[2])
        {
            writeAttribBinary(context, mbCopy.get(), 2, i, 2u);
        }
        if (_vaidToWrite[3])
        {
            writeAttribBinary(context, mbCopy.get(), 3, i, 3u, flipVectors);
        }
    }

    constexpr uint8_t listSize = 3u;
    void* indices = _indices;
    if (_forceFaces)
    {
        indices = _NBL_ALIGNED_MALLOC((_idxType == asset::EIT_32BIT ? 4 : 2) * listSize * _fcCount,_NBL_SIMD_ALIGNMENT);
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
            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, &listSize, context.fileOffset, sizeof(listSize));
                context.fileOffset += success.getBytesProcessed();
            }

            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, ind, context.fileOffset, listSize * 4);
                context.fileOffset += success.getBytesProcessed();
            }

            ind += listSize;
        }
    }
    else
    {
        uint16_t* ind = (uint16_t*)indices;
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, &listSize, context.fileOffset, sizeof(listSize));
                context.fileOffset += success.getBytesProcessed();
            }

            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, ind, context.fileOffset, listSize * 2);
                context.fileOffset += success.getBytesProcessed();
            }
            
            ind += listSize;
        }
    }

    if (_forceFaces)
        _NBL_ALIGNED_FREE(indices);
}

void CPLYMeshWriter::writeText(const asset::ICPUMeshBuffer* _mbuf, size_t _vtxCount, size_t _fcCount, asset::E_INDEX_TYPE _idxType, void* const _indices, bool _forceFaces, const bool _vaidToWrite[4], SContext& context) const
{
    auto mbCopy = createCopyMBuffNormalizedReplacedWithTrueInt(_mbuf);

    auto writefunc = [&context, &mbCopy, this](uint32_t _vaid, size_t _ix, size_t _cpa)
    {
		bool flipVerteciesAndNormals = false;
		if (!(context.writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
			if(_vaid == 0u || _vaid == 3u)
				flipVerteciesAndNormals = true;

        uint32_t ui[4];
        core::vectorSIMDf f;
        const asset::E_FORMAT t = mbCopy->getAttribFormat(_vaid);
        if (asset::isScaledFormat(t) || asset::isIntegerFormat(t))
        {
            mbCopy->getAttribute(ui, _vaid, _ix);
            if (!asset::isSignedFormat(t))
                writeVectorAsText(context, ui, _cpa, flipVerteciesAndNormals);
            else
            {
                int32_t ii[4];
                memcpy(ii, ui, 4*4);
                writeVectorAsText(context, ii, _cpa, flipVerteciesAndNormals);
            }
        }
        else
        {
            mbCopy->getAttribute(f, _vaid, _ix);
            writeVectorAsText(context, f.pointer, _cpa, flipVerteciesAndNormals);
        }
    };

    const size_t colCpa = asset::getFormatChannelCount(_mbuf->getAttribFormat(1));

    for (size_t i = 0u; i < _vtxCount; ++i)
    {
        core::vectorSIMDf f;
        uint32_t ui[4];
        if (_vaidToWrite[0])
        {
            writefunc(0, i, 3u);
        }
        if (_vaidToWrite[1])
        {
            writefunc(1, i, colCpa);
        }
        if (_vaidToWrite[2])
        {
            writefunc(2, i, 2u);
        }
        if (_vaidToWrite[3])
        {
            writefunc(3, i, 3u);
        }

        {
            system::IFile::success_t success;
            context.writeContext.outputFile->write(success, "\n", context.fileOffset, 1);
            context.fileOffset += success.getBytesProcessed();
        }
    }

    const char* listSize = "3 ";
    void* indices = _indices;
    if (_forceFaces)
    {
        indices = _NBL_ALIGNED_MALLOC((_idxType == asset::EIT_32BIT ? 4 : 2) * 3 * _fcCount,_NBL_SIMD_ALIGNMENT);
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
            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, listSize, context.fileOffset, 2);
                context.fileOffset += success.getBytesProcessed();
            }

            writeVectorAsText(context, ind, 3);

            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, "\n", context.fileOffset, 1);
                context.fileOffset += success.getBytesProcessed();
            }

            ind += 3;
        }
    }
    else
    {
        uint16_t* ind = (uint16_t*)indices;
        for (size_t i = 0u; i < _fcCount; ++i)
        {
            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, listSize, context.fileOffset, 2);
                context.fileOffset += success.getBytesProcessed();
            }

            writeVectorAsText(context, ind, 3);

            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, "\n", context.fileOffset, 1);
                context.fileOffset += success.getBytesProcessed();
            }

            ind += 3;
        }
    }

    if (_forceFaces)
        _NBL_ALIGNED_FREE(indices);
}

void CPLYMeshWriter::writeAttribBinary(SContext& context, asset::ICPUMeshBuffer* _mbuf, uint32_t _vaid, size_t _ix, size_t _cpa, bool flipAttribute) const
{
    uint32_t ui[4];
    core::vectorSIMDf f;
    asset::E_FORMAT t = _mbuf->getAttribFormat(_vaid);

    if (asset::isScaledFormat(t) || asset::isIntegerFormat(t))
    {
        _mbuf->getAttribute(ui, _vaid, _ix);
        if (flipAttribute)
            ui[0] = -ui[0];

        const uint32_t bytesPerCh = asset::getTexelOrBlockBytesize(t)/asset::getFormatChannelCount(t);
        if (bytesPerCh == 1u || t == asset::EF_A2B10G10R10_UINT_PACK32 || t == asset::EF_A2B10G10R10_SINT_PACK32 || t == asset::EF_A2B10G10R10_SSCALED_PACK32 || t == asset::EF_A2B10G10R10_USCALED_PACK32)
        {
            uint8_t a[4];
            for (uint32_t k = 0u; k < _cpa; ++k)
                a[k] = ui[k];

            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, a, context.fileOffset, _cpa);
                context.fileOffset += success.getBytesProcessed();
            }
        }
        else if (bytesPerCh == 2u)
        {
            uint16_t a[4];
            for (uint32_t k = 0u; k < _cpa; ++k)
                a[k] = ui[k];

            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, a, context.fileOffset, 2 * _cpa);
                context.fileOffset += success.getBytesProcessed();
            }
        }
        else if (bytesPerCh == 4u)
        {
            {
                system::IFile::success_t success;
                context.writeContext.outputFile->write(success, ui, context.fileOffset, 4 * _cpa);
                context.fileOffset += success.getBytesProcessed();
            }
        }
    }
    else
    {
        _mbuf->getAttribute(f, _vaid, _ix);
        if (flipAttribute)
            f[0] = -f[0];

        {
            system::IFile::success_t success;
            context.writeContext.outputFile->write(success, f.pointer, context.fileOffset, 4 * _cpa);
            context.fileOffset += success.getBytesProcessed();
        }
    }
}

core::smart_refctd_ptr<asset::ICPUMeshBuffer> CPLYMeshWriter::createCopyMBuffNormalizedReplacedWithTrueInt(const asset::ICPUMeshBuffer* _mbuf)
{
    auto mbCopy = core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(_mbuf->clone(2));

    for (size_t i = 0; i < ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT; ++i)
    {
        auto vaid = i;
        asset::E_FORMAT t = _mbuf->getAttribFormat(vaid);
    
        if (_mbuf->getAttribBoundBuffer(vaid).buffer)
            mbCopy->getPipeline()->getCachedCreationParams().vertexInput.attributes[vaid].format = asset::isNormalizedFormat(t) ? impl::getCorrespondingIntegerFormat(t) : t;
    }

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

#endif // _NBL_COMPILE_WITH_PLY_WRITER_


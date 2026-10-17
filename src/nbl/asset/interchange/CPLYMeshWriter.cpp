// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CPLYMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

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

   const asset::ICPUPolygonGeometry* geom = IAsset::castDown<const ICPUPolygonGeometry>(_params.rootAsset);
   if (!geom)
      return false;

   system::IFile* file = _override->getOutputFile(_file, inCtx, {geom, 0u});
   if (!file || !geom)
      return false;

   SContext context = { SAssetWriteContext{ inCtx.params, file} };

   context.writeContext.params.logger.log("Writing PLY file", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

   const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(context.writeContext, geom, 0u);

   const auto& idxView = geom->getIndexView();
   const auto& vtxView = geom->getPositionView();
   const auto& normalView = geom->getNormalView();
   const auto& auxAttributes = geom->getAuxAttributeViews();

   // TODO: quads?
   const size_t faceCount = geom->getIndexCount() / 3;

   // indices:
   // 0 for vertices (always enabled)
   // 1 for normals (optional)
   // 2 for vertex colors (optional)
   // 3 for texcoords (optional)
   bool attrsToWrite[4] = { true, false, false, false };

   // write PLY header
   std::string header = "ply\n";
   header += (flags & asset::EWF_BINARY) ? "format binary_little_endian 1.0" : "format ascii 1.0";
   header += "\ncomment Nabla ";
   header +=  NABLA_SDK_VERSION;

   // vertex definition
   header += "\nelement vertex ";
   header += std::to_string(vtxView.getElementCount()) + '\n';

   std::string typeStr = getTypeString(vtxView.composed.format);
   header += "property " + typeStr + " x\n" +
      "property " + typeStr + " y\n" +
      "property " + typeStr + " z\n";

   if (normalView.getElementCount() > 0)
   {
      attrsToWrite[1] = true;
      std::string typeStr = getTypeString(normalView.composed.format);
      header += "property " + typeStr + " nx\n" +
         "property " + typeStr + " ny\n" +
         "property " + typeStr + " nz\n";
   }

   header += "element face " + std::to_string(faceCount) + '\n';
   header += "property list uchar " + getTypeString(idxView.composed.format) + " vertex_index\n";

   if (auxAttributes.size() > 0)
   for (auto& attr : auxAttributes)
   {
      // TODO: Texcoords, vertex colors and any additional properties
   }
  
   header += "end_header\n";
   
   {
      system::IFile::success_t success;
      context.writeContext.outputFile->write(success, header.data(), context.fileOffset, header.size());
      context.fileOffset += success.getBytesProcessed();
   }

   if (flags & asset::EWF_BINARY)
      writeBinary(geom, attrsToWrite, true, context);
   else
      writeText(geom, attrsToWrite, true, context);

   return true;
}

void CPLYMeshWriter::writeBinary(const ICPUPolygonGeometry* geom, const bool attrsToWrite[4], bool _forceFaces, SContext& context) const
{
#if 0
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
#endif
}

void CPLYMeshWriter::writeText(const ICPUPolygonGeometry* geom, const bool attrsToWrite[4], bool _forceFaces, SContext& context) const
{
   auto& posView = geom->getPositionView();
   auto& normalView = geom->getNormalView();
   auto& idxView = geom->getIndexView();

   std::span<const hlsl::float32_t3> positions(
      reinterpret_cast<const hlsl::float32_t3*>(posView.getPointer()), 
      reinterpret_cast<const hlsl::float32_t3*>(posView.getPointer(posView.getElementCount()))
   );

   const auto* idxBegPtr = reinterpret_cast<const uint32_t*>(idxView.getPointer());
   std::span<const uint32_t> indices(idxBegPtr, idxBegPtr + idxView.getElementCount());

   for (size_t i = 0; i < positions.size(); i++)
      writeVertexAsText(context, positions[i], (attrsToWrite[1] ? reinterpret_cast<const hlsl::float32_t3*>(normalView.getPointer(i)) : nullptr));

   // write indices
   for (size_t i = 0; i < indices.size(); i += 3)
   {
      std::string str = "3 ";
      for (size_t j = 0; j < 3; j++)
         str += std::to_string(indices[i + j]) + " ";
      str += "\n";

      system::IFile::success_t success;
      context.writeContext.outputFile->write(success, str.data(), context.fileOffset, str.size());
      context.fileOffset += success.getBytesProcessed();
   }
}

void CPLYMeshWriter::writeAttribBinary(SContext& context, ICPUPolygonGeometry* geom, uint32_t _vaid, size_t _ix, size_t _cpa, bool flipAttribute) const
{
#if 0
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
#endif
}

core::smart_refctd_ptr<ICPUPolygonGeometry> CPLYMeshWriter::createCopyNormalizedReplacedWithTrueInt(const ICPUPolygonGeometry* geom)
{
#if 0
    auto mbCopy = core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(_mbuf->clone(2));

    for (size_t i = 0; i < ICPUMeshBuffer::MAX_VERTEX_ATTRIB_COUNT; ++i)
    {
        auto vaid = i;
        asset::E_FORMAT t = _mbuf->getAttribFormat(vaid);
    
        if (_mbuf->getAttribBoundBuffer(vaid).buffer)
            mbCopy->getPipeline()->getCachedCreationParams().vertexInput.attributes[vaid].format = asset::isNormalizedFormat(t) ? impl::getCorrespondingIntegerFormat(t) : t;
    }

    return mbCopy;
#endif
    return nullptr;
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


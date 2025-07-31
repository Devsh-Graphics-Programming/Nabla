// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

#include "CSTLMeshWriter.h"
#include "SColor.h"

using namespace nbl;
using namespace nbl::asset;

#ifdef _NBL_COMPILE_WITH_STL_WRITER_
constexpr auto POSITION_ATTRIBUTE = 0;
constexpr auto COLOR_ATTRIBUTE = 1;
constexpr auto UV_ATTRIBUTE = 2;
constexpr auto NORMAL_ATTRIBUTE = 3;

CSTLMeshWriter::CSTLMeshWriter()
{
	#ifdef _NBL_DEBUG
	setDebugName("CSTLMeshWriter");
	#endif
}

CSTLMeshWriter::~CSTLMeshWriter()
{

}

//! writes a mesh
bool CSTLMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
    if (!_override)
        getDefaultOverride(_override);

    SAssetWriteContext inCtx{_params, _file};

    const asset::ICPUPolygonGeometry* mesh =
#   ifndef _NBL_DEBUG
        static_cast<const asset::ICPUPolygonGeometry*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::ICPUPolygonGeometry*>(_params.rootAsset);
#   endif
    assert(mesh);

	system::IFile* file = _override->getOutputFile(_file, inCtx, {mesh, 0u});

	if (!file)
		return false;

	SContext context = { SAssetWriteContext{ inCtx.params, file} };

	_params.logger.log("WRITING STL: writing the file %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

    const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(context.writeContext, mesh, 0u);
	if (flags & asset::EWF_BINARY)
		return writeMeshBinary(mesh, &context);
	else
		return writeMeshASCII(mesh, &context);
}

inline static hlsl::float32_t3 calculateNormal(const hlsl::float32_t3& p1, const hlsl::float32_t3& p2, const hlsl::float32_t3& p3)
{
	return hlsl::normalize(hlsl::cross(p2 - p1, p3 - p1));
}

namespace
{
template <typename IndexType>
inline void writeFacesBinary(const asset::ICPUPolygonGeometry* geom, const bool& noIndices, system::IFile* file, uint32_t _colorVaid, IAssetWriter::SAssetWriteContext* context, size_t* fileOffset)
{
	bool hasColor = inputParams.enabledAttribFlags & core::createBitmask({ COLOR_ATTRIBUTE });
	const asset::E_FORMAT colorType = static_cast<asset::E_FORMAT>(hasColor ? inputParams.attributes[COLOR_ATTRIBUTE].format : asset::EF_UNKNOWN);

	auto& posView = geom->getPositionView();
	auto& normalView = geom->getNormalView();
	auto& idxView = geom->getIndexView();
	
	const auto vertexCount = posView.getElementCount();
	const auto idxCount = idxView.getElementCount();

	// TODO: check if I can actually assume following types, if not, handle that
	const uint32_t* idxBufPtr = reinterpret_cast<const uint32_t*>(idxView.getPointer());
	const pos_t* vtxBufPtr = reinterpret_cast<const pos_t*>(posView.getPointer());

	for (size_t i = 0; i < idxCount; i+=3)
	{
		IndexType idx[3] = {};
		for (size_t j = 0; j < 3; j++)
			idx[i] = *(idxBufPtr + j + i);

		pos_t pos[3] = {};
		for (size_t j = 0; j < 3; j++)
			pos[j] = *(vtxBufPtr + idx[j]);

		// TODO: vertex color
		// TODO: I think I could get the normal from normalView, but I need to think how can I do that well

		normal_t n = calculateNormal(pos[1], pos[2], pos[3]);

		// success variable can be reused, no need to scope it
		system::IFile::success_t success{};

		// write normal
		file->write(success, &normal, *fileOffset, 12);
		*fileOffset += success.getBytesProcessed();

		// write positions
		for (size_t j = 0; j < 3; j++)
		{
			file->write(success, &pos[i], *fileOffset, 12);
			*fileOffset += success.getBytesProcessed();
		}
	}
#if 0
        uint16_t color = 0u;
        if (hasColor)
        {
            if (asset::isIntegerFormat(colorType))
            {
                uint32_t res[4];
                for (uint32_t i = 0u; i < 3u; ++i)
                {
                    uint32_t d[4];
                    buffer->getAttribute(d, _colorVaid, idx[i]);
                    res[0] += d[0]; res[1] += d[1]; res[2] += d[2];
                }
                color = video::RGB16(res[0]/3, res[1]/3, res[2]/3);
            }
            else
            {
                core::vectorSIMDf res;
                for (uint32_t i = 0u; i < 3u; ++i)
                {
                    core::vectorSIMDf d;
                    buffer->getAttribute(d, _colorVaid, idx[i]);
                    res += d;
                }
                res /= 3.f;
                color = video::RGB16(res.X, res.Y, res.Z);
            }
        }

		{
			system::IFile::success_t success;;
			file->write(success, &color, *fileOffset, 2); // saving color using non-standard VisCAM/SolidView trick
	
			*fileOffset += success.getBytesProcessed();
		}
#endif
}
}

bool CSTLMeshWriter::writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	// write STL MESH header
	const char headerTxt[] = "Irrlicht-baw Engine";
	constexpr size_t HEADER_SIZE = 80u;

	system::IFile::success_t success;

	context->writeContext.outputFile->write(success, headerTxt, context->fileOffset, sizeof(headerTxt));
	context->fileOffset += success.getBytesProcessed();

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string(); // TODO: check it
	const int32_t sizeleft = HEADER_SIZE - sizeof(headerTxt) - name.size();

	if (sizeleft < 0)
	{
		context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, HEADER_SIZE - sizeof(headerTxt));
		context->fileOffset += success.getBytesProcessed();
	}
	else
	{
		const char buf[80] = {0};

		context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, name.size());
		context->fileOffset += success.getBytesProcessed();

		context->writeContext.outputFile->write(success, buf, context->fileOffset, sizeleft);
		context->fileOffset += success.getBytesProcessed();
	}
	
	// TODO: is this method (writeFacesBinary) even necessary with new ICPUPolygonGeometry?
	size_t idxCount = geom->getIndexCount();
	size_t facesCount = idxCount / 3;

	if (idxCount > 0)
	{
		auto& idxView = geom->getIndexView();
		size_t idxSize = idxView.src.size / idxCount;

		if (idxSize == sizeof(uint16_t))
			writeFacesBinary<uint16_t>(geom, false, context->writeContext.outputFile, COLOR_ATTRIBUTE, &context->writeContext, &context->fileOffset);
		else
			writeFacesBinary<uint32_t>(geom, false, context->writeContext.outputFile, COLOR_ATTRIBUTE, &context->writeContext, &context->fileOffset);
	}
	else
	{
		writeFacesBinary<uint16_t>(geom, true, context->writeContext.outputFile, COLOR_ATTRIBUTE, &context->writeContext, &context->fileOffset);
	}

	return true;
}

bool CSTLMeshWriter::writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	using pos_t = hlsl::float32_t3;

	// write STL MESH header
   const char headerTxt[] = "Nabla Engine ";

	system::IFile::success_t success;
	
	context->writeContext.outputFile->write(success, "solid ", context->fileOffset, 6);
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, headerTxt, context->fileOffset, sizeof(headerTxt) - 1);
	context->fileOffset += success.getBytesProcessed();

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();
	context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, name.size());
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "\n", context->fileOffset, 1);	
	context->fileOffset += success.getBytesProcessed();

	// TODO: what if index count is 0
	auto& idxView = geom->getIndexView();

	const size_t idxCount = geom->getIndexCount();
	const size_t facesCount = idxCount / 3;
	const size_t idxSize = idxView.src.buffer->getSize() / idxCount;

	auto& posView = geom->getPositionView();

	for (size_t i = 0; i < facesCount; i++)
	{
		pos_t positions[3] = {};
		if (idxSize == sizeof(uint16_t))
		{
			for (size_t j = 0; j < 3; j++)
			{
				uint16_t idx = *reinterpret_cast<const uint16_t*>(idxView.getPointer(i + j));
				positions[j] = *reinterpret_cast<const pos_t*>(posView.getPointer(idx));
			}
		}
		else if (idxSize == sizeof(uint32_t))
		{
			for (size_t j = 0; j < 3; j++)
			{
				uint32_t idx = *reinterpret_cast<const uint32_t*>(idxView.getPointer(i + j));
				positions[j] = *reinterpret_cast<const pos_t*>(posView.getPointer(idx));
			}
		}
		else
		{
			// TODO: what do we do with unknown index type
			assert(false);
		}

		writeFaceText(positions[0], positions[1], positions[2], context);

		context->writeContext.outputFile->write(success, "\n", context->fileOffset, 1);
		context->fileOffset += success.getBytesProcessed();
	}

	context->writeContext.outputFile->write(success, "endsolid ", context->fileOffset, 9);
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, headerTxt, context->fileOffset, sizeof(headerTxt) - 1);
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, name.size());
	context->fileOffset += success.getBytesProcessed();

	return true;
}

void CSTLMeshWriter::getVectorAsStringLine(const pos_t& v, std::string& s) const
{
    std::ostringstream tmp;
    tmp << v.x << " " << v.y << " " << v.z << "\n";
    s = std::string(tmp.str().c_str());
}

void CSTLMeshWriter::writeFaceText(
		const pos_t& v1,
		const pos_t& v2,
		const pos_t& v3,
		SContext* context)
{
	pos_t vertex1 = v3;
	pos_t vertex2 = v2;
	pos_t vertex3 = v1;
	normal_t normal = calculateNormal(vertex1, vertex2, vertex3);
	std::string tmp;

	auto flipVectors = [&]()
	{
		vertex1.x = -vertex1.x;
		vertex2.x = -vertex2.x;
		vertex3.x = -vertex3.x;
		normal_t normal = calculateNormal(vertex1, vertex2, vertex3);
	};
	
	if (!(context->writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
		flipVectors();
	
	system::IFile::success_t success;

	context->writeContext.outputFile->write(success, "facet normal ", context->fileOffset, 13);
	context->fileOffset += success.getBytesProcessed();

	getVectorAsStringLine(normal, tmp);

	context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "  outer loop\n", context->fileOffset, 13);
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "    vertex ", context->fileOffset, 11);
	context->fileOffset += success.getBytesProcessed();

	getVectorAsStringLine(vertex1, tmp);

	context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "    vertex ", context->fileOffset, 11);
	context->fileOffset += success.getBytesProcessed();

	getVectorAsStringLine(vertex2, tmp);

	context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "    vertex ", context->fileOffset, 11);
	context->fileOffset += success.getBytesProcessed();

	getVectorAsStringLine(vertex3, tmp);

	context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "  endloop\n", context->fileOffset, 10);
	context->fileOffset += success.getBytesProcessed();

	context->writeContext.outputFile->write(success, "endfacet\n", context->fileOffset, 9);
	context->fileOffset += success.getBytesProcessed();
}

#endif

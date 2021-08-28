// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CSTLMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_STL_WRITER_

#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

namespace nbl
{
namespace asset
{

using namespace io;
using namespace core;

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

    const asset::ICPUMesh* mesh =
#   ifndef _NBL_DEBUG
        static_cast<const asset::ICPUMesh*>(_params.rootAsset);
#   else
        dynamic_cast<const asset::ICPUMesh*>(_params.rootAsset);
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

namespace
{
template <class I>
inline void writeFacesBinary(const asset::ICPUMeshBuffer* buffer, const bool& noIndices, system::IFile* file, uint32_t _colorVaid, IAssetWriter::SAssetWriteContext* context, size_t* fileOffset)
{
	auto& inputParams = buffer->getPipeline()->getVertexInputParams();
	bool hasColor = inputParams.enabledAttribFlags & core::createBitmask({ COLOR_ATTRIBUTE });
    const asset::E_FORMAT colorType = static_cast<asset::E_FORMAT>(hasColor ? inputParams.attributes[COLOR_ATTRIBUTE].format : asset::EF_UNKNOWN);

    const uint32_t indexCount = buffer->getIndexCount();
    for (uint32_t j = 0u; j < indexCount; j += 3u)
    {
        I idx[3];
        for (uint32_t i = 0u; i < 3u; ++i)
        {
            if (noIndices)
                idx[i] = j + i;
            else
                idx[i] = ((I*)buffer->getIndices())[j + i];
        }

        core::vectorSIMDf v[3];
        for (uint32_t i = 0u; i < 3u; ++i)
            v[i] = buffer->getPosition(idx[i]);

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

		core::vectorSIMDf normal = core::plane3dSIMDf(v[0], v[1], v[2]).getNormal();
		core::vectorSIMDf vertex1 = v[2];
		core::vectorSIMDf vertex2 = v[1];
		core::vectorSIMDf vertex3 = v[0];

		auto flipVectors = [&]()
		{
			vertex1.X = -vertex1.X;
			vertex2.X = -vertex2.X;
			vertex3.X = -vertex3.X;
			normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
		};

		if (!(context->params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
			flipVectors();

		system::future<size_t> future;

		file->write(future, &normal, *fileOffset, 12);
		{
			const auto bytesWritten = future.get();
			*fileOffset += bytesWritten;
		}

		file->write(future, &vertex1, *fileOffset, 12);
		{
			const auto bytesWritten = future.get();
			*fileOffset += bytesWritten;
		}

		file->write(future, &vertex2, *fileOffset, 12);
		{
			const auto bytesWritten = future.get();
			*fileOffset += bytesWritten;
		}

		file->write(future, &vertex3, *fileOffset, 12);
		{
			const auto bytesWritten = future.get();
			*fileOffset += bytesWritten;
		}

		file->write(future, &color, *fileOffset, 2); // saving color using non-standard VisCAM/SolidView trick
		{
			const auto bytesWritten = future.get();
			*fileOffset += bytesWritten;
		}
    }
}
}

bool CSTLMeshWriter::writeMeshBinary(const asset::ICPUMesh* mesh, SContext* context)
{
	// write STL MESH header
    const char headerTxt[] = "Irrlicht-baw Engine";
    constexpr size_t HEADER_SIZE = 80u;

	system::future<size_t> future;
	context->writeContext.outputFile->write(future, headerTxt, context->fileOffset, sizeof(headerTxt));
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string(); // TODO: check it
	const int32_t sizeleft = HEADER_SIZE - sizeof(headerTxt) - name.size();

	if (sizeleft < 0)
	{
		context->writeContext.outputFile->write(future, name.c_str(), context->fileOffset, HEADER_SIZE - sizeof(headerTxt));
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}
	else
	{
		const char buf[80] = {0};

		context->writeContext.outputFile->write(future, name.c_str(), context->fileOffset, name.size());
		{
			const auto bytesWritten = future.get();
			context->fileOffset += bytesWritten;
		}
		
		context->writeContext.outputFile->write(future, buf, context->fileOffset, sizeleft);
		{
			const auto bytesWritten = future.get();
			context->fileOffset += bytesWritten;
		}
	}

	uint32_t facenum = 0;
	for (auto& mb : mesh->getMeshBuffers())
		facenum += mb->getIndexCount()/3;

	context->writeContext.outputFile->write(future, &facenum, context->fileOffset, sizeof(facenum));
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	// write mesh buffers

	for (auto& buffer : mesh->getMeshBuffers())
	if (buffer)
	{
        asset::E_INDEX_TYPE type = buffer->getIndexType();
		if (!buffer->getIndexBufferBinding().buffer)
            type = asset::EIT_UNKNOWN;

		if (type== asset::EIT_16BIT)
            writeFacesBinary<uint16_t>(buffer, false, context->writeContext.outputFile, COLOR_ATTRIBUTE, &context->writeContext, &context->fileOffset);
		else if (type== asset::EIT_32BIT)
            writeFacesBinary<uint32_t>(buffer, false, context->writeContext.outputFile, COLOR_ATTRIBUTE, &context->writeContext, &context->fileOffset);
		else
            writeFacesBinary<uint16_t>(buffer, true, context->writeContext.outputFile, COLOR_ATTRIBUTE, &context->writeContext, &context->fileOffset); //template param doesn't matter if there's no indices
	}
	return true;
}

bool CSTLMeshWriter::writeMeshASCII(const asset::ICPUMesh* mesh, SContext* context)
{
	// write STL MESH header
    const char headerTxt[] = "Irrlicht-baw Engine ";

	system::future<size_t> future;

	context->writeContext.outputFile->write(future, "solid ", context->fileOffset, 6);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, headerTxt, context->fileOffset, sizeof(headerTxt) - 1);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();

	context->writeContext.outputFile->write(future, name.c_str(), context->fileOffset, name.size());
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "\n", context->fileOffset, 1);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	// write mesh buffers
	for (auto& buffer : mesh->getMeshBuffers())
	if (buffer)
	{
        asset::E_INDEX_TYPE type = buffer->getIndexType();
		if (!buffer->getIndexBufferBinding().buffer)
            type = asset::EIT_UNKNOWN;
		const uint32_t indexCount = buffer->getIndexCount();
		if (type==asset::EIT_16BIT)
		{
            //os::Printer::log("Writing mesh with 16bit indices");
            for (uint32_t j=0; j<indexCount; j+=3)
            {
                writeFaceText(
                    buffer->getPosition(((uint16_t*)buffer->getIndices())[j]),
                    buffer->getPosition(((uint16_t*)buffer->getIndices())[j+1]),
                    buffer->getPosition(((uint16_t*)buffer->getIndices())[j+2]),
					context
                );
            }
		}
		else if (type==asset::EIT_32BIT)
		{
            //os::Printer::log("Writing mesh with 32bit indices");
            for (uint32_t j=0; j<indexCount; j+=3)
            {
                writeFaceText(
                    buffer->getPosition(((uint32_t*)buffer->getIndices())[j]),
                    buffer->getPosition(((uint32_t*)buffer->getIndices())[j+1]),
                    buffer->getPosition(((uint32_t*)buffer->getIndices())[j+2]),
					context
                );
            }
		}
		else
        {
            //os::Printer::log("Writing mesh with no indices");
            for (uint32_t j=0; j<indexCount; j+=3)
            {
                writeFaceText(
                    buffer->getPosition(j),
                    buffer->getPosition(j+1ul),
                    buffer->getPosition(j+2ul),
					context
                );
            }
        }
		
		context->writeContext.outputFile->write(future, "\n", context->fileOffset, 1);
		{
			const auto bytesWritten = future.get();
			context->fileOffset += bytesWritten;
		}
	}

	context->writeContext.outputFile->write(future, "endsolid ", context->fileOffset, 9);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, headerTxt, context->fileOffset, sizeof(headerTxt) - 1);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, name.c_str(), context->fileOffset, name.size());
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	return true;
}

void CSTLMeshWriter::getVectorAsStringLine(const core::vectorSIMDf& v, std::string& s) const
{
    std::ostringstream tmp;
    tmp << v.X << " " << v.Y << " " << v.Z << "\n";
    s = std::string(tmp.str().c_str());
}

void CSTLMeshWriter::writeFaceText(
		const core::vectorSIMDf& v1,
		const core::vectorSIMDf& v2,
		const core::vectorSIMDf& v3,
		SContext* context)
{
	core::vectorSIMDf vertex1 = v3;
	core::vectorSIMDf vertex2 = v2;
	core::vectorSIMDf vertex3 = v1;
	core::vectorSIMDf normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
	std::string tmp;

	auto flipVectors = [&]()
	{
		vertex1.X = -vertex1.X;
		vertex2.X = -vertex2.X;
		vertex3.X = -vertex3.X;
		normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
	};
	
	if (!(context->writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED))
		flipVectors();

	system::future<size_t> future;

	context->writeContext.outputFile->write(future, "facet normal ", context->fileOffset, 13);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	getVectorAsStringLine(normal, tmp);

	context->writeContext.outputFile->write(future, tmp.c_str(), context->fileOffset, tmp.size());
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "  outer loop\n", context->fileOffset, 13);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "    vertex ", context->fileOffset, 11);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	getVectorAsStringLine(vertex1, tmp);

	context->writeContext.outputFile->write(future, tmp.c_str(), context->fileOffset, tmp.size());
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "    vertex ", context->fileOffset, 11);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	getVectorAsStringLine(vertex2, tmp);

	context->writeContext.outputFile->write(future, tmp.c_str(), context->fileOffset, tmp.size());
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "    vertex ", context->fileOffset, 11);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	getVectorAsStringLine(vertex3, tmp);

	context->writeContext.outputFile->write(future, tmp.c_str(), context->fileOffset, tmp.size());
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "  endloop\n", context->fileOffset, 10);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}

	context->writeContext.outputFile->write(future, "endfacet\n", context->fileOffset, 9);
	{
		const auto bytesWritten = future.get();
		context->fileOffset += bytesWritten;
	}
}

} // end namespace
} // end namespace

#endif

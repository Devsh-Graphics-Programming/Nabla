// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/system/ISystem.h"
#include "nbl/system/IFile.h"

#include "CSTLMeshWriter.h"

#include <sstream>

using namespace nbl;
using namespace nbl::asset;

#ifdef _NBL_COMPILE_WITH_STL_WRITER_

CSTLMeshWriter::CSTLMeshWriter()
{
	#ifdef _NBL_DEBUG
	setDebugName("CSTLMeshWriter");
	#endif
}

CSTLMeshWriter::~CSTLMeshWriter()
{
}

bool CSTLMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	if (!_override)
		getDefaultOverride(_override);

	SAssetWriteContext inCtx{_params, _file};

	const asset::ICPUPolygonGeometry* geom =
#ifndef _NBL_DEBUG
		static_cast<const asset::ICPUPolygonGeometry*>(_params.rootAsset);
#else
		dynamic_cast<const asset::ICPUPolygonGeometry*>(_params.rootAsset);
#endif
	if (!geom)
		return false;

	system::IFile* file = _override->getOutputFile(_file, inCtx, {geom, 0u});
	if (!file)
		return false;

	SContext context = { SAssetWriteContext{ inCtx.params, file} };

	_params.logger.log("WRITING STL: writing the file %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

	const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(context.writeContext, geom, 0u);
	if (flags & asset::EWF_BINARY)
		return writeMeshBinary(geom, &context);
	return writeMeshASCII(geom, &context);
}

namespace
{
inline bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, const uint32_t primIx, core::vectorSIMDf& out0, core::vectorSIMDf& out1, core::vectorSIMDf& out2, uint32_t* outIdx)
{
	uint32_t idx[3] = {};
	const auto& indexView = geom->getIndexView();
	const void* indexBuffer = indexView ? indexView.getPointer() : nullptr;
	const uint64_t indexSize = indexView ? indexView.composed.getStride() : 0u;
	IPolygonGeometryBase::IIndexingCallback::SContext<uint32_t> ctx = {
		.indexBuffer = indexBuffer,
		.indexSize = indexSize,
		.beginPrimitive = primIx,
		.endPrimitive = primIx + 1u,
		.out = idx
	};
	indexing->operator()(ctx);
	if (outIdx)
	{
		outIdx[0] = idx[0];
		outIdx[1] = idx[1];
		outIdx[2] = idx[2];
	}

	hlsl::float32_t3 p0 = {};
	hlsl::float32_t3 p1 = {};
	hlsl::float32_t3 p2 = {};
	if (!posView.decodeElement(idx[0], p0))
		return false;
	if (!posView.decodeElement(idx[1], p1))
		return false;
	if (!posView.decodeElement(idx[2], p2))
		return false;

	out0 = core::vectorSIMDf(p0.x, p0.y, p0.z, 1.f);
	out1 = core::vectorSIMDf(p1.x, p1.y, p1.z, 1.f);
	out2 = core::vectorSIMDf(p2.x, p2.y, p2.z, 1.f);
	return true;
}

inline bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, core::vectorSIMDf& outNormal)
{
	if (!normalView || !idx)
		return false;

	hlsl::float32_t3 n0 = {};
	hlsl::float32_t3 n1 = {};
	hlsl::float32_t3 n2 = {};
	if (!normalView.decodeElement(idx[0], n0))
		return false;
	if (!normalView.decodeElement(idx[1], n1))
		return false;
	if (!normalView.decodeElement(idx[2], n2))
		return false;

	auto normal = core::vectorSIMDf(n0.x, n0.y, n0.z, 0.f);
	if ((normal == core::vectorSIMDf(0.f)).all())
		normal = core::vectorSIMDf(n1.x, n1.y, n1.z, 0.f);
	if ((normal == core::vectorSIMDf(0.f)).all())
		normal = core::vectorSIMDf(n2.x, n2.y, n2.z, 0.f);
	if ((normal == core::vectorSIMDf(0.f)).all())
		return false;

	outNormal = normal;
	return true;
}
}

bool CSTLMeshWriter::writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	if (!geom)
		return false;

	const auto* indexing = geom->getIndexingCallback();
	if (!indexing || indexing->degree() != 3u)
		return false;

	const auto& posView = geom->getPositionView();
	if (!posView)
		return false;
	const auto& normalView = geom->getNormalView();
	const bool flipHandedness = !(context->writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

	// write STL MESH header
	const char headerTxt[] = "Irrlicht-baw Engine";
	constexpr size_t HEADER_SIZE = 80u;

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, headerTxt, context->fileOffset, sizeof(headerTxt));
		context->fileOffset += success.getBytesProcessed();
	}

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();
	const int32_t sizeleft = HEADER_SIZE - sizeof(headerTxt) - static_cast<int32_t>(name.size());

	if (sizeleft < 0)
	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, HEADER_SIZE - sizeof(headerTxt));
		context->fileOffset += success.getBytesProcessed();
	}
	else
	{
		const char buf[80] = {0};
		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, name.size());
			context->fileOffset += success.getBytesProcessed();
		}
		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, buf, context->fileOffset, sizeleft);
			context->fileOffset += success.getBytesProcessed();
		}
	}

	const uint32_t facenum = static_cast<uint32_t>(geom->getPrimitiveCount());
	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, &facenum, context->fileOffset, sizeof(facenum));
		context->fileOffset += success.getBytesProcessed();
	}

	for (uint32_t primIx = 0u; primIx < facenum; ++primIx)
	{
		core::vectorSIMDf v0;
		core::vectorSIMDf v1;
		core::vectorSIMDf v2;
		uint32_t idx[3] = {};
		if (!decodeTriangle(geom, indexing, posView, primIx, v0, v1, v2, idx))
			return false;

		core::vectorSIMDf vertex1 = v2;
		core::vectorSIMDf vertex2 = v1;
		core::vectorSIMDf vertex3 = v0;

		if (flipHandedness)
		{
			vertex1.X = -vertex1.X;
			vertex2.X = -vertex2.X;
			vertex3.X = -vertex3.X;
		}

		core::vectorSIMDf normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
		core::vectorSIMDf attrNormal;
		if (decodeTriangleNormal(normalView, idx, attrNormal))
		{
			if (flipHandedness)
				attrNormal.X = -attrNormal.X;
			if (core::dot(attrNormal, normal).X < 0.f)
				attrNormal = -attrNormal;
			normal = attrNormal;
		}

		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, &normal, context->fileOffset, 12);
			context->fileOffset += success.getBytesProcessed();
		}
		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, &vertex1, context->fileOffset, 12);
			context->fileOffset += success.getBytesProcessed();
		}
		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, &vertex2, context->fileOffset, 12);
			context->fileOffset += success.getBytesProcessed();
		}
		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, &vertex3, context->fileOffset, 12);
			context->fileOffset += success.getBytesProcessed();
		}
		{
			const uint16_t color = 0u;
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, &color, context->fileOffset, 2);
			context->fileOffset += success.getBytesProcessed();
		}
	}

	return true;
}

bool CSTLMeshWriter::writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	if (!geom)
		return false;

	const auto* indexing = geom->getIndexingCallback();
	if (!indexing || indexing->degree() != 3u)
		return false;

	const auto& posView = geom->getPositionView();
	if (!posView)
		return false;
	const auto& normalView = geom->getNormalView();
	const bool flipHandedness = !(context->writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

	const char headerTxt[] = "Irrlicht-baw Engine ";

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "solid ", context->fileOffset, 6);
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, headerTxt, context->fileOffset, sizeof(headerTxt) - 1);
		context->fileOffset += success.getBytesProcessed();
	}

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, name.size());
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "\n", context->fileOffset, 1);
		context->fileOffset += success.getBytesProcessed();
	}

	const uint32_t faceCount = static_cast<uint32_t>(geom->getPrimitiveCount());
	for (uint32_t primIx = 0u; primIx < faceCount; ++primIx)
	{
		core::vectorSIMDf v0;
		core::vectorSIMDf v1;
		core::vectorSIMDf v2;
		uint32_t idx[3] = {};
		if (!decodeTriangle(geom, indexing, posView, primIx, v0, v1, v2, idx))
			return false;
		writeFaceText(v0, v1, v2, idx, normalView, flipHandedness, context);
		{
			system::IFile::success_t success;;
			context->writeContext.outputFile->write(success, "\n", context->fileOffset, 1);
			context->fileOffset += success.getBytesProcessed();
		}
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "endsolid ", context->fileOffset, 9);
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, headerTxt, context->fileOffset, sizeof(headerTxt) - 1);
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, name.c_str(), context->fileOffset, name.size());
		context->fileOffset += success.getBytesProcessed();
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
		const uint32_t* idx,
		const asset::ICPUPolygonGeometry::SDataView& normalView,
		const bool flipHandedness,
		SContext* context)
{
	core::vectorSIMDf vertex1 = v3;
	core::vectorSIMDf vertex2 = v2;
	core::vectorSIMDf vertex3 = v1;
	std::string tmp;

	if (flipHandedness)
	{
		vertex1.X = -vertex1.X;
		vertex2.X = -vertex2.X;
		vertex3.X = -vertex3.X;
	}

	core::vectorSIMDf normal = core::plane3dSIMDf(vertex1, vertex2, vertex3).getNormal();
	core::vectorSIMDf attrNormal;
	if (decodeTriangleNormal(normalView, idx, attrNormal))
	{
		if (flipHandedness)
			attrNormal.X = -attrNormal.X;
		if (core::dot(attrNormal, normal).X < 0.f)
			attrNormal = -attrNormal;
		normal = attrNormal;
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "facet normal ", context->fileOffset, 13);
		context->fileOffset += success.getBytesProcessed();
	}

	getVectorAsStringLine(normal, tmp);
	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "  outer loop\n", context->fileOffset, 13);
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "    vertex ", context->fileOffset, 11);
		context->fileOffset += success.getBytesProcessed();
	}

	getVectorAsStringLine(vertex1, tmp);
	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "    vertex ", context->fileOffset, 11);
		context->fileOffset += success.getBytesProcessed();
	}

	getVectorAsStringLine(vertex2, tmp);
	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "    vertex ", context->fileOffset, 11);
		context->fileOffset += success.getBytesProcessed();
	}

	getVectorAsStringLine(vertex3, tmp);
	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, tmp.c_str(), context->fileOffset, tmp.size());
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "  endloop\n", context->fileOffset, 10);
		context->fileOffset += success.getBytesProcessed();
	}

	{
		system::IFile::success_t success;;
		context->writeContext.outputFile->write(success, "endfacet\n", context->fileOffset, 9);
		context->fileOffset += success.getBytesProcessed();
	}
}

#endif

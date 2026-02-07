// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/system/IFile.h"

#include "CSTLMeshWriter.h"

#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _NBL_COMPILE_WITH_STL_WRITER_

namespace nbl::asset
{

namespace
{

struct SContext
{
	IAssetWriter::SAssetWriteContext writeContext;
	SResolvedFileIOPolicy ioPlan = {};
	core::vector<uint8_t> ioBuffer = {};
	size_t fileOffset = 0ull;
};

}

static bool flushBytes(SContext* context);
static bool writeBytes(SContext* context, const void* data, size_t size);
static bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, core::vectorSIMDf& out0, core::vectorSIMDf& out1, core::vectorSIMDf& out2, uint32_t* outIdx);
static bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, core::vectorSIMDf& outNormal);
static bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context);
static bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context);
static void getVectorAsStringLine(const core::vectorSIMDf& v, std::string& s);
static bool writeFaceText(
	const core::vectorSIMDf& v1,
	const core::vectorSIMDf& v2,
	const core::vectorSIMDf& v3,
	const uint32_t* idx,
	const asset::ICPUPolygonGeometry::SDataView& normalView,
	const bool flipHandedness,
	SContext* context);

CSTLMeshWriter::CSTLMeshWriter()
{
	#ifdef _NBL_DEBUG
	setDebugName("CSTLMeshWriter");
	#endif
}

CSTLMeshWriter::~CSTLMeshWriter()
{
}

const char** CSTLMeshWriter::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "stl", nullptr };
	return ext;
}

uint32_t CSTLMeshWriter::getSupportedFlags()
{
	return asset::EWF_BINARY;
}

uint32_t CSTLMeshWriter::getForcedFlags()
{
	return 0u;
}

bool CSTLMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	if (!_override)
		getDefaultOverride(_override);

	IAssetWriter::SAssetWriteContext inCtx{_params, _file};

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

	SContext context = { IAssetWriter::SAssetWriteContext{ inCtx.params, file} };

	_params.logger.log("WRITING STL: writing the file %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

	const asset::E_WRITER_FLAGS flags = _override->getAssetWritingFlags(context.writeContext, geom, 0u);
	const bool binary = (flags & asset::EWF_BINARY) != 0u;

	uint64_t expectedSize = 0ull;
	bool sizeKnown = false;
	if (binary)
	{
		expectedSize = 84ull + static_cast<uint64_t>(geom->getPrimitiveCount()) * 50ull;
		sizeKnown = true;
	}

	context.ioPlan = resolveFileIOPolicy(_params.ioPolicy, expectedSize, sizeKnown);
	if (!context.ioPlan.valid)
	{
		_params.logger.log("STL writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), context.ioPlan.reason);
		return false;
	}

	if (context.ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile && sizeKnown)
		context.ioBuffer.reserve(static_cast<size_t>(expectedSize));
	else
		context.ioBuffer.reserve(static_cast<size_t>(std::min<uint64_t>(context.ioPlan.chunkSizeBytes, 1ull << 20)));

	const bool written = binary ? writeMeshBinary(geom, &context) : writeMeshASCII(geom, &context);
	if (!written)
		return false;

	return flushBytes(&context);
}

static bool flushBytes(SContext* context)
{
	if (!context)
		return false;
	if (context->ioBuffer.empty())
		return true;

	size_t bytesWritten = 0ull;
	const size_t totalBytes = context->ioBuffer.size();
	while (bytesWritten < totalBytes)
	{
		system::IFile::success_t success;
		context->writeContext.outputFile->write(
			success,
			context->ioBuffer.data() + bytesWritten,
			context->fileOffset + bytesWritten,
			totalBytes - bytesWritten);
		if (!success)
			return false;
		const size_t processed = success.getBytesProcessed();
		if (processed == 0ull)
			return false;
		bytesWritten += processed;
	}
	context->fileOffset += totalBytes;
	context->ioBuffer.clear();
	return true;
}

static bool writeBytes(SContext* context, const void* data, size_t size)
{
	if (!context || (!data && size != 0ull))
		return false;
	if (size == 0ull)
		return true;

	const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
	switch (context->ioPlan.strategy)
	{
		case SResolvedFileIOPolicy::Strategy::WholeFile:
		{
			const size_t oldSize = context->ioBuffer.size();
			context->ioBuffer.resize(oldSize + size);
			std::memcpy(context->ioBuffer.data() + oldSize, src, size);
			return true;
		}
		case SResolvedFileIOPolicy::Strategy::Chunked:
		default:
		{
			const size_t chunkSize = static_cast<size_t>(context->ioPlan.chunkSizeBytes);
			size_t remaining = size;
			while (remaining > 0ull)
			{
				const size_t freeSpace = chunkSize - context->ioBuffer.size();
				const size_t toCopy = std::min(freeSpace, remaining);
				const size_t oldSize = context->ioBuffer.size();
				context->ioBuffer.resize(oldSize + toCopy);
				std::memcpy(context->ioBuffer.data() + oldSize, src, toCopy);
				src += toCopy;
				remaining -= toCopy;

				if (context->ioBuffer.size() == chunkSize)
				{
					if (!flushBytes(context))
						return false;
				}
			}
			return true;
		}
	}
}

static bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, core::vectorSIMDf& out0, core::vectorSIMDf& out1, core::vectorSIMDf& out2, uint32_t* outIdx)
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

static bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, core::vectorSIMDf& outNormal)
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

static bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context)
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
	const uint32_t facenum = static_cast<uint32_t>(geom->getPrimitiveCount());

	// write STL MESH header
	const char headerTxt[] = "Irrlicht-baw Engine";
	constexpr size_t HEADER_SIZE = 80u;
	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();
	const int32_t sizeleft = HEADER_SIZE - sizeof(headerTxt) - static_cast<int32_t>(name.size());

	if (!writeBytes(context, headerTxt, sizeof(headerTxt)))
		return false;

	if (sizeleft < 0)
	{
		if (!writeBytes(context, name.c_str(), HEADER_SIZE - sizeof(headerTxt)))
			return false;
	}
	else
	{
		const char buf[80] = {0};
		if (!writeBytes(context, name.c_str(), name.size()))
			return false;
		if (!writeBytes(context, buf, sizeleft))
			return false;
	}

	if (!writeBytes(context, &facenum, sizeof(facenum)))
		return false;

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

		if (!writeBytes(context, &normal, 12))
			return false;
		if (!writeBytes(context, &vertex1, 12))
			return false;
		if (!writeBytes(context, &vertex2, 12))
			return false;
		if (!writeBytes(context, &vertex3, 12))
			return false;
		const uint16_t color = 0u;
		if (!writeBytes(context, &color, sizeof(color)))
			return false;
	}

	return true;
}

static bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context)
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

	if (!writeBytes(context, "solid ", 6))
		return false;

	if (!writeBytes(context, headerTxt, sizeof(headerTxt) - 1))
		return false;

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();

	if (!writeBytes(context, name.c_str(), name.size()))
		return false;

	if (!writeBytes(context, "\n", 1))
		return false;

	const uint32_t faceCount = static_cast<uint32_t>(geom->getPrimitiveCount());
	for (uint32_t primIx = 0u; primIx < faceCount; ++primIx)
	{
		core::vectorSIMDf v0;
		core::vectorSIMDf v1;
		core::vectorSIMDf v2;
		uint32_t idx[3] = {};
		if (!decodeTriangle(geom, indexing, posView, primIx, v0, v1, v2, idx))
			return false;
		if (!writeFaceText(v0, v1, v2, idx, normalView, flipHandedness, context))
			return false;
		if (!writeBytes(context, "\n", 1))
			return false;
	}

	if (!writeBytes(context, "endsolid ", 9))
		return false;

	if (!writeBytes(context, headerTxt, sizeof(headerTxt) - 1))
		return false;

	if (!writeBytes(context, name.c_str(), name.size()))
		return false;

	return true;
}

static void getVectorAsStringLine(const core::vectorSIMDf& v, std::string& s)
{
	std::ostringstream tmp;
	tmp << v.X << " " << v.Y << " " << v.Z << "\n";
	s = std::string(tmp.str().c_str());
}

static bool writeFaceText(
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

	if (!writeBytes(context, "facet normal ", 13))
		return false;

	getVectorAsStringLine(normal, tmp);
	if (!writeBytes(context, tmp.c_str(), tmp.size()))
		return false;

	if (!writeBytes(context, "  outer loop\n", 13))
		return false;

	if (!writeBytes(context, "    vertex ", 11))
		return false;

	getVectorAsStringLine(vertex1, tmp);
	if (!writeBytes(context, tmp.c_str(), tmp.size()))
		return false;

	if (!writeBytes(context, "    vertex ", 11))
		return false;

	getVectorAsStringLine(vertex2, tmp);
	if (!writeBytes(context, tmp.c_str(), tmp.size()))
		return false;

	if (!writeBytes(context, "    vertex ", 11))
		return false;

	getVectorAsStringLine(vertex3, tmp);
	if (!writeBytes(context, tmp.c_str(), tmp.size()))
		return false;

	if (!writeBytes(context, "  endloop\n", 10))
		return false;

	if (!writeBytes(context, "endfacet\n", 9))
		return false;

	return true;
}

}

#endif


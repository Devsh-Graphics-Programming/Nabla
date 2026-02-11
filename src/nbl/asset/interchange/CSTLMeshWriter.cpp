// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/system/IFile.h"

#include "CSTLMeshWriter.h"
#include "nbl/asset/interchange/SInterchangeIOCommon.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <string_view>

#ifdef _NBL_COMPILE_WITH_STL_WRITER_

namespace nbl::asset
{

namespace stl_writer_detail
{

struct SContext
{
	IAssetWriter::SAssetWriteContext writeContext;
	SResolvedFileIOPolicy ioPlan = {};
	core::vector<uint8_t> ioBuffer = {};
	size_t fileOffset = 0ull;
	double formatMs = 0.0;
	double encodeMs = 0.0;
	double writeMs = 0.0;
	SFileWriteTelemetry writeTelemetry = {};
};

constexpr size_t BinaryHeaderBytes = 80ull;
constexpr size_t BinaryTriangleCountBytes = sizeof(uint32_t);
constexpr size_t BinaryTriangleFloatCount = 12ull;
constexpr size_t BinaryTriangleFloatBytes = sizeof(float) * BinaryTriangleFloatCount;
constexpr size_t BinaryTriangleAttributeBytes = sizeof(uint16_t);
constexpr size_t BinaryTriangleRecordBytes = BinaryTriangleFloatBytes + BinaryTriangleAttributeBytes;
#pragma pack(push, 1)
struct SBinaryTriangleRecord
{
	float payload[BinaryTriangleFloatCount];
	uint16_t attribute = 0u;
};
#pragma pack(pop)
static_assert(sizeof(SBinaryTriangleRecord) == BinaryTriangleRecordBytes);
constexpr size_t BinaryPrefixBytes = BinaryHeaderBytes + BinaryTriangleCountBytes;
constexpr size_t IoFallbackReserveBytes = 1ull << 20;
constexpr size_t AsciiFaceTextMaxBytes = 1024ull;
constexpr char AsciiSolidPrefix[] = "solid ";
constexpr char AsciiEndSolidPrefix[] = "endsolid ";
constexpr char AsciiDefaultName[] = "nabla_mesh";

}

using SContext = stl_writer_detail::SContext;

bool flushBytes(SContext* context);
bool writeBytes(SContext* context, const void* data, size_t size);
const hlsl::float32_t3* getTightFloat3View(const ICPUPolygonGeometry::SDataView& view);
bool decodeTriangleIndices(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView& posView, core::vector<uint32_t>& indexData, const uint32_t*& outIndices, uint32_t& outFaceCount);
bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, core::vectorSIMDf& out0, core::vectorSIMDf& out1, core::vectorSIMDf& out2, uint32_t* outIdx);
bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, core::vectorSIMDf& outNormal);
bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context);
bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context);
bool writeFaceText(
	const core::vectorSIMDf& v1,
	const core::vectorSIMDf& v2,
	const core::vectorSIMDf& v3,
	const uint32_t* idx,
	const asset::ICPUPolygonGeometry::SDataView& normalView,
	const bool flipHandedness,
	SContext* context);

char* appendFloatFixed6ToBuffer(char* dst, char* const end, const float value);
bool appendLiteral(char*& cursor, char* const end, const char* text, const size_t textSize);
bool appendVectorAsAsciiLine(char*& cursor, char* const end, const core::vectorSIMDf& v);

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
	using clock_t = std::chrono::high_resolution_clock;
	const auto totalStart = clock_t::now();

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
	const auto formatStart = clock_t::now();

	uint64_t expectedSize = 0ull;
	bool sizeKnown = false;
	if (binary)
	{
		expectedSize = stl_writer_detail::BinaryPrefixBytes + static_cast<uint64_t>(geom->getPrimitiveCount()) * stl_writer_detail::BinaryTriangleRecordBytes;
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
		context.ioBuffer.reserve(static_cast<size_t>(std::min<uint64_t>(context.ioPlan.chunkSizeBytes, stl_writer_detail::IoFallbackReserveBytes)));
	context.formatMs = std::chrono::duration<double, std::milli>(clock_t::now() - formatStart).count();

	const bool written = binary ? writeMeshBinary(geom, &context) : writeMeshASCII(geom, &context);
	if (!written)
		return false;

	const bool flushed = flushBytes(&context);
	if (!flushed)
		return false;

	const double totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
	const double miscMs = std::max(0.0, totalMs - (context.formatMs + context.encodeMs + context.writeMs));
	const uint64_t ioMinWrite = context.writeTelemetry.getMinOrZero();
	const uint64_t ioAvgWrite = context.writeTelemetry.getAvgOrZero();
	if (isTinyIOTelemetryLikely(context.writeTelemetry, context.fileOffset))
	{
		_params.logger.log(
			"STL writer tiny-io guard: file=%s writes=%llu min=%llu avg=%llu",
			system::ILogger::ELL_WARNING,
			file->getFileName().string().c_str(),
			static_cast<unsigned long long>(context.writeTelemetry.callCount),
			static_cast<unsigned long long>(ioMinWrite),
			static_cast<unsigned long long>(ioAvgWrite));
	}
	_params.logger.log(
		"STL writer stats: file=%s bytes=%llu binary=%d io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		file->getFileName().string().c_str(),
		static_cast<unsigned long long>(context.fileOffset),
		binary ? 1 : 0,
		static_cast<unsigned long long>(context.writeTelemetry.callCount),
		static_cast<unsigned long long>(ioMinWrite),
		static_cast<unsigned long long>(ioAvgWrite),
		toString(_params.ioPolicy.strategy),
		toString(context.ioPlan.strategy),
		static_cast<unsigned long long>(context.ioPlan.chunkSizeBytes),
		context.ioPlan.reason);
	(void)totalMs;
	(void)miscMs;
	(void)context.formatMs;
	(void)context.encodeMs;
	(void)context.writeMs;

	return true;
}

bool flushBytes(SContext* context)
{
	if (!context)
		return false;
	if (context->ioBuffer.empty())
		return true;

	using clock_t = std::chrono::high_resolution_clock;
	const auto writeStart = clock_t::now();
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
		context->writeTelemetry.account(processed);
		bytesWritten += processed;
	}
	context->fileOffset += totalBytes;
	context->ioBuffer.clear();
	context->writeMs += std::chrono::duration<double, std::milli>(clock_t::now() - writeStart).count();
	return true;
}

bool writeBytes(SContext* context, const void* data, size_t size)
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

char* appendFloatFixed6ToBuffer(char* dst, char* const end, const float value)
{
	if (!dst || dst >= end)
		return end;

	const auto result = std::to_chars(dst, end, value, std::chars_format::fixed, 6);
	if (result.ec == std::errc())
		return result.ptr;

	const int written = std::snprintf(dst, static_cast<size_t>(end - dst), "%.6f", static_cast<double>(value));
	if (written <= 0)
		return dst;
	const size_t writeLen = static_cast<size_t>(written);
	return (writeLen < static_cast<size_t>(end - dst)) ? (dst + writeLen) : end;
}

bool appendLiteral(char*& cursor, char* const end, const char* text, const size_t textSize)
{
	if (!cursor || cursor + textSize > end)
		return false;
	std::memcpy(cursor, text, textSize);
	cursor += textSize;
	return true;
}

bool appendVectorAsAsciiLine(char*& cursor, char* const end, const core::vectorSIMDf& v)
{
	cursor = appendFloatFixed6ToBuffer(cursor, end, v.X);
	if (cursor >= end)
		return false;
	*(cursor++) = ' ';
	cursor = appendFloatFixed6ToBuffer(cursor, end, v.Y);
	if (cursor >= end)
		return false;
	*(cursor++) = ' ';
	cursor = appendFloatFixed6ToBuffer(cursor, end, v.Z);
	if (cursor >= end)
		return false;
	*(cursor++) = '\n';
	return true;
}

const hlsl::float32_t3* getTightFloat3View(const ICPUPolygonGeometry::SDataView& view)
{
	if (!view)
		return nullptr;
	if (view.composed.format != EF_R32G32B32_SFLOAT)
		return nullptr;
	if (view.composed.getStride() != sizeof(hlsl::float32_t3))
		return nullptr;
	return reinterpret_cast<const hlsl::float32_t3*>(view.getPointer());
}

bool decodeTriangleIndices(const ICPUPolygonGeometry* geom, const ICPUPolygonGeometry::SDataView& posView, core::vector<uint32_t>& indexData, const uint32_t*& outIndices, uint32_t& outFaceCount)
{
	const auto& indexView = geom->getIndexView();
	if (indexView)
	{
		const size_t indexCount = indexView.getElementCount();
		if ((indexCount % 3ull) != 0ull)
			return false;

		const void* src = indexView.getPointer();
		if (!src)
			return false;

		if (indexView.composed.format == EF_R32_UINT && indexView.composed.getStride() == sizeof(uint32_t))
		{
			outIndices = reinterpret_cast<const uint32_t*>(src);
		}
		else if (indexView.composed.format == EF_R16_UINT && indexView.composed.getStride() == sizeof(uint16_t))
		{
			indexData.resize(indexCount);
			const auto* src16 = reinterpret_cast<const uint16_t*>(src);
			for (size_t i = 0ull; i < indexCount; ++i)
				indexData[i] = src16[i];
			outIndices = indexData.data();
		}
		else
		{
			indexData.resize(indexCount);
			hlsl::vector<uint32_t, 1> decoded = {};
			for (size_t i = 0ull; i < indexCount; ++i)
			{
				if (!indexView.decodeElement(i, decoded))
					return false;
				indexData[i] = decoded.x;
			}
			outIndices = indexData.data();
		}
		outFaceCount = static_cast<uint32_t>(indexCount / 3ull);
		return true;
	}

	const size_t vertexCount = posView.getElementCount();
	if ((vertexCount % 3ull) != 0ull)
		return false;

	outIndices = nullptr;
	outFaceCount = static_cast<uint32_t>(vertexCount / 3ull);
	return true;
}

bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, core::vectorSIMDf& out0, core::vectorSIMDf& out1, core::vectorSIMDf& out2, uint32_t* outIdx)
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

bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, core::vectorSIMDf& outNormal)
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

bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	if (!geom || !context || !context->writeContext.outputFile)
		return false;
	using clock_t = std::chrono::high_resolution_clock;
	const auto encodeStart = clock_t::now();

	const auto& posView = geom->getPositionView();
	if (!posView)
		return false;

	const bool flipHandedness = !(context->writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
	const size_t vertexCount = posView.getElementCount();
	if (vertexCount == 0ull)
		return false;

	core::vector<uint32_t> indexData;
	const uint32_t* indices = nullptr;
	uint32_t facenum = 0u;
	if (!decodeTriangleIndices(geom, posView, indexData, indices, facenum))
		return false;

	const size_t outputSize = stl_writer_detail::BinaryPrefixBytes + static_cast<size_t>(facenum) * stl_writer_detail::BinaryTriangleRecordBytes;
	std::unique_ptr<uint8_t[]> output(new (std::nothrow) uint8_t[outputSize]);
	if (!output)
		return false;
	uint8_t* dst = output.get();

	std::memset(dst, 0, stl_writer_detail::BinaryHeaderBytes);
	dst += stl_writer_detail::BinaryHeaderBytes;

	std::memcpy(dst, &facenum, sizeof(facenum));
	dst += sizeof(facenum);

	const auto& normalView = geom->getNormalView();
	const bool hasNormals = static_cast<bool>(normalView);
	const hlsl::float32_t3* const tightPositions = getTightFloat3View(posView);
	const hlsl::float32_t3* const tightNormals = hasNormals ? getTightFloat3View(normalView) : nullptr;
	const float handednessSign = flipHandedness ? -1.f : 1.f;

	auto decodePosition = [&](const uint32_t ix, hlsl::float32_t3& out)->bool
	{
		if (tightPositions)
		{
			out = tightPositions[ix];
			return true;
		}
		return posView.decodeElement(ix, out);
	};

	auto decodeNormal = [&](const uint32_t ix, hlsl::float32_t3& out)->bool
	{
		if (!hasNormals)
			return false;
		if (tightNormals)
		{
			out = tightNormals[ix];
			return true;
		}
		return normalView.decodeElement(ix, out);
	};
	auto writeRecord = [&dst](const float nx, const float ny, const float nz, const float v1x, const float v1y, const float v1z, const float v2x, const float v2y, const float v2z, const float v3x, const float v3y, const float v3z)->void
	{
		const stl_writer_detail::SBinaryTriangleRecord record = {
			{
				nx, ny, nz,
				v1x, v1y, v1z,
				v2x, v2y, v2z,
				v3x, v3y, v3z
			},
			0u
		};
		std::memcpy(dst, &record, sizeof(record));
		dst += sizeof(record);
	};

	const bool hasFastTightPath = (indices == nullptr) && (tightPositions != nullptr) && (!hasNormals || (tightNormals != nullptr));
	if (hasFastTightPath && hasNormals)
	{
		bool allFastNormalsNonZero = true;
		const size_t normalCount = static_cast<size_t>(facenum) * 3ull;
		for (size_t i = 0ull; i < normalCount; ++i)
		{
			const auto& n = tightNormals[i];
			if (n.x == 0.f && n.y == 0.f && n.z == 0.f)
			{
				allFastNormalsNonZero = false;
				break;
			}
		}

		const hlsl::float32_t3* posTri = tightPositions;
		const hlsl::float32_t3* nrmTri = tightNormals;
		if (allFastNormalsNonZero)
		{
			for (uint32_t primIx = 0u; primIx < facenum; ++primIx, posTri += 3u, nrmTri += 3u)
			{
				const hlsl::float32_t3 vertex1 = posTri[2u];
				const hlsl::float32_t3 vertex2 = posTri[1u];
				const hlsl::float32_t3 vertex3 = posTri[0u];
				const float vertex1x = vertex1.x * handednessSign;
				const float vertex2x = vertex2.x * handednessSign;
				const float vertex3x = vertex3.x * handednessSign;

				hlsl::float32_t3 attrNormal = nrmTri[0u];
				if (flipHandedness)
					attrNormal.x = -attrNormal.x;

				writeRecord(
					attrNormal.x, attrNormal.y, attrNormal.z,
					vertex1x, vertex1.y, vertex1.z,
					vertex2x, vertex2.y, vertex2.z,
					vertex3x, vertex3.y, vertex3.z);
			}
		}
		else
		{
			for (uint32_t primIx = 0u; primIx < facenum; ++primIx, posTri += 3u, nrmTri += 3u)
			{
				const hlsl::float32_t3 vertex1 = posTri[2u];
				const hlsl::float32_t3 vertex2 = posTri[1u];
				const hlsl::float32_t3 vertex3 = posTri[0u];
				const float vertex1x = vertex1.x * handednessSign;
				const float vertex2x = vertex2.x * handednessSign;
				const float vertex3x = vertex3.x * handednessSign;

				float normalX = 0.f;
				float normalY = 0.f;
				float normalZ = 0.f;
				hlsl::float32_t3 attrNormal = nrmTri[0u];
				if (attrNormal.x == 0.f && attrNormal.y == 0.f && attrNormal.z == 0.f)
					attrNormal = nrmTri[1u];
				if (attrNormal.x == 0.f && attrNormal.y == 0.f && attrNormal.z == 0.f)
					attrNormal = nrmTri[2u];
				if (!(attrNormal.x == 0.f && attrNormal.y == 0.f && attrNormal.z == 0.f))
				{
					if (flipHandedness)
						attrNormal.x = -attrNormal.x;
					normalX = attrNormal.x;
					normalY = attrNormal.y;
					normalZ = attrNormal.z;
				}

				if (normalX == 0.f && normalY == 0.f && normalZ == 0.f)
				{
					const float edge21x = vertex2x - vertex1x;
					const float edge21y = vertex2.y - vertex1.y;
					const float edge21z = vertex2.z - vertex1.z;
					const float edge31x = vertex3x - vertex1x;
					const float edge31y = vertex3.y - vertex1.y;
					const float edge31z = vertex3.z - vertex1.z;

					normalX = edge21y * edge31z - edge21z * edge31y;
					normalY = edge21z * edge31x - edge21x * edge31z;
					normalZ = edge21x * edge31y - edge21y * edge31x;
					const float planeNormalLen2 = normalX * normalX + normalY * normalY + normalZ * normalZ;
					if (planeNormalLen2 > 0.f)
					{
						const float invLen = 1.f / std::sqrt(planeNormalLen2);
						normalX *= invLen;
						normalY *= invLen;
						normalZ *= invLen;
					}
				}

				writeRecord(
					normalX, normalY, normalZ,
					vertex1x, vertex1.y, vertex1.z,
					vertex2x, vertex2.y, vertex2.z,
					vertex3x, vertex3.y, vertex3.z);
			}
		}
	}
	else if (hasFastTightPath)
	{
		const hlsl::float32_t3* posTri = tightPositions;
		for (uint32_t primIx = 0u; primIx < facenum; ++primIx, posTri += 3u)
		{
			const hlsl::float32_t3 vertex1 = posTri[2u];
			const hlsl::float32_t3 vertex2 = posTri[1u];
			const hlsl::float32_t3 vertex3 = posTri[0u];
			const float vertex1x = vertex1.x * handednessSign;
			const float vertex2x = vertex2.x * handednessSign;
			const float vertex3x = vertex3.x * handednessSign;

			const float edge21x = vertex2x - vertex1x;
			const float edge21y = vertex2.y - vertex1.y;
			const float edge21z = vertex2.z - vertex1.z;
			const float edge31x = vertex3x - vertex1x;
			const float edge31y = vertex3.y - vertex1.y;
			const float edge31z = vertex3.z - vertex1.z;

			float normalX = edge21y * edge31z - edge21z * edge31y;
			float normalY = edge21z * edge31x - edge21x * edge31z;
			float normalZ = edge21x * edge31y - edge21y * edge31x;
			const float planeNormalLen2 = normalX * normalX + normalY * normalY + normalZ * normalZ;
			if (planeNormalLen2 > 0.f)
			{
				const float invLen = 1.f / std::sqrt(planeNormalLen2);
				normalX *= invLen;
				normalY *= invLen;
				normalZ *= invLen;
			}

			writeRecord(
				normalX, normalY, normalZ,
				vertex1x, vertex1.y, vertex1.z,
				vertex2x, vertex2.y, vertex2.z,
				vertex3x, vertex3.y, vertex3.z);
		}
	}
	else
	{
		for (uint32_t primIx = 0u; primIx < facenum; ++primIx)
		{
			const uint32_t i0 = indices ? indices[primIx * 3u + 0u] : (primIx * 3u + 0u);
			const uint32_t i1 = indices ? indices[primIx * 3u + 1u] : (primIx * 3u + 1u);
			const uint32_t i2 = indices ? indices[primIx * 3u + 2u] : (primIx * 3u + 2u);
			if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
				return false;

			hlsl::float32_t3 p0 = {};
			hlsl::float32_t3 p1 = {};
			hlsl::float32_t3 p2 = {};
			if (!decodePosition(i0, p0) || !decodePosition(i1, p1) || !decodePosition(i2, p2))
				return false;

			hlsl::float32_t3 vertex1 = p2;
			hlsl::float32_t3 vertex2 = p1;
			hlsl::float32_t3 vertex3 = p0;

			if (flipHandedness)
			{
				vertex1.x = -vertex1.x;
				vertex2.x = -vertex2.x;
				vertex3.x = -vertex3.x;
			}

			const hlsl::float32_t3 planeNormal = hlsl::cross(vertex2 - vertex1, vertex3 - vertex1);
			const float planeNormalLen2 = hlsl::dot(planeNormal, planeNormal);
			hlsl::float32_t3 normal = hlsl::float32_t3(0.f, 0.f, 0.f);
			if (!hasNormals)
			{
				if (planeNormalLen2 > 0.f)
					normal = hlsl::normalize(planeNormal);
			}

			if (hasNormals)
			{
				hlsl::float32_t3 n0 = {};
				if (!decodeNormal(i0, n0))
					return false;

				hlsl::float32_t3 attrNormal = n0;
				if (hlsl::dot(attrNormal, attrNormal) <= 0.f)
				{
					hlsl::float32_t3 n1 = {};
					if (!decodeNormal(i1, n1))
						return false;
					attrNormal = n1;
				}
				if (hlsl::dot(attrNormal, attrNormal) <= 0.f)
				{
					hlsl::float32_t3 n2 = {};
					if (!decodeNormal(i2, n2))
						return false;
					attrNormal = n2;
				}

				if (hlsl::dot(attrNormal, attrNormal) > 0.f)
				{
					if (flipHandedness)
						attrNormal.x = -attrNormal.x;
					if (planeNormalLen2 > 0.f && hlsl::dot(attrNormal, planeNormal) < 0.f)
						attrNormal = -attrNormal;
					normal = attrNormal;
				}
				else if (planeNormalLen2 > 0.f)
				{
					normal = hlsl::normalize(planeNormal);
				}
			}

			writeRecord(
				normal.x, normal.y, normal.z,
				vertex1.x, vertex1.y, vertex1.z,
				vertex2.x, vertex2.y, vertex2.z,
				vertex3.x, vertex3.y, vertex3.z);
		}
	}

	context->encodeMs += std::chrono::duration<double, std::milli>(clock_t::now() - encodeStart).count();
	const auto writeStart = clock_t::now();
	const bool writeOk = writeFileWithPolicy(context->writeContext.outputFile, context->ioPlan, output.get(), outputSize, &context->writeTelemetry);
	context->writeMs += std::chrono::duration<double, std::milli>(clock_t::now() - writeStart).count();
	if (writeOk)
		context->fileOffset += outputSize;
	return writeOk;
}

bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	if (!geom)
		return false;
	using clock_t = std::chrono::high_resolution_clock;
	const auto encodeStart = clock_t::now();

	const auto* indexing = geom->getIndexingCallback();
	if (!indexing || indexing->degree() != 3u)
		return false;

	const auto& posView = geom->getPositionView();
	if (!posView)
		return false;
	const auto& normalView = geom->getNormalView();
	const bool flipHandedness = !(context->writeContext.params.flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();
	const std::string_view solidName = name.empty() ? std::string_view(stl_writer_detail::AsciiDefaultName) : std::string_view(name);

	if (!writeBytes(context, stl_writer_detail::AsciiSolidPrefix, sizeof(stl_writer_detail::AsciiSolidPrefix) - 1ull))
		return false;

	if (!writeBytes(context, solidName.data(), solidName.size()))
		return false;

	if (!writeBytes(context, "\n", sizeof("\n") - 1ull))
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
		if (!writeBytes(context, "\n", sizeof("\n") - 1ull))
			return false;
	}

	if (!writeBytes(context, stl_writer_detail::AsciiEndSolidPrefix, sizeof(stl_writer_detail::AsciiEndSolidPrefix) - 1ull))
		return false;

	if (!writeBytes(context, solidName.data(), solidName.size()))
		return false;

	context->encodeMs += std::chrono::duration<double, std::milli>(clock_t::now() - encodeStart).count();
	return true;
}

bool writeFaceText(
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

	std::array<char, stl_writer_detail::AsciiFaceTextMaxBytes> faceText = {};
	char* cursor = faceText.data();
	char* const end = faceText.data() + faceText.size();
	if (!appendLiteral(cursor, end, "facet normal ", sizeof("facet normal ") - 1ull))
		return false;
	if (!appendVectorAsAsciiLine(cursor, end, normal))
		return false;
	if (!appendLiteral(cursor, end, "  outer loop\n", sizeof("  outer loop\n") - 1ull))
		return false;
	if (!appendLiteral(cursor, end, "    vertex ", sizeof("    vertex ") - 1ull))
		return false;
	if (!appendVectorAsAsciiLine(cursor, end, vertex1))
		return false;
	if (!appendLiteral(cursor, end, "    vertex ", sizeof("    vertex ") - 1ull))
		return false;
	if (!appendVectorAsAsciiLine(cursor, end, vertex2))
		return false;
	if (!appendLiteral(cursor, end, "    vertex ", sizeof("    vertex ") - 1ull))
		return false;
	if (!appendVectorAsAsciiLine(cursor, end, vertex3))
		return false;
	if (!appendLiteral(cursor, end, "  endloop\n", sizeof("  endloop\n") - 1ull))
		return false;
	if (!appendLiteral(cursor, end, "endfacet\n", sizeof("endfacet\n") - 1ull))
		return false;

	return writeBytes(context, faceText.data(), static_cast<size_t>(cursor - faceText.data()));
}

}

#endif


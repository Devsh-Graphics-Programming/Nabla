// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/system/IFile.h"

#include "CSTLMeshWriter.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "SSTLPolygonGeometryAuxLayout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
	SFileWriteTelemetry writeTelemetry = {};

	bool flush();
	bool write(const void* data, size_t size);
};

constexpr size_t BinaryHeaderBytes = 80ull;
constexpr size_t BinaryTriangleCountBytes = sizeof(uint32_t);
constexpr size_t BinaryTriangleFloatCount = 12ull;
constexpr size_t BinaryTriangleFloatBytes = sizeof(float) * BinaryTriangleFloatCount;
constexpr size_t BinaryTriangleAttributeBytes = sizeof(uint16_t);
constexpr size_t BinaryTriangleRecordBytes = BinaryTriangleFloatBytes + BinaryTriangleAttributeBytes;
static_assert(BinaryTriangleRecordBytes == 50ull);
constexpr size_t BinaryPrefixBytes = BinaryHeaderBytes + BinaryTriangleCountBytes;
constexpr size_t IoFallbackReserveBytes = 1ull << 20;
constexpr size_t AsciiFaceTextMaxBytes = 1024ull;
constexpr char AsciiSolidPrefix[] = "solid ";
constexpr char AsciiEndSolidPrefix[] = "endsolid ";
constexpr char AsciiDefaultName[] = "nabla_mesh";

}

using SContext = stl_writer_detail::SContext;

bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, hlsl::float32_t3& out0, hlsl::float32_t3& out1, hlsl::float32_t3& out2, uint32_t* outIdx);
bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, hlsl::float32_t3& outNormal);
double stlNormalizeColorComponentToUnit(double value);
uint16_t stlPackViscamColorFromB8G8R8A8(uint32_t color);
const ICPUPolygonGeometry::SDataView* stlGetColorView(const ICPUPolygonGeometry* geom, size_t vertexCount);
bool stlDecodeColorB8G8R8A8(const ICPUPolygonGeometry::SDataView& colorView, const uint32_t ix, uint32_t& outColor);
void stlDecodeColorUnitRGBAFromB8G8R8A8(uint32_t color, double (&out)[4]);
bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context);
bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context);
bool writeFaceText(
	const hlsl::float32_t3& v1,
	const hlsl::float32_t3& v2,
	const hlsl::float32_t3& v3,
	const uint32_t* idx,
	const asset::ICPUPolygonGeometry::SDataView& normalView,
	const bool flipHandedness,
	SContext* context);

bool appendLiteral(char*& cursor, char* const end, const char* text, const size_t textSize);
bool appendVectorAsAsciiLine(char*& cursor, char* const end, const hlsl::float32_t3& v);

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

writer_flags_t CSTLMeshWriter::getSupportedFlags()
{
	return asset::EWF_BINARY;
}

writer_flags_t CSTLMeshWriter::getForcedFlags()
{
	return EWF_NONE;
}

bool CSTLMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	if (!_override)
		getDefaultOverride(_override);

	IAssetWriter::SAssetWriteContext inCtx{_params, _file};

	const asset::ICPUPolygonGeometry* geom = IAsset::castDown<const asset::ICPUPolygonGeometry>(_params.rootAsset);
	if (!geom)
		return false;

	system::IFile* file = _override->getOutputFile(_file, inCtx, {geom, 0u});
	if (!file)
		return false;

	SContext context = { IAssetWriter::SAssetWriteContext{ inCtx.params, file} };

	_params.logger.log("WRITING STL: writing the file %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

	const auto flags = _override->getAssetWritingFlags(context.writeContext, geom, 0u);
	const bool binary = flags.hasAnyFlag(asset::EWF_BINARY);

	uint64_t expectedSize = 0ull;
	bool sizeKnown = false;
	if (binary)
	{
		expectedSize = stl_writer_detail::BinaryPrefixBytes + static_cast<uint64_t>(geom->getPrimitiveCount()) * stl_writer_detail::BinaryTriangleRecordBytes;
		sizeKnown = true;
	}

	const bool fileMappable = core::bitflag<system::IFile::E_CREATE_FLAGS>(file->getFlags()).hasAnyFlag(system::IFile::ECF_MAPPABLE);
    context.ioPlan = SResolvedFileIOPolicy(_params.ioPolicy, expectedSize, sizeKnown, fileMappable);
	if (!context.ioPlan.isValid())
	{
		_params.logger.log("STL writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), context.ioPlan.reason);
		return false;
	}

	if (context.ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile && sizeKnown)
		context.ioBuffer.reserve(static_cast<size_t>(expectedSize));
	else
		context.ioBuffer.reserve(static_cast<size_t>(std::min<uint64_t>(context.ioPlan.chunkSizeBytes(), stl_writer_detail::IoFallbackReserveBytes)));

	const bool written = binary ? writeMeshBinary(geom, &context) : writeMeshASCII(geom, &context);
	if (!written)
		return false;

	const bool flushed = context.flush();
	if (!flushed)
		return false;

	const uint64_t ioMinWrite = context.writeTelemetry.getMinOrZero();
	const uint64_t ioAvgWrite = context.writeTelemetry.getAvgOrZero();
	if (SInterchangeIO::isTinyIOTelemetryLikely(context.writeTelemetry, context.fileOffset, _params.ioPolicy))
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
		system::to_string(_params.ioPolicy.strategy).c_str(),
		system::to_string(context.ioPlan.strategy).c_str(),
		static_cast<unsigned long long>(context.ioPlan.chunkSizeBytes()),
		context.ioPlan.reason);

	return true;
}

bool stl_writer_detail::SContext::flush()
{
	if (ioBuffer.empty())
		return true;

	size_t bytesWritten = 0ull;
	const size_t totalBytes = ioBuffer.size();
	while (bytesWritten < totalBytes)
	{
		system::IFile::success_t success;
		writeContext.outputFile->write(
			success,
			ioBuffer.data() + bytesWritten,
			fileOffset + bytesWritten,
			totalBytes - bytesWritten);
		if (!success)
			return false;
		const size_t processed = success.getBytesProcessed();
		if (processed == 0ull)
			return false;
		writeTelemetry.account(processed);
		bytesWritten += processed;
	}
	fileOffset += totalBytes;
	ioBuffer.clear();
	return true;
}

bool stl_writer_detail::SContext::write(const void* data, size_t size)
{
	if (!data && size != 0ull)
		return false;
	if (size == 0ull)
		return true;

	const uint8_t* src = reinterpret_cast<const uint8_t*>(data);
	switch (ioPlan.strategy)
	{
		case SResolvedFileIOPolicy::Strategy::WholeFile:
		{
			const size_t oldSize = ioBuffer.size();
			ioBuffer.resize(oldSize + size);
			std::memcpy(ioBuffer.data() + oldSize, src, size);
			return true;
		}
		case SResolvedFileIOPolicy::Strategy::Chunked:
		default:
		{
			const size_t chunkSize = static_cast<size_t>(ioPlan.chunkSizeBytes());
			size_t remaining = size;
			while (remaining > 0ull)
			{
				const size_t freeSpace = chunkSize - ioBuffer.size();
				const size_t toCopy = std::min(freeSpace, remaining);
				const size_t oldSize = ioBuffer.size();
				ioBuffer.resize(oldSize + toCopy);
				std::memcpy(ioBuffer.data() + oldSize, src, toCopy);
				src += toCopy;
				remaining -= toCopy;

				if (ioBuffer.size() == chunkSize)
				{
					if (!flush())
						return false;
				}
			}
			return true;
		}
	}
}

bool appendLiteral(char*& cursor, char* const end, const char* text, const size_t textSize)
{
	if (!cursor || cursor + textSize > end)
		return false;
	std::memcpy(cursor, text, textSize);
	cursor += textSize;
	return true;
}

bool appendVectorAsAsciiLine(char*& cursor, char* const end, const hlsl::float32_t3& v)
{
	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, end, v.x);
	if (cursor >= end)
		return false;
	*(cursor++) = ' ';
	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, end, v.y);
	if (cursor >= end)
		return false;
	*(cursor++) = ' ';
	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, end, v.z);
	if (cursor >= end)
		return false;
	*(cursor++) = '\n';
	return true;
}

bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, hlsl::float32_t3& out0, hlsl::float32_t3& out1, hlsl::float32_t3& out2, uint32_t* outIdx)
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

	out0 = p0;
	out1 = p1;
	out2 = p2;
	return true;
}

bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const uint32_t* idx, hlsl::float32_t3& outNormal)
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

	auto normal = n0;
	if (normal.x == 0.f && normal.y == 0.f && normal.z == 0.f)
		normal = n1;
	if (normal.x == 0.f && normal.y == 0.f && normal.z == 0.f)
		normal = n2;
	if (normal.x == 0.f && normal.y == 0.f && normal.z == 0.f)
		return false;

	outNormal = normal;
	return true;
}

double stlNormalizeColorComponentToUnit(double value)
{
	if (!std::isfinite(value))
		return 0.0;
	if (value > 1.0)
		value /= 255.0;
	return std::clamp(value, 0.0, 1.0);
}

uint16_t stlPackViscamColorFromB8G8R8A8(const uint32_t color)
{
	const void* src[4] = { &color, nullptr, nullptr, nullptr };
	uint16_t packed = 0u;
	convertColor<EF_B8G8R8A8_UNORM, EF_A1R5G5B5_UNORM_PACK16>(src, &packed, 0u, 0u);
	packed |= 0x8000u;
	return packed;
}

const ICPUPolygonGeometry::SDataView* stlGetColorView(const ICPUPolygonGeometry* geom, const size_t vertexCount)
{
	const auto* view = SGeometryWriterCommon::getAuxViewAt(geom, SSTLPolygonGeometryAuxLayout::COLOR0, vertexCount);
	if (!view)
		return nullptr;
	return getFormatChannelCount(view->composed.format) >= 3u ? view : nullptr;
}

bool stlDecodeColorB8G8R8A8(const ICPUPolygonGeometry::SDataView& colorView, const uint32_t ix, uint32_t& outColor)
{
	if (colorView.composed.format == EF_B8G8R8A8_UNORM && colorView.composed.getStride() == sizeof(uint32_t))
	{
		const auto* const ptr = reinterpret_cast<const uint8_t*>(colorView.getPointer());
		if (!ptr)
			return false;
		std::memcpy(&outColor, ptr + static_cast<size_t>(ix) * sizeof(uint32_t), sizeof(outColor));
		return true;
	}

	hlsl::float64_t4 decoded = {};
	if (!colorView.decodeElement(ix, decoded))
		return false;
	const double rgbaUnit[4] = {
		stlNormalizeColorComponentToUnit(decoded.x),
		stlNormalizeColorComponentToUnit(decoded.y),
		stlNormalizeColorComponentToUnit(decoded.z),
		stlNormalizeColorComponentToUnit(decoded.w)
	};
	encodePixels<EF_B8G8R8A8_UNORM, double>(&outColor, rgbaUnit);
	return true;
}

void stlDecodeColorUnitRGBAFromB8G8R8A8(const uint32_t color, double (&out)[4])
{
	const void* src[4] = { &color, nullptr, nullptr, nullptr };
	decodePixels<EF_B8G8R8A8_UNORM, double>(src, out, 0u, 0u);
}

bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, SContext* context)
{
	if (!geom || !context || !context->writeContext.outputFile)
		return false;

	const auto& posView = geom->getPositionView();
	if (!posView)
		return false;

	const bool flipHandedness = !context->writeContext.params.flags.hasAnyFlag(E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
	const size_t vertexCount = posView.getElementCount();
	if (vertexCount == 0ull)
		return false;

	uint32_t facenum = 0u;
	size_t faceCount = 0ull;
	if (!SGeometryWriterCommon::getTriangleFaceCount(geom, faceCount))
		return false;
	if (faceCount > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
		return false;
	facenum = static_cast<uint32_t>(faceCount);

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
	const auto* const colorView = stlGetColorView(geom, vertexCount);
	const hlsl::float32_t3* const tightPositions = SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(posView);
	const hlsl::float32_t3* const tightNormals = hasNormals ? SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(normalView) : nullptr;
	const bool hasImplicitTriangleIndices = !geom->getIndexView();

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
	auto computeFaceColor = [&](const uint32_t i0, const uint32_t i1, const uint32_t i2, uint16_t& outColor)->bool
	{
		outColor = 0u;
		if (!colorView)
			return true;
		uint32_t c0 = 0u, c1 = 0u, c2 = 0u;
		if (!stlDecodeColorB8G8R8A8(*colorView, i0, c0))
			return false;
		if (!stlDecodeColorB8G8R8A8(*colorView, i1, c1))
			return false;
		if (!stlDecodeColorB8G8R8A8(*colorView, i2, c2))
			return false;
		double rgba0[4] = {};
		double rgba1[4] = {};
		double rgba2[4] = {};
		stlDecodeColorUnitRGBAFromB8G8R8A8(c0, rgba0);
		stlDecodeColorUnitRGBAFromB8G8R8A8(c1, rgba1);
		stlDecodeColorUnitRGBAFromB8G8R8A8(c2, rgba2);
		const double rgbaAvg[4] = {
			(rgba0[0] + rgba1[0] + rgba2[0]) / 3.0,
			(rgba0[1] + rgba1[1] + rgba2[1]) / 3.0,
			(rgba0[2] + rgba1[2] + rgba2[2]) / 3.0,
			1.0
		};
		uint32_t avgColor = 0u;
		encodePixels<EF_B8G8R8A8_UNORM, double>(&avgColor, rgbaAvg);
		outColor = stlPackViscamColorFromB8G8R8A8(avgColor);
		return true;
	};
	auto writeRecord = [&dst](const float nx, const float ny, const float nz, const float v1x, const float v1y, const float v1z, const float v2x, const float v2y, const float v2z, const float v3x, const float v3y, const float v3z, const uint16_t attribute)->void
	{
		const float payload[stl_writer_detail::BinaryTriangleFloatCount] = {
			nx, ny, nz,
			v1x, v1y, v1z,
			v2x, v2y, v2z,
			v3x, v3y, v3z
		};
		std::memcpy(dst, payload, stl_writer_detail::BinaryTriangleFloatBytes);
		dst += stl_writer_detail::BinaryTriangleFloatBytes;
		std::memcpy(dst, &attribute, stl_writer_detail::BinaryTriangleAttributeBytes);
		dst += stl_writer_detail::BinaryTriangleAttributeBytes;
	};
	auto prepareVertices = [&](const hlsl::float32_t3& p0, const hlsl::float32_t3& p1, const hlsl::float32_t3& p2, hlsl::float32_t3& vertex1, hlsl::float32_t3& vertex2, hlsl::float32_t3& vertex3)->void
	{
		vertex1 = p2;
		vertex2 = p1;
		vertex3 = p0;
		if (flipHandedness)
		{
			vertex1.x = -vertex1.x;
			vertex2.x = -vertex2.x;
			vertex3.x = -vertex3.x;
		}
	};
	auto computePlaneNormal = [&](const hlsl::float32_t3& vertex1, const hlsl::float32_t3& vertex2, const hlsl::float32_t3& vertex3)->hlsl::float32_t3
	{
		const hlsl::float32_t3 planeNormal = hlsl::cross(vertex2 - vertex1, vertex3 - vertex1);
		const float planeNormalLen2 = hlsl::dot(planeNormal, planeNormal);
		return planeNormalLen2 > 0.f ? hlsl::normalize(planeNormal) : hlsl::float32_t3(0.f, 0.f, 0.f);
	};

	const bool hasFastTightPath = hasImplicitTriangleIndices && (tightPositions != nullptr) && (!hasNormals || (tightNormals != nullptr));
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
				uint16_t faceColor = 0u;
				if (!computeFaceColor(primIx * 3u + 0u, primIx * 3u + 1u, primIx * 3u + 2u, faceColor))
					return false;

				hlsl::float32_t3 vertex1 = {};
				hlsl::float32_t3 vertex2 = {};
				hlsl::float32_t3 vertex3 = {};
				prepareVertices(posTri[0u], posTri[1u], posTri[2u], vertex1, vertex2, vertex3);

				hlsl::float32_t3 attrNormal = nrmTri[0u];
				if (flipHandedness)
					attrNormal.x = -attrNormal.x;

				writeRecord(
					attrNormal.x, attrNormal.y, attrNormal.z,
					vertex1.x, vertex1.y, vertex1.z,
					vertex2.x, vertex2.y, vertex2.z,
					vertex3.x, vertex3.y, vertex3.z,
					faceColor);
			}
		}
		else
		{
			for (uint32_t primIx = 0u; primIx < facenum; ++primIx, posTri += 3u, nrmTri += 3u)
			{
				uint16_t faceColor = 0u;
				if (!computeFaceColor(primIx * 3u + 0u, primIx * 3u + 1u, primIx * 3u + 2u, faceColor))
					return false;

				hlsl::float32_t3 vertex1 = {};
				hlsl::float32_t3 vertex2 = {};
				hlsl::float32_t3 vertex3 = {};
				prepareVertices(posTri[0u], posTri[1u], posTri[2u], vertex1, vertex2, vertex3);

				hlsl::float32_t3 normal = hlsl::float32_t3(0.f, 0.f, 0.f);
				hlsl::float32_t3 attrNormal = nrmTri[0u];
				if (attrNormal.x == 0.f && attrNormal.y == 0.f && attrNormal.z == 0.f)
					attrNormal = nrmTri[1u];
				if (attrNormal.x == 0.f && attrNormal.y == 0.f && attrNormal.z == 0.f)
					attrNormal = nrmTri[2u];
				if (!(attrNormal.x == 0.f && attrNormal.y == 0.f && attrNormal.z == 0.f))
				{
					if (flipHandedness)
						attrNormal.x = -attrNormal.x;
					normal = attrNormal;
				}

				if (normal.x == 0.f && normal.y == 0.f && normal.z == 0.f)
					normal = computePlaneNormal(vertex1, vertex2, vertex3);

				writeRecord(
					normal.x, normal.y, normal.z,
					vertex1.x, vertex1.y, vertex1.z,
					vertex2.x, vertex2.y, vertex2.z,
					vertex3.x, vertex3.y, vertex3.z,
					faceColor);
			}
		}
	}
	else if (hasFastTightPath)
	{
		const hlsl::float32_t3* posTri = tightPositions;
		for (uint32_t primIx = 0u; primIx < facenum; ++primIx, posTri += 3u)
		{
			uint16_t faceColor = 0u;
			if (!computeFaceColor(primIx * 3u + 0u, primIx * 3u + 1u, primIx * 3u + 2u, faceColor))
				return false;

			hlsl::float32_t3 vertex1 = {};
			hlsl::float32_t3 vertex2 = {};
			hlsl::float32_t3 vertex3 = {};
			prepareVertices(posTri[0u], posTri[1u], posTri[2u], vertex1, vertex2, vertex3);
			const hlsl::float32_t3 normal = computePlaneNormal(vertex1, vertex2, vertex3);

			writeRecord(
				normal.x, normal.y, normal.z,
				vertex1.x, vertex1.y, vertex1.z,
				vertex2.x, vertex2.y, vertex2.z,
				vertex3.x, vertex3.y, vertex3.z,
				faceColor);
		}
	}
	else
	{
		if (!SGeometryWriterCommon::visitTriangleIndices(geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2)->bool
		{
			uint16_t faceColor = 0u;
			if (!computeFaceColor(i0, i1, i2, faceColor))
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
				vertex3.x, vertex3.y, vertex3.z,
				faceColor);
			return true;
		}))
			return false;
	}

	const bool writeOk = SInterchangeIO::writeFileWithPolicy(context->writeContext.outputFile, context->ioPlan, output.get(), outputSize, &context->writeTelemetry);
	if (writeOk)
		context->fileOffset += outputSize;
	return writeOk;
}

bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, SContext* context)
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
	const bool flipHandedness = !context->writeContext.params.flags.hasAnyFlag(E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);

	const std::string name = context->writeContext.outputFile->getFileName().filename().replace_extension().string();
	const std::string_view solidName = name.empty() ? std::string_view(stl_writer_detail::AsciiDefaultName) : std::string_view(name);

	if (!context->write(stl_writer_detail::AsciiSolidPrefix, sizeof(stl_writer_detail::AsciiSolidPrefix) - 1ull))
		return false;

	if (!context->write(solidName.data(), solidName.size()))
		return false;

	if (!context->write("\n", sizeof("\n") - 1ull))
		return false;

	const uint32_t faceCount = static_cast<uint32_t>(geom->getPrimitiveCount());
	for (uint32_t primIx = 0u; primIx < faceCount; ++primIx)
	{
		hlsl::float32_t3 v0 = {};
		hlsl::float32_t3 v1 = {};
		hlsl::float32_t3 v2 = {};
		uint32_t idx[3] = {};
		if (!decodeTriangle(geom, indexing, posView, primIx, v0, v1, v2, idx))
			return false;
		if (!writeFaceText(v0, v1, v2, idx, normalView, flipHandedness, context))
			return false;
		if (!context->write("\n", sizeof("\n") - 1ull))
			return false;
	}

	if (!context->write(stl_writer_detail::AsciiEndSolidPrefix, sizeof(stl_writer_detail::AsciiEndSolidPrefix) - 1ull))
		return false;

	if (!context->write(solidName.data(), solidName.size()))
		return false;

	return true;
}

bool writeFaceText(
		const hlsl::float32_t3& v1,
		const hlsl::float32_t3& v2,
		const hlsl::float32_t3& v3,
		const uint32_t* idx,
		const asset::ICPUPolygonGeometry::SDataView& normalView,
		const bool flipHandedness,
		SContext* context)
{
	hlsl::float32_t3 vertex1 = v3;
	hlsl::float32_t3 vertex2 = v2;
	hlsl::float32_t3 vertex3 = v1;

	if (flipHandedness)
	{
		vertex1.x = -vertex1.x;
		vertex2.x = -vertex2.x;
		vertex3.x = -vertex3.x;
	}

	const hlsl::float32_t3 planeNormal = hlsl::cross(vertex2 - vertex1, vertex3 - vertex1);
	const float planeNormalLen2 = hlsl::dot(planeNormal, planeNormal);
	hlsl::float32_t3 normal = hlsl::float32_t3(0.f, 0.f, 0.f);
	if (planeNormalLen2 > 0.f)
		normal = hlsl::normalize(planeNormal);

	hlsl::float32_t3 attrNormal = {};
	if (decodeTriangleNormal(normalView, idx, attrNormal))
	{
		if (flipHandedness)
			attrNormal.x = -attrNormal.x;
		if (planeNormalLen2 > 0.f && hlsl::dot(attrNormal, planeNormal) < 0.f)
			attrNormal = -attrNormal;
		normal = attrNormal;
	}

	std::array<char, stl_writer_detail::AsciiFaceTextMaxBytes> faceText = {};
	char* cursor = faceText.data();
	char* const end = faceText.data() + faceText.size();
	const hlsl::float32_t3 vertices[3] = { vertex1, vertex2, vertex3 };
	if (!appendLiteral(cursor, end, "facet normal ", sizeof("facet normal ") - 1ull))
		return false;
	if (!appendVectorAsAsciiLine(cursor, end, normal))
		return false;
	if (!appendLiteral(cursor, end, "  outer loop\n", sizeof("  outer loop\n") - 1ull))
		return false;
	for (const auto& vertex : vertices)
		if (!appendLiteral(cursor, end, "    vertex ", sizeof("    vertex ") - 1ull) || !appendVectorAsAsciiLine(cursor, end, vertex))
			return false;
	if (!appendLiteral(cursor, end, "  endloop\n", sizeof("  endloop\n") - 1ull))
		return false;
	if (!appendLiteral(cursor, end, "endfacet\n", sizeof("endfacet\n") - 1ull))
		return false;

	return context->write(faceText.data(), static_cast<size_t>(cursor - faceText.data()));
}

}

#endif

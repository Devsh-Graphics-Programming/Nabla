#ifdef _NBL_COMPILE_WITH_STL_WRITER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "nbl/system/IFile.h"

#include "CSTLMeshWriter.h"
#include "impl/SFileAccess.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string_view>

namespace nbl::asset
{

namespace
{

struct Parse
{
	static constexpr uint32_t COLOR0 = 0u;
	struct Context
	{
		IAssetWriter::SAssetWriteContext writeContext;
		SResolvedFileIOPolicy ioPlan = {};
		core::vector<uint8_t> ioBuffer = {};
		size_t fileOffset = 0ull;
		SFileWriteTelemetry writeTelemetry = {};

		bool flush()
		{
			if (ioBuffer.empty())
				return true;
			size_t bytesWritten = 0ull;
			const size_t totalBytes = ioBuffer.size();
			while (bytesWritten < totalBytes)
			{
				system::IFile::success_t success;
				writeContext.outputFile->write(success, ioBuffer.data() + bytesWritten, fileOffset + bytesWritten, totalBytes - bytesWritten);
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
		bool write(const void* data, size_t size)
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
						if (ioBuffer.size() == chunkSize && !flush())
							return false;
					}
					return true;
				}
			}
		}
	};

	struct TriangleData
	{
		hlsl::float32_t3 normal = {};
		hlsl::float32_t3 vertex1 = {};
		hlsl::float32_t3 vertex2 = {};
		hlsl::float32_t3 vertex3 = {};
	};

	static constexpr size_t BinaryHeaderBytes = 80ull;
	static constexpr size_t BinaryTriangleCountBytes = sizeof(uint32_t);
	static constexpr size_t BinaryTriangleFloatCount = 12ull;
	static constexpr size_t BinaryTriangleFloatBytes = sizeof(float) * BinaryTriangleFloatCount;
	static constexpr size_t BinaryTriangleAttributeBytes = sizeof(uint16_t);
	static constexpr size_t BinaryTriangleRecordBytes = BinaryTriangleFloatBytes + BinaryTriangleAttributeBytes;
	static constexpr size_t BinaryPrefixBytes = BinaryHeaderBytes + BinaryTriangleCountBytes;
	static constexpr size_t IoFallbackReserveBytes = 1ull << 20;
	static constexpr size_t AsciiFaceTextMaxBytes = 1024ull;
	static constexpr char AsciiSolidPrefix[] = "solid ";
	static constexpr char AsciiEndSolidPrefix[] = "endsolid ";
	static constexpr char AsciiDefaultName[] = "nabla_mesh";
	static_assert(BinaryTriangleRecordBytes == 50ull);

	static bool appendLiteral(char*& cursor, char* const end, const char* text, const size_t textSize)
	{
		if (!cursor || cursor + textSize > end)
			return false;
		std::memcpy(cursor, text, textSize);
		cursor += textSize;
		return true;
	}

	static bool appendVectorAsAsciiLine(char*& cursor, char* const end, const hlsl::float32_t3& v)
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

	static bool decodeTriangle(const ICPUPolygonGeometry* geom, const IPolygonGeometryBase::IIndexingCallback* indexing, const ICPUPolygonGeometry::SDataView& posView, uint32_t primIx, hlsl::float32_t3& out0, hlsl::float32_t3& out1, hlsl::float32_t3& out2, hlsl::uint32_t3* outIdx)
	{
		hlsl::uint32_t3 idx(0u);
		const auto& indexView = geom->getIndexView();
		const void* indexBuffer = indexView ? indexView.getPointer() : nullptr;
		const uint64_t indexSize = indexView ? indexView.composed.getStride() : 0u;
		IPolygonGeometryBase::IIndexingCallback::SContext<uint32_t> ctx = {.indexBuffer = indexBuffer, .indexSize = indexSize, .beginPrimitive = primIx, .endPrimitive = primIx + 1u, .out = &idx.x};
		indexing->operator()(ctx);
		if (outIdx)
			*outIdx = idx;

		std::array<hlsl::float32_t3, 3> positions = {};
		if (!decodeIndexedTriple(idx, [&posView](const uint32_t vertexIx, hlsl::float32_t3& out) -> bool { return posView.decodeElement(vertexIx, out); }, positions.data()))
			return false;
		out0 = positions[0];
		out1 = positions[1];
		out2 = positions[2];
		return true;
	}

	template<typename DecodeFn, typename T>
	static bool decodeIndexedTriple(const hlsl::uint32_t3& idx, DecodeFn&& decode, T* out)
	{
		return out && decode(idx.x, out[0]) && decode(idx.y, out[1]) && decode(idx.z, out[2]);
	}

	static bool decodeTriangleNormal(const ICPUPolygonGeometry::SDataView& normalView, const hlsl::uint32_t3& idx, hlsl::float32_t3& outNormal)
	{
		if (!normalView)
			return false;
		std::array<hlsl::float32_t3, 3> normals = {};
		if (!decodeIndexedTriple(idx, [&normalView](const uint32_t vertexIx, hlsl::float32_t3& out) -> bool { return normalView.decodeElement(vertexIx, out); }, normals.data()))
			return false;
		return selectFirstValidNormal(normals.data(), static_cast<uint32_t>(normals.size()), outNormal);
	}

	static bool selectFirstValidNormal(const hlsl::float32_t3* const normals, const uint32_t count, hlsl::float32_t3& outNormal)
	{
		if (!normals || count == 0u)
			return false;
		for (uint32_t i = 0u; i < count; ++i)
		{
			if (hlsl::dot(normals[i], normals[i]) > 0.f)
			{
				outNormal = normals[i];
				return true;
			}
		}
		return false;
	}

	static void prepareVertices(const hlsl::float32_t3& p0, const hlsl::float32_t3& p1, const hlsl::float32_t3& p2, const bool flipHandedness, hlsl::float32_t3& vertex1, hlsl::float32_t3& vertex2, hlsl::float32_t3& vertex3)
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
	}

	static hlsl::float32_t3 computePlaneNormal(const hlsl::float32_t3& vertex1, const hlsl::float32_t3& vertex2, const hlsl::float32_t3& vertex3, float* const planeNormalLen2 = nullptr)
	{
		const hlsl::float32_t3 planeNormal = hlsl::cross(vertex2 - vertex1, vertex3 - vertex1);
		const float len2 = hlsl::dot(planeNormal, planeNormal);
		if (planeNormalLen2)
		{
			*planeNormalLen2 = len2;
			return planeNormal;
		}
		return len2 > 0.f ? hlsl::normalize(planeNormal) : hlsl::float32_t3(0.f, 0.f, 0.f);
	}

	static hlsl::float32_t3 resolveTriangleNormal(const hlsl::float32_t3& planeNormal, const float planeNormalLen2, const hlsl::float32_t3* const attrNormals, const uint32_t attrNormalCount, const bool flipHandedness, const bool alignToPlane)
	{
		hlsl::float32_t3 attrNormal = {};
		if (selectFirstValidNormal(attrNormals, attrNormalCount, attrNormal))
		{
			if (flipHandedness)
				attrNormal.x = -attrNormal.x;
			if (alignToPlane && planeNormalLen2 > 0.f && hlsl::dot(attrNormal, planeNormal) < 0.f)
				attrNormal = -attrNormal;
			return attrNormal;
		}
		return planeNormalLen2 > 0.f ? hlsl::normalize(planeNormal) : hlsl::float32_t3(0.f, 0.f, 0.f);
	}

	static void buildTriangle(const hlsl::float32_t3& p0, const hlsl::float32_t3& p1, const hlsl::float32_t3& p2, const hlsl::float32_t3* const attrNormals, const uint32_t attrNormalCount, const bool flipHandedness, const bool alignToPlane, TriangleData& triangle)
	{
		prepareVertices(p0, p1, p2, flipHandedness, triangle.vertex1, triangle.vertex2, triangle.vertex3);
		float planeNormalLen2 = 0.f;
		const hlsl::float32_t3 planeNormal = computePlaneNormal(triangle.vertex1, triangle.vertex2, triangle.vertex3, &planeNormalLen2);
		triangle.normal = resolveTriangleNormal(planeNormal, planeNormalLen2, attrNormals, attrNormalCount, flipHandedness, alignToPlane);
	}

	static double normalizeColorComponentToUnit(double value)
	{
		if (!std::isfinite(value))
			return 0.0;
		if (value > 1.0)
			value /= 255.0;
		return std::clamp(value, 0.0, 1.0);
	}

	static uint16_t packViscamColorFromB8G8R8A8(const uint32_t color)
	{
		const void* src[4] = {&color, nullptr, nullptr, nullptr};
		uint16_t packed = 0u;
		convertColor<EF_B8G8R8A8_UNORM, EF_A1R5G5B5_UNORM_PACK16>(src, &packed, 0u, 0u);
		packed |= 0x8000u;
		return packed;
	}

	static const ICPUPolygonGeometry::SDataView* getColorView(const ICPUPolygonGeometry* geom, const size_t vertexCount)
	{
		const auto* view = SGeometryWriterCommon::getAuxViewAt(geom, Parse::COLOR0, vertexCount);
		return view && getFormatChannelCount(view->composed.format) >= 3u ? view : nullptr;
	}

	static bool decodeColorB8G8R8A8(const ICPUPolygonGeometry::SDataView& colorView, const uint32_t ix, uint32_t& outColor)
	{
		if (colorView.composed.format == EF_B8G8R8A8_UNORM && colorView.composed.getStride() == sizeof(uint32_t))
		{
			const auto* const ptr = reinterpret_cast<const uint8_t*>(colorView.getPointer());
			if (!ptr)
				return false;
			std::memcpy(&outColor, ptr + static_cast<size_t>(ix) * sizeof(uint32_t), sizeof(outColor));
			return true;
		}
		hlsl::float32_t4 decoded = {};
		if (!colorView.decodeElement(ix, decoded))
			return false;
		const double rgbaUnit[4] = {normalizeColorComponentToUnit(decoded.x), normalizeColorComponentToUnit(decoded.y), normalizeColorComponentToUnit(decoded.z), normalizeColorComponentToUnit(decoded.w)};
		encodePixels<EF_B8G8R8A8_UNORM, double>(&outColor, rgbaUnit);
		return true;
	}

	static void decodeColorUnitRGBAFromB8G8R8A8(const uint32_t color, double* out)
	{
		const void* src[4] = {&color, nullptr, nullptr, nullptr};
		decodePixels<EF_B8G8R8A8_UNORM, double>(src, out, 0u, 0u);
	}

	static bool writeMeshBinary(const asset::ICPUPolygonGeometry* geom, Context* context)
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
		size_t faceCount = 0ull;
		if (!SGeometryWriterCommon::getTriangleFaceCount(geom, faceCount))
			return false;
		if (faceCount > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
			return false;
		const uint32_t facenum = static_cast<uint32_t>(faceCount);

		const size_t outputSize = BinaryPrefixBytes + static_cast<size_t>(facenum) * BinaryTriangleRecordBytes;
		std::unique_ptr<uint8_t[]> output(new (std::nothrow) uint8_t[outputSize]);
		if (!output)
			return false;
		uint8_t* dst = output.get();
		std::memset(dst, 0, BinaryHeaderBytes);
		dst += BinaryHeaderBytes;
		std::memcpy(dst, &facenum, sizeof(facenum));
		dst += sizeof(facenum);

		const auto& normalView = geom->getNormalView();
		const bool hasNormals = static_cast<bool>(normalView);
		const auto* const colorView = getColorView(geom, vertexCount);
		const hlsl::float32_t3* const tightPositions = SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(posView);
		const hlsl::float32_t3* const tightNormals = hasNormals ? SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(normalView) : nullptr;
		const bool hasFastTightPath = !geom->getIndexView() && tightPositions && (!hasNormals || tightNormals);

		auto decodePosition = [&](const uint32_t ix, hlsl::float32_t3& out) -> bool {
			if (tightPositions)
			{
				out = tightPositions[ix];
				return true;
			}
			return posView.decodeElement(ix, out);
		};
		auto decodeNormal = [&](const uint32_t ix, hlsl::float32_t3& out) -> bool {
			if (!hasNormals)
				return false;
			if (tightNormals)
			{
				out = tightNormals[ix];
				return true;
			}
			return normalView.decodeElement(ix, out);
		};
		auto computeFaceColor = [&](const hlsl::uint32_t3& idx, uint16_t& outColor) -> bool {
			outColor = 0u;
			if (!colorView)
				return true;
			const std::array<uint32_t, 3> vertexIx = {idx.x, idx.y, idx.z};
			std::array<double, 4> rgbaAvg = {};
			for (uint32_t corner = 0u; corner < vertexIx.size(); ++corner)
			{
				uint32_t color = 0u;
				if (!decodeColorB8G8R8A8(*colorView, vertexIx[corner], color))
					return false;
				std::array<double, 4> rgba = {};
				decodeColorUnitRGBAFromB8G8R8A8(color, rgba.data());
				rgbaAvg[0] += rgba[0];
				rgbaAvg[1] += rgba[1];
				rgbaAvg[2] += rgba[2];
			}
			rgbaAvg[0] /= 3.0;
			rgbaAvg[1] /= 3.0;
			rgbaAvg[2] /= 3.0;
			rgbaAvg[3] = 1.0;
			uint32_t avgColor = 0u;
			encodePixels<EF_B8G8R8A8_UNORM, double>(&avgColor, rgbaAvg.data());
			outColor = packViscamColorFromB8G8R8A8(avgColor);
			return true;
		};
		auto writeRecord = [&dst](const hlsl::float32_t3& normal, const hlsl::float32_t3& vertex1, const hlsl::float32_t3& vertex2, const hlsl::float32_t3& vertex3, const uint16_t attribute) -> void {
			const float payload[BinaryTriangleFloatCount] = {normal.x, normal.y, normal.z, vertex1.x, vertex1.y, vertex1.z, vertex2.x, vertex2.y, vertex2.z, vertex3.x, vertex3.y, vertex3.z};
			std::memcpy(dst, payload, BinaryTriangleFloatBytes);
			dst += BinaryTriangleFloatBytes;
			std::memcpy(dst, &attribute, BinaryTriangleAttributeBytes);
			dst += BinaryTriangleAttributeBytes;
		};
		auto emitTriangle = [&](const hlsl::float32_t3& p0, const hlsl::float32_t3& p1, const hlsl::float32_t3& p2, const hlsl::uint32_t3& idx, const hlsl::float32_t3* const attrNormals, const uint32_t attrNormalCount, const bool alignToPlane) -> bool {
			uint16_t faceColor = 0u;
			if (!computeFaceColor(idx, faceColor))
				return false;
			TriangleData triangle = {};
			buildTriangle(p0, p1, p2, attrNormals, attrNormalCount, flipHandedness, alignToPlane, triangle);
			writeRecord(triangle.normal, triangle.vertex1, triangle.vertex2, triangle.vertex3, faceColor);
			return true;
		};

		if (hasFastTightPath)
		{
			const hlsl::float32_t3* posTri = tightPositions;
			const hlsl::float32_t3* nrmTri = tightNormals;
			for (uint32_t primIx = 0u; primIx < facenum; ++primIx, posTri += 3u)
			{
				const hlsl::uint32_t3 idx(primIx * 3u + 0u, primIx * 3u + 1u, primIx * 3u + 2u);
				if (!emitTriangle(posTri[0u], posTri[1u], posTri[2u], idx, nrmTri, hasNormals ? 3u : 0u, false))
					return false;
				if (nrmTri)
					nrmTri += 3u;
			}
		}
		else if (!SGeometryWriterCommon::visitTriangleIndices(geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2) -> bool {
			const hlsl::uint32_t3 idx(i0, i1, i2);
			std::array<hlsl::float32_t3, 3> positions = {};
			if (!decodeIndexedTriple(idx, decodePosition, positions.data()))
				return false;
			std::array<hlsl::float32_t3, 3> normals = {};
			if (hasNormals && !decodeIndexedTriple(idx, decodeNormal, normals.data()))
				return false;
			return emitTriangle(positions[0], positions[1], positions[2], idx, hasNormals ? normals.data() : nullptr, hasNormals ? 3u : 0u, true);
		}))
			return false;

		const bool writeOk = SInterchangeIO::writeFileWithPolicy(context->writeContext.outputFile, context->ioPlan, output.get(), outputSize, &context->writeTelemetry);
		if (writeOk)
			context->fileOffset += outputSize;
		return writeOk;
	}

	static bool writeMeshASCII(const asset::ICPUPolygonGeometry* geom, Context* context)
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
		const std::string_view solidName = name.empty() ? std::string_view(AsciiDefaultName) : std::string_view(name);
		if (!context->write(AsciiSolidPrefix, sizeof(AsciiSolidPrefix) - 1ull))
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
			hlsl::uint32_t3 idx(0u);
			if (!decodeTriangle(geom, indexing, posView, primIx, v0, v1, v2, &idx))
				return false;
			if (!writeFaceText(v0, v1, v2, idx, normalView, flipHandedness, context))
				return false;
			if (!context->write("\n", sizeof("\n") - 1ull))
				return false;
		}

		if (!context->write(AsciiEndSolidPrefix, sizeof(AsciiEndSolidPrefix) - 1ull))
			return false;
		if (!context->write(solidName.data(), solidName.size()))
			return false;
		return true;
	}

	static bool writeFaceText(const hlsl::float32_t3& v1, const hlsl::float32_t3& v2, const hlsl::float32_t3& v3, const hlsl::uint32_t3& idx, const asset::ICPUPolygonGeometry::SDataView& normalView, const bool flipHandedness, Context* context)
	{
		hlsl::float32_t3 attrNormal = {};
		TriangleData triangle = {};
		const hlsl::float32_t3* const attrNormalPtr = decodeTriangleNormal(normalView, idx, attrNormal) ? &attrNormal : nullptr;
		buildTriangle(v1, v2, v3, attrNormalPtr, attrNormalPtr ? 1u : 0u, flipHandedness, true, triangle);
		std::array<char, AsciiFaceTextMaxBytes> faceText = {};
		char* cursor = faceText.data();
		char* const end = faceText.data() + faceText.size();
		const std::array vertices = {triangle.vertex1, triangle.vertex2, triangle.vertex3};
		if (!appendLiteral(cursor, end, "facet normal ", sizeof("facet normal ") - 1ull))
			return false;
		if (!appendVectorAsAsciiLine(cursor, end, triangle.normal))
			return false;
		if (!appendLiteral(cursor, end, "  outer loop\n", sizeof("  outer loop\n") - 1ull))
			return false;
		for (const auto& vertex : vertices)
		{
			if (!appendLiteral(cursor, end, "    vertex ", sizeof("    vertex ") - 1ull))
				return false;
			if (!appendVectorAsAsciiLine(cursor, end, vertex))
				return false;
		}
		if (!appendLiteral(cursor, end, "  endloop\n", sizeof("  endloop\n") - 1ull))
			return false;
		if (!appendLiteral(cursor, end, "endfacet\n", sizeof("endfacet\n") - 1ull))
			return false;
		return context->write(faceText.data(), static_cast<size_t>(cursor - faceText.data()));
	}
};

}

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
	using Context = Parse::Context;

	if (!_override)
		getDefaultOverride(_override);

	IAssetWriter::SAssetWriteContext inCtx{_params, _file};
	const asset::ICPUPolygonGeometry* geom = IAsset::castDown<const asset::ICPUPolygonGeometry>(_params.rootAsset);
	if (!geom)
		return false;

	system::IFile* file = _override->getOutputFile(_file, inCtx, {geom, 0u});
	if (!file)
		return false;

	Context context = {IAssetWriter::SAssetWriteContext{inCtx.params, file}};
	_params.logger.log("WRITING STL: writing the file %s", system::ILogger::ELL_INFO, file->getFileName().string().c_str());

	const auto flags = _override->getAssetWritingFlags(context.writeContext, geom, 0u);
	const bool binary = flags.hasAnyFlag(asset::EWF_BINARY);

	uint64_t expectedSize = 0ull;
	bool sizeKnown = false;
	if (binary)
	{
		expectedSize = Parse::BinaryPrefixBytes + static_cast<uint64_t>(geom->getPrimitiveCount()) * Parse::BinaryTriangleRecordBytes;
		sizeKnown = true;
	}

	context.ioPlan = impl::SFileAccess::resolvePlan(_params.ioPolicy, expectedSize, sizeKnown, file);
	if (impl::SFileAccess::logInvalidPlan(_params.logger, "STL writer", file->getFileName().string().c_str(), context.ioPlan))
		return false;

	if (context.ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile && sizeKnown)
		context.ioBuffer.reserve(static_cast<size_t>(expectedSize));
	else
		context.ioBuffer.reserve(static_cast<size_t>(std::min<uint64_t>(context.ioPlan.chunkSizeBytes(), Parse::IoFallbackReserveBytes)));

	const bool written = binary ? Parse::writeMeshBinary(geom, &context) : Parse::writeMeshASCII(geom, &context);
	if (!written)
		return false;
	if (!context.flush())
		return false;

	const uint64_t ioMinWrite = context.writeTelemetry.getMinOrZero();
	const uint64_t ioAvgWrite = context.writeTelemetry.getAvgOrZero();
	impl::SFileAccess::logTinyIO(_params.logger, "STL writer", file->getFileName().string().c_str(), context.writeTelemetry, context.fileOffset, _params.ioPolicy, "writes");
	_params.logger.log("STL writer stats: file=%s bytes=%llu binary=%d io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE, file->getFileName().string().c_str(), static_cast<unsigned long long>(context.fileOffset), binary ? 1 : 0,
		static_cast<unsigned long long>(context.writeTelemetry.callCount), static_cast<unsigned long long>(ioMinWrite), static_cast<unsigned long long>(ioAvgWrite),
		system::to_string(_params.ioPolicy.strategy).c_str(), system::to_string(context.ioPlan.strategy).c_str(), static_cast<unsigned long long>(context.ioPlan.chunkSizeBytes()), context.ioPlan.reason);

	return true;
}

}

#endif

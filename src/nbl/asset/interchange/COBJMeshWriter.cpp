// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/COBJMeshWriter.h"
#include "nbl/asset/interchange/SInterchangeIOCommon.h"

#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <limits>
#include <system_error>

namespace nbl::asset
{

COBJMeshWriter::COBJMeshWriter()
{
	#ifdef _NBL_DEBUG
	setDebugName("COBJMeshWriter");
	#endif
}

const char** COBJMeshWriter::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "obj", nullptr };
	return ext;
}

uint32_t COBJMeshWriter::getSupportedFlags()
{
	return 0u;
}

uint32_t COBJMeshWriter::getForcedFlags()
{
	return 0u;
}

namespace obj_writer_detail
{

constexpr size_t ApproxObjBytesPerVertex = 96ull;
constexpr size_t ApproxObjBytesPerFace = 48ull;
constexpr size_t MaxUInt32Chars = 10ull;
constexpr size_t MaxFloatFixed6Chars = 48ull;
constexpr size_t MaxIndexTokenBytes = MaxUInt32Chars * 3ull + 2ull;

struct SIndexStringRef
{
	uint32_t offset = 0u;
	uint16_t length = 0u;
};

bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
	out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
	return view.decodeElement(ix, out);
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

const hlsl::float32_t2* getTightFloat2View(const ICPUPolygonGeometry::SDataView& view)
{
	if (!view)
		return nullptr;
	if (view.composed.format != EF_R32G32_SFLOAT)
		return nullptr;
	if (view.composed.getStride() != sizeof(hlsl::float32_t2))
		return nullptr;
	return reinterpret_cast<const hlsl::float32_t2*>(view.getPointer());
}

char* appendUIntToBuffer(char* dst, char* const end, const uint32_t value)
{
	if (!dst || dst >= end)
		return end;

	const auto result = std::to_chars(dst, end, value);
	if (result.ec == std::errc())
		return result.ptr;

	const int written = std::snprintf(dst, static_cast<size_t>(end - dst), "%u", value);
	if (written <= 0)
		return dst;
	const size_t writeLen = static_cast<size_t>(written);
	return (writeLen < static_cast<size_t>(end - dst)) ? (dst + writeLen) : end;
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

void appendVec3Line(std::string& out, const char* prefix, const size_t prefixSize, const float x, const float y, const float z)
{
	const size_t oldSize = out.size();
	out.resize(oldSize + prefixSize + (3ull * MaxFloatFixed6Chars) + 3ull);
	char* const lineBegin = out.data() + oldSize;
	char* cursor = lineBegin;
	char* const lineEnd = out.data() + out.size();

	std::memcpy(cursor, prefix, prefixSize);
	cursor += prefixSize;

	cursor = appendFloatFixed6ToBuffer(cursor, lineEnd, x);
	if (cursor < lineEnd)
		*(cursor++) = ' ';
	cursor = appendFloatFixed6ToBuffer(cursor, lineEnd, y);
	if (cursor < lineEnd)
		*(cursor++) = ' ';
	cursor = appendFloatFixed6ToBuffer(cursor, lineEnd, z);
	if (cursor < lineEnd)
		*(cursor++) = '\n';

	out.resize(oldSize + static_cast<size_t>(cursor - lineBegin));
}

void appendVec2Line(std::string& out, const char* prefix, const size_t prefixSize, const float x, const float y)
{
	const size_t oldSize = out.size();
	out.resize(oldSize + prefixSize + (2ull * MaxFloatFixed6Chars) + 2ull);
	char* const lineBegin = out.data() + oldSize;
	char* cursor = lineBegin;
	char* const lineEnd = out.data() + out.size();

	std::memcpy(cursor, prefix, prefixSize);
	cursor += prefixSize;

	cursor = appendFloatFixed6ToBuffer(cursor, lineEnd, x);
	if (cursor < lineEnd)
		*(cursor++) = ' ';
	cursor = appendFloatFixed6ToBuffer(cursor, lineEnd, y);
	if (cursor < lineEnd)
		*(cursor++) = '\n';

	out.resize(oldSize + static_cast<size_t>(cursor - lineBegin));
}

void appendFaceLine(std::string& out, const std::string& storage, const core::vector<SIndexStringRef>& refs, const uint32_t i0, const uint32_t i1, const uint32_t i2)
{
	const auto& ref0 = refs[i0];
	const auto& ref1 = refs[i1];
	const auto& ref2 = refs[i2];
	const size_t oldSize = out.size();
	const size_t lineSize = 2ull + static_cast<size_t>(ref0.length) + 1ull + static_cast<size_t>(ref1.length) + 1ull + static_cast<size_t>(ref2.length) + 1ull;
	out.resize(oldSize + lineSize);
	char* cursor = out.data() + oldSize;
	*(cursor++) = 'f';
	*(cursor++) = ' ';
	std::memcpy(cursor, storage.data() + ref0.offset, ref0.length);
	cursor += ref0.length;
	*(cursor++) = ' ';
	std::memcpy(cursor, storage.data() + ref1.offset, ref1.length);
	cursor += ref1.length;
	*(cursor++) = ' ';
	std::memcpy(cursor, storage.data() + ref2.offset, ref2.length);
	cursor += ref2.length;
	*(cursor++) = '\n';
}

void appendIndexTokenToStorage(std::string& storage, core::vector<SIndexStringRef>& refs, const uint32_t objIx, const bool hasUVs, const bool hasNormals)
{
	SIndexStringRef ref = {};
	ref.offset = static_cast<uint32_t>(storage.size());
	{
		const size_t oldSize = storage.size();
		storage.resize(oldSize + MaxIndexTokenBytes);
		char* const token = storage.data() + oldSize;
		char* const tokenEnd = token + MaxIndexTokenBytes;
		char* cursor = token;
		cursor = appendUIntToBuffer(cursor, tokenEnd, objIx);
		if (hasUVs && hasNormals)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = appendUIntToBuffer(cursor, tokenEnd, objIx);
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = appendUIntToBuffer(cursor, tokenEnd, objIx);
		}
		else if (hasUVs)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = appendUIntToBuffer(cursor, tokenEnd, objIx);
		}
		else if (hasNormals)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = appendUIntToBuffer(cursor, tokenEnd, objIx);
		}
		storage.resize(oldSize + static_cast<size_t>(cursor - token));
	}
	ref.length = static_cast<uint16_t>(storage.size() - ref.offset);
	refs.push_back(ref);
}

} // namespace obj_writer_detail

bool COBJMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	using namespace obj_writer_detail;
	using clock_t = std::chrono::high_resolution_clock;

	const auto totalStart = clock_t::now();
	double encodeMs = 0.0;
	double formatMs = 0.0;
	double writeMs = 0.0;
	SFileWriteTelemetry ioTelemetry = {};

	if (!_override)
		getDefaultOverride(_override);

	if (!_file || !_params.rootAsset)
		return false;

	const auto* geom = IAsset::castDown<const ICPUPolygonGeometry>(_params.rootAsset);
	if (!geom || !geom->valid())
		return false;

	SAssetWriteContext ctx = { _params, _file };
	system::IFile* file = _override->getOutputFile(_file, ctx, { geom, 0u });
	if (!file)
		return false;

	const auto& positionView = geom->getPositionView();
	if (!positionView)
		return false;

	const auto& normalView = geom->getNormalView();
	const bool hasNormals = static_cast<bool>(normalView);

	const auto& auxViews = geom->getAuxAttributeViews();
	const ICPUPolygonGeometry::SDataView* uvView = nullptr;
	for (const auto& view : auxViews)
	{
		if (!view)
			continue;
		const auto channels = getFormatChannelCount(view.composed.format);
		if (channels >= 2u)
		{
			uvView = &view;
			break;
		}
	}
	const bool hasUVs = uvView != nullptr;

	const size_t vertexCount = positionView.getElementCount();
	if (vertexCount == 0)
		return false;
	if (hasNormals && normalView.getElementCount() != vertexCount)
		return false;
	if (hasUVs && uvView->getElementCount() != vertexCount)
		return false;

	const auto* indexing = geom->getIndexingCallback();
	if (!indexing)
		return false;
	if (indexing->knownTopology() != E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST)
		return false;

	const auto& indexView = geom->getIndexView();
	core::vector<uint32_t> indexData;
	const uint32_t* indices = nullptr;
	size_t faceCount = 0;
	const auto encodeStart = clock_t::now();

	if (indexView)
	{
		const size_t indexCount = indexView.getElementCount();
		if (indexCount % 3u != 0u)
			return false;

		const void* src = indexView.getPointer();
		if (!src)
			return false;

		if (indexView.composed.format == EF_R32_UINT && indexView.composed.getStride() == sizeof(uint32_t))
		{
			indices = reinterpret_cast<const uint32_t*>(src);
		}
		else if (indexView.composed.format == EF_R16_UINT && indexView.composed.getStride() == sizeof(uint16_t))
		{
			indexData.resize(indexCount);
			const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
			for (size_t i = 0; i < indexCount; ++i)
				indexData[i] = src16[i];
			indices = indexData.data();
		}
		else
		{
			indexData.resize(indexCount);
			hlsl::vector<uint32_t, 1> decoded = {};
			for (size_t i = 0; i < indexCount; ++i)
			{
				if (!indexView.decodeElement(i, decoded))
					return false;
				indexData[i] = decoded.x;
			}
			indices = indexData.data();
		}
		faceCount = indexCount / 3u;
	}
	else
	{
		if (vertexCount % 3u != 0u)
			return false;

		indexData.resize(vertexCount);
		for (size_t i = 0; i < vertexCount; ++i)
			indexData[i] = static_cast<uint32_t>(i);

		indices = indexData.data();
		faceCount = vertexCount / 3u;
	}
	encodeMs = std::chrono::duration<double, std::milli>(clock_t::now() - encodeStart).count();

	const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
	const bool flipHandedness = !(flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
	std::string output;
	const auto formatStart = clock_t::now();
	output.reserve(vertexCount * ApproxObjBytesPerVertex + faceCount * ApproxObjBytesPerFace);

	output.append("# Nabla OBJ\n");

	hlsl::float64_t4 tmp = {};
	const hlsl::float32_t3* const tightPositions = getTightFloat3View(positionView);
	const hlsl::float32_t3* const tightNormals = hasNormals ? getTightFloat3View(normalView) : nullptr;
	const hlsl::float32_t2* const tightUV = hasUVs ? getTightFloat2View(*uvView) : nullptr;
	for (size_t i = 0u; i < vertexCount; ++i)
	{
		float x = 0.f;
		float y = 0.f;
		float z = 0.f;
		if (tightPositions)
		{
			x = tightPositions[i].x;
			y = tightPositions[i].y;
			z = tightPositions[i].z;
		}
		else
		{
			if (!decodeVec4(positionView, i, tmp))
				return false;
			x = static_cast<float>(tmp.x);
			y = static_cast<float>(tmp.y);
			z = static_cast<float>(tmp.z);
		}
		if (flipHandedness)
			x = -x;

		appendVec3Line(output, "v ", sizeof("v ") - 1ull, x, y, z);
	}

	if (hasUVs)
	{
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			float u = 0.f;
			float v = 0.f;
			if (tightUV)
			{
				u = tightUV[i].x;
				v = 1.f - tightUV[i].y;
			}
			else
			{
				if (!decodeVec4(*uvView, i, tmp))
					return false;
				u = static_cast<float>(tmp.x);
				v = 1.f - static_cast<float>(tmp.y);
			}

			appendVec2Line(output, "vt ", sizeof("vt ") - 1ull, u, v);
		}
	}

	if (hasNormals)
	{
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			float x = 0.f;
			float y = 0.f;
			float z = 0.f;
			if (tightNormals)
			{
				x = tightNormals[i].x;
				y = tightNormals[i].y;
				z = tightNormals[i].z;
			}
			else
			{
				if (!decodeVec4(normalView, i, tmp))
					return false;
				x = static_cast<float>(tmp.x);
				y = static_cast<float>(tmp.y);
				z = static_cast<float>(tmp.z);
			}
			if (flipHandedness)
				x = -x;

			appendVec3Line(output, "vn ", sizeof("vn ") - 1ull, x, y, z);
		}
	}

	core::vector<SIndexStringRef> faceIndexRefs;
	faceIndexRefs.reserve(vertexCount);
	std::string faceIndexStorage;
	faceIndexStorage.reserve(vertexCount * 24ull);
	for (size_t i = 0u; i < vertexCount; ++i)
	{
		const uint32_t objIx = static_cast<uint32_t>(i + 1u);
		appendIndexTokenToStorage(faceIndexStorage, faceIndexRefs, objIx, hasUVs, hasNormals);
	}

	for (size_t i = 0u; i < faceCount; ++i)
	{
		const uint32_t i0 = indices[i * 3u + 0u];
		const uint32_t i1 = indices[i * 3u + 1u];
		const uint32_t i2 = indices[i * 3u + 2u];

		const uint32_t f0 = i2;
		const uint32_t f1 = i1;
		const uint32_t f2 = i0;
		if (f0 >= faceIndexRefs.size() || f1 >= faceIndexRefs.size() || f2 >= faceIndexRefs.size())
			return false;

		appendFaceLine(output, faceIndexStorage, faceIndexRefs, f0, f1, f2);
	}
	formatMs = std::chrono::duration<double, std::milli>(clock_t::now() - formatStart).count();

	const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(output.size()), true);
	if (!ioPlan.valid)
	{
		_params.logger.log("OBJ writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
		return false;
	}

	const auto writeStart = clock_t::now();
	const bool writeOk = writeFileWithPolicy(file, ioPlan, reinterpret_cast<const uint8_t*>(output.data()), output.size(), &ioTelemetry);
	writeMs = std::chrono::duration<double, std::milli>(clock_t::now() - writeStart).count();

	const double totalMs = std::chrono::duration<double, std::milli>(clock_t::now() - totalStart).count();
	const double miscMs = std::max(0.0, totalMs - (encodeMs + formatMs + writeMs));
	const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
	const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
	if (isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(output.size())))
	{
		_params.logger.log(
			"OBJ writer tiny-io guard: file=%s writes=%llu min=%llu avg=%llu",
			system::ILogger::ELL_WARNING,
			file->getFileName().string().c_str(),
			static_cast<unsigned long long>(ioTelemetry.callCount),
			static_cast<unsigned long long>(ioMinWrite),
			static_cast<unsigned long long>(ioAvgWrite));
	}
	_params.logger.log(
		"OBJ writer stats: file=%s bytes=%llu vertices=%llu faces=%llu io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		file->getFileName().string().c_str(),
		static_cast<unsigned long long>(output.size()),
		static_cast<unsigned long long>(vertexCount),
		static_cast<unsigned long long>(faceCount),
		static_cast<unsigned long long>(ioTelemetry.callCount),
		static_cast<unsigned long long>(ioMinWrite),
		static_cast<unsigned long long>(ioAvgWrite),
		toString(_params.ioPolicy.strategy),
		toString(ioPlan.strategy),
		static_cast<unsigned long long>(ioPlan.chunkSizeBytes),
		ioPlan.reason);
	(void)totalMs;
	(void)encodeMs;
	(void)formatMs;
	(void)writeMs;
	(void)miscMs;

	return writeOk;
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_OBJ_WRITER_


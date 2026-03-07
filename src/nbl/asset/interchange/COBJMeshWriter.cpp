// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/COBJMeshWriter.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "SOBJPolygonGeometryAuxLayout.h"

#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <charconv>
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

uint64_t COBJMeshWriter::getSupportedAssetTypesBitfield() const
{
	// OBJ can store a single geometry, a geometry collection, or a flattened scene export.
	return IAsset::ET_GEOMETRY | IAsset::ET_GEOMETRY_COLLECTION | IAsset::ET_SCENE;
}

const char** COBJMeshWriter::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "obj", nullptr };
	return ext;
}

writer_flags_t COBJMeshWriter::getSupportedFlags()
{
	return EWF_NONE;
}

writer_flags_t COBJMeshWriter::getForcedFlags()
{
	return EWF_NONE;
}

namespace obj_writer_detail
{

constexpr size_t ApproxObjBytesPerVertex = 96ull;
constexpr size_t ApproxObjBytesPerFace = 48ull;
constexpr size_t MaxFloatTextChars = std::numeric_limits<float>::max_digits10 + 8ull;
constexpr size_t MaxUInt32Chars = std::numeric_limits<uint32_t>::digits10 + 1ull;
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

void appendVec3Line(std::string& out, const char* prefix, const size_t prefixSize, const hlsl::float32_t3& v)
{
	const size_t oldSize = out.size();
	out.resize(oldSize + prefixSize + (3ull * MaxFloatTextChars) + 3ull);
	char* const lineBegin = out.data() + oldSize;
	char* cursor = lineBegin;
	char* const lineEnd = out.data() + out.size();

	std::memcpy(cursor, prefix, prefixSize);
	cursor += prefixSize;

	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, lineEnd, v.x);
	if (cursor < lineEnd)
		*(cursor++) = ' ';
	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, lineEnd, v.y);
	if (cursor < lineEnd)
		*(cursor++) = ' ';
	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, lineEnd, v.z);
	if (cursor < lineEnd)
		*(cursor++) = '\n';

	out.resize(oldSize + static_cast<size_t>(cursor - lineBegin));
}

void appendVec2Line(std::string& out, const char* prefix, const size_t prefixSize, const hlsl::float32_t2& v)
{
	const size_t oldSize = out.size();
	out.resize(oldSize + prefixSize + (2ull * MaxFloatTextChars) + 2ull);
	char* const lineBegin = out.data() + oldSize;
	char* cursor = lineBegin;
	char* const lineEnd = out.data() + out.size();

	std::memcpy(cursor, prefix, prefixSize);
	cursor += prefixSize;

	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, lineEnd, v.x);
	if (cursor < lineEnd)
		*(cursor++) = ' ';
	cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, lineEnd, v.y);
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

void appendIndexTokenToStorage(std::string& storage, core::vector<SIndexStringRef>& refs, const uint32_t positionIx, const bool hasUVs, const uint32_t uvIx, const bool hasNormals, const uint32_t normalIx)
{
	SIndexStringRef ref = {};
	ref.offset = static_cast<uint32_t>(storage.size());
	{
		const size_t oldSize = storage.size();
		storage.resize(oldSize + MaxIndexTokenBytes);
		char* const token = storage.data() + oldSize;
		char* const tokenEnd = token + MaxIndexTokenBytes;
		char* cursor = token;
		cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, positionIx);
		if (hasUVs && hasNormals)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, uvIx);
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, normalIx);
		}
		else if (hasUVs)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, uvIx);
		}
		else if (hasNormals)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, normalIx);
		}
		storage.resize(oldSize + static_cast<size_t>(cursor - token));
	}
	ref.length = static_cast<uint16_t>(storage.size() - ref.offset);
	refs.push_back(ref);
}

void appendObjectHeader(std::string& out, const SGeometryWriterCommon::SPolygonGeometryWriteItem& item)
{
	char name[128] = {};
	if (item.instanceIx != ~0u)
		std::snprintf(name, sizeof(name), "o instance_%u_target_%u_geometry_%u\n", item.instanceIx, item.targetIx, item.geometryIx);
	else
		std::snprintf(name, sizeof(name), "o geometry_%u\n", item.geometryIx);
	out.append(name);
}

struct SGeometryTransformState
{
	hlsl::float32_t3x4 transform;
	hlsl::float32_t3x3 linear;
	bool identity = true;
	bool reverseWinding = false;
	hlsl::math::linalg::cofactors_base<float, 3> normalTransform;
};

inline SGeometryTransformState createTransformState(const hlsl::float32_t3x4& transform)
{
	const auto linear = hlsl::float32_t3x3(transform);
	return {
		.transform = transform,
		.linear = linear,
		.identity = SGeometryWriterCommon::isIdentityTransform(transform),
		.reverseWinding = hlsl::determinant(linear) < 0.f,
		.normalTransform = hlsl::math::linalg::cofactors_base<float, 3>::create(linear)
	};
}

inline hlsl::float32_t3 applyPositionTransform(const SGeometryTransformState& state, const hlsl::float32_t3& value)
{
	if (state.identity)
		return value;

	return hlsl::float32_t3(
		state.transform[0].x * value.x + state.transform[0].y * value.y + state.transform[0].z * value.z + state.transform[0].w,
		state.transform[1].x * value.x + state.transform[1].y * value.y + state.transform[1].z * value.z + state.transform[1].w,
		state.transform[2].x * value.x + state.transform[2].y * value.y + state.transform[2].z * value.z + state.transform[2].w
	);
}

inline hlsl::float32_t3 applyNormalTransform(const SGeometryTransformState& state, const hlsl::float32_t3& value)
{
	return state.identity ? value : state.normalTransform.normalTransform(value);
}

} // namespace obj_writer_detail

bool COBJMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	using namespace obj_writer_detail;
	SFileWriteTelemetry ioTelemetry = {};

	if (!_override)
		getDefaultOverride(_override);

	if (!_file || !_params.rootAsset)
		return false;

	// Scene input is flattened here by baking transforms and writing every collected
	// polygon geometry as its own OBJ object block.
	const auto items = SGeometryWriterCommon::collectPolygonGeometryWriteItems(_params.rootAsset);
	if (items.empty())
		return false;

	SAssetWriteContext ctx = { _params, _file };
	system::IFile* file = _override->getOutputFile(_file, ctx, { _params.rootAsset, 0u });
	if (!file)
		return false;

	std::string output;
	output.append("# Nabla OBJ\n");
	uint64_t totalVertexCount = 0ull;
	uint64_t totalFaceCount = 0ull;
	uint32_t positionBase = 1u;
	uint32_t uvBase = 1u;
	uint32_t normalBase = 1u;
	hlsl::float64_t4 tmp = {};
	for (size_t itemIx = 0u; itemIx < items.size(); ++itemIx)
	{
		const auto& item = items[itemIx];
		const auto* geom = item.geometry;
		if (!geom || !geom->valid())
			return false;

		const auto& positionView = geom->getPositionView();
		if (!positionView)
			return false;

		const auto& normalView = geom->getNormalView();
		const bool hasNormals = static_cast<bool>(normalView);
		const size_t vertexCount = positionView.getElementCount();
		const ICPUPolygonGeometry::SDataView* uvView = SGeometryWriterCommon::getAuxViewAt(geom, SOBJPolygonGeometryAuxLayout::UV0, vertexCount);
		if (uvView && getFormatChannelCount(uvView->composed.format) != 2u)
			uvView = nullptr;
		const bool hasUVs = uvView != nullptr;
		if (vertexCount == 0ull)
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

		size_t faceCount = 0ull;
		if (!SGeometryWriterCommon::getTriangleFaceCount(geom, faceCount))
			return false;

		const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
		const bool flipHandedness = !flags.hasAnyFlag(E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
		const auto transformState = createTransformState(item.transform);
		const hlsl::float32_t3* const tightPositions = SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(positionView);
		const hlsl::float32_t3* const tightNormals = hasNormals ? SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(normalView) : nullptr;
		const hlsl::float32_t2* const tightUV = hasUVs ? SGeometryWriterCommon::getTightView<hlsl::float32_t2, EF_R32G32_SFLOAT>(*uvView) : nullptr;

		if (itemIx != 0u)
			output.push_back('\n');
		appendObjectHeader(output, item);

		for (size_t i = 0u; i < vertexCount; ++i)
		{
			hlsl::float32_t3 vertex = {};
			if (tightPositions)
			{
				vertex = tightPositions[i];
			}
			else
			{
				if (!decodeVec4(positionView, i, tmp))
					return false;
				vertex = hlsl::float32_t3(static_cast<float>(tmp.x), static_cast<float>(tmp.y), static_cast<float>(tmp.z));
			}
			vertex = applyPositionTransform(transformState, vertex);
			if (flipHandedness)
				vertex.x = -vertex.x;

			appendVec3Line(output, "v ", sizeof("v ") - 1ull, vertex);
		}

		if (hasUVs)
		{
			for (size_t i = 0u; i < vertexCount; ++i)
			{
				hlsl::float32_t2 uv = {};
				if (tightUV)
				{
					uv = hlsl::float32_t2(tightUV[i].x, 1.f - tightUV[i].y);
				}
				else
				{
					if (!decodeVec4(*uvView, i, tmp))
						return false;
					uv = hlsl::float32_t2(static_cast<float>(tmp.x), 1.f - static_cast<float>(tmp.y));
				}

				appendVec2Line(output, "vt ", sizeof("vt ") - 1ull, uv);
			}
		}

		if (hasNormals)
		{
			for (size_t i = 0u; i < vertexCount; ++i)
			{
				hlsl::float32_t3 normal = {};
				if (tightNormals)
				{
					normal = tightNormals[i];
				}
				else
				{
					if (!decodeVec4(normalView, i, tmp))
						return false;
					normal = hlsl::float32_t3(static_cast<float>(tmp.x), static_cast<float>(tmp.y), static_cast<float>(tmp.z));
				}
				normal = applyNormalTransform(transformState, normal);
				if (flipHandedness)
					normal.x = -normal.x;

				appendVec3Line(output, "vn ", sizeof("vn ") - 1ull, normal);
			}
		}

		core::vector<SIndexStringRef> faceIndexRefs;
		faceIndexRefs.reserve(vertexCount);
		std::string faceIndexStorage;
		faceIndexStorage.reserve(vertexCount * 24ull);
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			const uint32_t positionIx = positionBase + static_cast<uint32_t>(i);
			const uint32_t uvIx = hasUVs ? (uvBase + static_cast<uint32_t>(i)) : 0u;
			const uint32_t normalIx = hasNormals ? (normalBase + static_cast<uint32_t>(i)) : 0u;
			appendIndexTokenToStorage(faceIndexStorage, faceIndexRefs, positionIx, hasUVs, uvIx, hasNormals, normalIx);
		}

		if (!SGeometryWriterCommon::visitTriangleIndices(geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2)->bool
		{
			const uint32_t f0 = transformState.reverseWinding ? i0 : i2;
			const uint32_t f1 = i1;
			const uint32_t f2 = transformState.reverseWinding ? i2 : i0;
			if (f0 >= faceIndexRefs.size() || f1 >= faceIndexRefs.size() || f2 >= faceIndexRefs.size())
				return false;

			appendFaceLine(output, faceIndexStorage, faceIndexRefs, f0, f1, f2);
			return true;
		}))
			return false;

		positionBase += static_cast<uint32_t>(vertexCount);
		if (hasUVs)
			uvBase += static_cast<uint32_t>(vertexCount);
		if (hasNormals)
			normalBase += static_cast<uint32_t>(vertexCount);
		totalVertexCount += vertexCount;
		totalFaceCount += faceCount;
	}

	const bool fileMappable = core::bitflag<system::IFile::E_CREATE_FLAGS>(file->getFlags()).hasAnyFlag(system::IFile::ECF_MAPPABLE);
    const auto ioPlan = SResolvedFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(output.size()), true, fileMappable);
	if (!ioPlan.isValid())
	{
		_params.logger.log("OBJ writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
		return false;
	}

	const bool writeOk = SInterchangeIO::writeFileWithPolicy(file, ioPlan, reinterpret_cast<const uint8_t*>(output.data()), output.size(), &ioTelemetry);
	const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
	const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
	if (SInterchangeIO::isTinyIOTelemetryLikely(ioTelemetry, static_cast<uint64_t>(output.size()), _params.ioPolicy))
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
		"OBJ writer stats: file=%s bytes=%llu vertices=%llu faces=%llu geometries=%llu io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE,
		file->getFileName().string().c_str(),
		static_cast<unsigned long long>(output.size()),
		static_cast<unsigned long long>(totalVertexCount),
		static_cast<unsigned long long>(totalFaceCount),
		static_cast<unsigned long long>(items.size()),
		static_cast<unsigned long long>(ioTelemetry.callCount),
		static_cast<unsigned long long>(ioMinWrite),
		static_cast<unsigned long long>(ioAvgWrite),
		system::to_string(_params.ioPolicy.strategy).c_str(),
		system::to_string(ioPlan.strategy).c_str(),
		static_cast<unsigned long long>(ioPlan.chunkSizeBytes()),
		ioPlan.reason);

	return writeOk;
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_OBJ_WRITER_

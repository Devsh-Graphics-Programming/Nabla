#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/COBJMeshWriter.h"
#include "nbl/asset/interchange/SGeometryViewDecode.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "impl/SFileAccess.h"
#include "nbl/builtin/hlsl/array_accessors.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

#include "nbl/system/IFile.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
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

namespace
{

struct Parse
{
	static constexpr uint32_t UV0 = 0u;
	static constexpr size_t MaxFloatTextChars = std::numeric_limits<float>::max_digits10 + 8ull;
	static constexpr size_t MaxUInt32Chars = std::numeric_limits<uint32_t>::digits10 + 1ull;
	static constexpr size_t MaxIndexTokenBytes = MaxUInt32Chars * 3ull + 2ull;

	struct IndexStringRef
	{
		uint32_t offset = 0u;
		uint16_t length = 0u;
	};

	struct GeometryTransformState
	{
		hlsl::float32_t3x4 transform;
		hlsl::float32_t3x3 linear;
		bool identity = true;
		bool reverseWinding = false;
		hlsl::math::linalg::cofactors_base<float, 3> normalTransform;
	};

	template<typename Vec>
	static void appendVecLine(std::string& out, const char* prefix, const size_t prefixSize, const Vec& values)
	{
		constexpr size_t N = hlsl::vector_traits<Vec>::Dimension;
		const size_t oldSize = out.size();
		out.resize(oldSize + prefixSize + (N * MaxFloatTextChars) + N);
		char* const lineBegin = out.data() + oldSize;
		char* cursor = lineBegin;
		char* const lineEnd = out.data() + out.size();
		hlsl::array_get<Vec, float> getter;

		std::memcpy(cursor, prefix, prefixSize);
		cursor += prefixSize;

		for (size_t i = 0ull; i < N; ++i)
		{
			cursor = SGeometryWriterCommon::appendFloatToBuffer(cursor, lineEnd, getter(values, static_cast<uint32_t>(i)));
			if (cursor < lineEnd)
				*(cursor++) = (i + 1ull < N) ? ' ' : '\n';
		}

		out.resize(oldSize + static_cast<size_t>(cursor - lineBegin));
	}

	static void appendFaceLine(std::string& out, const std::string& storage, const core::vector<IndexStringRef>& refs, const hlsl::uint32_t3& face)
	{
		const auto& ref0 = refs[face.x];
		const auto& ref1 = refs[face.y];
		const auto& ref2 = refs[face.z];
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

	static void appendIndexToken(std::string& storage, core::vector<IndexStringRef>& refs, const uint32_t positionIx, const bool hasUVs, const uint32_t uvIx, const bool hasNormals, const uint32_t normalIx)
	{
		IndexStringRef ref = {};
		ref.offset = static_cast<uint32_t>(storage.size());
		const size_t oldSize = storage.size();
		storage.resize(oldSize + MaxIndexTokenBytes);
		char* const token = storage.data() + oldSize;
		char* const tokenEnd = token + MaxIndexTokenBytes;
		char* cursor = token;
		cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, positionIx);
		if (hasUVs || hasNormals)
		{
			if (cursor < tokenEnd)
				*(cursor++) = '/';
			if (hasUVs)
				cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, uvIx);
			if (hasNormals)
			{
				if (cursor < tokenEnd)
					*(cursor++) = '/';
				cursor = SGeometryWriterCommon::appendUIntToBuffer(cursor, tokenEnd, normalIx);
			}
		}
		storage.resize(oldSize + static_cast<size_t>(cursor - token));
		ref.length = static_cast<uint16_t>(storage.size() - ref.offset);
		refs.push_back(ref);
	}

	static void appendHeader(std::string& out, const SGeometryWriterCommon::SPolygonGeometryWriteItem& item)
	{
		std::array<char, 128> name = {};
		if (item.instanceIx != ~0u)
			std::snprintf(name.data(), name.size(), "o instance_%u_target_%u_geometry_%u\n", item.instanceIx, item.targetIx, item.geometryIx);
		else
			std::snprintf(name.data(), name.size(), "o geometry_%u\n", item.geometryIx);
		out.append(name.data());
	}

	static GeometryTransformState createTransformState(const hlsl::float32_t3x4& transform)
	{
		const auto linear = hlsl::float32_t3x3(transform);
		return {.transform = transform, .linear = linear, .identity = SGeometryWriterCommon::isIdentityTransform(transform), .reverseWinding = hlsl::determinant(linear) < 0.f, .normalTransform = hlsl::math::linalg::cofactors_base<float, 3>::create(linear)};
	}

	static hlsl::float32_t3 applyPosition(const GeometryTransformState& state, const hlsl::float32_t3& value)
	{
		if (state.identity)
			return value;
		return hlsl::mul(state.transform, hlsl::float32_t4(value.x, value.y, value.z, 1.f));
	}

	static hlsl::float32_t3 applyNormal(const GeometryTransformState& state, const hlsl::float32_t3& value)
	{
		return state.identity ? value : state.normalTransform.normalTransform(value);
	}
};

}

bool COBJMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	SFileWriteTelemetry ioTelemetry = {};

	if (!_override)
		getDefaultOverride(_override);

	if (!_file || !_params.rootAsset)
		return false;

	const auto items = SGeometryWriterCommon::collectPolygonGeometryWriteItems(_params.rootAsset);
	if (items.empty())
		return false;

	SAssetWriteContext ctx = {_params, _file};
	system::IFile* file = _override->getOutputFile(_file, ctx, {_params.rootAsset, 0u});
	if (!file)
		return false;

	std::string output;
	output.append("# Nabla OBJ\n");
	uint64_t totalVertexCount = 0ull;
	uint64_t totalFaceCount = 0ull;
	uint32_t positionBase = 1u;
	uint32_t uvBase = 1u;
	uint32_t normalBase = 1u;
	using SemanticDecode = SGeometryViewDecode::Prepared<SGeometryViewDecode::EMode::Semantic>;
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
		const ICPUPolygonGeometry::SDataView* uvView = SGeometryWriterCommon::getAuxViewAt(geom, Parse::UV0, vertexCount);
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
		const auto transformState = Parse::createTransformState(item.transform);
		const hlsl::float32_t3* const tightPositions = SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(positionView);
		const hlsl::float32_t3* const tightNormals = hasNormals ? SGeometryWriterCommon::getTightView<hlsl::float32_t3, EF_R32G32B32_SFLOAT>(normalView) : nullptr;
		const hlsl::float32_t2* const tightUV = hasUVs ? SGeometryWriterCommon::getTightView<hlsl::float32_t2, EF_R32G32_SFLOAT>(*uvView) : nullptr;
		const SemanticDecode positionDecode = tightPositions ? SemanticDecode{} : SGeometryViewDecode::prepare<SGeometryViewDecode::EMode::Semantic>(positionView);
		const SemanticDecode uvDecode = (!hasUVs || tightUV) ? SemanticDecode{} : SGeometryViewDecode::prepare<SGeometryViewDecode::EMode::Semantic>(*uvView);
		const SemanticDecode normalDecode = (!hasNormals || tightNormals) ? SemanticDecode{} : SGeometryViewDecode::prepare<SGeometryViewDecode::EMode::Semantic>(normalView);

		if (itemIx != 0u)
			output.push_back('\n');
		Parse::appendHeader(output, item);

		for (size_t i = 0u; i < vertexCount; ++i)
		{
			hlsl::float32_t3 vertex = {};
			if (tightPositions)
				vertex = tightPositions[i];
			else if (!positionDecode.decode(i, vertex))
				return false;
			vertex = Parse::applyPosition(transformState, vertex);
			if (flipHandedness)
				vertex.x = -vertex.x;
			Parse::appendVecLine<hlsl::float32_t3>(output, "v ", sizeof("v ") - 1ull, vertex);
		}

		if (hasUVs)
		{
			for (size_t i = 0u; i < vertexCount; ++i)
			{
				hlsl::float32_t2 uv = {};
				if (tightUV)
					uv = hlsl::float32_t2(tightUV[i].x, 1.f - tightUV[i].y);
				else if (!uvDecode.decode(i, uv))
					return false;
				if (!tightUV)
					uv.y = 1.f - uv.y;
				Parse::appendVecLine<hlsl::float32_t2>(output, "vt ", sizeof("vt ") - 1ull, uv);
			}
		}

		if (hasNormals)
		{
			for (size_t i = 0u; i < vertexCount; ++i)
			{
				hlsl::float32_t3 normal = {};
				if (tightNormals)
					normal = tightNormals[i];
				else if (!normalDecode.decode(i, normal))
					return false;
				normal = Parse::applyNormal(transformState, normal);
				if (flipHandedness)
					normal.x = -normal.x;
				Parse::appendVecLine<hlsl::float32_t3>(output, "vn ", sizeof("vn ") - 1ull, normal);
			}
		}

		core::vector<Parse::IndexStringRef> faceIndexRefs;
		faceIndexRefs.reserve(vertexCount);
		std::string faceIndexStorage;
		faceIndexStorage.reserve(vertexCount * 24ull);
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			const uint32_t positionIx = positionBase + static_cast<uint32_t>(i);
			const uint32_t uvIx = hasUVs ? (uvBase + static_cast<uint32_t>(i)) : 0u;
			const uint32_t normalIx = hasNormals ? (normalBase + static_cast<uint32_t>(i)) : 0u;
			Parse::appendIndexToken(faceIndexStorage, faceIndexRefs, positionIx, hasUVs, uvIx, hasNormals, normalIx);
		}
		const hlsl::uint32_t3 faceLimit(static_cast<uint32_t>(faceIndexRefs.size()));

		if (!SGeometryWriterCommon::visitTriangleIndices(geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2) -> bool {
			const hlsl::uint32_t3 face(transformState.reverseWinding ? i0 : i2, i1, transformState.reverseWinding ? i2 : i0);
			if (hlsl::any(glm::greaterThanEqual(face, faceLimit)))
				return false;
			Parse::appendFaceLine(output, faceIndexStorage, faceIndexRefs, face);
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

	const auto ioPlan = impl::SFileAccess::resolvePlan(_params.ioPolicy, static_cast<uint64_t>(output.size()), true, file);
	if (impl::SFileAccess::logInvalidPlan(_params.logger, "OBJ writer", file->getFileName().string().c_str(), ioPlan))
		return false;

	const bool writeOk = SInterchangeIO::writeFileWithPolicy(file, ioPlan, output.data(), output.size(), &ioTelemetry);
	const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
	const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
	impl::SFileAccess::logTinyIO(_params.logger, "OBJ writer", file->getFileName().string().c_str(), ioTelemetry, static_cast<uint64_t>(output.size()), _params.ioPolicy, "writes");
	_params.logger.log("OBJ writer stats: file=%s bytes=%llu vertices=%llu faces=%llu geometries=%llu io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
		system::ILogger::ELL_PERFORMANCE, file->getFileName().string().c_str(), static_cast<unsigned long long>(output.size()),
		static_cast<unsigned long long>(totalVertexCount), static_cast<unsigned long long>(totalFaceCount), static_cast<unsigned long long>(items.size()),
		static_cast<unsigned long long>(ioTelemetry.callCount), static_cast<unsigned long long>(ioMinWrite), static_cast<unsigned long long>(ioAvgWrite),
		system::to_string(_params.ioPolicy.strategy).c_str(), system::to_string(ioPlan.strategy).c_str(), static_cast<unsigned long long>(ioPlan.chunkSizeBytes()), ioPlan.reason);

	return writeOk;
}

}

#endif // _NBL_COMPILE_WITH_OBJ_WRITER_

#ifdef _NBL_COMPILE_WITH_PLY_WRITER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "CPLYMeshWriter.h"
#include "nbl/asset/interchange/SGeometryViewDecode.h"
#include "nbl/asset/interchange/SPLYPolygonGeometryAuxLayout.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "impl/SFileAccess.h"
#include "nbl/system/IFile.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <system_error>
namespace nbl::asset
{
CPLYMeshWriter::CPLYMeshWriter()
{
	#ifdef _NBL_DEBUG
	setDebugName("CPLYMeshWriter");
	#endif
}
const char** CPLYMeshWriter::getAssociatedFileExtensions() const
{
	static const char* ext[] = { "ply", nullptr };
	return ext;
}
writer_flags_t CPLYMeshWriter::getSupportedFlags()
{
	return asset::EWF_BINARY;
}
writer_flags_t CPLYMeshWriter::getForcedFlags()
{
	return EWF_NONE;
}
namespace
{
struct Parse
{
	enum class ScalarType : uint8_t { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };
	using SemanticDecode = SGeometryViewDecode::Prepared<SGeometryViewDecode::EMode::Semantic>;
	using StoredDecode = SGeometryViewDecode::Prepared<SGeometryViewDecode::EMode::Stored>;
	struct ScalarMeta { const char* name = "float32"; uint32_t byteSize = sizeof(float); bool integer = false; bool signedType = true; };
	struct ExtraAuxView { const ICPUPolygonGeometry::SDataView* view = nullptr; uint32_t components = 0u; uint32_t auxIndex = 0u; ScalarType scalarType = ScalarType::Float32; };
	struct WriteInput { const ICPUPolygonGeometry* geom = nullptr; ScalarType positionScalarType = ScalarType::Float32; const ICPUPolygonGeometry::SDataView* uvView = nullptr; ScalarType uvScalarType = ScalarType::Float32; const core::vector<ExtraAuxView>* extraAuxViews = nullptr; bool writeNormals = false; ScalarType normalScalarType = ScalarType::Float32; size_t vertexCount = 0ull; const uint32_t* indices = nullptr; size_t faceCount = 0ull; bool write16BitIndices = false; bool flipVectors = false; };
	static constexpr size_t ApproxTextBytesPerVertex = sizeof("0.000000 0.000000 0.000000 0.000000 0.000000 0.000000\n") - 1ull;
	static constexpr size_t ApproxTextBytesPerFace = sizeof("3 4294967295 4294967295 4294967295\n") - 1ull;
	static constexpr size_t MaxFloatTextChars = std::numeric_limits<double>::max_digits10 + 16ull;
	template<typename T>
	static void appendIntegral(std::string& out, const T value) { std::array<char, 32> buf = {}; const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value); if (res.ec == std::errc()) out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data())); }
	static void appendFloat(std::string& out, double value)
	{
		const size_t oldSize = out.size();
		out.resize(oldSize + MaxFloatTextChars);
		char* const begin = out.data() + oldSize;
		char* const end = begin + MaxFloatTextChars;
		char* const cursor = SGeometryWriterCommon::appendFloatToBuffer(begin, end, value);
		out.resize(oldSize + static_cast<size_t>(cursor - begin));
	}
	static ScalarMeta getScalarMeta(const ScalarType type)
	{
		switch (type)
		{
			case ScalarType::Int8: return {"int8", sizeof(int8_t), true, true};
			case ScalarType::UInt8: return {"uint8", sizeof(uint8_t), true, false};
			case ScalarType::Int16: return {"int16", sizeof(int16_t), true, true};
			case ScalarType::UInt16: return {"uint16", sizeof(uint16_t), true, false};
			case ScalarType::Int32: return {"int32", sizeof(int32_t), true, true};
			case ScalarType::UInt32: return {"uint32", sizeof(uint32_t), true, false};
			case ScalarType::Float64: return {"float64", sizeof(double), false, true};
			default: return {"float32", sizeof(float), false, true};
		}
	}
	struct PreparedView
	{
		const ICPUPolygonGeometry::SDataView* view = nullptr;
		uint32_t componentCount = 0u;
		ScalarType scalarType = ScalarType::Float32;
		bool flipVectors = false;
		SemanticDecode semantic = {};
		StoredDecode stored = {};
		static inline PreparedView create(const ICPUPolygonGeometry::SDataView& view, const uint32_t componentCount, const ScalarType scalarType, const bool flipVectors)
		{
			PreparedView retval = {.view = &view, .componentCount = componentCount, .scalarType = scalarType, .flipVectors = flipVectors};
			const auto meta = getScalarMeta(scalarType);
			if (meta.integer)
				retval.stored = SGeometryViewDecode::prepare<SGeometryViewDecode::EMode::Stored>(view);
			else
				retval.semantic = SGeometryViewDecode::prepare<SGeometryViewDecode::EMode::Semantic>(view);
			return retval;
		}
	};
	static bool isSupportedScalarFormat(const E_FORMAT format)
	{
		if (format == EF_UNKNOWN)
			return false;
		const uint32_t channels = getFormatChannelCount(format);
		if (channels == 0u)
			return false;
		if (!(isIntegerFormat(format) || isFloatingPointFormat(format) || isNormalizedFormat(format) || isScaledFormat(format)))
			return false;
		const auto bytesPerPixel = getBytesPerPixel(format);
		if (bytesPerPixel.getDenominator() != 1u)
			return false;
		const uint32_t pixelBytes = bytesPerPixel.getNumerator();
		if (pixelBytes == 0u || (pixelBytes % channels) != 0u)
			return false;
		const uint32_t bytesPerChannel = pixelBytes / channels;
		return bytesPerChannel == 1u || bytesPerChannel == 2u || bytesPerChannel == 4u || bytesPerChannel == 8u;
	}
	static ScalarType selectScalarType(const E_FORMAT format)
	{
		if (!isSupportedScalarFormat(format))
			return ScalarType::Float32;
		if (isNormalizedFormat(format) || isScaledFormat(format))
			return ScalarType::Float32;
		const uint32_t channels = getFormatChannelCount(format);
		if (channels == 0u)
		{
			assert(format == EF_UNKNOWN);
			return ScalarType::Float32;
		}
		const auto bytesPerPixel = getBytesPerPixel(format);
		if (bytesPerPixel.getDenominator() != 1u)
			return ScalarType::Float32;
		const uint32_t pixelBytes = bytesPerPixel.getNumerator();
		if (pixelBytes == 0u || (pixelBytes % channels) != 0u)
			return ScalarType::Float32;
		const uint32_t bytesPerChannel = pixelBytes / channels;
		if (isIntegerFormat(format))
		{
			const bool signedType = isSignedFormat(format);
			switch (bytesPerChannel)
			{
				case 1u: return signedType ? ScalarType::Int8 : ScalarType::UInt8;
				case 2u: return signedType ? ScalarType::Int16 : ScalarType::UInt16;
				case 4u: return signedType ? ScalarType::Int32 : ScalarType::UInt32;
				default: return ScalarType::Float64;
			}
		}
		if (isFloatingPointFormat(format))
			return bytesPerChannel >= 8u ? ScalarType::Float64 : ScalarType::Float32;
		return ScalarType::Float32;
	}
	static bool isDirectScalarFormat(const E_FORMAT format, const ScalarType scalarType, const uint32_t componentCount, uint32_t& outByteSize)
	{
		outByteSize = 0u;
		if (format == EF_UNKNOWN || componentCount == 0u)
			return false;
		if (isNormalizedFormat(format) || isScaledFormat(format))
			return false;
		const uint32_t channels = getFormatChannelCount(format);
		if (channels < componentCount)
			return false;
		const auto bytesPerPixel = getBytesPerPixel(format);
		if (bytesPerPixel.getDenominator() != 1u)
			return false;
		const uint32_t pixelBytes = bytesPerPixel.getNumerator();
		if (pixelBytes == 0u || (pixelBytes % channels) != 0u)
			return false;
		const uint32_t byteSize = pixelBytes / channels;
		const auto meta = getScalarMeta(scalarType);
		if (byteSize != meta.byteSize)
			return false;
		switch (scalarType)
		{
			case ScalarType::Float32:
			case ScalarType::Float64:
				if (!isFloatingPointFormat(format))
					return false;
				break;
			case ScalarType::Int8:
			case ScalarType::Int16:
			case ScalarType::Int32:
				if (!isIntegerFormat(format) || !isSignedFormat(format))
					return false;
				break;
			case ScalarType::UInt8:
			case ScalarType::UInt16:
			case ScalarType::UInt32:
				if (!isIntegerFormat(format) || isSignedFormat(format))
					return false;
				break;
		}
		outByteSize = byteSize;
		return true;
	}
	static bool writeDirectBinaryView(const ICPUPolygonGeometry::SDataView& view, const size_t ix, const uint32_t componentCount, const ScalarType scalarType, const bool flipVectors, uint8_t*& dst)
	{
		if (flipVectors || !dst || !view.composed.isFormatted())
			return false;
		uint32_t byteSize = 0u;
		if (!isDirectScalarFormat(view.composed.format, scalarType, componentCount, byteSize))
			return false;
		const uint32_t pixelBytes = getBytesPerPixel(view.composed.format).getNumerator();
		if (view.composed.getStride() != pixelBytes)
			return false;
		const void* src = view.getPointer(ix);
		if (!src)
			return false;
		const size_t copyBytes = static_cast<size_t>(componentCount) * byteSize;
		std::memcpy(dst, src, copyBytes);
		dst += copyBytes;
		return true;
	}
	static bool writeTypedViewBinary(const PreparedView& prepared, const size_t ix, uint8_t*& dst)
	{
		if (!prepared.view || !dst)
			return false;
		const auto& view = *prepared.view;
		const auto componentCount = prepared.componentCount;
		const auto scalarType = prepared.scalarType;
		const auto flipVectors = prepared.flipVectors;
		if (!dst)
			return false;
		if (writeDirectBinaryView(view, ix, componentCount, scalarType, flipVectors, dst))
			return true;
		switch (scalarType)
		{
			case ScalarType::Float64:
			case ScalarType::Float32:
			{
				std::array<double, 4> tmp = {};
				if (!prepared.semantic.decode(ix, tmp))
					return false;
				for (uint32_t c = 0u; c < componentCount; ++c)
				{
					double value = tmp[c];
					if (flipVectors && c == 0u)
						value = -value;
					if (scalarType == ScalarType::Float64)
					{
						std::memcpy(dst, &value, sizeof(value));
						dst += sizeof(value);
					}
					else
					{
						const float typed = static_cast<float>(value);
						std::memcpy(dst, &typed, sizeof(typed));
						dst += sizeof(typed);
					}
				}
				return true;
			}
			case ScalarType::Int8:
			case ScalarType::Int16:
			case ScalarType::Int32:
			{
				std::array<int64_t, 4> tmp = {};
				if (!prepared.stored.decode(ix, tmp))
					return false;
				for (uint32_t c = 0u; c < componentCount; ++c)
				{
					int64_t value = tmp[c];
					if (flipVectors && c == 0u)
						value = -value;
					switch (scalarType)
					{
						case ScalarType::Int8:
						{
							const int8_t typed = static_cast<int8_t>(value);
							std::memcpy(dst, &typed, sizeof(typed));
							dst += sizeof(typed);
						} break;
						case ScalarType::Int16:
						{
							const int16_t typed = static_cast<int16_t>(value);
							std::memcpy(dst, &typed, sizeof(typed));
							dst += sizeof(typed);
						} break;
						default:
						{
							const int32_t typed = static_cast<int32_t>(value);
							std::memcpy(dst, &typed, sizeof(typed));
							dst += sizeof(typed);
						} break;
					}
				}
				return true;
			}
			case ScalarType::UInt8:
			case ScalarType::UInt16:
			case ScalarType::UInt32:
			{
				std::array<uint64_t, 4> tmp = {};
				if (!prepared.stored.decode(ix, tmp))
					return false;
				for (uint32_t c = 0u; c < componentCount; ++c)
				{
					switch (scalarType)
					{
						case ScalarType::UInt8:
						{
							const uint8_t typed = static_cast<uint8_t>(tmp[c]);
							std::memcpy(dst, &typed, sizeof(typed));
							dst += sizeof(typed);
						} break;
						case ScalarType::UInt16:
						{
							const uint16_t typed = static_cast<uint16_t>(tmp[c]);
							std::memcpy(dst, &typed, sizeof(typed));
							dst += sizeof(typed);
						} break;
						default:
						{
							const uint32_t typed = static_cast<uint32_t>(tmp[c]);
							std::memcpy(dst, &typed, sizeof(typed));
							dst += sizeof(typed);
						} break;
					}
				}
				return true;
			}
		}
		return false;
	}
	static bool writeTypedViewText(std::string& output, const PreparedView& prepared, const size_t ix)
	{
		if (!prepared.view)
			return false;
		const auto componentCount = prepared.componentCount;
		const auto scalarType = prepared.scalarType;
		const auto flipVectors = prepared.flipVectors;
		switch (scalarType)
		{
			case ScalarType::Float64:
			case ScalarType::Float32:
			{
				std::array<double, 4> tmp = {};
				if (!prepared.semantic.decode(ix, tmp))
					return false;
				for (uint32_t c = 0u; c < componentCount; ++c)
				{
					double value = tmp[c];
					if (flipVectors && c == 0u)
						value = -value;
					appendFloat(output, value);
					output.push_back(' ');
				}
				return true;
			}
			case ScalarType::Int8:
			case ScalarType::Int16:
			case ScalarType::Int32:
			{
				std::array<int64_t, 4> tmp = {};
				if (!prepared.stored.decode(ix, tmp))
					return false;
				for (uint32_t c = 0u; c < componentCount; ++c)
				{
					int64_t value = tmp[c];
					if (flipVectors && c == 0u)
						value = -value;
					appendIntegral(output, value);
					output.push_back(' ');
				}
				return true;
			}
			case ScalarType::UInt8:
			case ScalarType::UInt16:
			case ScalarType::UInt32:
			{
				std::array<uint64_t, 4> tmp = {};
				if (!prepared.stored.decode(ix, tmp))
					return false;
				for (uint32_t c = 0u; c < componentCount; ++c)
				{
					appendIntegral(output, tmp[c]);
					output.push_back(' ');
				}
				return true;
			}
		}
		return false;
	}
	static bool writeBinaryFast(const WriteInput& input, uint8_t*& dst)
	{
		if (!input.geom || !input.indices || !input.extraAuxViews || !dst || input.flipVectors || input.writeNormals || input.uvView || !input.extraAuxViews->empty() || input.positionScalarType != ScalarType::Float32)
			return false;
		const auto& positionView = input.geom->getPositionView();
		if (!positionView.composed.isFormatted() || positionView.composed.format != EF_R32G32B32_SFLOAT || positionView.composed.getStride() != sizeof(hlsl::float32_t3))
			return false;
		const void* src = positionView.getPointer();
		if (!src)
			return false;
		const size_t vertexBytes = input.vertexCount * sizeof(hlsl::float32_t3);
		std::memcpy(dst, src, vertexBytes);
		dst += vertexBytes;
		for (size_t i = 0u; i < input.faceCount; ++i)
		{
			*dst++ = 3u;
			const uint32_t* tri = input.indices + i * 3u;
			if (input.write16BitIndices)
			{
				const uint16_t tri16[3] = {static_cast<uint16_t>(tri[0]), static_cast<uint16_t>(tri[1]), static_cast<uint16_t>(tri[2])};
				std::memcpy(dst, tri16, sizeof(tri16));
				dst += sizeof(tri16);
			}
			else
			{
				std::memcpy(dst, tri, sizeof(uint32_t) * 3u);
				dst += sizeof(uint32_t) * 3u;
			}
		}
		return true;
	}
	static bool writeBinary(const WriteInput& input, uint8_t* dst)
	{
		if (!input.geom || !input.extraAuxViews || !dst)
			return false;
		if (writeBinaryFast(input, dst))
			return true;
		const auto& positionView = input.geom->getPositionView();
		const auto& normalView = input.geom->getNormalView();
		const auto& extraAuxViews = *input.extraAuxViews;
		const PreparedView preparedPosition = PreparedView::create(positionView, 3u, input.positionScalarType, input.flipVectors);
		const PreparedView preparedNormal = input.writeNormals ? PreparedView::create(normalView, 3u, input.normalScalarType, input.flipVectors) : PreparedView{};
		const PreparedView preparedUV = input.uvView ? PreparedView::create(*input.uvView, 2u, input.uvScalarType, false) : PreparedView{};
		core::vector<PreparedView> preparedExtraAuxViews;
		preparedExtraAuxViews.reserve(extraAuxViews.size());
		for (const auto& extra : extraAuxViews)
		{
			if (!extra.view)
				return false;
			preparedExtraAuxViews.push_back(PreparedView::create(*extra.view, extra.components, extra.scalarType, false));
		}
		for (size_t i = 0u; i < input.vertexCount; ++i)
		{
			if (!writeTypedViewBinary(preparedPosition, i, dst))
				return false;
			if (input.writeNormals && !writeTypedViewBinary(preparedNormal, i, dst))
				return false;
			if (input.uvView && !writeTypedViewBinary(preparedUV, i, dst))
				return false;
			for (const auto& extra : preparedExtraAuxViews)
				if (!writeTypedViewBinary(extra, i, dst))
					return false;
		}
		if (!input.indices)
			return false;
		for (size_t i = 0u; i < input.faceCount; ++i)
		{
			const uint8_t listSize = 3u;
			*dst++ = listSize;
			const uint32_t* tri = input.indices + i * 3u;
			if (input.write16BitIndices)
			{
				const uint16_t tri16[3] = {static_cast<uint16_t>(tri[0]), static_cast<uint16_t>(tri[1]), static_cast<uint16_t>(tri[2])};
				std::memcpy(dst, tri16, sizeof(tri16));
				dst += sizeof(tri16);
			}
			else
			{
				std::memcpy(dst, tri, sizeof(uint32_t) * 3u);
				dst += sizeof(uint32_t) * 3u;
			}
		}
		return true;
	}
	static bool writeText(const WriteInput& input, std::string& output)
	{
		if (!input.geom || !input.extraAuxViews)
			return false;
		const auto& extraAuxViews = *input.extraAuxViews;
		const PreparedView preparedPosition = PreparedView::create(input.geom->getPositionView(), 3u, input.positionScalarType, input.flipVectors);
		const PreparedView preparedNormal = input.writeNormals ? PreparedView::create(input.geom->getNormalView(), 3u, input.normalScalarType, input.flipVectors) : PreparedView{};
		const PreparedView preparedUV = input.uvView ? PreparedView::create(*input.uvView, 2u, input.uvScalarType, false) : PreparedView{};
		core::vector<PreparedView> preparedExtraAuxViews;
		preparedExtraAuxViews.reserve(extraAuxViews.size());
		for (const auto& extra : extraAuxViews)
		{
			if (!extra.view)
				return false;
			preparedExtraAuxViews.push_back(PreparedView::create(*extra.view, extra.components, extra.scalarType, false));
		}
		for (size_t i = 0u; i < input.vertexCount; ++i)
		{
			if (!writeTypedViewText(output, preparedPosition, i))
				return false;
			if (input.writeNormals && !writeTypedViewText(output, preparedNormal, i))
				return false;
			if (input.uvView && !writeTypedViewText(output, preparedUV, i))
				return false;
			for (const auto& extra : preparedExtraAuxViews)
				if (!writeTypedViewText(output, extra, i))
					return false;
			output.push_back('\n');
		}
		if (!input.indices)
			return false;
		for (size_t i = 0u; i < input.faceCount; ++i)
		{
			const uint32_t* tri = input.indices + i * 3u;
			output.append("3 ");
			appendIntegral(output, tri[0]);
			output.push_back(' ');
			appendIntegral(output, tri[1]);
			output.push_back(' ');
			appendIntegral(output, tri[2]);
			output.push_back('\n');
		}
		return true;
	}
};
}
bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	using ScalarType = Parse::ScalarType;
	using clock_t = std::chrono::high_resolution_clock;
	SFileWriteTelemetry ioTelemetry = {};
	if (!_override)
		getDefaultOverride(_override);
	if (!_file || !_params.rootAsset)
		return _params.logger.log("PLY writer: missing output file or root asset.", system::ILogger::ELL_ERROR), false;
	const auto items = SGeometryWriterCommon::collectPolygonGeometryWriteItems(_params.rootAsset);
	if (items.size() != 1u)
		return _params.logger.log("PLY writer: expected exactly one polygon geometry to write.", system::ILogger::ELL_ERROR), false;
	const auto& item = items.front();
	const auto* geom = item.geometry;
	if (!geom || !geom->valid())
		return _params.logger.log("PLY writer: root asset is not a valid polygon geometry.", system::ILogger::ELL_ERROR), false;
	if (!SGeometryWriterCommon::isIdentityTransform(item.transform))
		return _params.logger.log("PLY writer: transformed scene or collection export is not supported.", system::ILogger::ELL_ERROR), false;
	SAssetWriteContext ctx = {_params, _file};
	system::IFile* file = _override->getOutputFile(_file, ctx, {geom, 0u});
	if (!file)
		return _params.logger.log("PLY writer: output override returned null file.", system::ILogger::ELL_ERROR), false;
	const auto& positionView = geom->getPositionView();
	const auto& normalView = geom->getNormalView();
	const size_t vertexCount = positionView.getElementCount();
	if (vertexCount == 0ull)
		return _params.logger.log("PLY writer: empty position view.", system::ILogger::ELL_ERROR), false;
	const bool writeNormals = static_cast<bool>(normalView);
	if (writeNormals && normalView.getElementCount() != vertexCount)
		return _params.logger.log("PLY writer: normal vertex count mismatch.", system::ILogger::ELL_ERROR), false;
	const ICPUPolygonGeometry::SDataView* uvView = SGeometryWriterCommon::getAuxViewAt(geom, SPLYPolygonGeometryAuxLayout::UV0, vertexCount);
	if (uvView && getFormatChannelCount(uvView->composed.format) != 2u)
		uvView = nullptr;
	core::vector<Parse::ExtraAuxView> extraAuxViews;
	const auto& auxViews = geom->getAuxAttributeViews();
	extraAuxViews.reserve(auxViews.size());
	for (uint32_t auxIx = 0u; auxIx < static_cast<uint32_t>(auxViews.size()); ++auxIx)
	{
		const auto& view = auxViews[auxIx];
		if (!view || (uvView && auxIx == SPLYPolygonGeometryAuxLayout::UV0))
			continue;
		if (view.getElementCount() != vertexCount)
			continue;
		const uint32_t channels = getFormatChannelCount(view.composed.format);
		if (channels == 0u)
			continue;
		const uint32_t components = std::min(4u, channels);
		extraAuxViews.push_back({&view, components, auxIx, Parse::selectScalarType(view.composed.format)});
	}
	_params.logger.log("PLY writer input: file=%s pos_fmt=%u pos_stride=%u pos_count=%llu normal_fmt=%u normal_stride=%u normal_count=%llu uv_fmt=%u uv_stride=%u uv_count=%llu aux=%u",
		system::ILogger::ELL_INFO, file->getFileName().string().c_str(), static_cast<uint32_t>(positionView.composed.format), positionView.composed.getStride(),
		static_cast<unsigned long long>(positionView.getElementCount()), static_cast<uint32_t>(normalView.composed.format), normalView.composed.getStride(),
		static_cast<unsigned long long>(normalView.getElementCount()), uvView ? static_cast<uint32_t>(uvView->composed.format) : static_cast<uint32_t>(EF_UNKNOWN),
		uvView ? uvView->composed.getStride() : 0u, uvView ? static_cast<unsigned long long>(uvView->getElementCount()) : 0ull, static_cast<uint32_t>(extraAuxViews.size()));
	const auto* indexing = geom->getIndexingCallback();
	if (!indexing)
		return _params.logger.log("PLY writer: missing indexing callback.", system::ILogger::ELL_ERROR), false;
	if (indexing->knownTopology() != E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST)
		return _params.logger.log("PLY writer: only triangle-list topology is supported.", system::ILogger::ELL_ERROR), false;
	const auto& indexView = geom->getIndexView();
	core::vector<uint32_t> indexData;
	const uint32_t* indices = nullptr;
	size_t faceCount = 0ull;
	if (indexView)
	{
		const size_t indexCount = indexView.getElementCount();
		if ((indexCount % 3u) != 0u)
			return _params.logger.log("PLY writer: failed to validate triangle indexing.", system::ILogger::ELL_ERROR), false;
		const void* src = indexView.getPointer();
		if (!src)
			return _params.logger.log("PLY writer: missing index buffer pointer.", system::ILogger::ELL_ERROR), false;
		if (indexView.composed.format == EF_R32_UINT && indexView.composed.getStride() == sizeof(uint32_t))
			indices = reinterpret_cast<const uint32_t*>(src);
		else if (indexView.composed.format == EF_R16_UINT && indexView.composed.getStride() == sizeof(uint16_t))
		{
			const auto* src16 = reinterpret_cast<const uint16_t*>(src);
			indexData.resize(indexCount);
			for (size_t i = 0u; i < indexCount; ++i)
				indexData[i] = src16[i];
			indices = indexData.data();
		}
		else
		{
			indexData.resize(indexCount);
			for (size_t i = 0u; i < indexCount; ++i)
			{
				hlsl::uint32_t4 decoded = {};
				if (!indexView.decodeElement(i, decoded))
					return _params.logger.log("PLY writer: failed to decode index view.", system::ILogger::ELL_ERROR), false;
				indexData[i] = decoded.x;
			}
			indices = indexData.data();
		}
		faceCount = indexCount / 3u;
	}
	else
	{
		if ((vertexCount % 3u) != 0u)
			return _params.logger.log("PLY writer: failed to derive triangle indexing from positions.", system::ILogger::ELL_ERROR), false;
		indexData.resize(vertexCount);
		for (size_t i = 0u; i < vertexCount; ++i)
			indexData[i] = static_cast<uint32_t>(i);
		indices = indexData.data();
		faceCount = vertexCount / 3u;
	}
	const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
	const bool binary = flags.hasAnyFlag(E_WRITER_FLAGS::EWF_BINARY);
	const bool flipVectors = !flags.hasAnyFlag(E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
	const bool write16BitIndices = vertexCount <= static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1ull;
	ScalarType positionScalarType = Parse::selectScalarType(positionView.composed.format);
	if (flipVectors && Parse::getScalarMeta(positionScalarType).integer && !Parse::getScalarMeta(positionScalarType).signedType)
		positionScalarType = ScalarType::Float32;
	ScalarType normalScalarType = Parse::selectScalarType(normalView.composed.format);
	if (flipVectors && Parse::getScalarMeta(normalScalarType).integer && !Parse::getScalarMeta(normalScalarType).signedType)
		normalScalarType = ScalarType::Float32;
	const ScalarType uvScalarType = uvView ? Parse::selectScalarType(uvView->composed.format) : ScalarType::Float32;
	const auto positionMeta = Parse::getScalarMeta(positionScalarType);
	const auto normalMeta = Parse::getScalarMeta(normalScalarType);
	const auto uvMeta = Parse::getScalarMeta(uvScalarType);
	size_t extraAuxBytesPerVertex = 0ull;
	for (const auto& extra : extraAuxViews)
		extraAuxBytesPerVertex += static_cast<size_t>(extra.components) * Parse::getScalarMeta(extra.scalarType).byteSize;
	std::ostringstream headerBuilder;
	headerBuilder << "ply\n";
	headerBuilder << (binary ? "format binary_little_endian 1.0" : "format ascii 1.0");
	headerBuilder << "\ncomment Nabla " << NABLA_SDK_VERSION;
	headerBuilder << "\nelement vertex " << vertexCount << "\n";
	headerBuilder << "property " << positionMeta.name << " x\n";
	headerBuilder << "property " << positionMeta.name << " y\n";
	headerBuilder << "property " << positionMeta.name << " z\n";
	if (writeNormals)
	{
		headerBuilder << "property " << normalMeta.name << " nx\n";
		headerBuilder << "property " << normalMeta.name << " ny\n";
		headerBuilder << "property " << normalMeta.name << " nz\n";
	}
	if (uvView)
	{
		headerBuilder << "property " << uvMeta.name << " u\n";
		headerBuilder << "property " << uvMeta.name << " v\n";
	}
	for (const auto& extra : extraAuxViews)
	{
		const auto extraMeta = Parse::getScalarMeta(extra.scalarType);
		for (uint32_t component = 0u; component < extra.components; ++component)
		{
			headerBuilder << "property " << extraMeta.name << " aux" << extra.auxIndex;
			if (extra.components > 1u)
				headerBuilder << "_" << component;
			headerBuilder << "\n";
		}
	}
	headerBuilder << "element face " << faceCount;
	headerBuilder << (write16BitIndices ? "\nproperty list uchar uint16 vertex_indices\n" : "\nproperty list uchar uint32 vertex_indices\n");
	headerBuilder << "end_header\n";
	const std::string header = headerBuilder.str();
	const Parse::WriteInput input = {.geom = geom, .positionScalarType = positionScalarType, .uvView = uvView, .uvScalarType = uvScalarType, .extraAuxViews = &extraAuxViews, .writeNormals = writeNormals, .normalScalarType = normalScalarType, .vertexCount = vertexCount, .indices = indices, .faceCount = faceCount, .write16BitIndices = write16BitIndices, .flipVectors = flipVectors};
	bool writeOk = false;
	size_t outputBytes = 0ull;
	double writeIoMs = 0.0;
	auto writePayload = [&](const void* bodyData, const size_t bodySize) -> bool {
		const size_t outputSize = header.size() + bodySize;
		const auto ioPlan = impl::SFileAccess::resolvePlan(_params.ioPolicy, static_cast<uint64_t>(outputSize), true, file);
		if (impl::SFileAccess::logInvalidPlan(_params.logger, "PLY writer", file->getFileName().string().c_str(), ioPlan))
			return false;
		outputBytes = outputSize;
		const SInterchangeIO::SBufferRange writeBuffers[] = {{.data = header.data(), .byteCount = header.size()}, {.data = bodyData, .byteCount = bodySize}};
		const auto ioStart = clock_t::now();
		writeOk = SInterchangeIO::writeBuffersWithPolicy(file, ioPlan, writeBuffers, &ioTelemetry);
		writeIoMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
		const uint64_t ioMinWrite = ioTelemetry.getMinOrZero();
		const uint64_t ioAvgWrite = ioTelemetry.getAvgOrZero();
		impl::SFileAccess::logTinyIO(_params.logger, "PLY writer", file->getFileName().string().c_str(), ioTelemetry, static_cast<uint64_t>(outputBytes), _params.ioPolicy, "writes");
		_params.logger.log("PLY writer stats: file=%s bytes=%llu vertices=%llu faces=%llu binary=%d io_writes=%llu io_min_write=%llu io_avg_write=%llu io_req=%s io_eff=%s io_chunk=%llu io_reason=%s",
			system::ILogger::ELL_PERFORMANCE, file->getFileName().string().c_str(), static_cast<unsigned long long>(outputBytes),
			static_cast<unsigned long long>(vertexCount), static_cast<unsigned long long>(faceCount), binary ? 1 : 0,
			static_cast<unsigned long long>(ioTelemetry.callCount), static_cast<unsigned long long>(ioMinWrite), static_cast<unsigned long long>(ioAvgWrite),
			system::to_string(_params.ioPolicy.strategy).c_str(), system::to_string(ioPlan.strategy).c_str(), static_cast<unsigned long long>(ioPlan.chunkSizeBytes()), ioPlan.reason);
		return writeOk;
	};
	if (binary)
	{
		const size_t vertexStride = static_cast<size_t>(positionMeta.byteSize) * 3ull + (writeNormals ? static_cast<size_t>(normalMeta.byteSize) * 3ull : 0ull) + (uvView ? static_cast<size_t>(uvMeta.byteSize) * 2ull : 0ull) + extraAuxBytesPerVertex;
		const size_t faceStride = sizeof(uint8_t) + (write16BitIndices ? sizeof(uint16_t) : sizeof(uint32_t)) * 3u;
		const size_t bodySize = vertexCount * vertexStride + faceCount * faceStride;
		core::vector<uint8_t> body;
		const auto fillStart = clock_t::now();
		body.resize(bodySize);
		if (!Parse::writeBinary(input, body.data()))
			return _params.logger.log("PLY writer: binary payload generation failed.", system::ILogger::ELL_ERROR), false;
		const auto fillMs = std::chrono::duration<double, std::milli>(clock_t::now() - fillStart).count();
		const bool ok = writePayload(body.data(), body.size());
		_params.logger.log("PLY writer stages: file=%s header=%llu body=%llu fill=%.3f ms io=%.3f ms", system::ILogger::ELL_PERFORMANCE, file->getFileName().string().c_str(), static_cast<unsigned long long>(header.size()), static_cast<unsigned long long>(body.size()), fillMs, writeIoMs);
		return ok;
	}
	std::string body;
	body.reserve(vertexCount * Parse::ApproxTextBytesPerVertex + faceCount * Parse::ApproxTextBytesPerFace);
	const auto fillStart = clock_t::now();
	if (!Parse::writeText(input, body))
		return _params.logger.log("PLY writer: text payload generation failed.", system::ILogger::ELL_ERROR), false;
	const auto fillMs = std::chrono::duration<double, std::milli>(clock_t::now() - fillStart).count();
	const bool ok = writePayload(body.data(), body.size());
	_params.logger.log("PLY writer stages: file=%s header=%llu body=%llu fill=%.3f ms io=%.3f ms", system::ILogger::ELL_PERFORMANCE, file->getFileName().string().c_str(), static_cast<unsigned long long>(header.size()), static_cast<unsigned long long>(body.size()), fillMs, writeIoMs);
	return ok;
}
}
#endif // _NBL_COMPILE_WITH_PLY_WRITER_

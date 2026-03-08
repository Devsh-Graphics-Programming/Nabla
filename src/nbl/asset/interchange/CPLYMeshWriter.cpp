#ifdef _NBL_COMPILE_WITH_PLY_WRITER_
// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#include "CPLYMeshWriter.h"
#include "nbl/asset/interchange/SGeometryViewDecode.h"
#include "nbl/asset/interchange/SGeometryWriterCommon.h"
#include "nbl/asset/interchange/SInterchangeIO.h"
#include "impl/SBinaryData.h"
#include "impl/SFileAccess.h"
#include "nbl/system/IFile.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
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
	static constexpr uint32_t UV0 = 0u;
	using Binary = impl::BinaryData;
	using SemanticDecode = SGeometryViewDecode::Prepared<SGeometryViewDecode::EMode::Semantic>;
	using StoredDecode = SGeometryViewDecode::Prepared<SGeometryViewDecode::EMode::Stored>;
	enum class ScalarType : uint8_t { Int8, UInt8, Int16, UInt16, Int32, UInt32, Float32, Float64 };
	struct ScalarMeta { const char* name = "float32"; uint32_t byteSize = sizeof(float); bool integer = false; bool signedType = true; };
	struct ExtraAuxView { const ICPUPolygonGeometry::SDataView* view = nullptr; uint32_t components = 0u; uint32_t auxIndex = 0u; ScalarType scalarType = ScalarType::Float32; };
	struct WriteInput { const ICPUPolygonGeometry* geom = nullptr; ScalarType positionScalarType = ScalarType::Float32; const ICPUPolygonGeometry::SDataView* uvView = nullptr; ScalarType uvScalarType = ScalarType::Float32; const core::vector<ExtraAuxView>* extraAuxViews = nullptr; bool writeNormals = false; ScalarType normalScalarType = ScalarType::Float32; size_t vertexCount = 0ull; size_t faceCount = 0ull; bool write16BitIndices = false; bool flipVectors = false; };
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
	struct BinarySink
	{
		uint8_t* cursor = nullptr;
		template<typename T>
		inline bool append(const T value) { if (!cursor) return false; Binary::storeUnalignedAdvance(cursor, value); return true; }
		inline bool finishVertex() { return true; }
	};
	struct TextSink
	{
		std::string& output;
		template<typename T>
		inline bool append(const T value)
		{
			if constexpr (std::is_floating_point_v<T>) appendFloat(output, static_cast<double>(value));
			else appendIntegral(output, value);
			output.push_back(' ');
			return true;
		}
		inline bool finishVertex() { output.push_back('\n'); return true; }
	};
	template<typename Sink>
	struct PreparedView
	{
		using EmitFn = bool(*)(Sink&, const PreparedView&, size_t);
		uint32_t components = 0u;
		bool flipVectors = false;
		SemanticDecode semantic = {};
		StoredDecode stored = {};
		EmitFn emit = nullptr;
		inline explicit operator bool() const { return emit != nullptr && (static_cast<bool>(semantic) || static_cast<bool>(stored)); }
		inline bool operator()(Sink& sink, const size_t ix) const { return static_cast<bool>(*this) && emit(sink, *this, ix); }
		template<typename OutT, SGeometryViewDecode::EMode Mode>
		static bool emitDecode(Sink& sink, const auto& decode, const size_t ix, const uint32_t components, const bool flipVectors)
		{
			std::array<OutT, 4> decoded = {};
			if (!decode.decode(ix, decoded))
				return false;
			for (uint32_t c = 0u; c < components; ++c)
			{
				OutT value = decoded[c];
				if constexpr (std::is_signed_v<OutT> || std::is_floating_point_v<OutT>)
				{
					if (flipVectors && c == 0u)
						value = -value;
				}
				if (!sink.append(value))
					return false;
			}
			return true;
		}
		template<typename OutT, SGeometryViewDecode::EMode Mode>
		static bool emitPrepared(Sink& sink, const PreparedView& view, const size_t ix) { if constexpr (Mode == SGeometryViewDecode::EMode::Semantic) return emitDecode<OutT, Mode>(sink, view.semantic, ix, view.components, view.flipVectors); return emitDecode<OutT, Mode>(sink, view.stored, ix, view.components, view.flipVectors); }
		template<typename OutT, SGeometryViewDecode::EMode Mode>
		static inline void prepareDecode(PreparedView& view, const ICPUPolygonGeometry::SDataView& src, const bool flipVectors) { view.flipVectors = flipVectors; if constexpr (Mode == SGeometryViewDecode::EMode::Semantic) view.semantic = SGeometryViewDecode::prepare<Mode>(src); else view.stored = SGeometryViewDecode::prepare<Mode>(src); view.emit = &emitPrepared<OutT, Mode>; }
		static PreparedView create(const ICPUPolygonGeometry::SDataView* view, const uint32_t components, const ScalarType scalarType, const bool flipVectors)
		{
			PreparedView retval = {.components = components};
			if (!view)
				return retval;
			switch (scalarType)
			{
				case ScalarType::Float64: prepareDecode<double, SGeometryViewDecode::EMode::Semantic>(retval, *view, flipVectors); break;
				case ScalarType::Float32: prepareDecode<float, SGeometryViewDecode::EMode::Semantic>(retval, *view, flipVectors); break;
				case ScalarType::Int8: prepareDecode<int8_t, SGeometryViewDecode::EMode::Stored>(retval, *view, flipVectors); break;
				case ScalarType::UInt8: prepareDecode<uint8_t, SGeometryViewDecode::EMode::Stored>(retval, *view, false); break;
				case ScalarType::Int16: prepareDecode<int16_t, SGeometryViewDecode::EMode::Stored>(retval, *view, flipVectors); break;
				case ScalarType::UInt16: prepareDecode<uint16_t, SGeometryViewDecode::EMode::Stored>(retval, *view, false); break;
				case ScalarType::Int32: prepareDecode<int32_t, SGeometryViewDecode::EMode::Stored>(retval, *view, flipVectors); break;
				case ScalarType::UInt32: prepareDecode<uint32_t, SGeometryViewDecode::EMode::Stored>(retval, *view, false); break;
			}
			return retval;
		}
	};
	template<typename Sink>
	static bool emitVertices(const WriteInput& input, Sink& sink)
	{
		if (!input.geom || !input.extraAuxViews)
			return false;
		const auto& positionView = input.geom->getPositionView();
		const auto& normalView = input.geom->getNormalView();
		const auto& extraAuxViews = *input.extraAuxViews;
		const PreparedView<Sink> preparedPosition = PreparedView<Sink>::create(&positionView, 3u, input.positionScalarType, input.flipVectors);
		const PreparedView<Sink> preparedNormal = input.writeNormals ? PreparedView<Sink>::create(&normalView, 3u, input.normalScalarType, input.flipVectors) : PreparedView<Sink>{};
		const PreparedView<Sink> preparedUV = input.uvView ? PreparedView<Sink>::create(input.uvView, 2u, input.uvScalarType, false) : PreparedView<Sink>{};
		core::vector<PreparedView<Sink>> preparedExtraAuxViews;
		preparedExtraAuxViews.reserve(extraAuxViews.size());
		for (const auto& extra : extraAuxViews)
			preparedExtraAuxViews.push_back(extra.view ? PreparedView<Sink>::create(extra.view, extra.components, extra.scalarType, false) : PreparedView<Sink>{});
		for (size_t i = 0u; i < input.vertexCount; ++i)
		{
			if (!preparedPosition(sink, i))
				return false;
			if (input.writeNormals && !preparedNormal(sink, i))
				return false;
			if (input.uvView && !preparedUV(sink, i))
				return false;
			for (size_t extraIx = 0u; extraIx < extraAuxViews.size(); ++extraIx)
			{
				if (!extraAuxViews[extraIx].view || !preparedExtraAuxViews[extraIx](sink, i))
					return false;
			}
			if (!sink.finishVertex())
				return false;
		}
		return true;
	}
	static bool writeBinary(const WriteInput& input, uint8_t* dst)
	{
		BinarySink sink = {.cursor = dst};
		if (!emitVertices(input, sink))
			return false;
		return SGeometryWriterCommon::visitTriangleIndices(input.geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2) -> bool {
			if (!sink.append(static_cast<uint8_t>(3u)))
				return false;
			if (input.write16BitIndices)
			{
				if (!sink.append(static_cast<uint16_t>(i0)) || !sink.append(static_cast<uint16_t>(i1)) || !sink.append(static_cast<uint16_t>(i2)))
					return false;
			}
			else if (!sink.append(i0) || !sink.append(i1) || !sink.append(i2))
				return false;
			return true;
		});
	}
	static bool writeText(const WriteInput& input, std::string& output)
	{
		TextSink sink = {.output = output};
		if (!emitVertices(input, sink))
			return false;
		return SGeometryWriterCommon::visitTriangleIndices(input.geom, [&](const uint32_t i0, const uint32_t i1, const uint32_t i2) {
			output.append("3 ");
			appendIntegral(output, i0);
			output.push_back(' ');
			appendIntegral(output, i1);
			output.push_back(' ');
			appendIntegral(output, i2);
			output.push_back('\n');
		});
	}
};
}
bool CPLYMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
	using ScalarType = Parse::ScalarType;
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
	const ICPUPolygonGeometry::SDataView* uvView = SGeometryWriterCommon::getAuxViewAt(geom, Parse::UV0, vertexCount);
	if (uvView && getFormatChannelCount(uvView->composed.format) != 2u)
		uvView = nullptr;
	core::vector<Parse::ExtraAuxView> extraAuxViews;
	const auto& auxViews = geom->getAuxAttributeViews();
	extraAuxViews.reserve(auxViews.size());
	for (uint32_t auxIx = 0u; auxIx < static_cast<uint32_t>(auxViews.size()); ++auxIx)
	{
		const auto& view = auxViews[auxIx];
		if (!view || (uvView && auxIx == Parse::UV0))
			continue;
		if (view.getElementCount() != vertexCount)
			continue;
		const uint32_t channels = getFormatChannelCount(view.composed.format);
		if (channels == 0u)
			continue;
		const uint32_t components = std::min(4u, channels);
		extraAuxViews.push_back({&view, components, auxIx, Parse::selectScalarType(view.composed.format)});
	}
	const auto* indexing = geom->getIndexingCallback();
	if (!indexing)
		return _params.logger.log("PLY writer: missing indexing callback.", system::ILogger::ELL_ERROR), false;
	if (indexing->knownTopology() != E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST)
		return _params.logger.log("PLY writer: only triangle-list topology is supported.", system::ILogger::ELL_ERROR), false;
	size_t faceCount = 0ull;
	if (!SGeometryWriterCommon::getTriangleFaceCount(geom, faceCount))
		return _params.logger.log("PLY writer: failed to validate triangle indexing.", system::ILogger::ELL_ERROR), false;
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
	const Parse::WriteInput input = {.geom = geom, .positionScalarType = positionScalarType, .uvView = uvView, .uvScalarType = uvScalarType, .extraAuxViews = &extraAuxViews, .writeNormals = writeNormals, .normalScalarType = normalScalarType, .vertexCount = vertexCount, .faceCount = faceCount, .write16BitIndices = write16BitIndices, .flipVectors = flipVectors};
	bool writeOk = false;
	size_t outputBytes = 0ull;
	auto writePayload = [&](const void* bodyData, const size_t bodySize) -> bool {
		const size_t outputSize = header.size() + bodySize;
		const auto ioPlan = impl::SFileAccess::resolvePlan(_params.ioPolicy, static_cast<uint64_t>(outputSize), true, file);
		if (impl::SFileAccess::logInvalidPlan(_params.logger, "PLY writer", file->getFileName().string().c_str(), ioPlan))
			return false;
		outputBytes = outputSize;
		const SInterchangeIO::SBufferRange writeBuffers[] = {{.data = header.data(), .byteCount = header.size()}, {.data = bodyData, .byteCount = bodySize}};
		writeOk = SInterchangeIO::writeBuffersWithPolicy(file, ioPlan, writeBuffers, &ioTelemetry);
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
		body.resize(bodySize);
		if (!Parse::writeBinary(input, body.data()))
			return _params.logger.log("PLY writer: binary payload generation failed.", system::ILogger::ELL_ERROR), false;
		return writePayload(body.data(), body.size());
	}
	std::string body;
	body.reserve(vertexCount * Parse::ApproxTextBytesPerVertex + faceCount * Parse::ApproxTextBytesPerFace);
	if (!Parse::writeText(input, body))
		return _params.logger.log("PLY writer: text payload generation failed.", system::ILogger::ELL_ERROR), false;
	return writePayload(body.data(), body.size());
}
}
#endif // _NBL_COMPILE_WITH_PLY_WRITER_

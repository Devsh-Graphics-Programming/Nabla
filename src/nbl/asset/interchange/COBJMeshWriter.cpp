// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/COBJMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_

#include "nbl/system/IFile.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
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

static inline bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
	out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
	return view.decodeElement(ix, out);
}

static inline const hlsl::float32_t3* getTightFloat3View(const ICPUPolygonGeometry::SDataView& view)
{
	if (!view)
		return nullptr;
	if (view.composed.format != EF_R32G32B32_SFLOAT)
		return nullptr;
	if (view.composed.getStride() != sizeof(hlsl::float32_t3))
		return nullptr;
	return reinterpret_cast<const hlsl::float32_t3*>(view.getPointer());
}

static inline const hlsl::float32_t2* getTightFloat2View(const ICPUPolygonGeometry::SDataView& view)
{
	if (!view)
		return nullptr;
	if (view.composed.format != EF_R32G32_SFLOAT)
		return nullptr;
	if (view.composed.getStride() != sizeof(hlsl::float32_t2))
		return nullptr;
	return reinterpret_cast<const hlsl::float32_t2*>(view.getPointer());
}

static inline void appendUInt(std::string& out, const uint32_t value)
{
	std::array<char, 16> buf = {};
	const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value);
	if (res.ec == std::errc())
		out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
}

static inline void appendFloatFixed6(std::string& out, double value)
{
	std::array<char, 64> buf = {};
	const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), value, std::chars_format::fixed, 6);
	if (res.ec == std::errc())
	{
		out.append(buf.data(), static_cast<size_t>(res.ptr - buf.data()));
		return;
	}

	const int written = std::snprintf(buf.data(), buf.size(), "%.6f", value);
	if (written > 0)
		out.append(buf.data(), static_cast<size_t>(written));
}

static bool writeBufferWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* data, size_t byteCount);

bool COBJMeshWriter::writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override)
{
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

	const auto flags = _override->getAssetWritingFlags(ctx, geom, 0u);
	const bool flipHandedness = !(flags & E_WRITER_FLAGS::EWF_MESH_IS_RIGHT_HANDED);
	std::string output;
	output.reserve(vertexCount * 96ull + faceCount * 48ull);

	output += "# Nabla OBJ\n";

	hlsl::float64_t4 tmp = {};
	const hlsl::float32_t3* const tightPositions = getTightFloat3View(positionView);
	const hlsl::float32_t3* const tightNormals = hasNormals ? getTightFloat3View(normalView) : nullptr;
	const hlsl::float32_t2* const tightUV = hasUVs ? getTightFloat2View(*uvView) : nullptr;
	for (size_t i = 0u; i < vertexCount; ++i)
	{
		double x = 0.0;
		double y = 0.0;
		double z = 0.0;
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
			x = tmp.x;
			y = tmp.y;
			z = tmp.z;
		}
		if (flipHandedness)
			x = -x;

		output += "v ";
		appendFloatFixed6(output, x);
		output += " ";
		appendFloatFixed6(output, y);
		output += " ";
		appendFloatFixed6(output, z);
		output += "\n";
	}

	if (hasUVs)
	{
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			double u = 0.0;
			double v = 0.0;
			if (tightUV)
			{
				u = tightUV[i].x;
				v = 1.0 - tightUV[i].y;
			}
			else
			{
				if (!decodeVec4(*uvView, i, tmp))
					return false;
				u = tmp.x;
				v = 1.0 - tmp.y;
			}

			output += "vt ";
			appendFloatFixed6(output, u);
			output += " ";
			appendFloatFixed6(output, v);
			output += "\n";
		}
	}

	if (hasNormals)
	{
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			double x = 0.0;
			double y = 0.0;
			double z = 0.0;
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
				x = tmp.x;
				y = tmp.y;
				z = tmp.z;
			}
			if (flipHandedness)
				x = -x;

			output += "vn ";
			appendFloatFixed6(output, x);
			output += " ";
			appendFloatFixed6(output, y);
			output += " ";
			appendFloatFixed6(output, z);
			output += "\n";
		}
	}

	auto appendFaceIndex = [&](const uint32_t idx)
	{
		const uint32_t objIx = idx + 1u;
		appendUInt(output, objIx);
		if (hasUVs && hasNormals)
		{
			output += "/";
			appendUInt(output, objIx);
			output += "/";
			appendUInt(output, objIx);
		}
		else if (hasUVs)
		{
			output += "/";
			appendUInt(output, objIx);
		}
		else if (hasNormals)
		{
			output += "//";
			appendUInt(output, objIx);
		}
	};

	for (size_t i = 0u; i < faceCount; ++i)
	{
		const uint32_t i0 = indices[i * 3u + 0u];
		const uint32_t i1 = indices[i * 3u + 1u];
		const uint32_t i2 = indices[i * 3u + 2u];

		const uint32_t f0 = i2;
		const uint32_t f1 = i1;
		const uint32_t f2 = i0;

		output += "f ";
		appendFaceIndex(f0);
		output += " ";
		appendFaceIndex(f1);
		output += " ";
		appendFaceIndex(f2);
		output += "\n";
	}

	const auto ioPlan = resolveFileIOPolicy(_params.ioPolicy, static_cast<uint64_t>(output.size()), true);
	if (!ioPlan.valid)
	{
		_params.logger.log("OBJ writer: invalid io policy for %s reason=%s", system::ILogger::ELL_ERROR, file->getFileName().string().c_str(), ioPlan.reason);
		return false;
	}

	return writeBufferWithPolicy(file, ioPlan, reinterpret_cast<const uint8_t*>(output.data()), output.size());
}

static bool writeBufferWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* data, size_t byteCount)
{
	if (!file || (!data && byteCount != 0ull))
		return false;

	size_t fileOffset = 0ull;
	switch (ioPlan.strategy)
	{
		case SResolvedFileIOPolicy::Strategy::WholeFile:
		{
			system::IFile::success_t success;
			file->write(success, data, fileOffset, byteCount);
			return success && success.getBytesProcessed() == byteCount;
		}
		case SResolvedFileIOPolicy::Strategy::Chunked:
		default:
		{
			while (fileOffset < byteCount)
			{
				const size_t toWrite = static_cast<size_t>(std::min<uint64_t>(ioPlan.chunkSizeBytes, byteCount - fileOffset));
				system::IFile::success_t success;
				file->write(success, data + fileOffset, fileOffset, toWrite);
				if (!success)
					return false;
				const size_t written = success.getBytesProcessed();
				if (written == 0ull)
					return false;
				fileOffset += written;
			}
			return true;
		}
	}
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_OBJ_WRITER_


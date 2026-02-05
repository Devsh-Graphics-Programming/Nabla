// Copyright (C) 2025-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/COBJMeshWriter.h"

#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_

#include "nbl/system/IFile.h"

#include <sstream>
#include <iomanip>

namespace nbl::asset
{

COBJMeshWriter::COBJMeshWriter()
{
	#ifdef _NBL_DEBUG
	setDebugName("COBJMeshWriter");
	#endif
}

static inline bool decodeVec4(const ICPUPolygonGeometry::SDataView& view, const size_t ix, hlsl::float64_t4& out)
{
	out = hlsl::float64_t4(0.0, 0.0, 0.0, 0.0);
	return view.decodeElement(ix, out);
}

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

		indexData.resize(indexCount);
		const void* src = indexView.getPointer();
		if (!src)
			return false;

		if (indexView.composed.format == EF_R32_UINT)
		{
			memcpy(indexData.data(), src, indexCount * sizeof(uint32_t));
		}
		else if (indexView.composed.format == EF_R16_UINT)
		{
			const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
			for (size_t i = 0; i < indexCount; ++i)
				indexData[i] = src16[i];
		}
		else
		{
			return false;
		}

		indices = indexData.data();
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

	SAssetWriteContext writeCtx = { ctx.params, file };
	size_t fileOffset = 0u;

	auto writeString = [&](const std::string& str)
	{
		system::IFile::success_t success;
		writeCtx.outputFile->write(success, str.c_str(), fileOffset, str.size());
		fileOffset += success.getBytesProcessed();
	};

	{
		std::string header = "# Nabla OBJ\n";
		writeString(header);
	}

	hlsl::float64_t4 tmp = {};
	for (size_t i = 0u; i < vertexCount; ++i)
	{
		if (!decodeVec4(positionView, i, tmp))
			return false;

		double x = tmp.x;
		double y = tmp.y;
		double z = tmp.z;
		if (flipHandedness)
			x = -x;

		std::ostringstream ss;
		ss << std::fixed << std::setprecision(6);
		ss << "v " << x << " " << y << " " << z << "\n";
		writeString(ss.str());
	}

	if (hasUVs)
	{
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			if (!decodeVec4(*uvView, i, tmp))
				return false;
			const double u = tmp.x;
			const double v = 1.0 - tmp.y;

			std::ostringstream ss;
			ss << std::fixed << std::setprecision(6);
			ss << "vt " << u << " " << v << "\n";
			writeString(ss.str());
		}
	}

	if (hasNormals)
	{
		for (size_t i = 0u; i < vertexCount; ++i)
		{
			if (!decodeVec4(normalView, i, tmp))
				return false;

			double x = tmp.x;
			double y = tmp.y;
			double z = tmp.z;
			if (flipHandedness)
				x = -x;

			std::ostringstream ss;
			ss << std::fixed << std::setprecision(6);
			ss << "vn " << x << " " << y << " " << z << "\n";
			writeString(ss.str());
		}
	}

	for (size_t i = 0u; i < faceCount; ++i)
	{
		const uint32_t i0 = indices[i * 3u + 0u];
		const uint32_t i1 = indices[i * 3u + 1u];
		const uint32_t i2 = indices[i * 3u + 2u];

		const uint32_t f0 = i2;
		const uint32_t f1 = i1;
		const uint32_t f2 = i0;

		auto emitIndex = [&](std::ostringstream& ss, const uint32_t idx)
		{
			const uint32_t objIx = idx + 1u;
			if (hasUVs && hasNormals)
				ss << objIx << "/" << objIx << "/" << objIx;
			else if (hasUVs)
				ss << objIx << "/" << objIx;
			else if (hasNormals)
				ss << objIx << "//" << objIx;
			else
				ss << objIx;
		};

		std::ostringstream ss;
		ss << "f ";
		emitIndex(ss, f0);
		ss << " ";
		emitIndex(ss, f1);
		ss << " ";
		emitIndex(ss, f2);
		ss << "\n";
		writeString(ss.str());
	}

	return true;
}

} // namespace nbl::asset

#endif // _NBL_COMPILE_WITH_OBJ_WRITER_

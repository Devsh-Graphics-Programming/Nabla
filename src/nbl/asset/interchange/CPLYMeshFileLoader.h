// Copyright (C) 2019-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_C_PLY_MESH_FILE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_PLY_MESH_FILE_LOADER_H_INCLUDED_
#ifdef _NBL_COMPILE_WITH_PLY_LOADER_

#include "nbl/core/declarations.h"

#include "nbl/asset/interchange/IGeometryLoader.h"

#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/metadata/CPLYMetadata.h"

namespace nbl::asset
{

//! Meshloader capable of loading obj meshes.
class CPLYMeshFileLoader final : public IGeometryLoader
{
	public:
		inline CPLYMeshFileLoader() = default;

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "ply", nullptr };
			return ext;
		}

		//! creates/loads an animated mesh from the file.
		virtual SAssetBundle loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

private:
	enum E_TYPE { ET_POS = 0, ET_UV = 2, ET_NORM = 3, ET_COL = 1 };

 	bool readVertex(SContext& _ctx, const SPLYElement &Element, asset::SBufferBinding<asset::ICPUBuffer> outAttributes[4], const uint32_t& currentVertexIndex, const IAssetLoader::SAssetLoadParams& _params);
	bool readFace(SContext& _ctx, const SPLYElement &Element, core::vector<uint32_t>& _outIndices);

	void skipElement(SContext& _ctx, const SPLYElement &Element);
	void skipProperty(SContext& _ctx, const SPLYProperty &Property);
	float getFloat(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
	uint32_t getInt(SContext& _ctx, E_PLY_PROPERTY_TYPE t);
	void moveForward(SContext& _ctx, uint32_t bytes);
#if 0
	bool genVertBuffersForMBuffer(
		ICPUMeshBuffer* _mbuf,
		const asset::SBufferBinding<asset::ICPUBuffer> attributes[4],
		SContext& context
	) const;
#endif
	template<typename aType>
	static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
	{
		performOnCertainOrientation(varToHandle);
	}
};

} // end namespace nbl::asset
#endif
#endif

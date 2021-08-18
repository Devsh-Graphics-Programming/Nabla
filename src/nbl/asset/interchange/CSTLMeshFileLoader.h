// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_STL_MESH_FILE_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_STL_MESH_FILE_LOADER_H_INCLUDED__

#include "nbl/asset/interchange/IAssetLoader.h"
#include "nbl/asset/metadata/CSTLMetadata.h"

namespace nbl
{
namespace asset
{

//! Meshloader capable of loading STL meshes.
class CSTLMeshFileLoader final : public IRenderpassIndependentPipelineLoader
{
	public:

		CSTLMeshFileLoader(asset::IAssetManager* _m_assetMgr);

		asset::SAssetBundle loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

		bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "stl", nullptr };
			return ext;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return IAsset::ET_MESH; }

	private:

		virtual void initialize() override;

		const std::string_view getPipelineCacheKey(bool withColorAttribute) { return withColorAttribute ? "nbl/builtin/pipeline/loader/STL/color_attribute" : "nbl/builtin/pipeline/loader/STL/no_color_attribute"; }

		// skips to the first non-space character available
		void goNextWord(system::IFile* file) const;
		// returns the next word

		const std::string_view getNextToken(system::IFile* file, const std::string_view token) const;
		// skip to next printable character after the first line break
		void goNextLine(system::IFile* file) const;
		//! Read 3d vector of floats
		void getNextVector(system::IFile* file, core::vectorSIMDf& vec, bool binary) const;

		template<typename aType>
		static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
		{
			performOnCertainOrientation(varToHandle);
		}

		asset::IAssetManager* m_assetMgr;
};

}	// end namespace scene
}	// end namespace nbl

#endif


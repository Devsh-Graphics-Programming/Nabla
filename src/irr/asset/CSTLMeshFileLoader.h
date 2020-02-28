// Copyright (C) 2007-2012 Christian Stehno
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_STL_MESH_FILE_LOADER_H_INCLUDED__
#define __C_STL_MESH_FILE_LOADER_H_INCLUDED__

#include "irr/asset/IAssetLoader.h"

namespace irr
{
namespace asset
{
//! Meshloader capable of loading STL meshes.
class CSTLMeshFileLoader final : public IAssetLoader
{
	public:
		asset::SAssetBundle loadAsset(io::IReadFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

#ifdef NEW_SHADERS
		bool isALoadableFileFormat(io::IReadFile* _file) const override { return false; }
#else
		bool isALoadableFileFormat(io::IReadFile* _file) const override;
#endif

		const char** getAssociatedFileExtensions() const override
		{
			static const char* ext[]{ "stl", nullptr };
			return ext;
		}

		uint64_t getSupportedAssetTypesBitfield() const override { return IAsset::ET_MESH; }

	private:

		// skips to the first non-space character available
		void goNextWord(io::IReadFile* file) const;
		// returns the next word
		const core::stringc& getNextToken(io::IReadFile* file, core::stringc& token) const;
		// skip to next printable character after the first line break
		void goNextLine(io::IReadFile* file) const;

		//! Read 3d vector of floats
		void getNextVector(io::IReadFile* file, core::vectorSIMDf& vec, bool binary) const;

		template<typename aType>
		static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
		{
			performOnCertainOrientation(varToHandle);
		}
};

} // end namespace scene
} // end namespace irr

#endif


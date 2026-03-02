// Copyright (C) 2019-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors
#ifndef _NBL_ASSET_C_STL_MESH_FILE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_STL_MESH_FILE_LOADER_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/asset/interchange/IGeometryLoader.h"
#include "nbl/asset/metadata/CSTLMetadata.h"


namespace nbl::asset
{

//! Meshloader capable of loading STL meshes.
class CSTLMeshFileLoader final : public IGeometryLoader
{
   public:

      inline CSTLMeshFileLoader() = default;

      asset::SAssetBundle loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;

      bool isALoadableFileFormat(system::IFile* _file, const system::logger_opt_ptr logger) const override;

      const char** getAssociatedFileExtensions() const override
      {
         static const char* ext[]{ "stl", nullptr };
         return ext;
      }

   private:
      const std::string_view getPipelineCacheKey(bool withColorAttribute) { return withColorAttribute ? "nbl/builtin/pipeline/loader/STL/color_attribute" : "nbl/builtin/pipeline/loader/STL/no_color_attribute"; }

      asset::IAssetManager* m_assetMgr;

      template <typename aType>
      static inline void performActionBasedOnOrientationSystem(aType& varToHandle, void (*performOnCertainOrientation)(aType& varToHandle))
      {
         performOnCertainOrientation(varToHandle);
      }
};

}	// end namespace nbl::scene
#endif


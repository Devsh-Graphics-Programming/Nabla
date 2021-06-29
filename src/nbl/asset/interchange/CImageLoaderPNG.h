// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

// this file was created by rt (www.tomkorp.com), based on ttk's png-reader

#ifndef __NBL_ASSET_C_IMAGE_LOADER_PNG_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_LOADER_PNG_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef _NBL_COMPILE_WITH_PNG_LOADER_

#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl
{
namespace asset
{

//!  Surface Loader for PNG files
class CImageLoaderPng : public asset::IAssetLoader
{
        core::smart_refctd_ptr<system::ISystem> m_system;
    public:
        explicit CImageLoaderPng(core::smart_refctd_ptr<system::ISystem>&& sys) : m_system(std::move(sys)) {}
        virtual bool isALoadableFileFormat(system::IFile* _file) const override;

        virtual const char** getAssociatedFileExtensions() const override
        {
            static const char* ext[]{ "png", nullptr };
            return ext;
        }

        virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

        virtual asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};


} // end namespace video
} // end namespace nbl

#endif
#endif


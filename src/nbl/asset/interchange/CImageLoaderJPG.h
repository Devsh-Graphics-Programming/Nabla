// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_IMAGE_LOADER_JPG_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_LOADER_JPG_H_INCLUDED__

#include "nbl/core/core.h"

#ifdef _NBL_COMPILE_WITH_JPG_LOADER_

#include "nbl/asset/interchange/IAssetLoader.h"


namespace nbl
{
namespace asset
{

//! Surface Loader for JPG images
class CImageLoaderJPG : public asset::IAssetLoader
{
    protected:
	    //! destructor
	    virtual ~CImageLoaderJPG();

    public:
	    //! constructor
	    CImageLoaderJPG();

        virtual bool isALoadableFileFormat(system::IFile* _file) const override;

        virtual const char** getAssociatedFileExtensions() const override
        {
            static const char* ext[]{ "jpg", "jpeg", "jpe", "jif", "jfif", "jfi", nullptr };
            return ext;
        }

        virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

        virtual asset::SAssetBundle loadAsset(system::IFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

} // end namespace video
} // end namespace nbl


#endif
#endif


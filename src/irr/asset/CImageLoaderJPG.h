// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IMAGE_LOADER_JPG_H_INCLUDED__
#define __C_IMAGE_LOADER_JPG_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_JPG_LOADER_

#include "irr/asset/IAssetLoader.h"



namespace irr
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

    virtual bool isALoadableFileFormat(io::IReadFile* _file) const override;

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{ "jpg", "jpeg", "jpe", "jif", "jfif", "jfi", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE; }

    virtual asset::IAsset* loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};


} // end namespace video
} // end namespace irr


#endif
#endif


// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_IMAGE_WRITER_JPG_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_WRITER_JPG_H_INCLUDED__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_JPG_WRITER_

#include "nbl/asset/IAssetWriter.h"

namespace nbl
{
namespace asset
{

class CImageWriterJPG : public asset::IAssetWriter
{
public:
	//! constructor
	CImageWriterJPG();

    virtual const char** getAssociatedFileExtensions() const
    {
        static const char* ext[]{ "jpg", "jpeg", "jpe", "jif", "jfif", "jfi", nullptr };
        return ext;
    }

    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

    virtual uint32_t getSupportedFlags() override { return asset::EWF_COMPRESSED; }

    virtual uint32_t getForcedFlags() { return asset::EWF_BINARY; }

    virtual bool writeAsset(io::IWriteFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

}
}

#endif
#endif


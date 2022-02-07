// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IMAGE_WRITER_OPENEXR__
#define __NBL_ASSET_C_IMAGE_WRITER_OPENEXR__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_OPENEXR_WRITER_

#include "nbl/asset/interchange/IImageWriter.h"

namespace nbl
{
namespace asset
{
//! OpenEXR writer capable of saving .exr files
class CImageWriterOpenEXR final : public IImageWriter
{
protected:
    ~CImageWriterOpenEXR() {}

public:
    CImageWriterOpenEXR() {}

    const char** getAssociatedFileExtensions() const override
    {
        static const char* extensions[]{"exr", nullptr};
        return extensions;
    }

    uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

    uint32_t getSupportedFlags() override { return asset::EWF_BINARY; }

    uint32_t getForcedFlags() { return asset::EWF_BINARY; }

    bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

private:
    bool writeImageBinary(system::IFile* file, const asset::ICPUImage* image);
};

}
}

#endif  // _NBL_COMPILE_WITH_OPENEXR_WRITER_
#endif

// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_ASSET_C_IMAGE_WRITER_PNG_H_INCLUDED__
#define __NBL_ASSET_C_IMAGE_WRITER_PNG_H_INCLUDED__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_PNG_WRITER_

#include "nbl/asset/interchange/IAssetWriter.h"

namespace nbl
{
namespace asset
{

class CImageWriterPNG : public asset::IAssetWriter
{
    core::smart_refctd_ptr<system::ISystem> m_system;
public:
    struct SContext
    {
        SContext(system::ISystem* sys) : system(sys) {}
        system::ISystem* system;
        size_t file_pos = 0;
    };
    //! constructor
    explicit CImageWriterPNG(core::smart_refctd_ptr<system::ISystem>&& sys);
    
    virtual const char** getAssociatedFileExtensions() const
    {
        static const char* ext[]{ "png", nullptr };
        return ext;
    }
    
    virtual uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }
    
    virtual uint32_t getSupportedFlags() override { return 0u; }
    
    virtual uint32_t getForcedFlags() { return asset::EWF_BINARY; }
    
    virtual bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;
};

} // namespace video
} // namespace nbl

#endif
#endif


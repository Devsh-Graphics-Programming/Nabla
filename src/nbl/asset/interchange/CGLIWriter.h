// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IMAGE_WRITER_GLI__
#define __NBL_ASSET_C_IMAGE_WRITER_GLI__

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_GLI_WRITER_

#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/interchange/IAssetWriter.h"

namespace nbl
{
namespace asset
{
//! Texture writer capable of saving in .ktx, .dds and .kmg file extensions
class CGLIWriter final : public asset::IAssetWriter
{
    core::smart_refctd_ptr<system::ISystem> m_system;

protected:
    virtual ~CGLIWriter() {}

public:
    explicit CGLIWriter(core::smart_refctd_ptr<system::ISystem>&& sys)
        : m_system(std::move(sys)) {}

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* extensions[]{"ktx", "dds", "kmg", nullptr};
        return extensions;
    }

    uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_IMAGE_VIEW; }

    uint32_t getSupportedFlags() override { return asset::EWF_NONE | asset::EWF_BINARY; }

    uint32_t getForcedFlags() override { return asset::EWF_NONE | asset::EWF_BINARY; }

    bool writeAsset(system::IFile* _file, const SAssetWriteParams& _params, IAssetWriterOverride* _override = nullptr) override;

protected:
private:
    bool writeGLIFile(system::IFile* file, const asset::ICPUImageView* imageView, const system::logger_opt_ptr logger);

    static inline bool doesItHaveFaces(const IImageView<ICPUImage>::E_TYPE& type)
    {
        switch(type)
        {
            case ICPUImageView::ET_CUBE_MAP: return true;
            case ICPUImageView::ET_CUBE_MAP_ARRAY: return true;
            default: return false;
        }
    }
    static inline bool doesItHaveLayers(const IImageView<ICPUImage>::E_TYPE& type)
    {
        switch(type)
        {
            case ICPUImageView::ET_1D_ARRAY: return true;
            case ICPUImageView::ET_2D_ARRAY: return true;
            case ICPUImageView::ET_CUBE_MAP_ARRAY: return true;
            default: return false;
        }
    }
};

}
}

#endif  // _NBL_COMPILE_WITH_GLI_WRITER_
#endif

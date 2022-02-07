// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SPIR_V_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_SPIR_V_LOADER_H_INCLUDED__

#include "nbl/asset/interchange/IAssetLoader.h"

namespace nbl
{
namespace asset
{
class CSPVLoader final : public asset::IAssetLoader
{
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t SPV_MAGIC_NUMBER = 0x07230203u;

public:
    bool isALoadableFileFormat(io::IReadFile* _file) const override
    {
        uint32_t magicNumber = 0u;

        const size_t prevPos = _file->getPos();
        _file->seek(0u);
        _file->read(&magicNumber, sizeof(uint32_t));
        _file->seek(prevPos);

        return magicNumber == SPV_MAGIC_NUMBER;
    }

    const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{"spv", nullptr};
        return ext;
    }

    uint64_t getSupportedAssetTypesBitfield() const override { return asset::IAsset::ET_SHADER; }

    asset::SAssetBundle loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override = nullptr, uint32_t _hierarchyLevel = 0u) override;
};

}  // namespace asset
}  // namespace nbl

#endif

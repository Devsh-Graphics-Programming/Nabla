// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CBufferLoaderBIN.h"

namespace nbl
{
namespace asset
{
asset::SAssetBundle CBufferLoaderBIN::loadAsset(io::IReadFile* _file, const asset::IAssetLoader::SAssetLoadParams& _params, asset::IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
    if(!_file)
        return {};

    SContext ctx(_file->getSize());
    ctx.file = _file;

    ctx.file->read(ctx.sourceCodeBuffer.get()->getPointer(), ctx.file->getSize());

    return SAssetBundle(nullptr, {std::move(ctx.sourceCodeBuffer)});
}

bool CBufferLoaderBIN::isALoadableFileFormat(io::IReadFile* _file) const
{
    return true;  // validation if needed
}
}
}
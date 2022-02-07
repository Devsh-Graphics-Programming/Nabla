// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/bawformat/CBAWMeshFileLoader.h"

namespace nbl
{
namespace asset
{
template<>  // V2 to V3 conversion
io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<3>(SContext& _ctx, io::IReadFile* _baw2file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<2>& _common);

template<>  // V1 to V2 conversion
io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<2>(SContext& _ctx, io::IReadFile* _baw1file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<1>& _common);

template<>  // V0 to V1 conversion
io::IReadFile* CBAWMeshFileLoader::createConvertIntoVer_spec<1>(SContext& _ctx, io::IReadFile* _baw0file, asset::IAssetLoader::IAssetLoaderOverride* _override, const CommonDataTuple<0>& _common);
}
}

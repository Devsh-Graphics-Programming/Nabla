// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_HLSL_METADATA_H_INCLUDED__
#define __NBL_ASSET_C_HLSL_METADATA_H_INCLUDED__

#include "nbl/asset/metadata/IAssetMetadata.h"
#include "nbl/builtin/hlsl/enums.hlsl"

namespace nbl
{
namespace asset
{

class CHLSLMetadata final : public IAssetMetadata
{
    public:
        explicit CHLSLMetadata(hlsl::ShaderStage shaderStage): shaderStage(shaderStage) {}

        _NBL_STATIC_INLINE_CONSTEXPR const char* LoaderName = "CHLSLLoader";
        const char* getLoaderName() const override { return LoaderName; }
        
        hlsl::ShaderStage shaderStage;
};

}
}

#endif

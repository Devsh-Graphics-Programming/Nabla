// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GLSL_VERTEX_SHADER_BUILTIN_LOADER_H_INCLUDED__
#define __NBL_ASSET_C_GLSL_VERTEX_SHADER_BUILTIN_LOADER_H_INCLUDED__

#include "IVideoCapabilityReporter.h"
#include "nbl/asset/IBuiltinIncludeLoader.h"

#include <string>
#include <cstdint>
#include <cassert>

namespace nbl { namespace asset
{

class CGLSLSkinningBuiltinIncludeLoader : public asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/skinning/"; }

protected:
    core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override;

private:
    static std::string getLinearSkinningFunction(uint32_t maxBoneInfluences);
};

}} // nbl::asset

#endif

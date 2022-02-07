// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SPV_UTILS_H_INCLUDED__
#define __NBL_ASSET_C_SPV_UTILS_H_INCLUDED__

#include "nbl/asset/ISpecializedShader.h"
#include "nbl_spirv_cross/spirv_cross.hpp"

namespace nbl
{
namespace asset
{
inline IShader::E_SHADER_STAGE spvExecModel2ESS(spv::ExecutionModel _em)
{
    using namespace spv;
    switch(_em)
    {
        case ExecutionModelVertex: return IShader::ESS_VERTEX;
        case ExecutionModelTessellationControl: return IShader::ESS_TESSELATION_CONTROL;
        case ExecutionModelTessellationEvaluation: return IShader::ESS_TESSELATION_EVALUATION;
        case ExecutionModelGeometry: return IShader::ESS_GEOMETRY;
        case ExecutionModelFragment: return IShader::ESS_FRAGMENT;
        case ExecutionModelGLCompute: return IShader::ESS_COMPUTE;
        default: return IShader::ESS_UNKNOWN;
    }
}
inline spv::ExecutionModel ESS2spvExecModel(IShader::E_SHADER_STAGE _ss)
{
    using namespace spv;
    switch(_ss)
    {
        case IShader::ESS_VERTEX: return ExecutionModelVertex;
        case IShader::ESS_TESSELATION_CONTROL: return ExecutionModelTessellationControl;
        case IShader::ESS_TESSELATION_EVALUATION: return ExecutionModelTessellationEvaluation;
        case IShader::ESS_GEOMETRY: return ExecutionModelGeometry;
        case IShader::ESS_FRAGMENT: return ExecutionModelFragment;
        case IShader::ESS_COMPUTE: return ExecutionModelGLCompute;
        default: return ExecutionModelMax;
    }
}
}
}

#endif
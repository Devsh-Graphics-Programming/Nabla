#ifndef __C_SPV_UTILS_H_INCLUDED__
#define __C_SPV_UTILS_H_INCLUDED__

#include "irr/asset/ShaderCommons.h"
#include "spirv_cross/spirv_cross.hpp"

namespace irr { namespace asset
{
    inline E_SHADER_STAGE spvExecModel2ESS(spv::ExecutionModel _em)
    {
        using namespace spv;
        switch (_em)
        {
        case ExecutionModelVertex: return ESS_VERTEX;
        case ExecutionModelTessellationControl: return ESS_TESSELATION_CONTROL;
        case ExecutionModelTessellationEvaluation: return ESS_TESSELATION_EVALUATION;
        case ExecutionModelGeometry: return ESS_GEOMETRY;
        case ExecutionModelFragment: return ESS_FRAGMENT;
        case ExecutionModelGLCompute: return ESS_COMPUTE;
        default: return ESS_UNKNOWN;
        }
    }
    inline spv::ExecutionModel ESS2spvExecModel(E_SHADER_STAGE _ss)
    {
        using namespace spv;
        switch (_ss)
        {
        case ESS_VERTEX: return ExecutionModelVertex;
        case ESS_TESSELATION_CONTROL: return ExecutionModelTessellationControl;
        case ESS_TESSELATION_EVALUATION: return ExecutionModelTessellationEvaluation;
        case ESS_GEOMETRY: return ExecutionModelGeometry;
        case ESS_FRAGMENT: return ExecutionModelFragment;
        case ESS_COMPUTE: return ExecutionModelGLCompute;
        default: return ExecutionModelMax;
        }
    }
}}

#endif
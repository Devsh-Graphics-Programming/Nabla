#ifndef __C_SPV_UTILS_H_INCLUDED__
#define __C_SPV_UTILS_H_INCLUDED__

#include "irr/asset/ISpecializedShader.h"
#include "spirv_cross/spirv_cross.hpp"

namespace irr
{
namespace asset
{
    inline ISpecializedShader::E_SHADER_STAGE spvExecModel2ESS(spv::ExecutionModel _em)
    {
        using namespace spv;
        switch (_em)
        {
			case ExecutionModelVertex: return ISpecializedShader::ESS_VERTEX;
			case ExecutionModelTessellationControl: return ISpecializedShader::ESS_TESSELATION_CONTROL;
			case ExecutionModelTessellationEvaluation: return ISpecializedShader::ESS_TESSELATION_EVALUATION;
			case ExecutionModelGeometry: return ISpecializedShader::ESS_GEOMETRY;
			case ExecutionModelFragment: return ISpecializedShader::ESS_FRAGMENT;
			case ExecutionModelGLCompute: return ISpecializedShader::ESS_COMPUTE;
			default: return ISpecializedShader::ESS_UNKNOWN;
        }
    }
    inline spv::ExecutionModel ESS2spvExecModel(ISpecializedShader::E_SHADER_STAGE _ss)
    {
        using namespace spv;
        switch (_ss)
        {
			case ISpecializedShader::ESS_VERTEX: return ExecutionModelVertex;
			case ISpecializedShader::ESS_TESSELATION_CONTROL: return ExecutionModelTessellationControl;
			case ISpecializedShader::ESS_TESSELATION_EVALUATION: return ExecutionModelTessellationEvaluation;
			case ISpecializedShader::ESS_GEOMETRY: return ExecutionModelGeometry;
			case ISpecializedShader::ESS_FRAGMENT: return ExecutionModelFragment;
			case ISpecializedShader::ESS_COMPUTE: return ExecutionModelGLCompute;
			default: return ExecutionModelMax;
        }
    }
}}

#endif
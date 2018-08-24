#ifndef __IRR_C_GLSL_FUNCTION_GENERATOR_H_INCLUDED__
#define __IRR_C_GLSL_FUNCTION_GENERATOR_H_INCLUDED__

#include "IrrCompileConfig.h"

#include <string>
#include <cstdint>

namespace irr { namespace core
{

class CGLSLFunctionGenerator
{
    static std::string getLinearSkinningFunction(const uint32_t& maxBoneInfluences = 4u);
};

}} // irr::core

#endif // __IRR_C_GLSL_FUNCTION_GENERATOR_H_INCLUDED__

#ifndef __IRR_SHADERC_UTILS_H_INCLUDED__
#define __IRR_SHADERC_UTILS_H_INCLUDED__

//! This file is not supposed to be included in user-accesible header files

#include <shaderc/shaderc.hpp>
#include "irr/asset/ShaderCommons.h"
#include "irr/core/math/irrMath.h"

namespace irr { namespace asset
{

inline shaderc_shader_kind ESStoShadercEnum(E_SHADER_STAGE _ss)
{
    using T = std::underlying_type_t<E_SHADER_STAGE>;

    shaderc_shader_kind convert[6];
    convert[core::findLSB<uint32_t>(ESS_VERTEX)] = shaderc_vertex_shader;
    convert[core::findLSB<uint32_t>(ESS_TESSELATION_CONTROL)] = shaderc_tess_control_shader;
    convert[core::findLSB<uint32_t>(ESS_TESSELATION_EVALUATION)] = shaderc_tess_evaluation_shader;
    convert[core::findLSB<uint32_t>(ESS_GEOMETRY)] = shaderc_geometry_shader;
    convert[core::findLSB<uint32_t>(ESS_FRAGMENT)] = shaderc_fragment_shader;
    convert[core::findLSB<uint32_t>(ESS_COMPUTE)] = shaderc_compute_shader;

    return convert[core::findLSB<uint32_t>(_ss)];
}

}}

#endif // __IRR_SHADERC_UTILS_H_INCLUDED__


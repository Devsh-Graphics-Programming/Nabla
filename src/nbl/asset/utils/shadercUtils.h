// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_SHADERC_UTILS_H_INCLUDED__
#define __NBL_ASSET_SHADERC_UTILS_H_INCLUDED__

//! This file is not supposed to be included in user-accesible header files

#include <shaderc/shaderc.hpp>
#include "nbl/asset/ISpecializedShader.h"

namespace nbl
{
namespace asset
{

inline shaderc_shader_kind ESStoShadercEnum(IShader::E_SHADER_STAGE _ss)
{
    using T = core::bitflag<IShader::E_SHADER_STAGE>;

    shaderc_shader_kind convert[6];
    convert[core::findLSB<uint32_t>(IShader::ESS_VERTEX)] = shaderc_vertex_shader;
    convert[core::findLSB<uint32_t>(IShader::ESS_TESSELLATION_CONTROL)] = shaderc_tess_control_shader;
    convert[core::findLSB<uint32_t>(IShader::ESS_TESSELLATION_EVALUATION)] = shaderc_tess_evaluation_shader;
    convert[core::findLSB<uint32_t>(IShader::ESS_GEOMETRY)] = shaderc_geometry_shader;
    convert[core::findLSB<uint32_t>(IShader::ESS_FRAGMENT)] = shaderc_fragment_shader;
    convert[core::findLSB<uint32_t>(IShader::ESS_COMPUTE)] = shaderc_compute_shader;

    return convert[core::findLSB<uint32_t>(_ss)];
}

}}

#endif


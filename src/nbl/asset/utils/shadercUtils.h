// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_SHADERC_UTILS_H_INCLUDED_
#define _NBL_ASSET_SHADERC_UTILS_H_INCLUDED_

//! This file is not supposed to be included in user-accesible header files

#include <shaderc/shaderc.hpp>
#include "nbl/asset/IShader.h"

namespace nbl
{
namespace asset
{

inline shaderc_shader_kind ESStoShadercEnum(IShader::E_SHADER_STAGE _ss)
{
    using T = core::bitflag<IShader::E_SHADER_STAGE>;

    shaderc_shader_kind convert[6];
    convert[hlsl::findLSB<uint32_t>(IShader::ESS_VERTEX)] = shaderc_vertex_shader;
    convert[hlsl::findLSB<uint32_t>(IShader::ESS_TESSELLATION_CONTROL)] = shaderc_tess_control_shader;
    convert[hlsl::findLSB<uint32_t>(IShader::ESS_TESSELLATION_EVALUATION)] = shaderc_tess_evaluation_shader;
    convert[hlsl::findLSB<uint32_t>(IShader::ESS_GEOMETRY)] = shaderc_geometry_shader;
    convert[hlsl::findLSB<uint32_t>(IShader::ESS_FRAGMENT)] = shaderc_fragment_shader;
    convert[hlsl::findLSB<uint32_t>(IShader::ESS_COMPUTE)] = shaderc_compute_shader;

    return convert[hlsl::findLSB<uint32_t>(_ss)];
}

}}

#endif


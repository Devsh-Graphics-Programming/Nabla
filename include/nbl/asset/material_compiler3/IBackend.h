// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_I_BACKEND_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_I_BACKEND_H_INCLUDED_

#include "nbl/asset/material_compiler3/CTrueIR.h"

namespace nbl::asset::material_compiler3
{

class IBackend
{
public:
    struct IResult// : public core::IReferenceCounted
    {
        
    };

    IResult compile(const CTrueIR*, const std::span<const CTrueIR::SMaterialHandle>);
};

}

#endif

#ifndef __NBL_MATERIAL_COMPILER_I_FRONTEND_H_INCLUDED__
#define __NBL_MATERIAL_COMPILER_I_FRONTEND_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/asset/material_compiler/IR.h>

namespace nbl::asset::material_compiler
{

class IFrontend : public core::IReferenceCounted
{
    public:
        virtual core::smart_refctd_ptr<IR> compileToIR();
};

}

#endif
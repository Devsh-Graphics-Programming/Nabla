#ifndef __IRR_I_GPU_SHADER_H_INCLUDED__
#define __IRR_I_GPU_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ISPIR_VProgram.h"

namespace irr
{
namespace video
{

//! GPU Version of Unspecialized Shader
/*
	@see IReferenceCounted
*/

class IGPUShader : public asset::IShader
{
    protected:
        virtual ~IGPUShader() = default;
};

}
}

#endif//__IRR_I_GPU_SHADER_H_INCLUDED__

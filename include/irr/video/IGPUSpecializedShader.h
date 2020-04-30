#ifndef __IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__
#define __IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/ISpecializedShader.h"

namespace irr
{
namespace video
{

//! GPU Version of Specialized Shader
/*
	@see ISpecializedShader
*/

class IGPUSpecializedShader : public asset::ISpecializedShader
{
	public:
		IGPUSpecializedShader(asset::ISpecializedShader::E_SHADER_STAGE _stage) : m_stage(_stage) {}

		asset::ISpecializedShader::E_SHADER_STAGE getStage() const { return m_stage; }

	protected:
		virtual ~IGPUSpecializedShader() = default;

		const asset::ISpecializedShader::E_SHADER_STAGE m_stage;
};

}
}

#endif//__IRR_I_GPU_SPECIALIZED_SHADER_H_INCLUDED__


// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CCompilerSet.h"

using namespace nbl;
using namespace nbl::asset;

core::smart_refctd_ptr<ICPUBuffer> CCompilerSet::compileToSPIRV(core::smart_refctd_ptr<asset::ICPUShader> shader, const IShaderCompiler::SOptions& options)
{
	core::smart_refctd_ptr<ICPUBuffer> outSpirvShader;
	if (shader)
	{
		switch (shader->getContentType())
		{
		case IShader::E_CONTENT_TYPE::ECT_HLSL:
		{
			const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
			if (options.getCodeContentType() == IShader::E_CONTENT_TYPE::ECT_HLSL)
			{
				m_HLSLCompiler->compileToSPIRV(code, static_cast<const CHLSLCompiler::SOptions&>(options));
			}
			else
			{
				CHLSLCompiler::SOptions hlslCompilerOptions = {};
				hlslCompilerOptions.setCommonData(options);
				m_HLSLCompiler->compileToSPIRV(code, hlslCompilerOptions);
			}
		}
		break;
		case IShader::E_CONTENT_TYPE::ECT_GLSL:
		{
			const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
			if (options.getCodeContentType() == IShader::E_CONTENT_TYPE::ECT_GLSL)
			{
				m_GLSLCompiler->compileToSPIRV(code, static_cast<const CGLSLCompiler::SOptions&>(options));
			}
			else
			{
				CGLSLCompiler::SOptions glslCompilerOptions = {};
				glslCompilerOptions.setCommonData(options);
				m_GLSLCompiler->compileToSPIRV(code, glslCompilerOptions);
			}
		}
		break;
		case IShader::E_CONTENT_TYPE::ECT_SPIRV:
		{
			outSpirvShader = core::smart_refctd_ptr<ICPUBuffer>(const_cast<ICPUBuffer*>(shader->getContent()));
		}
		break;
		}
	}
	return outSpirvShader;
}
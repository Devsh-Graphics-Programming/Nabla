// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CCompilerSet.h"

using namespace nbl;
using namespace nbl::asset;

core::smart_refctd_ptr<IShader> CCompilerSet::compileToSPIRV(const IShader* shader, const IShaderCompiler::SCompilerOptions& options) const
{
	core::smart_refctd_ptr<IShader> outSpirvShader = nullptr;
	if (shader)
	{
		switch (shader->getContentType())
		{
			case IShader::E_CONTENT_TYPE::ECT_HLSL:
				{
#ifdef _NBL_PLATFORM_WINDOWS_
					const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
					outSpirvShader = m_HLSLCompiler->compileToSPIRV(code, options);
#endif
				}
				break;
			case IShader::E_CONTENT_TYPE::ECT_GLSL:
				{
					const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
					outSpirvShader = m_GLSLCompiler->compileToSPIRV(code, options);
				}
				break;
			case IShader::E_CONTENT_TYPE::ECT_SPIRV:
				{
					outSpirvShader = core::smart_refctd_ptr<IShader>(const_cast<IShader*>(shader));
				}
				break;
		}
	}
	return outSpirvShader;
}

core::smart_refctd_ptr<IShader> CCompilerSet::preprocessShader(const IShader* shader, hlsl::ShaderStage& stage, const IShaderCompiler::SPreprocessorOptions& preprocessOptions) const
{
	if (shader)
	{
		switch (shader->getContentType())
		{
			case IShader::E_CONTENT_TYPE::ECT_HLSL:
				{
#ifdef _NBL_PLATFORM_WINDOWS_
					const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
					auto resolvedCode = m_HLSLCompiler->preprocessShader(code, stage, preprocessOptions);
					return core::make_smart_refctd_ptr<IShader>(resolvedCode.c_str(), IShader::E_CONTENT_TYPE::ECT_HLSL, std::string(shader->getFilepathHint()));
#endif
				}
				break;
			case IShader::E_CONTENT_TYPE::ECT_GLSL:
				{
					const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
					auto resolvedCode = m_GLSLCompiler->preprocessShader(code, stage, preprocessOptions);
					return core::make_smart_refctd_ptr<IShader>(resolvedCode.c_str(), IShader::E_CONTENT_TYPE::ECT_GLSL, std::string(shader->getFilepathHint()));
				}
				break;
			case IShader::E_CONTENT_TYPE::ECT_SPIRV:
				return core::smart_refctd_ptr<IShader>(const_cast<IShader*>(shader));
			default:
				break;
		}
	}
	return nullptr;
}
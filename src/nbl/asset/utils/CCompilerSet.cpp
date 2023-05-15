// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/CCompilerSet.h"

using namespace nbl;
using namespace nbl::asset;

core::smart_refctd_ptr<ICPUShader> CCompilerSet::compileToSPIRV(const ICPUShader* shader, const IShaderCompiler::SCompilerOptions& options) const
{
	core::smart_refctd_ptr<ICPUShader> outSpirvShader = nullptr;
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
			outSpirvShader = core::smart_refctd_ptr<ICPUShader>(const_cast<ICPUShader*>(shader));
		}
		break;
		}
	}
	return outSpirvShader;
}

core::smart_refctd_ptr<ICPUShader> CCompilerSet::preprocessShader(const ICPUShader* shader, const IShaderCompiler::SPreprocessorOptions& preprocessOptions) const
{
	if (shader)
	{
		switch (shader->getContentType())
		{
		case IShader::E_CONTENT_TYPE::ECT_HLSL:
		{
#ifdef _NBL_PLATFORM_WINDOWS_
			const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
			auto stage = shader->getStage();
			auto resolvedCode = m_HLSLCompiler->preprocessShader(code, stage, preprocessOptions);
			return core::make_smart_refctd_ptr<ICPUShader>(resolvedCode.c_str(), stage, IShader::E_CONTENT_TYPE::ECT_HLSL, std::string(shader->getFilepathHint()));
#endif
		}
		break;
		case IShader::E_CONTENT_TYPE::ECT_GLSL:
		{
			const char* code = reinterpret_cast<const char*>(shader->getContent()->getPointer());
			auto stage = shader->getStage();
			auto resolvedCode = m_GLSLCompiler->preprocessShader(code, stage, preprocessOptions);
			return core::make_smart_refctd_ptr<ICPUShader>(resolvedCode.c_str(), stage, IShader::E_CONTENT_TYPE::ECT_GLSL, std::string(shader->getFilepathHint()));
		}
		break;
		case IShader::E_CONTENT_TYPE::ECT_SPIRV:
			return core::smart_refctd_ptr<ICPUShader>(const_cast<ICPUShader*>(shader));
		default:
			break;
		}
	}
	return nullptr;
}
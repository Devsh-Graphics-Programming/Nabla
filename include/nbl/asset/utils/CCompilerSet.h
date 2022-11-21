// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_COMPILER_SET_H_INCLUDED_
#define _NBL_ASSET_C_COMPILER_SET_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "CGLSLCompiler.h"
#include "CHLSLCompiler.h"

namespace nbl::asset
{
	class NBL_API2 CCompilerSet : public core::IReferenceCounted
	{
	public:
		CCompilerSet(core::smart_refctd_ptr<system::ISystem>&& sys)
			: m_HLSLCompiler(core::make_smart_refctd_ptr<CHLSLCompiler>(core::smart_refctd_ptr(sys)))
			, m_GLSLCompiler(core::make_smart_refctd_ptr<CGLSLCompiler>(core::smart_refctd_ptr(sys)))
		{}

		core::smart_refctd_ptr<ICPUBuffer> compileToSPIRV(core::smart_refctd_ptr<asset::ICPUShader> shader, const IShaderCompiler::SOptions& options);

		inline core::smart_refctd_ptr<IShaderCompiler> getShaderCompiler(IShader::E_CONTENT_TYPE contentType) const
		{
			if (contentType == IShader::E_CONTENT_TYPE::ECT_HLSL)
				return m_HLSLCompiler;
			else if (contentType == IShader::E_CONTENT_TYPE::ECT_GLSL)
				return m_GLSLCompiler;
			else
				return nullptr;
		}

	protected:
		core::smart_refctd_ptr<CHLSLCompiler> m_HLSLCompiler = nullptr;
		core::smart_refctd_ptr<CGLSLCompiler> m_GLSLCompiler = nullptr;
	};
}

#endif

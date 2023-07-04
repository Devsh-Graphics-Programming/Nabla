// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_HLSL_COMPILER_H_INCLUDED_
#define _NBL_ASSET_C_HLSL_COMPILER_H_INCLUDED_

#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/IShaderCompiler.h"

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl::asset::hlsl::impl
{
	class DXC;
}

namespace nbl::asset
{

class NBL_API2 CHLSLCompiler final : public IShaderCompiler
{
	public:
		IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_HLSL; };

		CHLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system);
		~CHLSLCompiler();

		struct SOptions : IShaderCompiler::SCompilerOptions
		{
			// TODO: Add extra dxc options
			IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_HLSL; };
		};

		core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const char* code, const IShaderCompiler::SCompilerOptions& options) const override;

		template<typename... Args>
		static core::smart_refctd_ptr<ICPUShader> createOverridenCopy(const ICPUShader* original, const char* fmt, Args... args)
		{
			return IShaderCompiler::createOverridenCopy(original, 0u, fmt, args...);
		}

		// TODO
		//static inline const char* getStorageImageFormatQualifier(const asset::E_FORMAT format)
		//{
		//	return "";
		//}

		std::string preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions) const override;

		void insertIntoStart(std::string& code, std::ostringstream&& ins) const override;
	protected:

		// This can't be a unique_ptr due to it being an undefined type 
		// when Nabla is used as a lib
		nbl::asset::hlsl::impl::DXC* m_dxcCompilerTypes;

		static CHLSLCompiler::SOptions option_cast(const IShaderCompiler::SCompilerOptions& options)
		{
			CHLSLCompiler::SOptions ret = {};
			if (options.getCodeContentType() == IShader::E_CONTENT_TYPE::ECT_GLSL)
				ret = static_cast<const CHLSLCompiler::SOptions&>(options);
			else
				ret.setCommonData(options);
			return ret;
		}
};

}

#endif

#endif
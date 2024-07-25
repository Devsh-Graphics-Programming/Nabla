// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_HLSL_COMPILER_H_INCLUDED_
#define _NBL_ASSET_C_HLSL_COMPILER_H_INCLUDED_

#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/IShaderCompiler.h"



#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl::asset::impl
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
			std::span<const std::string> dxcOptions;
			IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_HLSL; };
		};

		core::smart_refctd_ptr<ICPUShader> compileToSPIRV_impl(const std::string_view code, const IShaderCompiler::SCompilerOptions& options, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies = nullptr) const override;

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

		std::string preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies = nullptr) const override;
		std::string preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, std::vector<std::string>& dxc_compile_flags_override, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies = nullptr) const;
							
		void insertIntoStart(std::string& code, std::ostringstream&& ins) const override;
		constexpr static inline const wchar_t* RequiredArguments[] = {
			L"-spirv",
			L"-Zpr",
			L"-enable-16bit-types",
			L"-fvk-use-scalar-layout",
			L"-Wno-c++11-extensions",
			L"-Wno-c++1z-extensions",
			L"-Wno-c++14-extensions",
			L"-Wno-gnu-static-float-init",
			L"-fspv-target-env=vulkan1.3"
		};
		constexpr static inline uint32_t RequiredArgumentCount = sizeof(RequiredArguments) / sizeof(RequiredArguments[0]);
		
	protected:
		// This can't be a unique_ptr due to it being an undefined type 
		// when Nabla is used as a lib
		nbl::asset::impl::DXC* m_dxcCompilerTypes;

		static CHLSLCompiler::SOptions option_cast(const IShaderCompiler::SCompilerOptions& options)
		{
			CHLSLCompiler::SOptions ret = {};
			if (options.getCodeContentType() == IShader::E_CONTENT_TYPE::ECT_HLSL)
				ret = static_cast<const CHLSLCompiler::SOptions&>(options);
			else
				ret.setCommonData(options);
			return ret;
		}
};

}

#endif

#endif
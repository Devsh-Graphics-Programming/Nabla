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
			std::span<const std::string> dxcOptions; // TODO: span is a VIEW to memory, so to something which we should treat immutable - why not span of string_view then? Since its span we force users to keep those std::strings alive anyway but now we cannnot even make nice constexpr & pass such expression here directly
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

		static inline const char* getStorageImageFormatQualifier(const asset::E_FORMAT format)
		{
			switch (format)
			{
			case asset::EF_R32G32B32A32_SFLOAT:
				return "rgba32f";
			case asset::EF_R16G16B16A16_SFLOAT:
				return "rgba16f";
			case asset::EF_R32G32_SFLOAT:
				return "rg32f";
			case asset::EF_R16G16_SFLOAT:
				return "rg16f";
			case asset::EF_B10G11R11_UFLOAT_PACK32:
				return "r11g11b10f";
			case asset::EF_R32_SFLOAT:
				return "r32f";
			case asset::EF_R16_SFLOAT:
				return "r16f";
			case asset::EF_R16G16B16A16_UNORM:
				return "rgba16";
			case asset::EF_A2B10G10R10_UNORM_PACK32:
				return "rgb10a2";
			case asset::EF_R8G8B8A8_UNORM:
				return "rgba8";
			case asset::EF_R16G16_UNORM:
				return "rg16";
			case asset::EF_R8G8_UNORM:
				return "rg8";
			case asset::EF_R16_UNORM:
				return "r16";
			case asset::EF_R8_UNORM:
				return "r8";
			case asset::EF_R16G16B16A16_SNORM:
				return "rgba16snorm";
			case asset::EF_R8G8B8A8_SNORM:
				return "rgba8snorm";
			case asset::EF_R16G16_SNORM:
				return "rg16snorm";
			case asset::EF_R8G8_SNORM:
				return "rg8snorm";
			case asset::EF_R16_SNORM:
				return "r16snorm";
			case asset::EF_R8_UINT:
				return "r8ui";
			case asset::EF_R16_UINT:
				return "r16ui";
			case asset::EF_R32_UINT:
				return "r32ui";
			case asset::EF_R32G32_UINT:
				return "rg32ui";
			case asset::EF_R32G32B32A32_UINT:
				return "rgba32ui";
			default:
				assert(false);
				return "";
			}
		}

		static constexpr auto getRequiredArguments() //! returns required arguments for the compiler's backend
		{
			return std::span(RequiredArguments);
		}
		
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

	private:
		// we cannot have PUBLIC data symbol in header we do export - endpoint application will fail on linker with delayed DLL loading mechanism (thats why we trick it with private member hidden from the export + provide exported getter)
		// https://learn.microsoft.com/en-us/previous-versions/w59k653y(v=vs.100)?redirectedfrom=MSDN
		constexpr static inline auto RequiredArguments = std::to_array<const wchar_t*> // TODO: and if dxcOptions is span of std::string then why w_chars there? https://en.cppreference.com/w/cpp/string/basic_string
		({ 
			L"-spirv",
			L"-Zpr",
			L"-enable-16bit-types",
			L"-fvk-use-scalar-layout",
			L"-Wno-c++11-extensions",
			L"-Wno-c++1z-extensions",
			L"-Wno-c++14-extensions",
			L"-Wno-gnu-static-float-init",
			L"-fspv-target-env=vulkan1.3",
			L"-HV", L"202x"
		});
};

}

#endif

#endif
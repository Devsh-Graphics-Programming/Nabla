// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_GLSL_COMPILER_H_INCLUDED_
#define _NBL_ASSET_C_GLSL_COMPILER_H_INCLUDED_

#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/IShaderCompiler.h"
#include "nbl/asset/format/EFormat.h"

namespace nbl::asset
{

class NBL_API2 CGLSLCompiler final : public IShaderCompiler
{
	public:		

		IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_GLSL; };

		CGLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system);

		struct SOptions : IShaderCompiler::SCompilerOptions
		{
			IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_GLSL; };
		};

		/**
		If options.stage is ESS_UNKNOWN, then compiler will try to deduce shader stage from #pragma annotation, i.e.:
		#pragma shader_stage(vertex),       or
		#pragma shader_stage(tesscontrol),  or
		#pragma shader_stage(tesseval),     or
		#pragma shader_stage(geometry),     or
		#pragma shader_stage(fragment),     or
		#pragma shader_stage(compute)

		Such annotation should be placed right after #version directive.

		This function does NOT process #include directives! Use preprocessShader() first.

		@params code high level code
		@params options see IShaderCompiler::SCompilerOptions
			- entryPoint Must be "main" since shaderc does not allow other entry points for GLSL. Kept with hope that shaderc will drop that requirement.

		@returns Shader containing SPIR-V bytecode.
		*/

		core::smart_refctd_ptr<ICPUShader> compileToSPIRV_impl(const std::string_view code, const IShaderCompiler::SCompilerOptions& options, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const override;

		/*
		 If original code contains #version specifier,
			then the filled fmt will be placed onto the next line after #version in the output buffer. If not, fmt will be placed into the
			beginning of the output buffer.
		*/
		template<typename... Args>
		static core::smart_refctd_ptr<ICPUShader> createOverridenCopy(const ICPUShader* original, const char* fmt, Args... args)
		{
			uint32_t position = 0u;
			if (original != nullptr)
			{
				auto origCodeBuffer = original->getContent();
				auto origCode = std::string_view(reinterpret_cast<const char*>(origCodeBuffer->getPointer()), origCodeBuffer->getSize());
				auto start = origCode.find("#version");
				auto end = origCode.find("\n", start);
				if (end != std::string_view::npos)
					position = end + 1u;
			}

			return IShaderCompiler::createOverridenCopy(original, position, fmt, args...);
		}

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
				return "r11f_g11f_b10f";
			case asset::EF_R32_SFLOAT:
				return "r32f";
			case asset::EF_R16_SFLOAT:
				return "r16f";
			case asset::EF_R16G16B16A16_UNORM:
				return "rgba16";
			case asset::EF_A2B10G10R10_UNORM_PACK32:
				return "rgb10_a2";
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
				return "rgba16_snorm";
			case asset::EF_R8G8B8A8_SNORM:
				return "rgba8_snorm";
			case asset::EF_R16G16_SNORM:
				return "rg16_snorm";
			case asset::EF_R8G8_SNORM:
				return "rg8_snorm";
			case asset::EF_R16_SNORM:
				return "r16_snorm";
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

		std::string preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies = nullptr) const override;

		static std::string escapeFilename(std::string&& code);

		static void disableAllDirectivesExceptIncludes(std::string& _code);

		static void reenableDirectives(std::string& _code);

		static std::string encloseWithinExtraInclGuards(std::string&& _code, uint32_t _maxInclusions, const char* _identifier);

		static uint32_t encloseWithinExtraInclGuardsLeadingLines(uint32_t _maxInclusions);
	protected:

		void insertIntoStart(std::string& code, std::ostringstream&& ins) const override;

		static CGLSLCompiler::SOptions option_cast(const IShaderCompiler::SCompilerOptions& options)
		{
			CGLSLCompiler::SOptions ret = {};
			if (options.getCodeContentType() == IShader::E_CONTENT_TYPE::ECT_GLSL)
				ret = static_cast<const CGLSLCompiler::SOptions&>(options);
			else
				ret.setCommonData(options);
			return ret;
		}

};

}

#endif

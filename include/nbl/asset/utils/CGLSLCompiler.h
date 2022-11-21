// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_GLSL_COMPILER_H_INCLUDED_
#define _NBL_ASSET_C_GLSL_COMPILER_H_INCLUDED_

#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/IShaderCompiler.h"

namespace nbl::asset
{

class NBL_API CGLSLCompiler final : public IShaderCompiler
{
	public:
		IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_GLSL; };

		CGLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system);

		struct SOptions : IShaderCompiler::SOptions
		{
			virtual IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_GLSL; };
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

		This function does NOT process #include directives! Use resolveIncludeDirectives() first.

		@params code high level code
		@param options
			entryPoint Must be "main" since shaderc does not allow other entry points for GLSL. Kept with hope that shaderc will drop that requirement.
			compilationId String that will be printed along with possible errors as source identifier.
			genDebugInfo Requests compiler to generate debug info (most importantly objects' names).
				The engine, while running on OpenGL, won't be able to set push constants for shaders loaded as SPIR-V without debug info.
			outAssembly Optional parameter; if not nullptr, SPIR-V assembly is saved in there.

		@returns Shader containing SPIR-V bytecode.
		*/
		core::smart_refctd_ptr<ICPUBuffer> compileToSPIRV(const char* code, const CGLSLCompiler::SOptions& options) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVShader(const char* code, const CGLSLCompiler::SOptions& options) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVShader(system::IFile* sourceFile, const CGLSLCompiler::SOptions& options) const;

		// TODO: REMOVE
		core::smart_refctd_ptr<ICPUBuffer> compileSPIRVFromGLSL(
			const char* _glslCode,
			IShader::E_SHADER_STAGE _stage,
			const char* _entryPoint,
			const char* _compilationId,
			bool _genDebugInfo = true,
			std::string* _outAssembly = nullptr,
			system::logger_opt_ptr logger = nullptr,
			const E_SPIRV_VERSION targetSpirvVersion = E_SPIRV_VERSION::ESV_1_6) const;

		// TODO: REMOVE
		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(
			const char* _glslCode,
			IShader::E_SHADER_STAGE _stage,
			const char* _entryPoint,
			const char* _compilationId,
			const ISPIRVOptimizer* _opt = nullptr,
			bool _genDebugInfo = true,
			std::string* _outAssembly = nullptr,
			system::logger_opt_ptr logger = nullptr,
			const E_SPIRV_VERSION targetSpirvVersion = E_SPIRV_VERSION::ESV_1_6) const;

		// TODO: REMOVE
		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(
			system::IFile* _sourcefile,
			IShader::E_SHADER_STAGE _stage,
			const char* _entryPoint,
			const char* _compilationId,
			const ISPIRVOptimizer* _opt = nullptr,
			bool _genDebugInfo = true,
			std::string* _outAssembly = nullptr,
			system::logger_opt_ptr logger = nullptr,
			const E_SPIRV_VERSION targetSpirvVersion = E_SPIRV_VERSION::ESV_1_6) const;

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
};

}

#endif

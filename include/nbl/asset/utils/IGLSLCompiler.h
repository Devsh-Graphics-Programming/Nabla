// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_GLSL_COMPILER_H_INCLUDED__
#define __NBL_ASSET_I_GLSL_COMPILER_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/system/declarations.h"

#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

#include "nbl/asset/ICPUSpecializedShader.h"
#include "nbl/asset/utils/IIncludeHandler.h"

#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/IShaderCompiler.h"

namespace nbl::asset
{

//! Will be derivative of IShaderGenerator, but we have to establish interface first
class NBL_API IGLSLCompiler final : public IShaderCompiler
{
	public:
		enum E_SPIRV_VERSION
		{
			ESV_1_0 = 0x010000u,
			ESV_1_1 = 0x010100u,
			ESV_1_2 = 0x010200u,
			ESV_1_3 = 0x010300u,
			ESV_1_4 = 0x010400u,
			ESV_1_5 = 0x010500u,
			ESV_1_6 = 0x010600u,
			ESV_COUNT = 0x7FFFFFFFu
		};

		// TODO: make a new enum on compile type or something, doesn't make sense to have to compile ECT_SPIRV
		// And it's not "contentType" in this context. It's codeType (?)
		IShader::E_CONTENT_TYPE getContentType() const override { return IShader::ECT_GLSL;  };

		IGLSLCompiler(system::ISystem* _s);

		IIncludeHandler* getIncludeHandler() { return m_inclHandler.get(); }
		const IIncludeHandler* getIncludeHandler() const { return m_inclHandler.get(); }

		core::smart_refctd_ptr<ICPUBuffer> compileSPIRVFromGLSL(
			const char* _glslCode,
			IShader::E_SHADER_STAGE _stage,
			const char* _entryPoint,
			const char* _compilationId,
			bool _genDebugInfo = true,
			std::string* _outAssembly = nullptr,
			system::logger_opt_ptr logger = nullptr,
			const E_SPIRV_VERSION targetSpirvVersion = ESV_1_6) const;

		/**
		If _stage is ESS_UNKNOWN, then compiler will try to deduce shader stage from #pragma annotation, i.e.:
		#pragma shader_stage(vertex),       or
		#pragma shader_stage(tesscontrol),  or
		#pragma shader_stage(tesseval),     or
		#pragma shader_stage(geometry),     or
		#pragma shader_stage(fragment),     or
		#pragma shader_stage(compute)

		Such annotation should be placed right after #version directive.

		This function does NOT process #include directives! Use resolveIncludeDirectives() first.

		@param _entryPoint Must be "main" since shaderc does not allow other entry points for GLSL. Kept with hope that shaderc will drop that requirement.
		@param _compilationId String that will be printed along with possible errors as source identifier.
		@param _genDebugInfo Requests compiler to generate debug info (most importantly objects' names).
			The engine, while running on OpenGL, won't be able to set push constants for shaders loaded as SPIR-V without debug info.
		@param _outAssembly Optional parameter; if not nullptr, SPIR-V assembly is saved in there.

		@returns Shader containing SPIR-V bytecode.
		*/
		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(
			const char* _glslCode,
			IShader::E_SHADER_STAGE _stage,
			const char* _entryPoint,
			const char* _compilationId,
			const ISPIRVOptimizer* _opt = nullptr,
			bool _genDebugInfo = true,
			std::string* _outAssembly = nullptr,
			system::logger_opt_ptr logger = nullptr,
			const E_SPIRV_VERSION targetSpirvVersion = ESV_1_6) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(
			system::IFile* _sourcefile,
			IShader::E_SHADER_STAGE _stage,
			const char* _entryPoint,
			const char* _compilationId,
			const ISPIRVOptimizer* _opt = nullptr,
			bool _genDebugInfo = true,
			std::string* _outAssembly = nullptr,
			system::logger_opt_ptr logger = nullptr,
			const E_SPIRV_VERSION targetSpirvVersion = ESV_1_6) const;

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

	private:
		core::smart_refctd_ptr<IIncludeHandler> m_inclHandler;
};

}

#endif

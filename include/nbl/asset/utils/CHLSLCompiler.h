// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_HLSL_COMPILER_H_INCLUDED_
#define _NBL_ASSET_C_HLSL_COMPILER_H_INCLUDED_

#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/IShaderCompiler.h"

namespace nbl::asset
{

class NBL_API CHLSLCompiler final : public IShaderCompiler
{
	public:
		IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_HLSL; };

		CHLSLCompiler(core::smart_refctd_ptr<system::ISystem>&& system);

		struct SOptions : IShaderCompiler::SOptions
		{
			// TODO: Add extra dxc options
			virtual IShader::E_CONTENT_TYPE getCodeContentType() const override { return IShader::E_CONTENT_TYPE::ECT_HLSL; };
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
			entryPoint Must be "main" since shaderc does not allow other entry points for HLSL. Kept with hope that shaderc will drop that requirement.
			compilationId String that will be printed along with possible errors as source identifier.
			genDebugInfo Requests compiler to generate debug info (most importantly objects' names).
				The engine, while running on OpenGL, won't be able to set push constants for shaders loaded as SPIR-V without debug info.
			outAssembly Optional parameter; if not nullptr, SPIR-V assembly is saved in there.

		@returns Shader containing SPIR-V bytecode.
		*/
		core::smart_refctd_ptr<ICPUBuffer> compileToSPIRV(const char* code, const CHLSLCompiler::SOptions& options) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVShader(const char* code, const CHLSLCompiler::SOptions& options) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVShader(system::IFile* sourceFile, const CHLSLCompiler::SOptions& options) const;

		/*
		 If original code contains #version specifier,
			then the filled fmt will be placed onto the next line after #version in the output buffer. If not, fmt will be placed into the
			beginning of the output buffer.
		*/
		template<typename... Args>
		static core::smart_refctd_ptr<ICPUShader> createOverridenCopy(const ICPUShader* original, const char* fmt, Args... args)
		{
			return IShaderCompiler::createOverridenCopy(original, 0u, fmt, args...);
		}

		static inline const char* getStorageImageFormatQualifier(const asset::E_FORMAT format)
		{
			// TODO
			return "";
		}
};

}

#endif

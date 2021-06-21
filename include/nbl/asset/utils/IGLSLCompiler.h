// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_GLSL_COMPILER_H_INCLUDED__
#define __NBL_ASSET_I_GLSL_COMPILER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/system/system.h"

#include "nbl/system/IFile.h"
#include "IFileSystem.h"

#include "nbl/asset/ICPUSpecializedShader.h"
#include "nbl/asset/utils/IIncludeHandler.h"

#include "nbl/asset/utils/ISPIRVOptimizer.h"

namespace nbl
{

namespace asset
{

//! Will be derivative of IShaderGenerator, but we have to establish interface first
class IGLSLCompiler final : public core::IReferenceCounted
{
		core::smart_refctd_ptr<IIncludeHandler> m_inclHandler;
		const io::IFileSystem* m_fs;

	public:
		IGLSLCompiler(io::IFileSystem* _fs);

		IIncludeHandler* getIncludeHandler() { return m_inclHandler.get(); }
		const IIncludeHandler* getIncludeHandler() const { return m_inclHandler.get(); }

		core::smart_refctd_ptr<ICPUBuffer> compileSPIRVFromGLSL(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, bool _genDebugInfo = true, std::string* _outAssembly = nullptr) const;

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
		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(const char* _glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, const ISPIRVOptimizer* _opt = nullptr, bool _genDebugInfo = true, std::string* _outAssembly = nullptr) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(io::IReadFile* _sourcefile, ISpecializedShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, const ISPIRVOptimizer* _opt = nullptr, bool _genDebugInfo = true, std::string* _outAssembly = nullptr) const;

		/**
		Resolves ALL #include directives regardless of any other preprocessor directive.
		This is done in order to support `#include` AND simultaneulsy be able to store (serialize) such ICPUShader (mostly GLSL source) into ONE file which, upon loading, will compile on every hardware/driver predicted by shader's author.

		Internally function "disables" all preprocessor directives (so that they're not processed by preprocessor) except `#include` (and also `#version` and `#pragma shader_stage`).
		Note that among the directives there may be include guards. Because of that, _maxSelfInclusionCnt parameter is provided.

		@param _maxSelfInclusionCnt Max self-inclusion count of possible file being #include'd. If no self-inclusions are allowed, should be set to 0.

		@param _originFilepath Path to not necesarilly existing file whose directory will be base for relative (""-type) top-level #include's resolution.
			If _originFilepath is non-path-like string (e.g. "whatever" - no slashes), the base directory is assumed to be "." (working directory of your executable). It's important for it to be unique.

		@returns Shader containing logically same GLSL code as input but with #include directives resolved.
		*/
		core::smart_refctd_ptr<ICPUShader> resolveIncludeDirectives(std::string&& glslCode, ISpecializedShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt = 4u) const;

		core::smart_refctd_ptr<ICPUShader> resolveIncludeDirectives(io::IReadFile* _sourcefile, ISpecializedShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt = 4u) const;
};

}
}

#endif

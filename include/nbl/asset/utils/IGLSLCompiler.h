// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
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

namespace nbl::asset
{

//! Will be derivative of IShaderGenerator, but we have to establish interface first
class IGLSLCompiler final : public core::IReferenceCounted
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
			ESV_COUNT = 0xFFFFFFFFu
		};

		IGLSLCompiler(system::ISystem* _s, const E_SPIRV_VERSION targetSpirvVersion = ESV_1_3);

		IIncludeHandler* getIncludeHandler() { return m_inclHandler.get(); }
		const IIncludeHandler* getIncludeHandler() const { return m_inclHandler.get(); }

		core::smart_refctd_ptr<ICPUBuffer> compileSPIRVFromGLSL(const char* _glslCode, IShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, bool _genDebugInfo = true, std::string* _outAssembly = nullptr, system::logger_opt_ptr logger = nullptr) const;

		inline E_SPIRV_VERSION getTargetSpirvVersion() const
		{
			return m_targetSpirvVersion;
		}

		inline void setTargetSpirvVersion(const E_SPIRV_VERSION targetSpirvVersion)
		{
			m_targetSpirvVersion = targetSpirvVersion;
		}

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
		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(const char* _glslCode, IShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, const ISPIRVOptimizer* _opt = nullptr, bool _genDebugInfo = true, std::string* _outAssembly = nullptr, system::logger_opt_ptr logger = nullptr) const;

		core::smart_refctd_ptr<ICPUShader> createSPIRVFromGLSL(system::IFile* _sourcefile, IShader::E_SHADER_STAGE _stage, const char* _entryPoint, const char* _compilationId, const ISPIRVOptimizer* _opt = nullptr, bool _genDebugInfo = true, std::string* _outAssembly = nullptr, system::logger_opt_ptr logger = nullptr) const;

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
		core::smart_refctd_ptr<ICPUShader> resolveIncludeDirectives(std::string&& glslCode, IShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt = 4u, system::logger_opt_ptr logger = nullptr) const;

		core::smart_refctd_ptr<ICPUShader> resolveIncludeDirectives(system::IFile* _sourcefile, IShader::E_SHADER_STAGE _stage, const char* _originFilepath, uint32_t _maxSelfInclusionCnt = 4u, system::logger_opt_ptr logger = nullptr) const;
		
		/*
			Creates a formatted copy of the original

			@param original An original glsl shader (must contain glsl and not be a dummy object of be a nullptr).
			@param fmt A string with c-like format, which will be filled with data from ...args
			@param ...args Data to fill fmt with
			@returns shader containing fmt filled with ...args, placed before the original code.

			If original == nullptr, the output buffer will only contain the data from fmt. If original code contains #version specifier,
			then the filled fmt will be placed onto the next line after #version in the output buffer. If not, fmt will be placed into the
			beginning of the output buffer.
		*/
		template<typename... Args>
		static core::smart_refctd_ptr<ICPUShader> createOverridenCopy(const ICPUShader* original, const char* fmt, Args... args)
		{
			assert(original == nullptr || (!original->isADummyObjectForCache() && original->containsGLSL()));

			constexpr auto getMaxSize = [](auto num) -> size_t
			{
				using in_type_t = decltype(num);
				static_assert(std::is_fundamental_v<in_type_t> || std::is_same_v<in_type_t,const char*>);
				if constexpr (std::is_floating_point_v<in_type_t>)
				{
					return std::numeric_limits<decltype(num)>::max_digits10; // there is probably a better way to cope with scientific representation
				}
				else if constexpr (std::is_integral_v<in_type_t>)
				{
					return std::to_string(num).length();
				}
				else
				{
					return strlen(num);
				}
			};
			constexpr size_t templateArgsCount = sizeof...(Args);
			size_t origLen = original ? original->conservativeSizeEstimate():0u;
			size_t formatArgsCharSize = (getMaxSize(args) + ...);
			size_t formatSize = strlen(fmt);
			// 2 is an average size of a format (% and a letter) in chars. 
			// Assuming the format contains only one letter, but if it's 2, the outSize is gonna be a touch bigger.
			size_t outSize = origLen + formatArgsCharSize + formatSize - 2 * templateArgsCount;

			nbl::core::smart_refctd_ptr<ICPUBuffer> outBuffer = nbl::core::make_smart_refctd_ptr<ICPUBuffer>(outSize);

			size_t versionDirectiveLength = 0;

			std::string_view origCode;
			auto outCode = reinterpret_cast<char*>(outBuffer->getPointer());
			if (original!=nullptr)
			{
				origCode = std::string_view(reinterpret_cast<const char*>(original->getSPVorGLSL()->getPointer()),origLen);
				auto start = origCode.find("#version");
				auto end = origCode.find("\n",start);
				if (end!=std::string_view::npos)
					versionDirectiveLength = end+1u;
			}

			std::copy_n(origCode.data(),versionDirectiveLength,outCode);
			outCode += versionDirectiveLength;

			outCode += sprintf(outCode,fmt,std::forward<Args>(args)...);

			auto epilogueLen = origLen-versionDirectiveLength;
			std::copy_n(origCode.data()+versionDirectiveLength,epilogueLen,outCode);
			outCode += epilogueLen;
			*outCode = 0; // terminating char

			return nbl::core::make_smart_refctd_ptr<ICPUShader>(std::move(outBuffer), IShader::buffer_contains_glsl_t{}, original->getStage(), std::string(original->getFilepathHint()));
		}

	private:
		core::smart_refctd_ptr<IIncludeHandler> m_inclHandler;
		system::ISystem* m_system;
		E_SPIRV_VERSION m_targetSpirvVersion;
};

}

#endif

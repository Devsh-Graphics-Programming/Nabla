// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_SHADER_COMPILER_H_INCLUDED_
#define _NBL_ASSET_I_SHADER_COMPILER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/system/declarations.h"

#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

#include "nbl/asset/ICPUSpecializedShader.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

namespace nbl::asset
{

class NBL_API IShaderCompiler : public core::IReferenceCounted
{
	public:

		IShaderCompiler(core::smart_refctd_ptr<system::ISystem>&& system);

		/**
		Resolves ALL #include directives regardless of any other preprocessor directive.
		This is done in order to support `#include` AND simultaneulsy be able to store (serialize) such ICPUShader (mostly High Level source) into ONE file which, upon loading, will compile on every hardware/driver predicted by shader's author.

		Internally function "disables" all preprocessor directives (so that they're not processed by preprocessor) except `#include` (and also `#version` and `#pragma shader_stage`).
		Note that among the directives there may be include guards. Because of that, _maxSelfInclusionCnt parameter is provided.

		@param _maxSelfInclusionCnt Max self-inclusion count of possible file being #include'd. If no self-inclusions are allowed, should be set to 0.

		@param _originFilepath Path to not necesarilly existing file whose directory will be base for relative (""-type) top-level #include's resolution.
			If _originFilepath is non-path-like string (e.g. "whatever" - no slashes), the base directory is assumed to be "." (working directory of your executable). It's important for it to be unique.

		@returns Shader containing logically same High Level code as input but with #include directives resolved.
		*/
		core::smart_refctd_ptr<ICPUShader> resolveIncludeDirectives(
			std::string&& _code,
			IShader::E_SHADER_STAGE _stage,
			const char* _originFilepath,
			uint32_t _maxSelfInclusionCnt = 4u,
			system::logger_opt_ptr logger = nullptr) const;

		core::smart_refctd_ptr<ICPUShader> resolveIncludeDirectives(
			system::IFile* _sourcefile,
			IShader::E_SHADER_STAGE _stage,
			const char* _originFilepath,
			uint32_t _maxSelfInclusionCnt = 4u,
			system::logger_opt_ptr logger = nullptr) const;
		
		/*
			Creates a formatted copy of the original

			@param original An original High Level shader (must contain high level language code and must not be a nullptr).
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
			assert(original == nullptr || (!original->isADummyObjectForCache() && original->isContentHighLevelLanguage()));

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
			size_t origLen = original ? original->getContent()->getSize():0u;
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
				origCode = std::string_view(reinterpret_cast<const char*>(original->getContent()->getPointer()),origLen);
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

			return nbl::core::make_smart_refctd_ptr<ICPUShader>(std::move(outBuffer), original->getStage(), original->getContentType(), std::string(original->getFilepathHint()));
		}

		virtual IShader::E_CONTENT_TYPE getCodeContentType() const = 0;

		class NBL_API2 IIncludeLoader : public core::IReferenceCounted
		{
		public:
			virtual std::string getInclude(const system::path& searchPath, const std::string& includeName) const = 0;
		};

		class NBL_API2 IIncludeGenerator : public core::IReferenceCounted
		{
		public:
			// ! if includeName doesn't begin with prefix from `getPrefix` this function will return an empty string
			virtual std::string getInclude(const std::string& includeName) const;

			virtual std::string_view getPrefix() const = 0;

		protected:

			using HandleFunc_t = std::function<std::string(const std::string&)>;
			virtual core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const = 0;

			// ! Parses arguments from include path
			// ! template is path/to/shader.hlsl/arg0/arg1/...
			static core::vector<std::string> parseArgumentsFromPath(const std::string& _path);
		};

		class NBL_API2 CFileSystemIncludeLoader : public IIncludeLoader
		{
		public:
			CFileSystemIncludeLoader(core::smart_refctd_ptr<system::ISystem>&& system);

			std::string getInclude(const system::path& searchPath, const std::string& includeName) const override;

		protected:
			core::smart_refctd_ptr<system::ISystem> m_system;
		};

		class NBL_API2 CIncludeFinder : public core::IReferenceCounted
		{
		public:
			CIncludeFinder(core::smart_refctd_ptr<system::ISystem>&& system);

			// ! includes within <>
			// @param requestingSourceDir: the directory where the incude was requested
			// @param includeName: the string within <> of the include preprocessing directive
			std::string getIncludeStandard(const system::path& requestingSourceDir, const std::string& includeName) const;

			// ! includes within ""
			// @param requestingSourceDir: the directory where the incude was requested
			// @param includeName: the string within "" of the include preprocessing directive
			std::string getIncludeRelative(const system::path& requestingSourceDir, const std::string& includeName) const;

			inline core::smart_refctd_ptr<CFileSystemIncludeLoader> getDefaultFileSystemLoader() const { return m_defaultFileSystemLoader; }

			void addSearchPath(const std::string& searchPath, const core::smart_refctd_ptr<IIncludeLoader>& loader);

			void addGenerator(const core::smart_refctd_ptr<IIncludeGenerator>& generator);

		protected:

			std::string trySearchPaths(const std::string& includeName) const;

			std::string tryIncludeGenerators(const std::string& includeName) const;

			struct LoaderSearchPath
			{
				core::smart_refctd_ptr<IIncludeLoader> loader = nullptr;
				std::string searchPath = {};
			};

			std::vector<LoaderSearchPath> m_loaders;
			std::vector<core::smart_refctd_ptr<IIncludeGenerator>> m_generators;
			core::smart_refctd_ptr<CFileSystemIncludeLoader> m_defaultFileSystemLoader;
		};

		CIncludeFinder* getIncludeFinder() { return m_inclFinder.get(); }

		const CIncludeFinder* getIncludeFinder() const { return m_inclFinder.get(); }

	private:
		core::smart_refctd_ptr<system::ISystem> m_system;
		core::smart_refctd_ptr<CIncludeFinder> m_inclFinder;
};

}

#endif

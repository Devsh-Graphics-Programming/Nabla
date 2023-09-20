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

class NBL_API2 IShaderCompiler : public core::IReferenceCounted
{
	public:
		//string to be replaced with all "#" except those in "#include"
		static constexpr const char* PREPROC_DIRECTIVE_DISABLER = "_this_is_a_hash_";
		static constexpr const char* PREPROC_DIRECTIVE_ENABLER = PREPROC_DIRECTIVE_DISABLER;

		class NBL_API2 IIncludeLoader : public core::IReferenceCounted
		{
		public:
			virtual std::optional<std::string> getInclude(const system::path& searchPath, const std::string& includeName) const = 0;
		};

		class NBL_API2 IIncludeGenerator : public core::IReferenceCounted
		{
		public:
			// ! if includeName doesn't begin with prefix from `getPrefix` this function will return an empty string
			virtual std::optional<std::string> getInclude(const std::string& includeName) const;

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

			std::optional<std::string> getInclude(const system::path& searchPath, const std::string& includeName) const override;

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
			std::optional<std::string> getIncludeStandard(const system::path& requestingSourceDir, const std::string& includeName) const;

			// ! includes within ""
			// @param requestingSourceDir: the directory where the incude was requested
			// @param includeName: the string within "" of the include preprocessing directive
			std::optional<std::string> getIncludeRelative(const system::path& requestingSourceDir, const std::string& includeName) const;

			inline core::smart_refctd_ptr<CFileSystemIncludeLoader> getDefaultFileSystemLoader() const { return m_defaultFileSystemLoader; }

			void addSearchPath(const std::string& searchPath, const core::smart_refctd_ptr<IIncludeLoader>& loader);

			void addGenerator(const core::smart_refctd_ptr<IIncludeGenerator>& generator);

		protected:

			std::optional<std::string> trySearchPaths(const std::string& includeName) const;

			std::optional<std::string> tryIncludeGenerators(const std::string& includeName) const;

			struct LoaderSearchPath
			{
				core::smart_refctd_ptr<IIncludeLoader> loader = nullptr;
				std::string searchPath = {};
			};

			std::vector<LoaderSearchPath> m_loaders;
			std::vector<core::smart_refctd_ptr<IIncludeGenerator>> m_generators;
			core::smart_refctd_ptr<CFileSystemIncludeLoader> m_defaultFileSystemLoader;
		};

		enum class E_SPIRV_VERSION : uint32_t
		{
			ESV_1_0 = 0x010000u,
			ESV_1_1 = 0x010100u,
			ESV_1_2 = 0x010200u,
			ESV_1_3 = 0x010300u,
			ESV_1_4 = 0x010400u,
			ESV_1_5 = 0x010500u,
			ESV_1_6 = 0x010600u,
		};

		IShaderCompiler(core::smart_refctd_ptr<system::ISystem>&& system);

		struct SPreprocessorOptions
		{
			std::string_view sourceIdentifier = "";
			system::logger_opt_ptr logger = nullptr;
			const CIncludeFinder* includeFinder = nullptr;
			uint32_t maxSelfInclusionCount = 4u;
			core::SRange<const char* const> extraDefines = {nullptr, nullptr};
		};

		// https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#debugging
		enum class E_DEBUG_INFO_FLAGS : uint8_t
		{
			EDIF_NONE       = 0x00,
			EDIF_FILE_BIT   = 0x01,       //  for emitting full path of the main source file
			EDIF_SOURCE_BIT = 0x02,       //  for emitting preprocessed source code (turns on EDIF_FILE_BIT implicitly)
			EDIF_LINE_BIT   = 0x04,       //  for emitting line information (turns on EDIF_SOURCE_BIT implicitly)
			EDIF_TOOL_BIT   = 0x08,       //  for emitting Compiler Git commit hash and command-line options
			EDIF_NON_SEMANTIC_BIT = 0x10, // NonSemantic.Shader.DebugInfo.100 extended instructions, this option overrules the options above
		};

		/*
			@stage shaderStage
			@targetSpirvVersion spirv version
			@entryPoint entryPoint
			@outAssembly Optional parameter; if not nullptr, SPIR-V assembly is saved in there.
			@spirvOptimizer Optional parameter;
			@debugInfoFlags See E_DEBUG_INFO_FLAGS enum for more information on possible values
				Anything non-vulkan, basically you can't recover the names of original variables with CSPIRVIntrospector without debug info
				By variables we mean names of PC/SSBO/UBO blocks, as they're essentially instantiations of structs with custom packing.
			@preprocessorOptions
				@sourceIdentifier String that will be printed along with possible errors as source identifier, and used as include's requestingSrc
				@logger Optional parameter; used for logging errors/info
				@includeFinder Optional parameter; if not nullptr, it will resolve the includes in the code
				@maxSelfInclusionCount used only when includeFinder is not nullptr
				@extraDefines adds extra defines to the shader before compilation
		*/
		struct SCompilerOptions
		{
			IShader::E_SHADER_STAGE stage = IShader::E_SHADER_STAGE::ESS_UNKNOWN;
			E_SPIRV_VERSION targetSpirvVersion = E_SPIRV_VERSION::ESV_1_6;
			const ISPIRVOptimizer* spirvOptimizer = nullptr;
			core::bitflag<E_DEBUG_INFO_FLAGS> debugInfoFlags = core::bitflag<E_DEBUG_INFO_FLAGS>(E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT) | E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT;
			SPreprocessorOptions preprocessorOptions = {};

			void setCommonData(const SCompilerOptions& opt)
			{
				(*this) = opt;
			}

			virtual IShader::E_CONTENT_TYPE getCodeContentType() const { return IShader::E_CONTENT_TYPE::ECT_UNKNOWN; };
		};


		virtual core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const char* code, const SCompilerOptions& options) const = 0;

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(system::IFile* sourceFile, const SCompilerOptions& options) const
		{
			size_t fileSize = sourceFile->getSize();
			std::string code(fileSize, '\0');

			system::IFile::success_t success;
			sourceFile->read(success, code.data(), 0, fileSize);
			if (success)
				return compileToSPIRV(code.c_str(), options);
			else
				return nullptr;
		}

		/**
		Resolves ALL #include directives regardless of any other preprocessor directive.
		This is done in order to support `#include` AND simultaneulsy be able to store (serialize) such ICPUShader (mostly High Level source) into ONE file which, upon loading, will compile on every hardware/driver predicted by shader's author.

		Internally function "disables" all preprocessor directives (so that they're not processed by preprocessor) except `#include` (and also `#version` and `#pragma shader_stage`).
		Note that among the directives there may be include guards. Because of that, maxSelfInclusionCount parameter is provided.
		
		@param preprocessOptions
		@maxSelfInclusionCount Max self-inclusion count of possible file being #include'd. If no self-inclusions are allowed, should be set to 0.
		@sourceIdentifier Path to not necesarilly existing file whose directory will be base for relative (""-type) top-level #include's resolution.
			If sourceIdentifier is non-path-like string (e.g. "whatever" - no slashes), the base directory is assumed to be "." (working directory of your executable). It's important for it to be unique.

		@returns Shader containing logically same High Level code as input but with #include directives resolved.
		*/
		virtual std::string preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions) const = 0;

		std::string preprocessShader(system::IFile* sourcefile, IShader::E_SHADER_STAGE stage, const SPreprocessorOptions& preprocessOptions) const;
		
		/*
			Creates a formatted copy of the original

			@param original An original High Level shader (must contain high level language code and must not be a nullptr).
			@param position if original != nullptr, is the position in the code to insert the formatted string to the original shader code
			@param fmt A string with c-like format, which will be filled with data from ...args
			@param ...args Data to fill fmt with
			@returns shader containing fmt filled with ...args, placed before the original code.

			If original == nullptr, the output buffer will only contain the data from fmt.
		*/
		template<typename... Args>
		static core::smart_refctd_ptr<ICPUShader> createOverridenCopy(const ICPUShader* original, uint32_t position, const char* fmt, Args... args)
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

			auto origCode = std::string_view(reinterpret_cast<const char*>(original->getContent()->getPointer()), origLen);
			auto outCode = reinterpret_cast<char*>(outBuffer->getPointer());

			if (position < origLen)
			{
				// Copy whatever comes before position
				std::copy_n(origCode.data(), position, outCode);
				outCode += position;

				// Copy formatted string
				outCode += sprintf(outCode,fmt,std::forward<Args>(args)...);

				// Copy the rest of the original code
				auto epilogueLen = origLen - position;
				std::copy_n(origCode.data() + position, epilogueLen, outCode);
				outCode += epilogueLen;

				// terminating char
				*outCode = 0;

				return nbl::core::make_smart_refctd_ptr<ICPUShader>(std::move(outBuffer), original->getStage(), original->getContentType(), std::string(original->getFilepathHint()));
			}
			else
			{
				// Position isn't valid.
				assert(false);
				return nullptr;
			}
		}

		static std::string escapeFilename(std::string&& code);

		static void disableAllDirectivesExceptIncludes(std::string& _code);

		static void reenableDirectives(std::string& _code);

		static std::string encloseWithinExtraInclGuards(std::string&& _code, uint32_t _maxInclusions, const char* _identifier);

		static uint32_t encloseWithinExtraInclGuardsLeadingLines(uint32_t _maxInclusions);

		virtual IShader::E_CONTENT_TYPE getCodeContentType() const = 0;

		CIncludeFinder* getDefaultIncludeFinder() { return m_defaultIncludeFinder.get(); }

		const CIncludeFinder* getDefaultIncludeFinder() const { return m_defaultIncludeFinder.get(); }
	protected:

		virtual void insertIntoStart(std::string& code, std::ostringstream&& ins) const = 0;

		void insertExtraDefines(std::string& code, const core::SRange<const char* const>& defines) const;

		core::smart_refctd_ptr<system::ISystem> m_system;
	private:
		core::smart_refctd_ptr<CIncludeFinder> m_defaultIncludeFinder;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IShaderCompiler::E_DEBUG_INFO_FLAGS)

}

#endif

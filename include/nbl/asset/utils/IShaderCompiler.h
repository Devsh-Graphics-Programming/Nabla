// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_SHADER_COMPILER_H_INCLUDED_
#define _NBL_ASSET_I_SHADER_COMPILER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/system/declarations.h"

#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

#include "nbl/asset/ICPUShader.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include <json_struct/include/json_struct/json_struct.h>

#include "nbl/core/xxHash256.h"

namespace nbl::asset
{

class NBL_API2 IShaderCompiler : public core::IReferenceCounted
{
	public:
		//string to be replaced with all "#" except those in "#include"
		static constexpr const char* PREPROC_DIRECTIVE_DISABLER = "_this_is_a_hash_"; // TODO: remove/move to GLSL Compiler
		static constexpr const char* PREPROC_DIRECTIVE_ENABLER = PREPROC_DIRECTIVE_DISABLER; // TODO: remove/move to GLSL Compiler

		class NBL_API2 IIncludeLoader : public core::IReferenceCounted
		{
			public:
				struct found_t
				{
					system::path absolutePath = {};
					std::string contents = {};
					std::array<uint64_t, 4> hash = {}; // TODO: implement this! especially using the `IFile::getPrecomputedHash()`

					explicit inline operator bool() const {return !absolutePath.empty();}
				};
				virtual found_t getInclude(const system::path& searchPath, const std::string& includeName) const = 0;
		};

		class NBL_API2 IIncludeGenerator : public core::IReferenceCounted
		{
			public:
				// ! if includeName doesn't begin with prefix from `getPrefix` this function will return an empty string
				virtual IIncludeLoader::found_t getInclude(const std::string& includeName) const;

				virtual std::string_view getPrefix() const = 0;

			protected:

				using HandleFunc_t = std::function<std::string(const std::string&)>;
				virtual core::vector<std::pair<std::regex,HandleFunc_t>> getBuiltinNamesToFunctionMapping() const = 0;

				// ! Parses arguments from include path
				// ! template is path/to/shader.hlsl/arg0/arg1/...
				static core::vector<std::string> parseArgumentsFromPath(const std::string& _path);
		};

		class NBL_API2 CFileSystemIncludeLoader : public IIncludeLoader
		{
			public:
				CFileSystemIncludeLoader(core::smart_refctd_ptr<system::ISystem>&& system);

				IIncludeLoader::found_t getInclude(const system::path& searchPath, const std::string& includeName) const override;

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
				IIncludeLoader::found_t getIncludeStandard(const system::path& requestingSourceDir, const std::string& includeName) const;

				// ! includes within ""
				// @param requestingSourceDir: the directory where the incude was requested
				// @param includeName: the string within "" of the include preprocessing directive
				IIncludeLoader::found_t getIncludeRelative(const system::path& requestingSourceDir, const std::string& includeName) const;

				inline core::smart_refctd_ptr<CFileSystemIncludeLoader> getDefaultFileSystemLoader() const { return m_defaultFileSystemLoader; }

				void addSearchPath(const std::string& searchPath, const core::smart_refctd_ptr<IIncludeLoader>& loader);

				void addGenerator(const core::smart_refctd_ptr<IIncludeGenerator>& generator);

			protected:
				IIncludeLoader::found_t trySearchPaths(const std::string& includeName) const;

				IIncludeLoader::found_t tryIncludeGenerators(const std::string& includeName) const;

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

			inline bool operator==(const SPreprocessorOptions& other) const {
				if(sourceIdentifier != other.sourceIdentifier) return false;
				if (extraDefines.size() != other.extraDefines.size()) return false;
				for (auto definesIt = extraDefines.begin(), otherDefinesIt = other.extraDefines.begin(); definesIt != extraDefines.end(); definesIt++, otherDefinesIt++) {
					if (definesIt->identifier != otherDefinesIt->identifier || definesIt->definition != otherDefinesIt->definition) return false;
				}
				return true;
			}

			// We don't need to hash the preprocessor on its own but in conjunction with the compiler options. We return data to perform the hash on from SCompilerOptions
			inline std::vector<char> getHashable() const
			{
				std::vector<char> hashable;
				hashable.insert(hashable.end(), sourceIdentifier.data()[0], sourceIdentifier.data()[sourceIdentifier.size()]);
				core::vector<SMacroDefinition> sortedExtraDefines;
				sortedExtraDefines.assign(extraDefines.begin(), extraDefines.end());
				// Sort them by identifier so the hash is not order-sensitive!
				std::sort(sortedExtraDefines.begin(),sortedExtraDefines.end(),[](const SMacroDefinition& lhs, const SMacroDefinition& rhs){return lhs.identifier<rhs.identifier;});
				for (const auto& defines : sortedExtraDefines) {
					hashable.insert(hashable.end(), defines.identifier.data()[0], defines.identifier.data()[defines.identifier.size()]);
					hashable.insert(hashable.end(), defines.definition.data()[0], defines.definition.data()[defines.definition.size()]);
				}

				return hashable;
			}

			std::string_view sourceIdentifier = "";
			system::logger_opt_ptr logger = nullptr;
			const CIncludeFinder* includeFinder = nullptr;
			struct SMacroDefinition
			{
				std::string_view identifier;
				std::string_view definition;
			};
			std::span<const SMacroDefinition> extraDefines = {};
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
			inline bool operator==(const SCompilerOptions& other) const {
				if (stage != other.stage || targetSpirvVersion != other.targetSpirvVersion || debugInfoFlags != other.debugInfoFlags || preprocessorOptions != other.preprocessorOptions) return false;
				if (spirvOptimizer) {
					if (!other.spirvOptimizer) return false;
					auto passes = spirvOptimizer->getPasses();
					auto otherPasses = other.spirvOptimizer->getPasses();
					if (passes.size() != otherPasses.size()) return false;
					for (auto passesIt = passes.begin(), otherPassesIt = otherPasses.begin(); passesIt != passes.end(); passesIt++, otherPassesIt++) {
						if (*passesIt != *otherPassesIt) return false;
					}
				}
				else {
					if (other.spirvOptimizer) return false;
				}
				return true;
			}

			inline std::vector<char> getHashable() const {
				std::vector<char> hashable = extraHashable_impl();
				std::vector<char> preprocHashable = preprocessorOptions.getHashable();
				hashable.insert(hashable.end(), preprocHashable.data()[0], preprocHashable.data()[preprocHashable.size()]);
				auto stageString = std::to_string(stage);
				hashable.insert(hashable.end(), stageString.data()[0], stageString.data()[stageString.size()]);
				auto versionString = std::to_string(static_cast<uint32_t>(targetSpirvVersion));
				hashable.insert(hashable.end(), versionString.data()[0], versionString.data()[versionString.size()]);
				auto debugString = std::to_string(static_cast<uint8_t>(debugInfoFlags.value));
				hashable.insert(hashable.end(), debugString.data()[0], debugString.data()[debugString.size()]);
				if (spirvOptimizer != nullptr) {
					auto passes = spirvOptimizer->getPasses();
					for (auto passesIt = passes.begin(); passesIt != passes.end(); passesIt++) {
						auto passString = std::to_string(*passesIt);
						hashable.insert(hashable.end(), passString.data()[0], passString.data()[passString.size()]);
					}
				}
				return hashable;
			}

			virtual std::vector<char> extraHashable_impl() const {
				return {};
			}; 


			inline void setCommonData(const SCompilerOptions& opt)
			{
				(*this) = opt;
			}

			virtual IShader::E_CONTENT_TYPE getCodeContentType() const { return IShader::E_CONTENT_TYPE::ECT_UNKNOWN; };

			IShader::E_SHADER_STAGE stage = IShader::E_SHADER_STAGE::ESS_UNKNOWN;
			E_SPIRV_VERSION targetSpirvVersion = E_SPIRV_VERSION::ESV_1_6;
			const ISPIRVOptimizer* spirvOptimizer = nullptr;
			core::bitflag<E_DEBUG_INFO_FLAGS> debugInfoFlags = core::bitflag<E_DEBUG_INFO_FLAGS>(E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT) | E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT;
			SPreprocessorOptions preprocessorOptions = {};
		};

		class CCache final
		{
		public:
			using hash_t = std::array<uint64_t, 4>;

			struct SEntry
			{
				struct SDependency
				{
					// Perf note: hashing while preprocessor lexing is likely to be slower than just hashing the whole array like this 
					inline SDependency(const system::path& _requestingSourceDir, const std::string_view& _identifier, const std::string_view& _contents) :
						requestingSourceDir(_requestingSourceDir), identifier(_identifier), contents(_contents)
					{
						assert(!_contents.empty());
						const auto reqDirStr = requestingSourceDir.make_preferred().string();
						std::vector<char> hashable;
						hashable.insert(hashable.end(), reqDirStr.data()[0], reqDirStr.data()[reqDirStr.size()]);
						hashable.insert(hashable.end(), identifier.data()[0], identifier.data()[identifier.size()]);
						hashable.insert(hashable.end(), _contents.data()[0], _contents.data()[_contents.size()]);
						// Can't static cast here?
						hash = nbl::core::XXHash_256((uint8_t*)(hashable.data()), hashable.size() * (sizeof(char) / sizeof(uint8_t)));
					}

					inline SDependency(SDependency&&) = default;
					inline SDependency& operator=(SDependency&&) = default;

					inline bool operator==(const SDependency& other) const
					{
						return hash == other.hash && identifier == identifier && contents == contents;
					}

					// path or identifier
					system::path requestingSourceDir;
					std::string identifier;
					// file contents
					std::string contents;
					// hash of the contents
					hash_t hash;
					// If true, then `getIncludeStandard` was used to find, otherwise `getIncludeRelative`
					bool standardInclude;
				};

				using dependency_container_t = core::smart_refctd_dynamic_array<const SDependency>;
				template<typename Container>
				inline SEntry(const Container& _dependencies) : dependencies(core::make_refctd_dynamic_array<dependency_container_t>(_dependencies))
				{
					// we must at last have the "main" file as a dependency
					assert(dependencies && dependencies->size() > 0);
					// the main file is not an include
					assert(!dependencies->front().standardInclude);
				}

				inline SEntry(SEntry&&) = default;
				inline SEntry& operator=(SEntry&&) = default;

				// default Equality operator
				inline bool operator==(const SEntry& other)
				{
					if (dependencies->size() != other.dependencies->size())
						return false;
					for (auto i = 0; i != dependencies->size(); i++)
						if (dependencies->operator[](i) != other.dependencies->operator[](i))
							return false;
					return true;
				}

				// The ordering is important here, the dependencies MUST be added to the array IN THE ORDER THE PREPROCESSOR INCLUDED THEM!
				// Obviously the first dependency is the main source file itself! AND ITS HASH MUST INCLUDE THE COMPILE OPTIONS!
				dependency_container_t dependencies;
				// We delay loading the shader at runtime until someone tries to compile it
				std::optional<core::smart_refctd_ptr<asset::ICPUShader>> value;
				system::path mainFilePath;
				asset::IShaderCompiler::SCompilerOptions compilerOptions;
			};

			inline void insert(SEntry&& entry)
			{
				m_container.insert(std::move(entry));
			}

			// can move to .cpp and have it not inline
			inline asset::ICPUShader* find(const SEntry::SDependency& mainFile, CIncludeFinder* finder) const
			{
				auto foundRange = m_container.equal_range(mainFile);
				for (auto found = foundRange.first; found != foundRange.second; found++)
				{
					bool allDependenciesMatch = true;
					// go through all dependencies
					for (auto i = 1; i != found->dependencies->size(); i++)
					{
						const auto& dependency = found->dependencies->operator[](i);

						IIncludeLoader::found_t header;
						if (dependency.standardInclude)
							header = finder->getIncludeStandard(dependency.requestingSourceDir, dependency.identifier);
						else
							header = finder->getIncludeRelative(dependency.requestingSourceDir, dependency.identifier);

						if (header.hash != dependency.hash || header.contents != dependency.contents)
						{
							allDependenciesMatch = false;
							break;
						}
					}
					if (allDependenciesMatch)
						return nullptr;//found->value;
				}
				return nullptr;
			}

			// TODO: add methods as needed, e.g. to serialize and deserialize to/from a pointer

		private:
			// we only do lookups based on main file path + compiler options
			struct Hash
			{
				inline size_t operator()(const SEntry& entry) const noexcept
				{
					std::vector<char> hashable = entry.compilerOptions.getHashable();
					std::string hashableString(hashable.begin(), hashable.end());
					auto pathString = entry.mainFilePath.string();
					hashableString += pathString;
					return std::hash<std::string>{}(hashableString);
				}
			};
			struct KeyEqual
			{
				// used for insertions
				inline bool operator()(const SEntry& lhs, const SEntry& rhs) const
				{
					return lhs.mainFilePath == rhs.mainFilePath && lhs.compilerOptions == rhs.compilerOptions;
				}
			};
			core::unordered_multiset<SEntry, Hash, KeyEqual> m_container;
		};

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const std::string_view code, const SCompilerOptions& options, core::smart_refctd_ptr<CCache> cache = nullptr) const
		{
			core::vector<CCache::SEntry::SDependency> dependencies;
			if (options.cache)
			{
				const auto& mainDep = dependencies.emplace_back(code,options.hash());
				auto found = options.cache->find(mainDep);
				if (found)
					return core::smart_refctd_ptr<ICPUShader>(found);
			}
			auto retval = compileToSPIRV_impl(code,options,dependencies);
			if (options.cache)
				options.cache->insert(CCache::SEntry(dependencies));
			return retval;
		}

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const char* code, const SCompilerOptions& options, core::smart_refctd_ptr<CCache> cache = nullptr) const
		{
			if (!code)
				return nullptr;
			return compileToSPIRV({code,strlen(code)},options,cache);
		}

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(system::IFile* sourceFile, const SCompilerOptions& options, core::smart_refctd_ptr<CCache> cache = nullptr) const
		{
			size_t fileSize = sourceFile->getSize();
			std::string code(fileSize,'\0');

			system::IFile::success_t success;
			sourceFile->read(success, code.data(), 0, fileSize);
			if (success)
				return compileToSPIRV(code,options,cache);
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
			if (!original || original->isADummyObjectForCache() || !original->isContentHighLevelLanguage())
				return nullptr;

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
			const size_t origLen = original ? original->getContent()->getSize():0u;
			const size_t formatArgsCharSize = (getMaxSize(args) + ...);
			const size_t formatSize = strlen(fmt);
			// 2 is an average size of a format (% and a letter) in chars. 
			// Assuming the format contains only one letter, but if it's 2, the outSize is gonna be a touch bigger.
			constexpr size_t nullTerminatorSize = 1u;
			size_t outSize = origLen + formatArgsCharSize + formatSize + nullTerminatorSize - 2 * templateArgsCount;

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

		// TODO: push this crap into CGLSLCompiler, HLSL doesn't use it
		static std::string escapeFilename(std::string&& code);

		static void disableAllDirectivesExceptIncludes(std::string& _code);

		static void reenableDirectives(std::string& _code);

		static std::string encloseWithinExtraInclGuards(std::string&& _code, uint32_t _maxInclusions, const char* _identifier);

		static uint32_t encloseWithinExtraInclGuardsLeadingLines(uint32_t _maxInclusions);
		// end of TODO

		virtual IShader::E_CONTENT_TYPE getCodeContentType() const = 0;

		CIncludeFinder* getDefaultIncludeFinder() { return m_defaultIncludeFinder.get(); }

		const CIncludeFinder* getDefaultIncludeFinder() const { return m_defaultIncludeFinder.get(); }

	protected:
		virtual void insertIntoStart(std::string& code, std::ostringstream&& ins) const = 0;

		// `mainDep` passed just so it doesn't have to be recomputed
		virtual core::smart_refctd_ptr<ICPUShader> compileToSPIRV_impl(const std::string_view code, const SCompilerOptions& options, CCache::SEntry::SDependency&& mainDep) const = 0;

		core::smart_refctd_ptr<system::ISystem> m_system;

	private:
		core::smart_refctd_ptr<CIncludeFinder> m_defaultIncludeFinder;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IShaderCompiler::E_DEBUG_INFO_FLAGS)

}

#endif

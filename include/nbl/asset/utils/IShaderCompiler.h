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

#include "nbl/core/xxHash256.h"

namespace nbl::asset
{

class NBL_API2 IShaderCompiler : public core::IReferenceCounted
{
	public:

		class NBL_API2 IIncludeLoader : public core::IReferenceCounted
		{
			public:
				struct found_t
				{
					system::path absolutePath = {};
					std::string contents = {};
					std::array<uint64_t, 4> hash = {}; // TODO: we're not yet using IFile::getPrecomputedHash(), so for builtins we can maybe use that in the future

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

		class CCache final : public IReferenceCounted
		{
		public:
			using hash_t = std::array<uint64_t, 4>;

			struct SEntry
			{

				struct SPreprocessingDependency
				{
					// Perf note: hashing while preprocessor lexing is likely to be slower than just hashing the whole array like this 
					inline SPreprocessingDependency(const system::path& _requestingSourceDir, const std::string_view& _identifier, const std::string_view& _contents) :
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

					SPreprocessingDependency(SPreprocessingDependency&) = delete;
					SPreprocessingDependency& operator=(SPreprocessingDependency&) = delete;
				
					inline SPreprocessingDependency(SPreprocessingDependency&&) = default;
					inline SPreprocessingDependency& operator=(SPreprocessingDependency&&) = default;

					inline bool operator==(const SPreprocessingDependency& other) const
					{
						return hash == other.hash && identifier == identifier && contents == contents;
					}

					// Needed for json vector serialization I believe
					SPreprocessingDependency() {}

					// path or identifier
					system::path requestingSourceDir = "";
					std::string identifier = "";
					// file contents
					std::string contents = "";
					// hash of the contents
					std::array<uint64_t, 4> hash = {};
					// If true, then `getIncludeStandard` was used to find, otherwise `getIncludeRelative`
					bool standardInclude = false;
					nbl::system::IFileBase::time_point_t lastWriteTime = {};
				};

				struct SMacroData {
					std::string identifier;
					std::string definition;
				};

				struct SPreprocessorData {

					inline bool operator==(const SPreprocessorData& other) const {
						if (sourceIdentifier != other.sourceIdentifier) return false;
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
						core::vector<SMacroData> sortedExtraDefines;
						sortedExtraDefines.assign(extraDefines.begin(), extraDefines.end());
						// Sort them by identifier so the hash is not order-sensitive!
						std::sort(sortedExtraDefines.begin(), sortedExtraDefines.end(), [](const SMacroData& lhs, const SMacroData& rhs) {return lhs.identifier < rhs.identifier; });
						for (const auto& defines : sortedExtraDefines) {
							hashable.insert(hashable.end(), defines.identifier.data()[0], defines.identifier.data()[defines.identifier.size()]);
							hashable.insert(hashable.end(), defines.definition.data()[0], defines.definition.data()[defines.definition.size()]);
						}

						return hashable;
					}

					std::string sourceIdentifier;
					std::vector<SMacroData> extraDefines;
				};

				struct SCompilerData {

					inline bool operator==(const SCompilerData& other) const {
						if (stage != other.stage || targetSpirvVersion != other.targetSpirvVersion || debugInfoFlags != other.debugInfoFlags || preprocessorData != other.preprocessorData) return false;
						if (optimizerPasses.size() != other.optimizerPasses.size()) return false;
						for (auto passesIt = optimizerPasses.begin(), otherPassesIt = other.optimizerPasses.begin(); passesIt != optimizerPasses.end(); passesIt++, otherPassesIt++) {
							if (*passesIt != *otherPassesIt) return false;
						}
						return true;
					}

					inline std::vector<char> getHashable() const {
						std::vector<char> hashable = preprocessorData.getHashable();
						auto stageString = std::to_string(stage);
						hashable.insert(hashable.end(), stageString.data()[0], stageString.data()[stageString.size()]);
						auto versionString = std::to_string(static_cast<uint32_t>(targetSpirvVersion));
						hashable.insert(hashable.end(), versionString.data()[0], versionString.data()[versionString.size()]);
						auto debugString = std::to_string(static_cast<uint8_t>(debugInfoFlags.value));
						hashable.insert(hashable.end(), debugString.data()[0], debugString.data()[debugString.size()]);
						for (auto pass : optimizerPasses) {
							auto passString = std::to_string(pass);
							hashable.insert(hashable.end(), passString.data()[0], passString.data()[passString.size()]);
						}
						return hashable;
					}

					IShader::E_SHADER_STAGE stage;
					E_SPIRV_VERSION targetSpirvVersion;
					std::vector<ISPIRVOptimizer::E_OPTIMIZER_PASS> optimizerPasses;
					core::bitflag<E_DEBUG_INFO_FLAGS> debugInfoFlags;
					SPreprocessorData preprocessorData;
				};

				// The ordering is important here, the dependencies MUST be added to the array IN THE ORDER THE PREPROCESSOR INCLUDED THEM!
				using dependency_container_t = std::vector<SPreprocessingDependency>;
				inline SEntry(std::string_view _mainFileContents, dependency_container_t&& _dependencies, SCompilerData&& compilerData) : mainFileContents(std::move(std::string(_mainFileContents))), dependencies(std::move(_dependencies))
				{
				}

				inline SEntry(SEntry&&) = default;
				inline SEntry& operator=(SEntry&&) = default;

				// This next bit is a bit of a Frankenstein. We serialize shader creation parameters into a json, while the actual shader code goes in another file
				struct CPUShaderCreationParams {
					IShader::E_SHADER_STAGE stage;
					IShader::E_CONTENT_TYPE contentType; //I think this one could be skipped since it's always going to be SPIR-V
					std::string filepathHint;
					uint64_t codeByteSize = 0;
					uint64_t offset = 0; // Offset into the serialized .bin for the Cache where code starts
				};

				std::string mainFileContents;
				SCompilerData compilerData;
				// Keeping this one commented out. Could be useful for lazy-loading of dependencies if it takes up too much memory
				// uint64_t entryID;

				dependency_container_t dependencies;
				mutable CPUShaderCreationParams shaderParams;
				bool serialized = false;
				mutable core::smart_refctd_ptr<asset::ICPUShader> value = nullptr;
			};

			inline void insert(SEntry&& entry)
			{
				m_container.insert(std::move(entry));
			}

			// can move to .cpp and have it not inline
			inline core::smart_refctd_ptr<asset::ICPUShader> find(const SEntry& mainFile, CIncludeFinder* finder)
			{
				auto foundRange = m_container.equal_range(mainFile);
				for (auto& found = foundRange.first; found != foundRange.second; found++)
				{
					bool allDependenciesMatch = true;
					// go through all dependencies
					for (auto i = 1; i != found->dependencies.size(); i++)
					{
						const auto& dependency = found->dependencies[i];

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
					if (allDependenciesMatch) {
						if (!found->value) { // Load shader if not already loaded
							// If the Cache has no storageBuffer, then it's fresh new (not picked up from serialization): all compiled shaders should be loaded
							// If codeByteSize is 0, then the Entry found lives in the Cache but not in the serialization, so the Entry was generated in the current runtime:
							// This means the shader should be loaded
							assert(storageBuffer && found->shaderParams.codeByteSize != 0);
							auto code = core::make_smart_refctd_ptr<ICPUBuffer>(found->shaderParams.codeByteSize);
							memcpy(code->getPointer(), (uint8_t*)storageBuffer->getPointer() + containerJsonSize + found->shaderParams.offset, found->shaderParams.codeByteSize);
							found->value = core::make_smart_refctd_ptr<ICPUShader>(std::move(code), found->shaderParams.stage, found->shaderParams.contentType, std::move(std::string(found->shaderParams.filepathHint)));
						}
						return found->value;
					}	
				}
				return nullptr;
			}

			// TODO: add methods as needed, e.g. to serialize and deserialize to/from a pointer
			std::vector<uint8_t> serialize();
			static CCache deserialize(std::vector<uint8_t>& serializedCache);
		private:
			// we only do lookups based on main file contents + compiler options
			struct Hash
			{
				inline size_t operator()(const SEntry& entry) const noexcept
				{
					std::vector<char> hashable = entry.compilerData.getHashable();
					std::string hashableString(hashable.begin(), hashable.end());
					hashableString += entry.mainFileContents;
					return std::hash<std::string>{}(hashableString);
				}
				
			};
			struct KeyEqual
			{
				// used for insertions
				inline bool operator()(const SEntry& lhs, const SEntry& rhs) const
				{
					return lhs.compilerData == rhs.compilerData && lhs.mainFileContents == rhs.mainFileContents;
				}
				
			};

			core::unordered_multiset<SEntry, Hash, KeyEqual> m_container;
			// If provided at creation time, this will hold the json representing entries in m_container + all SPIRV shaders
			core::smart_refctd_ptr<ICPUBuffer> storageBuffer = nullptr;
			// Specifies the size in bytes of the container json. This is so we know the offset into the buffer where shader bytecode starts
			uint64_t containerJsonSize = 0;
		};

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const std::string_view code, const SCompilerOptions& options, core::smart_refctd_ptr<CCache> cache = nullptr) const
		{
			/*core::vector<CCache::SEntry::SPreprocessingDependency> dependencies;
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
			return retval;*/
			return compileToSPIRV_impl(code, options, nullptr);
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
		virtual std::string preprocessShader(std::string&& code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies = nullptr) const = 0;

		std::string preprocessShader(system::IFile* sourcefile, IShader::E_SHADER_STAGE stage, const SPreprocessorOptions& preprocessOptions, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies = nullptr) const;
		
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

		virtual IShader::E_CONTENT_TYPE getCodeContentType() const = 0;

		CIncludeFinder* getDefaultIncludeFinder() { return m_defaultIncludeFinder.get(); }

		const CIncludeFinder* getDefaultIncludeFinder() const { return m_defaultIncludeFinder.get(); }

	protected:
		virtual void insertIntoStart(std::string& code, std::ostringstream&& ins) const = 0;

		virtual core::smart_refctd_ptr<ICPUShader> compileToSPIRV_impl(const std::string_view code, const SCompilerOptions& options, std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const = 0;

		core::smart_refctd_ptr<system::ISystem> m_system;

	private:
		core::smart_refctd_ptr<CIncludeFinder> m_defaultIncludeFinder;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IShaderCompiler::E_DEBUG_INFO_FLAGS)

}

#endif

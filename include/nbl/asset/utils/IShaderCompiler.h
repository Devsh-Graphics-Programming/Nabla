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

// Less leakage than "nlohmann/json.hpp" only forward declarations
#include "nlohmann/json_fwd.hpp"

namespace nbl::asset
{

class NBL_API2 IShaderCompiler : public core::IReferenceCounted
{
	public:
		IShaderCompiler(core::smart_refctd_ptr<system::ISystem>&& system);

		class NBL_API2 IIncludeLoader : public core::IReferenceCounted
		{
			public:
				struct found_t
				{
					system::path absolutePath = {};
					std::string contents = {};
					std::array<uint64_t, 4> hash = {}; // TODO: we're not yet using IFile::getPrecomputedHash(), so for builtins we can maybe use that in the future
					// Could be used in the future for early rejection of cache hit
					//nbl::system::IFileBase::time_point_t lastWriteTime = {};

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

		//
		struct SMacroDefinition
		{
			friend void to_json(nlohmann::json&, const SMacroDefinition&);
			friend void from_json(const nlohmann::json&, SMacroDefinition&);

			std::string_view identifier;
			std::string_view definition;
		};

		//
		struct SPreprocessorOptions
		{
			std::string_view sourceIdentifier = "";
			system::logger_opt_ptr logger = nullptr;
			const CIncludeFinder* includeFinder = nullptr;
			std::span<const SMacroDefinition> extraDefines = {};
		};

		//
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

		// Forward declaration for SCompilerOptions use
		struct CCache;
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
			CCache* readCache = nullptr;
			CCache* writeCache = nullptr;
		};

		class CCache final : public IReferenceCounted
		{
			public:
				// Used to check compatibility of Caches before reading
				constexpr static inline std::string_view VERSION = "1.0.0";
        
				using hash_t = std::array<uint64_t,4>;
				static auto const SHADER_BUFFER_SIZE_BYTES = sizeof(uint64_t) / sizeof(uint8_t); // It's obviously 8

				struct SEntry
				{
					friend class CCache;

					struct SPreprocessingDependency
					{
						public:
							// Perf note: hashing while preprocessor lexing is likely to be slower than just hashing the whole array like this 
							inline SPreprocessingDependency(const system::path& _requestingSourceDir, const std::string_view& _identifier, const std::string_view& _contents, bool _standardInclude, std::array<uint64_t, 4> _hash) :
								requestingSourceDir(_requestingSourceDir), identifier(_identifier), contents(_contents), standardInclude(_standardInclude), hash(_hash)
							{
								assert(!_contents.empty());
							}

							inline SPreprocessingDependency(SPreprocessingDependency&) = default;
							inline SPreprocessingDependency& operator=(SPreprocessingDependency&) = delete;
							inline SPreprocessingDependency(SPreprocessingDependency&&) = default;
							inline SPreprocessingDependency& operator=(SPreprocessingDependency&&) = default;

							// Needed for json vector serialization. Making it private and declaring from_json(_, SEntry&) as friend didn't work
							inline SPreprocessingDependency() {}

						private:
							friend void to_json(nlohmann::json& j, const SEntry::SPreprocessingDependency& dependency);
							friend void from_json(const nlohmann::json& j, SEntry::SPreprocessingDependency& dependency);
							friend class CCache;

							// path or identifier
							system::path requestingSourceDir = "";
							std::string identifier = "";
							// file contents
							// TODO: change to `core::vector<uint8_t>` a compressed blob of LZMA, and store all contents together in the `SEntry`
							std::string contents = ""; 
							// hash of the contents - used to check against a found_t
							std::array<uint64_t, 4> hash = {};
							// If true, then `getIncludeStandard` was used to find, otherwise `getIncludeRelative`
							bool standardInclude = false;
					};

					struct SCompilerArgs; // Forward declaration for SPreprocessorArgs's friend declaration

					struct SPreprocessorArgs final
					{
						public:					
							inline bool operator==(const SPreprocessorArgs& other) const
							{
								if (sourceIdentifier != other.sourceIdentifier) return false;

								if (extraDefines.size() != other.extraDefines.size()) return false;

								for (auto definesIt = extraDefines.begin(), otherDefinesIt = other.extraDefines.begin(); definesIt != extraDefines.end(); definesIt++, otherDefinesIt++)
								if (definesIt->identifier != otherDefinesIt->identifier || definesIt->definition != otherDefinesIt->definition)
									return false;

								return true;
							}

						private:
							friend class SCompilerArgs;
							friend class SEntry;
							friend void to_json(nlohmann::json&, const SPreprocessorArgs&);
							friend void from_json(const nlohmann::json&, SPreprocessorArgs&);

							// Default constructor needed for json serialization of SCompilerArgs
							SPreprocessorArgs() {};

							inline SPreprocessorArgs(const SPreprocessorArgs&) = default;
							inline SPreprocessorArgs& operator=(const SPreprocessorArgs&) = delete;
							inline SPreprocessorArgs(SPreprocessorArgs&&) = delete;
							inline SPreprocessorArgs& operator=(SPreprocessorArgs&&) = default;

							// Only SCompilerArgs should instantiate this struct
							SPreprocessorArgs(const SPreprocessorOptions& options) : sourceIdentifier(options.sourceIdentifier)
							{
								for (auto define : options.extraDefines)
									extraDefines.emplace_back(std::string(define.identifier), std::string(define.definition));

								// Sort them so equality and hashing are well defined
								std::sort(extraDefines.begin(), extraDefines.end(), [](const SMacroDefinition& lhs, const SMacroDefinition& rhs) {return lhs.identifier < rhs.identifier; });
							};
							std::string sourceIdentifier;
							std::vector<SMacroDefinition> extraDefines;
					};
					// TODO: SPreprocessorArgs could just be folded into `SCompilerArgs` to have less classes and operators
					struct SCompilerArgs final
					{
						public:
							inline bool operator==(const SCompilerArgs& other) const {
								bool retVal = true;
								if (stage != other.stage || targetSpirvVersion != other.targetSpirvVersion || debugInfoFlags != other.debugInfoFlags || preprocessorArgs != other.preprocessorArgs) retVal = false;
								if (optimizerPasses.size() != other.optimizerPasses.size()) retVal = false;
								for (auto passesIt = optimizerPasses.begin(), otherPassesIt = other.optimizerPasses.begin(); passesIt != optimizerPasses.end(); passesIt++, otherPassesIt++) {
									if (*passesIt != *otherPassesIt) {
										retVal = false;
										break;
									}
								}
								return retVal;
							}

						private:
							friend class SEntry;
							friend void to_json(nlohmann::json&, const SCompilerArgs&);
							friend void from_json(const nlohmann::json&, SCompilerArgs&);

							// Default constructor needed for json serialization of SEntry
							SCompilerArgs() {}

							// Default copy needed for SEntry cloning
							inline SCompilerArgs(const SCompilerArgs&) = default;
							inline SCompilerArgs& operator=(const SCompilerArgs&) = delete;
							inline SCompilerArgs(SCompilerArgs&&) = delete;
							inline SCompilerArgs& operator=(SCompilerArgs&&) = default;

							// Only SEntry should instantiate this struct
							SCompilerArgs(const SCompilerOptions& options)
								: stage(options.stage), targetSpirvVersion(options.targetSpirvVersion), debugInfoFlags(options.debugInfoFlags), preprocessorArgs(options.preprocessorOptions)
							{
								if (options.spirvOptimizer) {
									for (auto pass : options.spirvOptimizer->getPasses())
										optimizerPasses.push_back(pass);
								}
							}

							IShader::E_SHADER_STAGE stage;
							E_SPIRV_VERSION targetSpirvVersion;
							std::vector<ISPIRVOptimizer::E_OPTIMIZER_PASS> optimizerPasses;
							core::bitflag<E_DEBUG_INFO_FLAGS> debugInfoFlags;
							SPreprocessorArgs preprocessorArgs;
					};

					// The ordering is important here, the dependencies MUST be added to the array IN THE ORDER THE PREPROCESSOR INCLUDED THEM!
					using dependency_container_t = std::vector<SPreprocessingDependency>;
				
					// Lookup Hash is precompued at entry creation time. Even though in a preprocessing pass the compiler options' stage might change from a #pragma, 
					// we can't update the info that the Cache uses to find entries post preprocessing: this would make it so that we need to preprocess entries
					// to get the "real" stage before lookup in the cache, defeating its purpose
					inline SEntry(const std::string_view _mainFileContents, const SCompilerOptions& compilerOptions) : mainFileContents(std::move(std::string(_mainFileContents))), compilerArgs(compilerOptions)
					{
						// Form the hashable for the compiler data
						size_t preprocessorArgsHashableSize = compilerArgs.preprocessorArgs.sourceIdentifier.size() + compilerArgs.preprocessorArgs.extraDefines.size() * sizeof(SMacroDefinition);
						size_t compilerArgsHashableSize = sizeof(compilerArgs.stage) + sizeof(compilerArgs.targetSpirvVersion) + sizeof(compilerArgs.debugInfoFlags.value) + compilerArgs.optimizerPasses.size();
						std::vector<uint8_t> hashable;
						hashable.reserve(preprocessorArgsHashableSize + compilerArgsHashableSize + mainFileContents.size());
					
						// Insert preproc stuff
						hashable.insert(hashable.end(), compilerArgs.preprocessorArgs.sourceIdentifier.begin(), compilerArgs.preprocessorArgs.sourceIdentifier.end());
						for (const auto& defines : compilerArgs.preprocessorArgs.extraDefines)
						{
							hashable.insert(hashable.end(), defines.identifier.begin(), defines.identifier.end());
							hashable.insert(hashable.end(), defines.definition.begin(), defines.definition.end());
						}

						// Insert rest of stuff from this struct. We're going to treat stage, targetSpirvVersion and debugInfoFlags.value as byte arrays for simplicity
						hashable.insert(hashable.end(), reinterpret_cast<uint8_t*>(&compilerArgs.stage), reinterpret_cast<uint8_t*>(&compilerArgs.stage) + sizeof(compilerArgs.stage));
						hashable.insert(hashable.end(), reinterpret_cast<uint8_t*>(&compilerArgs.targetSpirvVersion), reinterpret_cast<uint8_t*>(&compilerArgs.targetSpirvVersion) + sizeof(compilerArgs.targetSpirvVersion));
						hashable.insert(hashable.end(), reinterpret_cast<uint8_t*>(&compilerArgs.debugInfoFlags.value), reinterpret_cast<uint8_t*>(&compilerArgs.debugInfoFlags.value) + sizeof(compilerArgs.debugInfoFlags.value));
						for (auto pass : compilerArgs.optimizerPasses) {
							hashable.push_back(static_cast<uint8_t>(pass));
						}

						// Now add the mainFileContents and produce both lookup and early equality rejection hashes
						hashable.insert(hashable.end(), mainFileContents.begin(), mainFileContents.end());
						hash = nbl::core::XXHash_256(hashable.data(), hashable.size());
						lookupHash = hash[0];
						for (auto i = 1u; i < 4; i++) {
							core::hash_combine<uint64_t>(lookupHash, hash[i]);
						}
					}

					// Needed to get the vector deserialization automatically
					inline SEntry() {}

					// Making the copy constructor deep-copy everything but the shader 
					inline SEntry(const SEntry& other) 
						: mainFileContents(other.mainFileContents), compilerArgs(other.compilerArgs), hash(other.hash), lookupHash(other.lookupHash), 
						  dependencies(other.dependencies), value(other.value) {}
				
					inline SEntry& operator=(SEntry& other) = delete;
					inline SEntry(SEntry&& other) = default;
					// Used for late initialization while looking up a cache, so as not to always initialize an entry even if caching was not requested
					inline SEntry& operator=(SEntry&& other) = default;

					// TODO: make some of these private
					std::string mainFileContents;
					SCompilerArgs compilerArgs;
					std::array<uint64_t,4> hash;
					size_t lookupHash;
					dependency_container_t dependencies;
					core::smart_refctd_ptr<asset::ICPUShader> value;
				};

				inline void insert(SEntry&& entry)
				{
					m_container.insert(std::move(entry));
				}

				// For now, the merge incorporates what it can. Once we have lastWriteTime going, matching entries could be replaced by the most recent one
				// Alternatively, adding the time an SEntry entered the cache could also serve this purpose
				inline void merge(const CCache* other)
				{
					for (auto& entry : other->m_container)
						m_container.emplace(entry);
				}

				inline core::smart_refctd_ptr<CCache> clone()
				{
					auto retVal = core::make_smart_refctd_ptr<CCache>();
					for (auto& entry : m_container)
						retVal->m_container.emplace(entry);
					return retVal;
				}

				NBL_API2 core::smart_refctd_ptr<asset::ICPUShader> find(const SEntry& mainFile, const CIncludeFinder* finder) const;
		
				inline CCache() {}

				// De/serialization methods
				NBL_API2 core::smart_refctd_ptr<ICPUBuffer> serialize() const;
				NBL_API2 static core::smart_refctd_ptr<CCache> deserialize(const std::span<const uint8_t> serializedCache);

			private:
				// we only do lookups based on main file contents + compiler options
				struct Hash
				{
					inline size_t operator()(const SEntry& entry) const noexcept
					{
						return entry.lookupHash;
					}
				
				};
				struct KeyEqual
				{
					// used for insertions
					inline bool operator()(const SEntry& lhs, const SEntry& rhs) const
					{
						return lhs.compilerArgs == rhs.compilerArgs && lhs.mainFileContents == rhs.mainFileContents;
					}
				
				};
				core::unordered_multiset<SEntry,Hash,KeyEqual> m_container;
		};

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const std::string_view code, const SCompilerOptions& options) const
		{
			CCache::SEntry entry;
			std::vector<CCache::SEntry::SPreprocessingDependency> dependencies;
			if (options.readCache or options.writeCache)
				entry = std::move(CCache::SEntry(code, options));
			if (options.readCache)
			{
				auto found = options.readCache->find(entry, options.preprocessorOptions.includeFinder);
				if (found)
					return found;
			}
			auto retVal = compileToSPIRV_impl(code, options, options.writeCache ? &dependencies : nullptr);
			if (options.writeCache)
			{
				entry.dependencies = std::move(dependencies);
				entry.value = retVal;
				options.writeCache->insert(std::move(entry));
			}
			return retVal;
		}

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(const char* code, const SCompilerOptions& options) const
		{
			if (!code)
				return nullptr;
			return compileToSPIRV({code,strlen(code)},options);
		}

		inline core::smart_refctd_ptr<ICPUShader> compileToSPIRV(system::IFile* sourceFile, const SCompilerOptions& options) const
		{
			size_t fileSize = sourceFile->getSize();
			std::string code(fileSize,'\0');

			system::IFile::success_t success;
			sourceFile->read(success, code.data(), 0, fileSize);
			if (success)
				return compileToSPIRV(code,options);
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

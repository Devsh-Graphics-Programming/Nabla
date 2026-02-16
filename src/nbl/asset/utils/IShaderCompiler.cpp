// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/IShaderCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"

#include <sstream>
#include <regex>
#include <iterator>
#include <filesystem>
#include <algorithm>

#include <lzma/C/LzmaEnc.h>
#include <lzma/C/LzmaDec.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using SEntry = nbl::asset::IShaderCompiler::CCache::SEntry;

using namespace nbl;
using namespace nbl::asset;

// -> serialization
// SMacroData, simple container used in SPreprocessorArgs
namespace nbl::system::json {
    template<>
    struct adl_serializer<IShaderCompiler::SMacroDefinition>
    {
        using value_t = IShaderCompiler::SMacroDefinition;

        static inline void to_json(::json& j, const value_t& p)
        {
            j = ::json{
                { "identifier", p.identifier },
                { "definition", p.definition },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            j.at("identifier").get_to(p.identifier);
            j.at("definition").get_to(p.definition);
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<IShaderCompiler::SMacroDefinition>)

// SPreprocessorData, holds serialized info for Preprocessor options used during compilation
namespace nbl::system::json {
    template<>
    struct adl_serializer<SEntry::SPreprocessorArgs>
    {
        using value_t = SEntry::SPreprocessorArgs;

        static inline void to_json(::json& j, const value_t& p)
        {
            j = ::json{
                { "sourceIdentifier", p.sourceIdentifier },
                { "extraDefines", p.extraDefines},
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            j.at("sourceIdentifier").get_to(p.sourceIdentifier);
            j.at("extraDefines").get_to(p.extraDefines);
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<SEntry::SPreprocessorArgs>)

// Optimizer pass has its own method for easier vector serialization
namespace nbl::system::json {
    template<>
    struct adl_serializer<ISPIRVOptimizer::E_OPTIMIZER_PASS>
    {
        using value_t = ISPIRVOptimizer::E_OPTIMIZER_PASS;

        static inline void to_json(::json& j, const value_t& p)
        {
            uint32_t value = static_cast<uint32_t>(p);
            j = ::json{
                { "optPass", value },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            uint32_t aux;
            j.at("optPass").get_to(aux);
            p = static_cast<ISPIRVOptimizer::E_OPTIMIZER_PASS>(aux);
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<ISPIRVOptimizer::E_OPTIMIZER_PASS>)

// SCompilerArgs, holds serialized info for all Compilation options
namespace nbl::system::json {
    template<>
    struct adl_serializer<SEntry::SCompilerArgs>
    {
        using value_t = SEntry::SCompilerArgs;

        static inline void to_json(::json& j, const value_t& p)
        {
            uint32_t shaderStage = static_cast<uint32_t>(p.stage);
            uint32_t spirvVersion = static_cast<uint32_t>(p.targetSpirvVersion);
            uint32_t debugFlags = static_cast<uint32_t>(p.debugInfoFlags.value);

            j = ::json{
                { "shaderStage", shaderStage },
                { "spirvVersion", spirvVersion },
                { "optimizerPasses", p.optimizerPasses },
                { "debugFlags", debugFlags },
                { "preprocessorArgs", p.preprocessorArgs },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            uint32_t shaderStage, spirvVersion, debugFlags;
            j.at("shaderStage").get_to(shaderStage);
            j.at("spirvVersion").get_to(spirvVersion);
            j.at("optimizerPasses").get_to(p.optimizerPasses);
            j.at("debugFlags").get_to(debugFlags);
            j.at("preprocessorArgs").get_to(p.preprocessorArgs);
            p.stage = static_cast<IShader::E_SHADER_STAGE>(shaderStage);
            p.targetSpirvVersion = static_cast<IShaderCompiler::E_SPIRV_VERSION>(spirvVersion);
            p.debugInfoFlags = core::bitflag<IShaderCompiler::E_DEBUG_INFO_FLAGS>(debugFlags);
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<SEntry::SCompilerArgs>)

// Serialize clock's time point
using time_point_t = nbl::system::IFileBase::time_point_t;
namespace nbl::system::json {
    template<>
    struct adl_serializer<time_point_t>
    {
        using value_t = time_point_t;

        static inline void to_json(::json& j, const value_t& p)
        {
            auto ticks = p.time_since_epoch().count();
            j = ::json{
                { "ticks", ticks },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            uint64_t ticks;
            j.at("ticks").get_to(ticks);
            p = time_point_t(time_point_t::clock::duration(ticks));
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<time_point_t>)

// SDependency serialization. Dependencies will be saved in a vector for easier vectorization
namespace nbl::system::json {
    template<>
    struct adl_serializer<SEntry::SPreprocessingDependency>
    {
        using value_t = SEntry::SPreprocessingDependency;

        static inline void to_json(::json& j, const value_t& p)
        {
            j = ::json{
                { "requestingSourceDir", p.requestingSourceDir },
                { "identifier", p.identifier },
                { "hash", p.hash.data },
                { "standardInclude", p.standardInclude },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            j.at("requestingSourceDir").get_to(p.requestingSourceDir);
            j.at("identifier").get_to(p.identifier);
            j.at("hash").get_to(p.hash.data);
            j.at("standardInclude").get_to(p.standardInclude);
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<SEntry::SPreprocessingDependency>)

// We serialize shader creation parameters into a json, along with indexing info into the .bin buffer where the cache is serialized
struct CPUShaderCreationParams {
    IShader::E_SHADER_STAGE stage;
    std::string filepathHint;
    uint64_t codeByteSize = 0;
    uint64_t offset = 0; // Offset into the serialized .bin for the Cache where code starts

    CPUShaderCreationParams(IShader::E_SHADER_STAGE _stage, std::string_view _filepathHint, uint64_t _codeByteSize, uint64_t _offset)
        : stage(_stage), filepathHint(_filepathHint), codeByteSize(_codeByteSize), offset(_offset) {}
    CPUShaderCreationParams() {};
};

namespace nbl::system::json {
    template<>
    struct adl_serializer<CPUShaderCreationParams>
    {
        using value_t = CPUShaderCreationParams;

        static inline void to_json(::json& j, const value_t& p)
        {
            uint32_t stage = static_cast<uint32_t>(p.stage);
            j = ::json{
                { "stage", stage },
                { "filepathHint", p.filepathHint },
                { "codeByteSize", p.codeByteSize },
                { "offset", p.offset },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            uint32_t stage;
            j.at("stage").get_to(stage);
            j.at("filepathHint").get_to(p.filepathHint);
            j.at("codeByteSize").get_to(p.codeByteSize);
            j.at("offset").get_to(p.offset);
            p.stage = static_cast<IShader::E_SHADER_STAGE>(stage);
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<CPUShaderCreationParams>)

// Serialize SEntry, keeping some fields as extra serialization to keep them separate on disk
namespace nbl::system::json {
    template<>
    struct adl_serializer<SEntry>
    {
        using value_t = SEntry;

        static inline void to_json(::json& j, const value_t& p)
        {
            j = ::json{
                { "mainFileContents", p.mainFileContents },
                { "compilerArgs", p.compilerArgs },
                { "hash", p.hash.data },
                { "lookupHash", p.lookupHash },
                { "dependencies", p.dependencies },
                { "uncompressedContentHash", p.uncompressedContentHash.data },
                { "uncompressedSize", p.uncompressedSize },
            };
        }

        static inline void from_json(const ::json& j, value_t& p)
        {
            j.at("mainFileContents").get_to(p.mainFileContents);
            j.at("compilerArgs").get_to(p.compilerArgs);
            j.at("hash").get_to(p.hash.data);
            j.at("lookupHash").get_to(p.lookupHash);
            j.at("dependencies").get_to(p.dependencies);
            j.at("uncompressedContentHash").get_to(p.uncompressedContentHash.data);
            j.at("uncompressedSize").get_to(p.uncompressedSize);
            p.spirv = nullptr;
        }
    };
}
NBL_JSON_IMPL_BIND_ADL_SERIALIZER(::nbl::system::json::adl_serializer<SEntry>)
// <- serialization

IShaderCompiler::IShaderCompiler(core::smart_refctd_ptr<system::ISystem>&& system)
    : m_system(std::move(system))
{
    m_defaultIncludeFinder = core::make_smart_refctd_ptr<CIncludeFinder>(core::smart_refctd_ptr(m_system));
}

bool IShaderCompiler::writeDepfile(
	const DepfileWriteParams& params,
	const CCache::SEntry::dependency_container_t& dependencies,
	const CIncludeFinder* includeFinder,
	system::logger_opt_ptr logger)
{
	std::string depfilePathString;
	if (!params.depfilePath.empty())
		depfilePathString = std::string(params.depfilePath);
	else
		depfilePathString = std::string(params.outputPath) + ".d";

	if (depfilePathString.empty())
	{
		logger.log("Depfile path is empty.", system::ILogger::ELL_ERROR);
		return false;
	}

	const auto parentDirectory = std::filesystem::path(depfilePathString).parent_path();
	if (!parentDirectory.empty() && !std::filesystem::exists(parentDirectory))
	{
		if (!std::filesystem::create_directories(parentDirectory))
		{
			logger.log("Failed to create parent directory for depfile.", system::ILogger::ELL_ERROR);
			return false;
		}
	}

	std::vector<std::string> depPaths;
	depPaths.reserve(dependencies.size() + 1);

	auto addDepPath = [&depPaths, &params](std::filesystem::path path)
	{
		if (path.empty())
			return;
		if (path.is_relative())
		{
			if (params.workingDirectory.empty())
				return;
			path = std::filesystem::path(params.workingDirectory) / path;
		}
		std::error_code ec;
		std::filesystem::path normalized = std::filesystem::weakly_canonical(path, ec);
		if (ec)
		{
			normalized = std::filesystem::absolute(path, ec);
			if (ec)
				return;
		}
		if (normalized.empty() || !std::filesystem::exists(normalized))
			return;
		auto normalizedString = normalized.generic_string();
		if (normalizedString.find_first_of("\r\n") != std::string::npos)
			return;
		depPaths.emplace_back(std::move(normalizedString));
	};

	if (!params.sourceIdentifier.empty())
	{
		std::filesystem::path rootPath{std::string(params.sourceIdentifier)};
		if (rootPath.is_relative())
		{
			if (!params.workingDirectory.empty())
				rootPath = std::filesystem::absolute(std::filesystem::path(params.workingDirectory) / rootPath);
			else
				rootPath = std::filesystem::absolute(rootPath);
		}
		addDepPath(rootPath);
	}

	for (const auto& dep : dependencies)
	{
		if (includeFinder)
		{
			IShaderCompiler::IIncludeLoader::found_t header = dep.isStandardInclude() ?
				includeFinder->getIncludeStandard(dep.getRequestingSourceDir(), std::string(dep.getIdentifier())) :
				includeFinder->getIncludeRelative(dep.getRequestingSourceDir(), std::string(dep.getIdentifier()));

			if (!header)
				continue;
			addDepPath(header.absolutePath);
		}
		else
		{
			std::filesystem::path candidate = dep.isStandardInclude() ? std::filesystem::path(std::string(dep.getIdentifier())) : (dep.getRequestingSourceDir() / std::string(dep.getIdentifier()));
			if (candidate.is_relative())
			{
				if (!params.workingDirectory.empty())
					candidate = std::filesystem::absolute(std::filesystem::path(params.workingDirectory) / candidate);
				else
					candidate = std::filesystem::absolute(candidate);
			}
			addDepPath(candidate);
		}
	}

	std::sort(depPaths.begin(), depPaths.end());
	depPaths.erase(std::unique(depPaths.begin(), depPaths.end()), depPaths.end());

	auto escapeDepPath = [](const std::string& path) -> std::string
	{
		std::string normalized = path;
		std::replace(normalized.begin(), normalized.end(), '\\', '/');
		std::string out;
		out.reserve(normalized.size());
		for (const char c : normalized)
		{
			if (c == ' ' || c == '#')
				out.push_back('\\');
			if (c == '$')
			{
				out.push_back('$');
				out.push_back('$');
				continue;
			}
			out.push_back(c);
		}
		return out;
	};

	if (!params.system)
	{
		logger.log("Depfile system is null.", system::ILogger::ELL_ERROR);
		return false;
	}

	const auto depfilePath = std::filesystem::path(depfilePathString);
	auto tempPath = depfilePath;
	tempPath += ".tmp";
	params.system->deleteFile(tempPath);

	core::smart_refctd_ptr<system::IFile> depfile;
	{
		system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
		params.system->createFile(future, tempPath, system::IFileBase::ECF_WRITE);
		if (!future.wait())
		{
			logger.log("Failed to open depfile: %s", system::ILogger::ELL_ERROR, depfilePathString.c_str());
			return false;
		}
		future.acquire().move_into(depfile);
	}
	if (!depfile)
	{
		logger.log("Failed to open depfile: %s", system::ILogger::ELL_ERROR, depfilePathString.c_str());
		return false;
	}

	std::string targetPathString;
	if (params.outputPath.empty())
	{
		std::filesystem::path targetPath = depfilePathString;
		if (targetPath.extension() == ".d")
			targetPath.replace_extension();
		targetPathString = targetPath.generic_string();
	}
	else
	{
		targetPathString = std::string(params.outputPath);
	}
	if (targetPathString.empty())
	{
		logger.log("Depfile target path is empty.", system::ILogger::ELL_ERROR);
		return false;
	}
	const std::string target = escapeDepPath(std::filesystem::path(targetPathString).generic_string());
	std::vector<std::string> escapedDeps;
	escapedDeps.reserve(depPaths.size());
	for (const auto& depPath : depPaths)
		escapedDeps.emplace_back(escapeDepPath(depPath));

	std::string depfileContents;
	depfileContents.append(target);
	depfileContents.append(":");
	if (!escapedDeps.empty())
	{
		depfileContents.append(" \\\n");
		for (size_t index = 0; index < escapedDeps.size(); ++index)
		{
			depfileContents.append(" ");
			depfileContents.append(escapedDeps[index]);
			if (index + 1 < escapedDeps.size())
				depfileContents.append(" \\\n");
		}
	}
	depfileContents.append("\n");

	system::IFile::success_t success;
	depfile->write(success, depfileContents.data(), 0, depfileContents.size());
	if (!success)
	{
		logger.log("Failed to write depfile: %s", system::ILogger::ELL_ERROR, depfilePathString.c_str());
		return false;
	}
	depfile = nullptr;

	params.system->deleteFile(depfilePath);
	const std::error_code moveError = params.system->moveFileOrDirectory(tempPath, depfilePath);
	if (moveError)
	{
		logger.log("Failed to replace depfile: %s", system::ILogger::ELL_ERROR, depfilePathString.c_str());
		return false;
	}
	return true;
}

core::smart_refctd_ptr<IShader> nbl::asset::IShaderCompiler::compileToSPIRV(const std::string_view code, const SCompilerOptions& options) const
{
	const bool depfileEnabled = options.preprocessorOptions.depfile;
	const bool supportsDependencies = options.getCodeContentType() == IShader::E_CONTENT_TYPE::ECT_HLSL;

	auto writeDepfileFromDependencies = [&](const CCache::SEntry::dependency_container_t& dependencies) -> bool
	{
		if (!depfileEnabled)
			return true;

		if (options.preprocessorOptions.depfilePath.empty())
		{
			options.preprocessorOptions.logger.log("Depfile path is empty.", system::ILogger::ELL_ERROR);
			return false;
		}

		IShaderCompiler::DepfileWriteParams params = {};
		const std::string depfilePathString = options.preprocessorOptions.depfilePath.generic_string();
		params.depfilePath = depfilePathString;
		params.sourceIdentifier = options.preprocessorOptions.sourceIdentifier;
		if (!params.sourceIdentifier.empty())
			params.workingDirectory = std::filesystem::path(std::string(params.sourceIdentifier)).parent_path();
		params.system = m_system.get();
		return IShaderCompiler::writeDepfile(params, dependencies, options.preprocessorOptions.includeFinder, options.preprocessorOptions.logger);
	};

	CCache::SEntry entry;
	if (options.readCache || options.writeCache)
		entry = CCache::SEntry(code, options);

	if (options.readCache)
	{
		auto found = options.readCache->find_impl(entry, options.preprocessorOptions.includeFinder);
		if (found != options.readCache->m_container.end())
		{
			if (options.writeCache)
			{
				CCache::SEntry writeEntry = *found;
				options.writeCache->insert(std::move(writeEntry));
			}
			auto shader = found->decompressShader();
			if (depfileEnabled && !writeDepfileFromDependencies(found->dependencies))
				return nullptr;
			return shader;
		}
	}

	CCache::SEntry::dependency_container_t depfileDependencies;
	CCache::SEntry::dependency_container_t* dependenciesPtr = nullptr;
	if (options.writeCache)
		dependenciesPtr = &entry.dependencies;
	else if (depfileEnabled && supportsDependencies)
		dependenciesPtr = &depfileDependencies;

	auto retVal = compileToSPIRV_impl(code, options, dependenciesPtr);
	if (retVal)
	{
		auto backingBuffer = retVal->getContent();
		const_cast<ICPUBuffer*>(backingBuffer)->setContentHash(backingBuffer->computeContentHash());
	}

	if (retVal && depfileEnabled && supportsDependencies)
	{
		const auto* deps = options.writeCache ? &entry.dependencies : &depfileDependencies;
		if (!writeDepfileFromDependencies(*deps))
			return nullptr;
	}

	if (options.writeCache)
	{
		if (entry.setContent(retVal->getContent()))
			options.writeCache->insert(std::move(entry));
	}

	return retVal;
}

std::string IShaderCompiler::preprocessShader(
    system::IFile* sourcefile,
    IShader::E_SHADER_STAGE stage,
    const SPreprocessorOptions& preprocessOptions,
    std::vector<CCache::SEntry::SPreprocessingDependency>* dependencies) const
{
    std::string code(sourcefile->getSize(), '\0');

    system::IFile::success_t success;
    sourcefile->read(success, code.data(), 0, sourcefile->getSize());
    if (!success)
        return nullptr;

    return preprocessShader(std::move(code), stage, preprocessOptions, dependencies);
}
auto IShaderCompiler::IIncludeGenerator::getInclude(const std::string& includeName) const -> IIncludeLoader::found_t
{
    core::vector<std::pair<std::regex, HandleFunc_t>> builtinNames = getBuiltinNamesToFunctionMapping();
    for (const auto& pattern : builtinNames)
        if (std::regex_match(includeName, pattern.first))
        {
            if (auto contents = pattern.second(includeName); !contents.empty())
            {
                // Welcome, you've came to a very disused piece of code, please check the first parameter (path) makes sense!
                _NBL_DEBUG_BREAK_IF(true);
                return { includeName,contents };
            }
        }

    return {};
}

core::vector<std::string> IShaderCompiler::IIncludeGenerator::parseArgumentsFromPath(const std::string& _path)
{
    core::vector<std::string> args;

    std::stringstream ss{ _path };
    std::string arg;
    while (std::getline(ss, arg, '/'))
        args.emplace_back(std::move(arg));

    return args;
}

IShaderCompiler::CFileSystemIncludeLoader::CFileSystemIncludeLoader(core::smart_refctd_ptr<system::ISystem>&& system) : m_system(std::move(system))
{}

auto IShaderCompiler::CFileSystemIncludeLoader::getInclude(const system::path& searchPath, const std::string& includeName) const -> found_t
{
    system::path path = searchPath / includeName;
    if (std::filesystem::exists(path))
        path = std::filesystem::canonical(path);

    core::smart_refctd_ptr<system::IFile> f;
    {
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_system->createFile(future, path.c_str(), system::IFile::ECF_READ);
        if (!future.wait())
            return {};
        future.acquire().move_into(f);
    }
    if (!f)
        return {};
    const size_t size = f->getSize();

    std::string contents(size, '\0');
    system::IFile::success_t succ;
    f->read(succ, contents.data(), 0, size);
    const bool success = bool(succ);
    assert(success);

    return { f->getFileName(),std::move(contents) };
}

IShaderCompiler::CIncludeFinder::CIncludeFinder(core::smart_refctd_ptr<system::ISystem>&& system)
    : m_defaultFileSystemLoader(core::make_smart_refctd_ptr<CFileSystemIncludeLoader>(std::move(system)))
{
    addSearchPath("", m_defaultFileSystemLoader);
}

// ! includes within <>
// @param requestingSourceDir: the directory where the incude was requested
// @param includeName: the string within <> of the include preprocessing directive
// @param 
auto IShaderCompiler::CIncludeFinder::getIncludeStandard(const system::path& requestingSourceDir, const std::string& includeName) const -> IIncludeLoader::found_t
{
    IShaderCompiler::IIncludeLoader::found_t retVal;
    if (auto contents = tryIncludeGenerators(includeName))
        retVal = std::move(contents);
    else if (auto contents = trySearchPaths(includeName))
        retVal = std::move(contents);
    else retVal = m_defaultFileSystemLoader->getInclude(requestingSourceDir.string(), includeName);


    core::blake3_hasher hasher;
    hasher.update(reinterpret_cast<uint8_t*>(retVal.contents.data()), retVal.contents.size() * (sizeof(char) / sizeof(uint8_t)));
    retVal.hash = static_cast<core::blake3_hash_t>(hasher);
    return retVal;
}

// ! includes within ""
// @param requestingSourceDir: the directory where the incude was requested
// @param includeName: the string within "" of the include preprocessing directive
auto IShaderCompiler::CIncludeFinder::getIncludeRelative(const system::path& requestingSourceDir, const std::string& includeName) const -> IIncludeLoader::found_t
{
    IShaderCompiler::IIncludeLoader::found_t retVal;
    if (auto contents = m_defaultFileSystemLoader->getInclude(requestingSourceDir.string(), includeName))
        retVal = std::move(contents);
    else retVal = std::move(trySearchPaths(includeName));

    core::blake3_hasher hasher;
    hasher.update(reinterpret_cast<uint8_t*>(retVal.contents.data()), retVal.contents.size() * (sizeof(char) / sizeof(uint8_t)));
    retVal.hash = static_cast<core::blake3_hash_t>(hasher);
    return retVal;
}

void IShaderCompiler::CIncludeFinder::addSearchPath(const std::string& searchPath, const core::smart_refctd_ptr<IIncludeLoader>& loader)
{
    if (!loader)
        return;
    m_loaders.emplace_back(LoaderSearchPath{ loader, searchPath });
}

void IShaderCompiler::CIncludeFinder::addGenerator(const core::smart_refctd_ptr<IIncludeGenerator>& generatorToAdd)
{
    if (!generatorToAdd)
        return;

    // this will find the place of first generator with prefix <= generatorToAdd or end
    auto found = std::lower_bound(m_generators.begin(), m_generators.end(), generatorToAdd->getPrefix(),
        [](const core::smart_refctd_ptr<IIncludeGenerator>& generator, const std::string_view& value)
        {
            auto element = generator->getPrefix();
            return element.compare(value) > 0; // first to return false is lower_bound -> first element that is <= value
        });

    m_generators.insert(found, generatorToAdd);
}

auto IShaderCompiler::CIncludeFinder::trySearchPaths(const std::string& includeName) const -> IIncludeLoader::found_t
{
    for (const auto& itr : m_loaders)
        if (auto contents = itr.loader->getInclude(itr.searchPath, includeName))
            return contents;
    return {};
}

auto IShaderCompiler::CIncludeFinder::tryIncludeGenerators(const std::string& includeName) const -> IIncludeLoader::found_t
{
    // Need custom function because std::filesystem doesn't consider the parameters we use after the extension like CustomShader.hlsl/512/64
    auto removeExtension = [](const std::string& str)
        {
            return str.substr(0, str.find_last_of('.'));
        };

    auto standardizePrefix = [](const std::string_view& prefix) -> std::string
        {
            std::string ret(prefix);
            // Remove Trailing '/' if any, to compare to filesystem paths
            if (*ret.rbegin() == '/' && ret.size() > 1u)
                ret.resize(ret.size() - 1u);
            return ret;
        };

    auto extension_removed_path = system::path(removeExtension(includeName));
    system::path path = extension_removed_path.parent_path();

    // Try Generators with Matching Prefixes:
    // Uses a "Path Peeling" method which goes one level up the directory tree until it finds a suitable generator
    auto end = m_generators.begin();
    while (!path.empty() && path.root_name().empty() && end != m_generators.end())
    {
        auto begin = std::lower_bound(end, m_generators.end(), path.string(),
            [&standardizePrefix](const core::smart_refctd_ptr<IIncludeGenerator>& generator, const std::string& value)
            {
                const auto element = standardizePrefix(generator->getPrefix());
                return element.compare(value) > 0; // first to return false is lower_bound -> first element that is <= value
            });

        // search from new beginning to real end
        end = std::upper_bound(begin, m_generators.end(), path.string(),
            [&standardizePrefix](const std::string& value, const core::smart_refctd_ptr<IIncludeGenerator>& generator)
            {
                const auto element = standardizePrefix(generator->getPrefix());
                return value.compare(element) > 0; // first to return true is upper_bound -> first element that is < value
            });

        for (auto generatorIt = begin; generatorIt != end; generatorIt++)
        {
            if (auto contents = (*generatorIt)->getInclude(includeName))
                return contents;
        }

        path = path.parent_path();
    }

    return {};
}

core::smart_refctd_ptr<asset::IShader> IShaderCompiler::CCache::find(const SEntry& mainFile, const IShaderCompiler::CIncludeFinder* finder) const
{
    const auto found = find_impl(mainFile, finder);
    if (found==m_container.end())
        return nullptr;
    return found->decompressShader();
}

IShaderCompiler::CCache::EntrySet::const_iterator IShaderCompiler::CCache::find_impl(const SEntry& mainFile, const IShaderCompiler::CIncludeFinder* finder) const
{
    auto found = m_container.find(mainFile);
    // go through all dependencies
    if (found!=m_container.end())
    {
        for (const auto& dependency : found->dependencies)
        {
            IIncludeLoader::found_t header;
            if (dependency.standardInclude)
                header = finder->getIncludeStandard(dependency.requestingSourceDir, dependency.identifier);
            else
                header = finder->getIncludeRelative(dependency.requestingSourceDir, dependency.identifier);

            if (header.hash != dependency.hash)
            {
                return m_container.end();
            }
        }
    }

    return found;
}

core::smart_refctd_ptr<ICPUBuffer> IShaderCompiler::CCache::serialize() const
{
    size_t shaderBufferSize = 0;
    core::vector<size_t> offsets(m_container.size());
    core::vector<uint64_t> sizes(m_container.size());
    json entries;
    core::vector<CPUShaderCreationParams> shaderCreationParams;

    // In a first loop over entries we add all entries and their shader creation parameters to a json, and get the size of the shaders buffer
    size_t i = 0u;
    for (auto& entry : m_container) {
        // Add the entry as a json array
        entries.emplace_back(entry);

        // We keep a copy of the offsets and the sizes of each shader. This is so that later on, when we add the shaders to the buffer after json creation
        // (where the params array has been moved) we don't have to read the json to get the offsets again
        offsets[i] = shaderBufferSize;
        sizes[i] = entry.spirv->getSize();

        // And add the params to the shader creation parameters array
        shaderCreationParams.emplace_back(entry.compilerArgs.stage, entry.compilerArgs.preprocessorArgs.sourceIdentifier.data(), sizes[i], shaderBufferSize);
        // Enlarge the shader buffer by the size of the current shader
        shaderBufferSize += sizes[i];
        i++;
    }

    // Create the containerJson
    json containerJson{
        { "version", VERSION },
        { "entries", std::move(entries) },
        { "shaderCreationParams", std::move(shaderCreationParams) },
    };
    std::string dumpedContainerJson = std::move(containerJson.dump());
    uint64_t dumpedContainerJsonLength = dumpedContainerJson.size();

    // Create a buffer able to hold all shaders + the containerJson
    size_t retValSize = shaderBufferSize + SHADER_BUFFER_SIZE_BYTES + dumpedContainerJsonLength;
    core::vector<uint8_t> retVal(retValSize);

    // first SHADER_BUFFER_SIZE_BYTES (8) in the buffer are the size of the shader buffer
    memcpy(retVal.data(), &shaderBufferSize, SHADER_BUFFER_SIZE_BYTES);

    // Loop over entries again, adding each one's shader to the buffer. 
    i = 0u;
    for (auto& entry : m_container) {
        memcpy(retVal.data() + SHADER_BUFFER_SIZE_BYTES + offsets[i], entry.spirv->getPointer(), sizes[i]);
        i++;
    }

    // Might as well memcpy everything
    memcpy(retVal.data() + SHADER_BUFFER_SIZE_BYTES + shaderBufferSize, dumpedContainerJson.data(), dumpedContainerJsonLength);

    auto memoryResource = core::make_smart_refctd_ptr<core::adoption_memory_resource<decltype(retVal)>>(std::move(retVal));
    return ICPUBuffer::create({ { retValSize }, memoryResource->getBacker().data(),std::move(memoryResource)}, core::adopt_memory);
}

core::smart_refctd_ptr<IShaderCompiler::CCache> IShaderCompiler::CCache::deserialize(const std::span<const uint8_t> serializedCache)
{
    auto retVal = core::make_smart_refctd_ptr<CCache>();

    // First get the size of the shader buffer, stored in the first 8 bytes
    const uint64_t* cacheStart = reinterpret_cast<const uint64_t*>(serializedCache.data());
    uint64_t shaderBufferSize = cacheStart[0];
    // Next up get the json that stores the container data
    std::span<const char> cacheAsChar = { reinterpret_cast<const char*>(serializedCache.data()), serializedCache.size() };
    std::string_view containerJsonString(cacheAsChar.begin() + SHADER_BUFFER_SIZE_BYTES + shaderBufferSize, cacheAsChar.end());
    json containerJson = json::parse(containerJsonString);

    // Check that this cache is from the currently supported version
    {
        std::string version;
        containerJson.at("version").get_to(version);
        if (version != VERSION) {
            return nullptr;
        }
    }

    // Now retrieve two vectors, one with the entries and one with the extra data to recreate the CPUShaders
    std::vector<SEntry> entries;
    std::vector<CPUShaderCreationParams> shaderCreationParams;
    containerJson.at("entries").get_to(entries);
    containerJson.at("shaderCreationParams").get_to(shaderCreationParams);

    // We must now recreate the shaders, add them to each entry, then move the entry into the multiset
    for (auto i = 0u; i < entries.size(); i++) {
        // Create buffer to hold the code
        auto code = ICPUBuffer::create({ shaderCreationParams[i].codeByteSize });
        // Copy the shader bytecode into the buffer

        memcpy(code->getPointer(), serializedCache.data() + SHADER_BUFFER_SIZE_BYTES + shaderCreationParams[i].offset, shaderCreationParams[i].codeByteSize);
        code->setContentHash(code->computeContentHash());
        entries[i].spirv = std::move(code);

        retVal->insert(std::move(entries[i]));
    }

    return retVal;
}

static void* SzAlloc(ISzAllocPtr p, size_t size) { p = p; return _NBL_ALIGNED_MALLOC(size, _NBL_SIMD_ALIGNMENT); }
static void SzFree(ISzAllocPtr p, void* address) { p = p; _NBL_ALIGNED_FREE(address); }

bool nbl::asset::IShaderCompiler::CCache::SEntry::setContent(const asset::ICPUBuffer* uncompressedSpirvBuffer)
{
    uncompressedContentHash = uncompressedSpirvBuffer->getContentHash();
    uncompressedSize = uncompressedSpirvBuffer->getSize();

    size_t propsSize = LZMA_PROPS_SIZE;
    size_t destLen = uncompressedSpirvBuffer->getSize() + uncompressedSpirvBuffer->getSize() / 3 + 128;
    core::vector<uint8_t> compressedSpirv(propsSize + destLen);

    CLzmaEncProps props;
    LzmaEncProps_Init(&props);
    props.dictSize = 1 << 16; // 64KB
    props.writeEndMark = 1;

    ISzAlloc sz_alloc = { SzAlloc, SzFree };
    int res = LzmaEncode(
        compressedSpirv.data() + LZMA_PROPS_SIZE, &destLen,
        reinterpret_cast<const unsigned char*>(uncompressedSpirvBuffer->getPointer()), uncompressedSpirvBuffer->getSize(),
        &props, compressedSpirv.data(), &propsSize, props.writeEndMark,
        nullptr, &sz_alloc, &sz_alloc);

    if (res != SZ_OK || propsSize != LZMA_PROPS_SIZE) return false;
    compressedSpirv.resize(propsSize + destLen);

    auto memoryResource = core::make_smart_refctd_ptr<core::adoption_memory_resource<decltype(compressedSpirv)>>(std::move(compressedSpirv));
    spirv = ICPUBuffer::create({ { propsSize + destLen }, memoryResource->getBacker().data(),std::move(memoryResource)}, core::adopt_memory);

    return true;
}

core::smart_refctd_ptr<IShader> nbl::asset::IShaderCompiler::CCache::SEntry::decompressShader() const
{
    auto uncompressedBuf = ICPUBuffer::create({ uncompressedSize });
    uncompressedBuf->setContentHash(uncompressedContentHash);

    size_t dstSize = uncompressedBuf->getSize();
    size_t srcSize = spirv->getSize() - LZMA_PROPS_SIZE;
    ELzmaStatus status;
    ISzAlloc alloc = { SzAlloc, SzFree };
    SRes res = LzmaDecode(
        reinterpret_cast<unsigned char*>(uncompressedBuf->getPointer()), &dstSize,
        reinterpret_cast<const unsigned char*>(spirv->getPointer()) + LZMA_PROPS_SIZE, &srcSize,
        reinterpret_cast<const unsigned char*>(spirv->getPointer()), LZMA_PROPS_SIZE,
        LZMA_FINISH_ANY, &status, &alloc);
    assert(res == SZ_OK);
    return core::make_smart_refctd_ptr<asset::IShader>(std::move(uncompressedBuf), IShader::E_CONTENT_TYPE::ECT_SPIRV, compilerArgs.preprocessorArgs.sourceIdentifier.data());
}

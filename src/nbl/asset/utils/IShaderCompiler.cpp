// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/IShaderCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#include "nbl/asset/utils/shaderCompiler_serialization.h"
#include "nbl/core/hash/blake.h"

#include <sstream>
#include <regex>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <fstream>

#include <lzma/C/LzmaEnc.h>
#include <lzma/C/LzmaDec.h>

using namespace nbl;
using namespace nbl::asset;

namespace
{
	void splitPrefix(std::string_view code, std::string_view& prefix, std::string_view& body)
	{
		size_t pos = 0;
		size_t prefixEnd = 0;
		bool inContinuation = false;
		bool inBlockComment = false;

		while (pos < code.size())
		{
			const size_t lineStart = pos;
			size_t lineEnd = code.find('\n', pos);
			if (lineEnd == std::string_view::npos)
				lineEnd = code.size();

			std::string_view line = code.substr(lineStart, lineEnd - lineStart);
			if (!line.empty() && line.back() == '\r')
				line.remove_suffix(1);

			bool directiveLine = false;
			if (inContinuation || inBlockComment)
			{
				directiveLine = true;
			}
			else
			{
				size_t i = 0;
				if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
					static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF)
					i = 3;
				while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
					++i;
				if (i == line.size())
				{
					directiveLine = true;
				}
				else if (line[i] == '#')
				{
					directiveLine = true;
				}
				else if (line[i] == '/' && i + 1 < line.size() && line[i + 1] == '/')
				{
					directiveLine = true;
				}
				else if (line[i] == '/' && i + 1 < line.size() && line[i + 1] == '*')
				{
					directiveLine = true;
					if (line.find("*/", i + 2) == std::string_view::npos)
						inBlockComment = true;
				}
			}

			if (!directiveLine)
				break;

			prefixEnd = lineEnd < code.size() ? lineEnd + 1 : lineEnd;

			if (inBlockComment && line.find("*/") != std::string_view::npos)
				inBlockComment = false;

			bool continuation = false;
			if (!line.empty())
			{
				size_t j = line.size();
				while (j > 0 && (line[j - 1] == ' ' || line[j - 1] == '\t'))
					--j;
				if (j > 0 && line[j - 1] == '\\')
					continuation = true;
			}
			inContinuation = continuation;
			if (lineEnd == code.size())
				break;
			pos = lineEnd + 1;
		}

		prefix = code.substr(0, prefixEnd);
		body = code.substr(prefixEnd);
	}
}

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
	const auto* dependencyOverrides = options.dependencyOverrides;

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

	const std::string_view cacheCode = options.preprocessorOptions.codeForCache.empty() ? code : options.preprocessorOptions.codeForCache;
	CCache::SEntry entry;
	if (options.readCache || options.writeCache)
		entry = CCache::SEntry(cacheCode, options);

	if (options.cacheHit)
		*options.cacheHit = false;

	if (options.readCache)
	{
		auto found = options.readCache->find_impl(entry, options.preprocessorOptions.includeFinder);
		if (found != options.readCache->m_container.end())
		{
			if (options.cacheHit)
				*options.cacheHit = true;
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
	if (!dependencyOverrides)
	{
		if (options.writeCache)
			dependenciesPtr = &entry.dependencies;
		else if (depfileEnabled && supportsDependencies)
			dependenciesPtr = &depfileDependencies;
	}

	auto retVal = compileToSPIRV_impl(code, options, dependenciesPtr);
	if (retVal)
	{
		auto backingBuffer = retVal->getContent();
		const_cast<ICPUBuffer*>(backingBuffer)->setContentHash(backingBuffer->computeContentHash());
	}

	if (retVal && options.writeCache && dependencyOverrides)
	{
		entry.dependencies.clear();
		entry.dependencies.reserve(dependencyOverrides->size());
		for (const auto& dep : *dependencyOverrides)
			entry.dependencies.emplace_back(dep.getRequestingSourceDir(), dep.getIdentifier(), dep.isStandardInclude(), dep.getHash());
	}

	if (retVal && depfileEnabled && supportsDependencies)
	{
		const auto* deps = dependencyOverrides ? dependencyOverrides : (options.writeCache ? &entry.dependencies : &depfileDependencies);
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

bool IShaderCompiler::CCache::contains(const SEntry& mainFile, const IShaderCompiler::CIncludeFinder* finder) const
{
    return find_impl(mainFile, finder) != m_container.end();
}

bool IShaderCompiler::CCache::findEntryForCode(std::string_view code, const SCompilerOptions& options, const IShaderCompiler::CIncludeFinder* finder, SEntry& outEntry) const
{
    const std::string_view cacheCode = options.preprocessorOptions.codeForCache.empty() ? code : options.preprocessorOptions.codeForCache;
    const CCache::SEntry entry(cacheCode, options);
    const auto found = find_impl(entry, finder);
    if (found == m_container.end())
        return false;
    outEntry = SEntry(*found);
    return true;
}

core::smart_refctd_ptr<asset::IShader> IShaderCompiler::CCache::decompressEntry(const SEntry& entry) const
{
    return entry.decompressShader();
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

static std::string normalizeLinePath(std::string_view path)
{
    std::string out(path);
    std::replace(out.begin(), out.end(), '\\', '/');
	return out;
}

std::string IShaderCompiler::applyForceIncludes(std::string_view code, std::span<const std::string> forceIncludes)
{
    if (forceIncludes.empty())
        return std::string(code);

    size_t reserveSize = code.size();
    for (const auto& inc : forceIncludes)
        reserveSize += inc.size() + 16;

    std::string out;
    out.reserve(reserveSize);
    for (const auto& inc : forceIncludes)
    {
        const auto incPath = std::filesystem::path(inc).generic_string();
        out.append("#include \"");
        out.append(incPath);
        out.append("\"\n");
    }
    out.append(code);
    return out;
}

bool IShaderCompiler::probeShaderCache(const CCache* cache, std::string_view code, const SCompilerOptions& options, const CIncludeFinder* finder)
{
    if (!cache)
        return false;
    const std::string_view cacheCode = options.preprocessorOptions.codeForCache.empty() ? code : options.preprocessorOptions.codeForCache;
    const CCache::SEntry entry(cacheCode, options);
    return cache->contains(entry, finder);
}

bool IShaderCompiler::preprocessPrefixForCache(std::string_view code, IShader::E_SHADER_STAGE& stage, const SPreprocessorOptions& preprocessOptions, CPreprocessCache::SEntry& outEntry) const
{
    outEntry = {};
    std::vector<CCache::SEntry::SPreprocessingDependency> deps;
    auto text = preprocessShader(std::string(code), stage, preprocessOptions, &deps);
    if (text.empty())
        return false;
    outEntry.preprocessedPrefix = std::move(text);
    outEntry.dependencies = std::move(deps);
    outEntry.pragmaStage = static_cast<uint32_t>(stage);
    return true;
}

IShaderCompiler::CPreprocessCache::SProbeResult IShaderCompiler::CPreprocessCache::probe(std::string_view code, const CPreprocessCache* cache, ELoadStatus loadStatus, const SPreprocessorOptions& preprocessOptions)
{
	SProbeResult result = {};
	const CIncludeFinder* finder = preprocessOptions.includeFinder;
	std::string_view codeToSplit = code;
	if (preprocessOptions.applyForceIncludes && !preprocessOptions.forceIncludes.empty())
	{
		result.codeStorage = applyForceIncludes(code, preprocessOptions.forceIncludes);
		codeToSplit = result.codeStorage;
	}
	splitPrefix(codeToSplit, result.prefix, result.body);
	result.hasPrefix = !result.prefix.empty();
	if (!result.hasPrefix)
	{
		result.status = EProbeStatus::NoPrefix;
		result.cacheHit = false;
		return result;
	}

	{
		core::blake3_hasher hasher;
		hasher.update(result.prefix.data(), result.prefix.size());
		result.prefixHash = static_cast<core::blake3_hash_t>(hasher);
	}
	const bool hasEntry = cache && cache->hasEntry();
	if (!hasEntry)
	{
		result.cacheHit = false;
		if (loadStatus == ELoadStatus::Missing)
			result.status = EProbeStatus::Missing;
		else if (loadStatus == ELoadStatus::Invalid)
			result.status = EProbeStatus::Invalid;
		else
			result.status = EProbeStatus::EntryInvalid;
		return result;
	}

	const bool prefixMatch = cache->getEntry().prefixHash == result.prefixHash;
	const bool depsValid = cache->validateDependencies(finder);
	if (prefixMatch && depsValid)
	{
		result.cacheHit = true;
		result.status = EProbeStatus::Hit;
		return result;
	}

	result.cacheHit = false;
	if (!prefixMatch)
		result.status = EProbeStatus::PrefixChanged;
	else if (!depsValid)
		result.status = EProbeStatus::DependenciesChanged;
	else
		result.status = EProbeStatus::EntryInvalid;

	return result;
}

const char* IShaderCompiler::CPreprocessCache::getProbeReason(EProbeStatus status)
{
	switch (status)
	{
	case EProbeStatus::Missing:
		return "cache file missing; first build, cleaned, output moved, or out of date";
	case EProbeStatus::Invalid:
		return "cache file invalid or version mismatch";
	case EProbeStatus::PrefixChanged:
		return "prefix changed; cache invalidated";
	case EProbeStatus::DependenciesChanged:
		return "dependencies changed; cache invalidated";
	case EProbeStatus::EntryInvalid:
		return "cache entry invalid";
	case EProbeStatus::NoPrefix:
		return "no prefix";
	case EProbeStatus::Hit:
		return "hit";
	default:
		return "unknown";
	}
}

IShaderCompiler::SPreprocessCacheResult IShaderCompiler::preprocessWithCache(std::string_view code, IShader::E_SHADER_STAGE stage, const SPreprocessorOptions& preprocessOptions, CPreprocessCache& cache, CPreprocessCache::ELoadStatus loadStatus, std::string_view sourceIdentifier) const
{
    SPreprocessCacheResult result = {};
    result.stage = stage;

    const auto probe = CPreprocessCache::probe(code, &cache, loadStatus, preprocessOptions);
    result.status = probe.status;
    if (!probe.hasPrefix)
        return result;

    if (probe.cacheHit)
    {
        result.cacheHit = true;
        result.cacheUsed = true;
    }
    else
    {
        CPreprocessCache::SEntry entry;
        IShader::E_SHADER_STAGE prefixStage = stage;
        SPreprocessorOptions preCacheOpt = preprocessOptions;
        preCacheOpt.depfile = false;
        if (!preprocessPrefixForCache(probe.prefix, prefixStage, preCacheOpt, entry))
        {
            result.ok = false;
            return result;
        }
        entry.prefixHash = probe.prefixHash;
        entry.pragmaStage = static_cast<uint32_t>(prefixStage);
        cache.setEntry(std::move(entry));
        result.cacheUsed = true;
        result.cacheUpdated = true;
    }

    if (!cache.hasEntry())
    {
        result.ok = false;
        return result;
    }

    result.code = cache.buildCombinedCode(probe.body, sourceIdentifier);
    if (result.code.empty())
    {
        result.ok = false;
        return result;
    }

    const auto& entry = cache.getEntry();
    if (entry.pragmaStage != static_cast<uint32_t>(IShader::E_SHADER_STAGE::ESS_UNKNOWN))
        result.stage = static_cast<IShader::E_SHADER_STAGE>(entry.pragmaStage);

    return result;
}

core::smart_refctd_ptr<ICPUBuffer> IShaderCompiler::CPreprocessCache::serialize() const
{
    if (!m_hasEntry)
        return nullptr;

    auto write_bytes = [](std::vector<uint8_t>& out, const void* data, size_t size)
    {
        const auto* ptr = reinterpret_cast<const uint8_t*>(data);
        out.insert(out.end(), ptr, ptr + size);
    };
    auto write_u32 = [&write_bytes](std::vector<uint8_t>& out, uint32_t value)
    {
        write_bytes(out, &value, sizeof(value));
    };
    auto write_string = [&write_u32, &write_bytes](std::vector<uint8_t>& out, std::string_view value)
    {
        write_u32(out, static_cast<uint32_t>(value.size()));
        if (!value.empty())
            write_bytes(out, value.data(), value.size());
    };

    std::vector<uint8_t> out;
    out.reserve(m_entry.preprocessedPrefix.size() + 256);
    const uint32_t magic = 0x50435250u;
    write_u32(out, magic);
    write_string(out, VERSION);
    write_bytes(out, &m_entry.prefixHash, sizeof(m_entry.prefixHash));
    write_u32(out, m_entry.pragmaStage);
    write_string(out, m_entry.preprocessedPrefix);

    write_u32(out, static_cast<uint32_t>(m_entry.macroDefs.size()));
    for (const auto& macro : m_entry.macroDefs)
        write_string(out, macro);

    write_u32(out, static_cast<uint32_t>(m_entry.dxcFlags.size()));
    for (const auto& flag : m_entry.dxcFlags)
        write_string(out, flag);

    write_u32(out, static_cast<uint32_t>(m_entry.dependencies.size()));
    for (const auto& dep : m_entry.dependencies)
    {
        const auto dir = dep.getRequestingSourceDir().generic_string();
        write_string(out, dir);
        write_string(out, dep.getIdentifier());
        const uint8_t standardInclude = dep.isStandardInclude() ? 1u : 0u;
        write_bytes(out, &standardInclude, sizeof(standardInclude));
        write_bytes(out, dep.getHash().data, sizeof(dep.getHash().data));
    }

    auto buffer = ICPUBuffer::create({ out.size() });
    if (!buffer)
        return nullptr;
    std::memcpy(buffer->getPointer(), out.data(), out.size());
    return buffer;
}

core::smart_refctd_ptr<IShaderCompiler::CPreprocessCache> IShaderCompiler::CPreprocessCache::deserialize(const std::span<const uint8_t> serializedCache)
{
    if (serializedCache.empty())
        return nullptr;

    auto read_bytes = [](const std::span<const uint8_t> data, size_t& offset, void* dst, size_t size) -> bool
    {
        if (offset + size > data.size())
            return false;
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
        return true;
    };
    auto read_u32 = [&read_bytes](const std::span<const uint8_t> data, size_t& offset, uint32_t& out) -> bool
    {
        return read_bytes(data, offset, &out, sizeof(out));
    };
    auto read_string = [&read_u32, &read_bytes](const std::span<const uint8_t> data, size_t& offset, std::string& out) -> bool
    {
        uint32_t size = 0;
        if (!read_u32(data, offset, size))
            return false;
        if (offset + size > data.size())
            return false;
        out.assign(reinterpret_cast<const char*>(data.data() + offset), size);
        offset += size;
        return true;
    };

    size_t offset = 0;
    uint32_t magic = 0;
    if (!read_u32(serializedCache, offset, magic))
        return nullptr;
    if (magic != 0x50435250u)
        return nullptr;

    std::string version;
    if (!read_string(serializedCache, offset, version))
        return nullptr;
    if (version != VERSION)
        return nullptr;

    auto retVal = core::make_smart_refctd_ptr<CPreprocessCache>();
    auto& entry = retVal->m_entry;
    if (!read_bytes(serializedCache, offset, &entry.prefixHash, sizeof(entry.prefixHash)))
        return nullptr;
    if (!read_u32(serializedCache, offset, entry.pragmaStage))
        return nullptr;
    if (!read_string(serializedCache, offset, entry.preprocessedPrefix))
        return nullptr;

    uint32_t macroCount = 0;
    if (!read_u32(serializedCache, offset, macroCount))
        return nullptr;
    entry.macroDefs.clear();
    entry.macroDefs.reserve(macroCount);
    for (uint32_t i = 0; i < macroCount; ++i)
    {
        std::string macro;
        if (!read_string(serializedCache, offset, macro))
            return nullptr;
        entry.macroDefs.emplace_back(std::move(macro));
    }

    uint32_t flagCount = 0;
    if (!read_u32(serializedCache, offset, flagCount))
        return nullptr;
    entry.dxcFlags.clear();
    entry.dxcFlags.reserve(flagCount);
    for (uint32_t i = 0; i < flagCount; ++i)
    {
        std::string flag;
        if (!read_string(serializedCache, offset, flag))
            return nullptr;
        entry.dxcFlags.emplace_back(std::move(flag));
    }

    uint32_t depCount = 0;
    if (!read_u32(serializedCache, offset, depCount))
        return nullptr;
    entry.dependencies.clear();
    entry.dependencies.reserve(depCount);
    for (uint32_t i = 0; i < depCount; ++i)
    {
        std::string dir;
        std::string identifier;
        if (!read_string(serializedCache, offset, dir))
            return nullptr;
        if (!read_string(serializedCache, offset, identifier))
            return nullptr;
        uint8_t standardInclude = 0;
        if (!read_bytes(serializedCache, offset, &standardInclude, sizeof(standardInclude)))
            return nullptr;
        core::blake3_hash_t hash = {};
        if (!read_bytes(serializedCache, offset, hash.data, sizeof(hash.data)))
            return nullptr;
        entry.dependencies.emplace_back(system::path(dir), identifier, standardInclude != 0, hash);
    }

    retVal->m_hasEntry = true;
    return retVal;
}

core::smart_refctd_ptr<IShaderCompiler::CPreprocessCache> IShaderCompiler::CPreprocessCache::loadFromFile(const system::path& path, ELoadStatus& status)
{
    status = ELoadStatus::Missing;
    if (!std::filesystem::exists(path))
        return nullptr;

    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    in.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    if (!size)
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    std::vector<uint8_t> data(size);
    if (!in.read(reinterpret_cast<char*>(data.data()), data.size()))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    auto cache = deserialize(std::span<const uint8_t>(data.data(), data.size()));
    if (!cache)
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    status = ELoadStatus::Loaded;
    return cache;
}

bool IShaderCompiler::CPreprocessCache::writeToFile(const system::path& path, const CPreprocessCache& cache)
{
    auto buffer = cache.serialize();
    if (!buffer)
        return false;

    const auto parent = path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent))
        std::filesystem::create_directories(parent);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;

    out.write(reinterpret_cast<const char*>(buffer->getPointer()), buffer->getSize());
    return bool(out);
}

bool IShaderCompiler::CPreprocessCache::validateDependencies(const CIncludeFinder* finder) const
{
    if (!m_hasEntry || !finder)
        return false;

    for (const auto& dep : m_entry.dependencies)
    {
        IIncludeLoader::found_t header;
        if (dep.isStandardInclude())
            header = finder->getIncludeStandard(dep.getRequestingSourceDir(), std::string(dep.getIdentifier()));
        else
            header = finder->getIncludeRelative(dep.getRequestingSourceDir(), std::string(dep.getIdentifier()));

        if (!header || header.hash != dep.getHash())
            return false;
    }
    return true;
}

std::string IShaderCompiler::CPreprocessCache::buildCombinedCode(std::string_view body, std::string_view sourceIdentifier) const
{
    if (!m_hasEntry)
        return std::string(body);

    std::string out;
    size_t reserve = m_entry.preprocessedPrefix.size() + body.size();
    for (const auto& m : m_entry.macroDefs)
        reserve += m.size() + 16;
    for (const auto& f : m_entry.dxcFlags)
        reserve += f.size() + 1;
    reserve += 64;
    out.reserve(reserve);

    if (!m_entry.dxcFlags.empty())
    {
        out.append("#pragma dxc_compile_flags ");
        for (size_t i = 0; i < m_entry.dxcFlags.size(); ++i)
        {
            if (i)
                out.push_back(' ');
            out.append(m_entry.dxcFlags[i]);
        }
        out.push_back('\n');
    }

    if (!m_entry.preprocessedPrefix.empty())
    {
        out.append(m_entry.preprocessedPrefix);
        if (out.back() != '\n')
            out.push_back('\n');
    }

    for (const auto& macro : m_entry.macroDefs)
    {
        const auto eq = macro.find('=');
        std::string_view name = eq == std::string::npos ? std::string_view(macro) : std::string_view(macro).substr(0, eq);
        std::string_view def = eq == std::string::npos ? std::string_view() : std::string_view(macro).substr(eq + 1);
        out.append("#define ");
        out.append(name);
        if (!def.empty())
        {
            out.push_back(' ');
            out.append(def);
        }
        out.push_back('\n');
    }

    if (!sourceIdentifier.empty())
    {
        out.append("#line 1 \"");
        out.append(normalizeLinePath(sourceIdentifier));
        out.append("\"\n");
    }

    out.append(body);
    return out;
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

// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/IShaderCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#include "nbl/asset/utils/shaderCompiler_serialization.h"
#include "nbl/core/hash/blake.h"
#include "nbl/core/hash/xxHash256.h"
#include <sstream>
#include <regex>
#include <iterator>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <cstring>
#include <array>
#include <atomic>
#include <mutex>
#include <thread>

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/builtin/CArchive.h"
#include "spirv/builtin/CArchive.h"
#include "boost/builtin/CArchive.h"
#include "nbl/devicegen/builtin/CArchive.h"
#endif

#ifdef _WIN32
#include <Windows.h>
#endif

#include <lzma/C/LzmaEnc.h>
#include <lzma/C/LzmaDec.h>

namespace
{
struct FileInfoCacheEntry
{
    uint64_t size = 0;
    int64_t ticks = 0;
    bool ok = false;
};

std::unordered_map<nbl::system::path, FileInfoCacheEntry> g_fileInfoCache;
std::mutex g_fileInfoCacheMutex;

struct IncludeCacheEntry
{
    uint64_t size = 0;
    int64_t ticks = 0;
    nbl::core::blake3_hash_t hash = {};
    std::string contents;
};

std::unordered_map<nbl::system::path, IncludeCacheEntry> g_includeCache;
std::mutex g_includeCacheMutex;

#ifdef NBL_EMBED_BUILTIN_RESOURCES
inline bool tryGetBuiltinResource(const std::string& normalized, const nbl::system::SBuiltinFile*& outFile, std::string& outRel, std::string_view& outPrefix)
{
    auto tryNamespace = [&](std::string_view prefix, const nbl::system::SBuiltinFile& (*getResource)(const std::string&)) -> bool
    {
        if (normalized.rfind(prefix, 0) != 0)
            return false;
        std::string rel = normalized.substr(prefix.size());
        if (!rel.empty() && (rel.front() == '/' || rel.front() == '\\'))
            rel.erase(rel.begin());
        const auto& resource = getResource(rel);
        if (!resource.contents || resource.size == 0)
            return false;
        outFile = &resource;
        outRel = std::move(rel);
        outPrefix = prefix;
        return true;
    };

    if (tryNamespace(nbl::builtin::pathPrefix, nbl::builtin::get_resource_runtime))
        return true;
    if (tryNamespace(spirv::builtin::pathPrefix, spirv::builtin::get_resource_runtime))
        return true;
    if (tryNamespace(boost::builtin::pathPrefix, boost::builtin::get_resource_runtime))
        return true;
    if (tryNamespace(nbl::devicegen::builtin::pathPrefix, nbl::devicegen::builtin::get_resource_runtime))
        return true;

    return false;
}

inline bool tryGetBuiltinResourceHash(const nbl::system::path& path, nbl::core::blake3_hash_t& outHash)
{
    if (path.empty())
        return false;
    const std::string normalized = path.generic_string();
    const nbl::system::SBuiltinFile* resource = nullptr;
    std::string rel;
    std::string_view prefix;
    if (!tryGetBuiltinResource(normalized, resource, rel, prefix))
        return false;
    std::memcpy(outHash.data, resource->xx256Hash.data(), sizeof(outHash.data));
    return true;
}

inline bool matchBuiltinResourceHash(const nbl::system::path& path, const nbl::core::blake3_hash_t& expected)
{
    nbl::core::blake3_hash_t hash = {};
    if (!tryGetBuiltinResourceHash(path, hash))
        return false;
    return hash == expected;
}

class CBuiltinArchiveIncludeLoader final : public nbl::asset::IShaderCompiler::IIncludeLoader
{
    public:
        using IIncludeLoader = nbl::asset::IShaderCompiler::IIncludeLoader;

        IIncludeLoader::found_t getInclude(const nbl::system::path& searchPath, const std::string& includeName) const override
        {
            std::string normalized = nbl::system::path(includeName).generic_string();
            if (!searchPath.empty())
            {
                const std::string search = nbl::system::path(searchPath).generic_string();
                if (normalized.rfind(search, 0) != 0)
                    normalized = (nbl::system::path(search) / includeName).generic_string();
            }

            const nbl::system::SBuiltinFile* resource = nullptr;
            std::string rel;
            std::string_view prefix;
            if (!tryGetBuiltinResource(normalized, resource, rel, prefix))
                return {};

            IIncludeLoader::found_t ret = {};
            ret.absolutePath = nbl::system::path(std::string(prefix)) / rel;
            ret.contents.assign(reinterpret_cast<const char*>(resource->contents), resource->size);
            if (!ret.contents.empty() && ret.contents.back() != '\n' && ret.contents.back() != '\r')
                ret.contents.push_back('\n');
            std::memcpy(ret.hash.data, resource->xx256Hash.data(), sizeof(ret.hash.data));
            ret.hasHash = true;
            ret.fileSize = resource->size;
            ret.hasFileInfo = false;
            return ret;
        }
};
#endif

inline bool getFileInfoFast(const nbl::system::path& path, uint64_t& sizeOut, int64_t& timeOut)
{
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA data = {};
    if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &data))
        return false;
    ULARGE_INTEGER size = {};
    size.HighPart = data.nFileSizeHigh;
    size.LowPart = data.nFileSizeLow;
    ULARGE_INTEGER time = {};
    time.HighPart = data.ftLastWriteTime.dwHighDateTime;
    time.LowPart = data.ftLastWriteTime.dwLowDateTime;
    sizeOut = size.QuadPart;
    using file_clock = std::chrono::file_clock;
    const auto duration = file_clock::duration{ static_cast<file_clock::rep>(time.QuadPart) };
    const auto fileTp = std::chrono::time_point<file_clock>{ duration };
    const auto utcTp = std::chrono::clock_cast<nbl::system::IFileBase::time_point_t::clock>(fileTp);
    timeOut = utcTp.time_since_epoch().count();
    return true;
#else
    std::error_code ec;
    std::filesystem::directory_entry entry(path, ec);
    if (ec)
        return false;
    const auto time = entry.last_write_time(ec);
    if (ec)
        return false;
    const auto size = entry.file_size(ec);
    if (ec)
        return false;
    sizeOut = size;
    const auto utcTp = std::chrono::clock_cast<nbl::system::IFileBase::time_point_t::clock>(time);
    timeOut = utcTp.time_since_epoch().count();
    return true;
#endif
}

inline bool getFileInfoFast(const nbl::system::path& path, uint64_t& sizeOut, int64_t& timeOut, nbl::system::ISystem* system)
{
    if (getFileInfoFast(path, sizeOut, timeOut))
        return true;
    if (!system || path.empty())
        return false;

    nbl::system::ISystem::future_t<nbl::core::smart_refctd_ptr<nbl::system::IFile>> future;
    system->createFile(future, path, nbl::system::IFile::ECF_READ);
    if (!future.wait())
        return false;
    nbl::core::smart_refctd_ptr<nbl::system::IFile> file;
    if (auto lock = future.acquire(); lock)
        lock.move_into(file);
    if (!file)
        return false;
    sizeOut = file->getSize();
    timeOut = file->getLastWriteTime().time_since_epoch().count();
    return true;
}

inline bool getFileInfoCached(const nbl::system::path& path, uint64_t& sizeOut, int64_t& timeOut, nbl::system::ISystem* system)
{
    if (path.empty())
        return false;

    {
        std::lock_guard<std::mutex> lock(g_fileInfoCacheMutex);
        const auto it = g_fileInfoCache.find(path);
        if (it != g_fileInfoCache.end())
        {
            if (!it->second.ok)
                return false;
            sizeOut = it->second.size;
            timeOut = it->second.ticks;
            return true;
        }
    }

    uint64_t size = 0;
    int64_t ticks = 0;
    const bool ok = getFileInfoFast(path, size, ticks, system);
    {
        std::lock_guard<std::mutex> lock(g_fileInfoCacheMutex);
        g_fileInfoCache.emplace(path, FileInfoCacheEntry{ size, ticks, ok });
    }
    if (!ok)
        return false;
    sizeOut = size;
    timeOut = ticks;
    return true;
}

template<typename DepContainer>
inline void collectFileInfoMismatchesParallel(const DepContainer& deps, std::vector<size_t>& out, nbl::system::ISystem* system)
{
    const size_t count = deps.size();
    if (!count)
        return;

    std::vector<size_t> fileInfoIndices;
    fileInfoIndices.reserve(count);
    std::unordered_map<nbl::system::path, bool> seenPaths;
    seenPaths.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        const auto& dep = deps[i];
        const auto& path = dep.getAbsolutePath();
        const bool hasAbsolutePath = !path.empty() && path.is_absolute();
        const bool hasFileInfo = dep.getHasFileInfo() && hasAbsolutePath;
        if (!hasFileInfo)
        {
#ifdef NBL_EMBED_BUILTIN_RESOURCES
            if (!path.empty())
            {
                if (matchBuiltinResourceHash(path, dep.getHash()))
                    continue;
            }
            else
            {
                const nbl::system::path logicalPath(dep.getIdentifier());
                if (matchBuiltinResourceHash(logicalPath, dep.getHash()))
                    continue;
            }
#endif
            out.push_back(i);
            continue;
        }
        if (seenPaths.emplace(path, true).second)
            fileInfoIndices.push_back(i);
    }

    const size_t fileCount = fileInfoIndices.size();
    if (!fileCount)
        return;

    unsigned threads = std::thread::hardware_concurrency();
    if (!threads)
        threads = 1u;
    if (threads > 32u)
        threads = 32u;
    if (threads > fileCount)
        threads = static_cast<unsigned>(fileCount);

    if (threads <= 1u || fileCount < 32u)
    {
        for (size_t k = 0; k < fileCount; ++k)
        {
            const size_t i = fileInfoIndices[k];
            const auto& dep = deps[i];
            const auto& path = dep.getAbsolutePath();
            uint64_t size = 0;
            int64_t ticks = 0;
            if (path.empty() || !getFileInfoFast(path, size, ticks) || dep.getLastWriteTime() != ticks || dep.getFileSize() != size)
                out.push_back(i);
        }
        return;
    }

    std::vector<std::vector<size_t>> perThread(threads);
    const size_t chunk = (fileCount + threads - 1u) / threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (unsigned t = 0; t < threads; ++t)
    {
        const size_t start = t * chunk;
        if (start >= fileCount)
            break;
        const size_t end = std::min(start + chunk, fileCount);
        workers.emplace_back([&deps, &perThread, &fileInfoIndices, t, start, end, system]()
        {
            auto& local = perThread[t];
            for (size_t k = start; k < end; ++k)
            {
                const size_t i = fileInfoIndices[k];
                const auto& dep = deps[i];
                const auto& path = dep.getAbsolutePath();
                uint64_t size = 0;
                int64_t ticks = 0;
                if (path.empty() || !getFileInfoFast(path, size, ticks) || dep.getLastWriteTime() != ticks || dep.getFileSize() != size)
                    local.push_back(i);
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    for (auto& local : perThread)
        out.insert(out.end(), local.begin(), local.end());
}
}

using namespace nbl;
using namespace nbl::asset;

namespace
{
	std::string buildMacroBlock(const std::vector<std::string>& macros)
	{
		if (macros.empty())
			return {};
		size_t reserve = 0;
		for (const auto& macro : macros)
			reserve += macro.size() + 12;
		std::string out;
		out.reserve(reserve);
		for (const auto& macro : macros)
		{
			const size_t eq = macro.find('=');
			const std::string_view name = eq == std::string::npos ? std::string_view(macro) : std::string_view(macro).substr(0, eq);
			const std::string_view def = eq == std::string::npos ? std::string_view() : std::string_view(macro).substr(eq + 1);
			out.append("#define ");
			out.append(name);
			if (!def.empty())
			{
				out.push_back(' ');
				out.append(def);
			}
			out.push_back('\n');
		}
		return out;
	}

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

core::smart_refctd_ptr<IShader> nbl::asset::IShaderCompiler::compileToSPIRV(const std::string_view code, const SCompilerOptions& options) const
{
	const auto* dependencyOverrides = options.dependencyOverrides;

	const std::string_view cacheCode = options.preprocessorOptions.codeForCache.empty() ? code : options.preprocessorOptions.codeForCache;
	CCache::SEntry entry;
	if (options.readCache || options.writeCache)
		entry = CCache::SEntry(cacheCode, options);

	if (options.cacheHit)
		*options.cacheHit = false;

	if (options.readCache)
	{
		auto found = options.readCache->find_impl(entry, options.preprocessorOptions.includeFinder, true, nullptr);
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
			return shader;
		}
	}

	CCache::SEntry::dependency_container_t* dependenciesPtr = nullptr;
	if (!dependencyOverrides)
	{
		if (options.writeCache)
			dependenciesPtr = &entry.dependencies;
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
			entry.dependencies.emplace_back(dep.getRequestingSourceDir(), dep.getIdentifier(), dep.isStandardInclude(), dep.getHash(), dep.getAbsolutePath(), dep.getFileSize(), dep.getLastWriteTime(), dep.getHasFileInfo());
	}

	if (options.writeCache)
	{
		entry.compression = options.writeCache->getDefaultCompression();
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

    uint64_t fileSize = 0;
    int64_t lastWriteTime = 0;
    const bool infoOk = getFileInfoFast(path, fileSize, lastWriteTime, m_system.get());
    if (infoOk)
    {
        std::lock_guard<std::mutex> lock(g_includeCacheMutex);
        auto it = g_includeCache.find(path);
        if (it != g_includeCache.end() && it->second.size == fileSize && it->second.ticks == lastWriteTime)
        {
            found_t ret = {};
            ret.absolutePath = path;
            ret.contents = it->second.contents;
            ret.hash = it->second.hash;
            ret.hasHash = true;
            ret.fileSize = fileSize;
            ret.lastWriteTime = lastWriteTime;
            ret.hasFileInfo = true;
            return ret;
        }
    }

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
    if (!contents.empty() && contents.back() != '\n' && contents.back() != '\r')
        contents.push_back('\n');

    found_t ret = {};
    ret.absolutePath = path;
    ret.contents = std::move(contents);
    if (auto precomputed = f->getPrecomputedHash())
    {
        static_assert(sizeof(ret.hash.data) == sizeof(*precomputed));
        std::memcpy(ret.hash.data, &(*precomputed), sizeof(ret.hash.data));
        ret.hasHash = true;
        ret.hasFileInfo = false;
    }
    else
    {
        ret.fileSize = infoOk ? fileSize : size;
        ret.lastWriteTime = infoOk ? lastWriteTime : f->getLastWriteTime().time_since_epoch().count();
        ret.hasFileInfo = true;
    }
    if (!ret.hasHash)
    {
        std::array<uint64_t, 4> hash = {};
        core::XXHash_256(ret.contents.data(), ret.contents.size(), hash.data());
        std::memcpy(ret.hash.data, hash.data(), sizeof(ret.hash.data));
        ret.hasHash = true;
    }
    if (infoOk)
    {
        IncludeCacheEntry entry = {};
        entry.size = fileSize;
        entry.ticks = lastWriteTime;
        entry.hash = ret.hash;
        entry.contents = ret.contents;
        std::lock_guard<std::mutex> lock(g_includeCacheMutex);
        g_includeCache[path] = std::move(entry);
    }
    return ret;
}

IShaderCompiler::CIncludeFinder::CIncludeFinder(core::smart_refctd_ptr<system::ISystem>&& system)
    : m_defaultFileSystemLoader(core::make_smart_refctd_ptr<CFileSystemIncludeLoader>(core::smart_refctd_ptr(system)))
    , m_system(std::move(system))
{
#ifdef NBL_EMBED_BUILTIN_RESOURCES
    auto builtinLoader = core::make_smart_refctd_ptr<CBuiltinArchiveIncludeLoader>();
    addSearchPath(std::string(nbl::builtin::pathPrefix), builtinLoader);
    addSearchPath(std::string(spirv::builtin::pathPrefix), builtinLoader);
    addSearchPath(std::string(boost::builtin::pathPrefix), builtinLoader);
    addSearchPath(std::string(nbl::devicegen::builtin::pathPrefix), builtinLoader);
#endif
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

    if (retVal.fileSize == 0 && !retVal.contents.empty())
        retVal.fileSize = retVal.contents.size();
    if (!retVal.hasFileInfo && !retVal.absolutePath.empty() && !retVal.hasHash)
    {
        std::error_code ec;
        const auto fileTime = std::filesystem::last_write_time(retVal.absolutePath, ec);
        if (!ec)
        {
            retVal.lastWriteTime = fileTime.time_since_epoch().count();
            retVal.hasFileInfo = true;
        }
    }

    if (!retVal.hasHash)
    {
        std::array<uint64_t, 4> hash = {};
        core::XXHash_256(retVal.contents.data(), retVal.contents.size(), hash.data());
        std::memcpy(retVal.hash.data, hash.data(), sizeof(retVal.hash.data));
        retVal.hasHash = true;
    }
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

    if (retVal.fileSize == 0 && !retVal.contents.empty())
        retVal.fileSize = retVal.contents.size();
    if (!retVal.hasFileInfo && !retVal.absolutePath.empty() && !retVal.hasHash)
    {
        std::error_code ec;
        const auto fileTime = std::filesystem::last_write_time(retVal.absolutePath, ec);
        if (!ec)
        {
            retVal.lastWriteTime = fileTime.time_since_epoch().count();
            retVal.hasFileInfo = true;
        }
    }

    if (!retVal.hasHash)
    {
        std::array<uint64_t, 4> hash = {};
        core::XXHash_256(retVal.contents.data(), retVal.contents.size(), hash.data());
        std::memcpy(retVal.hash.data, hash.data(), sizeof(retVal.hash.data));
        retVal.hasHash = true;
    }
    return retVal;
}

void IShaderCompiler::CIncludeFinder::addSearchPath(const std::string& searchPath, const core::smart_refctd_ptr<IIncludeLoader>& loader)
{
    if (!loader)
        return;
    if (searchPath.empty())
    {
        m_loaders.emplace_back(LoaderSearchPath{ loader, searchPath });
        return;
    }
    const auto insertPos = std::find_if(m_loaders.begin(), m_loaders.end(), [](const LoaderSearchPath& entry)
        {
            return entry.searchPath.empty();
        });
    m_loaders.insert(insertPos, LoaderSearchPath{ loader, searchPath });
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
    const auto found = find_impl(mainFile, finder, true, nullptr);
    if (found==m_container.end())
        return nullptr;
    return found->decompressShader();
}

bool IShaderCompiler::CCache::contains(const SEntry& mainFile, const IShaderCompiler::CIncludeFinder* finder) const
{
    return find_impl(mainFile, finder, true, nullptr) != m_container.end();
}

bool IShaderCompiler::CCache::findEntryForCode(std::string_view code, const SCompilerOptions& options, const IShaderCompiler::CIncludeFinder* finder, SEntry& outEntry, bool validateDependencies, bool* depsUpdated) const
{
    const std::string_view cacheCode = options.preprocessorOptions.codeForCache.empty() ? code : options.preprocessorOptions.codeForCache;
    const CCache::SEntry entry(cacheCode, options);
    const auto found = find_impl(entry, finder, validateDependencies, depsUpdated);
    if (found == m_container.end())
        return false;
    outEntry = SEntry(*found);
    return true;
}

core::smart_refctd_ptr<asset::IShader> IShaderCompiler::CCache::decompressEntry(const SEntry& entry) const
{
    return entry.decompressShader();
}

IShaderCompiler::CCache::EntrySet::const_iterator IShaderCompiler::CCache::find_impl(const SEntry& mainFile, const IShaderCompiler::CIncludeFinder* finder, bool validateDependencies, bool* depsUpdated) const
{
    auto found = m_container.find(mainFile);
    if (found == m_container.end() || !validateDependencies)
        return found;
    if (depsUpdated)
        *depsUpdated = false;
    bool updated = false;
    auto* system = finder ? finder->getSystem() : nullptr;
    // go through all dependencies
    if (found!=m_container.end())
    {
        std::vector<size_t> mismatches;
        mismatches.reserve(found->dependencies.size());
        collectFileInfoMismatchesParallel(found->dependencies, mismatches, system);
        if (mismatches.empty())
            return found;
        if (!finder)
            return m_container.end();

        std::unordered_map<system::path, bool> fileStatus;
        std::unordered_map<std::string, bool> logicalStatus;
        fileStatus.reserve(mismatches.size());
        logicalStatus.reserve(mismatches.size());
        for (size_t idx : mismatches)
        {
            const auto& dependency = found->dependencies[idx];
            auto makeLogicalKey = [&dependency]()
            {
                std::string key;
                key.reserve(dependency.getIdentifier().size() + dependency.getRequestingSourceDir().string().size() + 4);
                key.append(dependency.getRequestingSourceDir().string());
                key.push_back('|');
                key.append(dependency.getIdentifier());
                key.push_back('|');
                key.push_back(dependency.isStandardInclude() ? '1' : '0');
                return key;
            };

            if (dependency.getHasFileInfo() && !dependency.getAbsolutePath().empty())
            {
                if (auto it = fileStatus.find(dependency.getAbsolutePath()); it != fileStatus.end())
                {
                    if (!it->second)
                        return m_container.end();
                    continue;
                }
            }
            else
            {
                auto key = makeLogicalKey();
                if (auto it = logicalStatus.find(key); it != logicalStatus.end())
                {
                    if (!it->second)
                        return m_container.end();
                    continue;
                }
            }

            bool valid = false;
            bool precomputedChecked = false;
            if (!dependency.getAbsolutePath().empty())
            {
                if (auto* system = finder->getSystem())
                {
                    system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
                    system->createFile(future, dependency.getAbsolutePath(), system::IFile::ECF_READ);
                    if (future.wait())
                    {
                        core::smart_refctd_ptr<system::IFile> file;
                        if (auto lock = future.acquire(); lock)
                            lock.move_into(file);
                        if (file)
                        {
                            if (auto precomputed = file->getPrecomputedHash())
                            {
                                precomputedChecked = true;
                                core::blake3_hash_t hash = {};
                                std::memcpy(hash.data, &(*precomputed), sizeof(hash.data));
                                if (hash == dependency.getHash())
                                    valid = true;
                                else
                                {
                                    if (dependency.getHasFileInfo() && !dependency.getAbsolutePath().empty())
                                        fileStatus.emplace(dependency.getAbsolutePath(), false);
                                    else
                                        logicalStatus.emplace(makeLogicalKey(), false);
                                    return m_container.end();
                                }
                            }
                        }
                    }
                }
            }

            if (!valid && !precomputedChecked)
            {
                IIncludeLoader::found_t header;
                if (dependency.standardInclude)
                    header = finder->getIncludeStandard(dependency.requestingSourceDir, dependency.identifier);
                else
                    header = finder->getIncludeRelative(dependency.requestingSourceDir, dependency.identifier);

                if (header.hash != dependency.hash)
                {
                    if (dependency.getHasFileInfo() && !dependency.getAbsolutePath().empty())
                        fileStatus.emplace(dependency.getAbsolutePath(), false);
                    else
                        logicalStatus.emplace(makeLogicalKey(), false);
                    return m_container.end();
                }

                valid = true;
                if (header.hasFileInfo && dependency.getAbsolutePath().is_absolute())
                {
                    dependency.setFileInfo(header.fileSize, header.lastWriteTime, true);
                    updated = true;
                }
            }

            if (valid && dependency.getHasFileInfo() && dependency.getAbsolutePath().is_absolute())
            {
                uint64_t size = 0;
                int64_t ticks = 0;
                if (getFileInfoCached(dependency.getAbsolutePath(), size, ticks, system) &&
                    (dependency.getFileSize() != size || dependency.getLastWriteTime() != ticks))
                {
                    dependency.setFileInfo(size, ticks, true);
                    updated = true;
                }
            }

            if (dependency.getHasFileInfo() && !dependency.getAbsolutePath().empty())
                fileStatus.emplace(dependency.getAbsolutePath(), true);
            else
                logicalStatus.emplace(makeLogicalKey(), true);
        }
    }

    if (depsUpdated)
        *depsUpdated = updated;
    return found;
}

core::smart_refctd_ptr<ICPUBuffer> IShaderCompiler::CCache::serialize() const
{
    size_t shaderBufferSize = 0;
    core::vector<size_t> offsets(m_container.size());
    core::vector<uint64_t> sizes(m_container.size());
    json entries = json::array();
    core::vector<CPUShaderCreationParams> shaderCreationParams;
    std::vector<uint8_t> depsBuffer;
    depsBuffer.reserve(m_container.size() * 64u);

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

    write_u32(depsBuffer, static_cast<uint32_t>(m_container.size()));
    size_t i = 0u;
    for (auto& entry : m_container) {
        json entryJson{
            { "mainFileContents", entry.mainFileContents },
            { "compilerArgs", entry.compilerArgs },
            { "hash", entry.hash.data },
            { "lookupHash", entry.lookupHash },
            { "uncompressedContentHash", entry.uncompressedContentHash.data },
            { "uncompressedSize", entry.uncompressedSize },
            { "compression", static_cast<uint32_t>(entry.compression) },
        };
        entries.emplace_back(std::move(entryJson));

        offsets[i] = shaderBufferSize;
        sizes[i] = entry.spirv->getSize();
        shaderCreationParams.emplace_back(entry.compilerArgs.stage, entry.compilerArgs.preprocessorArgs.sourceIdentifier.data(), sizes[i], shaderBufferSize);
        shaderBufferSize += sizes[i];

        write_u32(depsBuffer, static_cast<uint32_t>(entry.dependencies.size()));
        for (const auto& dep : entry.dependencies)
        {
            const auto dir = dep.getRequestingSourceDir().generic_string();
            write_string(depsBuffer, dir);
            write_string(depsBuffer, dep.getIdentifier());
            const auto abs = dep.getAbsolutePath().generic_string();
            write_string(depsBuffer, abs);
            const uint8_t standardInclude = dep.isStandardInclude() ? 1u : 0u;
            write_bytes(depsBuffer, &standardInclude, sizeof(standardInclude));
            write_bytes(depsBuffer, dep.getHash().data, sizeof(dep.getHash().data));
            const uint64_t fileSize = dep.getFileSize();
            write_bytes(depsBuffer, &fileSize, sizeof(fileSize));
            const int64_t lastWriteTime = dep.getLastWriteTime();
            write_bytes(depsBuffer, &lastWriteTime, sizeof(lastWriteTime));
            const uint8_t hasFileInfo = dep.getHasFileInfo() ? 1u : 0u;
            write_bytes(depsBuffer, &hasFileInfo, sizeof(hasFileInfo));
        }
        i++;
    }

    json containerJson{
        { "version", VERSION },
        { "entries", std::move(entries) },
        { "shaderCreationParams", std::move(shaderCreationParams) },
    };
    std::string dumpedContainerJson = std::move(containerJson.dump());
    uint64_t dumpedContainerJsonLength = dumpedContainerJson.size();

    size_t retValSize = shaderBufferSize + SHADER_BUFFER_SIZE_BYTES + sizeof(uint64_t) + dumpedContainerJsonLength + depsBuffer.size();
    core::vector<uint8_t> retVal(retValSize);

    memcpy(retVal.data(), &shaderBufferSize, SHADER_BUFFER_SIZE_BYTES);
    memcpy(retVal.data() + SHADER_BUFFER_SIZE_BYTES, &dumpedContainerJsonLength, sizeof(uint64_t));

    i = 0u;
    const size_t shaderOffset = SHADER_BUFFER_SIZE_BYTES + sizeof(uint64_t);
    for (auto& entry : m_container) {
        memcpy(retVal.data() + shaderOffset + offsets[i], entry.spirv->getPointer(), sizes[i]);
        i++;
    }

    const size_t jsonOffset = shaderOffset + shaderBufferSize;
    memcpy(retVal.data() + jsonOffset, dumpedContainerJson.data(), dumpedContainerJsonLength);
    if (!depsBuffer.empty())
        memcpy(retVal.data() + jsonOffset + dumpedContainerJsonLength, depsBuffer.data(), depsBuffer.size());

    auto memoryResource = core::make_smart_refctd_ptr<core::adoption_memory_resource<decltype(retVal)>>(std::move(retVal));
    return ICPUBuffer::create({ { retValSize }, memoryResource->getBacker().data(),std::move(memoryResource)}, core::adopt_memory);
}

core::smart_refctd_ptr<IShaderCompiler::CCache> IShaderCompiler::CCache::deserialize(const std::span<const uint8_t> serializedCache, bool skipDependencies)
{
    auto retVal = core::make_smart_refctd_ptr<CCache>();

    if (serializedCache.size() < SHADER_BUFFER_SIZE_BYTES)
        return nullptr;

    uint64_t shaderBufferSize = 0;
    std::memcpy(&shaderBufferSize, serializedCache.data(), SHADER_BUFFER_SIZE_BYTES);

    const size_t minOldHeader = SHADER_BUFFER_SIZE_BYTES + shaderBufferSize;
    if (serializedCache.size() < minOldHeader)
        return nullptr;

    bool hasBinaryDeps = false;
    uint64_t jsonSize = 0;
    size_t jsonOffset = 0;
    size_t depsOffset = 0;
    size_t shaderOffset = SHADER_BUFFER_SIZE_BYTES;

    const size_t minNewHeader = SHADER_BUFFER_SIZE_BYTES + sizeof(uint64_t) + shaderBufferSize;
    if (serializedCache.size() >= minNewHeader)
    {
        std::memcpy(&jsonSize, serializedCache.data() + SHADER_BUFFER_SIZE_BYTES, sizeof(jsonSize));
        const size_t candidateJsonOffset = SHADER_BUFFER_SIZE_BYTES + sizeof(uint64_t) + shaderBufferSize;
        if (candidateJsonOffset + jsonSize <= serializedCache.size())
        {
            hasBinaryDeps = true;
            jsonOffset = candidateJsonOffset;
            depsOffset = candidateJsonOffset + jsonSize;
            shaderOffset = SHADER_BUFFER_SIZE_BYTES + sizeof(uint64_t);
        }
    }

    if (!hasBinaryDeps)
    {
        jsonOffset = SHADER_BUFFER_SIZE_BYTES + shaderBufferSize;
        jsonSize = serializedCache.size() - jsonOffset;
        shaderOffset = SHADER_BUFFER_SIZE_BYTES;
    }

    std::string_view containerJsonString(reinterpret_cast<const char*>(serializedCache.data() + jsonOffset), jsonSize);
    json containerJson;
    if (skipDependencies)
    {
        bool skipNext = false;
        auto cb = [&skipNext](int, json::parse_event_t event, json& parsed)
        {
            if (event == json::parse_event_t::key && parsed.is_string() && parsed.get_ref<const std::string&>() == "dependencies")
            {
                skipNext = true;
                return true;
            }
            if (skipNext)
            {
                skipNext = false;
                return false;
            }
            return true;
        };
        containerJson = json::parse(containerJsonString, cb, true, true);
    }
    else
    {
        containerJson = json::parse(containerJsonString);
    }

    std::string version;
    containerJson.at("version").get_to(version);
    if (version != VERSION)
        return nullptr;

    std::vector<SEntry> entries;
    std::vector<CPUShaderCreationParams> shaderCreationParams;
    containerJson.at("entries").get_to(entries);
    containerJson.at("shaderCreationParams").get_to(shaderCreationParams);

    for (auto i = 0u; i < entries.size(); i++) {
        auto code = ICPUBuffer::create({ shaderCreationParams[i].codeByteSize });
        memcpy(code->getPointer(), serializedCache.data() + shaderOffset + shaderCreationParams[i].offset, shaderCreationParams[i].codeByteSize);
        code->setContentHash(code->computeContentHash());
        entries[i].spirv = std::move(code);
    }

    if (hasBinaryDeps && !skipDependencies)
    {
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

        size_t offset = depsOffset;
        uint32_t entryCount = 0;
        if (!read_u32(serializedCache, offset, entryCount))
            return nullptr;
        if (entryCount != entries.size())
            return nullptr;

        for (uint32_t i = 0; i < entryCount; ++i)
        {
            uint32_t depCount = 0;
            if (!read_u32(serializedCache, offset, depCount))
                return nullptr;
            entries[i].dependencies.clear();
            entries[i].dependencies.reserve(depCount);
            for (uint32_t d = 0; d < depCount; ++d)
            {
                std::string dir;
                std::string identifier;
                std::string absolutePath;
                if (!read_string(serializedCache, offset, dir))
                    return nullptr;
                if (!read_string(serializedCache, offset, identifier))
                    return nullptr;
                if (!read_string(serializedCache, offset, absolutePath))
                    return nullptr;
                uint8_t standardInclude = 0;
                if (!read_bytes(serializedCache, offset, &standardInclude, sizeof(standardInclude)))
                    return nullptr;
                core::blake3_hash_t hash = {};
                if (!read_bytes(serializedCache, offset, hash.data, sizeof(hash.data)))
                    return nullptr;
                uint64_t fileSize = 0;
                if (!read_bytes(serializedCache, offset, &fileSize, sizeof(fileSize)))
                    return nullptr;
                int64_t lastWriteTime = 0;
                if (!read_bytes(serializedCache, offset, &lastWriteTime, sizeof(lastWriteTime)))
                    return nullptr;
                uint8_t hasFileInfo = 0;
                if (!read_bytes(serializedCache, offset, &hasFileInfo, sizeof(hasFileInfo)))
                    return nullptr;
                entries[i].dependencies.emplace_back(system::path(dir), identifier, standardInclude != 0, hash, system::path(absolutePath), fileSize, lastWriteTime, hasFileInfo != 0);
            }
        }
    }

    for (auto& entry : entries)
        retVal->insert(std::move(entry));

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
	if (!prefixMatch)
	{
		result.cacheHit = false;
		result.status = EProbeStatus::PrefixChanged;
		return result;
	}
	bool depsUpdated = false;
	const bool depsValid = cache->validateDependencies(finder, &depsUpdated);
	result.depsUpdated = depsUpdated;
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
        if (probe.depsUpdated)
            result.cacheUpdated = true;
    }
    else
    {
        CPreprocessCache::SEntry entry;
        IShader::E_SHADER_STAGE prefixStage = stage;
        SPreprocessorOptions preCacheOpt = preprocessOptions;
        preCacheOpt.depfile = false;
        preCacheOpt.applyForceIncludes = false;
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
    ensurePrefixLoaded();

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
    const uint32_t prefixSize = static_cast<uint32_t>(m_entry.preprocessedPrefix.size());
    write_u32(out, prefixSize);

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
        const auto abs = dep.getAbsolutePath().generic_string();
        write_string(out, abs);
        const uint8_t standardInclude = dep.isStandardInclude() ? 1u : 0u;
        write_bytes(out, &standardInclude, sizeof(standardInclude));
        write_bytes(out, dep.getHash().data, sizeof(dep.getHash().data));
        const uint64_t fileSize = dep.getFileSize();
        write_bytes(out, &fileSize, sizeof(fileSize));
        const int64_t lastWriteTime = dep.getLastWriteTime();
        write_bytes(out, &lastWriteTime, sizeof(lastWriteTime));
        const uint8_t hasFileInfo = dep.getHasFileInfo() ? 1u : 0u;
        write_bytes(out, &hasFileInfo, sizeof(hasFileInfo));
    }
    if (prefixSize)
        write_bytes(out, m_entry.preprocessedPrefix.data(), m_entry.preprocessedPrefix.size());

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
    uint32_t prefixSize = 0;
    if (!read_u32(serializedCache, offset, prefixSize))
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
    entry.macroBlock = buildMacroBlock(entry.macroDefs);

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
        std::string absolutePath;
        if (!read_string(serializedCache, offset, absolutePath))
            return nullptr;
        uint8_t standardInclude = 0;
        if (!read_bytes(serializedCache, offset, &standardInclude, sizeof(standardInclude)))
            return nullptr;
        core::blake3_hash_t hash = {};
        if (!read_bytes(serializedCache, offset, hash.data, sizeof(hash.data)))
            return nullptr;
        uint64_t fileSize = 0;
        if (!read_bytes(serializedCache, offset, &fileSize, sizeof(fileSize)))
            return nullptr;
        int64_t lastWriteTime = 0;
        if (!read_bytes(serializedCache, offset, &lastWriteTime, sizeof(lastWriteTime)))
            return nullptr;
        uint8_t hasFileInfo = 0;
        if (!read_bytes(serializedCache, offset, &hasFileInfo, sizeof(hasFileInfo)))
            return nullptr;
        entry.dependencies.emplace_back(system::path(dir), identifier, standardInclude != 0, hash, system::path(absolutePath), fileSize, lastWriteTime, hasFileInfo != 0);
    }

    if (offset + prefixSize > serializedCache.size())
        return nullptr;
    if (prefixSize)
    {
        entry.preprocessedPrefix.assign(reinterpret_cast<const char*>(serializedCache.data() + offset), prefixSize);
        offset += prefixSize;
    }

    retVal->m_prefixLoaded = true;
    retVal->m_backingPath.clear();
    retVal->m_prefixOffset = 0;
    retVal->m_prefixSize = 0;
    retVal->m_hasEntry = true;
    return retVal;
}

core::smart_refctd_ptr<IShaderCompiler::CPreprocessCache> IShaderCompiler::CPreprocessCache::loadFromFile(const system::path& path, ELoadStatus& status, bool loadPrefix)
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

    auto read_bytes = [&in](void* dst, size_t count) -> bool
    {
        return bool(in.read(reinterpret_cast<char*>(dst), count));
    };
    auto read_u32 = [&read_bytes](uint32_t& out) -> bool
    {
        return read_bytes(&out, sizeof(out));
    };
    auto read_string = [&read_u32, &read_bytes](std::string& out) -> bool
    {
        uint32_t len = 0;
        if (!read_u32(len))
            return false;
        if (!len)
        {
            out.clear();
            return true;
        }
        out.resize(len);
        return read_bytes(out.data(), len);
    };

    uint32_t magic = 0;
    if (!read_u32(magic) || magic != 0x50435250u)
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    std::string version;
    if (!read_string(version) || version != VERSION)
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    auto retVal = core::make_smart_refctd_ptr<CPreprocessCache>();
    auto& entry = retVal->m_entry;
    if (!read_bytes(&entry.prefixHash, sizeof(entry.prefixHash)))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }
    if (!read_u32(entry.pragmaStage))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    uint32_t prefixSize = 0;
    if (!read_u32(prefixSize))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    uint32_t macroCount = 0;
    if (!read_u32(macroCount))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }
    entry.macroDefs.clear();
    entry.macroDefs.reserve(macroCount);
    for (uint32_t i = 0; i < macroCount; ++i)
    {
        std::string macro;
        if (!read_string(macro))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        entry.macroDefs.emplace_back(std::move(macro));
    }
    entry.macroBlock = buildMacroBlock(entry.macroDefs);

    uint32_t flagCount = 0;
    if (!read_u32(flagCount))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }
    entry.dxcFlags.clear();
    entry.dxcFlags.reserve(flagCount);
    for (uint32_t i = 0; i < flagCount; ++i)
    {
        std::string flag;
        if (!read_string(flag))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        entry.dxcFlags.emplace_back(std::move(flag));
    }

    uint32_t depCount = 0;
    if (!read_u32(depCount))
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }
    entry.dependencies.clear();
    entry.dependencies.reserve(depCount);
    for (uint32_t i = 0; i < depCount; ++i)
    {
        std::string dir;
        std::string identifier;
        if (!read_string(dir) || !read_string(identifier))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        std::string absolutePath;
        if (!read_string(absolutePath))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        uint8_t standardInclude = 0;
        if (!read_bytes(&standardInclude, sizeof(standardInclude)))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        core::blake3_hash_t hash = {};
        if (!read_bytes(hash.data, sizeof(hash.data)))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        uint64_t fileSize = 0;
        if (!read_bytes(&fileSize, sizeof(fileSize)))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        int64_t lastWriteTime = 0;
        if (!read_bytes(&lastWriteTime, sizeof(lastWriteTime)))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        uint8_t hasFileInfo = 0;
        if (!read_bytes(&hasFileInfo, sizeof(hasFileInfo)))
        {
            status = ELoadStatus::Invalid;
            return nullptr;
        }
        entry.dependencies.emplace_back(system::path(dir), identifier, standardInclude != 0, hash, system::path(absolutePath), fileSize, lastWriteTime, hasFileInfo != 0);
    }

    const auto prefixOffset = static_cast<uint64_t>(in.tellg());
    if (prefixOffset + prefixSize > size)
    {
        status = ELoadStatus::Invalid;
        return nullptr;
    }

    if (loadPrefix)
    {
        entry.preprocessedPrefix.clear();
        if (prefixSize)
        {
            entry.preprocessedPrefix.resize(prefixSize);
            if (!read_bytes(entry.preprocessedPrefix.data(), prefixSize))
            {
                status = ELoadStatus::Invalid;
                return nullptr;
            }
        }
        retVal->m_prefixLoaded = true;
        retVal->m_backingPath.clear();
        retVal->m_prefixOffset = 0;
        retVal->m_prefixSize = 0;
    }
    else
    {
        if (prefixSize)
            in.seekg(static_cast<std::streamoff>(prefixSize), std::ios::cur);
        retVal->m_prefixLoaded = false;
        retVal->m_backingPath = path;
        retVal->m_prefixOffset = prefixOffset;
        retVal->m_prefixSize = prefixSize;
    }

    retVal->m_hasEntry = true;
    status = ELoadStatus::Loaded;
    return retVal;
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

bool IShaderCompiler::CPreprocessCache::validateDependencies(const CIncludeFinder* finder, bool* depsUpdated) const
{
    if (!m_hasEntry || !finder)
        return false;
    if (depsUpdated)
        *depsUpdated = false;
    bool updated = false;
    auto* system = finder->getSystem();

    std::vector<size_t> mismatches;
    mismatches.reserve(m_entry.dependencies.size());
    collectFileInfoMismatchesParallel(m_entry.dependencies, mismatches, system);
    if (mismatches.empty())
        return true;

    std::unordered_map<system::path, bool> fileStatus;
    std::unordered_map<std::string, bool> logicalStatus;
    fileStatus.reserve(mismatches.size());
    logicalStatus.reserve(mismatches.size());
    for (size_t idx : mismatches)
    {
        const auto& dep = m_entry.dependencies[idx];
        auto makeLogicalKey = [&dep]()
        {
            std::string key;
            key.reserve(dep.getIdentifier().size() + dep.getRequestingSourceDir().string().size() + 4);
            key.append(dep.getRequestingSourceDir().string());
            key.push_back('|');
            key.append(dep.getIdentifier());
            key.push_back('|');
            key.push_back(dep.isStandardInclude() ? '1' : '0');
            return key;
        };

        if (dep.getHasFileInfo() && !dep.getAbsolutePath().empty())
        {
            if (auto it = fileStatus.find(dep.getAbsolutePath()); it != fileStatus.end())
            {
                if (!it->second)
                    return false;
                continue;
            }
        }
        else
        {
            auto key = makeLogicalKey();
            if (auto it = logicalStatus.find(key); it != logicalStatus.end())
            {
                if (!it->second)
                    return false;
                continue;
            }
        }

        bool valid = false;
        bool precomputedChecked = false;
        if (system && !dep.getAbsolutePath().empty())
        {
            system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
            system->createFile(future, dep.getAbsolutePath(), system::IFile::ECF_READ);
            if (future.wait())
            {
                core::smart_refctd_ptr<system::IFile> file;
                if (auto lock = future.acquire(); lock)
                    lock.move_into(file);
                if (file)
                {
                    if (auto precomputed = file->getPrecomputedHash())
                    {
                        precomputedChecked = true;
                        core::blake3_hash_t hash = {};
                        std::memcpy(hash.data, &(*precomputed), sizeof(hash.data));
                        if (hash == dep.getHash())
                        {
                            valid = true;
                            if (!dep.getHasFileInfo() && dep.getAbsolutePath().is_absolute())
                            {
                                dep.setFileInfo(file->getSize(), file->getLastWriteTime().time_since_epoch().count(), true);
                                updated = true;
                            }
                        }
                        else
                        {
                            if (dep.getHasFileInfo() && !dep.getAbsolutePath().empty())
                                fileStatus.emplace(dep.getAbsolutePath(), false);
                            else
                                logicalStatus.emplace(makeLogicalKey(), false);
                            return false;
                        }
                    }
                }
            }
        }

        if (!valid && !precomputedChecked)
        {
            const std::string identifier(dep.getIdentifier());
            IIncludeLoader::found_t header;
            if (dep.isStandardInclude())
                header = finder->getIncludeStandard(dep.getRequestingSourceDir(), identifier);
            else
                header = finder->getIncludeRelative(dep.getRequestingSourceDir(), identifier);
            if (header.hash != dep.getHash())
            {
                if (dep.getHasFileInfo() && !dep.getAbsolutePath().empty())
                    fileStatus.emplace(dep.getAbsolutePath(), false);
                else
                    logicalStatus.emplace(makeLogicalKey(), false);
                return false;
            }

            valid = true;
            if (header.hasFileInfo && dep.getAbsolutePath().is_absolute())
            {
                dep.setFileInfo(header.fileSize, header.lastWriteTime, true);
                updated = true;
            }
        }

        if (valid && dep.getHasFileInfo() && dep.getAbsolutePath().is_absolute())
        {
            uint64_t size = 0;
            int64_t ticks = 0;
            if (getFileInfoCached(dep.getAbsolutePath(), size, ticks, system) &&
                (dep.getFileSize() != size || dep.getLastWriteTime() != ticks))
            {
                dep.setFileInfo(size, ticks, true);
                updated = true;
            }
        }

        if (dep.getHasFileInfo() && !dep.getAbsolutePath().empty())
            fileStatus.emplace(dep.getAbsolutePath(), true);
        else
            logicalStatus.emplace(makeLogicalKey(), true);
    }

    if (depsUpdated)
        *depsUpdated = updated;
    return true;
}

void IShaderCompiler::CPreprocessCache::ensurePrefixLoaded() const
{
    if (m_prefixLoaded)
        return;
    if (m_prefixSize == 0)
    {
        m_prefixLoaded = true;
        return;
    }
    if (m_backingPath.empty())
        return;

    std::ifstream in(m_backingPath, std::ios::binary);
    if (!in)
        return;
    in.seekg(static_cast<std::streamoff>(m_prefixOffset), std::ios::beg);
    if (!in)
        return;

    std::string prefix;
    prefix.resize(m_prefixSize);
    if (!in.read(prefix.data(), prefix.size()))
        return;

    m_entry.preprocessedPrefix = std::move(prefix);
    m_prefixLoaded = true;
}


std::string IShaderCompiler::CPreprocessCache::buildCombinedCode(std::string_view body, std::string_view sourceIdentifier) const
{
    if (!m_hasEntry)
        return std::string(body);

    ensurePrefixLoaded();
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

    if (compression == ECompression::RAW)
    {
        spirv = core::smart_refctd_ptr<asset::ICPUBuffer>(const_cast<asset::ICPUBuffer*>(uncompressedSpirvBuffer));
        return static_cast<bool>(spirv);
    }

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
    if (compression == ECompression::RAW)
    {
        if (!spirv)
            return nullptr;
        auto buffer = spirv;
        return core::make_smart_refctd_ptr<asset::IShader>(std::move(buffer), IShader::E_CONTENT_TYPE::ECT_SPIRV, compilerArgs.preprocessorArgs.sourceIdentifier.data());
    }

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

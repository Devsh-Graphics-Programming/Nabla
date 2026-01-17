#include "nabla.h"
#include "nbl/system/IApplicationFramework.h"
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <cstring>
#include <cstdarg>
#include <cctype>
#include <vector>
#include <argparse/argparse.hpp>
#include "nbl/asset/metadata/CHLSLMetadata.h"
#include "nbl/asset/utils/shaderCompiler_serialization.h"
#include "nbl/core/hash/blake.h"
#include "nbl/core/hash/fnv1a64.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using namespace nbl;
using namespace nbl::system;
using namespace nbl::core;
using namespace nbl::asset;

class TrimStdoutLogger final : public CStdoutLogger
{
public:
    TrimStdoutLogger(const bitflag<E_LOG_LEVEL> logLevelMask) : CStdoutLogger(logLevelMask) {}

protected:
    void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
    {
        const auto str = constructLogString(fmt, logLevel, args);
        size_t size = str.size();
        while (size && str[size - 1] == '\0')
            --size;
        if (!size)
            return;
        std::fwrite(str.data(), 1, size, stdout);
        std::fflush(stdout);
    }
};

class TrimFileLogger final : public CFileLogger
{
public:
    using CFileLogger::CFileLogger;

protected:
    void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
    {
        const auto str = constructLogString(fmt, logLevel, args);
        size_t size = str.size();
        while (size && str[size - 1] == '\0')
            --size;
        if (!size)
            return;
        IFile::success_t succ;
        m_file->write(succ, str.data(), m_pos, size);
        m_pos += succ.getBytesProcessed();
    }
};

class ShaderLogger final : public IThreadsafeLogger
{
public:
    ShaderLogger(smart_refctd_ptr<ISystem> system, path logPath, const bitflag<E_LOG_LEVEL> fileMask, const bitflag<E_LOG_LEVEL> consoleMask, const bool noLog)
        : IThreadsafeLogger(fileMask | consoleMask), m_system(std::move(system)), m_logPath(std::move(logPath)), m_fileMask(fileMask), m_consoleMask(consoleMask), m_noLog(noLog)
    {
        m_stdoutLogger = make_smart_refctd_ptr<TrimStdoutLogger>(m_consoleMask);
		beginBuild();
    }

    void beginBuild()
    {
        m_fileLogger = nullptr;
        m_file = nullptr;

        if (m_noLog)
            return;
        if (!m_system || m_logPath.empty())
            return;

        const auto parent = std::filesystem::path(m_logPath).parent_path();
        if (!parent.empty() && m_system && !m_system->exists(parent, IFileBase::ECF_READ))
            m_system->createDirectory(parent);

        for (auto attempt = 0u; attempt < kDeleteRetries; ++attempt)
        {
            if (m_system->deleteFile(m_logPath))
                break;
            std::this_thread::sleep_for(kDeleteDelay);
        }

        ISystem::future_t<smart_refctd_ptr<IFile>> fut;
        m_system->createFile(fut, m_logPath, kLogFlags);

        if (fut.wait())
        {
            auto lk = fut.acquire();
            if (lk)
                lk.move_into(m_file);
        }

        if (!m_file)
            return;

        std::error_code ec;
        std::filesystem::resize_file(m_logPath, 0, ec);

        m_fileLogger = make_smart_refctd_ptr<TrimFileLogger>(smart_refctd_ptr(m_file), true, m_fileMask);
    }

private:
    static constexpr auto kDeleteRetries = 3u;
    static constexpr auto kDeleteDelay = std::chrono::milliseconds(100);
    static constexpr auto kLogFlags = bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_WRITE) | IFileBase::ECF_SHARE_READ_WRITE | IFileBase::ECF_SHARE_DELETE;

    static inline std::string formatMessageOnly(const std::string_view& fmt, va_list args)
    {
        va_list a;
        va_copy(a, args);
        const int n = std::vsnprintf(nullptr, 0, fmt.data(), a);
        va_end(a);
        if (n <= 0)
            return {};
        std::string s(size_t(n) + 1u, '\0');
        std::vsnprintf(s.data(), s.size(), fmt.data(), args);
        s.resize(size_t(n));
        return s;
    }

    void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
    {
        const auto msg = formatMessageOnly(fmt, args);
        if (msg.empty())
            return;

        if (m_stdoutLogger && (logLevel & m_consoleMask.value))
            m_stdoutLogger->log("%s", logLevel, msg.c_str());

        if (m_noLog || !(logLevel & m_fileMask.value) || !m_fileLogger)
            return;

        m_fileLogger->log("%s", logLevel, msg.c_str());
    }

    smart_refctd_ptr<ISystem> m_system;
    smart_refctd_ptr<IFile> m_file;
    smart_refctd_ptr<TrimStdoutLogger> m_stdoutLogger;
    smart_refctd_ptr<TrimFileLogger> m_fileLogger;
    path m_logPath;
    bitflag<E_LOG_LEVEL> m_fileMask;
    bitflag<E_LOG_LEVEL> m_consoleMask;
    bool m_noLog = false;
};

class ShaderCompiler final : public IApplicationFramework
{
    using base_t = IApplicationFramework;

public:
    using base_t::base_t;

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        const auto rawArgs = std::vector<std::string>(argv.begin(), argv.end());
        const auto expandedArgs = expandJoinedArgs(rawArgs);
        m_logger = make_smart_refctd_ptr<TrimStdoutLogger>(bitflag(ILogger::ELL_ALL));

        argparse::ArgumentParser program("nsc");
        program.add_argument("--dump-build-info").default_value(false).implicit_value(true);
        program.add_argument("--file").default_value(std::string{});
        program.add_argument("-P").default_value(false).implicit_value(true);
        program.add_argument("-no-nbl-builtins").default_value(false).implicit_value(true);
        program.add_argument("-MD").default_value(false).implicit_value(true);
        program.add_argument("-M").default_value(false).implicit_value(true);
        program.add_argument("-MF").default_value(std::string{});
        program.add_argument("-Fo").default_value(std::string{});
        program.add_argument("-Fc").default_value(std::string{});
        program.add_argument("-log").default_value(std::string{});
        program.add_argument("-nolog").default_value(false).implicit_value(true);
        program.add_argument("-quiet").default_value(false).implicit_value(true);
        program.add_argument("-verbose").default_value(false).implicit_value(true);
        program.add_argument("-shader-cache").default_value(false).implicit_value(true);
        program.add_argument("-shader-cache-file").default_value(std::string{});
        program.add_argument("-preprocess-cache").default_value(false).implicit_value(true);
        program.add_argument("-preprocess-cache-file").default_value(std::string{});

        std::vector<std::string> unknownArgs;
        try
        {
            unknownArgs = program.parse_known_args(expandedArgs);
        }
        catch (const std::runtime_error& err)
        {
            std::ostringstream usage;
            usage << program;
            if (m_logger)
                m_logger->log("%s\n%s", ILogger::ELL_ERROR, err.what(), usage.str().c_str());
            return false;
        }

        if (!isAPILoaded())
        {
            if (m_logger)
                m_logger->log("Could not load Nabla API, terminating!", ILogger::ELL_ERROR);
            return false;
        }

        m_system = system ? std::move(system) : IApplicationFramework::createSystem();
        if (!m_system)
        {
            if (m_logger)
                m_logger->log("Failed to create system.", ILogger::ELL_ERROR);
            return false;
        }

        if (program.get<bool>("--dump-build-info"))
        {
            dumpBuildInfo(program);
            std::exit(0);
        }

        if (rawArgs.size() < 2)
        {
            if (m_logger)
                m_logger->log("Insufficient arguments.", ILogger::ELL_ERROR);
            return false;
        }

        const std::string fileToCompile = rawArgs.back();
        if (!m_system->exists(fileToCompile, IFileBase::ECF_READ))
        {
            if (m_logger)
                m_logger->log("Input shader file does not exist: %s", ILogger::ELL_ERROR, fileToCompile.c_str());
            return false;
        }

        const bool preprocessOnly = program.get<bool>("-P");
        const bool hasFc = program.is_used("-Fc");
        const bool hasFo = program.is_used("-Fo");

        if (hasFc == hasFo)
        {
            if (hasFc)
            {
                if (m_logger)
                    m_logger->log("Invalid arguments. Passed both -Fo and -Fc.", ILogger::ELL_ERROR);
            }
            else
            {
                if (m_logger)
                    m_logger->log("Missing arguments. Expecting `-Fc {filename}` or `-Fo {filename}`.", ILogger::ELL_ERROR);
            }
            return false;
        }

        const std::string outputFilepath = hasFc ? program.get<std::string>("-Fc") : program.get<std::string>("-Fo");
        if (outputFilepath.empty())
        {
            if (m_logger)
                m_logger->log("Invalid output file path.", ILogger::ELL_ERROR);
            return false;
        }

        const bool quiet = program.get<bool>("-quiet");
        const bool verbose = program.get<bool>("-verbose");
        bool shaderCacheEnabled = program.get<bool>("-shader-cache");
        const std::string shaderCachePathOverride = program.is_used("-shader-cache-file") ? program.get<std::string>("-shader-cache-file") : std::string{};
        if (!shaderCachePathOverride.empty())
            shaderCacheEnabled = true;
        bool preprocessCacheEnabled = program.get<bool>("-preprocess-cache");
        const std::string preprocessCachePathOverride = program.is_used("-preprocess-cache-file") ? program.get<std::string>("-preprocess-cache-file") : std::string{};
        if (!preprocessCachePathOverride.empty())
            preprocessCacheEnabled = true;
        if (quiet && verbose)
        {
            if (m_logger)
                m_logger->log("Invalid arguments. Passed both -quiet and -verbose.", ILogger::ELL_ERROR);
            return false;
        }

        const bool noLog = program.get<bool>("-nolog");
        const std::string logPathOverride = program.is_used("-log") ? program.get<std::string>("-log") : std::string{};
        if (noLog && !logPathOverride.empty())
        {
            if (m_logger)
                m_logger->log("Invalid arguments. Passed both -nolog and -log.", ILogger::ELL_ERROR);
            return false;
        }

        const auto logPath = logPathOverride.empty() ? std::filesystem::path(outputFilepath).concat(".log") : std::filesystem::path(logPathOverride);
        const auto fileMask = bitflag(ILogger::ELL_ALL);
        const auto consoleMask = bitflag(ILogger::ELL_WARNING) | ILogger::ELL_ERROR;

        m_logger = make_smart_refctd_ptr<ShaderLogger>(m_system, logPath, fileMask, consoleMask, noLog);
        const auto configName = std::filesystem::path(outputFilepath).parent_path().filename().string();
        const auto configLabel = configName.empty() ? "Unknown" : configName;

        m_arguments = std::move(unknownArgs);
        if (!m_arguments.empty() && m_arguments.back() == fileToCompile)
            m_arguments.pop_back();
        if (!m_arguments.empty())
        {
            std::vector<std::string> filteredArgs;
            for (size_t i = 0; i < m_arguments.size(); ++i)
            {
                const auto& arg = m_arguments[i];
                if (arg == "-FI" || arg == "-include" || arg == "/FI")
                {
                    if (i + 1 >= m_arguments.size())
                    {
                        if (m_logger)
                            m_logger->log("Missing argument for %s.", ILogger::ELL_ERROR, arg.c_str());
                        return false;
                    }
                    m_force_includes.push_back(m_arguments[i + 1]);
                    ++i;
                    continue;
                }
                if ((arg.rfind("-FI", 0) == 0 || arg.rfind("/FI", 0) == 0) && arg.size() > 3)
                {
                    m_force_includes.push_back(arg.substr(3));
                    continue;
                }
                if (arg.rfind("-include", 0) == 0 && arg.size() > 8)
                {
                    m_force_includes.push_back(arg.substr(8));
                    continue;
                }
                filteredArgs.push_back(arg);
            }
            m_arguments = std::move(filteredArgs);
        }

        bool noNblBuiltins = program.get<bool>("-no-nbl-builtins");
        if (noNblBuiltins)
        {
            m_logger->log("Unmounting builtins.");
            m_system->unmountBuiltins();
        }

        DepfileConfig dep;
        if (program.get<bool>("-MD") || program.get<bool>("-M") || program.is_used("-MF"))
            dep.enabled = true;
        if (program.is_used("-MF"))
            dep.path = program.get<std::string>("-MF");
        if (dep.enabled && dep.path.empty())
            dep.path = outputFilepath + ".dep";

        ShaderCacheConfig shaderCache;
        shaderCache.enabled = shaderCacheEnabled && !preprocessOnly;
        shaderCache.verbose = verbose;
        if (shaderCache.enabled)
            shaderCache.path = shaderCachePathOverride.empty() ? makeCachePath(outputFilepath) : std::filesystem::path(shaderCachePathOverride);

        PreprocessCacheConfig preCache;
        preCache.enabled = preprocessCacheEnabled && !preprocessOnly;
        preCache.verbose = verbose;
        if (preCache.enabled)
            preCache.path = preprocessCachePathOverride.empty() ? makePreprocessCachePath(outputFilepath) : std::filesystem::path(preprocessCachePathOverride);

#ifndef NBL_EMBED_BUILTIN_RESOURCES
        if (!noNblBuiltins)
        {
            m_system->unmountBuiltins();
            noNblBuiltins = true;
            m_logger->log("nsc.exe was compiled with builtin resources disabled. Force enabling -no-nbl-builtins.", ILogger::ELL_WARNING);
        }
#endif

        if (std::find(m_arguments.begin(), m_arguments.end(), "-E") == m_arguments.end())
        {
            m_arguments.push_back("-E");
            m_arguments.push_back("main");
        }

        for (size_t i = 0; i + 1 < m_arguments.size(); ++i)
        {
            if (m_arguments[i] == "-I")
                m_include_search_paths.emplace_back(m_arguments[i + 1]);
        }

        auto addIncludePath = [&](const std::filesystem::path& path)
        {
            if (path.empty())
                return;
            std::error_code ec;
            const auto normalized = std::filesystem::weakly_canonical(path, ec).generic_string();
            if (normalized.empty())
                return;
            if (std::find(m_include_search_paths.begin(), m_include_search_paths.end(), normalized) == m_include_search_paths.end())
                m_include_search_paths.emplace_back(normalized);
        };

        if (!rawArgs.empty())
        {
            std::error_code ec;
            std::filesystem::path exePath = rawArgs.front();
            if (std::filesystem::exists(exePath, ec))
            {
                exePath = std::filesystem::weakly_canonical(exePath, ec);
                if (!ec)
                {
                    const auto root = exePath.parent_path().parent_path().parent_path();
                    addIncludePath(root / "include");
                }
            }
        }

        if (verbose)
        {
            auto join = [](const std::vector<std::string>& items)
            {
                std::string out;
                for (const auto& item : items)
                {
                    if (!out.empty())
                        out.push_back(' ');
                    out.append(item);
                }
                return out;
            };
            m_logger->log("Verbose logging enabled.", ILogger::ELL_DEBUG);
            m_logger->log("Variant: %s", ILogger::ELL_DEBUG, configLabel.c_str());
            if (!rawArgs.empty())
                m_logger->log("Compiler: %s", ILogger::ELL_DEBUG, rawArgs.front().c_str());
            m_logger->log("Command line: %s", ILogger::ELL_DEBUG, join(rawArgs).c_str());
            m_logger->log("Input: %s", ILogger::ELL_DEBUG, fileToCompile.c_str());
            m_logger->log("Output: %s", ILogger::ELL_DEBUG, outputFilepath.c_str());
            if (dep.enabled)
                m_logger->log("Depfile: %s", ILogger::ELL_DEBUG, dep.path.c_str());
            if (shaderCache.enabled)
                m_logger->log("Shader Cache: %s", ILogger::ELL_DEBUG, shaderCache.path.string().c_str());
            if (preCache.enabled)
                m_logger->log("Preprocess cache: %s", ILogger::ELL_DEBUG, preCache.path.string().c_str());
        }

        const char* const action = preprocessOnly ? "Preprocessing" : "Compiling";
        const char* const outType = preprocessOnly ? "Preprocessed" : "Compiled";
        m_logger->log("%s the input file.", ILogger::ELL_INFO, action);

        auto [shader, shaderStage] = open_shader_file(fileToCompile);
        if (!shader || shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_HLSL)
        {
            m_logger->log("Error. Loaded shader file content is not HLSL.", ILogger::ELL_ERROR);
            return false;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        const std::string preprocessedOutputPath = outputFilepath + ".pre.hlsl";
        const auto job = runShaderJob(shader.get(), shaderStage, fileToCompile, dep, shaderCache, preCache, preprocessOnly, outputFilepath, preprocessedOutputPath, verbose);
        const auto end = std::chrono::high_resolution_clock::now();

        const char* const op = preprocessOnly ? "preprocessing" : "compilation";
        if (!job.ok)
        {
            m_logger->log("Shader %s failed.", ILogger::ELL_ERROR, op);
            return false;
        }

        m_logger->log("Shader %s successful.", ILogger::ELL_INFO, op);
        if (dep.enabled)
        {
            const bool depWritten = m_system->exists(dep.path, IFileBase::ECF_READ);
            if (!depWritten)
                m_logger->log("Dependency file missing at %s", ILogger::ELL_WARNING, dep.path.c_str());
            m_logger->log(depWritten ? "Depfile written successfully." : "Depfile write failed.", depWritten ? ILogger::ELL_INFO : ILogger::ELL_WARNING);
        }

        const auto outParent = std::filesystem::path(outputFilepath).parent_path();
        if (!outParent.empty() && m_system && !m_system->exists(outParent, IFileBase::ECF_READ))
        {
            if (!m_system->createDirectory(outParent))
            {
                m_logger->log("Failed to create parent directory for output %s.", ILogger::ELL_ERROR, outputFilepath.c_str());
                return false;
            }
        }

        if (!job.view.empty())
        {
            const auto writeStart = std::chrono::high_resolution_clock::now();
            if (!writeBinaryFile(m_system.get(), std::filesystem::path(outputFilepath), job.view.data(), job.view.size()))
            {
                m_logger->log("Failed to write output file: %s", ILogger::ELL_ERROR, outputFilepath.c_str());
                return false;
            }
            OutputHashRecord record = {};
            record.size = job.view.size();
            {
                core::blake3_hasher hasher;
                hasher.update(job.view.data(), job.view.size());
                record.hash = static_cast<core::blake3_hash_t>(hasher);
            }
            const auto hashPath = makeOutputHashPath(std::filesystem::path(outputFilepath));
            if (!writeBinaryFile(m_system.get(), hashPath, &record, sizeof(record)))
                m_logger->log("Failed to write output hash file: %s", ILogger::ELL_WARNING, hashPath.string().c_str());
            const auto writeEnd = std::chrono::high_resolution_clock::now();
            if (verbose)
            {
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(writeEnd - writeStart).count();
                m_logger->log("Write output took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(duration));
            }
        }
        else if (verbose)
        {
            m_logger->log("Output up to date. Skipping write.", ILogger::ELL_DEBUG);
        }

        const auto took = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        m_logger->log("Total took: %s ms.", ILogger::ELL_PERFORMANCE, took.c_str());

        flushSystemQueue(m_system.get(), std::filesystem::path(outputFilepath));

        return true;
    }

    void workLoopBody() override {}
    bool keepRunning() override { return false; }

private:
    struct DepfileConfig
    {
        bool enabled = false;
        std::string path;
    };

    struct ShaderCacheConfig
    {
        bool enabled = false;
        bool verbose = false;
        std::filesystem::path path;
    };

    struct PreprocessCacheConfig
    {
        bool enabled = false;
        bool verbose = false;
        std::filesystem::path path;
    };

    enum class CacheLoadStatus : uint8_t
    {
        Missing,
        Invalid,
        Loaded
    };

    struct RunResult
    {
        bool ok = false;
        std::string text;
        smart_refctd_ptr<IShader> compiled;
        std::string_view view;
    };

    struct OutputHashRecord
    {
        core::blake3_hash_t hash = {};
        uint64_t size = 0;
    };

    static std::filesystem::path makeCachePath(std::filesystem::path outputPath)
    {
        outputPath += ".ppcache";
        return outputPath;
    }

    static std::filesystem::path makeOutputHashPath(std::filesystem::path outputPath)
    {
        outputPath += ".hash";
        return outputPath;
    }

    static std::filesystem::path makePreprocessCachePath(std::filesystem::path outputPath)
    {
        outputPath += ".ppcache.pre";
        return outputPath;
    }

    static smart_refctd_ptr<IShaderCompiler::CCache> loadShaderCache(system::ISystem* system, const std::filesystem::path& path, CacheLoadStatus& status, bool skipDependencies)
    {
        status = CacheLoadStatus::Missing;
        if (!system)
        {
            status = CacheLoadStatus::Invalid;
            return nullptr;
        }

        if (!system->exists(path, IFileBase::ECF_READ))
            return nullptr;

        auto openFile = [&](const core::bitflag<IFileBase::E_CREATE_FLAGS> flags) -> smart_refctd_ptr<IFile>
        {
            ISystem::future_t<smart_refctd_ptr<IFile>> future;
            system->createFile(future, path, flags);
            if (!future.wait())
                return nullptr;
            smart_refctd_ptr<IFile> file;
            if (auto lock = future.acquire(); lock)
                lock.move_into(file);
            return file;
        };

        smart_refctd_ptr<IFile> file = openFile(bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_READ) | IFileBase::ECF_MAPPABLE);
        if (!file)
            file = openFile(bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_READ));
        if (!file)
        {
            status = CacheLoadStatus::Invalid;
            return nullptr;
        }

        const size_t size = file->getSize();
        if (!size)
        {
            status = CacheLoadStatus::Invalid;
            return nullptr;
        }

        const auto* mapped = static_cast<const uint8_t*>(file->getMappedPointer());
        std::vector<uint8_t> data;
        std::span<const uint8_t> serialized;
        if (mapped)
        {
            serialized = std::span<const uint8_t>(mapped, size);
        }
        else
        {
            data.resize(size);
            IFile::success_t succ;
            file->read(succ, data.data(), 0, size);
            if (!succ || succ.getBytesProcessed(true) != size)
            {
                status = CacheLoadStatus::Invalid;
                return nullptr;
            }
            serialized = std::span<const uint8_t>(data.data(), data.size());
        }

        auto cache = IShaderCompiler::CCache::deserialize(serialized, skipDependencies);
        if (!cache)
        {
            status = CacheLoadStatus::Invalid;
            return nullptr;
        }

        status = CacheLoadStatus::Loaded;
        return cache;
    }

    static bool getFileInfo(system::ISystem* system, const std::filesystem::path& path, uint64_t& sizeOut, int64_t& timeOut)
    {
        if (!system || !system->exists(path, IFileBase::ECF_READ))
            return false;

        ISystem::future_t<smart_refctd_ptr<IFile>> future;
        system->createFile(future, path, IFileBase::ECF_READ);
        if (!future.wait())
            return false;

        smart_refctd_ptr<IFile> file;
        if (auto lock = future.acquire(); lock)
            lock.move_into(file);
        if (!file)
            return false;

        sizeOut = file->getSize();
        timeOut = file->getLastWriteTime().time_since_epoch().count();
        return sizeOut != 0;
    }

    static bool readBinaryFile(system::ISystem* system, const std::filesystem::path& path, void* data, size_t size)
    {
        if (!system)
            return false;
        if (!system->exists(path, IFileBase::ECF_READ))
            return false;

        ISystem::future_t<smart_refctd_ptr<IFile>> future;
        system->createFile(future, path, IFileBase::ECF_READ);
        if (!future.wait())
            return false;

        smart_refctd_ptr<IFile> file;
        if (auto lock = future.acquire(); lock)
            lock.move_into(file);
        if (!file || file->getSize() != size)
            return false;

        IFile::success_t succ;
        file->read(succ, data, 0, size);
        return succ.getBytesProcessed(true) == size;
    }

    static bool writeBinaryFile(system::ISystem* system, const std::filesystem::path& path, const void* data, size_t size)
    {
        if (!system)
            return false;

        const auto parent = path.parent_path();
        if (!parent.empty() && !system->exists(parent, IFileBase::ECF_READ))
            system->createDirectory(parent);

        if (!system->exists(path, IFileBase::ECF_READ))
        {
            ISystem::future_t<smart_refctd_ptr<IFile>> future;
            system->createFile(future, path, bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_WRITE) | IFileBase::ECF_SHARE_READ_WRITE | IFileBase::ECF_SHARE_DELETE);
            if (!future.wait())
                return false;

            smart_refctd_ptr<IFile> file;
            if (auto lock = future.acquire(); lock)
                lock.move_into(file);
            if (!file)
                return false;

            IFile::success_t succ;
            file->write(succ, data, 0, size);
            return succ.getBytesProcessed(true) == size;
        }

        std::filesystem::path tempPath = path;
        tempPath += ".tmp";
        tempPath += std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        system->deleteFile(tempPath);

        ISystem::future_t<smart_refctd_ptr<IFile>> future;
        system->createFile(future, tempPath, bitflag<IFileBase::E_CREATE_FLAGS>(IFileBase::ECF_WRITE) | IFileBase::ECF_SHARE_READ_WRITE | IFileBase::ECF_SHARE_DELETE);
        if (!future.wait())
            return false;

        smart_refctd_ptr<IFile> file;
        if (auto lock = future.acquire(); lock)
            lock.move_into(file);
        if (!file)
            return false;

        IFile::success_t succ;
        file->write(succ, data, 0, size);
        if (succ.getBytesProcessed(true) != size)
        {
            system->deleteFile(tempPath);
            return false;
        }

        file = nullptr;
        system->deleteFile(path);
        const std::error_code moveError = system->moveFileOrDirectory(tempPath, path);
        if (moveError)
        {
            system->deleteFile(tempPath);
            return false;
        }
        return true;
    }

    static void flushSystemQueue(system::ISystem* system, const std::filesystem::path& path)
    {
        if (!system)
            return;

        ISystem::future_t<smart_refctd_ptr<IFile>> future;
        system->createFile(future, path, IFileBase::ECF_READ);
        if (!future.wait())
            return;
        if (auto lock = future.acquire(); lock)
            lock.discard();
    }

    static bool writeShaderCache(system::ISystem* system, const std::filesystem::path& path, const IShaderCompiler::CCache& cache)
    {
        auto buffer = cache.serialize();
        if (!buffer)
            return false;
        return writeBinaryFile(system, path, buffer->getPointer(), buffer->getSize());
    }


    static std::vector<std::string> expandJoinedArgs(const std::vector<std::string>& args)
    {
        std::vector<std::string> out;
        out.reserve(args.size());

        auto split = [&](const std::string& a, const char* p)
        {
            const size_t n = std::strlen(p);
            if (a.rfind(p, 0) == 0 && a.size() > n)
            {
                out.emplace_back(p);
                out.emplace_back(a.substr(n));
                return true;
            }
            return false;
        };

        for (const auto& a : args)
        {
            if (split(a, "-I")) continue;
            if (split(a, "-MF")) continue;
            if (split(a, "-Fo")) continue;
            if (split(a, "-Fc")) continue;
            out.push_back(a);
        }

        return out;
    }

    void dumpBuildInfo(const argparse::ArgumentParser& program)
    {
        json j;
        auto& modules = j["modules"];

        auto serialize = [&](const gtml::GitInfo& info, std::string_view target)
        {
            auto& s = modules[target.data()];
            s["isPopulated"] = info.isPopulated;
            s["hasUncommittedChanges"] = info.hasUncommittedChanges.has_value() ? json(info.hasUncommittedChanges.value()) : json("UNKNOWN, BUILT WITHOUT DIRTY-CHANGES CAPTURE");
            s["commitAuthorName"] = info.commitAuthorName;
            s["commitAuthorEmail"] = info.commitAuthorEmail;
            s["commitHash"] = info.commitHash;
            s["commitShortHash"] = info.commitShortHash;
            s["commitDate"] = info.commitDate;
            s["commitSubject"] = info.commitSubject;
            s["commitBody"] = info.commitBody;
            s["describe"] = info.describe;
            s["branchName"] = info.branchName;
            s["latestTag"] = info.latestTag;
            s["latestTagName"] = info.latestTagName;
        };

        serialize(gtml::nabla_git_info, "nabla");
        serialize(gtml::dxc_git_info, "dxc");

        const auto pretty = j.dump(4);
        std::cout << pretty << std::endl;

        std::filesystem::path oPath = "build-info.json";
        if (program.is_used("--file"))
        {
            const auto filePath = program.get<std::string>("--file");
            if (!filePath.empty())
                oPath = filePath;
        }

        if (!m_system)
        {
            if (m_logger)
                m_logger->log("Failed to create system for writing \"%s\"", ILogger::ELL_ERROR, oPath.string().c_str());
            std::exit(-1);
        }

        if (!writeBinaryFile(m_system.get(), oPath, pretty.data(), pretty.size()))
        {
            if (m_logger)
                m_logger->log("Failed to write \"%s\"", ILogger::ELL_ERROR, oPath.string().c_str());
            std::exit(-1);
        }

        if (m_logger)
            m_logger->log("Saved \"%s\"", ILogger::ELL_INFO, oPath.string().c_str());
    }

    RunResult runShaderJob(const IShader* shader, hlsl::ShaderStage shaderStage, std::string_view sourceIdentifier, const DepfileConfig& dep, const ShaderCacheConfig& shaderCache, const PreprocessCacheConfig& preCache, const bool preprocessOnly, std::string_view outputFilepath, std::string_view preprocessedOutputPath, const bool verbose)
    {
        RunResult r;
        auto makeIncludeFinder = [&]()
        {
            auto finder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
            auto loader = finder->getDefaultFileSystemLoader();
            for (const auto& p : m_include_search_paths)
                finder->addSearchPath(p, loader);
            return finder;
        };

        const char* codePtr = (const char*)shader->getContent()->getPointer();
        const size_t codeSize = shader->getContent()->getSize();
        std::string_view code(codePtr, codeSize);
        if (!code.empty() && code.back() == '\0')
            code.remove_suffix(1);
        CHLSLCompiler::SPreprocessorOptions preOpt = {};
        preOpt.sourceIdentifier = sourceIdentifier;
        preOpt.logger = m_logger.get();
        preOpt.forceIncludes = std::span<const std::string>(m_force_includes);
        preOpt.depfile = false;
        preOpt.depfilePath = dep.path;
        preOpt.codeForCache = code;

        CHLSLCompiler::SOptions opt = {};
        opt.stage = static_cast<IShader::E_SHADER_STAGE>(shaderStage);
        opt.preprocessorOptions = preOpt;
        opt.debugInfoFlags = bitflag<IShaderCompiler::E_DEBUG_INFO_FLAGS>(IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT);
        opt.dxcOptions = std::span<std::string>(m_arguments);

        auto writeTextFile = [&](std::string_view path, std::string_view contents) -> bool
        {
            if (path.empty())
                return false;
            return writeBinaryFile(m_system.get(), std::filesystem::path(std::string(path)), contents.data(), contents.size());
        };

        const bool useShaderCache = shaderCache.enabled && !preprocessOnly;
        const bool usePreCache = preCache.enabled && !preprocessOnly;
        const bool validateCacheDeps = true;

        struct ShaderCacheProbeResult
        {
            CacheLoadStatus status = CacheLoadStatus::Missing;
            bool hit = false;
            bool entryReady = false;
            bool depsUpdated = false;
            smart_refctd_ptr<IShaderCompiler::CCache> cacheObj;
            IShaderCompiler::CCache::SEntry entry;
            std::chrono::nanoseconds duration = {};
            std::chrono::nanoseconds loadDuration = {};
            std::chrono::nanoseconds validateDuration = {};
        };

        struct PreprocessCacheProbeResult
        {
            bool skipped = false;
            bool updateSkipped = false;
            bool ok = false;
            IShaderCompiler::SPreprocessCacheResult result = {};
            IShaderCompiler::CPreprocessCache::ELoadStatus loadStatus = IShaderCompiler::CPreprocessCache::ELoadStatus::Missing;
            smart_refctd_ptr<IShaderCompiler::CPreprocessCache> cacheObj;
            std::chrono::nanoseconds duration = {};
        };

        ShaderCacheProbeResult shaderProbe;
        PreprocessCacheProbeResult preProbe;
        using clock_t = std::chrono::high_resolution_clock;
        const auto probeStart = clock_t::now();

        if (useShaderCache)
        {
            const auto start = clock_t::now();
            const auto loadStart = clock_t::now();
            shaderProbe.cacheObj = loadShaderCache(m_system.get(), shaderCache.path, shaderProbe.status, false);
            const auto loadEnd = clock_t::now();
            if (!shaderProbe.cacheObj)
                shaderProbe.cacheObj = make_smart_refctd_ptr<IShaderCompiler::CCache>();
            if (shaderProbe.status == CacheLoadStatus::Loaded)
            {
                auto finder = makeIncludeFinder();
                const auto validateStart = clock_t::now();
                shaderProbe.hit = shaderProbe.cacheObj->findEntryForCode(code, opt, finder.get(), shaderProbe.entry, validateCacheDeps, &shaderProbe.depsUpdated);
                const auto validateEnd = clock_t::now();
                shaderProbe.entryReady = shaderProbe.hit;
                shaderProbe.validateDuration = validateEnd - validateStart;
            }
            shaderProbe.loadDuration = loadEnd - loadStart;
            shaderProbe.duration = clock_t::now() - start;
        }

        if (usePreCache)
        {
            if (useShaderCache && shaderProbe.hit)
            {
                preProbe.skipped = true;
                preProbe.ok = true;
                preProbe.duration = {};
            }
            else
            {
                const auto start = clock_t::now();
                auto finder = makeIncludeFinder();
                preProbe.cacheObj = IShaderCompiler::CPreprocessCache::loadFromFile(preCache.path, preProbe.loadStatus, false);
                if (!preProbe.cacheObj)
                    preProbe.cacheObj = make_smart_refctd_ptr<IShaderCompiler::CPreprocessCache>();

                auto localCompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));
                CHLSLCompiler::SPreprocessorOptions preOptThread = preOpt;
                preOptThread.includeFinder = finder.get();
                IShader::E_SHADER_STAGE stageOverrideThread = static_cast<IShader::E_SHADER_STAGE>(shaderStage);
                preProbe.result = localCompiler->preprocessWithCache(code, stageOverrideThread, preOptThread, *preProbe.cacheObj, preProbe.loadStatus, sourceIdentifier);
                preProbe.ok = preProbe.result.ok;
                preProbe.duration = clock_t::now() - start;
            }
        }

        const auto probeEnd = clock_t::now();

        std::string preprocessedCode;
        bool preprocessedReady = false;
        std::string_view codeToCompile = code;
        smart_refctd_ptr<IShaderCompiler::CPreprocessCache> preCacheObj;
        IShader::E_SHADER_STAGE stageOverride = static_cast<IShader::E_SHADER_STAGE>(shaderStage);
        auto cacheMissReason = [](CacheLoadStatus status) -> const char*
        {
            if (status == CacheLoadStatus::Missing)
                return "cache file missing; first build, cleaned, output moved, or out of date";
            if (status == CacheLoadStatus::Invalid)
                return "cache file invalid or version mismatch";
            return "input/deps/options changed; cache invalidated";
        };

        auto toMs = [](const std::chrono::nanoseconds duration) -> long long
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        };

        auto writeDepfileFromDependencies = [&](const IShaderCompiler::CCache::SEntry::dependency_container_t& dependencies, bool allowSkipIfExists) -> bool
        {
            if (!dep.enabled)
                return true;
            if (preOpt.depfilePath.empty())
            {
                m_logger->log("Depfile path is empty.", ILogger::ELL_ERROR);
                return false;
            }
            if (allowSkipIfExists && m_system && m_system->exists(preOpt.depfilePath, IFileBase::ECF_READ))
                return true;

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

            std::vector<std::string> depPaths;
            depPaths.reserve(dependencies.size() + 1);

            auto addDepPath = [&](std::filesystem::path path)
            {
                if (path.empty())
                    return;
                if (path.is_relative())
                    return;
                auto normalized = path.generic_string();
                if (normalized.empty() || normalized.find_first_of("\r\n") != std::string::npos)
                    return;
                depPaths.emplace_back(std::move(normalized));
            };

            if (!preOpt.sourceIdentifier.empty())
                addDepPath(std::filesystem::path(std::string(preOpt.sourceIdentifier)));

            for (const auto& depEntry : dependencies)
            {
                if (!depEntry.getHasFileInfo())
                    continue;
                const auto& absPath = depEntry.getAbsolutePath();
                if (absPath.empty())
                    continue;
                addDepPath(absPath);
            }

            std::sort(depPaths.begin(), depPaths.end());
            depPaths.erase(std::unique(depPaths.begin(), depPaths.end()), depPaths.end());

            std::filesystem::path targetPath = preOpt.depfilePath;
            if (targetPath.extension() == ".d")
                targetPath.replace_extension();
            const std::string target = escapeDepPath(targetPath.generic_string());

            std::string depfileContents;
            depfileContents.append(target);
            depfileContents.append(":");
            if (!depPaths.empty())
            {
                depfileContents.append(" \\\n");
                for (size_t index = 0; index < depPaths.size(); ++index)
                {
                    depfileContents.append(" ");
                    depfileContents.append(escapeDepPath(depPaths[index]));
                    if (index + 1 < depPaths.size())
                        depfileContents.append(" \\\n");
                }
            }
            depfileContents.append("\n");

            return writeBinaryFile(m_system.get(), std::filesystem::path(preOpt.depfilePath), depfileContents.data(), depfileContents.size());
        };

        auto isOutputUpToDate = [&](const IShaderCompiler::CCache::SEntry& entry) -> bool
        {
            if (outputFilepath.empty())
                return false;
            uint64_t outSize = 0;
            int64_t outTime = 0;
            if (!getFileInfo(m_system.get(), std::filesystem::path(outputFilepath), outSize, outTime))
                return false;
            if (entry.uncompressedSize == 0 || outSize != entry.uncompressedSize)
                return false;
            const auto hashPath = makeOutputHashPath(std::filesystem::path(outputFilepath));
            OutputHashRecord record = {};
            const bool hashOk = readBinaryFile(m_system.get(), hashPath, &record, sizeof(record));
            if (!hashOk || record.size != entry.uncompressedSize || record.hash != entry.uncompressedContentHash)
                return false;
            uint64_t hashSize = 0;
            int64_t hashTime = 0;
            if (!getFileInfo(m_system.get(), hashPath, hashSize, hashTime))
                return false;
            return outTime <= hashTime;
        };

        if (verbose && (useShaderCache || usePreCache))
        {
            if (useShaderCache)
            {
                if (shaderProbe.loadDuration.count())
                    m_logger->log("Shader cache load took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(toMs(shaderProbe.loadDuration)));
                if (shaderProbe.validateDuration.count())
                    m_logger->log("Shader cache validate took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(toMs(shaderProbe.validateDuration)));
                m_logger->log("Shader cache lookup took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(toMs(shaderProbe.duration)));
            }
            if (usePreCache)
                m_logger->log("Preprocess cache lookup took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(toMs(preProbe.duration)));
            m_logger->log("Total cache probe took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(toMs(std::chrono::duration_cast<std::chrono::nanoseconds>(probeEnd - probeStart))));
        }

        smart_refctd_ptr<IShaderCompiler::CCache> cacheObj = shaderProbe.cacheObj;
        if (!cacheObj && dep.enabled && !preprocessOnly)
            cacheObj = make_smart_refctd_ptr<IShaderCompiler::CCache>();
        CacheLoadStatus cacheStatus = shaderProbe.status;
        const bool shaderCacheHitExpected = shaderProbe.hit;

        if (usePreCache && preCache.verbose && useShaderCache)
        {
            if (shaderCacheHitExpected)
                m_logger->log("Cache hit! Preprocess cache skipped.", ILogger::ELL_DEBUG);
            else
                m_logger->log("Cache miss! Cold run (%s). Checking preprocess cache.", ILogger::ELL_DEBUG, cacheMissReason(cacheStatus));
        }

        if (usePreCache && !shaderCacheHitExpected)
        {
            if (!preProbe.ok)
                return r;
            if (preCache.verbose)
            {
                if (preProbe.result.cacheHit)
                    m_logger->log("Preprocess cache hit!", ILogger::ELL_DEBUG);
                else
                    m_logger->log("Preprocess cache miss! Cold run (%s).", ILogger::ELL_DEBUG, IShaderCompiler::CPreprocessCache::getProbeReason(preProbe.result.status));
            }
            if (preProbe.result.cacheUsed)
            {
                preprocessedCode = std::move(preProbe.result.code);
                preprocessedReady = true;
                stageOverride = preProbe.result.stage;
                preCacheObj = preProbe.cacheObj;
                if (!preprocessedOutputPath.empty() && !writeTextFile(preprocessedOutputPath, preprocessedCode))
                    return r;
            }
        }
        else if (usePreCache && preCache.verbose)
        {
            if (preProbe.skipped)
            {
                m_logger->log("Preprocess cache lookup skipped (shader cache hit).", ILogger::ELL_DEBUG);
            }
            else if (preProbe.ok)
            {
                if (preProbe.result.cacheHit)
                    m_logger->log("Preprocess cache hit (ignored, shader cache hit).", ILogger::ELL_DEBUG);
                else
                    m_logger->log("Preprocess cache miss! Cold run (%s). (ignored, shader cache hit).", ILogger::ELL_DEBUG, IShaderCompiler::CPreprocessCache::getProbeReason(preProbe.result.status));
            }
            else
            {
                m_logger->log("Preprocess cache failed (ignored, shader cache hit).", ILogger::ELL_DEBUG);
            }
        }
        if (usePreCache && preProbe.result.cacheUpdated && preProbe.cacheObj)
            IShaderCompiler::CPreprocessCache::writeToFile(preCache.path, *preProbe.cacheObj);

        if (useShaderCache && shaderProbe.hit && shaderProbe.entryReady)
        {
            if (verbose)
                m_logger->log("Shader cache hit: using cached SPIR-V.", ILogger::ELL_DEBUG);
            if (shaderProbe.depsUpdated)
            {
                const auto cacheWriteStart = clock_t::now();
                if (!writeShaderCache(m_system.get(), shaderCache.path, *cacheObj))
                    m_logger->log("Failed to write shader cache: %s", ILogger::ELL_WARNING, shaderCache.path.string().c_str());
                if (verbose)
                {
                    const auto cacheWriteEnd = clock_t::now();
                    m_logger->log("Shader cache write took: %lld ms.", ILogger::ELL_PERFORMANCE,
                        static_cast<long long>(toMs(cacheWriteEnd - cacheWriteStart)));
                }
            }
            if (isOutputUpToDate(shaderProbe.entry))
            {
                const auto hitDepfileStart = clock_t::now();
                if (!writeDepfileFromDependencies(shaderProbe.entry.dependencies, true))
                    return r;
                const auto hitDepfileEnd = clock_t::now();
                r.ok = true;
                if (verbose)
                {
                    m_logger->log("HIT timings: decompress=0 ms, depfile=%lld ms.", ILogger::ELL_PERFORMANCE,
                        static_cast<long long>(toMs(hitDepfileEnd - hitDepfileStart)));
                }
                return r;
            }
            const auto hitDecompressStart = clock_t::now();
            r.compiled = cacheObj->decompressEntry(shaderProbe.entry);
            const auto hitDecompressEnd = clock_t::now();
            r.ok = bool(r.compiled);
            if (!r.ok)
                return r;
            r.view = { (const char*)r.compiled->getContent()->getPointer(), r.compiled->getContent()->getSize() };

            const auto hitDepfileStart = clock_t::now();
            if (!writeDepfileFromDependencies(shaderProbe.entry.dependencies, true))
                return r;
            const auto hitDepfileEnd = clock_t::now();
            if (verbose)
            {
                m_logger->log("HIT timings: decompress=%lld ms, depfile=%lld ms.", ILogger::ELL_PERFORMANCE,
                    static_cast<long long>(toMs(hitDecompressEnd - hitDecompressStart)),
                    static_cast<long long>(toMs(hitDepfileEnd - hitDepfileStart)));
            }

            return r;
        }

        auto hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

        if (preprocessOnly)
        {
            const auto preprocessStart = std::chrono::high_resolution_clock::now();
            auto finder = makeIncludeFinder();
            preOpt.includeFinder = finder.get();
            r.text = hlslcompiler->preprocessShader(std::string(code), shaderStage, preOpt, nullptr);
            r.ok = !r.text.empty();
            r.view = r.text;
            const auto preprocessEnd = std::chrono::high_resolution_clock::now();
            if (verbose)
            {
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(preprocessEnd - preprocessStart).count();
                m_logger->log("Preprocess took: %lld ms.", ILogger::ELL_PERFORMANCE, static_cast<long long>(duration));
            }
            return r;
        }

        opt.stage = stageOverride;

        bool cacheHit = false;
        if (shaderCache.enabled && cacheObj)
        {
            opt.readCache = cacheObj.get();
            opt.writeCache = cacheObj.get();
            opt.cacheHit = &cacheHit;
        }
        else if (dep.enabled && cacheObj)
        {
            opt.writeCache = cacheObj.get();
        }

        if (preprocessedReady)
        {
            opt.preprocessorOptions.applyForceIncludes = false;
            if (preCacheObj && preCacheObj->hasEntry())
                opt.dependencyOverrides = &preCacheObj->getEntry().dependencies;
            codeToCompile = preprocessedCode;
        }

        auto compileFinder = makeIncludeFinder();
        opt.preprocessorOptions.includeFinder = compileFinder.get();
        const auto compileStart = clock_t::now();
        r.compiled = hlslcompiler->compileToSPIRV(codeToCompile, opt);
        const auto compileEnd = clock_t::now();
        r.ok = bool(r.compiled);
        if (r.ok)
            r.view = { (const char*)r.compiled->getContent()->getPointer(), r.compiled->getContent()->getSize() };

        if (shaderCache.enabled && cacheObj)
        {
            const bool logShaderCache = verbose && !usePreCache;
            if (logShaderCache)
            {
                if (cacheHit)
                {
                    m_logger->log("Cache hit!", ILogger::ELL_DEBUG);
                }
                else
                {
                    m_logger->log("Cache miss! Cold run (%s).", ILogger::ELL_DEBUG, cacheMissReason(cacheStatus));
                }
            }
            if (!writeShaderCache(m_system.get(), shaderCache.path, *cacheObj))
                m_logger->log("Failed to write shader cache: %s", ILogger::ELL_WARNING, shaderCache.path.string().c_str());
        }

        if (dep.enabled && r.ok)
        {
            const IShaderCompiler::CCache::SEntry::dependency_container_t* deps = nullptr;
            IShaderCompiler::CCache::SEntry depEntry;
            if (preCacheObj && preCacheObj->hasEntry())
            {
                deps = &preCacheObj->getEntry().dependencies;
            }
            else if (cacheObj)
            {
                if (cacheObj->findEntryForCode(code, opt, compileFinder.get(), depEntry, validateCacheDeps))
                    deps = &depEntry.dependencies;
            }

            if (!deps)
            {
                m_logger->log("Depfile requested but dependencies unavailable.", ILogger::ELL_ERROR);
                r.ok = false;
                return r;
            }

            if (!writeDepfileFromDependencies(*deps, false))
            {
                r.ok = false;
                return r;
            }
        }

        return r;
    }

    std::tuple<smart_refctd_ptr<const IShader>, hlsl::ShaderStage> open_shader_file(std::string filepath)
    {
        m_assetMgr = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(m_system));

        IAssetLoader::SAssetLoadParams lp = {};
        lp.logger = m_logger.get();
        lp.workingDirectory = localInputCWD;

        auto bundle = m_assetMgr->getAsset(filepath, lp);
        const auto assets = bundle.getContents();
        const auto* metadata = bundle.getMetadata();

        if (assets.empty())
        {
            m_logger->log("Could not load shader %s", ILogger::ELL_ERROR, filepath.c_str());
            return { nullptr, hlsl::ShaderStage::ESS_UNKNOWN };
        }

        if (bundle.getAssetType() == IAsset::ET_BUFFER)
        {
            auto buf = IAsset::castDown<ICPUBuffer>(assets[0]);
            std::string source;
            source.resize(buf->getSize() + 1);
            std::memcpy(source.data(), buf->getPointer(), buf->getSize());
            return { make_smart_refctd_ptr<IShader>(source.data(), IShader::E_CONTENT_TYPE::ECT_HLSL, std::move(filepath)), hlsl::ShaderStage::ESS_UNKNOWN };
        }

        if (bundle.getAssetType() == IAsset::ET_SHADER)
        {
            const auto hlslMetadata = static_cast<const CHLSLMetadata*>(metadata);
            return { smart_refctd_ptr_static_cast<IShader>(assets[0]), hlslMetadata->shaderStages->front() };
        }

        m_logger->log("file '%s' is an asset that is neither a buffer or a shader.", ILogger::ELL_ERROR, filepath.c_str());
        return { nullptr, hlsl::ShaderStage::ESS_UNKNOWN };
    }

    smart_refctd_ptr<ISystem> m_system;
    smart_refctd_ptr<ILogger> m_logger;
    std::vector<std::string> m_arguments, m_include_search_paths, m_force_includes;
    smart_refctd_ptr<IAssetManager> m_assetMgr;
};

NBL_MAIN_FUNC(ShaderCompiler)

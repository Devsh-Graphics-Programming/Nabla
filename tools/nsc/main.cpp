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
#include <cstring>
#include <cstdarg>
#include <argparse/argparse.hpp>
#include "nbl/asset/metadata/CHLSLMetadata.h"
#include "nlohmann/json.hpp"

using json = ::nlohmann::json;
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

        const auto parent = m_logPath.parent_path();
        if (!parent.empty() && !m_system->isDirectory(parent))
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

static bool writeFileWithSystem(ISystem* system, const path& outputPath, const std::string_view content)
{
    if (!system || outputPath.empty())
        return false;

    const auto parent = outputPath.parent_path();
    if (!parent.empty() && !system->isDirectory(parent))
    {
        if (!system->createDirectory(parent))
            return false;
    }

    auto tempPath = outputPath;
    tempPath += ".tmp";
    system->deleteFile(tempPath);

    smart_refctd_ptr<IFile> outputFile;
    {
        ISystem::future_t<smart_refctd_ptr<IFile>> future;
        system->createFile(future, tempPath, IFileBase::ECF_WRITE);
        if (!future.wait())
            return false;

        auto lock = future.acquire();
        if (!lock)
            return false;
        lock.move_into(outputFile);
    }
    if (!outputFile)
        return false;

    if (!content.empty())
    {
        IFile::success_t success;
        outputFile->write(success, content.data(), 0ull, content.size());
        if (!success)
        {
            outputFile = nullptr;
            system->deleteFile(tempPath);
            return false;
        }
    }
    outputFile = nullptr;

    system->deleteFile(outputPath);
    const auto moveError = system->moveFileOrDirectory(tempPath, outputPath);
    if (moveError)
    {
        system->deleteFile(tempPath);
        return false;
    }

    return true;
}

class ShaderCompiler final : public IApplicationFramework
{
    using base_t = IApplicationFramework;

public:
    using base_t::base_t;

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        const auto rawArgs = std::vector<std::string>(argv.begin(), argv.end());
        const auto expandedArgs = expandJoinedArgs(rawArgs);

        argparse::ArgumentParser program("nsc");
        program.add_argument("--dump-build-info").default_value(false).implicit_value(true);
        program.add_argument("--self-test-unmount-builtins").default_value(false).implicit_value(true);
        program.add_argument("--dump-preprocessed-on-failure").default_value(false).implicit_value(true);
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

        std::vector<std::string> unknownArgs;
        try
        {
            unknownArgs = program.parse_known_args(expandedArgs);
        }
        catch (const std::runtime_error& err)
        {
            std::cerr << err.what() << std::endl << program;
            return false;
        }

        m_system = system ? std::move(system) : IApplicationFramework::createSystem();
        if (!m_system)
            return false;

        if (program.get<bool>("--dump-build-info"))
        {
            return dumpBuildInfo(program);
        }

        if (!isAPILoaded())
        {
            std::cerr << "Could not load Nabla API, terminating!";
            return false;
        }

        if (program.get<bool>("--self-test-unmount-builtins"))
        {
            const auto mountedBuiltinArchiveCount = m_system->getMountedBuiltinArchiveCount();
            if (!mountedBuiltinArchiveCount)
            {
                std::cerr << "Builtins were not mounted at startup. builtin_mount_count=0 total_mount_count=" << m_system->getMountedArchiveCount() << "\n";
                return false;
            }

            m_system->unmountBuiltins();

            if (const auto remainingBuiltinArchiveCount = m_system->getMountedBuiltinArchiveCount(); remainingBuiltinArchiveCount != 0ull)
            {
                std::cerr << "Builtins unmount self-test failed. remaining_builtin_mount_count=" << remainingBuiltinArchiveCount
                          << " total_mount_count=" << m_system->getMountedArchiveCount() << "\n";
                return false;
            }

            return true;
        }

        if (rawArgs.size() < 2)
        {
            std::cerr << "Insufficient arguments.\n";
            return false;
        }

        const std::string fileToCompile = rawArgs.back();
        if (!m_system->exists(fileToCompile, IFileBase::ECF_READ))
        {
            std::cerr << "Input shader file does not exist: " << fileToCompile << "\n";
            return false;
        }

        const bool preprocessOnly = program.get<bool>("-P");
        const bool dumpPreprocessedOnFailure = program.get<bool>("--dump-preprocessed-on-failure");
        const bool hasFc = program.is_used("-Fc");
        const bool hasFo = program.is_used("-Fo");

        if (hasFc == hasFo)
        {
            if (hasFc)
                std::cerr << "Invalid arguments. Passed both -Fo and -Fc.\n";
            else
                std::cerr << "Missing arguments. Expecting `-Fc {filename}` or `-Fo {filename}`.\n";
            return false;
        }
        if (dumpPreprocessedOnFailure && !preprocessOnly)
        {
            std::cerr << "Invalid arguments. --dump-preprocessed-on-failure requires -P.\n";
            return false;
        }

        const std::string outputFilepath = hasFc ? program.get<std::string>("-Fc") : program.get<std::string>("-Fo");
        if (outputFilepath.empty())
        {
            std::cerr << "Invalid output file path.\n";
            return false;
        }

        const bool quiet = program.get<bool>("-quiet");
        const bool verbose = program.get<bool>("-verbose");
        if (quiet && verbose)
        {
            std::cerr << "Invalid arguments. Passed both -quiet and -verbose.\n";
            return false;
        }

        const bool noLog = program.get<bool>("-nolog");
        const std::string logPathOverride = program.is_used("-log") ? program.get<std::string>("-log") : std::string{};
        if (noLog && !logPathOverride.empty())
        {
            std::cerr << "Invalid arguments. Passed both -nolog and -log.\n";
            return false;
        }

        const auto logPath = logPathOverride.empty() ? path(outputFilepath).concat(".log") : path(logPathOverride);
        const auto fileMask = bitflag(ILogger::ELL_ALL);
        const auto consoleMask = bitflag(ILogger::ELL_WARNING) | ILogger::ELL_ERROR;

        m_logger = make_smart_refctd_ptr<ShaderLogger>(m_system, logPath, fileMask, consoleMask, noLog);

        m_arguments = std::move(unknownArgs);
        if (!m_arguments.empty() && m_arguments.back() == fileToCompile)
            m_arguments.pop_back();

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
            dep.path = outputFilepath + ".d";
        if (dep.enabled)
            m_logger->log("Dependency file will be saved to %s", ILogger::ELL_INFO, dep.path.c_str());

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

        std::vector<std::string> normalizedArguments;
        normalizedArguments.reserve(m_arguments.size());
        for (size_t i = 0; i < m_arguments.size(); ++i)
        {
            const auto& argument = m_arguments[i];
            if ((argument == "-I" || argument == "-isystem") && i + 1 < m_arguments.size())
            {
                const auto classification = IShaderCompiler::IncludeClassification{
                    IShaderCompiler::IncludeRootOrigin::User,
                    argument == "-isystem" ? IShaderCompiler::HeaderClass::System : IShaderCompiler::HeaderClass::User
                };
                m_include_search_paths.push_back({ m_arguments[i + 1],classification });
                ++i;
                continue;
            }

            normalizedArguments.emplace_back(argument);
        }
        m_arguments = std::move(normalizedArguments);

        const char* const action = preprocessOnly ? "Preprocessing" : "Compiling";
        const char* const outType = preprocessOnly ? "Preprocessed" : "Compiled";
        m_logger->log("%s %s", ILogger::ELL_INFO, action, fileToCompile.c_str());
        m_logger->log("%s shader code will be saved to %s", ILogger::ELL_INFO, outType, outputFilepath.c_str());
        if (dumpPreprocessedOnFailure)
            m_logger->log("Partial preprocessed output will be written to %s if preprocessing fails.", ILogger::ELL_INFO, outputFilepath.c_str());

        auto [shader, shaderStage] = open_shader_file(fileToCompile);
        if (!shader || shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_HLSL)
        {
            m_logger->log("Error. Loaded shader file content is not HLSL.", ILogger::ELL_ERROR);
            return false;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        const auto job = runShaderJob(shader.get(), shaderStage, fileToCompile, dep, preprocessOnly, dumpPreprocessedOnFailure);
        const auto end = std::chrono::high_resolution_clock::now();

        const char* const op = preprocessOnly ? "preprocessing" : "compilation";
        if (!job.ok)
        {
            if (job.writeOutputOnFailure)
            {
                if (!writeOutputFile(outputFilepath, job.view, "partial preprocess dump"))
                    return false;

                if (job.view.empty())
                    m_logger->log("Shader preprocessing failed before emitting any output. Empty dump written to %s.", ILogger::ELL_WARNING, outputFilepath.c_str());
                else
                    m_logger->log("Shader preprocessing failed after emitting %zu bytes. Partial dump written to %s.", ILogger::ELL_WARNING, job.view.size(), outputFilepath.c_str());
            }
            m_logger->log("Shader %s failed.", ILogger::ELL_ERROR, op);
            return false;
        }

        const auto took = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        m_logger->log("Shader %s successful.", ILogger::ELL_INFO, op);
        m_logger->log("Took %s ms.", ILogger::ELL_PERFORMANCE, took.c_str());

        if (!writeOutputFile(outputFilepath, job.view, "output"))
            return false;

        if (dep.enabled)
            m_logger->log("Dependency file written to %s", ILogger::ELL_INFO, dep.path.c_str());

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

    struct RunResult
    {
        bool ok = false;
        bool writeOutputOnFailure = false;
        std::string text;
        smart_refctd_ptr<IShader> compiled;
        std::string_view view;
    };

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
            if (split(a, "-isystem")) continue;
            if (split(a, "-I")) continue;
            if (split(a, "-MF")) continue;
            if (split(a, "-Fo")) continue;
            if (split(a, "-Fc")) continue;
            out.push_back(a);
        }

        return out;
    }

    bool dumpBuildInfo(const argparse::ArgumentParser& program)
    {
        ::json j;
        auto& modules = j["modules"];

        auto serialize = [&](const gtml::GitInfo& info, std::string_view target)
        {
            auto& s = modules[target.data()];
            s["isPopulated"] = info.isPopulated;
            s["hasUncommittedChanges"] = info.hasUncommittedChanges.has_value() ? ::json(info.hasUncommittedChanges.value()) : ::json("UNKNOWN, BUILT WITHOUT DIRTY-CHANGES CAPTURE");
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

        path oPath = "build-info.json";
        if (program.is_used("--file"))
        {
            const auto filePath = program.get<std::string>("--file");
            if (!filePath.empty())
                oPath = filePath;
        }

        if (!writeFileWithSystem(m_system.get(), oPath, pretty))
        {
            std::printf("Failed to write \"%s\"\n", oPath.string().c_str());
            return false;
        }

        std::printf("Saved \"%s\"\n", oPath.string().c_str());
        return true;
    }

    bool writeOutputFile(const std::string& outputFilepath, const std::string_view content, const char* const description)
    {
        if (!writeFileWithSystem(m_system.get(), outputFilepath, content))
        {
            m_logger->log("Failed to write %s file: %s", ILogger::ELL_ERROR, description, outputFilepath.c_str());
            return false;
        }

        return true;
    }

    RunResult runShaderJob(const IShader* shader, hlsl::ShaderStage shaderStage, std::string_view sourceIdentifier, const DepfileConfig& dep, const bool preprocessOnly, const bool dumpPreprocessedOnFailure)
    {
        RunResult r;
        auto hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

        auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
        auto includeLoader = includeFinder->getDefaultFileSystemLoader();
        for (const auto& searchPath : m_include_search_paths)
            includeFinder->addSearchPath(searchPath.path, includeLoader, searchPath.classification);

        // need this struct becuase fields of IShaderCompiler::SMacroDefinition are string views
        struct SMacroDefinitionBuffer
        {
            std::string identifier;
            std::string definition;
        };

        core::vector<SMacroDefinitionBuffer> macroDefinitionBuffers;
        core::vector<IShaderCompiler::SMacroDefinition> macroDefinitions;
        {
            for (const auto& argument : m_arguments)
            {
                if (argument.rfind("-D", 0) != 0)
                    continue;

                std::string argumentTmp = argument.substr(2);

                std::string identifier;
                std::string definition;

                const size_t equalPos = argumentTmp.find('=');
                if (equalPos == std::string::npos)
                {
                    identifier = argumentTmp;
                    definition = "1";
                }
                else
                {
                    identifier = argumentTmp.substr(0, equalPos);
                    definition = argumentTmp.substr(equalPos + 1);
                }

                macroDefinitionBuffers.emplace_back(identifier, definition);
            }

            macroDefinitions.reserve(macroDefinitionBuffers.size());

            for (const auto& macroDefinitionBuffer : macroDefinitionBuffers)
                macroDefinitions.emplace_back(macroDefinitionBuffer.identifier, macroDefinitionBuffer.definition);
        }

        if (preprocessOnly)
        {
            CHLSLCompiler::SPreprocessorOptions opt = {};
            opt.sourceIdentifier = sourceIdentifier;
            opt.logger = m_logger.get();
            opt.includeFinder = includeFinder.get();
            opt.depfile = dep.enabled;
            opt.depfilePath = dep.path;
            opt.extraDefines = macroDefinitions;

            const char* codePtr = (const char*)shader->getContent()->getPointer();
            std::string_view code(codePtr, std::strlen(codePtr));
            std::string partialOutputOnFailure;
            if (dumpPreprocessedOnFailure)
            {
                opt.onPartialOutputOnFailure = [&](const std::string_view partialOutput)
                {
                    partialOutputOnFailure.assign(partialOutput);
                };
            }

            r.text = hlslcompiler->preprocessShader(std::string(code), shaderStage, opt, nullptr);
            r.ok = !r.text.empty();
            if (!r.ok && dumpPreprocessedOnFailure)
            {
                r.text = std::move(partialOutputOnFailure);
                r.writeOutputOnFailure = true;
            }
            r.view = r.text;
            return r;
        }

        CHLSLCompiler::SOptions opt = {};
        opt.stage = shaderStage;
        opt.preprocessorOptions.sourceIdentifier = sourceIdentifier;
        opt.preprocessorOptions.logger = m_logger.get();
        opt.preprocessorOptions.includeFinder = includeFinder.get();
        opt.preprocessorOptions.depfile = dep.enabled;
        opt.preprocessorOptions.depfilePath = dep.path;
        opt.preprocessorOptions.extraDefines = macroDefinitions;
        opt.debugInfoFlags = bitflag<IShaderCompiler::E_DEBUG_INFO_FLAGS>(IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT);
        opt.dxcOptions = std::span<std::string>(m_arguments);

        r.compiled = hlslcompiler->compileToSPIRV((const char*)shader->getContent()->getPointer(), opt);
        r.ok = bool(r.compiled);
        if (r.ok)
            r.view = { (const char*)r.compiled->getContent()->getPointer(), r.compiled->getContent()->getSize() };

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
    struct SearchPathArgument
    {
        std::string path;
        IShaderCompiler::IncludeClassification classification = {};
    };

    std::vector<std::string> m_arguments;
    std::vector<SearchPathArgument> m_include_search_paths;
    smart_refctd_ptr<IAssetManager> m_assetMgr;
};

NBL_MAIN_FUNC(ShaderCompiler)

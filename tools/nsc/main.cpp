#include "nabla.h"
#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <argparse/argparse.hpp>

#include "nbl/asset/metadata/CHLSLMetadata.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

using namespace nbl;
using namespace nbl::system;
using namespace nbl::core;
using namespace nbl::asset;

class NscLogger final : public system::IThreadsafeLogger
{
public:
	NscLogger(core::smart_refctd_ptr<system::IFile>&& logFile, const core::bitflag<E_LOG_LEVEL> logLevelMask, const core::bitflag<E_LOG_LEVEL> consoleMask)
		: IThreadsafeLogger(logLevelMask), m_logFile(std::move(logFile)), m_logPos(m_logFile ? m_logFile->getSize() : 0ull), m_consoleMask(consoleMask)
	{
	}

private:
	void threadsafeLog_impl(const std::string_view& fmt, E_LOG_LEVEL logLevel, va_list args) override
	{
		const auto line = constructLogString(fmt, logLevel, args);
		size_t lineSize = line.size();
		while (lineSize > 0 && line[lineSize - 1] == '\0')
			--lineSize;
		if (lineSize == 0)
			return;
		if (m_logFile)
		{
			system::IFile::success_t succ;
			m_logFile->write(succ, line.data(), m_logPos, lineSize);
			m_logPos += succ.getBytesProcessed();
		}
		if (logLevel & m_consoleMask.value)
		{
			std::fwrite(line.data(), 1, lineSize, stdout);
			std::fflush(stdout);
		}
	}

	core::smart_refctd_ptr<system::IFile> m_logFile;
	size_t m_logPos = 0ull;
	core::bitflag<E_LOG_LEVEL> m_consoleMask;
};

class ShaderCompiler final : public system::IApplicationFramework
{
	using base_t = system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		const auto rawArgs = std::vector<std::string>(argv.begin(), argv.end());
		auto expandArgs = [](const std::vector<std::string>& args)
		{
			std::vector<std::string> expanded;
			expanded.reserve(args.size());
			for (const auto& arg : args)
			{
				if (arg.rfind("-MF", 0) == 0 && arg.size() > 3)
				{
					expanded.push_back("-MF");
					expanded.push_back(arg.substr(3));
					continue;
				}
				if (arg.rfind("-Fo", 0) == 0 && arg.size() > 3)
				{
					expanded.push_back("-Fo");
					expanded.push_back(arg.substr(3));
					continue;
				}
				if (arg.rfind("-Fc", 0) == 0 && arg.size() > 3)
				{
					expanded.push_back("-Fc");
					expanded.push_back(arg.substr(3));
					continue;
				}
				expanded.push_back(arg);
			}
			return expanded;
		};

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

		std::vector<std::string> unknownArgs;
		try
		{
			unknownArgs = program.parse_known_args(expandArgs(rawArgs));
		}
		catch (const std::runtime_error& err)
		{
			std::cerr << err.what() << std::endl << program;
			return false;
		}

		if (program.get<bool>("--dump-build-info"))
		{
			json j;

			auto& modules = j["modules"];

			auto serialize = [&](const gtml::GitInfo& info, std::string_view target) -> void 
			{
				auto& s = modules[target.data()];

				s["isPopulated"] = info.isPopulated;
				if (info.hasUncommittedChanges.has_value()) 
					s["hasUncommittedChanges"] = info.hasUncommittedChanges.value();
				else
					s["hasUncommittedChanges"] = "UNKNOWN, BUILT WITHOUT DIRTY-CHANGES CAPTURE";

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

			std::ofstream outFile(oPath);
			if (outFile.is_open()) 
			{
				outFile << pretty;
				outFile.close();
				printf("Saved \"%s\"\n", oPath.string().c_str());
			}
			else
			{
				printf("Failed to open \"%s\" for writing\n", oPath.string().c_str());
				exit(-1);
			}

			exit(0);
		}

		if (not isAPILoaded())
		{
			std::cerr << "Could not load Nabla API, terminating!";
			return false;
		}

		if (system)
			m_system = std::move(system);
		else
			m_system = system::IApplicationFramework::createSystem();

		if (!m_system)
			return false;

		const auto defaultConsoleMask = core::bitflag(ILogger::ELL_WARNING) | ILogger::ELL_ERROR;
		m_logger = make_smart_refctd_ptr<CStdoutLogger>(defaultConsoleMask);

		if (rawArgs.size() < 2)
		{
			m_logger->log("Insufficient arguments.", ILogger::ELL_ERROR);
			return false;
		}
		std::string file_to_compile = rawArgs.back();

		if (!m_system->exists(file_to_compile, IFileBase::ECF_READ))
		{
			m_logger->log("Input shader file does not exist: %s", ILogger::ELL_ERROR, file_to_compile.c_str());
			return false;
		}

		const bool preprocessOnly = program.get<bool>("-P");
		const bool outputFlagFc = program.is_used("-Fc");
		const bool outputFlagFo = program.is_used("-Fo");
		if (outputFlagFc && outputFlagFo)
		{
			m_logger->log("Invalid arguments. Passed both -Fo and -Fc.", ILogger::ELL_ERROR);
			return false;
		}
		if (!outputFlagFc && !outputFlagFo)
		{
			m_logger->log("Missing arguments. Expecting `-Fc {filename}` or `-Fo {filename}`.", ILogger::ELL_ERROR);
			return false;
		}

		std::string output_filepath = outputFlagFc ? program.get<std::string>("-Fc") : program.get<std::string>("-Fo");
		if (output_filepath.empty())
		{
			m_logger->log("Invalid output file path.", ILogger::ELL_ERROR);
			return false;
		}

		const bool quietFlag = program.get<bool>("-quiet");
		const bool verboseFlag = program.get<bool>("-verbose");
		if (quietFlag && verboseFlag)
		{
			m_logger->log("Invalid arguments. Passed both -quiet and -verbose.", ILogger::ELL_ERROR);
			return false;
		}

		LogConfig logConfig;
		if (verboseFlag)
			logConfig.quiet = false;
		if (quietFlag)
			logConfig.quiet = true;

		logConfig.noLog = program.get<bool>("-nolog");
		if (program.is_used("-log"))
		{
			logConfig.path = program.get<std::string>("-log");
			if (logConfig.path.empty())
			{
				m_logger->log("Incorrect arguments. Expecting filename after -log.", ILogger::ELL_ERROR);
				return false;
			}
		}

		if (logConfig.noLog && !logConfig.path.empty())
		{
			m_logger->log("Invalid arguments. Passed both -nolog and -log.", ILogger::ELL_ERROR);
			return false;
		}

		const auto consoleMask = logConfig.quiet ? (core::bitflag(ILogger::ELL_WARNING) | ILogger::ELL_ERROR) : core::bitflag(ILogger::ELL_ALL);
		m_logger = make_smart_refctd_ptr<CStdoutLogger>(consoleMask);

		if (!logConfig.noLog)
		{
			const std::filesystem::path logPath = logConfig.path.empty() ? std::filesystem::path(output_filepath).concat(".log") : std::filesystem::path(logConfig.path);
			const auto parentDirectory = logPath.parent_path();
			if (!parentDirectory.empty() && !std::filesystem::exists(parentDirectory))
				std::filesystem::create_directories(parentDirectory);

			m_system->deleteFile(logPath);

			system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
			m_system->createFile(future, logPath, system::IFileBase::ECF_WRITE);
			core::smart_refctd_ptr<system::IFile> logFile;
			if (future.wait())
				future.acquire().move_into(logFile);

			if (logFile)
				m_logger = make_smart_refctd_ptr<NscLogger>(std::move(logFile), core::bitflag(ILogger::ELL_ALL), consoleMask);
			else
				m_logger->log("Failed to open log file: %s", ILogger::ELL_ERROR, logPath.string().c_str());
		}

		const char* action = preprocessOnly ? "Preprocessing" : "Compiling";
		m_logger->log("%s %s", ILogger::ELL_INFO, action, file_to_compile.c_str());
		const char* outputType = preprocessOnly ? "Preprocessed" : "Compiled";
		m_logger->log("%s shader code will be saved to %s", ILogger::ELL_INFO, outputType, output_filepath.c_str());

		m_arguments = std::move(unknownArgs);
		if (!m_arguments.empty() && m_arguments.back() == file_to_compile)
			m_arguments.pop_back();

		no_nbl_builtins = program.get<bool>("-no-nbl-builtins");
		if (no_nbl_builtins)
		{
			m_logger->log("Unmounting builtins.");
			m_system->unmountBuiltins();
		}

		DepfileConfig depfileConfig;
		if (program.get<bool>("-MD") || program.get<bool>("-M") || program.is_used("-MF"))
			depfileConfig.enabled = true;
		if (program.is_used("-MF"))
			depfileConfig.path = program.get<std::string>("-MF");
		if (depfileConfig.enabled && depfileConfig.path.empty())
			depfileConfig.path = output_filepath + ".d";
		if (depfileConfig.enabled)
			m_logger->log("Dependency file will be saved to %s", ILogger::ELL_INFO, depfileConfig.path.c_str());

#ifndef NBL_EMBED_BUILTIN_RESOURCES
		if (!no_nbl_builtins) {
			m_system->unmountBuiltins();
			no_nbl_builtins = true;
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
			const auto& arg = m_arguments[i];
			if (arg == "-I")
				m_include_search_paths.emplace_back(m_arguments[i + 1]);
		}

		auto [shader, shaderStage] = open_shader_file(file_to_compile);
		if (shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_HLSL)
		{
			m_logger->log("Error. Loaded shader file content is not HLSL.", ILogger::ELL_ERROR);
			return false;
		}

		auto start = std::chrono::high_resolution_clock::now();
		smart_refctd_ptr<IShader> compilation_result;
		std::string preprocessing_result;
		std::string_view result_view;
		if (preprocessOnly)
		{
			preprocessing_result = preprocess_shader(shader.get(), shaderStage, file_to_compile, depfileConfig);
			result_view = preprocessing_result;
		}
		else
		{
			compilation_result = compile_shader(shader.get(), shaderStage, file_to_compile, depfileConfig);
			result_view = { (const char*)compilation_result->getContent()->getPointer(), compilation_result->getContent()->getSize() };
		}
		auto end = std::chrono::high_resolution_clock::now();

		std::string operationType = preprocessOnly ? "preprocessing" : "compilation";
		const bool success = preprocessOnly ? preprocessing_result != std::string{} : bool(compilation_result);
		if (success) 
		{
			m_logger->log("Shader " + operationType + " successful.", ILogger::ELL_INFO);
			const auto took = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
			m_logger->log("Took %s ms.", ILogger::ELL_PERFORMANCE, took.c_str());
			{
				const auto location = std::filesystem::path(output_filepath);
				const auto parentDirectory = location.parent_path();

				if (!std::filesystem::exists(parentDirectory))
				{
					if (!std::filesystem::create_directories(parentDirectory))
					{
						m_logger->log("Failed to create parent directory for the " + output_filepath + "output!", ILogger::ELL_ERROR);
						return false;
					}
				}
			}

			std::fstream output_file(output_filepath, std::ios::out | std::ios::binary);

			if (!output_file.is_open()) 
			{
				m_logger->log("Failed to open output file: " + output_filepath, ILogger::ELL_ERROR);
				return false;
			}

			output_file.write(result_view.data(), result_view.size());

			if (output_file.fail()) 
			{
				m_logger->log("Failed to write to output file: " + output_filepath, ILogger::ELL_ERROR);
				output_file.close();
				return false;
			}

			output_file.close();

			if (output_file.fail()) 
			{
				m_logger->log("Failed to close output file: " + output_filepath, ILogger::ELL_ERROR);
				return false;
			}

			if (depfileConfig.enabled)
				m_logger->log("Dependency file written to %s", ILogger::ELL_INFO, depfileConfig.path.c_str());

			return true;
		}
		else 
		{
			m_logger->log("Shader " + operationType + " failed.", ILogger::ELL_ERROR);
			return false;
		}
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }


private:

	struct LogConfig
	{
		bool quiet = true;
		bool noLog = false;
		std::string path;
	};

	struct DepfileConfig
	{
		bool enabled = false;
		std::string path;
	};

	std::string preprocess_shader(const IShader* shader, hlsl::ShaderStage shaderStage, std::string_view sourceIdentifier, const DepfileConfig& depfileConfig) {
		smart_refctd_ptr<CHLSLCompiler> hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

		CHLSLCompiler::SPreprocessorOptions options = {};
		options.sourceIdentifier = sourceIdentifier;
		options.logger = m_logger.get();

		auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
		auto includeLoader = includeFinder->getDefaultFileSystemLoader();

		// because before real compilation we do preprocess the input it doesn't really matter we proxy include search direcotries further with dxcOptions since at the end all includes are resolved to single file
		for (const auto& it : m_include_search_paths)
			includeFinder->addSearchPath(it, includeLoader);

		options.includeFinder = includeFinder.get();
		options.depfile = depfileConfig.enabled;
		options.depfilePath = depfileConfig.path;

		const char* code_ptr = (const char*)shader->getContent()->getPointer();
		std::string_view code({ code_ptr, strlen(code_ptr)});

		return hlslcompiler->preprocessShader(std::string(code), shaderStage, options, nullptr);
	}

	core::smart_refctd_ptr<IShader> compile_shader(const IShader* shader, hlsl::ShaderStage shaderStage, std::string_view sourceIdentifier, const DepfileConfig& depfileConfig) {
		smart_refctd_ptr<CHLSLCompiler> hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

		CHLSLCompiler::SOptions options = {};
		options.stage = shaderStage;
		options.preprocessorOptions.sourceIdentifier = sourceIdentifier;
		options.preprocessorOptions.logger = m_logger.get();

		options.debugInfoFlags = core::bitflag<asset::IShaderCompiler::E_DEBUG_INFO_FLAGS>(asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_TOOL_BIT);
		options.dxcOptions = std::span<std::string>(m_arguments);

		auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
		auto includeLoader = includeFinder->getDefaultFileSystemLoader();

		// because before real compilation we do preprocess the input it doesn't really matter we proxy include search direcotries further with dxcOptions since at the end all includes are resolved to single file
		for(const auto& it : m_include_search_paths)
			includeFinder->addSearchPath(it, includeLoader);

		options.preprocessorOptions.includeFinder = includeFinder.get();
		options.preprocessorOptions.depfile = depfileConfig.enabled;
		options.preprocessorOptions.depfilePath = depfileConfig.path;

		return hlslcompiler->compileToSPIRV((const char*)shader->getContent()->getPointer(), options);
	}


	std::tuple<core::smart_refctd_ptr<const IShader>, hlsl::ShaderStage> open_shader_file(std::string filepath) {

		m_assetMgr = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));

		IAssetLoader::SAssetLoadParams lp = {};
		lp.logger = m_logger.get();
		lp.workingDirectory = localInputCWD;
		auto assetBundle = m_assetMgr->getAsset(filepath, lp);
		const auto assets = assetBundle.getContents();
		const auto* metadata = assetBundle.getMetadata();
		if (assets.empty()) {
			m_logger->log("Could not load shader %s", ILogger::ELL_ERROR, filepath);
			return {nullptr, hlsl::ShaderStage::ESS_UNKNOWN};
		}
		assert(assets.size() == 1);

		// could happen when the file is missing an extension and we can't deduce its a shader
		if (assetBundle.getAssetType() == IAsset::ET_BUFFER)
		{
			auto buf = IAsset::castDown<ICPUBuffer>(assets[0]);
			std::string source; source.resize(buf->getSize()+1);
			memcpy(source.data(),buf->getPointer(),buf->getSize());
			return { core::make_smart_refctd_ptr<IShader>(source.data(), IShader::E_CONTENT_TYPE::ECT_HLSL, std::move(filepath)), hlsl::ShaderStage::ESS_UNKNOWN};
		}
		else if (assetBundle.getAssetType() == IAsset::ET_SHADER)
		{
			const auto hlslMetadata = static_cast<const CHLSLMetadata*>(metadata);
			return { smart_refctd_ptr_static_cast<IShader>(assets[0]), hlslMetadata->shaderStages->front()};
		} 
		else 
		{
			m_logger->log("file '%s' is an asset that is neither a buffer or a shader.", ILogger::ELL_ERROR, filepath);
		}

		return {nullptr, hlsl::ShaderStage::ESS_UNKNOWN};
	}


	bool no_nbl_builtins{ false };
	smart_refctd_ptr<ISystem> m_system;
	smart_refctd_ptr<ILogger> m_logger;
	std::vector<std::string> m_arguments, m_include_search_paths;
	core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;


};

NBL_MAIN_FUNC(ShaderCompiler)

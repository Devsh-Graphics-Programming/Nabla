#include "nabla.h"
#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>

#include "nbl/asset/metadata/CHLSLMetadata.h"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

using namespace nbl;
using namespace nbl::system;
using namespace nbl::core;
using namespace nbl::asset;

class ShaderCompiler final : public system::IApplicationFramework
{
	using base_t = system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		const auto argc = argv.size();
		const bool insufficientArguments = argc < 2;

		if (not insufficientArguments)
		{
			// 1) NOTE: imo each example should be able to dump build info & have such mode, maybe it could go straight to IApplicationFramework main
			// 2) TODO: this whole "serialize" logic should go to the GitInfo struct and be static or something, it should be standardized

			if (argv[1] == "--dump-build-info")
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

				// TOOD: use argparse for it
				if (argc > 3 && argv[2] == "--file")
					oPath = argv[3];

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

				// in this mode terminate with 0 if all good
				exit(0);
			}
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

		m_logger = make_smart_refctd_ptr<CStdoutLogger>(core::bitflag(ILogger::ELL_DEBUG) | ILogger::ELL_INFO | ILogger::ELL_WARNING |	ILogger::ELL_PERFORMANCE | ILogger::ELL_ERROR);

		if (insufficientArguments) 
		{
			m_logger->log("Insufficient arguments.", ILogger::ELL_ERROR);
			return false;
		}

		m_arguments = std::vector<std::string>(argv.begin() + 1, argv.end()-1); // turn argv into vector for convenience

		std::string file_to_compile = argv.back();

		if (!m_system->exists(file_to_compile, IFileBase::ECF_READ)) {
			m_logger->log("Incorrect arguments. Expecting last argument to be filename of the shader intended to compile.", ILogger::ELL_ERROR);
			return false;
		}
		std::string output_filepath = "";

		auto builtin_flag_pos = std::find(m_arguments.begin(), m_arguments.end(), "-no-nbl-builtins");
		if (builtin_flag_pos != m_arguments.end()) {
			m_logger->log("Unmounting builtins.");
			m_system->unmountBuiltins();
			no_nbl_builtins = true;
			m_arguments.erase(builtin_flag_pos);
		}

		auto split = [&](const std::string& str, char delim) 
		{
			std::vector<std::string> strings;
			size_t start, end = 0;
		
			while ((start = str.find_first_not_of(delim, end)) != std::string::npos) 
			{
			    end = str.find(delim, start);
			    strings.push_back(str.substr(start, end - start));
			}
		
			return strings;
		};
		
		auto findOutputFlag = [&](const std::string_view& outputFlag)
		{
			return std::find_if(m_arguments.begin(), m_arguments.end(), [&](const std::string& argument) 
			{
				return argument.find(outputFlag.data()) != std::string::npos;
			});
		};
		
		auto output_flag_pos_fc = findOutputFlag("-Fc");
		auto output_flag_pos_fo = findOutputFlag("-Fo");
		if (output_flag_pos_fc != m_arguments.end() && output_flag_pos_fo != m_arguments.end()) {
			m_logger->log("Invalid arguments. Passed both -Fo and -Fc.", ILogger::ELL_ERROR);
			return false;
		}
		auto output_flag_pos = output_flag_pos_fc != m_arguments.end() ? output_flag_pos_fc : output_flag_pos_fo;
		if (output_flag_pos == m_arguments.end()) 
		{
			m_logger->log("Missing arguments. Expecting `-Fc {filename}` or `-Fo {filename}`.", ILogger::ELL_ERROR);
			return false;
		}
		else
		{
			// we need to assume -Fc may be passed with output file name quoted together with "", so we split it (DXC does it)
			const auto& outputFlag = *output_flag_pos;
			auto outputFlagVector = split(outputFlag, ' ');
		
			if(outputFlag == "-Fc" || outputFlag == "-Fo")
			{
			    if (output_flag_pos + 1 != m_arguments.end()) 
			    {
					output_filepath = *(output_flag_pos + 1);
			    }
			    else 
			    {
					m_logger->log("Incorrect arguments. Expecting filename after %s.", ILogger::ELL_ERROR, outputFlag);
					return false;
			    }
			}
			else
			{
			    output_filepath = outputFlagVector[1];
			}
			m_arguments.erase(output_flag_pos, output_flag_pos+2);

			if (output_filepath.empty())
			{
				m_logger->log("Invalid output file path!" + output_filepath, ILogger::ELL_ERROR);
				return false;
			}
		
			m_logger->log("Compiled shader code will be saved to " + output_filepath, ILogger::ELL_INFO);
		}

#ifndef NBL_EMBED_BUILTIN_RESOURCES
		if (!no_nbl_builtins) {
			m_system->unmountBuiltins();
			no_nbl_builtins = true;
			m_logger->log("nsc.exe was compiled with builtin resources disabled. Force enabling -no-nbl-builtins.", ILogger::ELL_WARNING);
		}
#endif
		if (std::find(m_arguments.begin(), m_arguments.end(), "-E") == m_arguments.end())
		{
			//Insert '-E main' into arguments if no entry point is specified
			m_arguments.push_back("-E");
			m_arguments.push_back("main");
		}

		for (size_t i = 0; i < m_arguments.size() - 1; ++i) // -I must be given with second arg, no need to include iteration over last one
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
		auto compilation_result = compile_shader(shader.get(), shaderStage, file_to_compile);

		// writie compiled shader to file as bytes
		if (compilation_result) 
		{
			m_logger->log("Shader compilation successful.", ILogger::ELL_INFO);
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

			output_file.write((const char*)compilation_result->getContent()->getPointer(), compilation_result->getContent()->getSize());

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

			return true;
		}
		else 
		{
			m_logger->log("Shader compilation failed.", ILogger::ELL_ERROR);
			return false;
		}
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }


private:

	core::smart_refctd_ptr<IShader> compile_shader(const IShader* shader, hlsl::ShaderStage shaderStage, std::string_view sourceIdentifier) {
		smart_refctd_ptr<CHLSLCompiler> hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

		CHLSLCompiler::SOptions options = {};
		options.stage = shaderStage;
		options.preprocessorOptions.sourceIdentifier = sourceIdentifier;
		options.preprocessorOptions.logger = m_logger.get();

		options.dxcOptions = std::span<std::string>(m_arguments);

		auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
		auto includeLoader = includeFinder->getDefaultFileSystemLoader();

		// because before real compilation we do preprocess the input it doesn't really matter we proxy include search direcotries further with dxcOptions since at the end all includes are resolved to single file
		for(const auto& it : m_include_search_paths)
			includeFinder->addSearchPath(it, includeLoader);

		options.preprocessorOptions.includeFinder = includeFinder.get();

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
	smart_refctd_ptr<CStdoutLogger> m_logger;
	std::vector<std::string> m_arguments, m_include_search_paths;
	core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;


};

NBL_MAIN_FUNC(ShaderCompiler)

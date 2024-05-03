#include "nabla.h"
#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdlib>
#include <string>

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
		if (system)
			m_system = std::move(system);
		else
			m_system = system::IApplicationFramework::createSystem();

		if (!m_system)
			return false;

		m_logger = make_smart_refctd_ptr<CStdoutLogger>();

		auto argc = argv.size();

//#ifndef NBL_DEBUG
//		std::string str = argv[0];
//		for (auto i=1; i<argc; i++)
//			str += "\n"+argv[i];
//		m_logger->log("Arguments Receive: %s", ILogger::ELL_DEBUG, str.c_str());
//#endif

		// expect the first argument to be nsc.exe
		// second argument should be input: filename of a shader to compile
		if (argc < 2) {
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
		
			m_logger->log("Compiled shader code will be saved to " + output_filepath);
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

		auto shader = open_shader_file(file_to_compile);
		if (shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_HLSL)
		{
			m_logger->log("Error. Loaded shader file content is not HLSL.", ILogger::ELL_ERROR);
			return false;
		}
		auto compilation_result = compile_shader(shader.get(), file_to_compile);

		// writie compiled shader to file as bytes
		if (compilation_result && !output_filepath.empty()) {
			std::fstream output_file(output_filepath, std::ios::out | std::ios::binary);
			output_file.write((const char*)compilation_result->getContent()->getPointer(), compilation_result->getContent()->getSize());
			output_file.close();
			m_logger->log("Shader compilation successful.");
			return true;
		}
		else {
			m_logger->log("Shader compilation failed.", ILogger::ELL_ERROR);
			return false;
		}
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }


private:

	core::smart_refctd_ptr<ICPUShader> compile_shader(const ICPUShader* shader, std::string_view sourceIdentifier) {
		smart_refctd_ptr<CHLSLCompiler> hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

		CHLSLCompiler::SOptions options = {};
		options.stage = shader->getStage();
		options.preprocessorOptions.sourceIdentifier = sourceIdentifier;
		options.preprocessorOptions.logger = m_logger.get();
		options.dxcOptions = std::span<std::string>(m_arguments);
		auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
		options.preprocessorOptions.includeFinder = includeFinder.get();

		return hlslcompiler->compileToSPIRV((const char*)shader->getContent()->getPointer(), options);
	}


	core::smart_refctd_ptr<const ICPUShader> open_shader_file(std::string filepath) {

		m_assetMgr = make_smart_refctd_ptr<asset::IAssetManager>(smart_refctd_ptr(m_system));

		IAssetLoader::SAssetLoadParams lp = {};
		lp.logger = m_logger.get();
		lp.workingDirectory = localInputCWD;
		auto assetBundle = m_assetMgr->getAsset(filepath, lp);
		const auto assets = assetBundle.getContents();
		if (assets.empty()) {
			m_logger->log("Could not load shader %s", ILogger::ELL_ERROR, filepath);
			return nullptr;
		}
		assert(assets.size() == 1);

		// could happen when the file is missing an extension and we can't deduce its a shader
		if (assetBundle.getAssetType() == IAsset::ET_BUFFER)
		{
			auto buf = IAsset::castDown<ICPUBuffer>(assets[0]);
			std::string source; source.resize(buf->getSize()+1);
			memcpy(source.data(),buf->getPointer(),buf->getSize());
			return core::make_smart_refctd_ptr<ICPUShader>(source.data(), IShader::ESS_UNKNOWN, IShader::E_CONTENT_TYPE::ECT_HLSL, std::move(filepath));
		}
		else if (assetBundle.getAssetType() == IAsset::ET_SHADER)
		{
			return smart_refctd_ptr_static_cast<ICPUShader>(assets[0]);
		} 
		else 
		{
			m_logger->log("file '%s' is an asset that is neither a buffer or a shader.", ILogger::ELL_ERROR, filepath);
		}

		return nullptr;
	}


	bool no_nbl_builtins{ false };
	smart_refctd_ptr<ISystem> m_system;
	smart_refctd_ptr<CStdoutLogger> m_logger;
	std::vector<std::string> m_arguments;
	core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;


};

NBL_MAIN_FUNC(ShaderCompiler)

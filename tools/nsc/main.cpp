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

		m_logger = make_smart_refctd_ptr<CStdoutLogger>();

		auto argc = argv.size();

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

		auto output_flag_pos = std::find(m_arguments.begin(), m_arguments.end(), "-Fo");
		if (output_flag_pos == m_arguments.end())
			output_flag_pos = std::find(m_arguments.begin(), m_arguments.end(), "-Fc");
			
		if (output_flag_pos == m_arguments.end()) {
			m_logger->log("Missing arguments. Expecting `-Fo {filename}` or `-Fc {filename}`.", ILogger::ELL_ERROR);
			return false;
		}

		if (output_flag_pos + 1 != m_arguments.end()) {
			output_filepath = *(output_flag_pos + 1);
			m_logger->log("Compiled shader code will be saved to " + output_filepath);
			m_arguments.erase(output_flag_pos, output_flag_pos+1);
		}
		else {
			m_logger->log("Incorrect arguments. Expecting filename after -Fo or -Fc.", ILogger::ELL_ERROR);
			return false;
		}

#ifndef NBL_EMBED_BUILTIN_RESOURCES
		if (!no_nbl_builtins) {
			m_system->unmountBuiltins();
			no_nbl_builtins = true;
			m_logger->log("nsc.exe was compiled with builtin resources disabled. Force enabling -no-nbl-builtins.", ILogger::ELL_WARNING);
		}
#endif

		const ICPUShader* shader = open_shader_file(file_to_compile);
		if (shader->getContentType() != IShader::E_CONTENT_TYPE::ECT_HLSL)
		{
			m_logger->log("Error. Loaded shader file content is not HLSL.", ILogger::ELL_ERROR);
			return false;
		}
		auto compilation_result = compile_shader(shader, file_to_compile);

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


	const ICPUShader* open_shader_file(std::string& filepath) {

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
		smart_refctd_ptr<ICPUSpecializedShader> source = IAsset::castDown<ICPUSpecializedShader>(assets[0]);

		return source->getUnspecialized();
	}


	bool no_nbl_builtins{ false };
	smart_refctd_ptr<ISystem> m_system;
	smart_refctd_ptr<CStdoutLogger> m_logger;
	std::vector<std::string> m_arguments;
	core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;


};

NBL_MAIN_FUNC(ShaderCompiler)

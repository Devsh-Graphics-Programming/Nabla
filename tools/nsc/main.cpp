// TODO: Cypi


// nsc input/simple_shader.hlsl -T ps_6_0 -E Main -Fo output/shader.ps

#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <streambuf>

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

		// expect the first argument to be
		// .exe
		// second the filename of a shader to compile
		if (argc < 2) {
			m_logger->log("Insufficient arguments.", ILogger::ELL_ERROR);
			return false;
		}

		m_arguments = std::vector<std::string>(argv.begin() + 2, argv.end()); // turn argv into vector for convenience
		std::string file_to_compile = argv[1];

		if (!m_system->exists(file_to_compile, IFileBase::ECF_READ)) {
			m_logger->log("Incorrect arguments. Expecting second argument to be filename of the shader intended to compile.", ILogger::ELL_ERROR);
			return false;
		}
		std::string output_filepath = "";
		for (auto i = 1; i < m_arguments.size(); i++)
		{
			if (m_arguments[i] == "-no-nbl-builtins")
			{
				// unmount builtins
				m_logger->log("Unmounting builtins.");
				m_system->unmountBuiltins();
				no_nbl_builtins = true;
				m_arguments.erase(m_arguments.begin() + i - 1);
				i--;
			}
			else if (m_arguments[i] == "-Fo")
			{
				// get filepath to save output to
				if (i + 1 < argc)
				{
					output_filepath = argv[i + 1];
					m_logger->log("Compiled shader code will be saved to " + output_filepath);
					m_arguments.erase(m_arguments.begin() + i, m_arguments.begin() + i + 1);
					i--;
				}
				else
				{
					m_logger->log("Incorrect arguments. Expecting filename after -Fo.", ILogger::ELL_ERROR);
				}
			}

		}

#ifndef NBL_EMBED_BUILTIN_RESOURCES
		if (!no_nbl_builtins) {
			m_system->unmountBuiltins();
			no_nbl_builtins = true;
			m_logger->log("nsc.exe was compiled with builtin resources disabled. Force enabling -no-nbl-builtins.", ILogger::ELL_WARNING);
		}
#endif
		string shader_code = open_shader_file(file_to_compile);
		auto compilation_result = compile_shader(shader_code, file_to_compile);

		// writie compiled shader to file as bytes
		if (compilation_result && !output_filepath.empty()) {
			std::fstream output_file(output_filepath, std::ios::out | std::ios::binary);
			output_file.write((const char*)compilation_result->getContent()->getPointer(), compilation_result->getContent()->getSize());
			output_file.close();
			m_logger->log("Shader compilation successful.");
		}
		else {
			m_logger->log("Shader compilation failed.", ILogger::ELL_ERROR);
		}
		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }


private:

	core::smart_refctd_ptr<ICPUShader> compile_shader(std::string& shader_code, std::string_view sourceIdentifier) {
		constexpr uint32_t WorkgroupSize = 256;
		constexpr uint32_t WorkgroupCount = 2048;
		const string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);

		smart_refctd_ptr<CHLSLCompiler> hlslcompiler = make_smart_refctd_ptr<CHLSLCompiler>(smart_refctd_ptr(m_system));

		CHLSLCompiler::SOptions options = {};
		options.stage = asset::IShader::E_SHADER_STAGE::ESS_UNKNOWN;
		//options.debugInfoFlags = IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
		options.preprocessorOptions.sourceIdentifier = sourceIdentifier;
		options.preprocessorOptions.logger = m_logger.get();
		options.dxcOptions = std::span<std::string>(m_arguments);
		auto includeFinder = make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(smart_refctd_ptr(m_system));
		options.preprocessorOptions.includeFinder = includeFinder.get();

		return hlslcompiler->compileToSPIRV(shader_code.c_str(), options);
	}


	std::string open_shader_file(std::string& filepath) {
		std::ifstream stream(filepath);
		std::string str((std::istreambuf_iterator<char>(stream)),
			std::istreambuf_iterator<char>());
		return str;
	}


	bool no_nbl_builtins{ false };
	smart_refctd_ptr<ISystem> m_system;
	smart_refctd_ptr<CStdoutLogger> m_logger;
	std::vector<std::string> m_arguments;

};

NBL_MAIN_FUNC(ShaderCompiler)
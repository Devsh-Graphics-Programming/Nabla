// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/utils/IGLSLCompiler.h"
#include "nbl/asset/utils/shadercUtils.h"
#include "nbl/asset/utils/CIncludeHandler.h"

#include "nbl/asset/utils/CGLSLVirtualTexturingBuiltinIncludeLoader.h"

#include <sstream>
#include <regex>
#include <iterator>


using namespace nbl;
using namespace nbl::asset;


IGLSLCompiler::IGLSLCompiler(system::ISystem* _s)
    : m_inclHandler(core::make_smart_refctd_ptr<CIncludeHandler>(_s)), IShaderCompiler(_s)
{
    m_inclHandler->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<asset::CGLSLVirtualTexturingBuiltinIncludeLoader>(_s));
}

core::smart_refctd_ptr<ICPUBuffer> IGLSLCompiler::compileSPIRVFromGLSL(
    const char* _glslCode,
    IShader::E_SHADER_STAGE _stage,
    const char* _entryPoint,
    const char* _compilationId,
    bool _genDebugInfo,
    std::string* _outAssembly,
    system::logger_opt_ptr logger,
    const E_SPIRV_VERSION targetSpirvVersion) const
{
    //shaderc requires entry point to be "main" in GLSL
    if (strcmp(_entryPoint, "main") != 0)
        return nullptr;

    shaderc::Compiler comp;
    shaderc::CompileOptions options;//default options
    assert(targetSpirvVersion < ESV_COUNT);
    options.SetTargetSpirv(static_cast<shaderc_spirv_version>(targetSpirvVersion));
    const shaderc_shader_kind stage = _stage==IShader::ESS_UNKNOWN ? shaderc_glsl_infer_from_source : ESStoShadercEnum(_stage);
    const size_t glsl_len = strlen(_glslCode);
    if (_genDebugInfo)
        options.SetGenerateDebugInfo();

    shaderc::AssemblyCompilationResult asm_res;
    shaderc::SpvCompilationResult bin_res;
    if (_outAssembly) {
        asm_res = comp.CompileGlslToSpvAssembly(_glslCode, glsl_len, stage, _compilationId ? _compilationId : "", options);
        _outAssembly->resize(std::distance(asm_res.cbegin(), asm_res.cend()));
        memcpy(_outAssembly->data(), asm_res.cbegin(), _outAssembly->size());
        bin_res = comp.AssembleToSpv(_outAssembly->data(), _outAssembly->size(), options);
    }
    else {
        bin_res = comp.CompileGlslToSpv(_glslCode, glsl_len, stage, _compilationId ? _compilationId : "", _entryPoint, options);
    }

    if (bin_res.GetCompilationStatus() != shaderc_compilation_status_success) {
        logger.log(bin_res.GetErrorMessage(), system::ILogger::ELL_ERROR);
        return nullptr;
    }

    auto spirv = core::make_smart_refctd_ptr<ICPUBuffer>(std::distance(bin_res.cbegin(), bin_res.cend())*sizeof(uint32_t));
    memcpy(spirv->getPointer(), bin_res.cbegin(), spirv->getSize());
	return spirv;
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::createSPIRVFromGLSL(
    const char* _glslCode,
    IShader::E_SHADER_STAGE _stage,
    const char* _entryPoint,
    const char* _compilationId,
    const ISPIRVOptimizer* _opt,
    bool _genDebugInfo,
    std::string* _outAssembly,
    system::logger_opt_ptr logger,
    const E_SPIRV_VERSION targetSpirvVersion) const
{
    auto spirvBuffer = compileSPIRVFromGLSL(_glslCode,_stage,_entryPoint,_compilationId,_genDebugInfo,_outAssembly,logger,targetSpirvVersion);
	if (!spirvBuffer)
		return nullptr;
    if (_opt)
        spirvBuffer = _opt->optimize(spirvBuffer.get(),logger) ;

    return core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirvBuffer), _stage, IShader::ECT_GLSL, _compilationId);
}

core::smart_refctd_ptr<ICPUShader> IGLSLCompiler::createSPIRVFromGLSL(
    system::IFile* _sourcefile,
    IShader::E_SHADER_STAGE _stage,
    const char* _entryPoint,
    const char* _compilationId,
    const ISPIRVOptimizer* _opt,
    bool _genDebugInfo,
    std::string* _outAssembly,
    system::logger_opt_ptr logger,
    const E_SPIRV_VERSION targetSpirvVersion) const
{
    size_t fileSize = _sourcefile->getSize();
    std::string glsl(fileSize, '\0');

    system::IFile::success_t success;
    _sourcefile->read(success, glsl.data(), 0, fileSize);
    if (!success)
        return nullptr;

    return createSPIRVFromGLSL(glsl.c_str(), _stage, _entryPoint, _compilationId, _opt, _genDebugInfo, _outAssembly, logger, targetSpirvVersion);
}
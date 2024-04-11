#ifndef  _NBL_ASSET_SHADER_COMPILER_SERIALIZATION_H_INCLUDED_
#define _NBL_ASSET_SHADER_COMPILER_SERIALIZATION_H_INCLUDED_

#include "nbl/asset/utils/IShaderCompiler.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using SEntry = nbl::asset::IShaderCompiler::CCache::SEntry;


namespace nbl::asset {

    // SMacroData, simple container used in SPreprocessorData

    void to_json(json& j, const SEntry::SMacroData& macroData)
    {
        j = json{
            { "identifier", macroData.identifier },
            { "definition", macroData.definition },
        };
    }

    void from_json(const json& j, SEntry::SMacroData& macroData)
    {
        j.at("identifier").get_to(macroData.identifier);
        j.at("definition").get_to(macroData.definition);
    }

    // SPreprocessorData, holds serialized info for Preprocessor options used during compilation

    void to_json(json& j, const SEntry::SPreprocessorData& preprocData)
    {
        j = json{
            { "sourceIdentifier", preprocData.sourceIdentifier },
            { "extraDefines", preprocData.extraDefines},
        };
    }

    void from_json(const json& j, SEntry::SPreprocessorData& preprocData)
    {
        j.at("sourceIdentifier").get_to(preprocData.sourceIdentifier);
        j.at("extraDefines").get_to(preprocData.extraDefines);
    }

    // Optimizer pass has its own method for easier vector serialization

    void to_json(json& j, const ISPIRVOptimizer::E_OPTIMIZER_PASS& optPass)
    {
        uint32_t value = static_cast<uint32_t>(optPass);
        j = json{
            { "optPass", optPass },
        };
    }

    void from_json(const json& j, ISPIRVOptimizer::E_OPTIMIZER_PASS& optPass)
    {
        uint32_t aux;
        j.at("optPass").get_to(aux);
        optPass = static_cast<ISPIRVOptimizer::E_OPTIMIZER_PASS>(aux);
    }

    // SCompilerData, holds serialized info for all Compilation options

    void to_json(json& j, const SEntry::SCompilerData& compilerData)
    {
        uint32_t shaderStage = static_cast<uint32_t>(compilerData.stage);
        uint32_t spirvVersion = static_cast<uint32_t>(compilerData.targetSpirvVersion);
        uint32_t debugFlags = static_cast<uint32_t>(compilerData.debugInfoFlags.value);

        j = json {
            { "shaderStage", shaderStage },
            { "spirvVersion", spirvVersion },
            { "optimizerPasses", compilerData.optimizerPasses },
            { "debugFlags", debugFlags },
            { "preprocessorData", compilerData.preprocessorData },
        };
    }

    void from_json(const json& j, SEntry::SCompilerData& compilerData)
    {
        uint32_t shaderStage, spirvVersion, debugFlags;
        j.at("shaderStage").get_to(shaderStage);
        j.at("spirvVersion").get_to(spirvVersion);
        j.at("optimizerPasses").get_to(compilerData.optimizerPasses);
        j.at("debugFlags").get_to(debugFlags);
        j.at("preprocessorData").get_to(compilerData.preprocessorData);
        compilerData.stage = static_cast<IShader::E_SHADER_STAGE>(shaderStage);
        compilerData.targetSpirvVersion = static_cast<IShaderCompiler::E_SPIRV_VERSION>(spirvVersion);
        compilerData.debugInfoFlags = core::bitflag<IShaderCompiler::E_DEBUG_INFO_FLAGS>(debugFlags);
    }

    // Serialize clock's time point
    using time_point_t = nbl::system::IFileBase::time_point_t;

    void to_json(json& j, const time_point_t& timePoint)
    {
        auto ticks = timePoint.time_since_epoch().count();
        j = json{
            { "ticks", ticks },
        };
    }

    void from_json(const json& j, time_point_t& timePoint)
    {
        uint64_t ticks;
        j.at("ticks").get_to(ticks);
        timePoint = time_point_t(time_point_t::clock::duration(ticks));
    }

    // Serialize (most) of SEntry, keeping some fields as extra serialization to keep them separate on disk

    void to_json(json& j, const IShaderCompiler::CCache::SEntry& entry)
    {
        // Serializing the write time by hand because compiler wasn't having it otherwise
        auto ticks = entry.lastWriteTime.time_since_epoch().count();
        j = json{
            { "mainFilePath", entry.mainFilePath },
            { "mainFileHash", entry.mainFileHash },
            { "lastWriteTimeTicks", ticks },
            { "compilerData", entry.compilerData },
            { "compilerDataHash", entry.compilerDataHash },
            { "storagePath", entry.storagePath },
        };
    }

    void from_json(const json& j, IShaderCompiler::CCache::SEntry& entry)
    {
        uint64_t ticks;
        j.at("lastWriteTimeTicks").get_to(ticks);
        entry.lastWriteTime = time_point_t(time_point_t::clock::duration(ticks));
        j.at("mainFilePath").get_to(entry.mainFilePath);
        j.at("mainFileHash").get_to(entry.mainFileHash);
        j.at("compilerData").get_to(entry.compilerData);
        j.at("compilerDataHash").get_to(entry.compilerDataHash);
        j.at("storagePath").get_to(entry.storagePath);
    }

    // Sdependency serialization. Dependencies will be saved in a vector for easier vectorization

    void to_json(json& j, const SPreprocessingDependency& dependency)
    {
        // Serializing the write time by hand because compiler wasn't having it otherwise
        auto ticks = dependency.lastWriteTime.time_since_epoch().count();
        j = json{
            { "requestingSourceDir", dependency.requestingSourceDir },
            { "identifier", dependency.identifier },
            { "contents", dependency.contents },
            { "hash", dependency.hash },
            { "standardInclude", dependency.standardInclude },
            { "lastWriteTimeTicks", ticks },
        };
    }

    void from_json(const json& j, SPreprocessingDependency& dependency)
    {
        uint64_t ticks;
        j.at("lastWriteTimeTicks").get_to(ticks);
        dependency.lastWriteTime = std::chrono::utc_clock::time_point(std::chrono::utc_clock::duration(ticks));
        j.at("requestingSourceDir").get_to(dependency.requestingSourceDir);
        j.at("identifier").get_to(dependency.identifier);
        j.at("contents").get_to(dependency.contents);
        j.at("hash").get_to(dependency.hash);
        j.at("standardInclude").get_to(dependency.standardInclude);
    }

    // We do a bit of a Frankenstein for CPU Shader serialization. We serialize creation parameters into a json, but binary data into a .bin file so it takes up less space

    void to_json(json& j, const IShaderCompiler::CCache::SEntry::CPUShaderCreationParams& creationParams)
    {
        uint32_t stage = static_cast<uint32_t>(creationParams.stage);
        uint32_t contentType = static_cast<uint32_t>(creationParams.contentType);
        j = json{
            { "stage", stage },
            { "contentType", contentType },
            { "filepathHint", creationParams.filepathHint },
            { "codeByteSize", creationParams.codeByteSize },
        };
    }

    void from_json(const json& j, IShaderCompiler::CCache::SEntry::CPUShaderCreationParams& creationParams)
    {
        uint32_t stage, contentType;
        j.at("stage").get_to(stage);
        j.at("contentType").get_to(contentType);
        j.at("filepathHint").get_to(creationParams.filepathHint);
        j.at("codeByteSize").get_to(creationParams.codeByteSize);
        creationParams.stage = static_cast<IShader::E_SHADER_STAGE>(stage);
        creationParams.contentType = static_cast<IShader::E_CONTENT_TYPE>(stage);
    }
}

#endif
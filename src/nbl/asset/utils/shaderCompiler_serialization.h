#ifndef _NBL_ASSET_SHADER_COMPILER_SERIALIZATION_H_INCLUDED_
#define _NBL_ASSET_SHADER_COMPILER_SERIALIZATION_H_INCLUDED_

#include "nbl/asset/utils/IShaderCompiler.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;
using SEntry = nbl::asset::IShaderCompiler::CCache::SEntry;


namespace nbl::asset
{

// TODO: use NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE whenever possible

// SMacroData, simple container used in SPreprocessorArgs

inline void to_json(json& j, const IShaderCompiler::SMacroDefinition& macroData)
{
    j = json{
        { "identifier", macroData.identifier },
        { "definition", macroData.definition },
    };
}

inline void from_json(const json& j, IShaderCompiler::SMacroDefinition& macroData)
{
    j.at("identifier").get_to(macroData.identifier);
    j.at("definition").get_to(macroData.definition);
}

// SPreprocessorData, holds serialized info for Preprocessor options used during compilation
inline void to_json(json& j, const SEntry::SPreprocessorArgs& preprocArgs)
{
    j = json{
        { "sourceIdentifier", preprocArgs.sourceIdentifier },
        { "extraDefines", preprocArgs.extraDefines},
    };
}

inline void from_json(const json& j, SEntry::SPreprocessorArgs& preprocArgs)
{
    j.at("sourceIdentifier").get_to(preprocArgs.sourceIdentifier);
    j.at("extraDefines").get_to(preprocArgs.extraDefines);
}

// Optimizer pass has its own method for easier vector serialization

inline void to_json(json& j, const ISPIRVOptimizer::E_OPTIMIZER_PASS& optPass)
{
    uint32_t value = static_cast<uint32_t>(optPass);
    j = json{
        { "optPass", optPass },
    };
}

inline void from_json(const json& j, ISPIRVOptimizer::E_OPTIMIZER_PASS& optPass)
{
    uint32_t aux;
    j.at("optPass").get_to(aux);
    optPass = static_cast<ISPIRVOptimizer::E_OPTIMIZER_PASS>(aux);
}

// SCompilerArgs, holds serialized info for all Compilation options

inline void to_json(json& j, const SEntry::SCompilerArgs& compilerData)
{
    uint32_t shaderStage = static_cast<uint32_t>(compilerData.stage);
    uint32_t spirvVersion = static_cast<uint32_t>(compilerData.targetSpirvVersion);
    uint32_t debugFlags = static_cast<uint32_t>(compilerData.debugInfoFlags.value);

    j = json {
        { "shaderStage", shaderStage },
        { "spirvVersion", spirvVersion },
        { "optimizerPasses", compilerData.optimizerPasses },
        { "debugFlags", debugFlags },
        { "preprocessorArgs", compilerData.preprocessorArgs },
    };
}

inline void from_json(const json& j, SEntry::SCompilerArgs& compilerData)
{
    uint32_t shaderStage, spirvVersion, debugFlags;
    j.at("shaderStage").get_to(shaderStage);
    j.at("spirvVersion").get_to(spirvVersion);
    j.at("optimizerPasses").get_to(compilerData.optimizerPasses);
    j.at("debugFlags").get_to(debugFlags);
    j.at("preprocessorArgs").get_to(compilerData.preprocessorArgs);
    compilerData.stage = static_cast<IShader::E_SHADER_STAGE>(shaderStage);
    compilerData.targetSpirvVersion = static_cast<IShaderCompiler::E_SPIRV_VERSION>(spirvVersion);
    compilerData.debugInfoFlags = core::bitflag<IShaderCompiler::E_DEBUG_INFO_FLAGS>(debugFlags);
}

// Serialize clock's time point
using time_point_t = nbl::system::IFileBase::time_point_t;

inline void to_json(json& j, const time_point_t& timePoint)
{
    auto ticks = timePoint.time_since_epoch().count();
    j = json{
        { "ticks", ticks },
    };
}

inline void from_json(const json& j, time_point_t& timePoint)
{
    uint64_t ticks;
    j.at("ticks").get_to(ticks);
    timePoint = time_point_t(time_point_t::clock::duration(ticks));
}

// SDependency serialization. Dependencies will be saved in a vector for easier vectorization

inline void to_json(json& j, const SEntry::SPreprocessingDependency& dependency)
{
    j = json{
        { "requestingSourceDir", dependency.requestingSourceDir },
        { "identifier", dependency.identifier },
        { "contents", dependency.contents },
        { "hash", dependency.hash },
        { "standardInclude", dependency.standardInclude },
    };
}

inline void from_json(const json& j, SEntry::SPreprocessingDependency& dependency)
{
    j.at("requestingSourceDir").get_to(dependency.requestingSourceDir);
    j.at("identifier").get_to(dependency.identifier);
    j.at("contents").get_to(dependency.contents);
    j.at("hash").get_to(dependency.hash);
    j.at("standardInclude").get_to(dependency.standardInclude);
}

// We serialize shader creation parameters into a json, along with indexing info into the .bin buffer where the cache is serialized

struct CPUShaderCreationParams {
    IShader::E_SHADER_STAGE stage;
    IShader::E_CONTENT_TYPE contentType; //I think this one could be skipped since it's always going to be SPIR-V
    std::string filepathHint;
    uint64_t codeByteSize = 0;
    uint64_t offset = 0; // Offset into the serialized .bin for the Cache where code starts

    CPUShaderCreationParams(IShader::E_SHADER_STAGE _stage, IShader::E_CONTENT_TYPE _contentType, std::string_view _filepathHint, uint64_t _codeByteSize, uint64_t _offset)
        : stage(_stage), contentType(_contentType), filepathHint(_filepathHint), codeByteSize(_codeByteSize), offset(_offset)
    {}

    CPUShaderCreationParams() {};
};

inline void to_json(json& j, const CPUShaderCreationParams& creationParams)
{
    uint32_t stage = static_cast<uint32_t>(creationParams.stage);
    uint32_t contentType = static_cast<uint32_t>(creationParams.contentType);
    j = json{
        { "stage", stage },
        { "contentType", contentType },
        { "filepathHint", creationParams.filepathHint },
        { "codeByteSize", creationParams.codeByteSize },
        { "offset", creationParams.offset },
    };
}

inline void from_json(const json& j, CPUShaderCreationParams& creationParams)
{
    uint32_t stage, contentType;
    j.at("stage").get_to(stage);
    j.at("contentType").get_to(contentType);
    j.at("filepathHint").get_to(creationParams.filepathHint);
    j.at("codeByteSize").get_to(creationParams.codeByteSize);
    j.at("offset").get_to(creationParams.offset);
    creationParams.stage = static_cast<IShader::E_SHADER_STAGE>(stage);
    creationParams.contentType = static_cast<IShader::E_CONTENT_TYPE>(stage);
}

// Serialize SEntry, keeping some fields as extra serialization to keep them separate on disk

inline void to_json(json& j, const SEntry& entry)
{
    j = json{
        { "mainFileContents", entry.mainFileContents },
        { "compilerArgs", entry.compilerArgs },
        { "hash", entry.hash },
        { "lookupHash", entry.lookupHash },
        { "dependencies", entry.dependencies },
    };
}

inline void from_json(const json& j, SEntry& entry)
{
    j.at("mainFileContents").get_to(entry.mainFileContents);
    j.at("compilerArgs").get_to(entry.compilerArgs);
    j.at("hash").get_to(entry.hash);
    j.at("lookupHash").get_to(entry.lookupHash);
    j.at("dependencies").get_to(entry.dependencies);
    entry.value = nullptr;
}

}
#endif
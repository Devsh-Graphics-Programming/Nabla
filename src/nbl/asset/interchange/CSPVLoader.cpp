// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUShader.h"

#include "CSPVLoader.h"

using namespace nbl;
using namespace nbl::asset;

inline IShader::E_SHADER_STAGE getShaderStageFromSPIRVCrossExecutionModel(spv::ExecutionModel model)
{
    IShader::E_SHADER_STAGE shaderStage;
    switch (model)
    {
    case spv::ExecutionModelVertex:
        shaderStage = IShader::ESS_VERTEX; break;
    case spv::ExecutionModelTessellationControl:
        shaderStage = IShader::ESS_TESSELATION_CONTROL; break;
    case spv::ExecutionModelTessellationEvaluation:
        shaderStage = IShader::ESS_TESSELATION_EVALUATION; break;
    case spv::ExecutionModelGeometry:
        shaderStage = IShader::ESS_GEOMETRY; break;
    case spv::ExecutionModelFragment:
        shaderStage = IShader::ESS_FRAGMENT; break;
    case spv::ExecutionModelGLCompute:
        shaderStage = IShader::ESS_COMPUTE; break;
    case spv::ExecutionModelTaskNV:
        shaderStage = IShader::ESS_TASK; break;
    case spv::ExecutionModelMeshNV:
        shaderStage = IShader::ESS_MESH; break;
    case spv::ExecutionModelRayGenerationKHR:
        shaderStage = IShader::ESS_RAYGEN; break;
    case spv::ExecutionModelIntersectionKHR:
        shaderStage = IShader::ESS_INTERSECTION; break;
    case spv::ExecutionModelAnyHitKHR:
        shaderStage = IShader::ESS_ANY_HIT; break;
    case spv::ExecutionModelClosestHitKHR:
        shaderStage = IShader::ESS_MISS; break;
    case spv::ExecutionModelMissKHR:
        shaderStage = IShader::ESS_MISS; break;
    case spv::ExecutionModelCallableKHR:
        shaderStage = IShader::ESS_CALLABLE; break;
    case spv::ExecutionModelKernel:
    case spv::ExecutionModelMax:
    default:
        assert(!"Shader stage not supported!");
        shaderStage = IShader::ESS_UNKNOWN;
        break;
    }
    return shaderStage;
}

// load in the image data
SAssetBundle CSPVLoader::loadAsset(system::IFile* _file, const IAssetLoader::SAssetLoadParams& _params, IAssetLoader::IAssetLoaderOverride* _override, uint32_t _hierarchyLevel)
{
	if (!_file)
        return {};
	
	auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(_file->getSize());
	
	system::future<size_t> future;
	_file->read(future, buffer->getPointer(), 0, _file->getSize());
	future.get();

	if (reinterpret_cast<uint32_t*>(buffer->getPointer())[0]!=SPV_MAGIC_NUMBER)
		return {};

	SPIRV_CROSS_NAMESPACE::Parser parser(reinterpret_cast<uint32_t*>(buffer->getPointer()), buffer->getSize() / 4ull);
	parser.parse();
	const SPIRV_CROSS_NAMESPACE::ParsedIR& parsedIR = parser.get_parsed_ir();
	SPIRV_CROSS_NAMESPACE::SPIREntryPoint defaultEntryPoint = parsedIR.entry_points.at(parsedIR.default_entry_point);

    return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<ICPUShader>(std::move(buffer), getShaderStageFromSPIRVCrossExecutionModel(defaultEntryPoint.model), _file->getFileName().string())});
}

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUShader.h"
#include "nbl_spirv_cross/spirv.hpp"
#include "nbl_spirv_cross/spirv_parser.hpp"

#include "CSPVLoader.h"
#include "nbl_spirv_cross/spirv.hpp"
#include "nbl_spirv_cross/spirv_cfg.hpp"
#include "nbl_spirv_cross/spirv_parser.hpp"

using namespace nbl;
using namespace nbl::asset;

inline IShader::E_SHADER_STAGE getShaderStageFromSPIRVCrossExecutionModel(spv::ExecutionModel model)
{
    IShader::E_SHADER_STAGE shaderStage;
    switch (model)
    {
    case spv::ExecutionModelVertex:
        shaderStage = IShader::E_SHADER_STAGE::ESS_VERTEX; break;
    case spv::ExecutionModelTessellationControl:
        shaderStage = IShader::E_SHADER_STAGE::ESS_TESSELLATION_CONTROL; break;
    case spv::ExecutionModelTessellationEvaluation:
        shaderStage = IShader::E_SHADER_STAGE::ESS_TESSELLATION_EVALUATION; break;
    case spv::ExecutionModelGeometry:
        shaderStage = IShader::E_SHADER_STAGE::ESS_GEOMETRY; break;
    case spv::ExecutionModelFragment:
        shaderStage = IShader::E_SHADER_STAGE::ESS_FRAGMENT; break;
    case spv::ExecutionModelGLCompute:
        shaderStage = IShader::E_SHADER_STAGE::ESS_COMPUTE; break;
    case spv::ExecutionModelTaskNV:
        shaderStage = IShader::E_SHADER_STAGE::ESS_TASK; break;
    case spv::ExecutionModelMeshNV:
        shaderStage = IShader::E_SHADER_STAGE::ESS_MESH; break;
    case spv::ExecutionModelRayGenerationKHR:
        shaderStage = IShader::E_SHADER_STAGE::ESS_RAYGEN; break;
    case spv::ExecutionModelIntersectionKHR:
        shaderStage = IShader::E_SHADER_STAGE::ESS_INTERSECTION; break;
    case spv::ExecutionModelAnyHitKHR:
        shaderStage = IShader::E_SHADER_STAGE::ESS_ANY_HIT; break;
    case spv::ExecutionModelClosestHitKHR:
        shaderStage = IShader::E_SHADER_STAGE::ESS_MISS; break;
    case spv::ExecutionModelMissKHR:
        shaderStage = IShader::E_SHADER_STAGE::ESS_MISS; break;
    case spv::ExecutionModelCallableKHR:
        shaderStage = IShader::E_SHADER_STAGE::ESS_CALLABLE; break;
    case spv::ExecutionModelKernel:
    case spv::ExecutionModelMax:
    default:
        assert(!"Shader stage not supported!");
        shaderStage = IShader::E_SHADER_STAGE::ESS_UNKNOWN;
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
	
	system::IFile::success_t success;
	_file->read(success, buffer->getPointer(), 0, _file->getSize());
    if (!success)
        return {};

	if (reinterpret_cast<uint32_t*>(buffer->getPointer())[0]!=SPV_MAGIC_NUMBER)
		return {};

	SPIRV_CROSS_NAMESPACE::Parser parser(reinterpret_cast<uint32_t*>(buffer->getPointer()), buffer->getSize() / 4ull);
	parser.parse();
	const SPIRV_CROSS_NAMESPACE::ParsedIR& parsedIR = parser.get_parsed_ir();
	SPIRV_CROSS_NAMESPACE::SPIREntryPoint defaultEntryPoint = parsedIR.entry_points.at(parsedIR.default_entry_point);

    return SAssetBundle(nullptr,{core::make_smart_refctd_ptr<ICPUShader>(std::move(buffer), getShaderStageFromSPIRVCrossExecutionModel(defaultEntryPoint.model), asset::IShader::E_CONTENT_TYPE::ECT_SPIRV, _file->getFileName().string())});
}

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/CentralLimitBoxBlur/CBlurPerformer.h"

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::CentralLimitBoxBlur;

CBlurPerformer::CBlurPerformer(video::ILogicalDevice* device, uint32_t maxDimensionSize, bool useHalfStorage)
    : m_maxBlurLen(maxDimensionSize), m_halfFloatStorage(useHalfStorage)
{
    static IGPUDescriptorSetLayout::SBinding bnd[] =
    {
        {
            0u,
            EDT_STORAGE_BUFFER,
            1u,
            IShader::ESS_COMPUTE,
            nullptr
        },
        {
            1u,
            EDT_STORAGE_BUFFER,
            1u,
            IShader::ESS_COMPUTE,
            nullptr
        },
    };

    m_dsLayout = device->createDescriptorSetLayout(bnd, bnd + sizeof(bnd) / sizeof(IGPUDescriptorSetLayout::SBinding));

    auto pcRange = getDefaultPushConstantRanges();
    m_pplnLayout = device->createPipelineLayout(pcRange.begin(), pcRange.end(), core::smart_refctd_ptr(m_dsLayout));

    auto specShader = createSpecializedShader("nbl/builtin/glsl/ext/CentralLimitBoxBlur/default_compute_blur.comp", m_maxBlurLen, useHalfStorage ? 1u : 0u, device);

    m_ppln = device->createComputePipeline(nullptr, core::smart_refctd_ptr(m_pplnLayout), std::move(specShader));
}

core::SRange<const SPushConstantRange> CBlurPerformer::getDefaultPushConstantRanges()
{
    static const SPushConstantRange ranges[1] =
    {
        {
            IShader::ESS_COMPUTE,
            0u,
            sizeof(Parameters_t)
        },
    };
    return { ranges,ranges + 1 };
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> CBlurPerformer::createSpecializedShader(const char* shaderIncludePath, const uint32_t axisDim, const bool useHalfStorage, video::ILogicalDevice* device)
{
    std::ostringstream shaderSourceStream;
    shaderSourceStream
        << "#version 460 core\n"
        << "#define _NBL_GLSL_WORKGROUP_SIZE_ " << DefaultWorkgroupSize << "\n" // Todo(achal): Get the workgroup size from outside
        << "#define _NBL_GLSL_EXT_BLUR_PASSES_PER_AXIS_ " << PassesPerAxis << "\n" // Todo(achal): Get this from outside?
        << "#define _NBL_GLSL_EXT_BLUR_AXIS_DIM_ " << axisDim << "\n"
        << "#define _NBL_GLSL_EXT_BLUR_HALF_STORAGE_ " << (useHalfStorage ? 1 : 0) << "\n"
        << "#include \"" << shaderIncludePath << "\"\n";

    auto cpuShader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSourceStream.str().c_str(), asset::IShader::ESS_COMPUTE, "CBlurPerformer::createSpecializedShader");
    auto gpuUnspecShader = device->createShader(std::move(cpuShader));
    auto specShader = device->createSpecializedShader(gpuUnspecShader.get(), { nullptr, nullptr, "main" });

    return specShader;
}
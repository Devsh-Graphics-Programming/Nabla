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

    const char* sourceFmt =
R"===(#version 430 core
#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_BLUR_PASS_COUNT_ %u
#define _NBL_GLSL_EXT_BLUR_AXIS_DIM_ %u
#define _NBL_GLSL_EXT_BLUR_HALF_STORAGE_ %u
layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "nbl/builtin/glsl/ext/CentralLimitBoxBlur/default_compute_blur.comp"
)===";

    const size_t extraSize = 4u + 8u + 8u + 128u;

    auto source = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt) + extraSize + 1u);
    snprintf(reinterpret_cast<char*>(source->getPointer()), source->getSize(), sourceFmt, DefaultWorkgroupSize, PassesPerAxis, m_maxBlurLen,
        useHalfStorage ? 1u : 0u);

    // auto shader = device->createShader(core::make_smart_refctd_ptr<ICPUShader>(std::move(source), asset::ICPUShader::buffer_contains_glsl));
    core::smart_refctd_ptr<video::IGPUShader> shader = nullptr;

    auto specializedShader = device->createSpecializedShader(shader.get(),
        asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main"});

    m_ppln = device->createComputePipeline(nullptr, core::smart_refctd_ptr(m_pplnLayout), std::move(specializedShader));
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
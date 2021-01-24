// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/FFT/FFT.h"
#include "../../../../source/Nabla/COpenGLExtensionHandler.h"

#include <cstdio>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::FFT;

core::SRange<const asset::SPushConstantRange> FFT::getDefaultPushConstantRanges()
{
    static const asset::SPushConstantRange range =
    {
        ISpecializedShader::ESS_COMPUTE,
        0u,
        sizeof(uint32_t)
    };
    return {&range,&range+1};
}

core::SRange<const video::IGPUDescriptorSetLayout::SBinding> FFT::getDefaultBindings(video::IVideoDriver* driver)
{
    static core::smart_refctd_ptr<IGPUSampler> sampler;
    static const IGPUDescriptorSetLayout::SBinding bnd[] =
    {
        {
            0u,
            EDT_UNIFORM_BUFFER_DYNAMIC,
            1u,
            ISpecializedShader::ESS_COMPUTE,
            nullptr
        },
        {
            1u,
            EDT_STORAGE_BUFFER_DYNAMIC,
            1u,
            ISpecializedShader::ESS_COMPUTE,
            nullptr
        },
        {
            2u,
            EDT_COMBINED_IMAGE_SAMPLER,
            1u,
            ISpecializedShader::ESS_COMPUTE,
            &sampler
        }
    };
    if (!sampler)
    {
        IGPUSampler::SParams params =
        {
            {
                ISampler::ETC_CLAMP_TO_EDGE,
                ISampler::ETC_CLAMP_TO_EDGE,
                ISampler::ETC_CLAMP_TO_EDGE,
                ISampler::ETBC_FLOAT_OPAQUE_BLACK,
                ISampler::ETF_LINEAR,
                ISampler::ETF_LINEAR,
                ISampler::ESMM_NEAREST,
                0u,
                0u,
                ISampler::ECO_ALWAYS
            }
        };
        sampler = driver->createGPUSampler(params);
    }
    return {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
}

core::smart_refctd_ptr<video::IGPUShader> FFT::createShader(asset::E_FORMAT format)
{
    const char* sourceFmt =
R"===(#version 430 core


#ifndef _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_ 16
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_ 16
#endif

#define _NBL_GLSL_WORKGROUP_SIZE_ (_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_*_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_)

#define _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT %d
#define _NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION %d


#define _NBL_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ %d
#define _NBL_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ %d

#ifndef _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_ %d
#endif

#include "nbl/builtin/glsl/colorspace/EOTF.glsl"
#include "nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "nbl/builtin/glsl/colorspace/OETF.glsl"

#define _NBL_GLSL_EXT_LUMA_METER_EOTF_DEFINED_ %s
#define _NBL_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_ %s
#define _NBL_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
#include "nbl/builtin/glsl/ext/LumaMeter/impl.glsl"


layout(local_size_x=_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_, local_size_y=_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_) in;



layout(set=_NBL_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_, binding=_NBL_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_) uniform Uniforms
{
    nbl_glsl_ext_LumaMeter_Uniforms_t inParams;
};


vec3 nbl_glsl_ext_LumaMeter_getColor(bool wgExecutionMask)
{
    vec3 retval;
    if (wgExecutionMask)
    {
        vec2 uv = vec2(gl_GlobalInvocationID.xy)*inParams.meteringWindowScale+inParams.meteringWindowOffset;
        retval = textureLod(inputImage,vec3(uv,float(gl_GlobalInvocationID.z)),0.0).rgb;
    }
    return retval;
}


void main()
{
    nbl_glsl_ext_LumaMeter(true);
}
)===";

    return {};
}

void FFT::defaultBarrier()
{
    COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
}
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
    static const IGPUDescriptorSetLayout::SBinding bnd[] =
    {
        {
            0u,
            EDT_UNIFORM_BUFFER,
            1u,
            ISpecializedShader::ESS_COMPUTE,
            nullptr
        },
        {
            1u,
            EDT_STORAGE_BUFFER,
            1u,
            ISpecializedShader::ESS_COMPUTE,
            nullptr
        },
        {
            2u,
            EDT_STORAGE_BUFFER,
            1u,
            ISpecializedShader::ESS_COMPUTE,
            nullptr
        },
    };
    
	return {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
}

core::smart_refctd_ptr<video::IGPUSpecializedShader> FFT::createShader(video::IVideoDriver* driver, asset::E_FORMAT format)
{
    const char* sourceFmt =
R"===(#version 430 core

// WorkGroup Size

#ifndef _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_
#define _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_ 16
#endif

#ifndef _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_
#define _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_
#define _NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_ 1
#endif

#define _NBL_GLSL_WORKGROUP_SIZE_ (_NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_*_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_*_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_)

layout(local_size_x=_NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_, local_size_y=_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Y_DEFINED_, local_size_z=_NBL_GLSL_EXT_FFT_BLOCK_SIZE_Z_DEFINED_) in;

#define _NBL_GLSL_EXT_FFT_GET_DATA_DEFINED_
#define _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_
#include "nbl/builtin/glsl/ext/FFT/fft.glsl"

// Uniform

#ifndef _NBL_GLSL_EXT_FFT_UNIFORM_SET_DEFINED_
#define _NBL_GLSL_EXT_FFT_UNIFORM_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_FFT_UNIFORM_BINDING_DEFINED_
#define _NBL_GLSL_EXT_FFT_UNIFORM_BINDING_DEFINED_ 0
#endif

layout(set=_NBL_GLSL_EXT_FFT_UNIFORM_SET_DEFINED_, binding=_NBL_GLSL_EXT_FFT_UNIFORM_BINDING_DEFINED_) uniform Uniforms
{
	nbl_glsl_ext_FFT_Uniforms_t inParams;
};


// Input Descriptor

struct nbl_glsl_ext_FFT_input_t
{
    float real_value;
};

#ifndef _NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_FFT_INPUT_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_FFT_INPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_FFT_INPUT_BINDING_DEFINED_) readonly restrict buffer InputBuffer
{
	nbl_glsl_ext_FFT_input_t inData[];
};

#endif

// Output Descriptor

struct nbl_glsl_ext_FFT_output_t
{
    vec2 complex_value;
};


#ifndef _NBL_GLSL_EXT_FFT_OUTPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_FFT_OUTPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_BINDING_DEFINED_ 2
#endif

#ifndef _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_FFT_OUTPUT_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_FFT_OUTPUT_SET_DEFINED_, binding=_NBL_GLSL_EXT_FFT_OUTPUT_BINDING_DEFINED_) restrict buffer OutputBuffer
{
	nbl_glsl_ext_FFT_output_t outData[];
};
#endif


// Get/Set Data Function

float nbl_glsl_ext_FFT_getData(in uvec3 coordinate, in uint channel)
{
    uvec3 dimension = inParams.dimension;
    uint index = coordinate.z * dimension.x * dimension.y + coordinate.y * dimension.x + coordinate.x;
    return inData[index].real_value;
}

void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in vec2 complex_value)
{
    uvec3 dimension = inParams.dimension;
    uint index = coordinate.z * dimension.x * dimension.y + coordinate.y * dimension.x + coordinate.x;
    outData[index].complex_value = complex_value;
}

void main()
{
	nbl_glsl_ext_FFT(inParams);
}

)===";
    
	const size_t extraSize = 0;

	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(),sourceFmt
	);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
    
	auto gpuShader = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
    
	auto gpuSpecializedShader = driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

    return gpuSpecializedShader;
}

void FFT::defaultBarrier()
{
    COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
}
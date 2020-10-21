// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __IRR_SCENE_I_PROPERTY_POOL_H_INCLUDED__
#define __IRR_SCENE_I_PROPERTY_POOL_H_INCLUDED__


#include "irr/asset/asset.h"

#include "IVideoDriver.h"
#include "irr/video/IGPUComputePipeline.h"


namespace irr
{
namespace video
{

//
constexpr char* copyCsSource = R"(
layout(local_size_x=_IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_, local_size_y=_IRR_BUILTIN_PROPERTY_COPY_COUNT_) in;

layout(set=0,binding=0) readonly restrict buffer Indices
{
    uint elementCount;
    uint indices[];
};


#define IRR_EVAL(ARG) ARG


#define DELCARE_PROPERTY(NUM) \
    layout(set=0,binding=2*(NUM)-1) readonly restrict buffer InData##NUM \
    { \
        IRR_EVAL(_IRR_BUILTIN_PROPERTY_COPY_TYPE_##NUM##_) in##NUM[]; \
    }; \
    layout(set=0,binding=2*(NUM)) writeonly restrict buffer OutData##NUM \
    { \
        IRR_EVAL(_IRR_BUILTIN_PROPERTY_COPY_TYPE_##NUM##_) out##NUM[]; \
    }

#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=1
    DELCARE_PROPERTY(1);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=2
    DELCARE_PROPERTY(2);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=3
    DELCARE_PROPERTY(3);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=4
    DELCARE_PROPERTY(4);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=5
    DELCARE_PROPERTY(5);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=6
    DELCARE_PROPERTY(6);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=7
    DELCARE_PROPERTY(7);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=8
    DELCARE_PROPERTY(8);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=9
    DELCARE_PROPERTY(9);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=10
    DELCARE_PROPERTY(10);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=11
    DELCARE_PROPERTY(11);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=12
    DELCARE_PROPERTY(12);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=13
    DELCARE_PROPERTY(13);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=14
    DELCARE_PROPERTY(14);
#endif
#ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=15
    DELCARE_PROPERTY(15);
#endif

#undef DELCARE_PROPERTY


shared uint workIndices[_IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_];


void main()
{
    uint propID = gl_LocationInvocationID.y;
    if (propID==0u)
        workIndices[gl_LocationInvocationID.x] = indices[gl_GlobalInvocationID.x];
    barrier();
    memoryBarrierShared();

    uint index = gl_GlobalInvocationID.x;
    if (index>=elementCount)
        return;

#ifdef _IRR_BUILTIN_PROPERTY_COPY_DOWNLOAD
    uint inIndex = workIndices[gl_LocationInvocationID.x];
    uint outIndex = index;
#else
    uint inIndex = index;
    uint outIndex = workIndices[gl_LocationInvocationID.x];
#endif


#define COPY_PROPERTY(NUM) \
        case NUM: \
            out##NUM[outIndex] = in##NUM[inIndex]; \
            break

    switch(propID)
    {
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=1
            COPY_PROPERTY(1);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=2
            COPY_PROPERTY(2);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=3
            COPY_PROPERTY(3);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=4
            COPY_PROPERTY(4);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=5
            COPY_PROPERTY(5);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=6
            COPY_PROPERTY(6);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=7
            COPY_PROPERTY(7);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=8
            COPY_PROPERTY(8);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=9
            COPY_PROPERTY(9);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=10
            COPY_PROPERTY(10);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=11
            COPY_PROPERTY(11);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=12
            COPY_PROPERTY(12);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=13
            COPY_PROPERTY(13);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=14
            COPY_PROPERTY(14);
        #endif
        #ifdef _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_>=15
            COPY_PROPERTY(15);
        #endif
    }

#undef COPY_PROPERTY
}
)";

//
core::smart_refctd_ptr<IGPUComputePipeline> IPropertyPool::getCopyPipeline(IVideoDriver* driver, IGPUPipelineCache* pipelineCache, const PipelineKey& key, bool canCompileNew)
{
    const auto propCount = key.getPropertyCount();
    if (!propCount)
    {
        #ifdef _IRR_DEBUG
            assert(false);
        #endif
        return nullptr;
    }

    auto found = copyPipelines.find(key);
    if (found!=copyPipelines.end())
        return found->second;

    if (!canCompileNew)
        return nullptr;

    std::string shaderSource;
    // property count
    shaderSource += "#define _IRR_BUILTIN_PROPERTY_COPY_COUNT_ ";
    shaderSource += std::to_string(propCount)+"\n";
    // workgroup sizes
    shaderSource += "#define _IRR_BUILTIN_PROPERTY_COPY_GROUP_SIZE_ ";
    shaderSource += std::to_string(getWorkGroupSizeX(propCount))+"\n";
    //
    shaderSource += copyCsSource;

    auto cpushader = core::make_smart_refctd_ptr<asset::ICPUShader>(shaderSource.c_str());

    auto shader = driver->createGPUShader(std::move(cpushader));
    auto specshader = driver->createGPUSpecializedShader(shader.get(),{nullptr,nullptr,"main",asset::ISpecializedShader::ESS_COMPUTE});

    IGPUDescriptorSetLayout::SBinding bindings[MaxPropertiesPerCS];
    auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings,bindings+propCount);

    auto layout = driver->createGPUPipelineLayout(nullptr,nullptr,std::move(descriptorSetLayout));

    auto pipeline = driver->createGPUComputePipeline(pipelineCache,std::move(layout),std::move(specshader));
    copyPipelines.insert({key,core::smart_refctd_ptr(pipeline)});
    return pipeline;
}



}
}

#endif
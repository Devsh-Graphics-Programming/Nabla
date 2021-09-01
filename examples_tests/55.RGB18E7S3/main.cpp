// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include <iostream>
#include <cstdio>

#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;

constexpr uint16_t MAX_TEST_RGB_VALUES = 32u;

#include "nbl/nblpack.h"
struct alignas(16) SShaderStorageBufferObject
{
    core::vector3df_SIMD rgb[MAX_TEST_RGB_VALUES]; //! buffer generated and filled on cpp side

    uint64_t rgb_cpp_encoded[MAX_TEST_RGB_VALUES];
    core::vector3df_SIMD rgb_cpp_decoded[MAX_TEST_RGB_VALUES];

    uint64_t rgb_glsl_encoded[MAX_TEST_RGB_VALUES];
    core::vector3df_SIMD rgb_glsl_decoded[MAX_TEST_RGB_VALUES];
} PACK_STRUCT;
#include "nbl/nblunpack.h"

int main()
{
    constexpr std::string_view APP_NAME = "RGB18E7S3 utility test";
    constexpr uint32_t FRAMES_IN_FLIGHT = 5u;

	auto initOutput = CommonAPI::Init(video::EAT_OPENGL, APP_NAME.data());
	auto system = std::move(initOutput.system);
    auto gl = std::move(initOutput.apiConnection);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto renderpass = std::move(initOutput.renderpass);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
    auto utilities = std::move(initOutput.utilities);

    auto gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
    auto gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    {
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
    }

    auto cpu2gpuWaitForFences = [&]() -> void
    {
        video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
        while (waitStatus != video::IGPUFence::ES_SUCCESS)
        {
            waitStatus = logicalDevice->waitForFences(1u, &gpuTransferFence.get(), false, 999999999ull);
            if (waitStatus == video::IGPUFence::ES_ERROR)
                assert(false);
            else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
                break;
        }

        waitStatus = video::IGPUFence::ES_NOT_READY;
        while (waitStatus != video::IGPUFence::ES_SUCCESS)
        {
            waitStatus = logicalDevice->waitForFences(1u, &gpuComputeFence.get(), false, 999999999ull);
            if (waitStatus == video::IGPUFence::ES_ERROR)
                assert(false);
            else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
                break;
        }
    };

    auto createDescriptorPool = [&](const uint32_t textureCount)
    {
        constexpr uint32_t maxItemCount = 256u;
        {
            nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
            poolSize.count = textureCount;
            poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
            return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
        }
    };

    auto getRandomRGB = [&]()
    {
        //! (-2^64,-FLT_MIN] U {0} U [FLT_MIN,2^64)
        //! is a valid range for testing

        static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        static std::uniform_real_distribution<float> distribution_range1(-std::pow(2, 64) + 1, -FLT_MIN);
        static std::uniform_real_distribution<float> distribution_range2(FLT_MIN, std::pow(2, 64));
        
        auto getRandomValue = [&]()
        {
            static const int32_t shift = static_cast<int32_t>(std::log2(RAND_MAX));
            const bool randomBool = (rand() >> shift) & 1;

            return randomBool ? distribution_range1(generator) : distribution_range2(generator);
        };
      
        return core::vector3df_SIMD(getRandomValue(), getRandomValue(), getRandomValue());
    };

    SShaderStorageBufferObject ssbo;

    for (size_t i = 0; i < MAX_TEST_RGB_VALUES; ++i)
    {
        const auto& rgb = ssbo.rgb[i] = getRandomRGB();
        const auto& encoded = ssbo.rgb_cpp_encoded[i] = rgb32f_to_rgb18e7s3(rgb.x, rgb.y, rgb.z);
        const auto& decoded = ssbo.rgb_cpp_decoded[i] = [&]()
        {
            const auto& rgb32f = rgb18e7s3_to_rgb32f(encoded);
            return core::vector3df_SIMD(rgb32f.x, rgb32f.y, rgb32f.z);
        }(); 

        std::cout << "references: " << rgb.x << " " << rgb.y << " " << rgb.z << "\n"
            << "cpp decoded: " << decoded.x << " " << decoded.y << " " << decoded.z << "\n\n";
    }

	return 0;
}
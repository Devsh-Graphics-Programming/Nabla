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

_NBL_STATIC_INLINE_CONSTEXPR size_t WORK_GROUP_SIZE = 32;				                 //! work-items per work-group
_NBL_STATIC_INLINE_CONSTEXPR size_t MAX_TEST_RGB_VALUES = WORK_GROUP_SIZE * 1;           //! total number of rgb values to test
_NBL_STATIC_INLINE_CONSTEXPR size_t FRAMES_IN_FLIGHT = 3u;          

enum E_SSBO
{
    ES_RGB,
    ES_RGB_CPP_DECODED,
    ES_RGB_GLSL_DECODED,
    ES_RGB_CPP_ENCODED,
    ES_RGB_GLSL_ENCODED,
    ES_COUNT
};

#include "nbl/nblpack.h"
struct alignas(16) SShaderStorageBufferObject
{
    core::vector4df_SIMD rgb[MAX_TEST_RGB_VALUES]; //! buffer generated and filled on cpp side
    core::vector4df_SIMD rgb_cpp_decoded[MAX_TEST_RGB_VALUES];
    core::vector4df_SIMD rgb_glsl_decoded[MAX_TEST_RGB_VALUES];

    uint64_t rgb_cpp_encoded[MAX_TEST_RGB_VALUES];
    uint64_t rgb_glsl_encoded[MAX_TEST_RGB_VALUES];
} PACK_STRUCT;
#include "nbl/nblunpack.h"

static_assert(sizeof(SShaderStorageBufferObject) == sizeof(SShaderStorageBufferObject::rgb) + sizeof(SShaderStorageBufferObject::rgb_cpp_encoded) + sizeof(SShaderStorageBufferObject::rgb_cpp_decoded) + sizeof(SShaderStorageBufferObject::rgb_glsl_encoded) + sizeof(SShaderStorageBufferObject::rgb_glsl_decoded), "There will be inproper alignment!");

int main()
{
    constexpr std::string_view APP_NAME = "RGB18E7S3 utility test";

	auto initOutput = CommonAPI::Init(video::EAT_OPENGL, APP_NAME.data());
	auto system = std::move(initOutput.system);
    auto gl = std::move(initOutput.apiConnection);
    auto logger = std::move(initOutput.logger);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto renderpass = std::move(initOutput.renderpass);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
    auto utilities = std::move(initOutput.utilities);

    core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence = nullptr;
    core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence = nullptr;

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    {
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
    }

    auto createDescriptorPool = [&](const uint32_t count, asset::E_DESCRIPTOR_TYPE type)
    {
        constexpr uint32_t maxItemCount = 256u;
        {
            nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
            poolSize.count = count;
            poolSize.type = type;
            return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
        }
    };

    auto computeShaderBundle = assetManager->getAsset("../computeShader.comp", {});
    {
        bool status = !computeShaderBundle.getContents().empty();
        assert(status);
    }

    smart_refctd_ptr<video::IGPUSpecializedShader> gpuComputeShader;
    {
        auto cpuComputeShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(computeShaderBundle.getContents().begin()[0]);

        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuComputeShader, &cpuComputeShader + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        cpu2gpuParams.waitForCreationToComplete();
        gpuComputeShader = (*gpu_array)[0];
    }

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
      
        return core::vector4df_SIMD(getRandomValue(), getRandomValue(), getRandomValue());
    };

    SShaderStorageBufferObject ssbo;
    {
        for (size_t i = 0; i < MAX_TEST_RGB_VALUES; ++i)
        {
            const auto& rgb = ssbo.rgb[i] = getRandomRGB();
            const auto& encoded = ssbo.rgb_cpp_encoded[i] = rgb32f_to_rgb18e7s3(rgb.x, rgb.y, rgb.z);
            const auto& decoded = ssbo.rgb_cpp_decoded[i] = [&]()
            {
                const auto& rgb32f = rgb18e7s3_to_rgb32f(encoded);
                return core::vector4df_SIMD(rgb32f.x, rgb32f.y, rgb32f.z);
            }();
        }
    }

    auto ssboMemoryReqs = logicalDevice->getDeviceLocalGPUMemoryReqs();
    ssboMemoryReqs.vulkanReqs.size = sizeof(SShaderStorageBufferObject);
    ssboMemoryReqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;

    video::IGPUBuffer::SCreationParams ssboCreationParams;
    ssboCreationParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
    ssboCreationParams.canUpdateSubRange = true;
    ssboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
    ssboCreationParams.queueFamilyIndexCount = 0u;
    ssboCreationParams.queueFamilyIndices = nullptr;

    auto gpuDownloadSSBOmapped = logicalDevice->createGPUBufferOnDedMem(ssboCreationParams,ssboMemoryReqs);

    video::IGPUDescriptorSetLayout::SBinding gpuBindingsLayout[ES_COUNT] =
    {
        {ES_RGB, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
        {ES_RGB_CPP_DECODED, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
        {ES_RGB_GLSL_DECODED, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
        {ES_RGB_CPP_ENCODED, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr},
        {ES_RGB_GLSL_ENCODED, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr}
    };

    auto gpuCDescriptorPool = createDescriptorPool(ES_COUNT, asset::EDT_STORAGE_BUFFER);
    auto gpuCDescriptorSetLayout = logicalDevice->createGPUDescriptorSetLayout(gpuBindingsLayout, gpuBindingsLayout + ES_COUNT);
    auto gpuCDescriptorSet = logicalDevice->createGPUDescriptorSet(gpuCDescriptorPool.get(), core::smart_refctd_ptr(gpuCDescriptorSetLayout));
    {
        video::IGPUDescriptorSet::SDescriptorInfo gpuDescriptorSetInfos[ES_COUNT];

        gpuDescriptorSetInfos[ES_RGB].desc = core::smart_refctd_ptr(gpuDownloadSSBOmapped);
        gpuDescriptorSetInfos[ES_RGB].buffer.size = sizeof(SShaderStorageBufferObject::rgb);
        gpuDescriptorSetInfos[ES_RGB].buffer.offset = 0;

        gpuDescriptorSetInfos[ES_RGB_CPP_DECODED].desc = core::smart_refctd_ptr(gpuDownloadSSBOmapped);
        gpuDescriptorSetInfos[ES_RGB_CPP_DECODED].buffer.size = sizeof(SShaderStorageBufferObject::rgb_cpp_decoded);
        gpuDescriptorSetInfos[ES_RGB_CPP_DECODED].buffer.offset = gpuDescriptorSetInfos[ES_RGB].buffer.size;

        gpuDescriptorSetInfos[ES_RGB_GLSL_DECODED].desc = core::smart_refctd_ptr(gpuDownloadSSBOmapped);
        gpuDescriptorSetInfos[ES_RGB_GLSL_DECODED].buffer.size = sizeof(SShaderStorageBufferObject::rgb_glsl_decoded);
        gpuDescriptorSetInfos[ES_RGB_GLSL_DECODED].buffer.offset = gpuDescriptorSetInfos[ES_RGB_CPP_DECODED].buffer.offset + gpuDescriptorSetInfos[ES_RGB_CPP_DECODED].buffer.size;

        gpuDescriptorSetInfos[ES_RGB_CPP_ENCODED].desc = core::smart_refctd_ptr(gpuDownloadSSBOmapped);
        gpuDescriptorSetInfos[ES_RGB_CPP_ENCODED].buffer.size = sizeof(SShaderStorageBufferObject::rgb_cpp_encoded);
        gpuDescriptorSetInfos[ES_RGB_CPP_ENCODED].buffer.offset = gpuDescriptorSetInfos[ES_RGB_GLSL_DECODED].buffer.offset + gpuDescriptorSetInfos[ES_RGB_GLSL_DECODED].buffer.size;

        gpuDescriptorSetInfos[ES_RGB_GLSL_ENCODED].desc = core::smart_refctd_ptr(gpuDownloadSSBOmapped);
        gpuDescriptorSetInfos[ES_RGB_GLSL_ENCODED].buffer.size = sizeof(SShaderStorageBufferObject::rgb_glsl_encoded);
        gpuDescriptorSetInfos[ES_RGB_GLSL_ENCODED].buffer.offset = gpuDescriptorSetInfos[ES_RGB_CPP_ENCODED].buffer.offset + gpuDescriptorSetInfos[ES_RGB_CPP_ENCODED].buffer.size;

        video::IGPUDescriptorSet::SWriteDescriptorSet gpuWrites[ES_COUNT];
        {
            for (uint32_t binding = 0u; binding < ES_COUNT; binding++)
                gpuWrites[binding] = { gpuCDescriptorSet.get(), binding, 0u, 1u, asset::EDT_STORAGE_BUFFER, gpuDescriptorSetInfos + binding };
            logicalDevice->updateDescriptorSets(ES_COUNT, gpuWrites, 0u, nullptr);
        }
    }

    auto gpuCPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, std::move(gpuCDescriptorSetLayout), nullptr, nullptr, nullptr);
    auto gpuComputePipeline = logicalDevice->createGPUComputePipeline(nullptr, std::move(gpuCPipelineLayout), std::move(gpuComputeShader));

    core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
    logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);
    auto gpuFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

    for(size_t i = 0; i < FRAMES_IN_FLIGHT; ++i)
    {
        auto& commandBuffer = commandBuffers[i];

        commandBuffer->begin(0);

        commandBuffer->bindComputePipeline(gpuComputePipeline.get());
        commandBuffer->bindDescriptorSets(asset::EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0, 1, &gpuCDescriptorSet.get(), nullptr);

        static_assert(MAX_TEST_RGB_VALUES % WORK_GROUP_SIZE == 0, "Inccorect amount!");
        _NBL_STATIC_INLINE_CONSTEXPR size_t groupCountX = MAX_TEST_RGB_VALUES / WORK_GROUP_SIZE;

        commandBuffer->updateBuffer(gpuDownloadSSBOmapped.get(), 0, sizeof(SShaderStorageBufferObject), &ssbo);
        commandBuffer->dispatch(groupCountX, 1, 1);
        commandBuffer->end();

        video::IGPUQueue::SSubmitInfo submit;
        {
            submit.commandBufferCount = 1u;
            submit.commandBuffers = &commandBuffer.get();
            submit.signalSemaphoreCount = {};
            submit.pSignalSemaphores = nullptr;
            asset::E_PIPELINE_STAGE_FLAGS dstWait = asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
            submit.waitSemaphoreCount = {};
            submit.pWaitSemaphores = nullptr;
            submit.pWaitDstStageMask = &dstWait;

            queues[decltype(initOutput)::EQT_COMPUTE]->submit(1u, &submit, gpuFence.get());
        }
    }

    {
        video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
        while (waitStatus != video::IGPUFence::ES_SUCCESS)
        {
            waitStatus = logicalDevice->waitForFences(1u, &gpuFence.get(), false, 99999999999ull);
            if (waitStatus == video::IGPUFence::ES_ERROR)
                assert(false);
            else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
                break;
        }
    }

    video::IDriverMemoryAllocation::MappedMemoryRange mappedMemoryRange(gpuDownloadSSBOmapped->getBoundMemory(), 0u, gpuDownloadSSBOmapped->getSize());
    logicalDevice->mapMemory(mappedMemoryRange, video::IDriverMemoryAllocation::EMCAF_READ);

    if (gpuDownloadSSBOmapped->getBoundMemory()->haveToMakeVisible())
        logicalDevice->invalidateMappedMemoryRanges(1u, &mappedMemoryRange);

    auto* gpuSsbo = reinterpret_cast<SShaderStorageBufferObject*>(gpuDownloadSSBOmapped->getBoundMemory()->getMappedPointer());
    {
        for (size_t i = 0; i < MAX_TEST_RGB_VALUES; ++i)
        {
            const auto& rgb_reference = gpuSsbo->rgb[i];

            const auto& cpp_encoded = gpuSsbo->rgb_cpp_encoded[i];
            const auto& cpp_decoded = gpuSsbo->rgb_cpp_decoded[i];

            const auto& glsl_encoded = gpuSsbo->rgb_glsl_encoded[i];
            const auto& glsl_decoded = gpuSsbo->rgb_cpp_decoded[i];

            const std::string logMessage_result = "rgb[" + std::to_string(i) + "] result";
            const std::string logMessage_performance =
                "\n[references]: " + std::to_string(rgb_reference.x) + " " + std::to_string(rgb_reference.y) + " " + std::to_string(rgb_reference.z) + "\n" +
                "[cpp encoded]: " + std::to_string(cpp_encoded) + "\n" +
                "[cpp decoded]: " + std::to_string(cpp_decoded.x) + " " + std::to_string(cpp_decoded.y) + " " + std::to_string(cpp_decoded.z) + "\n" +
                "[glsl encoded]: " + std::to_string(glsl_encoded) + "\n" +
                "[glsl decoded]: " + std::to_string(glsl_decoded.x) + " " + std::to_string(glsl_decoded.y) + " " + std::to_string(glsl_decoded.z) + "\n\n";

            logger->log(logMessage_result, system::ILogger::ELL_WARNING);
            logger->log(logMessage_performance, system::ILogger::ELL_PERFORMANCE);

        }
    }

    logicalDevice->unmapMemory(gpuDownloadSSBOmapped->getBoundMemory());

	return 0;
}
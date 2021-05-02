// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "../common/QToQuitEventReceiver.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "glm/glm.hpp"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

template <typename T>
static T* DebugGPUBufferDownload(smart_refctd_ptr<IGPUBuffer> buffer_to_download, size_t buffer_size, IVideoDriver* driver)
{
    constexpr uint64_t timeout_ns = 15000000000u;
    const uint32_t alignment = uint32_t(sizeof(T));
    auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
    auto downBuffer = downloadStagingArea->getBuffer();

    bool success = false;

    uint32_t array_size_32 = uint32_t(buffer_size);
    uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
    auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &array_size_32, &alignment);
    if (unallocatedSize)
    {
        os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
        exit(420);
    }

    driver->copyBuffer(buffer_to_download.get(), downBuffer, 0, address, array_size_32);

    auto downloadFence = driver->placeFence(true);
    auto result = downloadFence->waitCPU(timeout_ns, true);

    T* dataFromBuffer = nullptr;
    if (result != video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED && result != video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
    {
        if (downloadStagingArea->needsManualFlushOrInvalidate())
            driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,array_size_32} });

        dataFromBuffer = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address);
    }
    else
    {
        os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
    }

    downloadStagingArea->multi_free(1u, &address, &array_size_32, nullptr);

    return dataFromBuffer;
}

template <typename T>
static void DebugCompareGPUvsCPU(smart_refctd_ptr<IGPUBuffer> gpu_buffer, T* cpu_buffer, size_t buffer_size, IVideoDriver* driver)
{
    T* downloaded_buffer = DebugGPUBufferDownload<T>(gpu_buffer, buffer_size, driver);

    size_t buffer_count = buffer_size / sizeof(T);

    if (downloaded_buffer)
    {
        for (int i = 0; i < buffer_count; ++i)
        {
            const glm::vec4 error(1e-4f);
            glm::bvec4 result = glm::greaterThanEqual(glm::abs(downloaded_buffer[i] - cpu_buffer[i]), error);
            if (result.x || result.y || result.z)
                __debugbreak();

            // const float error = 1e-4f;
            // if (abs(downloaded_buffer[i] - cpu_buffer[i]) >= error)
            //     __debugbreak();
        }

        std::cout << "PASS" << std::endl;
    }
}

smart_refctd_ptr<IGPUSpecializedShader> createShader(const char* shader_file_path, IVideoDriver* driver, io::IFileSystem* filesystem, asset::IAssetManager* am)
{
    auto file = smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile(shader_file_path));

    asset::IAssetLoader::SAssetLoadParams lp;
    auto cs_bundle = am->getAsset(shader_file_path, lp);
    auto cs = smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
    auto cs_rawptr = cs.get();

    return driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
}

int main()
{
    nbl::SIrrlichtCreationParameters deviceParams;
    deviceParams.Bits = 24; //may have to set to 32bit for some platforms
    deviceParams.ZBufferBits = 24; //we'd like 32bit here
    deviceParams.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    deviceParams.WindowSize = dimension2d<uint32_t>(1024u, 1024u);
    deviceParams.Fullscreen = false;
    deviceParams.Vsync = true; //! If supported by target platform
    deviceParams.Doublebuffer = true;
    deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

    auto device = createDeviceEx(deviceParams);
    if (!device)
        return 1; // could not create selected driver.

    QToQuitEventReceiver receiver;
    device->setEventReceiver(&receiver);

    video::IVideoDriver* driver = device->getVideoDriver();

    nbl::io::IFileSystem* filesystem = device->getFileSystem();
    asset::IAssetManager* am = device->getAssetManager();

    const uint32_t in_count = 4096;
    const size_t in_size = in_count * sizeof(glm::vec4);
    // const size_t in_size = in_count * sizeof(float);

    std::vector<glm::vec4> in(in_count);
    // std::vector<float> in(in_count);
    for (uint32_t i = 0; i < in_count; ++i)
        // in[i] = float(i) / float(in_count);
        // in[i] = { float(i) / float(in_count), 2.f * float(i) / float(in_count), 3.f * float(i) / float(in_count) };
        in[i] = { float(i) / float(in_count), 0.f, 0.f, 0.f };

    auto in_gpu = driver->createFilledDeviceLocalGPUBufferOnDedMem(in_size, in.data());
    auto out_gpu = driver->createDeviceLocalGPUBufferOnDedMem(in_size);

    smart_refctd_ptr<IGPUDescriptorSet> ds = nullptr;
    smart_refctd_ptr<IGPUComputePipeline> pipeline = nullptr;
    {
        const uint32_t count = 2u;
        video::IGPUDescriptorSetLayout::SBinding binding[count];
        for (uint32_t i = 0; i < count; ++i)
            binding[i] = { i, asset::EDT_STORAGE_BUFFER, 1u, video::IGPUSpecializedShader::ESS_COMPUTE, nullptr };

        auto ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
        auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(ds_layout));

        ds = driver->createGPUDescriptorSet(ds_layout);
        pipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipeline_layout),
            createShader("../BlurPass.comp", driver, filesystem, am));
    }

    driver->beginScene();
    {
        {
            const uint32_t count = 2u;
            video::IGPUDescriptorSet::SDescriptorInfo ds_info[count];

            ds_info[0].desc = in_gpu;
            ds_info[0].buffer = { 0, in_gpu->getSize() };

            ds_info[1].desc = out_gpu;
            ds_info[1].buffer = { 0, out_gpu->getSize() };

            video::IGPUDescriptorSet::SWriteDescriptorSet writes[count];

            for (uint32_t i = 0; i < count; ++i)
                writes[i] = { ds.get(), i, 0u, 1u, asset::EDT_STORAGE_BUFFER, ds_info + i };

            driver->updateDescriptorSets(count, writes, 0u, nullptr);
        }

        driver->bindComputePipeline(pipeline.get());
        driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &ds.get(), nullptr);
        driver->dispatch(1u, 1u, 1u);

        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    driver->endScene();

    glm::vec4* out = DebugGPUBufferDownload<glm::vec4>(out_gpu, out_gpu->getSize(), driver);
    // float* out = DebugGPUBufferDownload<float>(out_gpu, out_gpu->getSize(), driver);

    std::vector<glm::vec4> blurred(in_count);
    memcpy(blurred.data(), in.data(), in_count * sizeof(glm::vec4));
    // std::vector<float> blurred(in_count);
    // memcpy(blurred.data(), in.data(), in_count * sizeof(float));

    const uint32_t channel_count = 3u;
    const uint32_t pass_count = 3u;
    for (uint32_t ch = 0; ch < channel_count; ++ch)
    {
        for (uint32_t pass = 0; pass < pass_count; ++pass)
        {
            float prefix_sum[in_count];
            float scan = 0.f;
            for (uint32_t i = 0; i < in_count; ++i)
            {
                scan += blurred[i][ch];
                // scan += blurred[i];
                prefix_sum[i] = scan;
            }

            const float RADIUS = 1.73f;
            const uint32_t last = in_count - 1u;

            for (uint32_t idx = 0; idx < in_count; ++idx)
            {
                float left = float(idx) - RADIUS - 1.f;
                float right = float(idx) + RADIUS;

                float result;
                if (right > last)
                {
                    result = (right - float(last)) * (prefix_sum[last] - prefix_sum[last - 1u]) + prefix_sum[last];
                }
                else
                {
                    uint32_t floored = uint32_t(floor(right));
                    result = prefix_sum[floored] * (1.f - fract(right)) + prefix_sum[floored + 1u] * fract(right);
                }

                if (left < 0)
                {
                    result -= (1.f - abs(left)) * prefix_sum[0u];
                }
                else
                {
                    uint32_t floored = uint32_t(floor(left));
                    result -= prefix_sum[floored] * (1.f - fract(left)) + prefix_sum[floored + 1u] * fract(left);
                }

                blurred[idx][ch] = result / (2.f * RADIUS + 1.f);
                // blurred[idx] = result / (2.f * RADIUS + 1.f);
            }
        }
    }
    
    DebugCompareGPUvsCPU<glm::vec4>(out_gpu, blurred.data(), out_gpu->getSize(), driver);
    // DebugCompareGPUvsCPU<float>(out_gpu, blurred.data(), out_gpu->getSize(), driver);


#if 0
    IAssetLoader::SAssetLoadParams lp;
    auto in_image_bundle = am->getAsset("../tex.jpg", lp);

    // Todo: Flip the image

    smart_refctd_ptr<IGPUImageView> in_image_view;
    {
        auto in_gpu_image = driver->getGPUObjectsFromAssets<ICPUImage>(in_image_bundle.getContents());

        IGPUImageView::SCreationParams in_image_view_info;
        in_image_view_info.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
        in_image_view_info.image = in_gpu_image->operator[](0u);
        in_image_view_info.viewType = IGPUImageView::ET_2D;
        in_image_view_info.format = in_image_view_info.image->getCreationParameters().format;
        in_image_view_info.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
        in_image_view_info.subresourceRange.baseMipLevel = 0;
        in_image_view_info.subresourceRange.levelCount = 1;
        in_image_view_info.subresourceRange.baseArrayLayer = 0;
        in_image_view_info.subresourceRange.layerCount = 1;
        in_image_view = driver->createGPUImageView(std::move(in_image_view_info));
    }

    const vector2d<uint32_t> blur_ds_factor = { 7u, 3u };
    const float blur_radius = 0.01f;
    const uint32_t passes_per_axis = 3u;

    auto in_dim = in_image_view->getCreationParameters().image->getCreationParameters().extent;
    vector2d<uint32_t> out_dim = vector2d<uint32_t>(in_dim.width, in_dim.height) / blur_ds_factor;

    // Create out image
    // Todo: Clean this up, a lot of this state could match with the input image
    smart_refctd_ptr<IGPUImage> out_image;
    smart_refctd_ptr<IGPUImageView> out_image_view;
    {
        IGPUImageView::SCreationParams out_image_view_info;
        out_image_view_info.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
        out_image_view_info.viewType = IGPUImageView::ET_2D;
        out_image_view_info.format = asset::EF_R16G16B16A16_SFLOAT;

        out_image_view_info.components.r = IGPUImageView::SComponentMapping::ES_R;
        out_image_view_info.components.g = IGPUImageView::SComponentMapping::ES_G;
        out_image_view_info.components.b = IGPUImageView::SComponentMapping::ES_B;
        out_image_view_info.components.a = IGPUImageView::SComponentMapping::ES_A;

        out_image_view_info.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
        out_image_view_info.subresourceRange.baseMipLevel = 0;
        out_image_view_info.subresourceRange.levelCount = 1;
        out_image_view_info.subresourceRange.baseArrayLayer = 0;
        out_image_view_info.subresourceRange.layerCount = 1;

        IImage::SCreationParams out_image_info;
        out_image_info.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
        out_image_info.type = IImage::ET_2D;
        out_image_info.format = asset::EF_R16G16B16A16_SFLOAT;
        out_image_info.extent = { out_dim.X, out_dim.Y, 1u };
        out_image_info.mipLevels = 1u; // Check this by inputting an image dims 146x341 and seeing what Nabla sets this value to
        out_image_info.arrayLayers = 1u;
        out_image_info.samples = IImage::ESCF_1_BIT;

        out_image = driver->createDeviceLocalGPUImageOnDedMem(std::move(out_image_info));
    
        out_image_view_info.image = out_image;
        out_image_view = driver->createGPUImageView(IGPUImageView::SCreationParams(out_image_view_info));
    }

    smart_refctd_ptr<IGPUDescriptorSet> ds = nullptr;
    smart_refctd_ptr<IGPUComputePipeline> pipeline = nullptr;
    {
        const uint32_t count = 2u;
        IGPUDescriptorSetLayout::SBinding binding[count] =
        {
            {
                0u,
                EDT_COMBINED_IMAGE_SAMPLER,
                1u,
                ISpecializedShader::ESS_COMPUTE,
                nullptr
            },
            {
                1u,
                EDT_STORAGE_IMAGE,
                1u,
                ISpecializedShader::ESS_COMPUTE,
                nullptr
            }
        };

        auto ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
        ds = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout));

        auto pipeline_layout = driver->createGPUPipelineLayout(nullptr, nullptr, std::move(ds_layout));
        pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), createShader("../BlurPass.comp", driver, filesystem, am));
    }

    auto blit_fbo = driver->addFrameBuffer();
    blit_fbo->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(out_image_view));

    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(false, false);

        IGPUSampler::SParams params =
        {
            {
                ISampler::ETC_CLAMP_TO_BORDER,
                ISampler::ETC_CLAMP_TO_BORDER,
                ISampler::ETC_CLAMP_TO_BORDER,
                ISampler::ETBC_FLOAT_OPAQUE_BLACK,
                ISampler::ETF_LINEAR,
                ISampler::ETF_LINEAR,
                ISampler::ESMM_LINEAR,
                8u,
                0u,
                ISampler::ECO_ALWAYS
            }
        };
        auto sampler = driver->createGPUSampler(std::move(params));

        constexpr auto descriptor_count = 2u;
        video::IGPUDescriptorSet::SDescriptorInfo ds_infos[descriptor_count];
        video::IGPUDescriptorSet::SWriteDescriptorSet ds_writes[descriptor_count];

        for (uint32_t i = 0; i < descriptor_count; ++i)
        {
            ds_writes[i].dstSet = ds.get();
            ds_writes[i].arrayElement = 0u;
            ds_writes[i].count = 1u;
            ds_writes[i].info = ds_infos + i;
        }

        // Input sampler2D
        ds_writes[0].binding = 0;
        ds_writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
        ds_writes[0].count = 1;

        ds_infos[0].desc = in_image_view;
        ds_infos[0].image.sampler = sampler;
        ds_infos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

        // Output image2D 
        ds_writes[1].binding = 1;
        ds_writes[1].descriptorType = asset::EDT_STORAGE_IMAGE;
        ds_writes[1].count = 1;

        ds_infos[1].desc = out_image_view;
        ds_infos[1].image.sampler = nullptr;
        ds_infos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

        driver->updateDescriptorSets(2u, ds_writes, 0u, nullptr);

        driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &ds.get(), nullptr);
        driver->bindComputePipeline(pipeline.get());
        driver->dispatch(1, out_dim.Y, 1);

        // Todo: You might also want to use texture fetch barrier bit
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        driver->blitRenderTargets(blit_fbo, nullptr, false, false);

        driver->endScene();
    }
#endif

    return 0;
}
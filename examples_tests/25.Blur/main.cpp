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

    const vector2d<uint32_t> blur_ds_factor = { 1u, 1u };// { 7u, 3u };
    // const float blur_radius = 0.01f;
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
        pipeline = driver->createGPUComputePipeline(nullptr, std::move(pipeline_layout), createShader("../BlurPassHorizontal.comp", driver, filesystem, am));
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
        driver->dispatch(out_dim.Y, 1, 1);

        // Todo(achal): You might also want to use texture fetch barrier bit
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        driver->blitRenderTargets(blit_fbo, nullptr, false, false);

        driver->endScene();
    }

    return 0;
}
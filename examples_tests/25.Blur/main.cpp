// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "../common/QToQuitEventReceiver.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

typedef uint32_t uint;
struct alignas(16) uvec4
{
    uint x, y, z, w;
};
#include "nbl/builtin/glsl/ext/CentralLimitBoxBlur/parameters_struct.glsl"
typedef nbl_glsl_ext_Blur_Parameters_t Parameters_t;

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

#define WG_SIZE 256
#define PASS_COUNT 3

Parameters_t blur_params = {};

class MouseEventReceiver : public IEventReceiver
{
    _NBL_STATIC_INLINE_CONSTEXPR float BLUR_RADIUS_MIN = 0.f, BLUR_RADIUS_MAX = 0.5f;

public:
    bool OnEvent(const SEvent& e)
    {
        if (e.EventType == nbl::EET_MOUSE_INPUT_EVENT)
        {
            const float r = blur_params.radius + e.MouseInput.Wheel / 500.f;
            blur_params.radius = core::max(BLUR_RADIUS_MIN, core::min(r, BLUR_RADIUS_MAX));

            // return true;
        }

        return false;
    }
};

smart_refctd_ptr<IGPUSpecializedShader> createShader(const char* shader_include_path, const uint32_t axis_dim, const bool use_half_storage, IVideoDriver* driver)
{
    const char* source_fmt =
R"===(#version 430 core
#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_BLUR_PASS_COUNT_ %u
#define _NBL_GLSL_EXT_BLUR_AXIS_DIM_ %u
#define _NBL_GLSL_EXT_BLUR_HALF_STORAGE_ %u
layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "%s"
)===";

    // Todo: This just the value I took from FFT example, don't know how it is being computed.
    const size_t extraSize = 4u + 8u + 8u + 128u;

    auto shader = core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(source_fmt) + extraSize + 1u);
    snprintf(reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), source_fmt, WG_SIZE, PASS_COUNT, axis_dim, use_half_storage ? 1u : 0u,
        shader_include_path);

    auto cpu_specialized_shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
        core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl),
        asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

    auto gpu_shader = driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
    return driver->createGPUSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());
}

void updateDescriptorSet(IGPUDescriptorSet* ds, smart_refctd_ptr<IGPUImageView> desc0, smart_refctd_ptr<IGPUImageView> desc1, smart_refctd_ptr<IGPUSampler> sampler,
    IVideoDriver* driver)
{
    constexpr uint32_t descriptor_count = 2u;
    IGPUDescriptorSet::SDescriptorInfo ds_infos[descriptor_count];
    IGPUDescriptorSet::SWriteDescriptorSet ds_writes[descriptor_count];

    for (uint32_t i = 0; i < descriptor_count; ++i)
    {
        ds_writes[i].dstSet = ds;
        ds_writes[i].arrayElement = 0u;
        ds_writes[i].count = 1u;
        ds_writes[i].info = ds_infos + i;
    }

    // Input sampler2D
    ds_writes[0].binding = 0;
    ds_writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
    ds_writes[0].count = 1;

    ds_infos[0].desc = desc0;
    ds_infos[0].image.sampler = sampler;
    ds_infos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

    // Output image2D 
    ds_writes[1].binding = 1;
    ds_writes[1].descriptorType = asset::EDT_STORAGE_IMAGE;
    ds_writes[1].count = 1;

    ds_infos[1].desc = desc1;
    ds_infos[1].image.sampler = nullptr;
    ds_infos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

    driver->updateDescriptorSets(descriptor_count, ds_writes, 0u, nullptr);
}

int main()
{
    nbl::SIrrlichtCreationParameters deviceParams;
    deviceParams.Bits = 24; //may have to set to 32bit for some platforms
    deviceParams.ZBufferBits = 24; //we'd like 32bit here
    deviceParams.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    deviceParams.WindowSize = dimension2d<uint32_t>(1024, 1024);
    deviceParams.Fullscreen = false;
    deviceParams.Vsync = true; //! If supported by target platform
    deviceParams.Doublebuffer = true;
    deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

    auto device = createDeviceEx(deviceParams);
    if (!device)
        return 1; // could not create selected driver.

    MouseEventReceiver receiver;
    device->setEventReceiver(&receiver);

    video::IVideoDriver* driver = device->getVideoDriver();

    nbl::io::IFileSystem* filesystem = device->getFileSystem();
    asset::IAssetManager* am = device->getAssetManager();

    IAssetLoader::SAssetLoadParams lp;
    auto in_image_bundle = am->getAsset("../cube_face.jpg", lp);

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

    const vector2d<uint32_t> blur_ds_factor = { 2u, 2u };
    const uint32_t passes_per_axis = 3u;
    const bool use_half_storage = false;
    const uint32_t channel_count = getFormatChannelCount(in_image_view->getCreationParameters().format);
    uint32_t blur_direction = 0u;

    auto in_dim = in_image_view->getCreationParameters().image->getCreationParameters().extent;
    vector2d<uint32_t> out_dim = vector2d<uint32_t>(in_dim.width, in_dim.height) / blur_ds_factor;

    // Create out image
    // Todo(achal): Clean this up, a lot of this state could match with the input image
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
        out_image_info.mipLevels = 1u;
        out_image_info.arrayLayers = 1u;
        out_image_info.samples = IImage::ESCF_1_BIT;

        out_image = driver->createDeviceLocalGPUImageOnDedMem(std::move(out_image_info));
    
        out_image_view_info.image = out_image;
        out_image_view = driver->createGPUImageView(IGPUImageView::SCreationParams(out_image_view_info));
    }

    const size_t scratch_samples_size = out_dim.X * out_dim.Y * channel_count * sizeof(float);
    auto scratch_samples_gpu = driver->createDeviceLocalGPUBufferOnDedMem(scratch_samples_size);

    const SPushConstantRange pc_range = { ISpecializedShader::ESS_COMPUTE, 0u, sizeof(Parameters_t) };

    // sampler2D -> SSBO
    smart_refctd_ptr<IGPUDescriptorSet> ds_horizontal = nullptr;
    smart_refctd_ptr<IGPUComputePipeline> pipeline_horizontal = nullptr;
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
                EDT_STORAGE_BUFFER,
                1u,
                ISpecializedShader::ESS_COMPUTE,
                nullptr
            }
        };

        auto ds_layout = driver->createGPUDescriptorSetLayout(binding, binding + count);
        ds_horizontal = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout));

        auto pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, std::move(ds_layout));
        pipeline_horizontal = driver->createGPUComputePipeline(nullptr, smart_refctd_ptr(pipeline_layout), createShader("../BlurPassHorizontal.comp", out_dim.X, use_half_storage, driver));
    }

    // SSBO -> image2D
    smart_refctd_ptr<IGPUDescriptorSet> ds_vertical = nullptr;
    smart_refctd_ptr<IGPUComputePipeline> pipeline_vertical = nullptr;
    {
        const uint32_t count = 2u;
        IGPUDescriptorSetLayout::SBinding binding[count] =
        {
            {
                0u,
                EDT_STORAGE_BUFFER,
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
        ds_vertical = driver->createGPUDescriptorSet(smart_refctd_ptr(ds_layout));

        auto pipeline_layout = driver->createGPUPipelineLayout(&pc_range, &pc_range + 1, std::move(ds_layout));
        pipeline_vertical = driver->createGPUComputePipeline(nullptr, smart_refctd_ptr(pipeline_layout), createShader("../BlurPassVertical.comp", out_dim.Y, use_half_storage, driver));
    }

    // buildParameters
    blur_params.input_dimensions = { out_dim.X, out_dim.Y, 1 };
    blur_params.input_dimensions.w = (blur_direction << 30) | (channel_count << 28);
    blur_params.radius = 0.1f;

    // Todo(achal): This forces the user to do the operation in the axis order: XYZ --I think
    // it could be more flexible?
    blur_params.input_strides = { 1, out_dim.X, out_dim.X * out_dim.Y, out_dim.X * out_dim.Y * 1 };
    blur_params.output_strides = blur_params.input_strides;

    // Update descriptor sets
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
                ISampler::ESMM_LINEAR,
                8u,
                0u,
                ISampler::ECO_ALWAYS
            }
        };
        auto sampler = driver->createGPUSampler(std::move(params));

        constexpr uint32_t descriptor_count = 2u;
        IGPUDescriptorSet::SDescriptorInfo ds_infos[descriptor_count];
        IGPUDescriptorSet::SWriteDescriptorSet ds_writes[descriptor_count];

        for (uint32_t i = 0; i < descriptor_count; ++i)
        {
            ds_writes[i].dstSet = ds_horizontal.get();
            ds_writes[i].arrayElement = 0u;
            ds_writes[i].count = 1u;
            ds_writes[i].info = ds_infos + i;
        }

        // Input sampler2D
        ds_infos[0].desc = in_image_view;
        ds_infos[0].image.sampler = sampler;
        ds_infos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

        ds_writes[0].binding = 0;
        ds_writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;

        // Output SSBO 
        ds_infos[1].desc = scratch_samples_gpu;
        ds_infos[1].buffer = { 0u, scratch_samples_gpu->getSize() };

        ds_writes[1].binding = 1;
        ds_writes[1].descriptorType = asset::EDT_STORAGE_BUFFER;

        driver->updateDescriptorSets(descriptor_count, ds_writes, 0u, nullptr);
    }

    {
        constexpr uint32_t descriptor_count = 2u;
        IGPUDescriptorSet::SDescriptorInfo ds_infos[descriptor_count];
        IGPUDescriptorSet::SWriteDescriptorSet ds_writes[descriptor_count];

        for (uint32_t i = 0; i < descriptor_count; ++i)
        {
            ds_writes[i].dstSet = ds_vertical.get();
            ds_writes[i].arrayElement = 0u;
            ds_writes[i].count = 1u;
            ds_writes[i].info = ds_infos + i;
        }

        // Input SSBO
        ds_infos[0].desc = scratch_samples_gpu;
        ds_infos[0].buffer = { 0u, scratch_samples_gpu->getSize() };

        ds_writes[0].binding = 0;
        ds_writes[0].descriptorType = asset::EDT_STORAGE_BUFFER;

        // Output image2D
        ds_writes[1].binding = 1;
        ds_writes[1].descriptorType = asset::EDT_STORAGE_IMAGE;

        ds_infos[1].desc = out_image_view;
        ds_infos[1].image.sampler = nullptr;
        ds_infos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

        driver->updateDescriptorSets(descriptor_count, ds_writes, 0u, nullptr);
    }

    auto blit_fbo = driver->addFrameBuffer();
    blit_fbo->attach(video::EFAP_COLOR_ATTACHMENT0, smart_refctd_ptr(out_image_view));

    while (device->run())
    {
        driver->beginScene(false, false);

        blur_direction = 0u;
        blur_params.input_dimensions.w = (blur_direction << 30) | (channel_count << 28);

        driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline_horizontal->getLayout(), 0u, 1u, &ds_horizontal.get(), nullptr);
        driver->bindComputePipeline(pipeline_horizontal.get());
        driver->pushConstants(pipeline_horizontal->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(Parameters_t), &blur_params);
        driver->dispatch(1, out_dim.Y, 1);
        
        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        blur_direction = 1u;
        blur_params.input_dimensions.w = (blur_direction << 30) | (channel_count << 28);

        driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline_vertical->getLayout(), 0u, 1u, &ds_vertical.get(), nullptr);
        driver->bindComputePipeline(pipeline_vertical.get());
        driver->pushConstants(pipeline_vertical->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(Parameters_t), &blur_params);
        driver->dispatch(out_dim.X, 1, 1);

        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        driver->blitRenderTargets(blit_fbo, nullptr, false, false);

        driver->endScene();
    }

    if (!ext::ScreenShot::createScreenShot(driver, am, out_image_view.get(), "../screenshot.png", asset::EF_R8G8B8_SRGB))
        std::cout << "Unable to create screenshot" << std::endl;

    return 0;
}
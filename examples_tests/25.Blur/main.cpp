// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/CentralLimitBoxBlur/CBlurPerformer.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

using BlurClass = ext::CentralLimitBoxBlur::CBlurPerformer;

class MouseEventReceiver : public IEventReceiver
{
    _NBL_STATIC_INLINE_CONSTEXPR float BLUR_RADIUS_MIN = 0.f, BLUR_RADIUS_MAX = 0.5f;

public:
    BlurClass::Parameters_t* pushConstants = nullptr;

    bool OnEvent(const SEvent& e)
    {
        if (e.EventType == nbl::EET_KEY_INPUT_EVENT && !e.KeyInput.PressedDown)
        {
            switch (e.KeyInput.Key)
            {
                case nbl::KEY_KEY_Q:
                    exit(0);

                default:
                    return false;
            }
        }

        if (e.EventType == nbl::EET_MOUSE_INPUT_EVENT && pushConstants)
        {
            for (uint32_t i = 0; i < 2u; ++i)
            {
                const float r = pushConstants[i].radius + e.MouseInput.Wheel / 500.f;
                pushConstants[i].radius = core::max(BLUR_RADIUS_MIN, core::min(r, BLUR_RADIUS_MAX));
            }
            return true;
        }

        return false;
    }
};

inline smart_refctd_ptr<IGPUSpecializedShader> createShader(const char* shader_include_path, const uint32_t axis_dim, const bool use_half_storage, IVideoDriver* driver)
{
    // Todo(achal): Like in FFT, `_NBL_GLSL_EXT_BLUR_AXIS_DIM_` should be `_NBL_GLSL_EXT_BLUR_MAX_AXIS_DIM_`
    // and you need to do virtual threads calculation in the shader.
    const char* sourceFmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_BLUR_PASS_COUNT_ %u
#define _NBL_GLSL_EXT_BLUR_AXIS_DIM_ %u
#define _NBL_GLSL_EXT_BLUR_HALF_STORAGE_ %u

layout (local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;
 
#include "%s"
)===";

    const size_t extraSize = 4u + 8u + 8u + 128u;

    auto shader = core::make_smart_refctd_ptr<asset::ICPUBuffer>(strlen(sourceFmt) + extraSize + 1u);
    snprintf(reinterpret_cast<char*>(shader->getPointer()), shader->getSize(), sourceFmt, BlurClass::DEFAULT_WORKGROUP_SIZE, BlurClass::PASSES_PER_AXIS,
        axis_dim, use_half_storage ? 1u : 0u, shader_include_path);

    auto cpu_specialized_shader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
        core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shader), asset::ICPUShader::buffer_contains_glsl),
        asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

    auto gpu_shader = driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpu_specialized_shader->getUnspecialized()));
    return driver->createGPUSpecializedShader(gpu_shader.get(), cpu_specialized_shader->getSpecializationInfo());
}

static void updateDescriptorSet_Horizontal(IVideoDriver* driver, IGPUDescriptorSet* set, core::smart_refctd_ptr<video::IGPUImageView> inputImageDescriptor,
    core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor)
{
    IGPUSampler::SParams params =
    {
        {
            // These wrapping params don't really matter for this example
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
        ds_writes[i].dstSet = set;
        ds_writes[i].arrayElement = 0u;
        ds_writes[i].count = 1u;
        ds_writes[i].info = ds_infos + i;
    }

    // Input sampler2D
    ds_infos[0].desc = inputImageDescriptor;
    ds_infos[0].image.sampler = sampler;
    ds_infos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

    ds_writes[0].binding = 0;
    ds_writes[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;

    // Output SSBO 
    ds_infos[1].desc = outputBufferDescriptor;
    ds_infos[1].buffer = { 0u, outputBufferDescriptor->getSize() };

    ds_writes[1].binding = 1;
    ds_writes[1].descriptorType = asset::EDT_STORAGE_BUFFER;

    driver->updateDescriptorSets(descriptor_count, ds_writes, 0u, nullptr);
}

static void updateDescriptorSet_Vertical(IVideoDriver* driver, IGPUDescriptorSet* set, core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
    core::smart_refctd_ptr<video::IGPUImageView> outputImageDescriptor)
{
    {
        constexpr uint32_t descriptor_count = 2u;
        IGPUDescriptorSet::SDescriptorInfo ds_infos[descriptor_count];
        IGPUDescriptorSet::SWriteDescriptorSet ds_writes[descriptor_count];

        for (uint32_t i = 0; i < descriptor_count; ++i)
        {
            ds_writes[i].dstSet = set;
            ds_writes[i].arrayElement = 0u;
            ds_writes[i].count = 1u;
            ds_writes[i].info = ds_infos + i;
        }

        // Input SSBO
        ds_infos[0].desc = inputBufferDescriptor;
        ds_infos[0].buffer = { 0u, inputBufferDescriptor->getSize() };

        ds_writes[0].binding = 0;
        ds_writes[0].descriptorType = asset::EDT_STORAGE_BUFFER;

        // Output image2D
        ds_writes[1].binding = 1;
        ds_writes[1].descriptorType = asset::EDT_STORAGE_IMAGE;

        ds_infos[1].desc = outputImageDescriptor;
        ds_infos[1].image.sampler = nullptr;
        ds_infos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

        driver->updateDescriptorSets(descriptor_count, ds_writes, 0u, nullptr);
    }
}

int main()
{
    nbl::SIrrlichtCreationParameters deviceParams;
    deviceParams.Bits = 24;
    deviceParams.ZBufferBits = 24;
    deviceParams.DriverType = video::EDT_OPENGL;
    deviceParams.WindowSize = dimension2d<uint32_t>(1024, 1024);
    deviceParams.Fullscreen = false;
    deviceParams.Vsync = true;
    deviceParams.Doublebuffer = true;
    deviceParams.Stencilbuffer = false;

    auto device = createDeviceEx(deviceParams);
    if (!device)
        return 1;

    MouseEventReceiver eventReceiver;
    device->setEventReceiver(&eventReceiver);

    video::IVideoDriver* driver = device->getVideoDriver();

    nbl::io::IFileSystem* filesystem = device->getFileSystem();
    asset::IAssetManager* am = device->getAssetManager();

    IAssetLoader::SAssetLoadParams lp;
    auto in_image_bundle = am->getAsset("../cube_face.jpg", lp);

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

    const asset::VkExtent3D blur_ds_factor = { 2u, 2u, 1u };
    auto in_dim = in_image_view->getCreationParameters().image->getCreationParameters().extent;

    VkExtent3D out_dim;
    for (uint32_t i = 0; i < 3u; ++i)
        (&out_dim.width)[i] = ((&in_dim.width)[i]) / ((&blur_ds_factor.width)[i]);

    // Create out image
    smart_refctd_ptr<IGPUImage> out_image;
    smart_refctd_ptr<IGPUImageView> out_image_view;

    IImage::SCreationParams out_image_info;
    {
        IGPUImageView::SCreationParams out_image_view_info = in_image_view->getCreationParameters();
        out_image_view_info.format = asset::EF_R16G16B16A16_SFLOAT;

        auto out_image_info = out_image_view_info.image->getCreationParameters();
        out_image_info.format = asset::EF_R16G16B16A16_SFLOAT;
        out_image_info.extent = out_dim;
        out_image_info.mipLevels = 1u;

        out_image = driver->createDeviceLocalGPUImageOnDedMem(std::move(out_image_info));
    
        out_image_view_info.image = out_image;
        out_image_view = driver->createGPUImageView(IGPUImageView::SCreationParams(out_image_view_info));
    }

    const bool use_half_storage = false;
    
    const uint32_t channelCount = getFormatChannelCount(in_image_view->getCreationParameters().format);
    auto scratchOutputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(BlurClass::getOutputBufferSize(out_dim, channelCount));

    core::SRange<const asset::SPushConstantRange> pcRange = BlurClass::getDefaultPushConstantRanges();

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

        auto pipeline_layout = driver->createGPUPipelineLayout(pcRange.begin(), pcRange.end(), std::move(ds_layout));
        pipeline_horizontal = driver->createGPUComputePipeline(nullptr, smart_refctd_ptr(pipeline_layout), createShader("../BlurPassHorizontal.comp", out_dim.width, use_half_storage, driver));
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

        auto pipeline_layout = driver->createGPUPipelineLayout(pcRange.begin(), pcRange.end(), std::move(ds_layout));
        pipeline_vertical = driver->createGPUComputePipeline(nullptr, smart_refctd_ptr(pipeline_layout), createShader("../BlurPassVertical.comp", out_dim.height, use_half_storage, driver));
    }

    const float blurRadius = 0.01f;

    BlurClass::Parameters_t pushConstants[2];
    BlurClass::DispatchInfo_t dispatchInfo[2];
    const ISampler::E_TEXTURE_CLAMP blurWrapMode[2] = { ISampler::ETC_REPEAT, ISampler::ETC_REPEAT };
    const uint32_t passCount = BlurClass::buildParameters(channelCount, out_dim, pushConstants, dispatchInfo, blurWrapMode, blurRadius);
    assert(passCount == 2u);

    eventReceiver.pushConstants = pushConstants;

    updateDescriptorSet_Horizontal(driver, ds_horizontal.get(), in_image_view, scratchOutputBuffer);
    updateDescriptorSet_Vertical(driver, ds_vertical.get(), scratchOutputBuffer, out_image_view);

    auto blit_fbo = driver->addFrameBuffer();
    blit_fbo->attach(video::EFAP_COLOR_ATTACHMENT0, smart_refctd_ptr(out_image_view));

    while (device->run())
    {
        driver->beginScene(false, false);

        driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline_horizontal->getLayout(), 0u, 1u, &ds_horizontal.get(), nullptr);
        driver->bindComputePipeline(pipeline_horizontal.get());
        BlurClass::dispatchHelper(driver, pipeline_horizontal->getLayout(), pushConstants[0], dispatchInfo[0]);

        driver->bindDescriptorSets(video::EPBP_COMPUTE, pipeline_vertical->getLayout(), 0u, 1u, &ds_vertical.get(), nullptr);
        driver->bindComputePipeline(pipeline_vertical.get());
        BlurClass::dispatchHelper(driver, pipeline_vertical->getLayout(), pushConstants[1], dispatchInfo[1], false);

        video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);

        driver->blitRenderTargets(blit_fbo, nullptr, false, false);

        driver->endScene();
    }

    if (!ext::ScreenShot::createScreenShot(driver, am, out_image_view.get(), "../screenshot.png", asset::EF_R8G8B8_SRGB))
        std::cout << "Unable to create screenshot" << std::endl;

    return 0;
}
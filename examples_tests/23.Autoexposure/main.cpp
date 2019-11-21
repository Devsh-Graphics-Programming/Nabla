#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>


//#include "../../ext/ScreenShot/ScreenShot.h"
#include "../source/Irrlicht/COpenGLDriver.h"
#include "../ext/AutoExposure/CToneMapper.h"

#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;


core::smart_refctd_ptr<video::IGPUImage> createImageForLackOfEXRLoader(asset::IAssetManager* am, video::IVideoDriver* driver);

int main()
{
	irr::SIrrlichtCreationParameters deviceParams;
	deviceParams.Bits = 24; //may have to set to 32bit for some platforms
	deviceParams.ZBufferBits = 24; //we'd like 32bit here
	deviceParams.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	deviceParams.WindowSize = dimension2d<uint32_t>(1280, 720);
	deviceParams.Fullscreen = false;
	deviceParams.Vsync = true; //! If supported by target platform
	deviceParams.Doublebuffer = true;
	deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

	IrrlichtDevice* device = createDeviceEx(deviceParams);
	if (device == 0)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();
	
	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();


	core::smart_refctd_ptr<video::IGPUImage> inImg;
	{
		inImg = createImageForLackOfEXRLoader(am,driver);

	}

	core::smart_refctd_ptr<video::IGPUImageView> imgToTonemap;
	core::smart_refctd_ptr<video::IGPUImageView> outImg,outImgStorage;
	{
		video::IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = core::smart_refctd_ptr(inImg); // copy smart ptr, still want to use it
		imgViewInfo.viewType = video::IGPUImageView::ET_2D;
		imgViewInfo.format = asset::EF_R32G32_UINT;
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		imgToTonemap = driver->createGPUImageView(std::move(imgViewInfo));

		video::IGPUImage::SCreationParams imgInfo = inImg->getCreationParameters();
		imgInfo.format = asset::EF_R8G8B8A8_SRGB;
		imgViewInfo.image = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));
		imgViewInfo.format = asset::EF_R8G8B8A8_SRGB;
		outImg = driver->createGPUImageView(std::move(imgViewInfo));
		imgViewInfo.format = asset::EF_R32_UINT;
		outImgStorage = driver->createGPUImageView(std::move(imgViewInfo));
	}


	auto tonemapper = ext::AutoExposure::CToneMapper::create(driver,inImg->getCreationParameters().format,am->getGLSLCompiler());

	// TODO: employ luma histograming
	auto params = ext::AutoExposure::ReinhardParams::fromKeyAndBurn(0.18, 0.95, 8.0, 16.0);
	auto parameterBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(ext::AutoExposure::ReinhardParams),&params);

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(tonemapper->getDescriptorSetLayout()));
	{
		video::IGPUDescriptorSet::SDescriptorInfo pInfos[3];
		pInfos[0].desc = parameterBuffer;
		pInfos[0].buffer.offset = 0u;
		pInfos[0].buffer.size = video::IGPUBufferView::whole_buffer;
		pInfos[1].desc = imgToTonemap;
		pInfos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
		using S = asset::ISampler;
		pInfos[1].image.sampler = driver->createGPUSampler({   {S::ETC_CLAMP_TO_EDGE,S::ETC_CLAMP_TO_EDGE,S::ETC_CLAMP_TO_EDGE,
																S::ETBC_FLOAT_OPAQUE_BLACK,
																S::ETF_NEAREST,S::ETF_NEAREST,S::ESMM_NEAREST,0u,
																false,asset::ECO_ALWAYS},
															0.f,-FLT_MAX,FLT_MAX});
		pInfos[2].desc = outImgStorage;
		pInfos[2].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
		pInfos[2].image.sampler = nullptr;

		video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[3];
		pWrites[0].dstSet = descriptorSet.get();
		pWrites[0].binding = 0u;
		pWrites[0].arrayElement = 0u;
		pWrites[0].count = 1u;
		pWrites[0].descriptorType = asset::EDT_UNIFORM_BUFFER_DYNAMIC;
		pWrites[0].info = pInfos+0u;
		pWrites[1].dstSet = descriptorSet.get();
		pWrites[1].binding = 1u;
		pWrites[1].arrayElement = 0u;
		pWrites[1].count = 1u;
		pWrites[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		pWrites[1].info = pInfos+1u;
		pWrites[2].dstSet = descriptorSet.get();
		pWrites[2].binding = 2u;
		pWrites[2].arrayElement = 0u;
		pWrites[2].count = 1u;
		pWrites[2].descriptorType = asset::EDT_STORAGE_IMAGE;
		pWrites[2].info = pInfos+2u;
		driver->updateDescriptorSets(3u,pWrites,0u,nullptr);
	}

	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImg));

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		tonemapper->tonemap(imgToTonemap.get(), descriptorSet.get(), 0u);

		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}
	// TODO: Histogramming
#if 0
        toneMapper->CalculateFrameExposureFactors(frameUniformBuffer,frameUniformBuffer,core::smart_refctd_ptr(hdrTex));


	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}
#endif
	device->drop();

	return 0;
}


core::smart_refctd_ptr<video::IGPUImage> createImageForLackOfEXRLoader(asset::IAssetManager* am, video::IVideoDriver* driver)
{
	auto viewportSize = driver->getScreenSize();

	video::IGPUImage::SCreationParams imgInfo;
	imgInfo.flags = static_cast<asset::ICPUImage::E_CREATE_FLAGS>(0u);
	imgInfo.type = video::IGPUImage::ET_2D;
	imgInfo.format = asset::EF_R16G16B16A16_SFLOAT;
	imgInfo.extent = {viewportSize.Width,viewportSize.Height,1};
	imgInfo.mipLevels = 1u;
	imgInfo.arrayLayers = 1u;
	imgInfo.samples = video::IGPUImage::ESCF_1_BIT;

	auto img = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));
	{
		video::IGPUDescriptorSetLayout::SBinding bnd;
		bnd.binding = 0u;
		bnd.type = asset::EDT_STORAGE_IMAGE;
		bnd.count = 1u;
		bnd.stageFlags = asset::ESS_COMPUTE;
		auto dsLayout = driver->createGPUDescriptorSetLayout(&bnd, &bnd+1);

		auto layout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(dsLayout));

		core::smart_refctd_ptr<video::IGPUSpecializedShader> shader;
		{
			const char* glsl = R"===(
#version 430 core
layout (local_size_x = 16, local_size_y = 16) in;

layout(set=3, binding=0, rgba16f) uniform writeonly restrict image2D outImg;

// The MIT License
// Copyright © 2015 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// A simple way to create color variation in a cheap way (yes, trigonometrics ARE cheap
// in the GPU, don't try to be smart and use a triangle wave instead).

// See http://iquilezles.org/www/articles/palettes/palettes.htm for more information


vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.28318*(c*t+d) );
}

void main()
{
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	ivec2 resolution = imageSize(outImg);
	if (any(greaterThanEqual(uv,resolution)))
		return;

	vec2 p = vec2(uv)/vec2(resolution);
    
    // compute colors
    vec3                col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.33,0.67) );
    if( p.y>(1.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.10,0.20) );
    if( p.y>(2.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.3,0.20,0.20) );
    if( p.y>(3.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,0.5),vec3(0.8,0.90,0.30) );
    if( p.y>(4.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,0.7,0.4),vec3(0.0,0.15,0.20) );
    if( p.y>(5.0/7.0) ) col = pal( p.x, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(2.0,1.0,0.0),vec3(0.5,0.20,0.25) );
    if( p.y>(6.0/7.0) ) col = pal( p.x, vec3(0.8,0.5,0.4),vec3(0.2,0.4,0.2),vec3(2.0,1.0,1.0),vec3(0.0,0.25,0.25) );
    

    // band
    float f = fract(p.y*7.0);
    // borders
    col *= smoothstep( 0.49, 0.47, abs(f-0.5) );
    // shadowing
    col *= 0.5 + 0.5*sqrt(4.0*f*(1.0-f));

	col *= 16.0;

	imageStore(outImg,uv,vec4(col,1.0));
}
)===";

			auto spirv = am->getGLSLCompiler()->createSPIRVFromGLSL(glsl, asset::ESS_COMPUTE, "main", "gradient");
			auto cs_unspec = driver->createGPUShader(std::move(spirv));

			auto specInfo = core::make_smart_refctd_ptr<asset::ISpecializationInfo>(core::vector<asset::SSpecializationMapEntry>{}, nullptr, "main", asset::ESS_COMPUTE);

			shader = driver->createGPUSpecializedShader(cs_unspec.get(),specInfo.get());
		}

		auto pipeline = driver->createGPUComputePipeline(nullptr, std::move(layout), std::move(shader));

		auto descriptorSet = driver->createGPUDescriptorSet(std::move(dsLayout));
		{
			video::IGPUDescriptorSet::SDescriptorInfo info;
			info.desc = driver->createGPUImageView({static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u),
													core::smart_refctd_ptr(img),
													video::IGPUImageView::ET_2D,
													asset::EF_R16G16B16A16_SFLOAT,
													{},
													{static_cast<asset::IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u}});
			info.image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);
			info.image.sampler = nullptr;

			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = descriptorSet.get();
			write.binding = 0u;
			write.arrayElement = 0u;
			write.count = 1u;
			write.descriptorType = asset::EDT_STORAGE_IMAGE;
			write.info = &info;
			driver->updateDescriptorSets(1u, &write, 0u, nullptr); 
		}

		driver->bindComputePipeline(pipeline.get());
		driver->bindDescriptorSets(video::EPBP_COMPUTE,pipeline->getLayout(),3u,1u,&descriptorSet.get(),nullptr);
		driver->dispatch(imgInfo.extent.width/16u,imgInfo.extent.height/16u,1u);
	}

	return img;
}
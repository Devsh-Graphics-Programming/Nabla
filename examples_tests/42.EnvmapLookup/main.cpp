#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"

#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../../ext/ScreenShot/ScreenShot.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

irr::video::IFrameBuffer* createHDRFramebuffer(core::smart_refctd_ptr<IrrlichtDevice> device, asset::E_FORMAT colorFormat)
{
	auto driver = device->getVideoDriver();

	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = driver->getScreenSize().Width;
		imgInfo.extent.height = driver->getScreenSize().Height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

		auto image = driver->createGPUImageOnDedMem(std::move(imgInfo),driver->getDeviceLocalGPUMemoryReqs());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = driver->createGPUImageView(std::move(imgViewInfo));
	}

	auto frameBuffer = driver->addFrameBuffer();
	frameBuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(gpuImageViewColorBuffer));

	return frameBuffer;
}


void APIENTRY openGLCBFunc(GLenum source, GLenum type, GLuint id, GLenum severity,
                           GLsizei length, const GLchar* message, const void* userParam)
{
    core::stringc outStr;
    switch (severity)
    {
        //case GL_DEBUG_SEVERITY_HIGH:
        case GL_DEBUG_SEVERITY_HIGH_ARB:
            outStr = "[H.I.G.H]";
            break;
        //case GL_DEBUG_SEVERITY_MEDIUM:
        case GL_DEBUG_SEVERITY_MEDIUM_ARB:
            outStr = "[MEDIUM]";
            break;
        //case GL_DEBUG_SEVERITY_LOW:
        case GL_DEBUG_SEVERITY_LOW_ARB:
            outStr = "[  LOW  ]";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            outStr = "[  LOW  ]";
            break;
        default:
            outStr = "[UNKNOWN]";
            break;
    }
    switch (source)
    {
        //case GL_DEBUG_SOURCE_API:
        case GL_DEBUG_SOURCE_API_ARB:
            switch (type)
            {
                //case GL_DEBUG_TYPE_ERROR:
                case GL_DEBUG_TYPE_ERROR_ARB:
                    outStr += "[OPENGL  API ERROR]\t\t";
                    break;
                //case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
                    outStr += "[OPENGL  DEPRECATED]\t\t";
                    break;
                //case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
                    outStr += "[OPENGL   UNDEFINED]\t\t";
                    break;
                //case GL_DEBUG_TYPE_PORTABILITY:
                case GL_DEBUG_TYPE_PORTABILITY_ARB:
                    outStr += "[OPENGL PORTABILITY]\t\t";
                    break;
                //case GL_DEBUG_TYPE_PERFORMANCE:
                case GL_DEBUG_TYPE_PERFORMANCE_ARB:
                    outStr += "[OPENGL PERFORMANCE]\t\t";
                    break;
                default:
                    outStr += "[OPENGL       OTHER]\t\t";
                    //! special sauce
                    //return;
                    break;
            }
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_SHADER_COMPILER:
        case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
            outStr += "[SHADER]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
            outStr += "[WINDOW SYS]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_THIRD_PARTY:
        case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
            outStr += "[3RDPARTY]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_APPLICATION:
        case GL_DEBUG_SOURCE_APPLICATION_ARB:
            outStr += "[APP]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_OTHER:
        case GL_DEBUG_SOURCE_OTHER_ARB:
            outStr += "[OTHER]\t\t";
            outStr += message;
            break;
        default:
            break;
    }
    outStr += "\n";
    printf("%s",outStr.c_str());
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 3; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;

int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 32;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(1920, 1080);
	params.Fullscreen = false;
	params.Doublebuffer = true;
	params.Vsync = true;
	params.Stencilbuffer = false;

	auto device = createDeviceEx(params);
	if (!device)
		return false;

	device->getCursorControl()->setVisible(false);
	auto driver = device->getVideoDriver();
#ifdef _DEBUG
	if (video::COpenGLExtensionHandler::FeatureAvailable[video::COpenGLExtensionHandler::IRR_KHR_debug])
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		video::COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, true);

		video::COpenGLExtensionHandler::pGlDebugMessageCallback(openGLCBFunc, NULL);
	}
	else
	{
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
		video::COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, true);

		video::COpenGLExtensionHandler::pGlDebugMessageCallbackARB(openGLCBFunc, NULL);
	}
#endif
	auto assetManager = device->getAssetManager();
	auto sceneManager = device->getSceneManager();
	auto geometryCreator = device->getAssetManager()->getGeometryCreator();

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
	camera->setLeftHanded(false);

	camera->setPosition(core::vector3df(0, 2, 3));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.03125f);
	camera->setFarValue(200.0f);
	camera->setFOV(core::radians(60.f));

	sceneManager->setActiveCamera(camera);

	IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
		{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr },
		{ 1u, EDT_UNIFORM_TEXEL_BUFFER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr },
		{ 2u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr }
	};
	IGPUDescriptorSetLayout::SBinding uboBinding {0, asset::EDT_UNIFORM_BUFFER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr};

	auto gpuDescriptorSetLayout1 = driver->createGPUDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
	auto gpuDescriptorSetLayout3 = driver->createGPUDescriptorSetLayout(descriptorSet3Bindings, descriptorSet3Bindings+3u);

	auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(device->getAssetManager(), device->getVideoDriver());

	auto createGpuResources = [&](std::string pathToShadersWithoutExtension) -> std::pair<core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>, core::smart_refctd_ptr<video::IGPUMeshBuffer>>
	{
		auto cpuFragmentSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(assetManager->getAsset(pathToShadersWithoutExtension + ".frag", {}).getContents().begin()[0]);
		ISpecializedShader::SInfo info = cpuFragmentSpecializedShader->getSpecializationInfo();
		info.m_backingBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ShaderParameters));
		memcpy(info.m_backingBuffer->getPointer(),&kShaderParameters,sizeof(ShaderParameters));
		info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
		for (uint32_t i=0; i<2; i++)
			info.m_entries->operator[](i) = {i,i*sizeof(uint32_t),sizeof(uint32_t)};
		cpuFragmentSpecializedShader->setSpecializationInfo(std::move(info));

		auto gpuFragmentSpecialedShader = driver->getGPUObjectsFromAssets(&cpuFragmentSpecializedShader.get(), &cpuFragmentSpecializedShader.get() + 1)->front();
		IGPUSpecializedShader* shaders[2] = { std::get<0>(fullScreenTriangle).get(), gpuFragmentSpecialedShader.get() };

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		asset::SBlendParams blendParams;
		SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = EFCM_NONE;
		rasterParams.depthCompareOp = ECO_ALWAYS;
		rasterParams.minSampleShading = 1.f;
		rasterParams.depthWriteEnable = false;
		rasterParams.depthTestEnable = false;


		auto gpuPipeline = driver->createGPURenderpassIndependentPipeline(
			nullptr, std::move(gpuPipelineLayout),
			shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(fullScreenTriangle), blendParams, std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);

		SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
		core::smart_refctd_ptr<video::IGPUMeshBuffer> gpuMeshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, nullptr, std::move(idxBinding));
		{
			gpuMeshBuffer->setIndexCount(3u);
		}

		return { gpuPipeline, gpuMeshBuffer };
	};

	auto gpuEnvmapResources = createGpuResources("../envCubeMapShaders/envmap");
	auto gpuEnvmapPipeline = gpuEnvmapResources.first;
	auto gpuEnvmapMeshBuffer = gpuEnvmapResources.second;

	auto createGPUImageView = [&](std::string pathToOpenEXRHDRIImage)
	{
		auto pathToTexture = pathToOpenEXRHDRIImage;
		IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
		auto cpuTextureContents = cpuTexture.getContents();

		auto asset = *cpuTextureContents.begin();

		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.viewType = IImageView<ICPUImage>::ET_2D;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;

		auto cpuImageView = ICPUImageView::create(std::move(viewParams));
		auto gpuImageView = driver->getGPUObjectsFromAssets(&cpuImageView.get(), &cpuImageView.get() + 1u)->front();

		return gpuImageView;
	};

	auto gpuEnvmapImageView = createGPUImageView("../../media/envmap/envmap_0.exr");

	smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView;
	{
		const uint32_t MaxDimensions = 3u<<kShaderParameters.MaxDepthLog2;
		const uint32_t MaxSamples = 1u<<kShaderParameters.MaxSamplesLog2;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*MaxDimensions*MaxSamples);
		
		core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
		//core::SobolSampler sampler(MaxDimensions);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (auto dim=0u; dim<MaxDimensions; dim++)
		for (uint32_t i=0; i<MaxSamples; i++)
		{
			out[i*MaxDimensions+dim] = sampler.sample(dim,i);
		}
		auto gpuSequenceBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
		gpuSequenceBufferView = driver->createGPUBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
	{
		IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = IImage::ET_2D;
		imgParams.format = EF_R32_UINT;
		imgParams.extent = {params.WindowSize.Width,params.WindowSize.Height,1u};
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = IImage::ESCF_1_BIT;

		IGPUImage::SBufferCopy region;
		region.imageExtent = imgParams.extent;
		region.imageSubresource.layerCount = 1u;

		const auto renderPixelCount = imgParams.extent.width * imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}
		auto buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(renderPixelCount*sizeof(uint32_t),random.data());

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = driver->createFilledDeviceLocalGPUImageOnDedMem(std::move(imgParams),buffer.get(),1u,&region);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = EF_R32_UINT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = driver->createGPUImageView(std::move(viewParams));
	}

	auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));
	auto uboDescriptorSet1 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
		uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
		uboWriteDescriptorSet.binding = 0;
		uboWriteDescriptorSet.count = 1u;
		uboWriteDescriptorSet.arrayElement = 0u;
		uboWriteDescriptorSet.descriptorType = asset::EDT_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuubo;
			info.buffer.offset = 0ull;
			info.buffer.size = sizeof(SBasicViewParameters);
		}
		uboWriteDescriptorSet.info = &info;
		driver->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	auto descriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));
	{
		constexpr auto kDescriptorCount = 3;
		IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet[kDescriptorCount];
		IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo[kDescriptorCount];
		for (auto i=0; i<kDescriptorCount; i++)
		{
			samplerWriteDescriptorSet[i].dstSet = descriptorSet3.get();
			samplerWriteDescriptorSet[i].binding = i;
			samplerWriteDescriptorSet[i].arrayElement = 0u;
			samplerWriteDescriptorSet[i].count = 1u;
			samplerWriteDescriptorSet[i].info = samplerDescriptorInfo+i;
		}
		samplerWriteDescriptorSet[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		samplerWriteDescriptorSet[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
		samplerWriteDescriptorSet[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;

		samplerDescriptorInfo[0].desc = gpuEnvmapImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[0].image.sampler = driver->createGPUSampler(samplerParams);
			samplerDescriptorInfo[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		samplerDescriptorInfo[1].desc = gpuSequenceBufferView;
		samplerDescriptorInfo[2].desc = gpuScrambleImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[2].image.sampler = driver->createGPUSampler(samplerParams);
			samplerDescriptorInfo[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		driver->updateDescriptorSets(kDescriptorCount, samplerWriteDescriptorSet, 0u, nullptr);
	}

	auto HDRFramebuffer = createHDRFramebuffer(device, asset::EF_R16G16B16A16_SFLOAT);
	float colorClearValues[] = { 1.f, 1.f, 1.f, 1.f };

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	if (device->isWindowFocused())
	{
		driver->setRenderTarget(HDRFramebuffer, false);
		driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, colorClearValues);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewMatrix = camera->getViewMatrix();
		const auto viewProjectionMatrix = camera->getConcatenatedMatrix();

		// cube envmap handle
		{
			auto mv = viewMatrix;
			auto mvp = viewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParameters uboData;
			memcpy(uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

			driver->bindGraphicsPipeline(gpuEnvmapPipeline.get());
			driver->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr);
			driver->bindDescriptorSets(EPBP_GRAPHICS, gpuEnvmapPipeline->getLayout(), 3u, 1u, &descriptorSet3.get(), nullptr);
			driver->drawMeshBuffer(gpuEnvmapMeshBuffer.get());
		}
		// TODO: tone mapping and stuff

		driver->setRenderTarget(nullptr, false);
		driver->blitRenderTargets(HDRFramebuffer, nullptr, false, false);

		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Envmap Example - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	// TODO: screenshot
	ext::ScreenShot::createScreenShot(device, HDRFramebuffer->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenshot.exr");
}

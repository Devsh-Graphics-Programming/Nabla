// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"
#include "irr/asset/CGeometryCreator.h"

/*
	General namespaces. Entire engine consists of those bellow.
*/

using namespace irr;
using namespace asset;
using namespace video;
using namespace core;


int main()
{
	/*
		 SIrrlichtCreationParameters holds some specific initialization information 
		 about driver being used, size of window, stencil buffer or depth buffer.
		 Used to create a device.
	*/

	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; 
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; 
	auto device = createDeviceEx(params);

	if (!device)
		return 0;

	device->getCursorControl()->setVisible(false);

	/*
		One of event receiver. Used to handle closing aplication event.
	*/

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	/*
		Most important objects to manage literally whole stuff are bellow. 
		By their usage you can create for example GPU objects, load or write
		assets or manage objects on a scene.
	*/

	auto driver = device->getVideoDriver();
	auto assetManager = device->getAssetManager();
	auto sceneManager = device->getSceneManager();

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(-5, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(1000.0f);

	sceneManager->setActiveCamera(camera);

	/*
		Helpfull class for managing basic geometry objects. 
		Thanks to it you can get half filled pipeline for your
		geometries such as cubes, cones or spheres.
	*/

	auto geometryCreator = device->getAssetManager()->getGeometryCreator();
	auto rectangleGeometry = geometryCreator->createRectangleMesh(irr::core::vector2df_SIMD(1.5, 3));

	/*
		Loading an asset bundle. You can specify some flags 
		and parameters to have an impact on extraordinary 
		tasks while loading for example. 
	*/

	asset::IAssetLoader::SAssetLoadParams loadingParams;
	auto images_bundle = assetManager->getAsset("../../media/color_space_test/R8G8B8A8_1.png", loadingParams);
	assert(!images_bundle.isEmpty());
	auto image = images_bundle.getContents().begin()[0];
	auto image_raw = static_cast<asset::ICPUImage*>(image.get());

	/*
		Specifing gpu image view parameters to create a gpu
		image view through the driver.
	*/

	auto gpuImage = driver->getGPUObjectsFromAssets(&image_raw, &image_raw + 1)->front();
	auto& gpuParams = gpuImage->getCreationParameters();

	IImageView<IGPUImage>::SCreationParams gpuImageViewParams = { static_cast<IGPUImageView::E_CREATE_FLAGS>(0), gpuImage, IImageView<IGPUImage>::ET_2D, gpuParams.format, {}, {static_cast<IImage::E_ASPECT_FLAGS>(0u), 0, gpuParams.mipLevels, 0, gpuParams.arrayLayers} };
	auto gpuImageView = driver->createGPUImageView(std::move(gpuImageViewParams));

	/*
		Specifying cache key to default exsisting cached asset bundle
		and specifying it's size where end is determined by 
		static_cast<IAsset::E_TYPE>(0u)
	*/

	const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };

	auto cpuVertexShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assetManager->findAssets("irr/builtin/materials/lambertian/singletexture/specializedshader.vert", types)->front().getContents().begin()[0]);
	auto cpuFragmentShader = core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(assetManager->findAssets("irr/builtin/materials/lambertian/singletexture/specializedshader.frag", types)->front().getContents().begin()[0]);

	auto gpuVertexShader = driver->getGPUObjectsFromAssets(&cpuVertexShader.get(), &cpuVertexShader.get() + 1)->front();
	auto gpuFragmentShader = driver->getGPUObjectsFromAssets(&cpuFragmentShader.get(), &cpuFragmentShader.get() + 1)->front();
	std::array<IGPUSpecializedShader*, 2> gpuShaders = { gpuVertexShader.get(), gpuFragmentShader.get() };

	/*
		Creating helpull variables for descriptor sets.
		We are using to descriptor sets, one for the texture
		(sampler) and one for UBO holding basic view parameters.
		Each uses 0 as index of binding.
	*/

	size_t ds0SamplerBinding = 0, ds1UboBinding = 0;
	auto createAndGetUsefullData = [&](asset::IGeometryCreator::return_type& geometryObject)
	{
		/*
			SBinding for the texture (sampler). 
		*/

		IGPUDescriptorSetLayout::SBinding gpuSamplerBinding;
		gpuSamplerBinding.binding = ds0SamplerBinding;
		gpuSamplerBinding.type = EDT_COMBINED_IMAGE_SAMPLER;
		gpuSamplerBinding.count = 1u;
		gpuSamplerBinding.stageFlags = static_cast<IGPUSpecializedShader::E_SHADER_STAGE>(IGPUSpecializedShader::ESS_FRAGMENT);
		gpuSamplerBinding.samplers = nullptr;	

		/*
			SBinding for UBO - basic view parameters.
		*/

		IGPUDescriptorSetLayout::SBinding gpuUboBinding;
		gpuUboBinding.count = 1u;
		gpuUboBinding.binding = ds1UboBinding;
		gpuUboBinding.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
		gpuUboBinding.type = asset::EDT_UNIFORM_BUFFER;

		/*
			Creating specific descriptor set layouts from specialized bindings.
			Those layouts needs to attached to pipeline layout if required by user.
			IrrlichtBaW provides 4 places for descriptor set layout usage.
		*/

		auto gpuDs1Layout = driver->createGPUDescriptorSetLayout(&gpuUboBinding, &gpuUboBinding + 1);
		auto gpuDs3Layout = driver->createGPUDescriptorSetLayout(&gpuSamplerBinding, &gpuSamplerBinding + 1);

		/*
			Creating gpu UBO with appropiate size.

			We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
		*/

		auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

		/*
			Creating descriptor sets - texture (sampler) and basic view parameters (UBO).
			Specifying info and write parameters for updating certain descriptor set to the driver.

			We know ahead of time that `SBasicViewParameters` struct is the expected structure of the only UBO block in the descriptor set nr. 1 of the shader.
		*/

		auto gpuDescriptorSet3 = driver->createGPUDescriptorSet(gpuDs3Layout);
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = gpuDescriptorSet3.get();
			write.binding = ds0SamplerBinding;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
			IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = std::move(gpuImageView);
				ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETC_CLAMP_TO_EDGE,ISampler::ETBC_FLOAT_OPAQUE_BLACK,ISampler::ETF_LINEAR,ISampler::ETF_LINEAR,ISampler::ESMM_LINEAR,0u,false,ECO_ALWAYS };
				info.image = { driver->createGPUSampler(samplerParams),EIL_SHADER_READ_ONLY_OPTIMAL };
			}
			write.info = &info;
			driver->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		auto gpuDescriptorSet1 = driver->createGPUDescriptorSet(gpuDs1Layout);
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = gpuDescriptorSet1.get();
			write.binding = ds1UboBinding;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_UNIFORM_BUFFER;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuubo;
				info.buffer.offset = 0ull;
				info.buffer.size = sizeof(SBasicViewParameters);
			}
			write.info = &info;
			driver->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, std::move(gpuDs1Layout), nullptr, std::move(gpuDs3Layout));

		/*
			Preparing required pipeline parameters and filling choosen one.
			Note that some of them are returned from geometry creator according 
			to what I mentioned in returning half pipeline parameters.
		*/

		asset::SBlendParams blendParams;
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		/*
			Creating gpu pipeline with it's pipeline layout and specilized parameters.
			Attaching vertex shader and fragment shaders.
		*/

		auto gpuPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), gpuShaders.data(), gpuShaders.data() + gpuShaders.size(), geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

		/*
			Creating gpu meshbuffer from parameters fetched from geometry creator return value.
		*/

		constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
		constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT + 1;
		core::vector<asset::ICPUBuffer*> cpubuffers;
		cpubuffers.reserve(MAX_DATA_BUFFERS);
		for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			auto buf = geometryObject.bindings[i].buffer.get();
			if (buf)
				cpubuffers.push_back(buf);
		}
		auto cpuindexbuffer = geometryObject.indexBuffer.buffer.get();
		if (cpuindexbuffer)
			cpubuffers.push_back(cpuindexbuffer);

		auto gpubuffers = driver->getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data() + cpubuffers.size());

		asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
		for (auto i = 0, j = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
		{
			if (!geometryObject.bindings[i].buffer)
				continue;
			auto buffPair = gpubuffers->operator[](j++);
			bindings[i].offset = buffPair->getOffset();
			bindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}
		if (cpuindexbuffer)
		{
			auto buffPair = gpubuffers->back();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
			bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
		}

		auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(gpuPipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
		{
			mb->setIndexType(geometryObject.indexType);
			mb->setIndexCount(geometryObject.indexCount);
			mb->setBoundingBox(geometryObject.bbox);
		}

		return std::make_tuple(mb, gpuPipeline, gpuubo, gpuDescriptorSet1, gpuDescriptorSet3);
	};

	auto gpuRectangle = createAndGetUsefullData(rectangleGeometry);
	auto gpuMeshBuffer = std::get<0>(gpuRectangle);
	auto gpuPipeline = std::get<1>(gpuRectangle);
	auto gpuubo = std::get<2>(gpuRectangle);
	auto gpuDescriptorSet1 = std::get<3>(gpuRectangle);
	auto gpuDescriptorSet3 = std::get<4>(gpuRectangle);

	/*
		Hot loop for rendering a scene.
	*/

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewProjection = camera->getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setRotation(irr::core::quaternion(0, 1, 0));

		auto mv = core::concatenateBFollowedByA(camera->getViewMatrix(), modelMatrix);
		auto mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		/*
			Updating UBO for basic view parameters and sending 
			updated data to staging buffer that will redirect
			the data to graphics card - to vertex shader.
		*/

		SBasicViewParameters uboData;
		memcpy(uboData.MV,mv.pointer(),sizeof(mv));
		memcpy(uboData.MVP,mvp.pointer(),sizeof(mvp));
		memcpy(uboData.NormalMat,normalMat.pointer(),sizeof(normalMat));
		driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

		/*
			Binding the most important objects needed to
			render anything on the screen with textures:

			- gpu pipeline
			- gpu descriptor sets
		*/

		driver->bindGraphicsPipeline(gpuPipeline.get());
		driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), nullptr);
		driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 3u, 1u, &gpuDescriptorSet3.get(), nullptr);

		/*
			Drawing a mesh (created rectangle) with it's gpu mesh buffer usage.
		*/

		driver->drawMeshBuffer(gpuMeshBuffer.get());
		
		driver->endScene();
	}
}

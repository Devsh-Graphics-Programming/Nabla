// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/QToQuitEventReceiver.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;

/*
	Define below to write assets
*/

// #define WRITE_ASSETS

int main()
{
	nbl::SIrrlichtCreationParameters params;
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
		return 1; 

	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	auto* driver = device->getVideoDriver();
	auto* smgr = device->getSceneManager();

    auto loadAndGetCpuMesh = [&](std::string path) -> std::pair<core::smart_refctd_ptr<asset::ICPUMesh>, const asset::IAssetMetadata*>
    {
        auto* am = device->getAssetManager();
        auto* fs = am->getFileSystem();

        asset::IAssetLoader::SAssetLoadParams lp;

        auto meshes_bundle = am->getAsset(path, lp);

        assert(!meshes_bundle.getContents().empty());

        return std::make_pair(core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(meshes_bundle.getContents().begin()[0]), meshes_bundle.getMetadata());
    };

	auto& cpuBundlePLYData = loadAndGetCpuMesh("../../media/ply/Industrial_compressor.ply");
	auto& cpuBundleSTLData = loadAndGetCpuMesh("../../media/extrusionLogo_TEST_fixed.stl");

    core::smart_refctd_ptr<asset::ICPUMesh> cpuMeshPly = cpuBundlePLYData.first;
	auto metadataPly = cpuBundlePLYData.second->selfCast<const asset::CPLYMetadata>();

	core::smart_refctd_ptr<asset::ICPUMesh> cpuMeshStl = cpuBundleSTLData.first;
	auto metadataStl = cpuBundleSTLData.second->selfCast<const asset::CSTLMetadata>();

	#ifdef WRITE_ASSETS
	{
		asset::IAssetWriter::SAssetWriteParams wp(cpuMeshStl.get());
		device->getAssetManager()->writeAsset("extrusionLogo_TEST_fixedTest.stl", wp);
	}

	{
		asset::IAssetWriter::SAssetWriteParams wp(cpuMeshPly.get());
		device->getAssetManager()->writeAsset("IndustrialWriteTest.ply", wp);
	}
	#endif // WRITE_ASSETS
	
	/*
		For the testing puposes we can safely assume all meshbuffers within mesh loaded from PLY & STL has same DS1 layout (used for camera-specific data)
	*/

	using DependentDrawData = std::tuple<core::smart_refctd_ptr<video::IGPUMesh>, core::smart_refctd_ptr<video::IGPUBuffer>, core::smart_refctd_ptr<video::IGPUDescriptorSet>, uint32_t, const asset::IRenderpassIndependentPipelineMetadata*>;

	auto getMeshDependentDrawData = [&](core::smart_refctd_ptr<asset::ICPUMesh> cpuMesh, bool isPLY) -> DependentDrawData
	{
		const asset::ICPUMeshBuffer* const firstMeshBuffer = cpuMesh->getMeshBuffers().begin()[0];
		const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u); //! DS1
		const asset::IRenderpassIndependentPipelineMetadata* pipelineMetadata;
		{
			if(isPLY)
				pipelineMetadata = metadataPly->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());
			else
				pipelineMetadata = metadataStl->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());
		}

		/*
			So we can create just one DescriptorSet
		*/

		auto getDS1UboBinding = [&]()
		{
			uint32_t ds1UboBinding = 0u;
			for (const auto& bnd : ds1layout->getBindings())
				if (bnd.type == asset::EDT_UNIFORM_BUFFER)
				{
					ds1UboBinding = bnd.binding;
					break;
				}
			return ds1UboBinding;
		};

		const uint32_t ds1UboBinding = getDS1UboBinding();

		auto getNeededDS1UboByteSize = [&]()
		{
			size_t neededDS1UboSize = 0ull;
			{
				for (const auto& shaderInputs : pipelineMetadata->m_inputSemantics)
					if (shaderInputs.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shaderInputs.descriptorSection.uniformBufferObject.set == 1u && shaderInputs.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
						neededDS1UboSize = std::max<size_t>(neededDS1UboSize, shaderInputs.descriptorSection.uniformBufferObject.relByteoffset + shaderInputs.descriptorSection.uniformBufferObject.bytesize);
			}
			return neededDS1UboSize;
		};

		const uint64_t uboDS1ByteSize = getNeededDS1UboByteSize();

		auto gpuds1layout = driver->getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1)->front();

		auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(uboDS1ByteSize);
		auto gpuds1 = driver->createGPUDescriptorSet(std::move(gpuds1layout));
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = gpuds1.get();
			write.binding = ds1UboBinding;
			write.count = 1u;
			write.arrayElement = 0u;
			write.descriptorType = asset::EDT_UNIFORM_BUFFER;
			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuubo;
				info.buffer.offset = 0ull;
				info.buffer.size = uboDS1ByteSize;
			}
			write.info = &info;
			driver->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		auto gpuMesh = driver->getGPUObjectsFromAssets(&cpuMesh.get(), &cpuMesh.get() + 1)->front();

		return std::make_tuple(gpuMesh, gpuubo, gpuds1, ds1UboBinding, pipelineMetadata);
	};

	auto& plyDrawData = getMeshDependentDrawData(cpuMeshPly, true);
	auto& stlDrawData = getMeshDependentDrawData(cpuMeshStl, false);

	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(1.f);
	camera->setFarValue(5000.0f);

	smgr->setActiveCamera(camera);

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewProjection = camera->getConcatenatedMatrix();

		auto renderMesh = [&](DependentDrawData& drawData, uint32_t index)
		{
			core::smart_refctd_ptr<video::IGPUMesh> gpuMesh = std::get<0>(drawData);
			core::smart_refctd_ptr<video::IGPUBuffer> gpuubo = std::get<1>(drawData);
			core::smart_refctd_ptr<video::IGPUDescriptorSet> gpuds1 = std::get<2>(drawData);
			uint32_t ds1UboBinding = std::get<3>(drawData);
			const asset::IRenderpassIndependentPipelineMetadata* pipelineMetadata = std::get<4>(drawData);

			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(index * 5, 0, 0, 0));

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);

			core::vector<uint8_t> uboData(gpuubo->getSize());
			for (const auto& shaderInputs : pipelineMetadata->m_inputSemantics)
			{
				if (shaderInputs.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shaderInputs.descriptorSection.uniformBufferObject.set == 1u && shaderInputs.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
				{
					switch (shaderInputs.type)
					{
						case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
						{
							memcpy(uboData.data() + shaderInputs.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shaderInputs.descriptorSection.uniformBufferObject.bytesize);
						} break;

						case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
						{
							core::matrix3x4SIMD MV = camera->getViewMatrix();
							memcpy(uboData.data() + shaderInputs.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shaderInputs.descriptorSection.uniformBufferObject.bytesize);
						} break;

						case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
						{
							core::matrix3x4SIMD MV = camera->getViewMatrix();
							memcpy(uboData.data() + shaderInputs.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shaderInputs.descriptorSection.uniformBufferObject.bytesize);
						} break;
					}
				}
			}
			driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

			for (auto gpuMeshBuffer : gpuMesh->getMeshBuffers())
			{
				const video::IGPURenderpassIndependentPipeline* gpuPipeline = gpuMeshBuffer->getPipeline();
				const video::IGPUDescriptorSet* gpuds3 = gpuMeshBuffer->getAttachedDescriptorSet();

				driver->bindGraphicsPipeline(gpuPipeline);
				const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
				driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);

				if (gpuds3) //! Execute if we have a texture attached as DS
					driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 3u, 1u, &gpuds3, nullptr);
				driver->pushConstants(gpuPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

				driver->drawMeshBuffer(gpuMeshBuffer);
			}
		};

		/*
			Render PLY and STL
		*/

		renderMesh(plyDrawData, 0);
		renderMesh(stlDrawData, 20);

		driver->endScene();

		/* 
			Display frames per second in window title
		*/

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"PLY & STL Test Demo - Nabla Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}
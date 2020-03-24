#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"
#include "irr/asset/CGeometryCreator.h"

using namespace irr;
using namespace asset;
using namespace video;
using namespace core;

int main()
{
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

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	auto driver = device->getVideoDriver();
	auto assetManager = device->getAssetManager();
	auto smgr = device->getSceneManager();

	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(0, -5, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(1000.0f);

	smgr->setActiveCamera(camera);

	auto geometryCreator = device->getAssetManager()->getGeometryCreator();
	auto rectangleGeometry = geometryCreator->createRectangleMesh(irr::core::vector2df_SIMD(1.5, 3));

	constexpr std::string_view cacheKey = "irr/builtin/materials/lambertian/singletexture/specializedshader";
	const IAsset::E_TYPE types[]{ IAsset::E_TYPE::ET_SPECIALIZED_SHADER, IAsset::E_TYPE::ET_SPECIALIZED_SHADER, static_cast<IAsset::E_TYPE>(0u) };

	auto vertexShader = core::smart_refctd_ptr<ICPUSpecializedShader>();
	auto fragmentShader = core::smart_refctd_ptr<ICPUSpecializedShader>();

	auto bundle = assetManager->findAssets(cacheKey.data(), types);

	auto refCountedBundle =
	{
		core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(bundle->begin()->getContents().first[0]),
		core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>((bundle->begin() + 1)->getContents().first[0])
	};

	for (auto& shader : refCountedBundle)
		if (shader->getStage() == ISpecializedShader::ESS_VERTEX)
			vertexShader = std::move(shader);
		else if (shader->getStage() == ISpecializedShader::ESS_FRAGMENT)
			fragmentShader = std::move(shader);

	size_t ds1UboBinding = 0, neededDS1UBOsz = 0;
	auto createGPUMeshBufferAndItsPipeline = [&](asset::IGeometryCreator::return_type& geometryObject)
	{
		asset::ICPUDescriptorSetLayout::SBinding binding1;
		binding1.count = 1u;
		binding1.binding = ds1UboBinding = 0u;
		binding1.stageFlags = static_cast<asset::ICPUSpecializedShader::E_SHADER_STAGE>(asset::ICPUSpecializedShader::ESS_VERTEX | asset::ICPUSpecializedShader::ESS_FRAGMENT);
		binding1.type = asset::EDT_UNIFORM_BUFFER;
		auto ds1Layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(&binding1, &binding1 + 1);

		auto pipelineLayout = core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(nullptr, nullptr, nullptr, std::move(ds1Layout), nullptr, nullptr); // doing so removes binding1 from ds1Layout placed in pipeline, why?
		auto gpuPipelineLayout = driver->getGPUObjectsFromAssets(&pipelineLayout.get(), &pipelineLayout.get() + 1)->front();

		auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(neededDS1UBOsz);
		auto rawds1 = pipelineLayout->getDescriptorSetLayout(1u);
		auto gpuDescriptorSet1 = driver->createGPUDescriptorSet(std::move(driver->getGPUObjectsFromAssets(&rawds1, &rawds1 + 1)->front()));
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
				info.buffer.size = neededDS1UBOsz;
			}
			write.info = &info;
			driver->updateDescriptorSets(1u, &write, 0u, nullptr);
		}

		constexpr size_t DS1_METADATA_ENTRY_CNT = 3ull;
		core::smart_refctd_dynamic_array<IPipelineMetadata::ShaderInputSemantic> shaderInputsMetadata = core::make_refctd_dynamic_array<decltype(shaderInputsMetadata)>(DS1_METADATA_ENTRY_CNT);
		{
			ICPUDescriptorSetLayout* ds1layout = pipelineLayout->getDescriptorSetLayout(1u);

			constexpr IPipelineMetadata::E_COMMON_SHADER_INPUT types[DS1_METADATA_ENTRY_CNT]{ IPipelineMetadata::ECSI_WORLD_VIEW_PROJ, IPipelineMetadata::ECSI_WORLD_VIEW, IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE };
			constexpr uint32_t sizes[DS1_METADATA_ENTRY_CNT]{ sizeof(SBasicViewParameters::MVP), sizeof(SBasicViewParameters::MV), sizeof(SBasicViewParameters::NormalMat) };
			constexpr uint32_t relOffsets[DS1_METADATA_ENTRY_CNT]{ offsetof(SBasicViewParameters,MVP), offsetof(SBasicViewParameters,MV), offsetof(SBasicViewParameters,NormalMat) };
			for (uint32_t i = 0u; i < DS1_METADATA_ENTRY_CNT; ++i)
			{
				auto& semantic = (shaderInputsMetadata->end() - i - 1u)[0];
				semantic.type = types[i];
				semantic.descriptorSection.type = IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER;
				semantic.descriptorSection.uniformBufferObject.binding = ds1layout->getBindings().begin()[0].binding;
				semantic.descriptorSection.uniformBufferObject.set = 1u;
				semantic.descriptorSection.uniformBufferObject.relByteoffset = relOffsets[i];
				semantic.descriptorSection.uniformBufferObject.bytesize = sizes[i];
				semantic.descriptorSection.shaderAccessFlags = ICPUSpecializedShader::ESS_VERTEX;

				neededDS1UBOsz += sizes[i];
			}
		}

		asset::SBlendParams blendParams;
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		auto pipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(pipelineLayout), nullptr, nullptr, geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);
		pipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_VERTEX_SHADER_IX, vertexShader.get());
		pipeline->setShaderAtIndex(ICPURenderpassIndependentPipeline::ESSI_FRAGMENT_SHADER_IX, fragmentShader.get());
		
		assetManager->setAssetMetadata(pipeline.get(), core::make_smart_refctd_ptr<CPLYPipelineMetadata>(1, std::move(shaderInputsMetadata)));
		auto metadata = pipeline->getMetadata();

		auto gpuPipeline = driver->getGPUObjectsFromAssets(&pipeline.get(), &pipeline.get() + 1)->front();

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

		return std::make_tuple(mb, gpuPipeline, gpuubo, metadata, gpuDescriptorSet1);
	};

	auto gpuRectangle = createGPUMeshBufferAndItsPipeline(rectangleGeometry);
	auto gpuMeshBuffer = std::get<0>(gpuRectangle);
	auto gpuPipeline = std::get<1>(gpuRectangle);
	auto gpuubo = std::get<2>(gpuRectangle);
	auto metadata = std::get<3>(gpuRectangle);
	auto gpuDescriptorSet1 = std::get<4>(gpuRectangle);

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 255, 255, 255));

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		core::vector<uint8_t> uboData(gpuubo->getSize());
		auto pipelineMetadata = static_cast<const asset::IPipelineMetadata*>(metadata);
		for (const auto& shdrIn : pipelineMetadata->getCommonRequiredInputs())
		{
			if (shdrIn.descriptorSection.type == asset::IPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
			{
				switch (shdrIn.type)
				{
				case asset::IPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
				{
					core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
					memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
				}
				break;
				case asset::IPipelineMetadata::ECSI_WORLD_VIEW:
				{
					core::matrix3x4SIMD MV = camera->getViewMatrix();
					memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
				}
				break;
				case asset::IPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
				{
					core::matrix3x4SIMD MV = camera->getViewMatrix();
					memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
				}
				break;
				}
			}
		}
		driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

		driver->bindGraphicsPipeline(gpuPipeline.get());
		driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), nullptr);
		driver->drawMeshBuffer(gpuMeshBuffer.get());
		
		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"GPU Mesh Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}

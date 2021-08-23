#define _IRR_STATIC_LIB_
#include <nabla.h>

#include "../include/nbl/asset/metadata/CGLTFMetadata.h"
#include "../common/QToQuitEventReceiver.h"
#include "../include/nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace asset;
using namespace video;
using namespace core;

#include "nbl/nblpack.h"
struct GraphicsData
{
	struct Mesh
	{
		struct Resources
		{
			const IGPUMeshBuffer* gpuMeshBuffer;
			const IGPURenderpassIndependentPipeline* gpuPipeline;
			const asset::CGLTFPipelineMetadata* pipelineMetadata;
		};

		core::vector<Resources> resources;
	};

	core::vector<Mesh> meshes;

} PACK_STRUCT;
#include "nbl/nblunpack.h"

int main()
{
	SIrrlichtCreationParameters params;
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
    auto* sceneManager = device->getSceneManager();
    auto* assetManager = device->getAssetManager();
    auto* fileSystem = assetManager->getFileSystem();

    asset::IAssetLoader::SAssetLoadParams loadingParams;

    auto meshes_bundle = assetManager->getAsset("../../../3rdparty/glTFSampleModels/2.0/Avocado/glTF/Avocado.gltf", loadingParams);
	{
		bool status = !meshes_bundle.getContents().empty();
		assert(status);
	}
   
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto* mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());
	auto* meta = meshes_bundle.getMetadata()->selfCast<asset::CGLTFMetadata>();

	auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));

	/*
		We can safely assume that all mesh buffers within mesh loaded from glTF has the same DS1 layout 
		used for camera-specific data, so we can create just one DS.
	*/

	auto cpuDescriptorSetLayout1 = mesh_raw->getMeshBuffers().begin()[0]->getPipeline()->getLayout()->getDescriptorSetLayout(1);
	auto gpuDescriptorSet1Layout = driver->getGPUObjectsFromAssets(&cpuDescriptorSetLayout1, &cpuDescriptorSetLayout1 + 1)->front();

	auto gpuDescriptorSet1 = driver->createGPUDescriptorSet(std::move(gpuDescriptorSet1Layout));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write;
		write.dstSet = gpuDescriptorSet1.get();
		write.binding = 0;
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

	/*
		TODO: MORE DS3s, use metadata to fetch translations, track it and set up in the graphicsData struct
	*/

	GraphicsData graphicsData;
	for (auto* asset = meshes_bundle.getContents().begin(); asset != meshes_bundle.getContents().end(); ++asset)
	{
		auto& graphicsDataMesh = graphicsData.meshes.emplace_back();

		auto cpuMesh = core::smart_refctd_ptr_static_cast<ICPUMesh>(*asset);
		{
			for (size_t i = 0; i < cpuMesh->getMeshBuffers().size(); ++i)
			{
				auto& graphicsResources = graphicsDataMesh.resources.emplace_back();
				auto* glTFMetaPushConstants = meta->getAssetSpecificMetadata(cpuMesh->getMeshBufferVector()[i]->getPipeline());
				graphicsResources.pipelineMetadata = glTFMetaPushConstants;
			}
		}

		auto gpuMesh = driver->getGPUObjectsFromAssets(&cpuMesh.get(), &cpuMesh.get() + 1)->front();

		for (size_t i = 0; i < gpuMesh->getMeshBuffers().size(); ++i)
		{
			auto* gpuMeshBuffer = graphicsDataMesh.resources[i].gpuMeshBuffer = (gpuMesh->getMeshBufferIterator() + i)->get();
			graphicsDataMesh.resources[i].gpuPipeline = gpuMeshBuffer->getPipeline();
		}
	}

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0,100.0f,0.001f);

	camera->setPosition(core::vector3df(-0.5,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10000.0f);

    sceneManager->setActiveCamera(camera);

	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewProjection = camera->getConcatenatedMatrix();
		core::matrix3x4SIMD modelMatrix;
		modelMatrix.setRotation(quaternion(0, 0, 0));

		auto mv = core::concatenateBFollowedByA(camera->getViewMatrix(), modelMatrix);
		auto mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
		core::matrix3x4SIMD normalMat;
		mv.getSub3x3InverseTranspose(normalMat);

		/*
			Camera data is shared between all meshes
		*/

		SBasicViewParameters uboData;
		memcpy(uboData.MV, mv.pointer(), sizeof(mv));
		memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
		memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
		driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

		for (auto& gpuMeshData : graphicsData.meshes)
		{
			for (auto& graphicsResource : gpuMeshData.resources)
			{
				driver->pushConstants(graphicsResource.gpuPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, sizeof(asset::CGLTFPipelineMetadata::SGLTFMaterialParameters), &graphicsResource.pipelineMetadata->m_materialParams);
				driver->bindGraphicsPipeline(graphicsResource.gpuPipeline);
				driver->bindDescriptorSets(video::EPBP_GRAPHICS, graphicsResource.gpuPipeline->getLayout(), 1u, 1u, &gpuDescriptorSet1.get(), nullptr);

				auto* gpuDescriptorSet3 = graphicsResource.gpuMeshBuffer->getAttachedDescriptorSet();
				if(gpuDescriptorSet3)
					driver->bindDescriptorSets(video::EPBP_GRAPHICS, graphicsResource.gpuPipeline->getLayout(), 3u, 1u, &gpuDescriptorSet3, nullptr);

				driver->drawMeshBuffer(graphicsResource.gpuMeshBuffer);
			}
		}
        
		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"glTF Demo - IrrlichtBAW Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}
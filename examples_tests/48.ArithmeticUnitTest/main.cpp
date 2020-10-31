#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../source/Irrlicht/COpenGLDriver.h"


using namespace irr;
using namespace core;
using namespace video;
using namespace asset;

enum TestOperation {
	TO_AND=1,
	TO_XOR,
	TO_OR,
	TO_ADD,
	TO_MUL,
	TO_MIN,
	TO_MAX
};

#define BUFFER_SIZE 128u*1024u*1024u

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();

	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();



	//TODO: create 8 128 *1024*1024 big buffers.
	auto inputDataBuffer...
	
	//TODO: fill the first buffer with random numbers
	//TODO: Create array with cpu stored numbers


	IGPUDescriptorSetLayout::SBinding binding[15] = {
		{0u,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},	//input with randomized numbers
		{TO_AND,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_XOR,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_OR,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_ADD,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MUL,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MIN,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
		{TO_MAX,EDT_STORAGE_BUFFER,1u,IGPUSpecializedShader::ESS_COMPUTE,nullptr},
	};
	auto gpuDSLayout = driver->createGPUDescriptorSetLayout(binding, binding + 15);
	constexpr uint32_t pushconstantSize = 1;
	SPushConstantRange pcRange[1] = { IGPUSpecializedShader::ESS_COMPUTE,0u,pushconstantSize };
	auto pipelineLayout = driver->createGPUPipelineLayout(pcRange, pcRange + sizeof(pcRange) / pushconstantSize, core::smart_refctd_ptr(gpuDSLayout));

	auto descriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDSLayout));
	{
		//TODO bind the buffers
		IGPUDescriptorSet::SDescriptorInfo infos[8];
		infos[0].desc = core::smart_refctd_ptr<IGPUBuffer>(inputDataBuffer.getObject());
		infos[0].buffer = { 0u, BUFFER_SIZE };
		for (size_t i = 1; i <= 7; i++)
		{
			infos[i].desc = core::smart_refctd_ptr<IGPUBuffer>(buffer[i].getObject());
			infos[i].buffer = { 0u,BUFFER_SIZE };

		}
		infos[2].desc = histogramBuffer;
		infos[2].buffer = { 0u,HistogramBufferSize };
		infos[3].desc = core::smart_refctd_ptr<IGPUBuffer>(intensityBuffer.getObject());
		infos[3].buffer = { 0u,intensityBuffer.getObject()->getMemoryReqs().vulkanReqs.size };
		IGPUDescriptorSet::SWriteDescriptorSet writes[SharedDescriptorSetDescCount];
		for (uint32_t i = 0u; i < SharedDescriptorSetDescCount; i++)
			writes[i] = { descriptorSet.get(),i,0u,1u,EDT_STORAGE_BUFFER,infos + i };
		driver->updateDescriptorSets(SharedDescriptorSetDescCount, writes, 0u, nullptr);
	}
	auto getGPUShader = [&](irr::io::path& filePath)
	{
		auto file = core::smart_refctd_ptr<io::IReadFile>(filesystem->createAndOpenFile(filePath));
		asset::IAssetLoader::SAssetLoadParams lp;
		auto cs_bundle = am->getAsset(filePath.c_str(), lp);
		auto cs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*cs_bundle.getContents().begin());
		auto cs_rawptr = cs.get();
		core::smart_refctd_ptr<IGPUSpecializedShader> shader = driver->getGPUObjectsFromAssets(&cs_rawptr, &cs_rawptr + 1)->front();
		return shader;
	};


	auto reducePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),		std::move(getGPUShader(io::path("../testReduce.comp"))));
	auto inclusivePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),	std::move(getGPUShader(io::path("../testInclusive.comp"))));
	auto exclusicePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),	std::move(getGPUShader(io::path("../testExclusive.comp"))));




	

	while (device->run())
	{
		driver->beginScene(true);

		//reduce test
		driver->bindComputePipeline(reducePipeline.get());
		const video::IGPUDescriptorSet* descriptorSet = ds0_gpu.get();
		driver->bindDescriptorSets(video::EPBP_COMPUTE, reducePipeline->getLayout(), 0u, 1u, &descriptorSet, nullptr);
		driver->pushConstants(compPipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(imgSize), imgSize);
		driver->dispatch((imgSize[0] + 15u) / 16u, (imgSize[1] + 15u) / 16u, 1u);

		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
		break;
	}

	return 0;
}

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <irrlicht.h>
#include "../common/QToQuitEventReceiver.h"
#include "COpenGLExtensionHandler.h"

using namespace irr;
using namespace core;

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

	//! disable mouse cursor, since camera will force it to the middle
	//! and we don't want a jittery cursor in the middle distracting us
	device->getCursorControl()->setVisible(false);

	//! Since our cursor will be enslaved, there will be no way to close the window
	//! So we listen for the "Q" key being pressed and exit the application
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();

	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();
	scene::ISceneManager* smgr = device->getSceneManager();

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(1.f);
	camera->setFarValue(5000.0f);

	auto glslExts = driver->getSupportedGLSLExtensions();
	asset::CShaderIntrospector introspector(am->getGLSLCompiler());

	core::smart_refctd_ptr<asset::ICPUShader> computeUnspec;
	{
		auto file = filesystem->createAndOpenFile("../particles.comp");
		//computeUnspec = am->getGLSLCompiler()->createSPIRVFromGLSL(file, asset::ISpecializedShader::ESS_COMPUTE, "main", file->getFileName().c_str());
		computeUnspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_COMPUTE, file->getFileName().c_str());
		file->drop();
	}
	const asset::CIntrospectionData* introspection = nullptr;
	{
		asset::CShaderIntrospector::SIntrospectionParams params;
		params.entryPoint = "main";
		params.filePathHint = "../particles.comp";
		params.GLSLextensions = glslExts;
		params.stage = asset::ISpecializedShader::ESS_COMPUTE;
		introspection = introspector.introspect(computeUnspec.get(), params);
	}
	constexpr uint32_t COMPUTE_SET = 0u;
	constexpr uint32_t PARTICLE_BUF_BINDING = 0u;
	constexpr uint32_t COMPUTE_DATA_UBO_BINDING = 1u;

	constexpr uint32_t WORKGROUP_SIZE = 256u;
	constexpr uint32_t PARTICLE_COUNT = 1u<<21;
	constexpr uint32_t PARTICLE_COUNT_PER_AXIS = 1u<<7;
	constexpr uint32_t POS_BUF_IX = 0u;
	constexpr uint32_t VEL_BUF_IX = 1u;
	constexpr uint32_t BUF_COUNT = 2u;
	asset::ISpecializedShader::SInfo specInfo;
	{
		struct SpecConstants
		{
			int32_t wg_size;
			int32_t particle_count;
			int32_t pos_buf_ix;
			int32_t vel_buf_ix;
			int32_t buf_count;
		};
		SpecConstants sc {WORKGROUP_SIZE, PARTICLE_COUNT, POS_BUF_IX, VEL_BUF_IX, BUF_COUNT};

		auto it_particleBufDescIntro = std::find_if(introspection->descriptorSetBindings[COMPUTE_SET].begin(), introspection->descriptorSetBindings[COMPUTE_SET].end(),
			[=] (auto b) { return b.binding==PARTICLE_BUF_BINDING; }
			);
		assert(it_particleBufDescIntro->descCountIsSpecConstant);
		const uint32_t buf_count_specID = it_particleBufDescIntro->count_specID;
		auto& particleDataArrayIntro = it_particleBufDescIntro->get<asset::ESRT_STORAGE_BUFFER>().members.array[0];
		assert(particleDataArrayIntro.countIsSpecConstant);
		const uint32_t particle_count_specID = particleDataArrayIntro.count_specID;

		auto backbuf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(sc));
		memcpy(backbuf->getPointer(), &sc, sizeof(sc));
		auto entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ISpecializedShader::SInfo::SMapEntry>>(5u);
		(*entries)[0] = {0u,offsetof(SpecConstants,wg_size),sizeof(int32_t)};//currently local_size_{x|y|z}_id is not queryable via introspection API
		(*entries)[1] = {particle_count_specID,offsetof(SpecConstants,particle_count),sizeof(int32_t)};
		(*entries)[2] = {2u,offsetof(SpecConstants,pos_buf_ix),sizeof(int32_t)};
		(*entries)[3] = {3u,offsetof(SpecConstants,vel_buf_ix),sizeof(int32_t)};
		(*entries)[4] = {buf_count_specID,offsetof(SpecConstants,buf_count),sizeof(int32_t)};

		specInfo = asset::ISpecializedShader::SInfo(std::move(entries), std::move(backbuf), "main", asset::ISpecializedShader::ESS_COMPUTE, "../particles.comp");
	}
	auto compute = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(computeUnspec), std::move(specInfo));

	auto computePipeline = introspector.createApproximateComputePipelineFromIntrospection(compute.get(), glslExts->begin(), glslExts->end());

	auto gpuComputePipeline = driver->getGPUObjectsFromAssets(&computePipeline.get(),&computePipeline.get()+1)->front();
	auto* ds0layoutCompute = computePipeline->getLayout()->getDescriptorSetLayout(0);
	auto gpuDs0layoutCompute = driver->getGPUObjectsFromAssets(&ds0layoutCompute,&ds0layoutCompute+1)->front();

	struct UBOCompute
	{
		//xyz - gravity point, w - dt
		core::vectorSIMDf gravPointAndDt;
	};
	UBOCompute uboComputeData;

	core::vector<core::vector3df_SIMD> particlePos;
	particlePos.reserve(PARTICLE_COUNT);
	for (int32_t i = 0; i < PARTICLE_COUNT_PER_AXIS; ++i)
	for (int32_t j = 0; j < PARTICLE_COUNT_PER_AXIS; ++j)
	for (int32_t k = 0; k < PARTICLE_COUNT_PER_AXIS; ++k)
		particlePos.push_back(core::vector3df_SIMD(i,j,k)*0.5f);

	constexpr size_t BUF_SZ = 4ull*sizeof(float)*PARTICLE_COUNT;
	auto gpuParticleBuf = driver->createDeviceLocalGPUBufferOnDedMem(2ull*BUF_SZ);
	driver->updateBufferRangeViaStagingBuffer(gpuParticleBuf.get(), POS_BUF_IX*BUF_SZ, BUF_SZ, particlePos.data());
	particlePos.clear();
	driver->fillBuffer(gpuParticleBuf.get(), VEL_BUF_IX*BUF_SZ, BUF_SZ, 0u);
	auto gpuUboCompute = driver->createDeviceLocalGPUBufferOnDedMem(core::roundUp(sizeof(UBOCompute),64ull));
	auto gpuds0Compute = driver->createGPUDescriptorSet(std::move(gpuDs0layoutCompute));
	{
		video::IGPUDescriptorSet::SDescriptorInfo i[3];
		video::IGPUDescriptorSet::SWriteDescriptorSet w[2];
		w[0].arrayElement = 0u;
		w[0].binding = PARTICLE_BUF_BINDING;
		w[0].count = BUF_COUNT;
		w[0].descriptorType = asset::EDT_STORAGE_BUFFER;
		w[0].dstSet = gpuds0Compute.get();
		w[0].info = i;
		i[0].desc = gpuParticleBuf;
		i[0].buffer.offset = 0ull;
		i[0].buffer.size = BUF_SZ;
		i[1].desc = gpuParticleBuf;
		i[1].buffer.offset = BUF_SZ;
		i[1].buffer.size = BUF_SZ;

		w[1].arrayElement = 0u;
		w[1].binding = COMPUTE_DATA_UBO_BINDING;
		w[1].count = 1u;
		w[1].descriptorType = asset::EDT_UNIFORM_BUFFER;
		w[1].dstSet = gpuds0Compute.get();
		w[1].info = i+2;
		i[2].buffer.offset = 0u;
		i[2].buffer.size = gpuUboCompute->getSize();
		i[2].desc = gpuUboCompute;

		driver->updateDescriptorSets(2u, w, 0u, nullptr);
	}

	asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	vtxBindings[0].buffer = gpuParticleBuf;
	vtxBindings[0].offset = 0u;
	auto meshbuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(nullptr, nullptr, vtxBindings, asset::SBufferBinding<video::IGPUBuffer>{});
	meshbuffer->setIndexCount(PARTICLE_COUNT);
	meshbuffer->setIndexType(asset::EIT_UNKNOWN);

	auto createSpecShader = [&](const char* filepath, asset::ISpecializedShader::E_SHADER_STAGE stage) {
		auto file = filesystem->createAndOpenFile(filepath);
		auto unspec = am->getGLSLCompiler()->resolveIncludeDirectives(file, stage, file->getFileName().c_str());
		//unspec = am->getGLSLCompiler()->createSPIRVFromGLSL(reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer()), stage, "main", file->getFileName().c_str());

		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", stage, file->getFileName().c_str());
		file->drop();
		return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspec), std::move(info));
	};
	auto vs = createSpecShader("../particles.vert", asset::ISpecializedShader::ESS_VERTEX);
	auto fs = createSpecShader("../particles.frag", asset::ISpecializedShader::ESS_FRAGMENT);
	asset::ICPUSpecializedShader* shaders[2] {vs.get(),fs.get()};
	auto pipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders, shaders+2, glslExts->begin(), glslExts->end());
	{
		auto& vtxParams = pipeline->getVertexInputParams();
		vtxParams.attributes[0].binding = 0u;
		vtxParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
		vtxParams.attributes[0].relativeOffset = 0u;
		vtxParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
		vtxParams.bindings[0].stride = 4u*sizeof(float);

		pipeline->getPrimitiveAssemblyParams().primitiveType = asset::EPT_POINT_LIST;
	}
	auto gpuPipeline = driver->getGPUObjectsFromAssets(&pipeline.get(),&pipeline.get()+1)->front();
	auto* ds0layoutGraphics = pipeline->getLayout()->getDescriptorSetLayout(0);
	auto gpuDs0layoutGraphics = driver->getGPUObjectsFromAssets(&ds0layoutGraphics,&ds0layoutGraphics+1)->front();
	auto gpuds0Graphics = driver->createGPUDescriptorSet(std::move(gpuDs0layoutGraphics));

	constexpr uint32_t GRAPHICS_SET = 0u;
	constexpr uint32_t GRAPHICS_DATA_UBO_BINDING = 0u;

	asset::SBasicViewParameters viewParams;
	auto gpuUboGaphics = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(viewParams));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet w;
		video::IGPUDescriptorSet::SDescriptorInfo i;
		w.arrayElement = 0u;
		w.binding = GRAPHICS_DATA_UBO_BINDING;
		w.count = 1u;
		w.descriptorType = asset::EDT_UNIFORM_BUFFER;
		w.dstSet = gpuds0Graphics.get();
		w.info = &i;
		i.desc = gpuUboGaphics;
		i.buffer.offset = 0u;
		i.buffer.size = gpuUboGaphics->getSize();

		driver->updateDescriptorSets(1u, &w, 0u, nullptr);
	}

	uint64_t lastTime = device->getTimer()->getRealTime64();
	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 0));

		//! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		auto camFront = camera->getViewMatrix()[2];
		core::vector3df_SIMD gravPoint = core::vector3df_SIMD(&camera->getPosition().X) + camFront*250.f;
		uint64_t time = device->getTimer()->getRealTime64();
		uboComputeData.gravPointAndDt = gravPoint;
		uboComputeData.gravPointAndDt.w = (time-lastTime)*1e-3f;
		lastTime = time;
		driver->updateBufferRangeViaStagingBuffer(gpuUboCompute.get(), 0u, sizeof(uboComputeData), &uboComputeData);

		driver->bindComputePipeline(gpuComputePipeline.get());
		driver->bindDescriptorSets(video::EPBP_COMPUTE, gpuComputePipeline->getLayout(), COMPUTE_SET, 1u, &gpuds0Compute.get(), nullptr);

		driver->dispatch(PARTICLE_COUNT/WORKGROUP_SIZE, 1u, 1u);

		video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

		memcpy(viewParams.MVP, camera->getConcatenatedMatrix().pointer(), sizeof(viewParams.MVP));
		driver->updateBufferRangeViaStagingBuffer(gpuUboGaphics.get(), 0u, gpuUboGaphics->getSize(), &viewParams);
		driver->bindGraphicsPipeline(gpuPipeline.get());
		driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), GRAPHICS_SET, 1u, &gpuds0Graphics.get(), nullptr);

		driver->drawMeshBuffer(meshbuffer.get());

		driver->endScene();

		if (time-lastFPSTime > 1000ull)
		{
			std::wostringstream str;
			str << L"Specialization Constants Demo - IrrlichtBAW Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}
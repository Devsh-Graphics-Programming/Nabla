// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "nbl/asset/CGeometryCreator.h"

using namespace nbl;
using namespace core;

#include "nbl/irrpack.h"
struct GPUObject
{
	core::smart_refctd_ptr<video::IGPUMeshBuffer> meshbuffer;
	core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline;
} PACK_STRUCT;

struct Objects
{
	enum E_OBJECT_INDEX
	{
		E_CUBE,
		E_SPHERE,
		E_CYLINDER,
		E_RECTANGLE,
		E_DISK,
		E_CONE,
		E_ARROW,
		E_ICOSPHERE,
		E_COUNT
	};

	Objects(std::initializer_list<std::pair<asset::IGeometryCreator::return_type, GPUObject>> _objects) : objects(_objects) {}

	const std::vector<std::pair<asset::IGeometryCreator::return_type, GPUObject>> objects;
} PACK_STRUCT;
#include "nbl/irrunpack.h"

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos;
layout(location = 3) in vec3 vNormal;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec3 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = PushConstants.modelViewProj*vPos;
    Color = vNormal*0.5+vec3(0.5);
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec3 Color;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(Color,1.0);
}
)===";

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
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

	auto* driver = device->getVideoDriver();
	auto* smgr = device->getSceneManager();

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);

	camera->setPosition(core::vector3df(-5, 0, 0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);

    smgr->setActiveCamera(camera);

	auto geometryCreator = device->getAssetManager()->getGeometryCreator();
	auto cubeGeometry = geometryCreator->createCubeMesh(vector3df(2,2,2));
	auto sphereGeometry = geometryCreator->createSphereMesh(2, 16, 16);
	auto cylinderGeometry = geometryCreator->createCylinderMesh(2, 2, 20);
	auto rectangleGeometry = geometryCreator->createRectangleMesh(nbl::core::vector2df_SIMD(1.5, 3));
	auto diskGeometry = geometryCreator->createDiskMesh(2, 30);
	auto coneGeometry = geometryCreator->createConeMesh(2, 3, 10);
	auto arrowGeometry = geometryCreator->createArrowMesh();
	auto icosphereGeometry = geometryCreator->createIcoSphere(1, 3, true);

	auto createGPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage) -> core::smart_refctd_ptr<video::IGPUSpecializedShader>
	{
		auto spirv = device->getAssetManager()->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		if (!spirv)
			return nullptr;
		auto unspec = driver->createGPUShader(std::move(spirv));
		return driver->createGPUSpecializedShader(unspec.get(), { nullptr, nullptr, "main", stage });
	};

	auto createGPUSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
	{
		auto resolved_includes = device->getAssetManager()->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
		return createGPUSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
	};

	core::smart_refctd_ptr<video::IGPUSpecializedShader> shaders[2] =
	{
		createGPUSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert"),
		createGPUSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT)
	};
	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader**>(shaders);

	auto createGPUMeshBufferAndItsPipeline = [&](asset::IGeometryCreator::return_type& geometryObject) -> GPUObject
	{
		asset::SBlendParams blendParams; 
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;

		asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
		auto pipeline = driver->createGPURenderpassIndependentPipeline(nullptr, driver->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
			shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
			geometryObject.inputParams, blendParams, geometryObject.assemblyParams, rasterParams);

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

		auto gpubuffers = driver->getGPUObjectsFromAssets(cpubuffers.data(), cpubuffers.data()+cpubuffers.size());

		asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
		for (auto i=0,j=0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
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

		auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(pipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
		{
			mb->setIndexType(geometryObject.indexType);
			mb->setIndexCount(geometryObject.indexCount);
			mb->setBoundingBox(geometryObject.bbox);
		}

		return { mb, pipeline };
	};

	auto gpuCube = createGPUMeshBufferAndItsPipeline(cubeGeometry);
	auto gpuSphere = createGPUMeshBufferAndItsPipeline(sphereGeometry);
	auto gpuCylinder = createGPUMeshBufferAndItsPipeline(cylinderGeometry);
	auto gpuRectangle = createGPUMeshBufferAndItsPipeline(rectangleGeometry);
	auto gpuDisk = createGPUMeshBufferAndItsPipeline(diskGeometry);
	auto gpuCone = createGPUMeshBufferAndItsPipeline(coneGeometry);
	auto gpuArrow = createGPUMeshBufferAndItsPipeline(arrowGeometry);
	auto gpuIcosphere = createGPUMeshBufferAndItsPipeline(icosphereGeometry);

	Objects cpuGpuObjects =
	{
		std::make_pair(cubeGeometry, gpuCube),
		std::make_pair(sphereGeometry, gpuSphere),
		std::make_pair(cylinderGeometry, gpuCylinder),
		std::make_pair(rectangleGeometry, gpuRectangle),
		std::make_pair(diskGeometry, gpuDisk),
		std::make_pair(coneGeometry, gpuCone),
		std::make_pair(arrowGeometry, gpuArrow), 
		std::make_pair(icosphereGeometry, gpuIcosphere)
	};
	
	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		// draw available objects placed in the vector
		const auto viewProjection = camera->getConcatenatedMatrix();
		for (auto index = 0u; index < cpuGpuObjects.objects.size(); ++index)
		{
			const auto iterator = cpuGpuObjects.objects[index];
			auto geometryObject = iterator.first;
			auto gpuObject = iterator.second;
			
			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(nbl::core::vectorSIMDf(index * 5, 0, 0, 0));

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);
			driver->bindGraphicsPipeline(gpuObject.pipeline.get());
			driver->pushConstants(gpuObject.pipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			driver->drawMeshBuffer(gpuObject.meshbuffer.get());
		}

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"GPU Mesh Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}

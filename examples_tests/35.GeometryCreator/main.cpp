#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "irr/asset/CGeometryCreator.h"

using namespace irr;
using namespace core;

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 3) in vec3 vNormal;

//#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec3 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    //gl_Position = irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4(PushConstants.modelViewProj)*vPos; //only thing preventing the shader from being core-compliant
    gl_Position = PushConstants.modelViewProj*vPos; //only thing preventing the shader from being core-compliant
    Color = vNormal*0.5+vec3(0.5);
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec3 Color; //per vertex output color, will be interpolated across the triangle

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(Color,1.0);
}
)===";

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

	auto* driver = device->getVideoDriver();
	auto* smgr = device->getSceneManager();

	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.001f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10.0f);

    smgr->setActiveCamera(camera);

	auto geometryCreator = device->getAssetManager()->getGeometryCreator();
	auto cubeGeom = geometryCreator->createCubeMesh(vector3df(2,2,2));
	auto inputParams = cubeGeom.inputParams;
	auto assemblyParams = cubeGeom.assemblyParams;
	auto vertices = cubeGeom.bindings[0];
	auto indices = cubeGeom.indexBuffer;


	auto createGPUSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage)
	{
		auto spirv = device->getAssetManager()->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
		auto unspec = driver->createGPUShader(std::move(spirv));
		return driver->createGPUSpecializedShader(unspec.get(), { core::vector<asset::ISpecializedShader::SInfo::SMapEntry>(),nullptr,"main",stage });
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
	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader * *>(shaders);

	asset::SBlendParams blendParams; // defaults are same
	asset::SRasterizationParams rasterParams;

	asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };
	auto pipeline = driver->createGPURenderpassIndependentPipeline(nullptr, driver->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
		shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
		inputParams, blendParams, assemblyParams, rasterParams);

	constexpr auto MAX_ATTR_BUF_BINDING_COUNT = video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT;
	constexpr auto MAX_DATA_BUFFERS = MAX_ATTR_BUF_BINDING_COUNT+1;
	asset::ICPUBuffer* cpubuffers[MAX_DATA_BUFFERS];
	for (auto i=0; i<MAX_ATTR_BUF_BINDING_COUNT; i++)
		cpubuffers[i] = cubeGeom.bindings[i].buffer.get();
	cpubuffers[MAX_ATTR_BUF_BINDING_COUNT] = cubeGeom.indexBuffer.buffer.get();
	
	auto gpubuffers = driver->getGPUObjectsFromAssets(cpubuffers,cpubuffers+MAX_DATA_BUFFERS);

	asset::SBufferBinding<video::IGPUBuffer> bindings[MAX_DATA_BUFFERS];
	for (auto i = 0; i < MAX_ATTR_BUF_BINDING_COUNT; i++)
	{
		auto buffPair = gpubuffers->operator[](i);
		if (!buffPair)
			continue;
		bindings[i].offset = buffPair->getOffset();
		bindings[i].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
	}
	auto buffPair = gpubuffers->operator[](MAX_ATTR_BUF_BINDING_COUNT);
	if (buffPair)
	{
		bindings[MAX_ATTR_BUF_BINDING_COUNT].offset = buffPair->getOffset();
		bindings[MAX_ATTR_BUF_BINDING_COUNT].buffer = core::smart_refctd_ptr<video::IGPUBuffer>(buffPair->getBuffer());
	}

	auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(core::smart_refctd_ptr(pipeline), nullptr, bindings, std::move(bindings[MAX_ATTR_BUF_BINDING_COUNT]));
	{
		mb->setIndexType(cubeGeom.indexType);
		mb->setIndexCount(cubeGeom.indexCount);
		mb->setBoundingBox(cubeGeom.bbox);
	}

	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();
		core::matrix4SIMD mvp = camera->getConcatenatedMatrix();

		driver->bindGraphicsPipeline(pipeline.get());
		driver->pushConstants(pipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
		driver->drawMeshBuffer(mb.get());

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

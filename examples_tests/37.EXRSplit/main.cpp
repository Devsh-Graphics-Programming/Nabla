#define _IRR_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;

#include "irr/irrpack.h"
struct VertexStruct
{
								/// every member needs to be at location aligned to its type size for GLSL
	float Pos[3];				/// uses float hence need 4 byte alignment
	uint8_t Col[2];				/// same logic needs 1 byte alignment
	uint8_t uselessPadding[2];	/// so if there is a member with 4 byte alignment then whole struct needs 4 byte align, so pad it
} PACK_STRUCT;

struct ImageData
{
	ImageData(core::smart_refctd_ptr<asset::IAsset> _image) : asset(std::move(_image))
	{
		image_raw = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset).get();
	}

	core::smart_refctd_ptr<asset::IAsset> asset;
	asset::ICPUImage* image_raw;
} PACK_STRUCT;
#include "irr/irrunpack.h"

const char* vertexSource = R"===(
#version 430 core
layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol;

//#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;

layout(location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle

void main()
{
    gl_Position = PushConstants.modelViewProj*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
}
)===";

const char* fragmentSource = R"===(
#version 430 core

layout(location = 0) in vec4 Color; //per vertex output color, will be interpolated across the triangle

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color;
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

	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	auto driver = device->getVideoDriver();
	auto smgr = device->getSceneManager();
	auto am = device->getAssetManager();

	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.001f);

	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);

	smgr->setActiveCamera(camera);

	/// OpenEXR image data ///
	asset::IAssetLoader::SAssetLoadParams lp;
	auto image_bundle = am->getAsset("../../media/daily_pt_1.exr", lp);
	assert(!image_bundle.isEmpty());

	core::vector<ImageData> data;
	for (auto i = 0ul; i < image_bundle.getSize(); ++i)
		data.push_back(image_bundle.getContents().first[i]);

	/// vertex and index data ///
	VertexStruct vertices[4];
	vertices[0] = VertexStruct{ {-1.f,-1.f,-1.f},{  0,  0} };
	vertices[1] = VertexStruct{ { 1.f,-1.f,-1.f},{127,  0} };
	vertices[2] = VertexStruct{ {-1.f, 1.f,-1.f},{255,  0} };
	vertices[3] = VertexStruct{ { 1.f, 1.f,-1.f},{  0,127} };

	uint16_t indices_indexed16[] =
	{
		0,1,2,1,2,3
	};

	/// some usefull buffer data and pipeline ///
	auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
	core::smart_refctd_ptr<video::IGPUBuffer> upStreamRef(upStreamBuff->getBuffer());

	const void* dataToPlace[2] = { vertices,indices_indexed16 };
	uint32_t offsets[2] = { video::StreamingTransientDataBufferMT<>::invalid_address,video::StreamingTransientDataBufferMT<>::invalid_address };
	uint32_t alignments[2] = { sizeof(decltype(vertices[0u])),sizeof(decltype(indices_indexed16[0u])) };
	uint32_t sizes[2] = { sizeof(vertices),sizeof(indices_indexed16) };
	upStreamBuff->multi_place(2u, (const void* const*)dataToPlace, (uint32_t*)offsets, (uint32_t*)sizes, (uint32_t*)alignments);
	if (upStreamBuff->needsManualFlushOrInvalidate())
	{
		auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
		driver->flushMappedMemoryRanges({ video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[0],sizes[0]),video::IDriverMemoryAllocation::MappedMemoryRange(upStreamMem,offsets[1],sizes[1]) });
	}

	asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD) };

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
	auto shadersPtr = reinterpret_cast<video::IGPUSpecializedShader**>(shaders);

	asset::SVertexInputParams inputParams;
	inputParams.enabledAttribFlags = 0b11u;
	inputParams.enabledBindingFlags = 0b1u;
	inputParams.attributes[0].binding = 0u;
	inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
	inputParams.attributes[0].relativeOffset = offsetof(VertexStruct, Pos[0]);
	inputParams.attributes[1].binding = 0u;
	inputParams.attributes[1].format = asset::EF_R8G8_UNORM;
	inputParams.attributes[1].relativeOffset = offsetof(VertexStruct, Col[0]);
	inputParams.bindings[0].stride = sizeof(VertexStruct);
	inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

	asset::SBlendParams blendParams; 
	asset::SPrimitiveAssemblyParams assemblyParams = { asset::EPT_TRIANGLE_LIST,false,1u };
	asset::SStencilOpParams defaultStencil;
	asset::SRasterizationParams rasterParams;

	rasterParams.faceCullingMode = asset::EFCM_NONE;
	auto pipeline = driver->createGPURenderpassIndependentPipeline(nullptr, driver->createGPUPipelineLayout(range, range + 1u, nullptr, nullptr, nullptr, nullptr),
		shadersPtr, shadersPtr + sizeof(shaders) / sizeof(core::smart_refctd_ptr<video::IGPUSpecializedShader>),
		inputParams, blendParams, assemblyParams, rasterParams);

	asset::SBufferBinding<video::IGPUBuffer> bindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = { offsets[0],upStreamRef };
	auto mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{offsets[1], upStreamRef});
	{
		mb->setIndexType(asset::EIT_16BIT);
		mb->setIndexCount(2 * 3);
	}

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 255, 255, 255));
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();		

		const auto viewProjection = camera->getConcatenatedMatrix();
		irr::core::vectorSIMDf shift;
		for (auto image : data)
		{
			core::matrix3x4SIMD modelMatrix;
			modelMatrix.setTranslation(irr::core::vectorSIMDf(shift.X, 0, 0, 0));
			shift.X += 5;

			core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjection, modelMatrix);

			driver->bindGraphicsPipeline(mb->getPipeline());
			driver->pushConstants(mb->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			driver->drawMeshBuffer(mb.get());
		}

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

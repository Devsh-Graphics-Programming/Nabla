// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

using namespace nbl;
using namespace core;


#include "nbl/nblpack.h"
struct VertexStruct
{
    /// every member needs to be at location aligned to its type size for GLSL
    float Pos[3]; /// uses float hence need 4 byte alignment
    uint8_t Col[2]; /// same logic needs 1 byte alignment
    uint8_t uselessPadding[2]; /// so if there is a member with 4 byte alignment then whole struct needs 4 byte align, so pad it
} PACK_STRUCT;
#include "nbl/nblunpack.h"

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

	//
	asset::SBufferBinding<asset::ICPUBuffer> inverseBindPoses,jointAABBs;
	core::smart_refctd_ptr<asset::ICPUSkeleton> skeleton;
	{
		constexpr uint32_t kJointCount = 2u;
		asset::SBufferBinding<asset::ICPUBuffer> parentIDs,defaultTransforms,inverseBindPoses,jointAABBs;
		{
			parentIDs.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(asset::ICPUSkeleton)*kJointCount);
			{
				asset::ICPUSkeleton::joint_id_t parentJointIDs[] = { asset::ICPUSkeleton::invalid_joint_id,0u };
				memcpy(parentIDs.buffer->getPointer(),parentJointIDs,sizeof(parentJointIDs));
			}
			defaultTransforms.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(matrix3x4SIMD)*kJointCount);
			inverseBindPoses.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(matrix3x4SIMD)*kJointCount);
			{
				auto* dftTransforms = reinterpret_cast<matrix3x4SIMD*>(defaultTransforms.buffer->getPointer());
				auto* invBindPoses = reinterpret_cast<matrix3x4SIMD*>(inverseBindPoses.buffer->getPointer());
				for (auto i=0u; i<kJointCount; i++)
				{
					dftTransforms[i] = matrix3x4SIMD();
					dftTransforms[i].setTranslation(core::vectorSIMDf(0.f,float(i)*2.f-1.f,0.f));
					dftTransforms[i].getInverse(invBindPoses[i]);
				}
			}
			jointAABBs.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(aabbox3df)*kJointCount);
		}
		const char* jointNames[] = {"root","bendy"};
		skeleton = core::make_smart_refctd_ptr<asset::ICPUSkeleton>(std::move(parentIDs),std::move(defaultTransforms),&jointNames[0],&jointNames[0]+kJointCount);
	}
	core::smart_refctd_ptr<video::IGPUAnimationLibrary> gpuanimations;
	{
		constexpr uint32_t kKeyframeCount = 16u;
		constexpr uint32_t kAnimationCount = 3u;
		core::smart_refctd_ptr<asset::ICPUAnimationLibrary> animations;
		{
			asset::SBufferBinding<asset::ICPUBuffer> keyframes = {0ull,core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(asset::ICPUAnimationLibrary::Keyframe)*kKeyframeCount)};
			asset::SBufferBinding<asset::ICPUBuffer> timestamps = {0ull,core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(asset::ICPUAnimationLibrary::timestamp_t)*kKeyframeCount)};
			asset::SBufferRange<asset::ICPUBuffer> namedAnims;
			namedAnims.offset = 0ull;
			namedAnims.size = sizeof(asset::ICPUAnimationLibrary::Animation)*kAnimationCount;
			namedAnims.buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(namedAnims.size);
			animations = core::make_smart_refctd_ptr<asset::ICPUAnimationLibrary>(std::move(keyframes),std::move(timestamps),kKeyframeCount,std::move(namedAnims));
		}
		{
			const uint32_t animationOffsets[] = { 0u,1u,2u };
			const asset::ICPUAnimationLibrary::Animation anims[] = {
				{0u,kKeyframeCount,asset::ICPUAnimationLibrary::Animation::EIM_NEAREST},
				{0u,kKeyframeCount,asset::ICPUAnimationLibrary::Animation::EIM_LINEAR},
				{0u,kKeyframeCount,asset::ICPUAnimationLibrary::Animation::EIM_CUBIC},
			};
			for (auto i=0u; i<kAnimationCount; i++)
				animations->getAnimation(animationOffsets[i]) = anims[i];
			const char* animationNames[] = { "moveNearest","moveLinear","moveCubic" };
			animations->addAnimationNames(animationNames,animationNames+kAnimationCount,animationOffsets);
			
			for (auto i=0u; i<kAnimationCount; i++)
			{
				assert(animations->getAnimationOffsetFromName(animationNames[i]) == animationOffsets[i]);
			}
		}
		gpuanimations = driver->getGPUObjectsFromAssets<asset::ICPUAnimationLibrary>(&animations,&animations+1u)->begin()[0];
	}

	//
	core::smart_refctd_ptr<video::IGPUMeshBuffer> gpumb;
    {
        VertexStruct vertices[8];
        vertices[0] = VertexStruct{{-1.f,-1.f,-1.f},{  0,  0}};
        vertices[1] = VertexStruct{{ 1.f,-1.f,-1.f},{127,  0}};
        vertices[2] = VertexStruct{{-1.f, 1.f,-1.f},{255,  0}};
        vertices[3] = VertexStruct{{ 1.f, 1.f,-1.f},{  0,127}};
        vertices[4] = VertexStruct{{-1.f,-1.f, 1.f},{127,127}};
        vertices[5] = VertexStruct{{ 1.f,-1.f, 1.f},{255,127}};
        vertices[6] = VertexStruct{{-1.f, 1.f, 1.f},{  0,255}};
        vertices[7] = VertexStruct{{ 1.f, 1.f, 1.f},{127,255}};
		asset::SBufferBinding<asset::ICPUBuffer> bindings[asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
		bindings[0u] = {0u,core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(sizeof(vertices),vertices,core::adopt_memory)};

        uint16_t indices_indexed16[] =
        {
            0,1,2,1,2,3,
            4,5,6,5,6,7,
            0,1,4,1,4,5,
            2,3,6,3,6,7,
            0,2,4,2,4,6,
            1,3,5,3,5,7
        };
		asset::SBufferBinding<asset::ICPUBuffer> indexBinding{0u,core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(sizeof(indices_indexed16),indices_indexed16,core::adopt_memory)};
			
		asset::SPushConstantRange range[1] = {asset::ISpecializedShader::ESS_VERTEX,0u,sizeof(core::matrix4SIMD)};

		auto createSpecializedShaderFromSource = [=](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage)
		{
			auto spirv = device->getAssetManager()->getGLSLCompiler()->createSPIRVFromGLSL(source, stage, "main", "runtimeID");
			return core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(spirv),asset::ICPUSpecializedShader::SInfo{ nullptr,nullptr,"main",stage });
		};
		// origFilepath is only relevant when you have filesystem #includes in your shader
		auto createSpecializedShaderFromSourceWithIncludes = [&](const char* source, asset::ISpecializedShader::E_SHADER_STAGE stage, const char* origFilepath)
		{
			auto resolved_includes = device->getAssetManager()->getGLSLCompiler()->resolveIncludeDirectives(source, stage, origFilepath);
			return createSpecializedShaderFromSource(reinterpret_cast<const char*>(resolved_includes->getSPVorGLSL()->getPointer()), stage);
		};
		constexpr uint32_t kShaderCount = 2u;
		core::smart_refctd_ptr<asset::ICPUSpecializedShader> shaders[kShaderCount] =
		{
			createSpecializedShaderFromSourceWithIncludes(vertexSource,asset::ISpecializedShader::ESS_VERTEX, "shader.vert"),
			createSpecializedShaderFromSource(fragmentSource,asset::ISpecializedShader::ESS_FRAGMENT)
		};
		auto shadersPtr = reinterpret_cast<asset::ICPUSpecializedShader**>(shaders);
			
		asset::SVertexInputParams inputParams;
		inputParams.enabledAttribFlags = 0b11u;
		inputParams.enabledBindingFlags = 0b1u;
		inputParams.attributes[0].binding = 0u;
		inputParams.attributes[0].format = asset::EF_R32G32B32_SFLOAT;
		inputParams.attributes[0].relativeOffset = offsetof(VertexStruct,Pos[0]);
		inputParams.attributes[1].binding = 0u;
		inputParams.attributes[1].format = asset::EF_R8G8_UNORM;
		inputParams.attributes[1].relativeOffset = offsetof(VertexStruct,Col[0]);
		inputParams.bindings[0].stride = sizeof(VertexStruct);
		inputParams.bindings[0].inputRate = asset::EVIR_PER_VERTEX;

		asset::SBlendParams blendParams; // defaults are sane

		asset::SPrimitiveAssemblyParams assemblyParams = {asset::EPT_TRIANGLE_LIST,false,1u};

		asset::SStencilOpParams defaultStencil;
		asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = asset::EFCM_NONE;
		auto pipeline = core::make_smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>(
			core::make_smart_refctd_ptr<asset::ICPUPipelineLayout>(range,range+1u,nullptr,nullptr,nullptr,nullptr),
			shadersPtr,shadersPtr+kShaderCount,
			inputParams,blendParams,assemblyParams,rasterParams
		);

		auto mb = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>(std::move(pipeline),nullptr,bindings,std::move(indexBinding));
		{
			mb->setSkin(std::move(inverseBindPoses),std::move(jointAABBs),std::move(skeleton),1u);
			mb->setIndexType(asset::EIT_16BIT);
			mb->setIndexCount(2*3*6);
		}
		gpumb = driver->getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(&mb,&mb+1u)->begin()[0];
	}


	//! we want to move around the scene and view it from different angles
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.001f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10.0f);

    smgr->setActiveCamera(camera);

	uint64_t lastFPSTime = 0;
	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );
        //! This animates (moves) the camera and sets the transforms
		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();
		core::matrix4SIMD mvp = camera->getConcatenatedMatrix();

        //! Stress test for memleaks aside from demo how to create meshes that live on the GPU RAM
        {
			driver->bindGraphicsPipeline(gpumb->getPipeline());
			driver->pushConstants(gpumb->getPipeline()->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			driver->drawMeshBuffer(gpumb.get());
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

	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		//ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	return 0;
}

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <irrlicht.h>

#include "nbl/ext/DebugDraw/CDraw3DLine.h"

#include "../common/QToQuitEventReceiver.h"


using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

vector3df camPos;
vector<vectorSIMDf> controlPts;
ISpline* spline = NULL;


template<typename IteratorType>
vector<vectorSIMDf> preprocessBSplineControlPoints(const IteratorType& _begin, const IteratorType& _end, bool loop=false, float relativeLen=0.25f)
{
	//assert(curveRelativeLen < 0.5f);
	auto ptCount = std::distance(_begin, _end);
	if (ptCount < 2u)
		return {};

	ptCount *= 2u;
	if (!loop)
		ptCount -= 2;
	core::vector<vectorSIMDf> retval(ptCount);
	auto out = retval.begin();

	auto it = _begin;
	auto _back = _end - 1;
	vectorSIMDf prev;
	if (loop)
		prev = *_back;
	else
	{
		prev = *_begin;
		*(out++) = *(it++);
	}

	auto addDoublePoint = [&](const vectorSIMDf& original, vectorSIMDf next)
	{
		auto deltaPrev = original - prev;
		auto deltaNext = next - original;
		float currentRelativeLen = core::min(core::length(deltaPrev).x, core::length(deltaNext).x) * relativeLen;
		auto tangent = core::normalize(next - prev) * currentRelativeLen;
		*(out++) = original - tangent;
		*(out++) = original + tangent;
	};
	while (it != _back)
	{
		const auto& orig = *(it++);
		addDoublePoint(orig, *it);
		prev = orig;
	}

	if (loop)
	{
		addDoublePoint(*_back, *_begin);
	}
	else
		*(out++) = *_back;

	return retval;
}


core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> lines;

class MyEventReceiver : public QToQuitEventReceiver
{
public:

	MyEventReceiver() : wasLeftPressedBefore(false)
	{
	}

	bool OnEvent(const SEvent& event)
	{
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
			auto useNewSpline = [&](auto replacementCreateFunc) -> bool
			{
				if (spline)
					delete spline;
				spline = nullptr;

				if (controlPts.size())
				{
					spline = replacementCreateFunc();

					printf("Total Len %f\n", spline->getSplineLength());
					for (size_t i = 0; i < spline->getSegmentCount(); i++)
						printf("Seg: %d \t\t %f\n", i, spline->getSegmentLength(i));

					lines.clear();
					auto wholeLen = spline->getSplineLength();
					constexpr auto limit = 1000u;
					for (auto i = 0u; i < limit; ++i)
					{
						auto computeLinePt = [&](float percentage)
						{
							ext::DebugDraw::S3DLineVertex vx = { {0.f,0.f,0.f},{1.f,0.f,0.f,1.f} };

							float segDist = percentage * wholeLen;
							uint32_t segID = 0u;

							core::vectorSIMDf pos;
							spline->getPos(pos, segDist, segID);
							memcpy(vx.Position, pos.pointer, sizeof(float) * 3u);
							return vx;
						};
						lines.emplace_back(computeLinePt((float(i)+0.1) / limit), computeLinePt((float(i)+0.9) / limit));
					}
					return true;
				}
				return false;
			};

			auto createLinear = []() {return new CLinearSpline(controlPts.data(), controlPts.size()); };
			auto createLinearLoop = []() {return new CLinearSpline(controlPts.data(), controlPts.size(), true); };
			auto createBSpline = []()
			{
				auto prep = preprocessBSplineControlPoints(controlPts.cbegin(), controlPts.cend());
				return new irr::core::CQuadraticBSpline(prep.data(), prep.size());
			};
			auto createBSplineLoop = []()
			{
				auto prep = preprocessBSplineControlPoints(controlPts.cbegin(), controlPts.cend(), true);
				return new irr::core::CQuadraticBSpline(prep.data(), prep.size(), true); //make it a loop
			};
            switch (event.KeyInput.Key)
            {
                case irr::KEY_KEY_Q: // switch wire frame mode
					return QToQuitEventReceiver::OnEvent(event);
                    break;
                case KEY_KEY_T:
                    {
						if (useNewSpline(createLinear))
	                        return true;
                    }
                    break;
                case KEY_KEY_Y:
                    {
						if (useNewSpline(createLinearLoop))
						return true;
                    }
                    break;
                case KEY_KEY_U:
                    {
						if (useNewSpline(createBSpline))
							return true;
                    }
                    break;
                case KEY_KEY_I:
                    {
						if (useNewSpline(createBSplineLoop))
							return true;
                    }
                case KEY_KEY_C:
                    {
                        controlPts.clear();
                        return true;
                    }
                    break;
                default:
                    break;
            }
        }
        else if (event.EventType == EET_MOUSE_INPUT_EVENT)
        {
            bool pressed = event.MouseInput.isLeftPressed();
            if (pressed && !wasLeftPressedBefore)
            {
                controlPts.push_back(core::vectorSIMDf(camPos.X,camPos.Y,camPos.Z));
            }
            wasLeftPressedBefore = pressed;
        }

		return false;
	}

private:
    bool wasLeftPressedBefore;
};



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

	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	auto assMgr = device->getAssetManager();
	auto driver = device->getVideoDriver();

	auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(driver);


	auto smgr = device->getSceneManager();
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);

	constexpr uint32_t PER_MATERIAL_DESC_SET = 2u; //don't touch this, will go into scene namespace

	auto layout = assMgr->findGPUObject("irr/builtin/materials/lambertian/singletexture/pipelinelayout",IAsset::ET_PIPELINE_LAYOUT);
	auto shaders = assMgr->findGPUObject("irr/builtin/materials/lambertian/singletexture/specializedshader",IAsset::ET_SPECIALIZED_SHADER);
	if (!layout->size() || !shaders->size())
		return 2;

	auto pLayout = core::smart_refctd_ptr_static_cast<IGPUPipelineLayout>(layout->front());
	//! Test Creation Of Builtin
	auto cubeParams = assMgr->getGeometryCreator()->createCubeMesh(core::vector3df(1.f));

	IGPUSpecializedShader** pShaders = reinterpret_cast<IGPUSpecializedShader**>(shaders->data()); // hack but works, shorter code

	SBlendParams blendParams; // gets sane defaults

	SRasterizationParams rasterParams; // default is depth testing
	auto pipeline = driver->createGPURenderpassIndependentPipeline(	nullptr,core::smart_refctd_ptr(pLayout),pShaders,pShaders+shaders->size(),
																	cubeParams.inputParams,blendParams,cubeParams.assemblyParams,rasterParams);

	auto sampler = assMgr->findGPUObject("irr/builtin/samplers/default",IAsset::ET_SAMPLER);

	struct alignas(64) CustomObject
	{
		core::smart_refctd_ptr<IGPUMeshBuffer> mb = nullptr;
		core::matrix3x4SIMD transform = core::matrix3x4SIMD();
	};
	
	auto createCube = [=](	const char* jpegPath, const core::quaternion& rotation,
							const core::vector3df_SIMD& scale=core::vector3df_SIMD(1.f,1.f,1.f)) -> CustomObject
	{
		asset::IAssetLoader::SAssetLoadParams lparams;
		auto contents = assMgr->getAsset(jpegPath, lparams).getContents();
		if (contents.begin() == contents.end())
			return {};

		auto cpuImg = static_cast<asset::ICPUImage*>(contents.begin()->get());
		auto gpuImages = driver->getGPUObjectsFromAssets(&cpuImg,&cpuImg+1u);
		if (!gpuImages->size())
			return {};

		auto gpuImage = core::smart_refctd_ptr_static_cast<IGPUImage>(gpuImages->front());
		const auto& imgCreationParams = gpuImage->getCreationParameters();
		auto imgView = driver->createGPUImageView({{},std::move(gpuImage),IGPUImageView::ET_2D,imgCreationParams.format,{},{{},0u,imgCreationParams.mipLevels,0u,1u}});

		core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
		{
			std::string path("irr/builtin/materials/lambertian/singletexture/descriptorsetlayout/");
			path += std::to_string(PER_MATERIAL_DESC_SET);
			auto layout = assMgr->findGPUObject(path.c_str(),IAsset::ET_PIPELINE_LAYOUT);
			assert(layout->front().get()==pLayout->getDescriptorSetLayout(PER_MATERIAL_DESC_SET));
			dsLayout = core::smart_refctd_ptr_dynamic_cast<IGPUDescriptorSetLayout>(layout->front());
		}
		auto ds = driver->createGPUDescriptorSet(std::move(dsLayout));

		IGPUDescriptorSet::SDescriptorInfo pInfo[1];
		pInfo[0].desc = std::move(imgView);
		pInfo[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		pInfo[0].image.sampler = core::smart_refctd_ptr_dynamic_cast<IGPUSampler>(sampler->front());
		IGPUDescriptorSet::SWriteDescriptorSet pWrites[1] = {{ds.get(),0u,0u,1u,EDT_COMBINED_IMAGE_SAMPLER,pInfo}};
		driver->updateDescriptorSets(1u,pWrites,0u,nullptr);

		auto getGPUBufferBinding = [&](const SBufferBinding<ICPUBuffer>& other)
		{
			return SBufferBinding<IGPUBuffer>{other.offset,core::smart_refctd_ptr_dynamic_cast<IGPUBuffer>(assMgr->findGPUObject(other.buffer.get()))};
		};
		SBufferBinding<IGPUBuffer> bindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
		for (size_t i=0ull; i<IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
			bindings[i] = getGPUBufferBinding(cubeParams.bindings[i]);
		CustomObject retval;
		retval.mb = core::make_smart_refctd_ptr<IGPUMeshBuffer>(core::smart_refctd_ptr(pipeline),std::move(ds),bindings,getGPUBufferBinding(cubeParams.indexBuffer));
		retval.transform.setRotation(rotation);
		retval.transform.concatenateAfter(core::matrix3x4SIMD().setScale(scale));
		return retval;
	};

	auto animatedCube = createCube("../../media/bumpmap.jpg",core::quaternion::fromEuler(core::radians(core::vector3df_SIMD(45.f,20.f,15.f))));
	auto centerCube = createCube("../../media/wall.jpg",core::quaternion(),core::vector3df_SIMD(2.f));

    float cubeDistance = 0.f;
    float cubeParameterHint = 0.f;
    uint32_t cubeSegment = 0;
    #define kCircleControlPts 4
    for (size_t i=0; i<kCircleControlPts; i++)
    {
        float x = float(i)*core::PI<float>()*2.f/float(kCircleControlPts);
        vectorSIMDf pos(sin(x),0,-cos(x)); pos *= 4.f;
        controlPts.push_back(pos);
    }

	uint64_t lastFPSTime = 0;

	auto timer = device->getTimer();
	auto lastTime = timer->getTime();

	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

		auto nowtime = timer->getTime();
		auto timeDelta = nowtime - lastTime;
		lastTime = nowtime;

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(nowtime).count());
		camPos = camera->getAbsolutePosition();

		camera->render();
		core::matrix4SIMD vp = camera->getConcatenatedMatrix();

		driver->bindGraphicsPipeline(pipeline.get());
		auto drawCube = [&](const CustomObject& cube) -> void
		{
			auto ds = cube.mb->getAttachedDescriptorSet();
			driver->bindDescriptorSets(EPBP_GRAPHICS,pLayout.get(),PER_MATERIAL_DESC_SET,1u,&ds,nullptr);
			core::matrix4SIMD mvp = core::concatenateBFollowedByA(vp, cube.transform);
			driver->pushConstants(pipeline->getLayout(), ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), mvp.pointer());
			driver->drawMeshBuffer(cube.mb.get());
		};
		drawCube(centerCube);

		if (spline)
        {
            vectorSIMDf newPos;
            cubeDistance += float(std::chrono::duration_cast<std::chrono::milliseconds>(timeDelta).count())*0.001f; //1 unit per second
            cubeSegment = spline->getPos(newPos,cubeDistance,cubeSegment,&cubeParameterHint);
            if (cubeSegment>=0xdeadbeefu) //reached end of non-loop, or spline changed
            {
                cubeDistance = 0;
                cubeParameterHint = 0;
                cubeSegment = 0;
                cubeSegment = spline->getPos(newPos,cubeDistance,cubeSegment,&cubeParameterHint);
            }

            vectorSIMDf forwardDir; 
			bool success = spline->getUnnormDirection_fromParameter(forwardDir,cubeSegment,cubeParameterHint);
            assert(success); //must be TRUE
            forwardDir = normalize(forwardDir); //must normalize after
            vectorSIMDf sideDir = normalize(cross(forwardDir,vectorSIMDf(0,1,0))); // predefined up vector
            vectorSIMDf pseudoUp = cross(sideDir,forwardDir);



            matrix4x3 mat;
            mat.getColumn(0) = reinterpret_cast<vector3df&>(forwardDir);
            mat.getColumn(1) = reinterpret_cast<vector3df&>(pseudoUp);
            mat.getColumn(2) = reinterpret_cast<vector3df&>(sideDir);
            mat.setTranslation(reinterpret_cast<const vector3df&>(newPos));
			animatedCube.transform.set(mat);
        }
		drawCube(animatedCube);

		draw3DLine->draw(vp,lines);

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:";
			str << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

    if (spline)
        delete spline;

	return 0;
}

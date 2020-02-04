#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../ext/DebugDraw/CDraw3DLine.h"
#include "../../ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"


using namespace irr;
using namespace core;

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
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	device->getCursorControl()->setVisible(false);

	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


	video::IVideoDriver* driver = device->getVideoDriver();

	auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(driver);


	scene::ISceneManager* smgr = device->getSceneManager();
	scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);

	//! Test Creation Of Builtin
	auto* cube = smgr->addCubeSceneNode(1.f, 0, -1);
	cube->setRotation(core::vector3df(45, 20, 15));

	auto assetMgr = device->getAssetManager();
	{
		asset::IAssetLoader::SAssetLoadParams lparams;
		asset::ICPUTexture* cputextures[]{
			static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_dn.jpg", lparams).getContents().first->get()),
			static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/wall.jpg", lparams).getContents().first->get())
		};
		auto gputextures = driver->getGPUObjectsFromAssets(cputextures, cputextures + 2);

		cube->getMesh()->getMeshBuffer(0)->getMaterial().setTexture(0, std::move(gputextures->operator[](0u)));

		auto* billboard = smgr->addCubeSceneNode(2.f, 0, -1, core::vector3df(0, 0, 0));
		billboard->getMesh()->getMeshBuffer(0)->getMaterial().setTexture(0, std::move(gputextures->operator[](1u)));
	}

    float cubeDistance = 0.f;
    float cubeParameterHint = 0.f;
    uint32_t cubeSegment = 0;
    #define kCircleControlPts 4
    for (size_t i=0; i<kCircleControlPts; i++)
    {
        float x = float(i)*core::PI<float>()*2.f/float(kCircleControlPts);
        vectorSIMDf pos(sin(x),0,-cos(x)); pos *= 4.f;
        controlPts.push_back(pos);
		smgr->addCubeSceneNode(0.5f, 0, -1, pos.getAsVector3df());
    }

	uint64_t lastFPSTime = 0;

	uint64_t lastTime = device->getTimer()->getRealTime();
    uint64_t timeDelta = 0;

	while(device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        uint64_t nowTime = device->getTimer()->getRealTime();
        timeDelta = nowTime-lastTime;
        lastTime = nowTime;

		if (spline)
        {
            vectorSIMDf newPos;
            cubeDistance += float(timeDelta)*0.001f; //1 unit per second
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
            cube->setRelativeTransformationMatrix(mat);
        }

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();
        camPos = camera->getAbsolutePosition();

		draw3DLine->draw(lines);

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

	//device->drop();

	return 0;
}

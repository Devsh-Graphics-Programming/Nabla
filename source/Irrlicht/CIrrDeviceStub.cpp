// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CIrrDeviceStub.h"
#include "CSceneManager.h"
#include "IEventReceiver.h"
#include "IFileSystem.h"
#include "os.h"
#include "IrrCompileConfig.h"
#include "CLogger.h"
#include "irr/asset/CIncludeHandler.h"
#include "irr/asset/CGLSLScanBuiltinIncludeLoader.h"
#include "irr/asset/CGLSLSkinningBuiltinIncludeLoader.h"

namespace irr
{
//! constructor
CIrrDeviceStub::CIrrDeviceStub(const SIrrlichtCreationParameters& params)
: IrrlichtDevice(), VideoDriver(0), SceneManager(0),
	Timer(0), CursorControl(0), UserReceiver(params.EventReceiver),
	Logger(0), Operator(0), FileSystem(0),
	InputReceivingSceneManager(0), VideoModeList(0),
	CreationParams(params), Close(false)
{
	Timer = new ITimer();
	if (os::Printer::Logger)
	{
		os::Printer::Logger->grab();
		Logger = (CLogger*)os::Printer::Logger;
		Logger->setReceiver(UserReceiver);
	}
	else
	{
		Logger = new CLogger(UserReceiver);
		os::Printer::Logger = Logger;
	}
	Logger->setLogLevel(CreationParams.LoggingLevel);

	os::Printer::Logger = Logger;

	FileSystem = io::createFileSystem();
	VideoModeList = new video::CVideoModeList();

	core::stringc s = "Irrlicht Engine version ";
	s.append(getVersion());
	os::Printer::log(s.c_str(), ELL_INFORMATION);

	checkVersion(params.SDK_version_do_not_use);

    IncludeHandler = core::make_smart_refctd_ptr<asset::CIncludeHandler>(FileSystem);

    //add builtin loaders
    asset::IBuiltinIncludeLoader* builtinLdr = new asset::CGLSLScanBuiltinIncludeLoader();
    IncludeHandler->addBuiltinIncludeLoader(builtinLdr);
    builtinLdr->drop();
    builtinLdr = new asset::CGLSLSkinningBuiltinIncludeLoader();
    IncludeHandler->addBuiltinIncludeLoader(builtinLdr);
    builtinLdr->drop();
}


CIrrDeviceStub::~CIrrDeviceStub()
{
	VideoModeList->drop();
	FileSystem->drop();

	if (Operator)
		Operator->drop();

	if (Timer)
		Timer->drop();

	if (Logger->drop())
		os::Printer::Logger = 0;
}


void CIrrDeviceStub::createGUIAndScene()
{
	// create Scene manager
	SceneManager = new scene::CSceneManager(this, VideoDriver, Timer, FileSystem, CursorControl);

	setEventReceiver(UserReceiver);
}


//! returns the video driver
video::IVideoDriver* CIrrDeviceStub::getVideoDriver()
{
	return VideoDriver;
}



//! return file system
io::IFileSystem* CIrrDeviceStub::getFileSystem()
{
	return FileSystem;
}




//! returns the scene manager
scene::ISceneManager* CIrrDeviceStub::getSceneManager()
{
	return SceneManager;
}


//! \return Returns a pointer to the ITimer object. With it the
//! current Time can be received.
ITimer* CIrrDeviceStub::getTimer()
{
	return Timer;
}


//! Returns the version of the engine.
const char* CIrrDeviceStub::getVersion() const
{
	return IRRLICHTBAW_SDK_VERSION;
}

//! \return Returns a pointer to the mouse cursor control interface.
gui::ICursorControl* CIrrDeviceStub::getCursorControl()
{
	return CursorControl;
}


//! \return Returns a pointer to a list with all video modes supported
//! by the gfx adapter.
video::IVideoModeList* CIrrDeviceStub::getVideoModeList()
{
	return VideoModeList;
}


//! checks version of sdk and prints warning if there might be a problem
bool CIrrDeviceStub::checkVersion(const char* version)
{
	if (strcmp(getVersion(), version))
	{
		core::stringc w;
		w = "Warning: The library version of the Irrlicht Engine (";
		w += getVersion();
		w += ") does not match the version the application was compiled with (";
		w += version;
		w += "). This may cause problems.";
		os::Printer::log(w.c_str(), ELL_WARNING);
		return false;
	}

	return true;
}


//! Compares to the last call of this function to return double and triple clicks.
uint32_t CIrrDeviceStub::checkSuccessiveClicks(int32_t mouseX, int32_t mouseY, EMOUSE_INPUT_EVENT inputEvent )
{
	const int32_t MAX_MOUSEMOVE = 3;

	uint32_t clickTime = getTimer()->getRealTime();

	if ( (clickTime-MouseMultiClicks.LastClickTime) < MouseMultiClicks.DoubleClickTime
		&& core::abs_(MouseMultiClicks.LastClick.X - mouseX ) <= MAX_MOUSEMOVE
		&& core::abs_(MouseMultiClicks.LastClick.Y - mouseY ) <= MAX_MOUSEMOVE
		&& MouseMultiClicks.CountSuccessiveClicks < 3
		&& MouseMultiClicks.LastMouseInputEvent == inputEvent
	   )
	{
		++MouseMultiClicks.CountSuccessiveClicks;
	}
	else
	{
		MouseMultiClicks.CountSuccessiveClicks = 1;
	}

	MouseMultiClicks.LastMouseInputEvent = inputEvent;
	MouseMultiClicks.LastClickTime = clickTime;
	MouseMultiClicks.LastClick.X = mouseX;
	MouseMultiClicks.LastClick.Y = mouseY;

	return MouseMultiClicks.CountSuccessiveClicks;
}


//! send the event to the right receiver
bool CIrrDeviceStub::postEventFromUser(const SEvent& event)
{
	bool absorbed = false;

	if (UserReceiver)
		absorbed = UserReceiver->OnEvent(event);

	scene::ISceneManager* inputReceiver = InputReceivingSceneManager ? InputReceivingSceneManager:SceneManager;

	if (!absorbed && inputReceiver)
		absorbed = inputReceiver->receiveIfEventReceiverDidNotAbsorb(event);

	return absorbed;
}


//! Sets a new event receiver to receive events
void CIrrDeviceStub::setEventReceiver(IEventReceiver* receiver)
{
	UserReceiver = receiver;
	Logger->setReceiver(receiver);
}


//! Returns poinhter to the current event receiver. Returns 0 if there is none.
IEventReceiver* CIrrDeviceStub::getEventReceiver()
{
	return UserReceiver;
}


//! \return Returns a pointer to the logger.
ILogger* CIrrDeviceStub::getLogger()
{
	return Logger;
}


//! Returns the operation system opertator object.
IOSOperator* CIrrDeviceStub::getOSOperator()
{
	return Operator;
}


//! Sets the input receiving scene manager.
void CIrrDeviceStub::setInputReceivingSceneManager(scene::ISceneManager* sceneManager)
{
    if (sceneManager)
        sceneManager->grab();
	if (InputReceivingSceneManager)
		InputReceivingSceneManager->drop();

	InputReceivingSceneManager = sceneManager;
}


//! Checks if the window is running in fullscreen mode
bool CIrrDeviceStub::isFullscreen() const
{
	return CreationParams.Fullscreen;
}


//! returns color format
asset::E_FORMAT CIrrDeviceStub::getColorFormat() const
{
	return asset::EF_B5G6R5_UNORM_PACK16;
}

//! No-op in this implementation
bool CIrrDeviceStub::activateJoysticks(core::vector<SJoystickInfo> & joystickInfo)
{
	return false;
}

/*!
*/
void CIrrDeviceStub::calculateGammaRamp ( uint16_t *ramp, float gamma, float relativebrightness, float relativecontrast )
{
	int32_t i;
	int32_t value;
	int32_t rbright = (int32_t) ( relativebrightness * (65535.f / 4 ) );
	float rcontrast = 1.f / (255.f - ( relativecontrast * 127.5f ) );

	gamma = gamma > 0.f ? 1.0f / gamma : 0.f;

	for ( i = 0; i < 256; ++i )
	{
		value = (int32_t)(pow( rcontrast * i, gamma)*65535.f + 0.5f );
		ramp[i] = (uint16_t) core::s32_clamp ( value + rbright, 0, 65535 );
	}

}

void CIrrDeviceStub::calculateGammaFromRamp ( float &gamma, const uint16_t *ramp )
{
	/* The following is adapted from a post by Garrett Bass on OpenGL
	Gamedev list, March 4, 2000.
	*/
	float sum = 0.0;
	int32_t i, count = 0;

	gamma = 1.0;
	for ( i = 1; i < 256; ++i ) {
		if ( (ramp[i] != 0) && (ramp[i] != 65535) ) {
			float B = (float)i / 256.f;
			float A = ramp[i] / 65535.f;
			sum += (float) ( logf(A) / logf(B) );
			count++;
		}
	}
	if ( count && sum ) {
		gamma = 1.0f / (sum / count);
	}

}


//! Set the maximal elapsed time between 2 clicks to generate doubleclicks for the mouse. It also affects tripleclick behavior.
void CIrrDeviceStub::setDoubleClickTime( uint32_t timeMs )
{
	MouseMultiClicks.DoubleClickTime = timeMs;
}

//! Get the maximal elapsed time between 2 clicks to generate double- and tripleclicks for the mouse.
uint32_t CIrrDeviceStub::getDoubleClickTime() const
{
	return MouseMultiClicks.DoubleClickTime;
}

//! Remove all messages pending in the system message loop
void CIrrDeviceStub::clearSystemMessages()
{
}



} // end namespace irr


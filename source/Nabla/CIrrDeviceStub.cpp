// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "nbl_os.h"
#include "nbl_CLogger.h"
#include "IFileSystem.h"
#include "CFileSystem.h"

#include "nbl/asset/utils/CIncludeHandler.h"

#include "IEventReceiver.h"

#include "CIrrDeviceStub.h"

#include "CSceneManager.h"

namespace nbl
{
	namespace video
	{
#ifdef _NBL_COMPILE_WITH_OPENGL_
		core::smart_refctd_ptr<IVideoDriver> createOpenGLDriver(const nbl::SIrrlichtCreationParameters& params,
			io::IFileSystem* io, CIrrDeviceStub* device, asset::IAssetManager* assmgr, const asset::IGLSLCompiler* glslcomp);
#endif
	}
} // end namespace nbl

namespace nbl
{
//! constructor
CIrrDeviceStub::CIrrDeviceStub(const SIrrlichtCreationParameters& params)
: IrrlichtDevice(), VideoDriver(0), SceneManager(0),
	Timer(0), CursorControl(0), UserReceiver(params.EventReceiver),
	Logger(0), Operator(0),
	InputReceivingSceneManager(0),
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

	FileSystem = core::make_smart_refctd_ptr<io::CFileSystem>(std::string(CreationParams.builtinResourceDirectoryPath));

	core::stringc s = "Nabla Engine version ";
	s.append(getVersion());
	os::Printer::log(s.c_str(), ELL_INFORMATION);

	checkVersion(params.SDK_version_do_not_use);

	// init EGL
	Display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	eglInitialize(Display, NULL, NULL);
}


CIrrDeviceStub::~CIrrDeviceStub()
{
	if (Operator)
		Operator->drop();

	if (Timer)
		Timer->drop();

	if (Logger->drop())
		os::Printer::Logger = 0;
}

//! create the driver
void CIrrDeviceStub::createDriver()
{
	switch (CreationParams.DriverType)
	{
	case video::EDT_OPENGL:

#ifdef _NBL_COMPILE_WITH_OPENGL_
		switchToFullscreen();

		VideoDriver = video::createOpenGLDriver(CreationParams, FileSystem.get(), this, getAssetManager(), getAssetManager()->getGLSLCompiler());
		if (!VideoDriver)
		{
			os::Printer::log("Could not create OpenGL driver.", ELL_ERROR);
		}
#else
		os::Printer::log("OpenGL driver was not compiled in.", ELL_ERROR);
#endif
		break;

	case video::EDT_NULL:
		// create null driver
		VideoDriver = video::createNullDriver(getAssetManager(), FileSystem.get(), CreationParams);
		break;

	default:
		os::Printer::log("Unable to create video driver of unknown type.", ELL_ERROR);
		break;
	}
}


void CIrrDeviceStub::createGUIAndScene()
{
	// create Scene manager
	SceneManager = new scene::CSceneManager(this, getVideoDriver(), Timer, getFileSystem(), CursorControl);

	setEventReceiver(UserReceiver);
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
	return NABLA_SDK_VERSION;
}

//! \return Returns a pointer to the mouse cursor control interface.
gui::ICursorControl* CIrrDeviceStub::getCursorControl()
{
	return CursorControl;
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
	const int32_t maxMOUSEMOVE = 3;

	uint32_t clickTime = getTimer()->getRealTime();

	if ( (clickTime-MouseMultiClicks.LastClickTime) < MouseMultiClicks.DoubleClickTime
		&& core::abs(MouseMultiClicks.LastClick.X - mouseX ) <= maxMOUSEMOVE
		&& core::abs(MouseMultiClicks.LastClick.Y - mouseY ) <= maxMOUSEMOVE
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



} // end namespace nbl


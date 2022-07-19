// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#include "CIrrDeviceLinux.h"

#include "CNullDriver.h"

#ifdef _NBL_COMPILE_WITH_X11_DEVICE_

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/utsname.h>
#include <time.h>
#include <sstream>
#include "IEventReceiver.h"
#include "ISceneManager.h"
#include "nbl_os.h"
#include "Keycodes.h"
#include "COSOperator.h"
#include "SIrrCreationParameters.h"
#include <X11/XKBlib.h>
#include <X11/Xatom.h>

#ifdef _NBL_LINUX_XCURSOR_
#include <X11/Xcursor/Xcursor.h>
#endif

#if defined _NBL_COMPILE_WITH_JOYSTICK_EVENTS_
#include <fcntl.h>
#include <unistd.h>

#ifdef __FreeBSD__
#include <sys/joystick.h>
#else

// linux/joystick.h includes linux/input.h, which #defines values for various KEY_FOO keys.
// These override the nbl::KEY_FOO equivalents, which stops key handling from working.
// As a workaround, defining _INPUT_H stops linux/input.h from being included; it
// doesn't actually seem to be necessary except to pull in sys/ioctl.h.
#define _INPUT_H
#include <sys/ioctl.h> // Would normally be included in linux/input.h
#include <linux/joystick.h>
#undef _INPUT_H
#endif

#endif // _NBL_COMPILE_WITH_JOYSTICK_EVENTS_

#ifdef _NBL_COMPILE_WITH_X11_

namespace nbl
{
	namespace video
	{
#ifdef _NBL_COMPILE_WITH_OPENGL_
		core::smart_refctd_ptr<IVideoDriver> createOpenGLDriver(const nbl::SIrrlichtCreationParameters& params,
			io::IFileSystem* io, CIrrDeviceStub* device, const asset::IGLSLCompiler* glslcomp);
#endif
	}
} // end namespace nbl
#endif // _NBL_COMPILE_WITH_X11_

namespace
{
	Atom X_ATOM_CLIPBOARD;
	Atom X_ATOM_TARGETS;
	Atom X_ATOM_UTF8_STRING;
	Atom X_ATOM_TEXT;
};

namespace nbl
{

const char* wmDeleteWindow = "WM_DELETE_WINDOW";

//! constructor
CIrrDeviceLinux::CIrrDeviceLinux(const SIrrlichtCreationParameters& param)
	: CIrrDeviceStub(param),
#ifdef _NBL_COMPILE_WITH_X11_
	Display(0), visual(0), screennr(0), window(0), StdHints(0),
	XInputMethod(0), XInputContext(0)
#endif
	Width(param.WindowSize.Width), Height(param.WindowSize.Height),
	WindowHasFocus(false), WindowMinimized(false),
	UseXVidMode(false), UseXRandR(false),
	ExternalWindow(false), AutorepeatSupport(0)
{
	#ifdef _NBL_DEBUG
	setDebugName("CIrrDeviceLinux");
	#endif

	// print version, distribution etc.
	// thx to LynxLuna for pointing me to the uname function
	core::stringc linuxversion;
	struct utsname LinuxInfo;
	uname(&LinuxInfo);

	linuxversion += LinuxInfo.sysname;
	linuxversion += " ";
	linuxversion += LinuxInfo.release;
	linuxversion += " ";
	linuxversion += LinuxInfo.version;
	linuxversion += " ";
	linuxversion += LinuxInfo.machine;

	Operator = new COSOperator(linuxversion, this);
	os::Printer::log(linuxversion.c_str(), ELL_INFORMATION);

	// create keymap
	createKeyMap();

	// create window
	if (CreationParams.DriverType != video::EDT_NULL)
	{
		// create the window, only if we do not use the null device
		if (!createWindow())
			return;
	}

	// create cursor control
	CursorControl = new CCursorControl(this, CreationParams.DriverType == video::EDT_NULL);

#ifdef _NBL_COMPILE_WITH_CUDA_
	cuda::CCUDAHandler::init();
#endif // _NBL_COMPILE_WITH_CUDA_
#ifdef _NBL_COMPILE_WITH_OPENCL_
    ocl::COpenCLHandler::enumeratePlatformsAndDevices();
#endif // _NBL_COMPILE_WITH_OPENCL_

	// create driver
	createDriver();

	if (!VideoDriver)
		return;

#ifdef _NBL_COMPILE_WITH_X11_
    createInputContext();
#endif // _NBL_COMPILE_WITH_X11_

	createGUIAndScene();
}


//! destructor
CIrrDeviceLinux::~CIrrDeviceLinux()
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (StdHints)
		XFree(StdHints);
	// Disable cursor (it is drop'ed in stub)
	if (CursorControl)
	{
		CursorControl->setVisible(false);
		static_cast<CCursorControl*>(CursorControl)->clearCursors();
		CursorControl->drop();
		CursorControl = NULL;
	}

	if (InputReceivingSceneManager)
		InputReceivingSceneManager->drop();

	// Must free OpenGL textures etc before destroying context, so can't wait for stub destructor
	if ( SceneManager )
	{
		SceneManager->drop();
		SceneManager = NULL;
	}
	if ( VideoDriver )
	{
		VideoDriver->drop();
		VideoDriver = NULL;
	}

	destroyInputContext();

	if (Display)
	{
		// Reset fullscreen resolution change
		switchToFullscreen(true);

		if (!ExternalWindow)
		{
			XDestroyWindow(Display,window);
			XCloseDisplay(Display);
		}
	}
	if (visual)
		XFree(visual);

#endif // #ifdef _NBL_COMPILE_WITH_X11_

#if defined(_NBL_COMPILE_WITH_JOYSTICK_EVENTS_)
	for (uint32_t joystick = 0; joystick < ActiveJoysticks.size(); ++joystick)
	{
		if (ActiveJoysticks[joystick].fd >= 0)
		{
			close(ActiveJoysticks[joystick].fd);
		}
	}
#endif
}

#ifdef _NBL_COMPILE_WITH_X11_
bool CIrrDeviceLinux::createInputContext()
{
	// One one side it would be nicer to let users do that - on the other hand
	// not setting the environment locale will not work when using i18n X11 functions.
	// So users would have to call it always or their input is broken badly.
	// We can restore immediately - so shouldn't mess with anything in users apps.
	std::string oldLocale(setlocale(LC_CTYPE, NULL));
	setlocale(LC_CTYPE, "");	// use environmenbt locale

	if ( !XSupportsLocale() )
	{
		os::Printer::log("Locale not supported. Falling back to non-i18n input.", ELL_WARNING);
		setlocale(LC_CTYPE, oldLocale.c_str());
		return false;
	}

	XInputMethod = XOpenIM(Display, NULL, NULL, NULL);
	if ( !XInputMethod )
	{
		setlocale(LC_CTYPE, oldLocale.c_str());
		os::Printer::log("XOpenIM failed to create an input method. Falling back to non-i18n input.", ELL_WARNING);
		return false;
	}

	XIMStyles *im_supported_styles;
	XGetIMValues(XInputMethod, XNQueryInputStyle, &im_supported_styles, (char*)NULL);
	XIMStyle bestStyle = 0;
	// TODO: If we want to support languages like chinese or japanese as well we probably have to work with callbacks here.
	XIMStyle supportedStyle = XIMPreeditNone | XIMStatusNone;
    for(int i=0; i < im_supported_styles->count_styles; ++i)
	{
        XIMStyle style = im_supported_styles->supported_styles[i];
        if ((style & supportedStyle) == style) // if we can handle it
		{
            bestStyle = style;
			break;
		}
    }
	XFree(im_supported_styles);

	if ( !bestStyle )
	{
		XDestroyIC(XInputContext);
		XInputContext = 0;

		os::Printer::log("XInputMethod has no input style we can use. Falling back to non-i18n input.", ELL_WARNING);
		setlocale(LC_CTYPE, oldLocale.c_str());
		return false;
	}

	XInputContext = XCreateIC(XInputMethod,
							XNInputStyle, bestStyle,
							XNClientWindow, window,
							(char*)NULL);
	if (!XInputContext )
	{
		os::Printer::log("XInputContext failed to create an input context. Falling back to non-i18n input.", ELL_WARNING);
		setlocale(LC_CTYPE, oldLocale.c_str());
		return false;
	}
	XSetICFocus(XInputContext);
	setlocale(LC_CTYPE, oldLocale.c_str());
	return true;
}

void CIrrDeviceLinux::destroyInputContext()
{
	if ( XInputContext )
	{
		XUnsetICFocus(XInputContext);
		XDestroyIC(XInputContext);
		XInputContext = 0;
	}
	if ( XInputMethod )
	{
		XCloseIM(XInputMethod);
		XInputMethod = 0;
	}
}

EKEY_CODE CIrrDeviceLinux::getKeyCode(const uint32_t& xEventKey)
{
	EKEY_CODE keyCode = (EKEY_CODE)0;

	KeySym x11Key = XkbKeycodeToKeysym(Display, xEventKey, 0, 0);

	core::unordered_map<KeySym,int32_t>::const_iterator it = KeyMap.find(x11Key);
	if (it != KeyMap.end())
	{
		keyCode = (EKEY_CODE)it->second;
	}
	if (keyCode == 0)
	{
		// Any value is better than none, that allows at least using the keys.
		// Worst case is that some keys will be identical, still better than _all_
		// unknown keys being identical.
		if ( !x11Key )
		{
			keyCode = (EKEY_CODE)(xEventKey+KEY_KEY_CODES_COUNT+1);
#ifdef _NBL_DEBUG
			os::Printer::log("No such X11Key, using event keycode", std::to_string(xEventKey), ELL_INFORMATION);
		}
		else if (it == KeyMap.end())
		{
			keyCode = (EKEY_CODE)(x11Key+KEY_KEY_CODES_COUNT+1);
			os::Printer::log("EKEY_CODE not found, using orig. X11 keycode", std::to_string(x11Key), ELL_INFORMATION);
#endif // _NBL_DEBUG
		}
		else
		{
			keyCode = (EKEY_CODE)(x11Key+KEY_KEY_CODES_COUNT+1);
#ifdef _NBL_DEBUG
			os::Printer::log("EKEY_CODE is 0, using orig. X11 keycode", std::to_string(x11Key), ELL_INFORMATION);
#endif // _NBL_DEBUG
		}
 	}
	return keyCode;
}
#endif


#if defined(_NBL_COMPILE_WITH_X11_)
int IrrPrintXError(Display *Display, XErrorEvent *event)
{
	char msg[256];
	char msg2[256];

	snprintf(msg, 256, "%d", event->request_code);
	XGetErrorDatabaseText(Display, "XRequest", msg, "unknown", msg2, 256);
	XGetErrorText(Display, event->error_code, msg, 256);
	os::Printer::log("X Error", msg, ELL_WARNING);
	os::Printer::log("From call ", msg2, ELL_WARNING);
	return 0;
}
#endif

bool CIrrDeviceLinux::switchToFullscreen(bool reset)
{
	if (!CreationParams.Fullscreen)
		return true;
	if (reset)
	{
#ifdef _NBL_LINUX_X11_VIDMODE_
		if (UseXVidMode && CreationParams.Fullscreen)
		{
			XF86VidModeSwitchToMode(Display, screennr, &oldVideoMode);
			XF86VidModeSetViewPort(Display, screennr, 0, 0);
		}
		#endif
		#ifdef _NBL_LINUX_X11_RANDR_
		if (UseXRandR && CreationParams.Fullscreen)
		{
			XRRScreenConfiguration *config=XRRGetScreenInfo(Display,DefaultRootWindow(Display));
			XRRSetScreenConfig(Display,config,DefaultRootWindow(Display),oldRandrMode,oldRandrRotation,CurrentTime);
			XRRFreeScreenConfigInfo(config);
		}
		#endif
		return true;
	}

	#if defined(_NBL_LINUX_X11_VIDMODE_) || defined(_NBL_LINUX_X11_RANDR_)
	int32_t eventbase, errorbase;
	int32_t bestMode = -1;
	#endif

	#ifdef _NBL_LINUX_X11_VIDMODE_
	if (XF86VidModeQueryExtension(Display, &eventbase, &errorbase))
	{
		// enumerate video modes
		int32_t modeCount;
		XF86VidModeModeInfo** modes;

		XF86VidModeGetAllModeLines(Display, screennr, &modeCount, &modes);

		// find fitting mode
		for (int32_t i = 0; i<modeCount; ++i)
		{
			if (bestMode==-1 && modes[i]->hdisplay >= Width && modes[i]->vdisplay >= Height)
				bestMode = i;
			else if (bestMode!=-1 &&
					modes[i]->hdisplay >= Width &&
					modes[i]->vdisplay >= Height &&
					modes[i]->hdisplay <= modes[bestMode]->hdisplay &&
					modes[i]->vdisplay <= modes[bestMode]->vdisplay)
				bestMode = i;
		}
		if (bestMode != -1)
		{
			os::Printer::log("Starting vidmode fullscreen mode...", ELL_INFORMATION);
			{
                std::ostringstream tmp;
                //tmp.seekp(0, std::ios_base::end);
                tmp << modes[bestMode]->hdisplay;
                os::Printer::log("hdisplay: ", tmp.str().c_str(), ELL_INFORMATION);
			}
			{
                std::ostringstream tmp;
                tmp << modes[bestMode]->vdisplay;
                os::Printer::log("vdisplay: ", tmp.str().c_str(), ELL_INFORMATION);
            }

			XF86VidModeSwitchToMode(Display, screennr, modes[bestMode]);
			XF86VidModeSetViewPort(Display, screennr, 0, 0);
			UseXVidMode=true;
		}
		else
		{
			os::Printer::log("Could not find specified video mode, running windowed.", ELL_WARNING);
			CreationParams.Fullscreen = false;
		}

		XFree(modes);
	}
	else
	#endif
	#ifdef _NBL_LINUX_X11_RANDR_
	if (XRRQueryExtension(Display, &eventbase, &errorbase))
	{
		int32_t modeCount;
		XRRScreenConfiguration *config=XRRGetScreenInfo(Display,DefaultRootWindow(Display));
		XRRScreenSize *modes=XRRConfigSizes(config,&modeCount);
		for (int32_t i = 0; i<modeCount; ++i)
		{
			if (bestMode==-1 && (uint32_t)modes[i].width >= Width && (uint32_t)modes[i].height >= Height)
				bestMode = i;
			else if (bestMode!=-1 &&
					(uint32_t)modes[i].width >= Width &&
					(uint32_t)modes[i].height >= Height &&
					modes[i].width <= modes[bestMode].width &&
					modes[i].height <= modes[bestMode].height)
				bestMode = i;
		}
		if (bestMode != -1)
		{
			os::Printer::log("Starting randr fullscreen mode...", ELL_INFORMATION);
			{
                std::ostringstream tmp;
                tmp << modes[bestMode].width;
                os::Printer::log("width: ", tmp.str().c_str(), ELL_INFORMATION);
			}
			{
                std::ostringstream tmp;
                tmp << modes[bestMode].height;
                os::Printer::log("height: ", tmp.str().c_str(), ELL_INFORMATION);
			}

			XRRSetScreenConfig(Display,config,DefaultRootWindow(Display),bestMode,oldRandrRotation,CurrentTime);
			UseXRandR=true;
		}
		XRRFreeScreenConfigInfo(config);
	}
	else
	#endif
	{
		os::Printer::log("VidMode or RandR extension must be installed to allow Irrlicht "
		"to switch to fullscreen mode. Running in windowed mode instead.", ELL_WARNING);
		CreationParams.Fullscreen = false;
	}
	return CreationParams.Fullscreen;
}


#if defined(_NBL_COMPILE_WITH_X11_)
void IrrPrintXGrabError(int grabResult, const char * grabCommand )
{
	if ( grabResult == GrabSuccess )
	{
//		os::Printer::log(grabCommand, ": GrabSuccess", ELL_INFORMATION);
		return;
	}

	switch ( grabResult )
	{
		case AlreadyGrabbed:
			os::Printer::log(grabCommand, ": AlreadyGrabbed", ELL_WARNING);
			break;
		case GrabNotViewable:
			os::Printer::log(grabCommand, ": GrabNotViewable", ELL_WARNING);
			break;
		case GrabFrozen:
			os::Printer::log(grabCommand, ": GrabFrozen", ELL_WARNING);
			break;
		case GrabInvalidTime:
			os::Printer::log(grabCommand, ": GrabInvalidTime", ELL_WARNING);
			break;
		default:
			os::Printer::log(grabCommand, ": grab failed with unknown problem", ELL_WARNING);
			break;
	}
}
#endif


bool CIrrDeviceLinux::createWindow()
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (CreationParams.AuxGLContexts)
		XInitThreads();

	os::Printer::log("Creating X window...", ELL_INFORMATION);
	XSetErrorHandler(IrrPrintXError);

	screennr = DefaultScreen(Display);

	switchToFullscreen();

	// create visual with standard X methods
	{
		os::Printer::log("Using plain X visual");
		XVisualInfo visTempl; //Template to hold requested values
		int visNumber; // Return value of available visuals

		visTempl.screen = screennr;
		// ARGB visuals should be avoided for usual applications
		visTempl.depth = CreationParams.WithAlphaChannel ? 32 : 24;
		while ((!visual) && (visTempl.depth >= 16))
		{
			visual = XGetVisualInfo(Display, VisualScreenMask | VisualDepthMask,
				&visTempl, &visNumber);
			visTempl.depth -= 8;
		}
	}

	if (!visual)
	{
		os::Printer::log("Fatal error, could not get visual.", ELL_ERROR);
		XCloseDisplay(Display);
		Display = 0;
		return false;
	}
#ifdef _NBL_DEBUG
	else
		os::Printer::log("Visual chosen: ", std::to_string(static_cast<uint32_t>(visual->visualid)), ELL_DEBUG);
#endif

	// create color map
	Colormap colormap;
	colormap = XCreateColormap(Display,
		RootWindow(Display, visual->screen),
		visual->visual, AllocNone);

	attributes.colormap = colormap;
	attributes.border_pixel = 0;
	attributes.event_mask = StructureNotifyMask | FocusChangeMask | ExposureMask;
	if (!CreationParams.IgnoreInput)
		attributes.event_mask |= PointerMotionMask |
		ButtonPressMask | KeyPressMask |
		ButtonReleaseMask | KeyReleaseMask;

	if (!CreationParams.WindowId)
	{
		// create new Window
		// Remove window manager decoration in fullscreen
		attributes.override_redirect = CreationParams.Fullscreen;
		window = XCreateWindow(Display,
			RootWindow(Display, visual->screen),
			0, 0, Width, Height, 0, visual->depth,
			InputOutput, visual->visual,
			CWBorderPixel | CWColormap | CWEventMask | CWOverrideRedirect,
			&attributes);
		XMapRaised(Display, window);
		CreationParams.WindowId = (void*)window;
		Atom wmDelete;
		wmDelete = XInternAtom(Display, wmDeleteWindow, True);
		XSetWMProtocols(Display, window, &wmDelete, 1);
		if (CreationParams.Fullscreen)
		{
			XSetInputFocus(Display, window, RevertToParent, CurrentTime);
			int grabKb = XGrabKeyboard(Display, window, True, GrabModeAsync,
				GrabModeAsync, CurrentTime);
			IrrPrintXGrabError(grabKb, "XGrabKeyboard");
			int grabPointer = XGrabPointer(Display, window, True, ButtonPressMask,
				GrabModeAsync, GrabModeAsync, window, None, CurrentTime);
			IrrPrintXGrabError(grabPointer, "XGrabPointer");
			XWarpPointer(Display, None, window, 0, 0, 0, 0, 0, 0);
		}
	}
	else
	{
		// attach external window
		window = (Window)CreationParams.WindowId;
		if (!CreationParams.IgnoreInput)
		{
			XCreateWindow(Display,
				window,
				0, 0, Width, Height, 0, visual->depth,
				InputOutput, visual->visual,
				CWBorderPixel | CWColormap | CWEventMask,
				&attributes);
		}
		XWindowAttributes wa;
		XGetWindowAttributes(Display, window, &wa);
		CreationParams.WindowSize.Width = wa.width;
		CreationParams.WindowSize.Height = wa.height;
		CreationParams.Fullscreen = false;
		ExternalWindow = true;
	}

	WindowMinimized = false;
	// Currently broken in X, see Bug ID 2795321
	// XkbSetDetectableAutoRepeat(Display, True, &AutorepeatSupport);

	Window tmp;
	uint32_t borderWidth;
	int x, y;
	unsigned int bits;

	XGetGeometry(Display, window, &tmp, &x, &y, &Width, &Height, &borderWidth, &bits);
	CreationParams.Bits = bits;
	CreationParams.WindowSize.Width = Width;
	CreationParams.WindowSize.Height = Height;

	StdHints = XAllocSizeHints();
	long num;
	XGetWMNormalHints(Display, window, StdHints, &num);


	initXAtoms();
#endif // #ifdef _NBL_COMPILE_WITH_X11_
	return true;
}


//! runs the device. Returns false if device wants to be deleted
bool CIrrDeviceLinux::run()
{
	Timer->tick();

#ifdef _NBL_COMPILE_WITH_X11_

	if ( CursorControl )
		static_cast<CCursorControl*>(CursorControl)->update();

	if ((CreationParams.DriverType != video::EDT_NULL) && Display)
	{
		SEvent irrevent;
		irrevent.MouseInput.ButtonStates = 0xffffffff;

		while (XPending(Display) > 0 && !Close)
		{
			XEvent event;
			XNextEvent(Display, &event);

			switch (event.type)
			{
			case ConfigureNotify:
				// check for changed window size
				if ((event.xconfigure.width != (int) Width) ||
					(event.xconfigure.height != (int) Height))
				{
					Width = event.xconfigure.width;
					Height = event.xconfigure.height;

					if (VideoDriver)
						VideoDriver->OnResize(core::dimension2d<uint32_t>(Width, Height));
				}
				break;

			case MapNotify:
				WindowMinimized=false;
				break;

			case UnmapNotify:
				WindowMinimized=true;
				break;

			case FocusIn:
				WindowHasFocus=true;
				break;

			case FocusOut:
				WindowHasFocus=false;
				break;

			case MotionNotify:
				irrevent.EventType = nbl::EET_MOUSE_INPUT_EVENT;
				irrevent.MouseInput.Event = nbl::EMIE_MOUSE_MOVED;
				irrevent.MouseInput.X = event.xmotion.x;
				irrevent.MouseInput.Y = event.xmotion.y;
				irrevent.MouseInput.Control = (event.xmotion.state & ControlMask) != 0;
				irrevent.MouseInput.Shift = (event.xmotion.state & ShiftMask) != 0;

				// mouse button states
				irrevent.MouseInput.ButtonStates = (event.xmotion.state & Button1Mask) ? nbl::EMBSM_LEFT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xmotion.state & Button3Mask) ? nbl::EMBSM_RIGHT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xmotion.state & Button2Mask) ? nbl::EMBSM_MIDDLE : 0;

				postEventFromUser(irrevent);
				break;

			case ButtonPress:
			case ButtonRelease:

				irrevent.EventType = nbl::EET_MOUSE_INPUT_EVENT;
				irrevent.MouseInput.X = event.xbutton.x;
				irrevent.MouseInput.Y = event.xbutton.y;
				irrevent.MouseInput.Control = (event.xbutton.state & ControlMask) != 0;
				irrevent.MouseInput.Shift = (event.xbutton.state & ShiftMask) != 0;

				// mouse button states
				// This sets the state which the buttons had _prior_ to the event.
				// So unlike on Windows the button which just got changed has still the old state here.
				// We handle that below by flipping the corresponding bit later.
				irrevent.MouseInput.ButtonStates = (event.xbutton.state & Button1Mask) ? nbl::EMBSM_LEFT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xbutton.state & Button3Mask) ? nbl::EMBSM_RIGHT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xbutton.state & Button2Mask) ? nbl::EMBSM_MIDDLE : 0;

				irrevent.MouseInput.Event = nbl::EMIE_COUNT;

				switch(event.xbutton.button)
				{
				case  Button1:
					irrevent.MouseInput.Event =
						(event.type == ButtonPress) ? nbl::EMIE_LMOUSE_PRESSED_DOWN : nbl::EMIE_LMOUSE_LEFT_UP;
					irrevent.MouseInput.ButtonStates ^= nbl::EMBSM_LEFT;
					break;

				case  Button3:
					irrevent.MouseInput.Event =
						(event.type == ButtonPress) ? nbl::EMIE_RMOUSE_PRESSED_DOWN : nbl::EMIE_RMOUSE_LEFT_UP;
					irrevent.MouseInput.ButtonStates ^= nbl::EMBSM_RIGHT;
					break;

				case  Button2:
					irrevent.MouseInput.Event =
						(event.type == ButtonPress) ? nbl::EMIE_MMOUSE_PRESSED_DOWN : nbl::EMIE_MMOUSE_LEFT_UP;
					irrevent.MouseInput.ButtonStates ^= nbl::EMBSM_MIDDLE;
					break;

				case  Button4:
					if (event.type == ButtonPress)
					{
						irrevent.MouseInput.Event = EMIE_MOUSE_WHEEL;
						irrevent.MouseInput.Wheel = 1.0f;
					}
					break;

				case  Button5:
					if (event.type == ButtonPress)
					{
						irrevent.MouseInput.Event = EMIE_MOUSE_WHEEL;
						irrevent.MouseInput.Wheel = -1.0f;
					}
					break;
				}

				if (irrevent.MouseInput.Event != nbl::EMIE_COUNT)
				{
					postEventFromUser(irrevent);

					if ( irrevent.MouseInput.Event >= EMIE_LMOUSE_PRESSED_DOWN && irrevent.MouseInput.Event <= EMIE_MMOUSE_PRESSED_DOWN )
					{
						uint32_t clicks = checkSuccessiveClicks(irrevent.MouseInput.X, irrevent.MouseInput.Y, irrevent.MouseInput.Event);
						if ( clicks == 2 )
						{
							irrevent.MouseInput.Event = (EMOUSE_INPUT_EVENT)(EMIE_LMOUSE_DOUBLE_CLICK + irrevent.MouseInput.Event-EMIE_LMOUSE_PRESSED_DOWN);
							postEventFromUser(irrevent);
						}
						else if ( clicks == 3 )
						{
							irrevent.MouseInput.Event = (EMOUSE_INPUT_EVENT)(EMIE_LMOUSE_TRIPLE_CLICK + irrevent.MouseInput.Event-EMIE_LMOUSE_PRESSED_DOWN);
							postEventFromUser(irrevent);
						}
					}
				}
				break;

			case MappingNotify:
				XRefreshKeyboardMapping (&event.xmapping) ;
				break;

			case KeyRelease:
				if (0 == AutorepeatSupport && (XPending( Display ) > 0) )
				{
					// check for Autorepeat manually
					// We'll do the same as Windows does: Only send KeyPressed
					// So every KeyRelease is a real release
					XEvent next_event;
					XPeekEvent (event.xkey.Display, &next_event);
					if ((next_event.type == KeyPress) &&
						(next_event.xkey.keycode == event.xkey.keycode) &&
						(next_event.xkey.time - event.xkey.time) < 2)	// usually same time, but on some systems a difference of 1 is possible
					{
						// Ignore the key release event
						break;
					}
				}

                irrevent.EventType = nbl::EET_KEY_INPUT_EVENT;
                irrevent.KeyInput.PressedDown = false;
                irrevent.KeyInput.Char = 0;	// on release that's undefined
                irrevent.KeyInput.Control = (event.xkey.state & ControlMask) != 0;
                irrevent.KeyInput.Shift = (event.xkey.state & ShiftMask) != 0;
                irrevent.KeyInput.Key = getKeyCode(event.xkey.keycode);

                postEventFromUser(irrevent);
                break;
			case KeyPress:
				{
                    KeySym x11Key;
					if ( XInputContext )
					{
					    wchar_t buf[8]={0};
						Status status;
						int strLen = XwcLookupString(XInputContext, &event.xkey, buf, sizeof(buf), &x11Key, &status);
						if ( status == XBufferOverflow )
 						{
 						    os::Printer::log("XwcLookupString needs a larger buffer", ELL_INFORMATION);
						}
						if ( strLen > 0 && (status == XLookupChars || status == XLookupBoth) )
						{
							if ( strLen > 1 )
								os::Printer::log("Additional returned characters dropped", ELL_INFORMATION);
							irrevent.KeyInput.Char = buf[0];
 						}
 						else
                        {
#if 0 // Most of those are fine - but useful to have the info when debugging Irrlicht itself.
							if ( status == XLookupNone )
								os::Printer::log("XLookupNone", ELL_INFORMATION);
							else if ( status ==  XLookupKeySym )
								// Getting this also when user did not set setlocale(LC_ALL, ""); and using an unknown locale
								// XSupportsLocale doesn't seeem to catch that unfortunately - any other ideas to catch it are welcome.
								os::Printer::log("XLookupKeySym", ELL_INFORMATION);
							else if ( status ==  XBufferOverflow )
								os::Printer::log("XBufferOverflow", ELL_INFORMATION);
							else if ( strLen == 0 )
								os::Printer::log("no string", ELL_INFORMATION);
#endif
							irrevent.KeyInput.Char = 0;
                        }
					}
                    else	// Old version without InputContext. Does not support i18n, but good to have as fallback.
					{
						char buf[8]={0};
						XLookupString(&event.xkey, buf, sizeof(buf), &x11Key, NULL);
						irrevent.KeyInput.Char = ((wchar_t*)(buf))[0];
					}

					irrevent.EventType = nbl::EET_KEY_INPUT_EVENT;
					irrevent.KeyInput.PressedDown = true;
					irrevent.KeyInput.Control = (event.xkey.state & ControlMask) != 0;
					irrevent.KeyInput.Shift = (event.xkey.state & ShiftMask) != 0;
					irrevent.KeyInput.Key = getKeyCode(event.xkey.keycode);

					postEventFromUser(irrevent);
				}
				break;

			case ClientMessage:
				{
					char *atom = XGetAtomName(Display, event.xclient.message_type);
					if (*atom == *wmDeleteWindow)
					{
						os::Printer::log("Quit message received.", ELL_INFORMATION);
						Close = true;
					}
					else
					{
						// we assume it's a user message
						irrevent.EventType = nbl::EET_USER_EVENT;
						irrevent.UserEvent.UserData1 = (int32_t)event.xclient.data.l[0];
						irrevent.UserEvent.UserData2 = (int32_t)event.xclient.data.l[1];
						postEventFromUser(irrevent);
					}
					XFree(atom);
				}
				break;

			case SelectionRequest:
				{
					XEvent respond;
					XSelectionRequestEvent *req = &(event.xselectionrequest);
					if (  req->target == XA_STRING)
					{
						XChangeProperty (Display,
								req->requestor,
								req->property, req->target,
								8, // format
								PropModeReplace,
								(unsigned char*) Clipboard.c_str(),
								Clipboard.size());
						respond.xselection.property = req->property;
					}
					else if ( req->target == X_ATOM_TARGETS )
					{
						long data[2];

						data[0] = X_ATOM_TEXT;
						data[1] = XA_STRING;

						XChangeProperty (Display, req->requestor,
								req->property, req->target,
								8, PropModeReplace,
								(unsigned char *) &data,
								sizeof (data));
						respond.xselection.property = req->property;
					}
					else
					{
						respond.xselection.property= None;
					}
					respond.xselection.type= SelectionNotify;
					respond.xselection.Display= req->Display;
					respond.xselection.requestor= req->requestor;
					respond.xselection.selection=req->selection;
					respond.xselection.target= req->target;
					respond.xselection.time = req->time;
					XSendEvent (Display, req->requestor,0,0,&respond);
					XFlush (Display);
				}
				break;

			default:
				break;
			} // end switch

		} // end while
	}
#endif //_NBL_COMPILE_WITH_X11_

	if (!Close)
		pollJoysticks();

	return !Close;
}


//! Pause the current process for the minimum time allowed only to allow other processes to execute
void CIrrDeviceLinux::yield()
{
	struct timespec ts = {0,1};
	nanosleep(&ts, NULL);
}


//! Pause execution and let other processes to run for a specified amount of time.
void CIrrDeviceLinux::sleep(uint32_t timeMs, bool pauseTimer=false)
{
	const bool wasStopped = Timer ? Timer->isStopped() : true;

	struct timespec ts;
	ts.tv_sec = (time_t) (timeMs / 1000);
	ts.tv_nsec = (long) (timeMs % 1000) * 1000000;

	if (pauseTimer && !wasStopped)
		Timer->stop();

	nanosleep(&ts, NULL);

	if (pauseTimer && !wasStopped)
		Timer->start();
}


//! sets the caption of the window
void CIrrDeviceLinux::setWindowCaption(const std::wstring& text)
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (CreationParams.DriverType == video::EDT_NULL)
		return;

    const wchar_t* tmpPtr = text.data();

	XTextProperty txt;
	if (Success==XwcTextListToTextProperty(Display, const_cast<wchar_t**>(&tmpPtr),
				1, XStdICCTextStyle, &txt))
	{
		XSetWMName(Display, window, &txt);
		XSetWMIconName(Display, window, &txt);
		XFree(txt.value);
	}
#endif
}


//! notifies the device that it should close itself
void CIrrDeviceLinux::closeDevice()
{
	Close = true;
}


//! returns if window is active. if not, nothing need to be drawn
bool CIrrDeviceLinux::isWindowActive() const
{
	return (WindowHasFocus && !WindowMinimized);
}


//! returns if window has focus.
bool CIrrDeviceLinux::isWindowFocused() const
{
	return WindowHasFocus;
}


//! returns if window is minimized.
bool CIrrDeviceLinux::isWindowMinimized() const
{
	return WindowMinimized;
}


//! returns color format of the window.
asset::E_FORMAT CIrrDeviceLinux::getColorFormat() const
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (visual && (visual->depth != 16))
		return asset::EF_R8G8B8_UNORM;
	else
#endif
		return asset::EF_R5G6B5_UNORM_PACK16;
}


//! Sets if the window should be resizable in windowed mode.
void CIrrDeviceLinux::setResizable(bool resize)
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (CreationParams.DriverType == video::EDT_NULL || CreationParams.Fullscreen )
		return;

	XUnmapWindow(Display, window);
	if ( !resize )
	{
		// Must be heap memory because data size depends on X Server
		XSizeHints *hints = XAllocSizeHints();
		hints->flags=PSize|PMinSize|PMaxSize;
		hints->min_width=hints->max_width=hints->base_width=Width;
		hints->min_height=hints->max_height=hints->base_height=Height;
		XSetWMNormalHints(Display, window, hints);
		XFree(hints);
	}
	else
	{
		XSetWMNormalHints(Display, window, StdHints);
	}
	XMapWindow(Display, window);
	XFlush(Display);
#endif // #ifdef _NBL_COMPILE_WITH_X11_
}

//! Minimize window
void CIrrDeviceLinux::minimizeWindow()
{
#ifdef _NBL_COMPILE_WITH_X11_
	XIconifyWindow(Display, window, screennr);
#endif
}


//! Maximize window
void CIrrDeviceLinux::maximizeWindow()
{
#ifdef _NBL_COMPILE_WITH_X11_
	XMapWindow(Display, window);
#endif
}


//! Restore original window size
void CIrrDeviceLinux::restoreWindow()
{
#ifdef _NBL_COMPILE_WITH_X11_
	XMapWindow(Display, window);
#endif
}


void CIrrDeviceLinux::createKeyMap()
{
	// I don't know if this is the best method  to create
	// the lookuptable, but I'll leave it like that until
	// I find a better version.
	// Search for missing numbers in keysymdef.h

#ifdef _NBL_COMPILE_WITH_X11_
	KeyMap.reserve(256);
	KeyMap[XK_BackSpace] = KEY_BACK;
	KeyMap[XK_Tab] = KEY_TAB;
	KeyMap[XK_ISO_Left_Tab] = KEY_TAB;
	KeyMap[XK_Linefeed] = 0; // ???
	KeyMap[XK_Clear] = KEY_CLEAR;
	KeyMap[XK_Return] = KEY_RETURN;
	KeyMap[XK_Pause] = KEY_PAUSE;
	KeyMap[XK_Scroll_Lock] = KEY_SCROLL;
	KeyMap[XK_Sys_Req] = 0; // ???
	KeyMap[XK_Escape] = KEY_ESCAPE;
	KeyMap[XK_Insert] = KEY_INSERT;
	KeyMap[XK_Delete] = KEY_DELETE;
	KeyMap[XK_Home] = KEY_HOME;
	KeyMap[XK_Left] = KEY_LEFT;
	KeyMap[XK_Up] = KEY_UP;
	KeyMap[XK_Right] = KEY_RIGHT;
	KeyMap[XK_Down] = KEY_DOWN;
	KeyMap[XK_Prior] = KEY_PRIOR;
	KeyMap[XK_Page_Up] = KEY_PRIOR;
	KeyMap[XK_Next] = KEY_NEXT;
	KeyMap[XK_Page_Down] = KEY_NEXT;
	KeyMap[XK_End] = KEY_END;
	KeyMap[XK_Begin] = KEY_HOME;
	KeyMap[XK_Num_Lock] = KEY_NUMLOCK;
	KeyMap[XK_KP_Space] = KEY_SPACE;
	KeyMap[XK_KP_Tab] = KEY_TAB;
	KeyMap[XK_KP_Enter] = KEY_RETURN;
	KeyMap[XK_KP_F1] = KEY_F1;
	KeyMap[XK_KP_F2] = KEY_F2;
	KeyMap[XK_KP_F3] = KEY_F3;
	KeyMap[XK_KP_F4] = KEY_F4;
	KeyMap[XK_KP_Home] = KEY_HOME;
	KeyMap[XK_KP_Left] = KEY_LEFT;
	KeyMap[XK_KP_Up] = KEY_UP;
	KeyMap[XK_KP_Right] = KEY_RIGHT;
	KeyMap[XK_KP_Down] = KEY_DOWN;
	KeyMap[XK_Print] = KEY_PRINT;
	KeyMap[XK_KP_Prior] = KEY_PRIOR;
	KeyMap[XK_KP_Page_Up] = KEY_PRIOR;
	KeyMap[XK_KP_Next] = KEY_NEXT;
	KeyMap[XK_KP_Page_Down] = KEY_NEXT;
	KeyMap[XK_KP_End] = KEY_END;
	KeyMap[XK_KP_Begin] = KEY_HOME;
	KeyMap[XK_KP_Insert] = KEY_INSERT;
	KeyMap[XK_KP_Delete] = KEY_DELETE;
	KeyMap[XK_KP_Equal] = 0; // ???
	KeyMap[XK_KP_Multiply] = KEY_MULTIPLY;
	KeyMap[XK_KP_Add] = KEY_ADD;
	KeyMap[XK_KP_Separator] = KEY_SEPARATOR;
	KeyMap[XK_KP_Subtract] = KEY_SUBTRACT;
	KeyMap[XK_KP_Decimal] = KEY_DECIMAL;
	KeyMap[XK_KP_Divide] = KEY_DIVIDE;
	KeyMap[XK_KP_0] = KEY_KEY_0;
	KeyMap[XK_KP_1] = KEY_KEY_1;
	KeyMap[XK_KP_2] = KEY_KEY_2;
	KeyMap[XK_KP_3] = KEY_KEY_3;
	KeyMap[XK_KP_4] = KEY_KEY_4;
	KeyMap[XK_KP_5] = KEY_KEY_5;
	KeyMap[XK_KP_6] = KEY_KEY_6;
	KeyMap[XK_KP_7] = KEY_KEY_7;
	KeyMap[XK_KP_8] = KEY_KEY_8;
	KeyMap[XK_KP_9] = KEY_KEY_9;
	KeyMap[XK_F1] = KEY_F1;
	KeyMap[XK_F2] = KEY_F2;
	KeyMap[XK_F3] = KEY_F3;
	KeyMap[XK_F4] = KEY_F4;
	KeyMap[XK_F5] = KEY_F5;
	KeyMap[XK_F6] = KEY_F6;
	KeyMap[XK_F7] = KEY_F7;
	KeyMap[XK_F8] = KEY_F8;
	KeyMap[XK_F9] = KEY_F9;
	KeyMap[XK_F10] = KEY_F10;
	KeyMap[XK_F11] = KEY_F11;
	KeyMap[XK_F12] = KEY_F12;
	KeyMap[XK_Shift_L] = KEY_LSHIFT;
	KeyMap[XK_Shift_R] = KEY_RSHIFT;
	KeyMap[XK_Control_L] = KEY_LCONTROL;
	KeyMap[XK_Control_R] = KEY_RCONTROL;
	KeyMap[XK_Caps_Lock] = KEY_CAPITAL;
	KeyMap[XK_Shift_Lock] = KEY_CAPITAL;
	KeyMap[XK_Meta_L] = KEY_LWIN;
	KeyMap[XK_Meta_R] = KEY_RWIN;
	KeyMap[XK_Alt_L] = KEY_LMENU;
	KeyMap[XK_Alt_R] = KEY_RMENU;
	KeyMap[XK_ISO_Level3_Shift] = KEY_RMENU;
	KeyMap[XK_Menu] = KEY_MENU;
	KeyMap[XK_space] = KEY_SPACE;
	KeyMap[XK_exclam] = 0; //?
	KeyMap[XK_quotedbl] = 0; //?
	KeyMap[XK_section] = 0; //?
	KeyMap[XK_numbersign] = KEY_OEM_2;
	KeyMap[XK_dollar] = 0; //?
	KeyMap[XK_percent] = 0; //?
	KeyMap[XK_ampersand] = 0; //?
	KeyMap[XK_apostrophe] = KEY_OEM_7;
	KeyMap[XK_parenleft] = 0; //?
	KeyMap[XK_parenright] = 0; //?
	KeyMap[XK_asterisk] = 0; //?
	KeyMap[XK_plus] = KEY_PLUS; //?
	KeyMap[XK_comma] = KEY_COMMA; //?
	KeyMap[XK_minus] = KEY_MINUS; //?
	KeyMap[XK_period] = KEY_PERIOD; //?
	KeyMap[XK_slash] = KEY_OEM_2; //?
	KeyMap[XK_0] = KEY_KEY_0;
	KeyMap[XK_1] = KEY_KEY_1;
	KeyMap[XK_2] = KEY_KEY_2;
	KeyMap[XK_3] = KEY_KEY_3;
	KeyMap[XK_4] = KEY_KEY_4;
	KeyMap[XK_5] = KEY_KEY_5;
	KeyMap[XK_6] = KEY_KEY_6;
	KeyMap[XK_7] = KEY_KEY_7;
	KeyMap[XK_8] = KEY_KEY_8;
	KeyMap[XK_9] = KEY_KEY_9;
	KeyMap[XK_colon] = 0; //?
	KeyMap[XK_semicolon] = KEY_OEM_1;
	KeyMap[XK_less] = KEY_OEM_102;
	KeyMap[XK_equal] = KEY_PLUS;
	KeyMap[XK_greater] = 0; //?
	KeyMap[XK_question] = 0; //?
	KeyMap[XK_at] = KEY_KEY_2; //?
	KeyMap[XK_mu] = 0; //?
	KeyMap[XK_EuroSign] = 0; //?
	KeyMap[XK_A] = KEY_KEY_A;
	KeyMap[XK_B] = KEY_KEY_B;
	KeyMap[XK_C] = KEY_KEY_C;
	KeyMap[XK_D] = KEY_KEY_D;
	KeyMap[XK_E] = KEY_KEY_E;
	KeyMap[XK_F] = KEY_KEY_F;
	KeyMap[XK_G] = KEY_KEY_G;
	KeyMap[XK_H] = KEY_KEY_H;
	KeyMap[XK_I] = KEY_KEY_I;
	KeyMap[XK_J] = KEY_KEY_J;
	KeyMap[XK_K] = KEY_KEY_K;
	KeyMap[XK_L] = KEY_KEY_L;
	KeyMap[XK_M] = KEY_KEY_M;
	KeyMap[XK_N] = KEY_KEY_N;
	KeyMap[XK_O] = KEY_KEY_O;
	KeyMap[XK_P] = KEY_KEY_P;
	KeyMap[XK_Q] = KEY_KEY_Q;
	KeyMap[XK_R] = KEY_KEY_R;
	KeyMap[XK_S] = KEY_KEY_S;
	KeyMap[XK_T] = KEY_KEY_T;
	KeyMap[XK_U] = KEY_KEY_U;
	KeyMap[XK_V] = KEY_KEY_V;
	KeyMap[XK_W] = KEY_KEY_W;
	KeyMap[XK_X] = KEY_KEY_X;
	KeyMap[XK_Y] = KEY_KEY_Y;
	KeyMap[XK_Z] = KEY_KEY_Z;
	KeyMap[XK_bracketleft] = KEY_OEM_4;
	KeyMap[XK_backslash] = KEY_OEM_5;
	KeyMap[XK_bracketright] = KEY_OEM_6;
	KeyMap[XK_asciicircum] = KEY_OEM_5;
	KeyMap[XK_dead_circumflex] = KEY_OEM_5;
	KeyMap[XK_degree] = 0; //?
	KeyMap[XK_underscore] = KEY_MINUS; //?
	KeyMap[XK_grave] = KEY_OEM_3;
	KeyMap[XK_dead_grave] = KEY_OEM_3;
	KeyMap[XK_acute] = KEY_OEM_6;
	KeyMap[XK_dead_acute] = KEY_OEM_6;
	KeyMap[XK_a] = KEY_KEY_A;
	KeyMap[XK_b] = KEY_KEY_B;
	KeyMap[XK_c] = KEY_KEY_C;
	KeyMap[XK_d] = KEY_KEY_D;
	KeyMap[XK_e] = KEY_KEY_E;
	KeyMap[XK_f] = KEY_KEY_F;
	KeyMap[XK_g] = KEY_KEY_G;
	KeyMap[XK_h] = KEY_KEY_H;
	KeyMap[XK_i] = KEY_KEY_I;
	KeyMap[XK_j] = KEY_KEY_J;
	KeyMap[XK_k] = KEY_KEY_K;
	KeyMap[XK_l] = KEY_KEY_L;
	KeyMap[XK_m] = KEY_KEY_M;
	KeyMap[XK_n] = KEY_KEY_N;
	KeyMap[XK_o] = KEY_KEY_O;
	KeyMap[XK_p] = KEY_KEY_P;
	KeyMap[XK_q] = KEY_KEY_Q;
	KeyMap[XK_r] = KEY_KEY_R;
	KeyMap[XK_s] = KEY_KEY_S;
	KeyMap[XK_t] = KEY_KEY_T;
	KeyMap[XK_u] = KEY_KEY_U;
	KeyMap[XK_v] = KEY_KEY_V;
	KeyMap[XK_w] = KEY_KEY_W;
	KeyMap[XK_x] = KEY_KEY_X;
	KeyMap[XK_y] = KEY_KEY_Y;
	KeyMap[XK_z] = KEY_KEY_Z;
	KeyMap[XK_ssharp] = KEY_OEM_4;
	KeyMap[XK_adiaeresis] = KEY_OEM_7;
	KeyMap[XK_odiaeresis] = KEY_OEM_3;
	KeyMap[XK_udiaeresis] = KEY_OEM_1;
	KeyMap[XK_Super_L] = KEY_LWIN;
	KeyMap[XK_Super_R] = KEY_RWIN;
#endif
}

bool CIrrDeviceLinux::activateJoysticks(core::vector<SJoystickInfo> & joystickInfo)
{
#if defined (_NBL_COMPILE_WITH_JOYSTICK_EVENTS_)

	joystickInfo.clear();

	uint32_t joystick;
	for (joystick = 0; joystick < 32; ++joystick)
	{
		// The joystick device could be here...
		core::stringc devName = "/dev/js";
		devName += joystick;

		SJoystickInfo returnInfo;
		JoystickInfo info;

		info.fd = open(devName.c_str(), O_RDONLY);
		if (-1 == info.fd)
		{
			// ...but Ubuntu and possibly other distros
			// create the devices in /dev/input
			devName = "/dev/input/js";
			devName += joystick;
			info.fd = open(devName.c_str(), O_RDONLY);
			if (-1 == info.fd)
			{
				// and BSD here
				devName = "/dev/joy";
				devName += joystick;
				info.fd = open(devName.c_str(), O_RDONLY);
			}
		}

		if (-1 == info.fd)
			continue;

#ifdef __FreeBSD__
		info.axes=2;
		info.buttons=2;
#else
		ioctl( info.fd, JSIOCGAXES, &(info.axes) );
		ioctl( info.fd, JSIOCGBUTTONS, &(info.buttons) );
		fcntl( info.fd, F_SETFL, O_NONBLOCK );
#endif

		(void)memset(&info.persistentData, 0, sizeof(info.persistentData));
		info.persistentData.EventType = nbl::EET_JOYSTICK_INPUT_EVENT;
		info.persistentData.JoystickEvent.Joystick = ActiveJoysticks.size();

		// There's no obvious way to determine which (if any) axes represent a POV
		// hat, so we'll just set it to "not used" and forget about it.
		info.persistentData.JoystickEvent.POV = 65535;

		ActiveJoysticks.push_back(info);

		returnInfo.Joystick = joystick;
		returnInfo.PovHat = SJoystickInfo::POV_HAT_UNKNOWN;
		returnInfo.Axes = info.axes;
		returnInfo.Buttons = info.buttons;

#ifndef __FreeBSD__
		char name[80];
		ioctl( info.fd, JSIOCGNAME(80), name);
		returnInfo.Name = name;
#endif

		joystickInfo.push_back(returnInfo);
	}

	for (joystick = 0; joystick < joystickInfo.size(); ++joystick)
	{
		char logString[256];
		(void)sprintf(logString, "Found joystick %u, %u axes, %u buttons '%s'",
			joystick, joystickInfo[joystick].Axes,
			joystickInfo[joystick].Buttons, joystickInfo[joystick].Name.c_str());
		os::Printer::log(logString, ELL_INFORMATION);
	}

	return true;
#else
	return false;
#endif // _NBL_COMPILE_WITH_JOYSTICK_EVENTS_
}


void CIrrDeviceLinux::pollJoysticks()
{
#if defined (_NBL_COMPILE_WITH_JOYSTICK_EVENTS_)
	if (0 == ActiveJoysticks.size())
		return;

	for (uint32_t j= 0; j< ActiveJoysticks.size(); ++j)
	{
		JoystickInfo & info =  ActiveJoysticks[j];

#ifdef __FreeBSD__
		struct joystick js;
		if (read(info.fd, &js, sizeof(js)) == sizeof(js))
		{
			info.persistentData.JoystickEvent.ButtonStates = js.b1 | (js.b2 << 1); /* should be a two-bit field */
			info.persistentData.JoystickEvent.Axis[0] = js.x; /* X axis */
			info.persistentData.JoystickEvent.Axis[1] = js.y; /* Y axis */
#else
		struct js_event event;
		while (sizeof(event) == read(info.fd, &event, sizeof(event)))
		{
			switch(event.type & ~JS_EVENT_INIT)
			{
			case JS_EVENT_BUTTON:
				if (event.value)
						info.persistentData.JoystickEvent.ButtonStates |= (1 << event.number);
				else
						info.persistentData.JoystickEvent.ButtonStates &= ~(1 << event.number);
				break;

			case JS_EVENT_AXIS:
				if (event.number < SEvent::SJoystickEvent::NUMBER_OF_AXES)
					info.persistentData.JoystickEvent.Axis[event.number] = event.value;
				break;

			default:
				break;
			}
		}
#endif

		// Send an irrlicht joystick event once per ::run() even if no new data were received.
		(void)postEventFromUser(info.persistentData);
	}
#endif // _NBL_COMPILE_WITH_JOYSTICK_EVENTS_
}


//! gets text from the clipboard
//! \return Returns 0 if no string is in there.
const char* CIrrDeviceLinux::getTextFromClipboard() const
{
#if defined(_NBL_COMPILE_WITH_X11_)
	Window ownerWindow = XGetSelectionOwner (Display, X_ATOM_CLIPBOARD);
	if ( ownerWindow ==  window )
	{
		return Clipboard.c_str();
	}
	Clipboard = "";
	if (ownerWindow != None )
	{
		XConvertSelection (Display, X_ATOM_CLIPBOARD, XA_STRING, XA_PRIMARY, ownerWindow, CurrentTime);
 		XFlush (Display);
		XFlush (Display);

		// check for data
		Atom type;
		int format;
		unsigned long numItems, bytesLeft, dummy;
		unsigned char *data;
		XGetWindowProperty (Display, ownerWindow,
				XA_PRIMARY, // property name
				0, // offset
				0, // length (we only check for data, so 0)
				0, // Delete 0==false
				AnyPropertyType, // AnyPropertyType or property identifier
				&type, // return type
				&format, // return format
				&numItems, // number items
				&bytesLeft, // remaining bytes for partial reads
				&data); // data
		if ( bytesLeft > 0 )
		{
			// there is some data to get
			int result = XGetWindowProperty (Display, ownerWindow, XA_PRIMARY, 0,
										bytesLeft, 0, AnyPropertyType, &type, &format,
										&numItems, &dummy, &data);
			if (result == Success)
				Clipboard = (char*)data;
			XFree (data);
		}
	}

	return Clipboard.c_str();

#else
	return 0;
#endif
}

//! copies text to the clipboard
void CIrrDeviceLinux::copyToClipboard(const char* text) const
{
#if defined(_NBL_COMPILE_WITH_X11_)
	// Actually there is no clipboard on X but applications just say they own the clipboard and return text when asked.
	// Which btw. also means that on X you lose clipboard content when closing applications.
	Clipboard = text;
	XSetSelectionOwner (Display, X_ATOM_CLIPBOARD, window, CurrentTime);
	XFlush (Display);
#endif
}

#ifdef _NBL_COMPILE_WITH_X11_
// return true if the passed event has the type passed in parameter arg
Bool PredicateIsEventType(Display *Display, XEvent *event, XPointer arg)
{
	if ( event && event->type == *(int*)arg )
	{
//		os::Printer::log("remove event:", core::stringc((int)arg).c_str(), ELL_INFORMATION);
		return True;
	}
	return False;
}
#endif //_NBL_COMPILE_WITH_X11_

//! Remove all messages pending in the system message loop
void CIrrDeviceLinux::clearSystemMessages()
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (CreationParams.DriverType != video::EDT_NULL)
	{
		XEvent event;
		int usrArg = ButtonPress;
		while ( XCheckIfEvent(Display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = ButtonRelease;
		while ( XCheckIfEvent(Display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = MotionNotify;
		while ( XCheckIfEvent(Display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = KeyRelease;
		while ( XCheckIfEvent(Display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = KeyPress;
		while ( XCheckIfEvent(Display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
	}
#endif //_NBL_COMPILE_WITH_X11_
}

void CIrrDeviceLinux::initXAtoms()
{
#ifdef _NBL_COMPILE_WITH_X11_
	X_ATOM_CLIPBOARD = XInternAtom(Display, "CLIPBOARD", False);
	X_ATOM_TARGETS = XInternAtom(Display, "TARGETS", False);
	X_ATOM_UTF8_STRING = XInternAtom (Display, "UTF8_STRING", False);
	X_ATOM_TEXT = XInternAtom (Display, "TEXT", False);
#endif
}


CIrrDeviceLinux::CCursorControl::CCursorControl(CIrrDeviceLinux* dev, bool null)
	: Device(dev)
#ifdef _NBL_COMPILE_WITH_X11_
	, PlatformBehavior(gui::ECPB_NONE), lastQuery(0)
#endif
	, IsVisible(true), Null(null), UseReferenceRect(false)
	, ActiveIcon(gui::ECI_NORMAL), ActiveIconStartTime(0)
{
#ifdef _NBL_COMPILE_WITH_X11_
	if (!Null)
	{
		XGCValues values;
		unsigned long valuemask = 0;

		XColor fg, bg;

		// this code, for making the cursor invisible was sent in by
		// Sirshane, thank your very much!


		Pixmap invisBitmap = XCreatePixmap(Device->Display, Device->window, 32, 32, 1);
		Pixmap maskBitmap = XCreatePixmap(Device->Display, Device->window, 32, 32, 1);
		Colormap screen_colormap = DefaultColormap( Device->Display, DefaultScreen( Device->Display ) );
		XAllocNamedColor( Device->Display, screen_colormap, "black", &fg, &fg );
		XAllocNamedColor( Device->Display, screen_colormap, "white", &bg, &bg );

		GC gc = XCreateGC( Device->Display, invisBitmap, valuemask, &values );

		XSetForeground( Device->Display, gc, BlackPixel( Device->Display, DefaultScreen( Device->Display ) ) );
		XFillRectangle( Device->Display, invisBitmap, gc, 0, 0, 32, 32 );
		XFillRectangle( Device->Display, maskBitmap, gc, 0, 0, 32, 32 );

		invisCursor = XCreatePixmapCursor( Device->Display, invisBitmap, maskBitmap, &fg, &bg, 1, 1 );
		XFreeGC(Device->Display, gc);
		XFreePixmap(Device->Display, invisBitmap);
		XFreePixmap(Device->Display, maskBitmap);

		initCursors();
	}
#endif
}

CIrrDeviceLinux::CCursorControl::~CCursorControl()
{
	// Do not clearCursors here as the Display is already closed
	// TODO (cutealien): droping cursorcontrol earlier might work, not sure about reason why that's done in stub currently.
}

#ifdef _NBL_COMPILE_WITH_X11_
void CIrrDeviceLinux::CCursorControl::clearCursors()
{
	if (!Null)
		XFreeCursor(Device->Display, invisCursor);
	for ( uint32_t i=0; i < Cursors.size(); ++i )
	{
		for ( uint32_t f=0; f < Cursors[i].Frames.size(); ++f )
		{
			XFreeCursor(Device->Display, Cursors[i].Frames[f].IconHW);
		}
	}
}

void CIrrDeviceLinux::CCursorControl::initCursors()
{
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_top_left_arrow)) ); //  (or XC_arrow?)
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_crosshair)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_hand2)) ); // (or XC_hand1? )
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_question_arrow)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_xterm)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_X_cursor)) );	//  (or XC_pirate?)
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_watch)) );	// (or XC_clock?)
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_fleur)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_top_right_corner)) );	// NESW not available in X11
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_top_left_corner)) );	// NWSE not available in X11
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_sb_v_double_arrow)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_sb_h_double_arrow)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->Display, XC_sb_up_arrow)) );	// (or XC_center_ptr?)
}

void CIrrDeviceLinux::CCursorControl::update()
{
	if ( (uint32_t)ActiveIcon < Cursors.size() && !Cursors[ActiveIcon].Frames.empty() && Cursors[ActiveIcon].FrameTime )
	{
		// update animated cursors. This could also be done by X11 in case someone wants to figure that out (this way was just easier to implement)
		uint32_t now = Device->getTimer()->getRealTime();
		uint32_t frame = ((now - ActiveIconStartTime) / Cursors[ActiveIcon].FrameTime) % Cursors[ActiveIcon].Frames.size();
		XDefineCursor(Device->Display, Device->window, Cursors[ActiveIcon].Frames[frame].IconHW);
	}
}
#endif

//! Sets the active cursor icon
void CIrrDeviceLinux::CCursorControl::setActiveIcon(gui::ECURSOR_ICON iconId)
{
#ifdef _NBL_COMPILE_WITH_X11_
	if ( iconId >= (int32_t)Cursors.size() )
		return;

	if ( Cursors[iconId].Frames.size() )
		XDefineCursor(Device->Display, Device->window, Cursors[iconId].Frames[0].IconHW);

	ActiveIconStartTime = Device->getTimer()->getRealTime();
	ActiveIcon = iconId;
#endif
}


//! Add a custom sprite as cursor icon.
gui::ECURSOR_ICON CIrrDeviceLinux::CCursorControl::addIcon(const gui::SCursorSprite& icon)
{/**
#ifdef _NBL_COMPILE_WITH_X11_
	if ( icon.SpriteId >= 0 )
	{
		CursorX11 cX11;
		cX11.FrameTime = icon.SpriteBank->getSprites()[icon.SpriteId].frameTime;
		for ( uint32_t i=0; i < icon.SpriteBank->getSprites()[icon.SpriteId].Frames.size(); ++i )
		{
			uint32_t texId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].textureNumber;
			uint32_t rectId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].rectNumber;
			nbl::core::rect<int32_t> rectIcon = icon.SpriteBank->getPositions()[rectId];
			Cursor cursor = Device->TextureToCursor(icon.SpriteBank->getTexture(texId), rectIcon, icon.HotSpot);
			cX11.Frames.push_back( CursorFrameX11(cursor) );
		}

		Cursors.push_back( cX11 );

		return (gui::ECURSOR_ICON)(Cursors.size() - 1);
	}
#endif**/
	return gui::ECI_NORMAL;
}

//! replace the given cursor icon.
void CIrrDeviceLinux::CCursorControl::changeIcon(gui::ECURSOR_ICON iconId, const gui::SCursorSprite& icon)
{/**
#ifdef _NBL_COMPILE_WITH_X11_
	if ( iconId >= (int32_t)Cursors.size() )
		return;

	for ( uint32_t i=0; i < Cursors[iconId].Frames.size(); ++i )
		XFreeCursor(Device->Display, Cursors[iconId].Frames[i].IconHW);

	if ( icon.SpriteId >= 0 )
	{
		CursorX11 cX11;
		cX11.FrameTime = icon.SpriteBank->getSprites()[icon.SpriteId].frameTime;
		for ( uint32_t i=0; i < icon.SpriteBank->getSprites()[icon.SpriteId].Frames.size(); ++i )
		{
			uint32_t texId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].textureNumber;
			uint32_t rectId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].rectNumber;
			nbl::core::rect<int32_t> rectIcon = icon.SpriteBank->getPositions()[rectId];
			Cursor cursor = Device->TextureToCursor(icon.SpriteBank->getTexture(texId), rectIcon, icon.HotSpot);
			cX11.Frames.push_back( CursorFrameX11(cursor) );
		}

		Cursors[iconId] = cX11;
	}
#endif**/
}


nbl::core::dimension2di CIrrDeviceLinux::CCursorControl::getSupportedIconSize() const
{
	// this returns the closest match that is smaller or same size, so we just pass a value which should be large enough for cursors
	unsigned int width=0, height=0;
#ifdef _NBL_COMPILE_WITH_X11_
	XQueryBestCursor(Device->Display, Device->window, 64, 64, &width, &height);
#endif
	return core::dimension2di(width, height);
}

} // end namespace

#endif // _NBL_COMPILE_WITH_X11_DEVICE_


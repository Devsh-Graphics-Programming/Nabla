// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CIrrDeviceLinux.h"

#ifdef _IRR_COMPILE_WITH_X11_DEVICE_

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/utsname.h>
#include <time.h>
#include <sstream>
#include "IEventReceiver.h"
#include "ISceneManager.h"
#include "os.h"
#include "coreutil.h"
#include "Keycodes.h"
#include "COSOperator.h"
#include "CColorConverter.h"
#include "SIrrCreationParameters.h"
#include <X11/XKBlib.h>
#include <X11/Xatom.h>

#ifdef _IRR_LINUX_XCURSOR_
#include <X11/Xcursor/Xcursor.h>
#endif

#if defined _IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#include <fcntl.h>
#include <unistd.h>

#ifdef __FreeBSD__
#include <sys/joystick.h>
#else

// linux/joystick.h includes linux/input.h, which #defines values for various KEY_FOO keys.
// These override the irr::KEY_FOO equivalents, which stops key handling from working.
// As a workaround, defining _INPUT_H stops linux/input.h from being included; it
// doesn't actually seem to be necessary except to pull in sys/ioctl.h.
#define _INPUT_H
#include <sys/ioctl.h> // Would normally be included in linux/input.h
#include <linux/joystick.h>
#undef _INPUT_H
#endif

#endif // _IRR_COMPILE_WITH_JOYSTICK_EVENTS_

#ifdef _IRR_COMPILE_WITH_X11_

#ifdef _IRR_COMPILE_WITH_OPENGL_
    #include "COpenGLDriver.h"
#endif // _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
	namespace video
	{
		IVideoDriver* createOpenGLDriver(const SIrrlichtCreationParameters& params,
				io::IFileSystem* io, CIrrDeviceLinux* device
#ifdef _IRR_COMPILE_WITH_OPENGL_
                ,COpenGLDriver::SAuxContext* auxCtxts
#endif // _IRR_COMPILE_WITH_OPENGL_
        );
	}
} // end namespace irr
#endif // _IRR_COMPILE_WITH_X11_

namespace
{
	Atom X_ATOM_CLIPBOARD;
	Atom X_ATOM_TARGETS;
	Atom X_ATOM_UTF8_STRING;
	Atom X_ATOM_TEXT;
};

namespace irr
{

const char* wmDeleteWindow = "WM_DELETE_WINDOW";

//! constructor
CIrrDeviceLinux::CIrrDeviceLinux(const SIrrlichtCreationParameters& param)
	: CIrrDeviceStub(param),
#ifdef _IRR_COMPILE_WITH_X11_
	display(0), visual(0), screennr(0), window(0), StdHints(0), SoftwareImage(0),
	XInputMethod(0), XInputContext(0),
#ifdef _IRR_COMPILE_WITH_OPENGL_
	glxWin(0),	Context(0), AuxContexts(0),
#endif
#endif
	Width(param.WindowSize.Width), Height(param.WindowSize.Height),
	WindowHasFocus(false), WindowMinimized(false),
	UseXVidMode(false), UseXRandR(false),
	ExternalWindow(false), AutorepeatSupport(0)
{
	#ifdef _IRR_DEBUG
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

#ifdef _IRR_COMPILE_WITH_OPENCL_
    ocl::COpenCLHandler::enumeratePlatformsAndDevices();
#endif // _IRR_COMPILE_WITH_OPENCL_

	// create driver
	createDriver();

	if (!VideoDriver)
		return;

#ifdef _IRR_COMPILE_WITH_X11_
    createInputContext();
#endif // _IRR_COMPILE_WITH_X11_

	createGUIAndScene();
}


//! destructor
CIrrDeviceLinux::~CIrrDeviceLinux()
{
#ifdef _IRR_COMPILE_WITH_X11_
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

	if (display)
	{
		#ifdef _IRR_COMPILE_WITH_OPENGL_
		if (Context)
		{
			if (glxWin)
			{
				if (!glXMakeContextCurrent(display, None, None, NULL))
					os::Printer::log("Could not release glx context.", ELL_WARNING);
			}
			else
			{
				if (!glXMakeCurrent(display, None, NULL))
					os::Printer::log("Could not release glx context.", ELL_WARNING);
			}
			glXDestroyContext(display, Context);

			if (glxWin)
				glXDestroyWindow(display, glxWin);
		}
		#endif // #ifdef _IRR_COMPILE_WITH_OPENGL_

		// Reset fullscreen resolution change
		switchToFullscreen(true);

		if (SoftwareImage)
			XDestroyImage(SoftwareImage);

		if (!ExternalWindow)
		{
			XDestroyWindow(display,window);
			XCloseDisplay(display);
		}
	}
	if (visual)
		XFree(visual);

#endif // #ifdef _IRR_COMPILE_WITH_X11_

#if defined(_IRR_COMPILE_WITH_JOYSTICK_EVENTS_)
	for (uint32_t joystick = 0; joystick < ActiveJoysticks.size(); ++joystick)
	{
		if (ActiveJoysticks[joystick].fd >= 0)
		{
			close(ActiveJoysticks[joystick].fd);
		}
	}
#endif
}

#ifdef _IRR_COMPILE_WITH_X11_
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

	XInputMethod = XOpenIM(display, NULL, NULL, NULL);
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

	KeySym x11Key = XkbKeycodeToKeysym(display, xEventKey, 0, 0);

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
#ifdef _IRR_DEBUG
			os::Printer::log("No such X11Key, using event keycode", std::to_string(xEventKey), ELL_INFORMATION);
		}
		else if (it == KeyMap.end())
		{
			keyCode = (EKEY_CODE)(x11Key+KEY_KEY_CODES_COUNT+1);
			os::Printer::log("EKEY_CODE not found, using orig. X11 keycode", std::to_string(x11Key), ELL_INFORMATION);
#endif // _IRR_DEBUG
		}
		else
		{
			keyCode = (EKEY_CODE)(x11Key+KEY_KEY_CODES_COUNT+1);
#ifdef _IRR_DEBUG
			os::Printer::log("EKEY_CODE is 0, using orig. X11 keycode", std::to_string(x11Key), ELL_INFORMATION);
#endif // _IRR_DEBUG
		}
 	}
	return keyCode;
}
#endif


#if defined(_IRR_COMPILE_WITH_X11_)
int IrrPrintXError(Display *display, XErrorEvent *event)
{
	char msg[256];
	char msg2[256];

	snprintf(msg, 256, "%d", event->request_code);
	XGetErrorDatabaseText(display, "XRequest", msg, "unknown", msg2, 256);
	XGetErrorText(display, event->error_code, msg, 256);
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
#ifdef _IRR_LINUX_X11_VIDMODE_
		if (UseXVidMode && CreationParams.Fullscreen)
		{
			XF86VidModeSwitchToMode(display, screennr, &oldVideoMode);
			XF86VidModeSetViewPort(display, screennr, 0, 0);
		}
		#endif
		#ifdef _IRR_LINUX_X11_RANDR_
		if (UseXRandR && CreationParams.Fullscreen)
		{
			XRRScreenConfiguration *config=XRRGetScreenInfo(display,DefaultRootWindow(display));
			XRRSetScreenConfig(display,config,DefaultRootWindow(display),oldRandrMode,oldRandrRotation,CurrentTime);
			XRRFreeScreenConfigInfo(config);
		}
		#endif
		return true;
	}

	getVideoModeList();
	#if defined(_IRR_LINUX_X11_VIDMODE_) || defined(_IRR_LINUX_X11_RANDR_)
	int32_t eventbase, errorbase;
	int32_t bestMode = -1;
	#endif

	#ifdef _IRR_LINUX_X11_VIDMODE_
	if (XF86VidModeQueryExtension(display, &eventbase, &errorbase))
	{
		// enumerate video modes
		int32_t modeCount;
		XF86VidModeModeInfo** modes;

		XF86VidModeGetAllModeLines(display, screennr, &modeCount, &modes);

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

			XF86VidModeSwitchToMode(display, screennr, modes[bestMode]);
			XF86VidModeSetViewPort(display, screennr, 0, 0);
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
	#ifdef _IRR_LINUX_X11_RANDR_
	if (XRRQueryExtension(display, &eventbase, &errorbase))
	{
		int32_t modeCount;
		XRRScreenConfiguration *config=XRRGetScreenInfo(display,DefaultRootWindow(display));
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

			XRRSetScreenConfig(display,config,DefaultRootWindow(display),bestMode,oldRandrRotation,CurrentTime);
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


#if defined(_IRR_COMPILE_WITH_X11_)
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
#ifdef _IRR_COMPILE_WITH_X11_
    if (CreationParams.AuxGLContexts)
        XInitThreads();

	os::Printer::log("Creating X window...", ELL_INFORMATION);
	XSetErrorHandler(IrrPrintXError);

	display = XOpenDisplay(0);
	if (!display)
	{
		os::Printer::log("Error: Need running XServer to start Irrlicht Engine.", ELL_ERROR);
		if (XDisplayName(0)[0])
			os::Printer::log("Could not open display", XDisplayName(0), ELL_ERROR);
		else
			os::Printer::log("Could not open display, set DISPLAY variable", ELL_ERROR);
		return false;
	}

	screennr = DefaultScreen(display);

	switchToFullscreen();

#ifdef _IRR_COMPILE_WITH_OPENGL_
    // attribute array for the draw buffer
    int visualAttrBuffer[] =
    {
        GLX_X_RENDERABLE    , True,
        GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
        GLX_RENDER_TYPE     , GLX_RGBA_BIT,
        GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
        GLX_RED_SIZE        , 8,
        GLX_GREEN_SIZE      , 8,
        GLX_BLUE_SIZE       , 8,
        GLX_ALPHA_SIZE      , CreationParams.WithAlphaChannel ? 8:0,
        GLX_DEPTH_SIZE      , CreationParams.ZBufferBits,
        GLX_STENCIL_SIZE    , CreationParams.Stencilbuffer ? 8:0,
        GLX_DOUBLEBUFFER    , CreationParams.Doublebuffer ? True:False,
        GLX_STEREO          , CreationParams.Stereobuffer ? True:False,
        GLX_SAMPLE_BUFFERS  , 0,
        GLX_SAMPLES         , 0,
        GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB, True,
        None
    };

    #define IRR_OGL_LOAD_EXTENSION(X) glXGetProcAddress(reinterpret_cast<const GLubyte*>(X))

    int major,minor;
	bool isAvailableGLX=false;
	GLXFBConfig bestFbc;
	if (CreationParams.DriverType==video::EDT_OPENGL)
	{
		isAvailableGLX=glXQueryExtension(display,&major,&minor);
		if (isAvailableGLX && glXQueryVersion(display, &major, &minor) &&
            (major>1 || (major==1&&minor>=3) )  )
		{
            int fbcount;
            GLXFBConfig* fbc = glXChooseFBConfig(display, DefaultScreen(display), visualAttrBuffer, &fbcount);
            if (!fbc)
            {
                if (CreationParams.Stencilbuffer)
                    os::Printer::log("No stencilbuffer available, disabling.", ELL_WARNING);
                CreationParams.Stencilbuffer = !CreationParams.Stencilbuffer;
                visualAttrBuffer[13] = CreationParams.Stencilbuffer ? 1:0;

                fbc = glXChooseFBConfig(display, DefaultScreen(display), visualAttrBuffer, &fbcount);
                if (!fbc && CreationParams.Doublebuffer)
                {
                    os::Printer::log("No doublebuffering available.", ELL_WARNING);
                    CreationParams.Doublebuffer=false;
                    visualAttrBuffer[14] = GLX_USE_GL;
                    fbc = glXChooseFBConfig(display, DefaultScreen(display), visualAttrBuffer, &fbcount);
                }
            }

            if (fbc)
            {
                int desiredSamples = 0;
                int bestSamples = 1024;
                int bestDepth = -1;
                int best_fbc = -1;

                int i;
                for (i=0; i<fbcount; ++i)
                {
                    XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
                    if ( vi )
                    {
                        int obtainedFBConfigAttrs[12];
                        glXGetFBConfigAttrib( display, fbc[i], GLX_RED_SIZE, obtainedFBConfigAttrs+0 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_GREEN_SIZE, obtainedFBConfigAttrs+1 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_BLUE_SIZE, obtainedFBConfigAttrs+2 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_ALPHA_SIZE, obtainedFBConfigAttrs+3 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_DEPTH_SIZE, obtainedFBConfigAttrs+4 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_STENCIL_SIZE, obtainedFBConfigAttrs+5 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_DOUBLEBUFFER, obtainedFBConfigAttrs+6 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_STEREO, obtainedFBConfigAttrs+7 );

                        glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, obtainedFBConfigAttrs+8 );
                        glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , obtainedFBConfigAttrs+9  );

                        glXGetFBConfigAttrib( display, fbc[i], GLX_FBCONFIG_ID       , obtainedFBConfigAttrs+10  );

                        glXGetFBConfigAttrib( display, fbc[i], GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB, obtainedFBConfigAttrs+11  );

                        if (CreationParams.WithAlphaChannel)
                        {
                            if (obtainedFBConfigAttrs[3]<8)
                            {
                                XFree( vi );
                                continue;
                            }

                            if (vi->depth==24)
                            {
                                XFree( vi );
                                continue;
                            }
                        }
                        else
                        {
                            if (obtainedFBConfigAttrs[3])
                            {
                                XFree( vi );
                                continue;
                            }

                            if (vi->depth==32)
                            {
                                XFree( vi );
                                continue;
                            }
                        }

                        if (best_fbc >= 0)
                        {
                            if (obtainedFBConfigAttrs[11]!=True)
                            {
                                XFree( vi );
                                continue;
                            }

                            if (desiredSamples>1) //want AA
                            {
                                if (obtainedFBConfigAttrs[8]!=1 || obtainedFBConfigAttrs[9]<desiredSamples || bestSamples<1024&&obtainedFBConfigAttrs[9]>bestSamples)
                                {
                                    XFree( vi );
                                    continue;
                                }
                            }
                            else if (obtainedFBConfigAttrs[8] || obtainedFBConfigAttrs[9]>1) //don't want AA
                            {
                                XFree( vi );
                                continue;
                            }

                            if (obtainedFBConfigAttrs[0]<8 || obtainedFBConfigAttrs[1]<8 || obtainedFBConfigAttrs[2]<8)
                            {
                                XFree( vi );
                                continue;
                            }

                            if (obtainedFBConfigAttrs[4]<CreationParams.ZBufferBits || bestDepth>=0&&obtainedFBConfigAttrs[4]>bestDepth)
                            {
                                XFree( vi );
                                continue;
                            }

                            if (CreationParams.Stencilbuffer)
                            {
                                if (obtainedFBConfigAttrs[5]<8)
                                {
                                    XFree( vi );
                                    continue;
                                }
                            }
                            else if (obtainedFBConfigAttrs[5])
                            {
                                XFree( vi );
                                continue;
                            }

                            if (CreationParams.Doublebuffer && !obtainedFBConfigAttrs[6])
                            {
                                XFree( vi );
                                continue;
                            }
                        }
/*
                        printf("%d===================================================================\n",obtainedFBConfigAttrs[10]);
                        printf("GLX_RED_SIZE \t\t%d\n",obtainedFBConfigAttrs[0]);
                        printf("GLX_GREEN_SIZE \t\t%d\n",obtainedFBConfigAttrs[1]);
                        printf("GLX_BLUE_SIZE \t\t%d\n",obtainedFBConfigAttrs[2]);
                        printf("GLX_ALPHA_SIZE \t\t%d\n",obtainedFBConfigAttrs[3]);
                        printf("GLX_DEPTH_SIZE \t\t%d\n",obtainedFBConfigAttrs[4]);
                        printf("GLX_STENCIL_SIZE \t\t%d\n",obtainedFBConfigAttrs[5]);
                        printf("GLX_DOUBLEBUFFER \t\t%d\n",obtainedFBConfigAttrs[6]);
                        printf("GLX_STEREO \t\t%d\n",obtainedFBConfigAttrs[7]);
                        printf("GLX_SAMPLE_BUFFERS \t\t%d\n",obtainedFBConfigAttrs[8]);
                        printf("GLX_SAMPLES \t\t%d\n",obtainedFBConfigAttrs[9]);
                        printf("=====================================================================\n");
*/
                        best_fbc = i;
                        bestDepth = obtainedFBConfigAttrs[4];
                        bestSamples = obtainedFBConfigAttrs[9];
                    }
                    XFree( vi );
                }

                //printf("best_fbc is %d\n",best_fbc);
                if (best_fbc<0)
                {
                    os::Printer::log("Couldn't find matching Framebuffer Config.", ELL_ERROR);
                }
                else
                {
                    bestFbc = fbc[ best_fbc ];

                    visual = glXGetVisualFromFBConfig( display, bestFbc );
                    //printf("Visual chosen %d\n",visual->visualid);
                }

                // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
                XFree( fbc );
            }
            else
                os::Printer::log("No GLX support available. OpenGL driver will not work.", ELL_ERROR);
		}
		else
			os::Printer::log("No GLX support available. OpenGL driver will not work.", ELL_ERROR);
	}
	// don't use the XVisual with OpenGL, because it ignores all requested
	// properties of the CreationParams
	else if (!visual)
#endif // _IRR_COMPILE_WITH_OPENGL_

	// create visual with standard X methods
	{
		os::Printer::log("Using plain X visual");
		XVisualInfo visTempl; //Template to hold requested values
		int visNumber; // Return value of available visuals

		visTempl.screen = screennr;
		// ARGB visuals should be avoided for usual applications
		visTempl.depth = CreationParams.WithAlphaChannel?32:24;
		while ((!visual) && (visTempl.depth>=16))
		{
			visual = XGetVisualInfo(display, VisualScreenMask|VisualDepthMask,
				&visTempl, &visNumber);
			visTempl.depth -= 8;
		}
	}

	if (!visual)
	{
		os::Printer::log("Fatal error, could not get visual.", ELL_ERROR);
		XCloseDisplay(display);
		display=0;
		return false;
	}
#ifdef _IRR_DEBUG
	else
		os::Printer::log("Visual chosen: ", std::to_string(static_cast<uint32_t>(visual->visualid)), ELL_DEBUG);
#endif

	// create color map
	Colormap colormap;
	colormap = XCreateColormap(display,
			RootWindow(display, visual->screen),
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
		window = XCreateWindow(display,
				RootWindow(display, visual->screen),
				0, 0, Width, Height, 0, visual->depth,
				InputOutput, visual->visual,
				CWBorderPixel | CWColormap | CWEventMask | CWOverrideRedirect,
				&attributes);
		XMapRaised(display, window);
		CreationParams.WindowId = (void*)window;
		Atom wmDelete;
		wmDelete = XInternAtom(display, wmDeleteWindow, True);
		XSetWMProtocols(display, window, &wmDelete, 1);
		if (CreationParams.Fullscreen)
		{
			XSetInputFocus(display, window, RevertToParent, CurrentTime);
			int grabKb = XGrabKeyboard(display, window, True, GrabModeAsync,
				GrabModeAsync, CurrentTime);
			IrrPrintXGrabError(grabKb, "XGrabKeyboard");
			int grabPointer = XGrabPointer(display, window, True, ButtonPressMask,
				GrabModeAsync, GrabModeAsync, window, None, CurrentTime);
			IrrPrintXGrabError(grabPointer, "XGrabPointer");
			XWarpPointer(display, None, window, 0, 0, 0, 0, 0, 0);
		}
	}
	else
	{
		// attach external window
		window = (Window)CreationParams.WindowId;
		if (!CreationParams.IgnoreInput)
		{
			XCreateWindow(display,
					window,
					0, 0, Width, Height, 0, visual->depth,
					InputOutput, visual->visual,
					CWBorderPixel | CWColormap | CWEventMask,
					&attributes);
		}
		XWindowAttributes wa;
		XGetWindowAttributes(display, window, &wa);
		CreationParams.WindowSize.Width = wa.width;
		CreationParams.WindowSize.Height = wa.height;
		CreationParams.Fullscreen = false;
		ExternalWindow = true;
	}

	WindowMinimized=false;
	// Currently broken in X, see Bug ID 2795321
	// XkbSetDetectableAutoRepeat(display, True, &AutorepeatSupport);

#ifdef _IRR_COMPILE_WITH_OPENGL_

	// connect glx context to window
	Context=0;
	if (isAvailableGLX && CreationParams.DriverType==video::EDT_OPENGL)
	{
		GLXContext tmpCtx = glXCreateContext(display, visual, NULL, True);
		glXMakeCurrent(display, window, tmpCtx);
        //if (glXMakeCurrent(display, window, Context))
            PFNGLXCREATECONTEXTATTRIBSARBPROC pGlxCreateContextAttribsARB = (PFNGLXCREATECONTEXTATTRIBSARBPROC)IRR_OGL_LOAD_EXTENSION("glXCreateContextAttribsARB");

		if (tmpCtx)
        {
            if (pGlxCreateContextAttribsARB)
            {
                int context_attribs[] =
                {
                    GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
                    GLX_CONTEXT_MINOR_VERSION_ARB, 6,
                    GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
                    None
                };

                // create rendering context
                Context = pGlxCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
                if (!Context)
                {
                    context_attribs[3] = 5;
                    Context = pGlxCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
                }
                if (!Context)
                {
                    context_attribs[3] = 4;
                    Context = pGlxCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
                }
                if (!Context)
                {
                    context_attribs[3] = 3;
                    Context = pGlxCreateContextAttribsARB( display, bestFbc, 0, True, context_attribs );
                } //! everything below will go!

                if (Context)
                {
                    AuxContexts = _IRR_NEW_ARRAY(video::COpenGLDriver::SAuxContext,CreationParams.AuxGLContexts+1);
                    {
                        reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[0].threadId = std::this_thread::get_id();
                        reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[0].ctx = Context;
                        reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[0].pbuff = 0ull;
                    }

                    const int pboAttribs[] =
                    {
                        GLX_PBUFFER_WIDTH,  128,
                        GLX_PBUFFER_HEIGHT, 128,
                        GLX_PRESERVED_CONTENTS, 0,
                        None
                    };

                    for (uint8_t i=1; i<=CreationParams.AuxGLContexts; i++)
                    {
                        reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[i].threadId = std::thread::id(); //invalid ID
                        reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[i].ctx = pGlxCreateContextAttribsARB( display, bestFbc, Context, True, context_attribs );
                        reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[i].pbuff = glXCreatePbuffer( display, bestFbc, pboAttribs);
                    }

                    if (!glXMakeCurrent(display, window, Context))
                    {
                        os::Printer::log("Could not make context current.", ELL_WARNING);

                        for (uint8_t i=1; i<=CreationParams.AuxGLContexts; i++)
                        {
                            glXDestroyPbuffer(display,reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[i].pbuff);
                            glXDestroyContext(display,reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts)[i].ctx);
                        }

                        _IRR_DELETE_ARRAY(reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts),CreationParams.AuxGLContexts+1);

                        glXDestroyContext(display, Context);
                        glXMakeCurrent(display, None, NULL);
                    }
                    glXDestroyContext(display, tmpCtx);
                }
                else
                {
                    glXMakeCurrent(display, None, NULL);
                    glXDestroyContext(display, tmpCtx);
                    os::Printer::log("Could not create GLX rendering context.", ELL_WARNING);
                }
            }
            else
            {
                glXMakeCurrent(display, None, NULL);
                glXDestroyContext(display, tmpCtx);
                os::Printer::log("Could not get pointer to glxCreateContextAttribsARB.", ELL_WARNING);
            }
        }
        else
        {
            glXMakeCurrent(display, None, NULL);
            glXDestroyContext(display, tmpCtx);
            os::Printer::log("Could not get pointer to glxCreateContextAttribsARB.", ELL_WARNING);
        }
	}
#endif // _IRR_COMPILE_WITH_OPENGL_

	Window tmp;
	uint32_t borderWidth;
	int x,y;
	unsigned int bits;

	XGetGeometry(display, window, &tmp, &x, &y, &Width, &Height, &borderWidth, &bits);
	CreationParams.Bits = bits;
	CreationParams.WindowSize.Width = Width;
	CreationParams.WindowSize.Height = Height;

	StdHints = XAllocSizeHints();
	long num;
	XGetWMNormalHints(display, window, StdHints, &num);

	// create an XImage for the software renderer
	//(thx to Nadav for some clues on how to do that!)

	if (CreationParams.DriverType == video::EDT_BURNINGSVIDEO)
	{
		SoftwareImage = XCreateImage(display,
			visual->visual, visual->depth,
			ZPixmap, 0, 0, Width, Height,
			BitmapPad(display), 0);

		// use malloc because X will free it later on
		if (SoftwareImage)
			SoftwareImage->data = (char*) malloc(SoftwareImage->bytes_per_line * SoftwareImage->height * sizeof(char));
	}

	initXAtoms();

#endif // #ifdef _IRR_COMPILE_WITH_X11_
	return true;
}


//! create the driver
void CIrrDeviceLinux::createDriver()
{
	switch(CreationParams.DriverType)
	{
#ifdef _IRR_COMPILE_WITH_X11_

	case video::EDT_BURNINGSVIDEO:
		#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_
		VideoDriver = video::createBurningVideoDriver(this, CreationParams, FileSystem, this);
		#else
		os::Printer::log("Burning's video driver was not compiled in.", ELL_ERROR);
		#endif
		break;

	case video::EDT_OPENGL:
		#ifdef _IRR_COMPILE_WITH_OPENGL_
		if (Context)
			VideoDriver = video::createOpenGLDriver(CreationParams, FileSystem, this, reinterpret_cast<video::COpenGLDriver::SAuxContext*>(AuxContexts));
		#else
		os::Printer::log("No OpenGL support compiled in.", ELL_ERROR);
		#endif
		break;

	case video::EDT_NULL:
		VideoDriver = video::createNullDriver(this, FileSystem, CreationParams.WindowSize);
		break;

	default:
		os::Printer::log("Unable to create video driver of unknown type.", ELL_ERROR);
		break;
#else
	case video::EDT_NULL:
		VideoDriver = video::createNullDriver(FileSystem, CreationParams.WindowSize);
		break;
	default:
		os::Printer::log("No X11 support compiled in. Only Null driver available.", ELL_ERROR);
		break;
#endif
	}
}


//! runs the device. Returns false if device wants to be deleted
bool CIrrDeviceLinux::run()
{
	Timer->tick();

#ifdef _IRR_COMPILE_WITH_X11_

	if ( CursorControl )
		static_cast<CCursorControl*>(CursorControl)->update();

	if ((CreationParams.DriverType != video::EDT_NULL) && display)
	{
		SEvent irrevent;
		irrevent.MouseInput.ButtonStates = 0xffffffff;

		while (XPending(display) > 0 && !Close)
		{
			XEvent event;
			XNextEvent(display, &event);

			switch (event.type)
			{
			case ConfigureNotify:
				// check for changed window size
				if ((event.xconfigure.width != (int) Width) ||
					(event.xconfigure.height != (int) Height))
				{
					Width = event.xconfigure.width;
					Height = event.xconfigure.height;

					// resize image data
					if (SoftwareImage)
					{
						XDestroyImage(SoftwareImage);

						SoftwareImage = XCreateImage(display,
							visual->visual, visual->depth,
							ZPixmap, 0, 0, Width, Height,
							BitmapPad(display), 0);

						// use malloc because X will free it later on
						if (SoftwareImage)
							SoftwareImage->data = (char*) malloc(SoftwareImage->bytes_per_line * SoftwareImage->height * sizeof(char));
					}

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
				irrevent.EventType = irr::EET_MOUSE_INPUT_EVENT;
				irrevent.MouseInput.Event = irr::EMIE_MOUSE_MOVED;
				irrevent.MouseInput.X = event.xmotion.x;
				irrevent.MouseInput.Y = event.xmotion.y;
				irrevent.MouseInput.Control = (event.xmotion.state & ControlMask) != 0;
				irrevent.MouseInput.Shift = (event.xmotion.state & ShiftMask) != 0;

				// mouse button states
				irrevent.MouseInput.ButtonStates = (event.xmotion.state & Button1Mask) ? irr::EMBSM_LEFT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xmotion.state & Button3Mask) ? irr::EMBSM_RIGHT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xmotion.state & Button2Mask) ? irr::EMBSM_MIDDLE : 0;

				postEventFromUser(irrevent);
				break;

			case ButtonPress:
			case ButtonRelease:

				irrevent.EventType = irr::EET_MOUSE_INPUT_EVENT;
				irrevent.MouseInput.X = event.xbutton.x;
				irrevent.MouseInput.Y = event.xbutton.y;
				irrevent.MouseInput.Control = (event.xbutton.state & ControlMask) != 0;
				irrevent.MouseInput.Shift = (event.xbutton.state & ShiftMask) != 0;

				// mouse button states
				// This sets the state which the buttons had _prior_ to the event.
				// So unlike on Windows the button which just got changed has still the old state here.
				// We handle that below by flipping the corresponding bit later.
				irrevent.MouseInput.ButtonStates = (event.xbutton.state & Button1Mask) ? irr::EMBSM_LEFT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xbutton.state & Button3Mask) ? irr::EMBSM_RIGHT : 0;
				irrevent.MouseInput.ButtonStates |= (event.xbutton.state & Button2Mask) ? irr::EMBSM_MIDDLE : 0;

				irrevent.MouseInput.Event = irr::EMIE_COUNT;

				switch(event.xbutton.button)
				{
				case  Button1:
					irrevent.MouseInput.Event =
						(event.type == ButtonPress) ? irr::EMIE_LMOUSE_PRESSED_DOWN : irr::EMIE_LMOUSE_LEFT_UP;
					irrevent.MouseInput.ButtonStates ^= irr::EMBSM_LEFT;
					break;

				case  Button3:
					irrevent.MouseInput.Event =
						(event.type == ButtonPress) ? irr::EMIE_RMOUSE_PRESSED_DOWN : irr::EMIE_RMOUSE_LEFT_UP;
					irrevent.MouseInput.ButtonStates ^= irr::EMBSM_RIGHT;
					break;

				case  Button2:
					irrevent.MouseInput.Event =
						(event.type == ButtonPress) ? irr::EMIE_MMOUSE_PRESSED_DOWN : irr::EMIE_MMOUSE_LEFT_UP;
					irrevent.MouseInput.ButtonStates ^= irr::EMBSM_MIDDLE;
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

				if (irrevent.MouseInput.Event != irr::EMIE_COUNT)
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
				if (0 == AutorepeatSupport && (XPending( display ) > 0) )
				{
					// check for Autorepeat manually
					// We'll do the same as Windows does: Only send KeyPressed
					// So every KeyRelease is a real release
					XEvent next_event;
					XPeekEvent (event.xkey.display, &next_event);
					if ((next_event.type == KeyPress) &&
						(next_event.xkey.keycode == event.xkey.keycode) &&
						(next_event.xkey.time - event.xkey.time) < 2)	// usually same time, but on some systems a difference of 1 is possible
					{
						// Ignore the key release event
						break;
					}
				}

                irrevent.EventType = irr::EET_KEY_INPUT_EVENT;
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

					irrevent.EventType = irr::EET_KEY_INPUT_EVENT;
					irrevent.KeyInput.PressedDown = true;
					irrevent.KeyInput.Control = (event.xkey.state & ControlMask) != 0;
					irrevent.KeyInput.Shift = (event.xkey.state & ShiftMask) != 0;
					irrevent.KeyInput.Key = getKeyCode(event.xkey.keycode);

					postEventFromUser(irrevent);
				}
				break;

			case ClientMessage:
				{
					char *atom = XGetAtomName(display, event.xclient.message_type);
					if (*atom == *wmDeleteWindow)
					{
						os::Printer::log("Quit message received.", ELL_INFORMATION);
						Close = true;
					}
					else
					{
						// we assume it's a user message
						irrevent.EventType = irr::EET_USER_EVENT;
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
						XChangeProperty (display,
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

						XChangeProperty (display, req->requestor,
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
					respond.xselection.display= req->display;
					respond.xselection.requestor= req->requestor;
					respond.xselection.selection=req->selection;
					respond.xselection.target= req->target;
					respond.xselection.time = req->time;
					XSendEvent (display, req->requestor,0,0,&respond);
					XFlush (display);
				}
				break;

			default:
				break;
			} // end switch

		} // end while
	}
#endif //_IRR_COMPILE_WITH_X11_

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
#ifdef _IRR_COMPILE_WITH_X11_
	if (CreationParams.DriverType == video::EDT_NULL)
		return;

    const wchar_t* tmpPtr = text.data();

	XTextProperty txt;
	if (Success==XwcTextListToTextProperty(display, const_cast<wchar_t**>(&tmpPtr),
				1, XStdICCTextStyle, &txt))
	{
		XSetWMName(display, window, &txt);
		XSetWMIconName(display, window, &txt);
		XFree(txt.value);
	}
#endif
}


//! presents a surface in the client area
bool CIrrDeviceLinux::present(video::IImage* image, void* windowId, core::rect<int32_t>* srcRect)
{
#ifdef _IRR_COMPILE_WITH_X11_
	// this is only necessary for software drivers.
	if (!SoftwareImage)
		return true;

	// thx to Nadav, who send me some clues of how to display the image
	// to the X Server.

	const uint32_t destwidth = SoftwareImage->width;
	const uint32_t minWidth = core::min_(image->getDimension().Width, destwidth);
	const uint32_t destPitch = SoftwareImage->bytes_per_line;

	asset::E_FORMAT destColor;
	switch (SoftwareImage->bits_per_pixel)
	{
		case 16:
			if (SoftwareImage->depth==16)
				destColor = asset::EF_R5G6B5_UNORM_PACK16;
			else
				destColor = asset::EF_A1R5G5B5_UNORM_PACK16;
		break;
		case 24: destColor = asset::EF_R8G8B8_UNORM; break;
		case 32: destColor = asset::EF_B8G8R8A8_UNORM; break;
		default:
			os::Printer::log("Unsupported screen depth.");
			return false;
	}

	uint8_t* srcdata = reinterpret_cast<uint8_t*>(image->getData());
	uint8_t* destData = reinterpret_cast<uint8_t*>(SoftwareImage->data);

	const uint32_t destheight = SoftwareImage->height;
	const uint32_t srcheight = core::min_(image->getDimension().Height, destheight);
	const uint32_t srcPitch = image->getPitch();
	for (uint32_t y=0; y!=srcheight; ++y)
	{
		video::CColorConverter::convert_viaFormat(srcdata,image->getColorFormat(), minWidth, destData, destColor);
		srcdata+=srcPitch;
		destData+=destPitch;
	}

	GC gc = DefaultGC(display, DefaultScreen(display));
	Window myWindow=window;
	if (windowId)
		myWindow = reinterpret_cast<Window>(windowId);
	XPutImage(display, myWindow, gc, SoftwareImage, 0, 0, 0, 0, destwidth, destheight);
#endif
	return true;
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
#ifdef _IRR_COMPILE_WITH_X11_
	if (visual && (visual->depth != 16))
		return asset::EF_R8G8B8_UNORM;
	else
#endif
		return asset::EF_R5G6B5_UNORM_PACK16;
}


//! Sets if the window should be resizable in windowed mode.
void CIrrDeviceLinux::setResizable(bool resize)
{
#ifdef _IRR_COMPILE_WITH_X11_
	if (CreationParams.DriverType == video::EDT_NULL || CreationParams.Fullscreen )
		return;

	XUnmapWindow(display, window);
	if ( !resize )
	{
		// Must be heap memory because data size depends on X Server
		XSizeHints *hints = XAllocSizeHints();
		hints->flags=PSize|PMinSize|PMaxSize;
		hints->min_width=hints->max_width=hints->base_width=Width;
		hints->min_height=hints->max_height=hints->base_height=Height;
		XSetWMNormalHints(display, window, hints);
		XFree(hints);
	}
	else
	{
		XSetWMNormalHints(display, window, StdHints);
	}
	XMapWindow(display, window);
	XFlush(display);
#endif // #ifdef _IRR_COMPILE_WITH_X11_
}


//! Return pointer to a list with all video modes supported by the gfx adapter.
video::IVideoModeList* CIrrDeviceLinux::getVideoModeList()
{
#ifdef _IRR_COMPILE_WITH_X11_
	if (!VideoModeList->getVideoModeCount())
	{
		bool temporaryDisplay = false;

		if (!display)
		{
			display = XOpenDisplay(0);
			temporaryDisplay=true;
		}
		if (display)
		{
			#if defined(_IRR_LINUX_X11_VIDMODE_) || defined(_IRR_LINUX_X11_RANDR_)
			int32_t eventbase, errorbase;
			int32_t defaultDepth=DefaultDepth(display,screennr);
			#endif

			#ifdef _IRR_LINUX_X11_VIDMODE_
			if (XF86VidModeQueryExtension(display, &eventbase, &errorbase))
			{
				// enumerate video modes
				int modeCount;
				XF86VidModeModeInfo** modes;

				XF86VidModeGetAllModeLines(display, screennr, &modeCount, &modes);

				// save current video mode
				oldVideoMode = *modes[0];

				// find fitting mode

				VideoModeList->setDesktop(defaultDepth, core::dimension2d<uint32_t>(
					modes[0]->hdisplay, modes[0]->vdisplay));
				for (int i = 0; i<modeCount; ++i)
				{
					VideoModeList->addMode(core::dimension2d<uint32_t>(
						modes[i]->hdisplay, modes[i]->vdisplay), defaultDepth);
				}
				XFree(modes);
			}
			else
			#endif
			#ifdef _IRR_LINUX_X11_RANDR_
			if (XRRQueryExtension(display, &eventbase, &errorbase))
			{
				int modeCount;
				XRRScreenConfiguration *config=XRRGetScreenInfo(display,DefaultRootWindow(display));
				oldRandrMode=XRRConfigCurrentConfiguration(config,&oldRandrRotation);
				XRRScreenSize *modes=XRRConfigSizes(config,&modeCount);
				VideoModeList->setDesktop(defaultDepth, core::dimension2d<uint32_t>(
					modes[oldRandrMode].width, modes[oldRandrMode].height));
				for (int i = 0; i<modeCount; ++i)
				{
					VideoModeList->addMode(core::dimension2d<uint32_t>(
						modes[i].width, modes[i].height), defaultDepth);
				}
				XRRFreeScreenConfigInfo(config);
			}
			else
			#endif
			{
				os::Printer::log("VidMode or RandR X11 extension requireed for VideoModeList." , ELL_WARNING);
			}
		}
		if (display && temporaryDisplay)
		{
			XCloseDisplay(display);
			display=0;
		}
	}
#endif

	return VideoModeList;
}

//! Minimize window
void CIrrDeviceLinux::minimizeWindow()
{
#ifdef _IRR_COMPILE_WITH_X11_
	XIconifyWindow(display, window, screennr);
#endif
}


//! Maximize window
void CIrrDeviceLinux::maximizeWindow()
{
#ifdef _IRR_COMPILE_WITH_X11_
	XMapWindow(display, window);
#endif
}


//! Restore original window size
void CIrrDeviceLinux::restoreWindow()
{
#ifdef _IRR_COMPILE_WITH_X11_
	XMapWindow(display, window);
#endif
}


void CIrrDeviceLinux::createKeyMap()
{
	// I don't know if this is the best method  to create
	// the lookuptable, but I'll leave it like that until
	// I find a better version.
	// Search for missing numbers in keysymdef.h

#ifdef _IRR_COMPILE_WITH_X11_
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
#if defined (_IRR_COMPILE_WITH_JOYSTICK_EVENTS_)

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
		info.persistentData.EventType = irr::EET_JOYSTICK_INPUT_EVENT;
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
#endif // _IRR_COMPILE_WITH_JOYSTICK_EVENTS_
}


void CIrrDeviceLinux::pollJoysticks()
{
#if defined (_IRR_COMPILE_WITH_JOYSTICK_EVENTS_)
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
#endif // _IRR_COMPILE_WITH_JOYSTICK_EVENTS_
}


//! gets text from the clipboard
//! \return Returns 0 if no string is in there.
const char* CIrrDeviceLinux::getTextFromClipboard() const
{
#if defined(_IRR_COMPILE_WITH_X11_)
	Window ownerWindow = XGetSelectionOwner (display, X_ATOM_CLIPBOARD);
	if ( ownerWindow ==  window )
	{
		return Clipboard.c_str();
	}
	Clipboard = "";
	if (ownerWindow != None )
	{
		XConvertSelection (display, X_ATOM_CLIPBOARD, XA_STRING, XA_PRIMARY, ownerWindow, CurrentTime);
 		XFlush (display);
		XFlush (display);

		// check for data
		Atom type;
		int format;
		unsigned long numItems, bytesLeft, dummy;
		unsigned char *data;
		XGetWindowProperty (display, ownerWindow,
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
			int result = XGetWindowProperty (display, ownerWindow, XA_PRIMARY, 0,
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
#if defined(_IRR_COMPILE_WITH_X11_)
	// Actually there is no clipboard on X but applications just say they own the clipboard and return text when asked.
	// Which btw. also means that on X you lose clipboard content when closing applications.
	Clipboard = text;
	XSetSelectionOwner (display, X_ATOM_CLIPBOARD, window, CurrentTime);
	XFlush (display);
#endif
}

#ifdef _IRR_COMPILE_WITH_X11_
// return true if the passed event has the type passed in parameter arg
Bool PredicateIsEventType(Display *display, XEvent *event, XPointer arg)
{
	if ( event && event->type == *(int*)arg )
	{
//		os::Printer::log("remove event:", core::stringc((int)arg).c_str(), ELL_INFORMATION);
		return True;
	}
	return False;
}
#endif //_IRR_COMPILE_WITH_X11_

//! Remove all messages pending in the system message loop
void CIrrDeviceLinux::clearSystemMessages()
{
#ifdef _IRR_COMPILE_WITH_X11_
	if (CreationParams.DriverType != video::EDT_NULL)
	{
		XEvent event;
		int usrArg = ButtonPress;
		while ( XCheckIfEvent(display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = ButtonRelease;
		while ( XCheckIfEvent(display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = MotionNotify;
		while ( XCheckIfEvent(display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = KeyRelease;
		while ( XCheckIfEvent(display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
		usrArg = KeyPress;
		while ( XCheckIfEvent(display, &event, PredicateIsEventType, XPointer(&usrArg)) == True ) {}
	}
#endif //_IRR_COMPILE_WITH_X11_
}

void CIrrDeviceLinux::initXAtoms()
{
#ifdef _IRR_COMPILE_WITH_X11_
	X_ATOM_CLIPBOARD = XInternAtom(display, "CLIPBOARD", False);
	X_ATOM_TARGETS = XInternAtom(display, "TARGETS", False);
	X_ATOM_UTF8_STRING = XInternAtom (display, "UTF8_STRING", False);
	X_ATOM_TEXT = XInternAtom (display, "TEXT", False);
#endif
}


#ifdef _IRR_COMPILE_WITH_X11_
Cursor CIrrDeviceLinux::TextureToMonochromeCursor(irr::video::IImage * tex, const core::rect<int32_t>& sourceRect, const core::position2d<int32_t> &hotspot)
{
	XImage * sourceImage = XCreateImage(display, visual->visual,
										1, // depth,
										ZPixmap,	// XYBitmap (depth=1), ZPixmap(depth=x)
										0, 0, sourceRect.getWidth(), sourceRect.getHeight(),
										32, // bitmap_pad,
										0// bytes_per_line (0 means continuos in memory)
										);
	sourceImage->data = new char[sourceImage->height * sourceImage->bytes_per_line];
	XImage * maskImage = XCreateImage(display, visual->visual,
										1, // depth,
										ZPixmap,
										0, 0, sourceRect.getWidth(), sourceRect.getHeight(),
										32, // bitmap_pad,
										0 // bytes_per_line
										);
	maskImage->data = new char[maskImage->height * maskImage->bytes_per_line];

	// write texture into XImage
	asset::E_FORMAT format = tex->getColorFormat();
	uint32_t bytesPerPixel = video::getBitsPerPixelFromFormat(format) / 8;
	uint32_t bytesLeftGap = sourceRect.UpperLeftCorner.X * bytesPerPixel;
	uint32_t bytesRightGap = tex->getPitch() - sourceRect.LowerRightCorner.X * bytesPerPixel;
	const uint8_t* data = (const uint8_t*)tex->getData();
	data += sourceRect.UpperLeftCorner.Y*tex->getPitch();
	for ( int32_t y = 0; y < sourceRect.getHeight(); ++y )
	{
		data += bytesLeftGap;
		for ( int32_t x = 0; x < sourceRect.getWidth(); ++x )
		{
			video::SColor pixelCol;
			pixelCol.setData((const void*)data, format);
			data += bytesPerPixel;

			if ( pixelCol.getAlpha() == 0 )	// transparent
			{
				XPutPixel(maskImage, x, y, 0);
				XPutPixel(sourceImage, x, y, 0);
			}
			else	// color
			{
				if ( pixelCol.getAverage() >= 127 )
					XPutPixel(sourceImage, x, y, 1);
				else
					XPutPixel(sourceImage, x, y, 0);
				XPutPixel(maskImage, x, y, 1);
			}
		}
		data += bytesRightGap;
	}

	Pixmap sourcePixmap = XCreatePixmap(display, window, sourceImage->width, sourceImage->height, sourceImage->depth);
	Pixmap maskPixmap = XCreatePixmap(display, window, maskImage->width, maskImage->height, maskImage->depth);

	XGCValues values;
	values.foreground = 1;
	values.background = 1;
	GC gc = XCreateGC( display, sourcePixmap, GCForeground | GCBackground, &values );

	XPutImage(display, sourcePixmap, gc, sourceImage, 0, 0, 0, 0, sourceImage->width, sourceImage->height);
	XPutImage(display, maskPixmap, gc, maskImage, 0, 0, 0, 0, maskImage->width, maskImage->height);

	XFreeGC(display, gc);
	XDestroyImage(sourceImage);
	XDestroyImage(maskImage);

	Cursor cursorResult = 0;
	XColor foreground, background;
	foreground.red = 65535;
	foreground.green = 65535;
	foreground.blue = 65535;
	foreground.flags = DoRed | DoGreen | DoBlue;
	background.red = 0;
	background.green = 0;
	background.blue = 0;
	background.flags = DoRed | DoGreen | DoBlue;

	cursorResult = XCreatePixmapCursor(display, sourcePixmap, maskPixmap, &foreground, &background, hotspot.X, hotspot.Y);

	XFreePixmap(display, sourcePixmap);
	XFreePixmap(display, maskPixmap);

	return cursorResult;
}

#ifdef _IRR_LINUX_XCURSOR_
Cursor CIrrDeviceLinux::TextureToARGBCursor(irr::video::IImage * tex, const core::rect<int32_t>& sourceRect, const core::position2d<int32_t> &hotspot)
{
	XcursorImage * image = XcursorImageCreate (sourceRect.getWidth(), sourceRect.getHeight());
	image->xhot = hotspot.X;
	image->yhot = hotspot.Y;

	// write texture into XcursorImage
	asset::E_FORMAT format = tex->getColorFormat();
	uint32_t bytesPerPixel = video::getBitsPerPixelFromFormat(format) / 8;
	uint32_t bytesLeftGap = sourceRect.UpperLeftCorner.X * bytesPerPixel;
	uint32_t bytesRightGap = tex->getPitch() - sourceRect.LowerRightCorner.X * bytesPerPixel;
	XcursorPixel* target = image->pixels;
	const uint8_t* data = (const uint8_t*)tex->lock(video::ETLM_READ_ONLY, 0);
	data += sourceRect.UpperLeftCorner.Y*tex->getPitch();
	for ( int32_t y = 0; y < sourceRect.getHeight(); ++y )
	{
		data += bytesLeftGap;
		for ( int32_t x = 0; x < sourceRect.getWidth(); ++x )
		{
			video::SColor pixelCol;
			pixelCol.setData((const void*)data, format);
			data += bytesPerPixel;

			*target = (XcursorPixel)pixelCol.color;
			++target;
		}
		data += bytesRightGap;
	}
	tex->unlock();

	Cursor cursorResult=XcursorImageLoadCursor(display, image);

	XcursorImageDestroy(image);


	return cursorResult;
}
#endif // #ifdef _IRR_LINUX_XCURSOR_

Cursor CIrrDeviceLinux::TextureToCursor(irr::video::IImage * tex, const core::rect<int32_t>& sourceRect, const core::position2d<int32_t> &hotspot)
{
#ifdef _IRR_LINUX_XCURSOR_
	return TextureToARGBCursor( tex, sourceRect, hotspot );
#else
	return TextureToMonochromeCursor( tex, sourceRect, hotspot );
#endif
}
#endif	// _IRR_COMPILE_WITH_X11_


CIrrDeviceLinux::CCursorControl::CCursorControl(CIrrDeviceLinux* dev, bool null)
	: Device(dev)
#ifdef _IRR_COMPILE_WITH_X11_
	, PlatformBehavior(gui::ECPB_NONE), lastQuery(0)
#endif
	, IsVisible(true), Null(null), UseReferenceRect(false)
	, ActiveIcon(gui::ECI_NORMAL), ActiveIconStartTime(0)
{
#ifdef _IRR_COMPILE_WITH_X11_
	if (!Null)
	{
		XGCValues values;
		unsigned long valuemask = 0;

		XColor fg, bg;

		// this code, for making the cursor invisible was sent in by
		// Sirshane, thank your very much!


		Pixmap invisBitmap = XCreatePixmap(Device->display, Device->window, 32, 32, 1);
		Pixmap maskBitmap = XCreatePixmap(Device->display, Device->window, 32, 32, 1);
		Colormap screen_colormap = DefaultColormap( Device->display, DefaultScreen( Device->display ) );
		XAllocNamedColor( Device->display, screen_colormap, "black", &fg, &fg );
		XAllocNamedColor( Device->display, screen_colormap, "white", &bg, &bg );

		GC gc = XCreateGC( Device->display, invisBitmap, valuemask, &values );

		XSetForeground( Device->display, gc, BlackPixel( Device->display, DefaultScreen( Device->display ) ) );
		XFillRectangle( Device->display, invisBitmap, gc, 0, 0, 32, 32 );
		XFillRectangle( Device->display, maskBitmap, gc, 0, 0, 32, 32 );

		invisCursor = XCreatePixmapCursor( Device->display, invisBitmap, maskBitmap, &fg, &bg, 1, 1 );
		XFreeGC(Device->display, gc);
		XFreePixmap(Device->display, invisBitmap);
		XFreePixmap(Device->display, maskBitmap);

		initCursors();
	}
#endif
}

CIrrDeviceLinux::CCursorControl::~CCursorControl()
{
	// Do not clearCursors here as the display is already closed
	// TODO (cutealien): droping cursorcontrol earlier might work, not sure about reason why that's done in stub currently.
}

#ifdef _IRR_COMPILE_WITH_X11_
void CIrrDeviceLinux::CCursorControl::clearCursors()
{
	if (!Null)
		XFreeCursor(Device->display, invisCursor);
	for ( uint32_t i=0; i < Cursors.size(); ++i )
	{
		for ( uint32_t f=0; f < Cursors[i].Frames.size(); ++f )
		{
			XFreeCursor(Device->display, Cursors[i].Frames[f].IconHW);
		}
	}
}

void CIrrDeviceLinux::CCursorControl::initCursors()
{
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_top_left_arrow)) ); //  (or XC_arrow?)
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_crosshair)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_hand2)) ); // (or XC_hand1? )
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_question_arrow)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_xterm)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_X_cursor)) );	//  (or XC_pirate?)
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_watch)) );	// (or XC_clock?)
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_fleur)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_top_right_corner)) );	// NESW not available in X11
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_top_left_corner)) );	// NWSE not available in X11
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_sb_v_double_arrow)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_sb_h_double_arrow)) );
	Cursors.push_back( CursorX11(XCreateFontCursor(Device->display, XC_sb_up_arrow)) );	// (or XC_center_ptr?)
}

void CIrrDeviceLinux::CCursorControl::update()
{
	if ( (uint32_t)ActiveIcon < Cursors.size() && !Cursors[ActiveIcon].Frames.empty() && Cursors[ActiveIcon].FrameTime )
	{
		// update animated cursors. This could also be done by X11 in case someone wants to figure that out (this way was just easier to implement)
		uint32_t now = Device->getTimer()->getRealTime();
		uint32_t frame = ((now - ActiveIconStartTime) / Cursors[ActiveIcon].FrameTime) % Cursors[ActiveIcon].Frames.size();
		XDefineCursor(Device->display, Device->window, Cursors[ActiveIcon].Frames[frame].IconHW);
	}
}
#endif

//! Sets the active cursor icon
void CIrrDeviceLinux::CCursorControl::setActiveIcon(gui::ECURSOR_ICON iconId)
{
#ifdef _IRR_COMPILE_WITH_X11_
	if ( iconId >= (int32_t)Cursors.size() )
		return;

	if ( Cursors[iconId].Frames.size() )
		XDefineCursor(Device->display, Device->window, Cursors[iconId].Frames[0].IconHW);

	ActiveIconStartTime = Device->getTimer()->getRealTime();
	ActiveIcon = iconId;
#endif
}


//! Add a custom sprite as cursor icon.
gui::ECURSOR_ICON CIrrDeviceLinux::CCursorControl::addIcon(const gui::SCursorSprite& icon)
{/**
#ifdef _IRR_COMPILE_WITH_X11_
	if ( icon.SpriteId >= 0 )
	{
		CursorX11 cX11;
		cX11.FrameTime = icon.SpriteBank->getSprites()[icon.SpriteId].frameTime;
		for ( uint32_t i=0; i < icon.SpriteBank->getSprites()[icon.SpriteId].Frames.size(); ++i )
		{
			uint32_t texId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].textureNumber;
			uint32_t rectId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].rectNumber;
			irr::core::rect<int32_t> rectIcon = icon.SpriteBank->getPositions()[rectId];
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
#ifdef _IRR_COMPILE_WITH_X11_
	if ( iconId >= (int32_t)Cursors.size() )
		return;

	for ( uint32_t i=0; i < Cursors[iconId].Frames.size(); ++i )
		XFreeCursor(Device->display, Cursors[iconId].Frames[i].IconHW);

	if ( icon.SpriteId >= 0 )
	{
		CursorX11 cX11;
		cX11.FrameTime = icon.SpriteBank->getSprites()[icon.SpriteId].frameTime;
		for ( uint32_t i=0; i < icon.SpriteBank->getSprites()[icon.SpriteId].Frames.size(); ++i )
		{
			uint32_t texId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].textureNumber;
			uint32_t rectId = icon.SpriteBank->getSprites()[icon.SpriteId].Frames[i].rectNumber;
			irr::core::rect<int32_t> rectIcon = icon.SpriteBank->getPositions()[rectId];
			Cursor cursor = Device->TextureToCursor(icon.SpriteBank->getTexture(texId), rectIcon, icon.HotSpot);
			cX11.Frames.push_back( CursorFrameX11(cursor) );
		}

		Cursors[iconId] = cX11;
	}
#endif**/
}


irr::core::dimension2di CIrrDeviceLinux::CCursorControl::getSupportedIconSize() const
{
	// this returns the closest match that is smaller or same size, so we just pass a value which should be large enough for cursors
	unsigned int width=0, height=0;
#ifdef _IRR_COMPILE_WITH_X11_
	XQueryBestCursor(Device->display, Device->window, 64, 64, &width, &height);
#endif
	return core::dimension2di(width, height);
}

} // end namespace

#endif // _IRR_COMPILE_WITH_X11_DEVICE_


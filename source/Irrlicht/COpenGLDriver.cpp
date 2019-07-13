// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "COpenGLDriver.h"
// needed here also because of the create methods' parameters
#include "CNullDriver.h"
#include "irr/video/CGPUSkinnedMesh.h"

#include "vectorSIMD.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGL1DTexture.h"
#include "COpenGL1DTextureArray.h"
#include "COpenGL2DTexture.h"
#include "COpenGL3DTexture.h"
#include "COpenGL2DTextureArray.h"
#include "COpenGLCubemapTexture.h"
#include "COpenGLCubemapArrayTexture.h"
#include "COpenGLMultisampleTexture.h"
#include "COpenGLMultisampleTextureArray.h"
#include "COpenGLTextureBufferObject.h"

#include "COpenGLBuffer.h"
#include "COpenGLFrameBuffer.h"
#include "COpenGLSLMaterialRenderer.h"
#include "COpenGLQuery.h"
#include "COpenGLTimestampQuery.h"
#include "os.h"

#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
#include "MacOSX/CIrrDeviceMacOSX.h"
#endif

#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
#include "CIrrDeviceSDL.h"
#include <SDL/SDL.h>
#endif

#if defined(_IRR_COMPILE_WITH_WINDOWS_DEVICE_)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "CIrrDeviceWin32.h"
#elif defined(_IRR_COMPILE_WITH_X11_DEVICE_)
#include "CIrrDeviceLinux.h"
#include <dlfcn.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#ifdef _IRR_LINUX_X11_RANDR_
#include <X11/extensions/Xrandr.h>
#endif
#endif

namespace irr
{
namespace video
{

//: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
//	CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(asset::EF_R8G8B8_UNORM), Params(params),
// -----------------------------------------------------------------------
// WINDOWS CONSTRUCTOR
// -----------------------------------------------------------------------
#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
//! Windows constructor and init code
COpenGLDriver::COpenGLDriver(const irr::SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceWin32* device)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
	runningInRenderDoc(false),  CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(asset::EF_R8G8B8_UNORM), Params(params),
	HDc(0), Window(static_cast<HWND>(params.WindowId)), Win32Device(device),
	DeviceType(EIDT_WIN32), AuxContexts(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("COpenGLDriver");
	#endif
}

bool COpenGLDriver::changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceWin32* device)
{
	if (videoData.OpenGLWin32.HWnd && videoData.OpenGLWin32.HDc && videoData.OpenGLWin32.HRc)
	{
		if (!wglMakeCurrent((HDC)videoData.OpenGLWin32.HDc, (HGLRC)videoData.OpenGLWin32.HRc))
		{
			os::Printer::log("Render Context switch failed.");
			return false;
		}
		else
		{
			HDc = (HDC)videoData.OpenGLWin32.HDc;
		}
	}
	// set back to main context
	else if (HDc != ExposedData.OpenGLWin32.HDc)
	{
		if (!wglMakeCurrent((HDC)ExposedData.OpenGLWin32.HDc, (HGLRC)ExposedData.OpenGLWin32.HRc))
		{
			os::Printer::log("Render Context switch failed.");
			return false;
		}
		else
		{
			HDc = (HDC)ExposedData.OpenGLWin32.HDc;
		}
	}
	return true;
}

//! inits the open gl driver
bool COpenGLDriver::initDriver(CIrrDeviceWin32* device)
{
	// Create a window to test antialiasing support
	const char* ClassName = __TEXT("GLCIrrDeviceWin32");
	HINSTANCE lhInstance = GetModuleHandle(0);

	// Register Class
	WNDCLASSEX wcex;
	wcex.cbSize        = sizeof(WNDCLASSEX);
	wcex.style         = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc   = (WNDPROC)DefWindowProc;
	wcex.cbClsExtra    = 0;
	wcex.cbWndExtra    = 0;
	wcex.hInstance     = lhInstance;
	wcex.hIcon         = NULL;
	wcex.hCursor       = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName  = 0;
	wcex.lpszClassName = ClassName;
	wcex.hIconSm       = 0;
	wcex.hIcon         = 0;
	RegisterClassEx(&wcex);

	RECT clientSize;
	clientSize.top = 0;
	clientSize.left = 0;
	clientSize.right = Params.WindowSize.Width;
	clientSize.bottom = Params.WindowSize.Height;

	DWORD style = WS_POPUP;
	if (!Params.Fullscreen)
		style = WS_SYSMENU | WS_BORDER | WS_CAPTION | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;

	AdjustWindowRect(&clientSize, style, FALSE);

	const int32_t realWidth = clientSize.right - clientSize.left;
	const int32_t realHeight = clientSize.bottom - clientSize.top;

	const int32_t windowLeft = (GetSystemMetrics(SM_CXSCREEN) - realWidth) / 2;
	const int32_t windowTop = (GetSystemMetrics(SM_CYSCREEN) - realHeight) / 2;

	HWND temporary_wnd=CreateWindow(ClassName, __TEXT(""), style, windowLeft,
			windowTop, realWidth, realHeight, NULL, NULL, lhInstance, NULL);

	if (!temporary_wnd)
	{
		os::Printer::log("Cannot create a temporary window.", ELL_ERROR);
		UnregisterClass(ClassName, lhInstance);
		return false;
	}

	HDc = GetDC(temporary_wnd);

	// Set up pixel format descriptor with desired parameters
	PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR),             // Size Of This Pixel Format Descriptor
		1,                                         // Version Number
		PFD_DRAW_TO_WINDOW |                       // Format Must Support Window
		PFD_SUPPORT_OPENGL |                       // Format Must Support OpenGL
		(Params.Doublebuffer?PFD_DOUBLEBUFFER:0) | // Must Support Double Buffering
		(Params.Stereobuffer?PFD_STEREO:0),        // Must Support Stereo Buffer
		PFD_TYPE_RGBA,                             // Request An RGBA Format
		Params.Bits,                               // Select Our Color Depth
		0, 0, 0, 0, 0, 0,                          // Color Bits Ignored
		0,                                         // No Alpha Buffer
		0,                                         // Shift Bit Ignored
		0,                                         // No Accumulation Buffer
		0, 0, 0, 0,	                               // Accumulation Bits Ignored
		Params.ZBufferBits,                        // Z-Buffer (Depth Buffer)
		BYTE(Params.Stencilbuffer ? 1 : 0),        // Stencil Buffer Depth
		0,                                         // No Auxiliary Buffer
		PFD_MAIN_PLANE,                            // Main Drawing Layer
		0,                                         // Reserved
		0, 0, 0                                    // Layer Masks Ignored
	};

	GLuint PixelFormat;

	for (uint32_t i=0; i<6; ++i)
	{
		if (i == 1)
		{
			if (Params.Stencilbuffer)
			{
				os::Printer::log("Cannot create a GL device with stencil buffer, disabling stencil shadows.", ELL_WARNING);
				Params.Stencilbuffer = false;
				pfd.cStencilBits = 0;
			}
			else
				continue;
		}
		else
		if (i == 2)
		{
			pfd.cDepthBits = 24;
		}
		else
		if (i == 3)
		{
			if (Params.Bits!=16)
				pfd.cDepthBits = 16;
			else
				continue;
		}
		else
		if (i == 4)
		{
			// try single buffer
			if (Params.Doublebuffer)
				pfd.dwFlags &= ~PFD_DOUBLEBUFFER;
			else
				continue;
		}
		else
		if (i == 5)
		{
			os::Printer::log("Cannot create a GL device context", "No suitable format for temporary window.", ELL_ERROR);
			ReleaseDC(temporary_wnd, HDc);
			DestroyWindow(temporary_wnd);
			UnregisterClass(ClassName, lhInstance);
			return false;
		}

		// choose pixelformat
		PixelFormat = ChoosePixelFormat(HDc, &pfd);
		if (PixelFormat)
			break;
	}

	SetPixelFormat(HDc, PixelFormat, &pfd);
	HGLRC hrc=wglCreateContext(HDc);
	if (!hrc)
	{
		os::Printer::log("Cannot create a temporary GL rendering context.", ELL_ERROR);
		ReleaseDC(temporary_wnd, HDc);
		DestroyWindow(temporary_wnd);
		UnregisterClass(ClassName, lhInstance);
		return false;
	}

	SExposedVideoData data;
	data.OpenGLWin32.HDc = HDc;
	data.OpenGLWin32.HRc = hrc;
	data.OpenGLWin32.HWnd = temporary_wnd;


	if (!changeRenderContext(data, device))
	{
		os::Printer::log("Cannot activate a temporary GL rendering context.", ELL_ERROR);
		wglDeleteContext(hrc);
		ReleaseDC(temporary_wnd, HDc);
		DestroyWindow(temporary_wnd);
		UnregisterClass(ClassName, lhInstance);
		return false;
	}

	core::stringc wglExtensions;
#ifdef WGL_ARB_extensions_string
	PFNWGLGETEXTENSIONSSTRINGARBPROC irrGetExtensionsString = (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
	if (irrGetExtensionsString)
		wglExtensions = irrGetExtensionsString(HDc);
#elif defined(WGL_EXT_extensions_string)
	PFNWGLGETEXTENSIONSSTRINGEXTPROC irrGetExtensionsString = (PFNWGLGETEXTENSIONSSTRINGEXTPROC)wglGetProcAddress("wglGetExtensionsStringEXT");
	if (irrGetExtensionsString)
		wglExtensions = irrGetExtensionsString(HDc);
#endif
	const bool pixel_format_supported = (wglExtensions.find("WGL_ARB_pixel_format") != -1);
#ifdef _IRR_DEBUG
	os::Printer::log("WGL_extensions", wglExtensions.c_str());
#endif

#ifdef WGL_ARB_pixel_format
	PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormat_ARB = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
	if (pixel_format_supported && wglChoosePixelFormat_ARB)
	{
		float fAttributes[] = {0.0, 0.0};
		int32_t iAttributes[] =
		{
			WGL_DRAW_TO_WINDOW_ARB,1,
			WGL_SUPPORT_OPENGL_ARB,1,
			WGL_ACCELERATION_ARB,WGL_FULL_ACCELERATION_ARB,
			WGL_COLOR_BITS_ARB,(Params.Bits==32) ? 24 : 15,
			WGL_ALPHA_BITS_ARB,(Params.Bits==32) ? 8 : 1,
			WGL_DEPTH_BITS_ARB,Params.ZBufferBits, // 10,11
			WGL_STENCIL_BITS_ARB,Params.Stencilbuffer ? 1 : 0,
			WGL_DOUBLE_BUFFER_ARB,Params.Doublebuffer ? 1 : 0,
			WGL_STEREO_ARB,Params.Stereobuffer ? 1 : 0,
			WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
			WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB, 1,
			0,0,0,0
		};

		// Try to get an acceptable pixel format
        int pixelFormat=0;
        UINT numFormats=0;
        const BOOL valid = wglChoosePixelFormat_ARB(HDc,iAttributes,fAttributes,1,&pixelFormat,&numFormats);
        if (valid && numFormats && pixelFormat)
            PixelFormat = pixelFormat;
	}
#endif

	PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribs_ARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
	wglMakeCurrent(HDc, NULL);
	wglDeleteContext(hrc);
	ReleaseDC(temporary_wnd, HDc);
	DestroyWindow(temporary_wnd);
	UnregisterClass(ClassName, lhInstance);

	if (!wglCreateContextAttribs_ARB)
	{
		os::Printer::log("Couldn't get wglCreateContextAttribs_ARB address.", ELL_ERROR);
		return false;
	}

	// get hdc
	HDc=GetDC(Window);
	if (!HDc)
	{
		os::Printer::log("Cannot create a GL device context.", ELL_ERROR);
		return false;
	}

	// search for pixel format the simple way
	if (PixelFormat==0 || (!SetPixelFormat(HDc, PixelFormat, &pfd)))
	{
		for (uint32_t i=0; i<5; ++i)
		{
			if (i == 1)
			{
				if (Params.Stencilbuffer)
				{
					os::Printer::log("Cannot create a GL device with stencil buffer, disabling stencil shadows.", ELL_WARNING);
					Params.Stencilbuffer = false;
					pfd.cStencilBits = 0;
				}
				else
					continue;
			}
			else
			if (i == 2)
			{
				pfd.cDepthBits = 24;
			}
			if (i == 3)
			{
				if (Params.Bits!=16)
					pfd.cDepthBits = 16;
				else
					continue;
			}
			else
			if (i == 4)
			{
				os::Printer::log("Cannot create a GL device context", "No suitable format.", ELL_ERROR);
				return false;
			}

			// choose pixelformat
			PixelFormat = ChoosePixelFormat(HDc, &pfd);
			if (PixelFormat)
				break;
		}

        // set pixel format
        if (!SetPixelFormat(HDc, PixelFormat, &pfd))
        {
            os::Printer::log("Cannot set the pixel format.", ELL_ERROR);
            return false;
        }
    }
	os::Printer::log("Pixel Format", std::to_string(PixelFormat), ELL_DEBUG);

	int iAttribs[] =
	{
		WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
		WGL_CONTEXT_MINOR_VERSION_ARB, 6,
		WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};
	// create rendering context
	hrc=wglCreateContextAttribs_ARB(HDc, 0, iAttribs);
	if (!hrc)
	{
		iAttribs[3] = 5;
		hrc=wglCreateContextAttribs_ARB(HDc, 0, iAttribs);
	}
	if (!hrc)
	{
		iAttribs[3] = 4;
		hrc=wglCreateContextAttribs_ARB(HDc, 0, iAttribs);
	}
	if (!hrc)
	{
		iAttribs[3] = 3;
		hrc=wglCreateContextAttribs_ARB(HDc, 0, iAttribs);
	}

	if (!hrc)
	{
		os::Printer::log("Cannot create a GL rendering context.", ELL_ERROR);
		return false;
	}

    AuxContexts = _IRR_NEW_ARRAY(SAuxContext,Params.AuxGLContexts+1);
    {
        AuxContexts[0].threadId = std::this_thread::get_id();
        AuxContexts[0].ctx = hrc;
    }
	for (size_t i=1; i<=Params.AuxGLContexts; i++)
    {
        AuxContexts[i].threadId = std::thread::id(); //invalid ID
        AuxContexts[i].ctx = wglCreateContextAttribs_ARB(HDc, hrc, iAttribs);
    }

	// set exposed data
	ExposedData.OpenGLWin32.HDc = HDc;
	ExposedData.OpenGLWin32.HRc = hrc;
	ExposedData.OpenGLWin32.HWnd = Window;

	// activate rendering context
	if (!changeRenderContext(ExposedData, device))
	{
		os::Printer::log("Cannot activate GL rendering context", ELL_ERROR);
		wglDeleteContext(hrc);
		_IRR_DELETE_ARRAY(AuxContexts,Params.AuxGLContexts+1);
		return false;
	}


	int pf = GetPixelFormat(HDc);
	DescribePixelFormat(HDc, pf, sizeof(PIXELFORMATDESCRIPTOR), &pfd);
	if (pfd.cAlphaBits != 0)
	{
		if (pfd.cRedBits == 8)
			ColorFormat = asset::EF_B8G8R8A8_UNORM;
		else
			ColorFormat = asset::EF_A1R5G5B5_UNORM_PACK16;
	}
	else
	{
		if (pfd.cRedBits == 8)
			ColorFormat = asset::EF_R8G8B8_UNORM;
		else
			ColorFormat = asset::EF_B5G6R5_UNORM_PACK16;
	}

#ifdef _IRR_COMPILE_WITH_OPENCL_
	ocl::COpenCLHandler::getCLDeviceFromGLContext(clDevice,hrc,HDc);
#endif // _IRR_COMPILE_WITH_OPENCL_
	genericDriverInit();

	extGlSwapInterval(Params.Vsync ? 1 : 0);
	return true;
}

bool COpenGLDriver::initAuxContext()
{
	if (!AuxContexts) // opengl dead and never inited
		return false;

    bool retval = false;
    glContextMutex->Get();
    SAuxContext* found = getThreadContext_helper(true,std::thread::id());
    if (found)
    {
        retval = wglMakeCurrent((HDC)ExposedData.OpenGLWin32.HDc,found->ctx);
        if (retval)
            found->threadId = std::this_thread::get_id();
    }
    glContextMutex->Release();
    return retval;
}

bool COpenGLDriver::deinitAuxContext()
{
    bool retval = false;
    glContextMutex->Get();
    SAuxContext* found = getThreadContext_helper(true);
    if (found)
    {
        glContextMutex->Release();
        cleanUpContextBeforeDelete();
        glContextMutex->Get();
        retval = wglMakeCurrent(NULL,NULL);
        if (retval)
            found->threadId = std::thread::id();
    }
    glContextMutex->Release();
    return retval;
}

#endif // _IRR_COMPILE_WITH_WINDOWS_DEVICE_

// -----------------------------------------------------------------------
// MacOSX CONSTRUCTOR
// -----------------------------------------------------------------------
#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
//! Windows constructor and init code
COpenGLDriver::COpenGLDriver(const SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceMacOSX *device)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
    runningInRenderDoc(false), CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(asset::EF_R8G8B8_UNORM),
	Params(params),
	OSXDevice(device), DeviceType(EIDT_OSX), AuxContexts(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("COpenGLDriver");
	#endif

	genericDriverInit();
}

#endif

// -----------------------------------------------------------------------
// LINUX CONSTRUCTOR
// -----------------------------------------------------------------------
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
//! Linux constructor and init code
COpenGLDriver::COpenGLDriver(const SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceLinux* device)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
	runningInRenderDoc(false),  CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(asset::EF_R8G8B8_UNORM),
	Params(params), X11Device(device), DeviceType(EIDT_X11), AuxContexts(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("COpenGLDriver");
	#endif
}


bool COpenGLDriver::changeRenderContext(const SExposedVideoData& videoData, CIrrDeviceLinux* device)
{
	if (videoData.OpenGLLinux.X11Window)
	{
		if (videoData.OpenGLLinux.X11Display && videoData.OpenGLLinux.X11Context)
		{
			if (!glXMakeCurrent((Display*)videoData.OpenGLLinux.X11Display, videoData.OpenGLLinux.X11Window, (GLXContext)videoData.OpenGLLinux.X11Context))
			{
				os::Printer::log("Render Context switch failed.");
				return false;
			}
			else
			{
				Drawable = videoData.OpenGLLinux.X11Window;
				X11Display = (Display*)videoData.OpenGLLinux.X11Display;
			}
		}
		else
		{
			// in case we only got a window ID, try with the existing values for display and context
			if (!glXMakeCurrent((Display*)ExposedData.OpenGLLinux.X11Display, videoData.OpenGLLinux.X11Window, (GLXContext)ExposedData.OpenGLLinux.X11Context))
			{
				os::Printer::log("Render Context switch failed.");
				return false;
			}
			else
			{
				Drawable = videoData.OpenGLLinux.X11Window;
				X11Display = (Display*)ExposedData.OpenGLLinux.X11Display;
			}
		}
	}
	// set back to main context
	else if (X11Display != ExposedData.OpenGLLinux.X11Display)
	{
		if (!glXMakeCurrent((Display*)ExposedData.OpenGLLinux.X11Display, ExposedData.OpenGLLinux.X11Window, (GLXContext)ExposedData.OpenGLLinux.X11Context))
		{
			os::Printer::log("Render Context switch failed.");
			return false;
		}
		else
		{
			Drawable = ExposedData.OpenGLLinux.X11Window;
			X11Display = (Display*)ExposedData.OpenGLLinux.X11Display;
		}
	}
	return true;
}


//! inits the open gl driver
bool COpenGLDriver::initDriver(CIrrDeviceLinux* device, SAuxContext* auxCtxts)
{
	ExposedData.OpenGLLinux.X11Context = glXGetCurrentContext();
	ExposedData.OpenGLLinux.X11Display = glXGetCurrentDisplay();
	ExposedData.OpenGLLinux.X11Window = (unsigned long)Params.WindowId;
	Drawable = glXGetCurrentDrawable();
	X11Display = (Display*)ExposedData.OpenGLLinux.X11Display;

    AuxContexts = auxCtxts;

#ifdef _IRR_COMPILE_WITH_OPENCL_
	if (!ocl::COpenCLHandler::getCLDeviceFromGLContext(clDevice,reinterpret_cast<GLXContext&>(ExposedData.OpenGLLinux.X11Context),(Display*)ExposedData.OpenGLLinux.X11Display))
        os::Printer::log("Couldn't find matching OpenCL device.\n");
#endif // _IRR_COMPILE_WITH_OPENCL_

	genericDriverInit();

	// set vsync
	//if (queryOpenGLFeature(IRR_))
        extGlSwapInterval(Params.Vsync ? -1 : 0);
	return true;
}

bool COpenGLDriver::initAuxContext()
{
	if (!AuxContexts) // opengl dead and never inited
		return false;

    bool retval = false;
    glContextMutex->Get();
    SAuxContext* found = getThreadContext_helper(true,std::thread::id());
    if (found)
    {
        retval = glXMakeCurrent((Display*)ExposedData.OpenGLLinux.X11Display, found->pbuff, found->ctx);
        if (retval)
            found->threadId = std::this_thread::get_id();
    }
    glContextMutex->Release();
    return retval;
}

bool COpenGLDriver::deinitAuxContext()
{
	if (!AuxContexts) // opengl dead and never inited
		return false;

    bool retval = false;
    glContextMutex->Get();
    SAuxContext* found = getThreadContext_helper(true);
    if (found)
    {
        glContextMutex->Release();
        cleanUpContextBeforeDelete();
        glContextMutex->Get();
        retval = glXMakeCurrent((Display*)ExposedData.OpenGLLinux.X11Display, None, NULL);
        if (retval)
            found->threadId = std::thread::id();
    }
    glContextMutex->Release();
    return retval;
}

#endif // _IRR_COMPILE_WITH_X11_DEVICE_


// -----------------------------------------------------------------------
// SDL CONSTRUCTOR
// -----------------------------------------------------------------------
#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
//! SDL constructor and init code
COpenGLDriver::COpenGLDriver(const SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceSDL* device)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
    runningInRenderDoc(false), CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(EF_R8G8B8_UNORM),
	CurrentTarget(ERT_FRAME_BUFFER), Params(params),
	SDLDevice(device), DeviceType(EIDT_SDL), AuxContexts(0)
{
	#ifdef _IRR_DEBUG
	setDebugName("COpenGLDriver");
	#endif

	genericDriverInit();
}

#endif // _IRR_COMPILE_WITH_SDL_DEVICE_


//! destructor
COpenGLDriver::~COpenGLDriver()
{
	if (!AuxContexts) //opengl dead and never initialized in the first place
		return;

    cleanUpContextBeforeDelete();

	deleteMaterialRenders();

    //! Spin wait for other contexts to deinit
    //! @TODO: Change trylock to semaphore
	while (true)
    {
        while (!glContextMutex->TryLock()) {}

        bool allDead = true;
        for (size_t i=1; i<=Params.AuxGLContexts; i++)
        {
            if (AuxContexts[i].threadId==std::thread::id())
                continue;

            // found one alive
            glContextMutex->Release();
            allDead = false;
            break;
        }

        if (allDead)
            break;
    }

#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
	if (DeviceType == EIDT_WIN32)
	{
        for (size_t i=1; i<=Params.AuxGLContexts; i++)
            wglDeleteContext(AuxContexts[i].ctx);

		if (ExposedData.OpenGLWin32.HRc)
		{
			if (!wglMakeCurrent(HDc, 0))
				os::Printer::log("Release of dc and rc failed.", ELL_WARNING);

			if (!wglDeleteContext((HGLRC)ExposedData.OpenGLWin32.HRc))
				os::Printer::log("Release of rendering context failed.", ELL_WARNING);
		}

		if (HDc)
			ReleaseDC(Window, HDc);

        //if (!ExternalWindow)
        //{
        //    DestroyWindow(temporary_wnd);
        //    UnregisterClass(ClassName, lhInstance);
        //}
	}
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
	else
#endif // _IRR_COMPILE_WITH_X11_DEVICE_
#endif
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
    if (DeviceType == EIDT_X11)
    {
        for (size_t i=1; i<=Params.AuxGLContexts; i++)
        {
            assert(AuxContexts[i].threadId==std::thread::id());
            glXDestroyPbuffer((Display*)ExposedData.OpenGLLinux.X11Display,AuxContexts[i].pbuff);
            glXDestroyContext((Display*)ExposedData.OpenGLLinux.X11Display,AuxContexts[i].ctx);
        }
    }
#endif // _IRR_COMPILE_WITH_X11_DEVICE_
    _IRR_DELETE_ARRAY(AuxContexts,Params.AuxGLContexts+1);
    glContextMutex->Release();
    _IRR_DELETE(glContextMutex);
}


// -----------------------------------------------------------------------
// METHODS
// -----------------------------------------------------------------------

uint16_t COpenGLDriver::retrieveDisplayRefreshRate() const
{
#if defined(_IRR_COMPILE_WITH_WINDOWS_DEVICE_)
    DEVMODEA dm;
    dm.dmSize = sizeof(DEVMODE);
    dm.dmDriverExtra = 0;
    if (!EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm))
        return 0u;
    return dm.dmDisplayFrequency;
#elif defined(_IRR_COMPILE_WITH_X11_DEVICE_)
#   ifdef _IRR_LINUX_X11_RANDR_
    Display* disp = XOpenDisplay(NULL);
    Window root = RootWindow(disp, 0);

    XRRScreenConfiguration* conf = XRRGetScreenInfo(disp, root);
    uint16_t rate = XRRConfigCurrentRate(conf);

    return rate;
#   else
#       ifdef _IRR_DEBUG
    os::Printer::log("Refresh rate retrieval without Xrandr compiled in is not supprted!\n", ELL_WARNING);
#       endif
    return 0u;
#   endif // _IRR_LINUX_X11_RANDR_
#else
    return 0u;
#endif
}

const COpenGLDriver::SAuxContext* COpenGLDriver::getThreadContext(const std::thread::id& tid) const
{
    glContextMutex->Get();
    for (size_t i=0; i<=Params.AuxGLContexts; i++)
    {
        if (AuxContexts[i].threadId==tid)
        {
            glContextMutex->Release();
            return AuxContexts+i;
        }
    }
    glContextMutex->Release();
    return NULL;
}

COpenGLDriver::SAuxContext* COpenGLDriver::getThreadContext_helper(const bool& alreadyLockedMutex, const std::thread::id& tid)
{
    if (!alreadyLockedMutex)
        glContextMutex->Get();
    for (size_t i=0; i<=Params.AuxGLContexts; i++)
    {
        if (AuxContexts[i].threadId==tid)
        {
            if (!alreadyLockedMutex)
                glContextMutex->Release();
            return AuxContexts+i;
        }
    }
    if (!alreadyLockedMutex)
        glContextMutex->Release();
    return NULL;
}

void COpenGLDriver::cleanUpContextBeforeDelete()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    found->CurrentRendertargetSize = ScreenSize;
    extGlBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (found->CurrentFBO)
    {
        found->CurrentFBO->drop();
        found->CurrentFBO = NULL;
    }

    extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);
    if (found->CurrentXFormFeedback)
    {
        if (!found->CurrentXFormFeedback->isEnded())
        {
            assert(found->CurrentXFormFeedback->isActive());
            found->CurrentXFormFeedback->endFeedback();
            found->XFormFeedbackRunning = false;
        }

        found->CurrentXFormFeedback->drop();
		found->CurrentXFormFeedback = NULL;
    }


    extGlUseProgram(0);
    removeAllFrameBuffers();

    extGlBindVertexArray(0);
    found->CurrentVAO = std::pair<COpenGLVAOSpec::HashAttribs,SAuxContext::COpenGLVAO*>(COpenGLVAOSpec::HashAttribs(),nullptr);
	for(auto it = found->VAOMap.begin(); it != found->VAOMap.end(); it++)
    {
        delete it->second;
    }
    found->VAOMap.clear();

	found->CurrentTexture.clear();

	for(core::unordered_map<uint64_t,GLuint>::iterator it = found->SamplerMap.begin(); it != found->SamplerMap.end(); it++)
    {
        extGlDeleteSamplers(1,&it->second);
    }
    found->SamplerMap.clear();

    glFinish();
}


bool COpenGLDriver::genericDriverInit()
{
	if (!AuxContexts) // opengl dead and never inited
		return false;

    glContextMutex = _IRR_NEW(FW_Mutex);

#ifdef _IRR_WINDOWS_API_
    if (GetModuleHandleA("renderdoc.dll"))
#elif defined(_IRR_ANDROID_PLATFORM_)
    if (dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD))
#elif defined(_IRR_LINUX_PLATFORM_)
    if (dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
#else
    if (false)
#endif // LINUX
        runningInRenderDoc = true;

	Name=L"OpenGL ";
	Name.append(glGetString(GL_VERSION));
	int32_t pos=Name.findNext(L' ', 7);
	if (pos != -1)
		Name=Name.subString(0, pos);
	printVersion();

	// print renderer information
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* vendor = glGetString(GL_VENDOR);
	if (renderer && vendor)
	{
		os::Printer::log(reinterpret_cast<const char*>(renderer), reinterpret_cast<const char*>(vendor), ELL_INFORMATION);
		VendorName = reinterpret_cast<const char*>(vendor);
	}


    maxConcurrentShaderInvocations = 4;
    maxALUShaderInvocations = 4;
    maxShaderComputeUnits = 1;
#ifdef _IRR_COMPILE_WITH_OPENCL_
    clPlatformIx = 0xdeadbeefu;
    clDeviceIx = 0xdeadbeefu;
    for (size_t i=0; i<ocl::COpenCLHandler::getPlatformCount(); i++)
    {
        const ocl::COpenCLHandler::SOpenCLPlatformInfo& platform = ocl::COpenCLHandler::getPlatformInfo(i);

        for (size_t j=0; j<platform.deviceCount; j++)
        {
            if (platform.devices[j]==clDevice)
            {
                clPlatformIx = i;
                clDeviceIx = j;
                maxALUShaderInvocations = platform.deviceInformation[j].MaxWorkGroupSize;
                maxShaderComputeUnits = platform.deviceInformation[j].MaxComputeUnits;
                maxConcurrentShaderInvocations = platform.deviceInformation[j].ProbableUnifiedShaders;
                break;
            }
        }
        if (clPlatformIx==i)
            break;
    }
#endif // _IRR_COMPILE_WITH_OPENCL_


	GLint num = 0;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &num);
	MaxTextureSizes[ITexture::ETT_1D][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_2D][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_2D][1] = static_cast<uint32_t>(num);

	MaxTextureSizes[ITexture::ETT_1D_ARRAY][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_2D_ARRAY][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_2D_ARRAY][1] = static_cast<uint32_t>(num);

	glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE , &num);
	MaxTextureSizes[ITexture::ETT_CUBE_MAP][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_CUBE_MAP][1] = static_cast<uint32_t>(num);

	MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][1] = static_cast<uint32_t>(num);

	glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &num);
	MaxTextureSizes[ITexture::ETT_1D_ARRAY][2] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_2D_ARRAY][2] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_CUBE_MAP_ARRAY][2] = static_cast<uint32_t>(num);

	glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &num);
	MaxTextureSizes[ITexture::ETT_3D][0] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_3D][1] = static_cast<uint32_t>(num);
	MaxTextureSizes[ITexture::ETT_3D][2] = static_cast<uint32_t>(num);


	glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE , &num);
	///MaxTextureSizes[ITexture::ETT_TEXTURE_BUFFER][0] = static_cast<uint32_t>(num);


	uint32_t i;
	// load extensions
	initExtensions(Params.Stencilbuffer);

    char buf[32];
    const uint32_t maj = ShaderLanguageVersion/10;
    snprintf(buf, 32, "%u.%u", maj, ShaderLanguageVersion-maj*10);
    os::Printer::log("GLSL version", buf, ELL_INFORMATION);

    if (Version<430)
    {
		os::Printer::log("OpenGL version is less than 4.3", ELL_ERROR);
		return false;
    }

	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// Reset The Current Viewport
	glViewport(0, 0, Params.WindowSize.Width, Params.WindowSize.Height);

	glEnable(GL_FRAMEBUFFER_SRGB);
    glDisable(GL_DITHER);
    glDisable(GL_MULTISAMPLE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    extGlClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
	glClearDepth(0.0);
	glDepthFunc(GL_GEQUAL);
	glDepthRange(1.0,0.0);
	glFrontFace(GL_CCW);

	// adjust flat coloring scheme to DirectX version
	///extGlProvokingVertex(GL_FIRST_VERTEX_CONVENTION_EXT);

	// create material renderers
	createMaterialRenderers();

	// set the renderstates
	setRenderStates3DMode();

	// We need to reset once more at the beginning of the first rendering.
	// This fixes problems with intermediate changes to the material during texture load.
	ResetRenderStates = true;

	// down
	{
        auto reqs = getDownStreamingMemoryReqs();
        reqs.vulkanReqs.size = Params.StreamingDownloadBufferSize;
        reqs.vulkanReqs.alignment = 64u*1024u; // if you need larger alignments then you're not right in the head
        defaultDownloadBuffer = new video::StreamingTransientDataBufferMT<>(this,reqs);
	}
	// up
	{
        auto reqs = getUpStreamingMemoryReqs();
        reqs.vulkanReqs.size = Params.StreamingUploadBufferSize;
        reqs.vulkanReqs.alignment = 64u*1024u; // if you need larger alignments then you're not right in the head
        defaultUploadBuffer = new video::StreamingTransientDataBufferMT<>(this,reqs);
	}

	return true;
}

class SimpleDummyCallBack : public video::IShaderConstantSetCallBack
{
protected:
    video::SConstantLocationNamePair mvpUniform[EMT_COUNT];
    video::E_MATERIAL_TYPE currentMatType;
public:
    SimpleDummyCallBack()
    {
        currentMatType = EMT_COUNT;
        for (size_t i=0; i<EMT_COUNT; i++)
            mvpUniform[i].location = -1;
    }
    virtual void OnUnsetMaterial()
    {
        currentMatType = EMT_COUNT;
    }
    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE &materialType, const core::vector<video::SConstantLocationNamePair> &constants)
    {
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="MVPMat")
            {
                mvpUniform[materialType] = constants[i];
                break;
            }
        }
    }
    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial &material, const video::SGPUMaterial &lastMaterial)
    {
        currentMatType = material.MaterialType;
	}
	virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
	{
	    if (currentMatType>=EMT_COUNT)
            return;

	    if (mvpUniform[currentMatType].location>=0)
            services->setShaderConstant(services->getVideoDriver()->getTransform(EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniform[currentMatType].location,mvpUniform[currentMatType].type);
	}
};

void COpenGLDriver::createMaterialRenderers()
{
	// create OpenGL material renderers
    const char* std_vert =
    "#version 430 core\n"
    "uniform mat4 MVPMat;\n"
    "layout(location = 0) in vec4 vPosAttr;\n"
    "layout(location = 2) in vec2 vTCAttr;\n"
    "layout(location = 1) in vec4 vColAttr;\n"
    "\n"
    "out vec4 vxCol;\n"
    "out vec2 tcCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
    "   gl_Position = MVPMat*vPosAttr;"
    "   vxCol = vColAttr;"
    "   tcCoord = vTCAttr;"
    "}";
    const char* std_solid_frag =
    "#version 430 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "layout(location = 0) uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   outColor = texture(tex0,tcCoord);"
    "}";
    const char* std_trans_add_frag =
    "#version 430 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "layout(location = 0) uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   outColor = texture(tex0,tcCoord);"
    "}";
    const char* std_trans_alpha_frag =
    "#version 430 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "layout(location = 0) uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   vec4 tmp = texture(tex0,tcCoord)*vxCol;\n"
    "   if (tmp.a<0.00000000000000000000000000000000001)\n"
    "       discard;\n"
    "   outColor = tmp;"
    "}";
    const char* std_trans_vertex_frag =
    "#version 430 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "layout(location = 0) uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   if (vxCol.a<0.00000000000000000000000000000000001)\n"
    "       discard;\n"
    "   outColor = vec4(texture(tex0,tcCoord).rgb,1.0)*vxCol;"
    "}";
    int32_t nr;

    SimpleDummyCallBack* sdCB = new SimpleDummyCallBack();

    COpenGLSLMaterialRenderer* rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_solid_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3,sdCB,EMT_SOLID);
    if (rdr)
        rdr->drop();

	rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_trans_add_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3,sdCB,EMT_TRANSPARENT_ADD_COLOR);
    if (rdr)
        rdr->drop();

	rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_trans_alpha_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3,sdCB,EMT_TRANSPARENT_ALPHA_CHANNEL);
    if (rdr)
        rdr->drop();

	rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_trans_vertex_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3,sdCB,EMT_TRANSPARENT_VERTEX_ALPHA);
    if (rdr)
        rdr->drop();

    sdCB->drop();
}


//! presents the rendered scene on the screen, returns false if failed
bool COpenGLDriver::endScene()
{
	CNullDriver::endScene();

#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
	if (DeviceType == EIDT_WIN32)
		return SwapBuffers(HDc) == TRUE;
#endif

#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
	if (DeviceType == EIDT_X11)
	{
		glXSwapBuffers(X11Display, Drawable);
		return true;
	}
#endif

#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
	if (DeviceType == EIDT_OSX)
	{
		OSXDevice->flush();
		return true;
	}
#endif

#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
	if (DeviceType == EIDT_SDL)
	{
		SDL_GL_SwapBuffers();
		return true;
	}
#endif

	// todo: console device present

	getThreadContext_helper(false)->freeUpVAOCache(false);

	return false;
}


//! init call for rendering start
bool COpenGLDriver::beginScene(bool backBuffer, bool zBuffer, SColor color,
		const SExposedVideoData& videoData, core::rect<int32_t>* sourceRect)
{
	CNullDriver::beginScene(backBuffer, zBuffer, color, videoData, sourceRect);
#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
	if (DeviceType==EIDT_OSX)
		changeRenderContext(videoData, (void*)0);
#endif // _IRR_COMPILE_WITH_OSX_DEVICE_

    if (zBuffer)
    {
        clearZBuffer(0.0);
    }

    if (backBuffer)
    {
        core::vectorSIMDf colorf(color.getRed(),color.getGreen(),color.getBlue(),color.getAlpha());
        colorf /= 255.f;
        clearScreen(Params.Doublebuffer ? ESB_BACK_LEFT:ESB_FRONT_LEFT,reinterpret_cast<float*>(&colorf));
    }
	return true;
}


IGPUBuffer* COpenGLDriver::createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData)
{
    auto extraMreqs = initialMreqs;

    if (extraMreqs.memoryHeapLocation==IDriverMemoryAllocation::ESMT_DONT_KNOW)
        extraMreqs.memoryHeapLocation = (initialMreqs.mappingCapability&IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ)!=0u ? IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL:IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;

    if ((extraMreqs.mappingCapability&IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ) && !runningInRenderDoc)
        extraMreqs.mappingCapability |= IDriverMemoryAllocation::EMCF_COHERENT;

    return new COpenGLBuffer(extraMreqs, canModifySubData);
}

void COpenGLDriver::flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges)
{
    for (uint32_t i=0; i<memoryRangeCount; i++)
    {
        auto range = pMemoryRanges+i;
        #ifdef _IRR_DEBUG
        if (!range->memory->haveToMakeVisible())
            os::Printer::log("Why are you flushing mapped memory that does not need to be flushed!?",ELL_WARNING);
        #endif // _IRR_DEBUG
        extGlFlushMappedNamedBufferRange(static_cast<COpenGLBuffer*>(range->memory)->getOpenGLName(),range->offset,range->length);
    }
}

void COpenGLDriver::invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges)
{
    for (uint32_t i=0; i<memoryRangeCount; i++)
    {
        auto range = pMemoryRanges+i;
        #ifdef _IRR_DEBUG
        if (!range->memory->haveToMakeVisible())
            os::Printer::log("Why are you invalidating mapped memory that does not need to be invalidated!?",ELL_WARNING);
        #endif // _IRR_DEBUG
        extGlMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
    }
}

void COpenGLDriver::copyBuffer(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, size_t readOffset, size_t writeOffset, size_t length)
{
    COpenGLBuffer* readbuffer = static_cast<COpenGLBuffer*>(readBuffer);
    COpenGLBuffer* writebuffer = static_cast<COpenGLBuffer*>(writeBuffer);
    extGlCopyNamedBufferSubData(readbuffer->getOpenGLName(),writebuffer->getOpenGLName(),readOffset,writeOffset,length);
}

IGPUMeshDataFormatDesc* COpenGLDriver::createGPUMeshDataFormatDesc(core::LeakDebugger* dbgr)
{
    return new COpenGLVAOSpec(dbgr);
}

IQueryObject* COpenGLDriver::createPrimitivesGeneratedQuery()
{
    return new COpenGLQuery(GL_PRIMITIVES_GENERATED);
}

IQueryObject* COpenGLDriver::createXFormFeedbackPrimitiveQuery()
{
    return new COpenGLQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
}

IQueryObject* COpenGLDriver::createElapsedTimeQuery()
{
    return new COpenGLQuery(GL_TIME_ELAPSED);
}

IGPUTimestampQuery* COpenGLDriver::createTimestampQuery()
{
    return new COpenGLTimestampQuery();
}

void COpenGLDriver::beginQuery(IQueryObject* query)
{
    if (!query)
        return; //error

    COpenGLQuery* queryGL = static_cast<COpenGLQuery*>(query);
    if (queryGL->getGLHandle()==0||queryGL->isActive())
        return;

    if (currentQuery[query->getQueryObjectType()][0])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()][0] = query;


    extGlBeginQuery(queryGL->getType(),queryGL->getGLHandle());
    queryGL->flagBegun();
}
void COpenGLDriver::endQuery(IQueryObject* query)
{
    if (!query)
        return; //error
    if (currentQuery[query->getQueryObjectType()][0]!=query)
        return; //error

    COpenGLQuery* queryGL = static_cast<COpenGLQuery*>(query);
    if (queryGL->getGLHandle()==0||!queryGL->isActive())
        return;

    if (currentQuery[query->getQueryObjectType()][0])
        currentQuery[query->getQueryObjectType()][0]->drop();
    currentQuery[query->getQueryObjectType()][0] = NULL;


    extGlEndQuery(queryGL->getType());
    queryGL->flagEnded();
}

void COpenGLDriver::beginQuery(IQueryObject* query, const size_t& index)
{
    if (index>=_IRR_XFORM_FEEDBACK_MAX_STREAMS_)
        return; //error

    if (!query||(query->getQueryObjectType()!=EQOT_PRIMITIVES_GENERATED&&query->getQueryObjectType()!=EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN))
        return; //error

    COpenGLQuery* queryGL = static_cast<COpenGLQuery*>(query);
    if (queryGL->getGLHandle()==0||queryGL->isActive())
        return;

    if (currentQuery[query->getQueryObjectType()][index])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()][index] = query;


    extGlBeginQueryIndexed(queryGL->getType(),index,queryGL->getGLHandle());
    queryGL->flagBegun();
}
void COpenGLDriver::endQuery(IQueryObject* query, const size_t& index)
{
    if (index>=_IRR_XFORM_FEEDBACK_MAX_STREAMS_)
        return; //error

    if (!query||(query->getQueryObjectType()!=EQOT_PRIMITIVES_GENERATED&&query->getQueryObjectType()!=EQOT_XFORM_FEEDBACK_PRIMITIVES_WRITTEN))
        return; //error
    if (currentQuery[query->getQueryObjectType()][index]!=query)
        return; //error

    COpenGLQuery* queryGL = static_cast<COpenGLQuery*>(query);
    if (queryGL->getGLHandle()==0||!queryGL->isActive())
        return;

    if (currentQuery[query->getQueryObjectType()][index])
        currentQuery[query->getQueryObjectType()][index]->drop();
    currentQuery[query->getQueryObjectType()][index] = NULL;


    extGlEndQueryIndexed(queryGL->getType(),index);
    queryGL->flagEnded();
}

// small helper function to create vertex buffer object adress offsets
static inline uint8_t* buffer_offset(const long offset)
{
	return ((uint8_t*)0 + offset);
}



void COpenGLDriver::drawMeshBuffer(const IGPUMeshBuffer* mb)
{
    if (mb && !mb->getInstanceCount())
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    const COpenGLVAOSpec* meshLayoutVAO = static_cast<const COpenGLVAOSpec*>(mb->getMeshDataAndFormat());
    if (!found->setActiveVAO(meshLayoutVAO,mb->isIndexCountGivenByXFormFeedback() ? mb:NULL))
        return;

#ifdef _IRR_DEBUG
	if (mb->getIndexCount() > getMaximalIndicesCount())
	{
		char tmp[1024];
		sprintf(tmp,"Could not draw, too many indices(%u), maxium is %u.", mb->getIndexCount(), getMaximalIndicesCount());
		os::Printer::log(tmp, ELL_ERROR);
	}
#endif // _IRR_DEBUG

	CNullDriver::drawMeshBuffer(mb);

	// draw everything
	setRenderStates3DMode();

	GLenum indexSize=0;
    if (meshLayoutVAO->getIndexBuffer())
    {
        switch (mb->getIndexType())
        {
            case asset::EIT_16BIT:
            {
                indexSize=GL_UNSIGNED_SHORT;
                break;
            }
            case asset::EIT_32BIT:
            {
                indexSize=GL_UNSIGNED_INT;
                break;
            }
            default:
                break;
        }
    }

    GLenum primType = primitiveTypeToGL(mb->getPrimitiveType());
	switch (mb->getPrimitiveType())
	{
		case asset::EPT_POINTS:
		{
			// prepare size and attenuation (where supported)
			GLfloat particleSize=Material.Thickness;
			extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);
			glPointSize(particleSize);

		}
			break;
		case asset::EPT_TRIANGLES:
        {
            if (static_cast<uint32_t>(Material.MaterialType) < MaterialRenderers.size())
            {
                COpenGLSLMaterialRenderer* shaderRenderer = static_cast<COpenGLSLMaterialRenderer*>(MaterialRenderers[Material.MaterialType].Renderer);
                if (shaderRenderer&&shaderRenderer->isTessellation())
                    primType = GL_PATCHES;
            }
        }
			break;
        default:
			break;
	}

    if (indexSize)
        extGlDrawElementsInstancedBaseVertexBaseInstance(primType,mb->getIndexCount(),indexSize,(void*)mb->getIndexBufferOffset(),mb->getInstanceCount(),mb->getBaseVertex(),mb->getBaseInstance());
    else if (mb->isIndexCountGivenByXFormFeedback())
    {
        COpenGLTransformFeedback* xfmFb = static_cast<COpenGLTransformFeedback*>(mb->getXFormFeedback());
#ifdef _IRR_DEBUG
        if (xfmFb->isEnded())
            os::Printer::log("Trying To DrawTransformFeedback which hasn't ended yet (call glEndTransformFeedback() on the damn thing)!\n",ELL_ERROR);
        if (mb->getXFormFeedbackStream()>=MaxVertexStreams)
            os::Printer::log("Trying to use more than GL_MAX_VERTEX_STREAMS vertex streams in transform feedback!\n",ELL_ERROR);
#endif // _IRR_DEBUG
        extGlDrawTransformFeedbackStreamInstanced(primType,xfmFb->getOpenGLHandle(),mb->getXFormFeedbackStream(),mb->getInstanceCount());
    }
    else
        extGlDrawArraysInstancedBaseInstance(primType, mb->getBaseVertex(), mb->getIndexCount(), mb->getInstanceCount(), mb->getBaseInstance());
}


//! Indirect Draw
void COpenGLDriver::drawArraysIndirect(  const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                         const asset::E_PRIMITIVE_TYPE& mode,
                                         const IGPUBuffer* indirectDrawBuff,
                                         const size_t& offset, const size_t& count, const size_t& stride)
{
    if (!indirectDrawBuff)
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    const COpenGLVAOSpec* meshLayoutVAO = static_cast<const COpenGLVAOSpec*>(vao);
    if (!found->setActiveVAO(meshLayoutVAO))
        return;

    found->setActiveIndirectDrawBuffer(static_cast<const COpenGLBuffer*>(indirectDrawBuff));

	// draw everything
	setRenderStates3DMode();

    GLenum primType = primitiveTypeToGL(mode);
	switch (mode)
	{
		case asset::EPT_POINTS:
		{
			// prepare size and attenuation (where supported)
			GLfloat particleSize=Material.Thickness;
			extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);
			glPointSize(particleSize);
		}
			break;
		case asset::EPT_TRIANGLES:
        {
            if (static_cast<uint32_t>(Material.MaterialType) < MaterialRenderers.size())
            {
                COpenGLSLMaterialRenderer* shaderRenderer = static_cast<COpenGLSLMaterialRenderer*>(MaterialRenderers[Material.MaterialType].Renderer);
                if (shaderRenderer&&shaderRenderer->isTessellation())
                    primType = GL_PATCHES;
            }
        }
			break;
        default:
			break;
	}


    //actual drawing
    extGlMultiDrawArraysIndirect(primType,(void*)offset,count,stride);
}


bool COpenGLDriver::queryFeature(const E_DRIVER_FEATURE &feature) const
{
	switch (feature)
	{
        case EDF_ALPHA_TO_COVERAGE:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_multisample]||true; //vulkan+android
        case EDF_GEOMETRY_SHADER:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_geometry_shader4]||true; //vulkan+android
        case EDF_TESSELLATION_SHADER:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_tessellation_shader]||true; //vulkan+android
        case EDF_GET_TEXTURE_SUB_IMAGE:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_get_texture_sub_image]; //only on OpenGL
        case EDF_TEXTURE_BARRIER:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_texture_barrier]||COpenGLExtensionHandler::FeatureAvailable[IRR_NV_texture_barrier]||Version>=450;
        case EDF_STENCIL_ONLY_TEXTURE:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_texture_stencil8]||Version>=440;
		case EDF_SHADER_DRAW_PARAMS:
			return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_shader_draw_parameters]||Version>=460;
		case EDF_MULTI_DRAW_INDIRECT_COUNT:
			return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_indirect_parameters]||Version>=460;
        case EDF_SHADER_GROUP_VOTE:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_NV_gpu_shader5]||COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_shader_group_vote]||Version>=460;
        case EDF_SHADER_GROUP_BALLOT:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_NV_shader_thread_group]||COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_shader_ballot];
		case EDF_SHADER_GROUP_SHUFFLE:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_NV_shader_thread_shuffle];
        case EDF_FRAGMENT_SHADER_INTERLOCK:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_INTEL_fragment_shader_ordering]||COpenGLExtensionHandler::FeatureAvailable[IRR_NV_fragment_shader_interlock]||COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_fragment_shader_interlock];
        case EDF_BINDLESS_TEXTURE:
            return COpenGLExtensionHandler::FeatureAvailable[IRR_ARB_bindless_texture]||Version>=450;
        case EDF_DYNAMIC_SAMPLER_INDEXING:
            return queryFeature(EDF_BINDLESS_TEXTURE);
        default:
            break;
	};
	return false;
}

void COpenGLDriver::drawIndexedIndirect(const asset::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const asset::E_PRIMITIVE_TYPE& mode,
                                        const asset::E_INDEX_TYPE& type, const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride)
{
    if (!indirectDrawBuff)
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    const COpenGLVAOSpec* meshLayoutVAO = static_cast<const COpenGLVAOSpec*>(vao);
    if (!found->setActiveVAO(meshLayoutVAO))
        return;

    found->setActiveIndirectDrawBuffer(static_cast<const COpenGLBuffer*>(indirectDrawBuff));

	// draw everything
	setRenderStates3DMode();

	GLenum indexSize = type!=asset::EIT_16BIT ? GL_UNSIGNED_INT:GL_UNSIGNED_SHORT;
    GLenum primType = primitiveTypeToGL(mode);
	switch (mode)
	{
		case asset::EPT_POINTS:
		{
			// prepare size and attenuation (where supported)
			GLfloat particleSize=Material.Thickness;
			extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);
			glPointSize(particleSize);
		}
			break;
		case asset::EPT_TRIANGLES:
        {
            if (static_cast<uint32_t>(Material.MaterialType) < MaterialRenderers.size())
            {
                COpenGLSLMaterialRenderer* shaderRenderer = static_cast<COpenGLSLMaterialRenderer*>(MaterialRenderers[Material.MaterialType].Renderer);
                if (shaderRenderer&&shaderRenderer->isTessellation())
                    primType = GL_PATCHES;
            }
        }
			break;
        default:
			break;
	}


    //actual drawing
    extGlMultiDrawElementsIndirect(primType,indexSize,(void*)offset,count,stride);
}


template<GLenum BIND_POINT,size_t BIND_POINTS>
void COpenGLDriver::SAuxContext::BoundIndexedBuffer<BIND_POINT,BIND_POINTS>::set(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** const buffers, const ptrdiff_t* const offsets, const ptrdiff_t* const sizes)
{
    if (!buffers)
    {
        bool needRebind = false;

        for (uint32_t i=0; i<count; i++)
        {
            uint32_t actualIx = i+first;
            if (boundBuffer[actualIx])
            {
                needRebind = true;
                boundBuffer[actualIx]->drop();
                boundBuffer[actualIx] = nullptr;
            }
        }

        if (needRebind)
            extGlBindBuffersRange(BIND_POINT,first,count,nullptr,nullptr,nullptr);
        return;
    }

    uint32_t newFirst = BIND_POINTS;
    uint32_t newLast = 0;

    GLuint toBind[BIND_POINTS];
    for (uint32_t i=0; i<count; i++)
    {
        toBind[i] = buffers[i] ? buffers[i]->getOpenGLName():0;

        uint32_t actualIx = i+first;
        if (boundBuffer[actualIx]!=buffers[i]) //buffers are different
        {
            if (buffers[i])
                buffers[i]->grab();
            if (boundBuffer[actualIx])
                boundBuffer[actualIx]->drop();
            boundBuffer[actualIx] = buffers[i];
        }
        else if (!buffers[i]) //change of range on a null binding doesn't matter
            continue;
        else if (offsets[i]==boundOffsets[actualIx]&&
                 sizes[i]==boundSizes[actualIx]&&
                 buffers[i]->getLastTimeReallocated()<=lastValidatedBuffer[actualIx]) //everything has to be the same and up to date
            continue;

        boundOffsets[actualIx] = offsets[i];
        boundSizes[actualIx] = sizes[i];
        lastValidatedBuffer[actualIx] = boundBuffer[actualIx]->getLastTimeReallocated();

        newLast = i;
        if (newFirst==BIND_POINTS)
            newFirst = i;
    }

    if (newFirst>newLast)
        return;

    extGlBindBuffersRange(BIND_POINT,first+newFirst,newLast-newFirst+1,toBind+newFirst,offsets+newFirst,sizes+newFirst);
}

template class COpenGLDriver::SAuxContext::BoundIndexedBuffer<GL_SHADER_STORAGE_BUFFER,OGL_MAX_BUFFER_BINDINGS>;
template class COpenGLDriver::SAuxContext::BoundIndexedBuffer<GL_UNIFORM_BUFFER,OGL_MAX_BUFFER_BINDINGS>;


template<GLenum BIND_POINT>
void COpenGLDriver::SAuxContext::BoundBuffer<BIND_POINT>::set(const COpenGLBuffer* buff)
{
    if (!buff)
    {
        if (boundBuffer)
        {
            boundBuffer->drop();
            boundBuffer = nullptr;
            extGlBindBuffer(BIND_POINT,0);
        }

        return;
    }

    if (boundBuffer!=buff)
    {
        buff->grab();
        if (boundBuffer)
            boundBuffer->drop();
        boundBuffer = buff;
    }
    else if (!boundBuffer||boundBuffer->getLastTimeReallocated()<=lastValidatedBuffer)
        return;

    extGlBindBuffer(BIND_POINT,boundBuffer->getOpenGLName());
    lastValidatedBuffer = boundBuffer->getLastTimeReallocated();
}


static GLenum formatEnumToGLenum(asset::E_FORMAT fmt)
{
    using namespace asset;
    switch (fmt)
    {
    case EF_R16_SFLOAT:
    case EF_R16G16_SFLOAT:
    case EF_R16G16B16_SFLOAT:
    case EF_R16G16B16A16_SFLOAT:
        return GL_HALF_FLOAT;
    case EF_R32_SFLOAT:
    case EF_R32G32_SFLOAT:
    case EF_R32G32B32_SFLOAT:
    case EF_R32G32B32A32_SFLOAT:
        return GL_FLOAT;
    case EF_B10G11R11_UFLOAT_PACK32:
        return GL_UNSIGNED_INT_10F_11F_11F_REV;
    case EF_R8_UNORM:
    case EF_R8_UINT:
    case EF_R8G8_UNORM:
    case EF_R8G8_UINT:
    case EF_R8G8B8_UNORM:
    case EF_R8G8B8_UINT:
    case EF_R8G8B8A8_UNORM:
    case EF_R8G8B8A8_UINT:
    case EF_R8_USCALED:
    case EF_R8G8_USCALED:
    case EF_R8G8B8_USCALED:
    case EF_R8G8B8A8_USCALED:
    case EF_B8G8R8A8_UNORM:
        return GL_UNSIGNED_BYTE;
    case EF_R8_SNORM:
    case EF_R8_SINT:
    case EF_R8G8_SNORM:
    case EF_R8G8_SINT:
    case EF_R8G8B8_SNORM:
    case EF_R8G8B8_SINT:
    case EF_R8G8B8A8_SNORM:
    case EF_R8G8B8A8_SINT:
    case EF_R8_SSCALED:
    case EF_R8G8_SSCALED:
    case EF_R8G8B8_SSCALED:
    case EF_R8G8B8A8_SSCALED:
        return GL_BYTE;
    case EF_R16_UNORM:
    case EF_R16_UINT:
    case EF_R16G16_UNORM:
    case EF_R16G16_UINT:
    case EF_R16G16B16_UNORM:
    case EF_R16G16B16_UINT:
    case EF_R16G16B16A16_UNORM:
    case EF_R16G16B16A16_UINT:
    case EF_R16_USCALED:
    case EF_R16G16_USCALED:
    case EF_R16G16B16_USCALED:
    case EF_R16G16B16A16_USCALED:
        return GL_UNSIGNED_SHORT;
    case EF_R16_SNORM:
    case EF_R16_SINT:
    case EF_R16G16_SNORM:
    case EF_R16G16_SINT:
    case EF_R16G16B16_SNORM:
    case EF_R16G16B16_SINT:
    case EF_R16G16B16A16_SNORM:
    case EF_R16G16B16A16_SINT:
    case EF_R16_SSCALED:
    case EF_R16G16_SSCALED:
    case EF_R16G16B16_SSCALED:
    case EF_R16G16B16A16_SSCALED:
        return GL_SHORT;
    case EF_R32_UINT:
    case EF_R32G32_UINT:
    case EF_R32G32B32_UINT:
    case EF_R32G32B32A32_UINT:
        return GL_UNSIGNED_INT;
    case EF_R32_SINT:
    case EF_R32G32_SINT:
    case EF_R32G32B32_SINT:
    case EF_R32G32B32A32_SINT:
        return GL_INT;
    case EF_A2R10G10B10_UNORM_PACK32:
    case EF_A2B10G10R10_UNORM_PACK32:
    case EF_A2B10G10R10_USCALED_PACK32:
    case EF_A2B10G10R10_UINT_PACK32:
        return GL_UNSIGNED_INT_2_10_10_10_REV;
    case EF_A2R10G10B10_SNORM_PACK32:
    case EF_A2B10G10R10_SNORM_PACK32:
    case EF_A2B10G10R10_SSCALED_PACK32:
    case EF_A2B10G10R10_SINT_PACK32:
        return GL_INT_2_10_10_10_REV;
    case EF_R64_SFLOAT:
    case EF_R64G64_SFLOAT:
    case EF_R64G64B64_SFLOAT:
    case EF_R64G64B64A64_SFLOAT:
        return GL_DOUBLE;

    default: return (GLenum)0;
    }
}

COpenGLDriver::SAuxContext::COpenGLVAO::COpenGLVAO(const COpenGLVAOSpec* spec)
        : vao(0), lastValidated(0)
#ifdef _IRR_DEBUG
            ,debugHash(spec->getHash())
#endif // _IRR_DEBUG
{
    extGlCreateVertexArrays(1,&vao);

    memcpy(attrOffset,&spec->getMappedBufferOffset(asset::EVAI_ATTR0),sizeof(attrOffset));
    for (asset::E_VERTEX_ATTRIBUTE_ID attrId=asset::EVAI_ATTR0; attrId<asset::EVAI_COUNT; attrId = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(attrId+1))
    {
        const IGPUBuffer* buf = spec->getMappedBuffer(attrId);
        mappedAttrBuf[attrId] = static_cast<const COpenGLBuffer*>(buf);
        if (mappedAttrBuf[attrId])
        {
            const asset::E_FORMAT format = spec->getAttribFormat(attrId);

            mappedAttrBuf[attrId]->grab();
            attrStride[attrId] = spec->getMappedBufferStride(attrId);

            extGlEnableVertexArrayAttrib(vao,attrId);
            extGlVertexArrayAttribBinding(vao,attrId,attrId);

            if (isFloatingPointFormat(format) && getTexelOrBlockSize(format)/getFormatChannelCount(format)==8u)//DOUBLE
                extGlVertexArrayAttribLFormat(vao, attrId, getFormatChannelCount(format), GL_DOUBLE, 0);
            else if (isFloatingPointFormat(format) || isScaledFormat(format) || isNormalizedFormat(format))//FLOATING-POINT, SCALED ("weak integer"), NORMALIZED
                extGlVertexArrayAttribFormat(vao, attrId, isBGRALayoutFormat(format) ? GL_BGRA : getFormatChannelCount(format), formatEnumToGLenum(format), isNormalizedFormat(format) ? GL_TRUE : GL_FALSE, 0);
            else if (isIntegerFormat(format))//INTEGERS
                extGlVertexArrayAttribIFormat(vao, attrId, getFormatChannelCount(format), formatEnumToGLenum(format), 0);

            extGlVertexArrayBindingDivisor(vao,attrId,spec->getAttribDivisor(attrId));
            extGlVertexArrayVertexBuffer(vao,attrId,mappedAttrBuf[attrId]->getOpenGLName(),attrOffset[attrId],attrStride[attrId]);
        }
        else
        {
            mappedAttrBuf[attrId] = nullptr;
            attrStride[attrId] = 16;
        }
    }


    mappedIndexBuf = static_cast<const COpenGLBuffer*>(spec->getIndexBuffer());
    if (mappedIndexBuf)
    {
        mappedIndexBuf->grab();
        extGlVertexArrayElementBuffer(vao,mappedIndexBuf->getOpenGLName());
    }
}

COpenGLDriver::SAuxContext::COpenGLVAO::~COpenGLVAO()
{
    if (vao)
        extGlDeleteVertexArrays(1,&vao);

    for (asset::E_VERTEX_ATTRIBUTE_ID attrId=asset::EVAI_ATTR0; attrId<asset::EVAI_COUNT; attrId = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(attrId+1))
    {
        if (!mappedAttrBuf[attrId])
            continue;

        mappedAttrBuf[attrId]->drop();
    }

    if (mappedIndexBuf)
        mappedIndexBuf->drop();
}

void COpenGLDriver::SAuxContext::COpenGLVAO::bindBuffers(   const COpenGLBuffer* indexBuf,
                                                            const COpenGLBuffer* const* attribBufs,
                                                            const size_t offsets[asset::EVAI_COUNT],
                                                            const uint32_t strides[asset::EVAI_COUNT])
{
    uint64_t beginStamp = CNullDriver::ReallocationCounter;

    for (asset::E_VERTEX_ATTRIBUTE_ID attrId=asset::EVAI_ATTR0; attrId<asset::EVAI_COUNT; attrId = static_cast<asset::E_VERTEX_ATTRIBUTE_ID>(attrId+1))
    {
#ifdef _IRR_DEBUG
        assert( (mappedAttrBuf[attrId]==NULL && attribBufs[attrId]==NULL)||
                (mappedAttrBuf[attrId]!=NULL && attribBufs[attrId]!=NULL));
#endif // _IRR_DEBUG
        if (!mappedAttrBuf[attrId])
            continue;

        bool rebind = false;
        if (mappedAttrBuf[attrId]!=attribBufs[attrId])
        {
            mappedAttrBuf[attrId]->drop();
            mappedAttrBuf[attrId] = attribBufs[attrId];
            mappedAttrBuf[attrId]->grab();
            rebind = true;
        }
        if (attrOffset[attrId]!=offsets[attrId])
        {
            attrOffset[attrId] = offsets[attrId];
            rebind = true;
        }
        if (attrStride[attrId]!=strides[attrId])
        {
            attrStride[attrId] = strides[attrId];
            rebind = true;
        }

        if (rebind||mappedAttrBuf[attrId]->getLastTimeReallocated()>lastValidated)
            extGlVertexArrayVertexBuffer(vao,attrId,mappedAttrBuf[attrId]->getOpenGLName(),attrOffset[attrId],attrStride[attrId]);
    }

    bool rebind = false;
    if (indexBuf!=mappedIndexBuf)
    {
        if (indexBuf)
            indexBuf->grab();
        if (mappedIndexBuf)
            mappedIndexBuf->drop();
        mappedIndexBuf = indexBuf;
        rebind = true;
    }
    else if (mappedIndexBuf&&mappedIndexBuf->getLastTimeReallocated()>lastValidated)
        rebind = true;

    if (rebind)
    {
        if (mappedIndexBuf)
            extGlVertexArrayElementBuffer(vao,mappedIndexBuf->getOpenGLName());
        else
            extGlVertexArrayElementBuffer(vao,0);
    }

    lastValidated = beginStamp;
}

bool COpenGLDriver::SAuxContext::setActiveVAO(const COpenGLVAOSpec* const spec, const IGPUMeshBuffer* correctOffsetsForXFormDraw)
{
    if (!spec)
    {
        CurrentVAO = HashVAOPair(COpenGLVAOSpec::HashAttribs(),nullptr);
        extGlBindVertexArray(0);
        freeUpVAOCache(true);
        return false;
    }

    const COpenGLVAOSpec::HashAttribs& hashVal = spec->getHash();
	if (CurrentVAO.first!=hashVal)
    {
        auto it = std::lower_bound(VAOMap.begin(),VAOMap.end(),HashVAOPair(hashVal,nullptr),[](HashVAOPair lhs, HashVAOPair rhs) -> bool { return lhs.first < rhs.first; });
        if (it != VAOMap.end() && it->first==hashVal)
            CurrentVAO = *it;
        else
        {
            COpenGLVAO* vao = new COpenGLVAO(spec);
            CurrentVAO = HashVAOPair(hashVal,vao);
            VAOMap.insert(it,CurrentVAO);
        }

        #ifdef _IRR_DEBUG
            assert(!(CurrentVAO.second->getDebugHash()!=hashVal));
        #endif // _IRR_DEBUG

        extGlBindVertexArray(CurrentVAO.second->getOpenGLName());
    }

    if (correctOffsetsForXFormDraw)
    {
        size_t offsets[asset::EVAI_COUNT] = {0};
        memcpy(offsets,&spec->getMappedBufferOffset(asset::EVAI_ATTR0),sizeof(offsets));
        for (size_t i=0; i<asset::EVAI_COUNT; i++)
        {
            if (!spec->getMappedBuffer((asset::E_VERTEX_ATTRIBUTE_ID)i))
                continue;

            if (spec->getAttribDivisor((asset::E_VERTEX_ATTRIBUTE_ID)i))
            {
                if (correctOffsetsForXFormDraw->getBaseInstance())
                    offsets[i] += spec->getMappedBufferStride((asset::E_VERTEX_ATTRIBUTE_ID)i)*correctOffsetsForXFormDraw->getBaseInstance();
            }
            else
            {
                if (correctOffsetsForXFormDraw->getBaseVertex())
                    offsets[i] = int64_t(offsets[i])+int64_t(spec->getMappedBufferStride((asset::E_VERTEX_ATTRIBUTE_ID)i))*correctOffsetsForXFormDraw->getBaseVertex();
            }
        }
        CurrentVAO.second->bindBuffers(static_cast<const COpenGLBuffer*>(spec->getIndexBuffer()),reinterpret_cast<const COpenGLBuffer* const*>(spec->getMappedBuffers()),offsets,&spec->getMappedBufferStride(asset::EVAI_ATTR0));
    }
    else
        CurrentVAO.second->bindBuffers(static_cast<const COpenGLBuffer*>(spec->getIndexBuffer()),reinterpret_cast<const COpenGLBuffer* const*>(spec->getMappedBuffers()),&spec->getMappedBufferOffset(asset::EVAI_ATTR0),&spec->getMappedBufferStride(asset::EVAI_ATTR0));

    return true;
}

//! Get native wrap mode value
inline GLint getTextureWrapMode(uint8_t clamp)
{
	GLint mode=GL_REPEAT;
	switch (clamp)
	{
		case ETC_REPEAT:
			mode=GL_REPEAT;
			break;
		case ETC_CLAMP_TO_EDGE:
			mode=GL_CLAMP_TO_EDGE;
			break;
		case ETC_CLAMP_TO_BORDER:
            mode=GL_CLAMP_TO_BORDER;
			break;
		case ETC_MIRROR:
            mode=GL_MIRRORED_REPEAT;
			break;
		case ETC_MIRROR_CLAMP_TO_EDGE:
			if (COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_EXT_texture_mirror_clamp])
				mode = GL_MIRROR_CLAMP_TO_EDGE_EXT;
			else if (COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_ATI_texture_mirror_once])
				mode = GL_MIRROR_CLAMP_TO_EDGE_ATI;
			else
				mode = GL_CLAMP;
			break;
		case ETC_MIRROR_CLAMP_TO_BORDER:
			if (COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_EXT_texture_mirror_clamp])
				mode = GL_MIRROR_CLAMP_TO_BORDER_EXT;
			else
				mode = GL_CLAMP;
			break;
	}
	return mode;
}


const GLuint& COpenGLDriver::SAuxContext::constructSamplerInCache(const uint64_t &hashVal)
{
    GLuint samplerHandle;
    extGlGenSamplers(1,&samplerHandle);

    const STextureSamplingParams* tmpTSP = reinterpret_cast<const STextureSamplingParams*>(&hashVal);

    switch (tmpTSP->MinFilter)
    {
        case ETFT_NEAREST_NO_MIP:
            extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            break;
        case ETFT_LINEAR_NO_MIP:
            extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            break;
        case ETFT_NEAREST_NEARESTMIP:
            extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
            break;
        case ETFT_LINEAR_NEARESTMIP:
            extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
            break;
        case ETFT_NEAREST_LINEARMIP:
            extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
            break;
        case ETFT_LINEAR_LINEARMIP:
            extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            break;
    }

    extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MAG_FILTER, tmpTSP->MaxFilter ? GL_LINEAR : GL_NEAREST);

    if (tmpTSP->AnisotropicFilter)
        extGlSamplerParameteri(samplerHandle, GL_TEXTURE_MAX_ANISOTROPY_EXT, core::min_(tmpTSP->AnisotropicFilter+1u,uint32_t(MaxAnisotropy)));

    extGlSamplerParameteri(samplerHandle, GL_TEXTURE_WRAP_S, getTextureWrapMode(tmpTSP->TextureWrapU));
    extGlSamplerParameteri(samplerHandle, GL_TEXTURE_WRAP_T, getTextureWrapMode(tmpTSP->TextureWrapV));
    extGlSamplerParameteri(samplerHandle, GL_TEXTURE_WRAP_R, getTextureWrapMode(tmpTSP->TextureWrapW));

    extGlSamplerParameterf(samplerHandle, GL_TEXTURE_LOD_BIAS, tmpTSP->LODBias);
    extGlSamplerParameteri(samplerHandle, GL_TEXTURE_CUBE_MAP_SEAMLESS, tmpTSP->SeamlessCubeMap);

    return (SamplerMap[hashVal] = samplerHandle);
}

bool COpenGLDriver::SAuxContext::setActiveTexture(uint32_t stage, video::IVirtualTexture* texture, const video::STextureSamplingParams &sampleParams)
{
	if (stage >= COpenGLExtensionHandler::MaxTextureUnits)
		return false;


    if (texture&&texture->getVirtualTextureType()==IVirtualTexture::EVTT_BUFFER_OBJECT&&!static_cast<COpenGLTextureBufferObject*>(texture)->rebindRevalidate())
        return false;

	if (CurrentTexture[stage]!=texture)
    {
        const video::COpenGLTexture* oldTexture = dynamic_cast<const COpenGLTexture*>(CurrentTexture[stage]);
        GLenum oldTexType = GL_INVALID_ENUM;
        if (oldTexture)
            oldTexType = oldTexture->getOpenGLTextureType();
        CurrentTexture.set(stage,texture);

        if (!texture)
        {
            if (oldTexture)
                extGlBindTextures(stage,1,NULL,&oldTexType);
        }
        else
        {
            if (texture->getDriverType() != EDT_OPENGL)
            {
                CurrentTexture.set(stage, 0);
                if (oldTexture)
                    extGlBindTextures(stage,1,NULL,&oldTexType);
                os::Printer::log("Fatal Error: Tried to set a texture not owned by this driver.", ELL_ERROR);
            }
            else
            {
                const video::COpenGLTexture* newTexture = dynamic_cast<const COpenGLTexture*>(texture);
                GLenum newTexType = newTexture->getOpenGLTextureType();

                if (Version<440 && !FeatureAvailable[IRR_ARB_multi_bind] && oldTexture && oldTexType!=newTexType)
                    extGlBindTextures(stage,1,NULL,&oldTexType);
                extGlBindTextures(stage,1,&newTexture->getOpenGLName(),&newTexType);
            }
        }
    }

    if (CurrentTexture[stage])
    {
        if (CurrentTexture[stage]->getVirtualTextureType()!=IVirtualTexture::EVTT_BUFFER_OBJECT&&
            CurrentTexture[stage]->getVirtualTextureType()!=IVirtualTexture::EVTT_2D_MULTISAMPLE)
        {
            uint64_t hashVal = sampleParams.calculateHash(CurrentTexture[stage]);
            if (CurrentSamplerHash[stage]!=hashVal)
            {
                CurrentSamplerHash[stage] = hashVal;
                auto it = SamplerMap.find(hashVal);
                if (it != SamplerMap.end())
                {
                    extGlBindSamplers(stage,1,&it->second);
                }
                else
                {
                    extGlBindSamplers(stage,1,&constructSamplerInCache(hashVal));
                }
            }
        }
    }
    else if (CurrentSamplerHash[stage]!=0xffffffffffffffffull)
    {
        CurrentSamplerHash[stage] = 0xffffffffffffffffull;
        extGlBindSamplers(stage,1,NULL);
    }

	return true;
}


void COpenGLDriver::SAuxContext::STextureStageCache::remove(const IVirtualTexture* tex)
{
    for (int32_t i = MATERIAL_MAX_TEXTURES-1; i>= 0; --i)
    {
        if (CurrentTexture[i] == tex)
        {
            GLenum target = dynamic_cast<const COpenGLTexture*>(tex)->getOpenGLTextureType();
            COpenGLExtensionHandler::extGlBindTextures(i,1,NULL,&target);
            COpenGLExtensionHandler::extGlBindSamplers(i,1,NULL);
            tex->drop();
            CurrentTexture[i] = 0;
        }
    }
}

void COpenGLDriver::SAuxContext::STextureStageCache::clear()
{
    // Drop all the CurrentTexture handles
    GLuint textures[MATERIAL_MAX_TEXTURES] = {0};
    GLenum targets[MATERIAL_MAX_TEXTURES];

    for (uint32_t i=0; i<MATERIAL_MAX_TEXTURES; ++i)
    {
        if (CurrentTexture[i])
        {
            targets[i] = dynamic_cast<const COpenGLTexture*>(CurrentTexture[i])->getOpenGLTextureType();
            CurrentTexture[i]->drop();
            CurrentTexture[i] = NULL;
        }
        else
            targets[i] = GL_INVALID_ENUM;
    }

    COpenGLExtensionHandler::extGlBindTextures(0,MATERIAL_MAX_TEXTURES,textures,targets);
    COpenGLExtensionHandler::extGlBindSamplers(0,MATERIAL_MAX_TEXTURES,NULL);
}


bool orderByMip(asset::CImageData* a, asset::CImageData* b)
{
    return a->getSupposedMipLevel() < b->getSupposedMipLevel();
}


//! returns a device dependent texture from a software surface (IImage)
video::ITexture* COpenGLDriver::createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels,
			const io::path& name, asset::E_FORMAT format)
{
#ifdef _IRR_DEBUG
    //if the max coords are not 0, then there is something seriously wrong
    switch (type)
    {
        case ITexture::ETT_1D:
            assert(size[0]>0);
            break;
        case ITexture::ETT_2D:
        case ITexture::ETT_1D_ARRAY:
            assert(size[0]>0&&size[1]>0);
            break;
        case ITexture::ETT_CUBE_MAP:
            assert(size[0]>0&&size[1]>0&&size[2]==6);
            break;
        case ITexture::ETT_CUBE_MAP_ARRAY:
            assert(size[0]>0&&size[1]>0&&size[2]&&(size[2]%6==0));
            break;
        default:
            assert(size[0]>0&&size[1]>0&&size[2]>0);
            break;
    }
#endif // _IRR_DEBUG
    //do the texture creation flag mumbo jumbo of death.
    if (mipmapLevels==0)
    {
        if (getTextureCreationFlag(ETCF_CREATE_MIP_MAPS))
        {
            uint32_t maxSideLen = size[0];
            switch (type)
            {
                case ITexture::ETT_1D:
                case ITexture::ETT_1D_ARRAY:
                case ITexture::ETT_CUBE_MAP:
                case ITexture::ETT_CUBE_MAP_ARRAY:
                    break;
                case ITexture::ETT_2D:
                case ITexture::ETT_2D_ARRAY:
                    if (maxSideLen < size[1])
                        maxSideLen = size[1];
                    break;
                case ITexture::ETT_3D:
                    if (maxSideLen < size[1])
                        maxSideLen = size[1];
                    if (maxSideLen < size[2])
                        maxSideLen = size[2];
                    break;
                default:
                    maxSideLen = 1;
                    break;
            }
            mipmapLevels = 1u+uint32_t(floorf(log2(float(maxSideLen))));
        }
        else
            mipmapLevels = 1;
    }

    switch (type)
    {
        case ITexture::ETT_1D:
            return new COpenGL1DTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format), size, mipmapLevels, name);
            break;
        case ITexture::ETT_2D:
            return new COpenGL2DTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format), size, mipmapLevels, name);
            break;
        case ITexture::ETT_3D:
            return new COpenGL3DTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        case ITexture::ETT_1D_ARRAY:
            return new COpenGL1DTextureArray(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        case ITexture::ETT_2D_ARRAY:
            return new COpenGL2DTextureArray(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        case ITexture::ETT_CUBE_MAP:
            return new COpenGLCubemapTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        case ITexture::ETT_CUBE_MAP_ARRAY:
            return new COpenGLCubemapArrayTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        default:// ETT_CUBE_MAP, ETT_CUBE_MAP_ARRAY, ETT_TEXTURE_BUFFER
            break;
    }

    return NULL;
}


//! Sets a material. All 3d drawing functions draw geometry now using this material.
void COpenGLDriver::setMaterial(const SGPUMaterial& material)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


	Material = material;

	for (int32_t i = MaxTextureUnits-1; i>= 0; --i)
	{
		found->setActiveTexture(i, material.getTexture(i), material.TextureLayer[i].SamplingParams);
	}
}

//! sets the needed renderstates
void COpenGLDriver::setRenderStates3DMode()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

	if (CurrentRenderMode != ERM_3D)
	{
		// Reset Texture Stages
		glDisable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		ResetRenderStates = true;
	}

	if (ResetRenderStates || LastMaterial != Material)
	{
		// unset old material

		if (LastMaterial.MaterialType != Material.MaterialType &&
				static_cast<uint32_t>(LastMaterial.MaterialType) < MaterialRenderers.size())
			MaterialRenderers[LastMaterial.MaterialType].Renderer->OnUnsetMaterial();

		// set new material.
		if (static_cast<uint32_t>(Material.MaterialType) < MaterialRenderers.size())
			MaterialRenderers[Material.MaterialType].Renderer->OnSetMaterial(
				Material, LastMaterial, ResetRenderStates, this);

		if (found->CurrentXFormFeedback&&found->XFormFeedbackRunning)
        {
            if (Material.MaterialType==found->CurrentXFormFeedback->getMaterialType())
            {
                if (!found->CurrentXFormFeedback->isActive())
                    found->CurrentXFormFeedback->beginResumeFeedback();
            }
            else if (found->CurrentXFormFeedback->isActive()) //Material Type not equal to intial
                found->CurrentXFormFeedback->pauseFeedback();
        }

		LastMaterial = Material;
		ResetRenderStates = false;
	}

	if (static_cast<uint32_t>(Material.MaterialType) < MaterialRenderers.size())
		MaterialRenderers[Material.MaterialType].Renderer->OnRender(this);

	CurrentRenderMode = ERM_3D;
}




//! Can be called by an IMaterialRenderer to make its work easier.
void COpenGLDriver::setBasicRenderStates(const SGPUMaterial& material, const SGPUMaterial& lastmaterial,
	bool resetAllRenderStates)
{
	// fillmode
	if (resetAllRenderStates || (lastmaterial.Wireframe != material.Wireframe) || (lastmaterial.PointCloud != material.PointCloud))
		glPolygonMode(GL_FRONT_AND_BACK, material.Wireframe ? GL_LINE : material.PointCloud? GL_POINT : GL_FILL);

	// zbuffer
	if (resetAllRenderStates || lastmaterial.ZBuffer != material.ZBuffer)
	{
		switch (material.ZBuffer)
		{
			case ECFN_NEVER:
				glDisable(GL_DEPTH_TEST);
				break;
			case ECFN_LESSEQUAL:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_LEQUAL);
				break;
			case ECFN_EQUAL:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_EQUAL);
				break;
			case ECFN_LESS:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_LESS);
				break;
			case ECFN_NOTEQUAL:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_NOTEQUAL);
				break;
			case ECFN_GREATEREQUAL:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_GEQUAL);
				break;
			case ECFN_GREATER:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_GREATER);
				break;
			case ECFN_ALWAYS:
				glEnable(GL_DEPTH_TEST);
				glDepthFunc(GL_ALWAYS);
				break;
		}
	}

	// zwrite
//	if (resetAllRenderStates || lastmaterial.ZWriteEnable != material.ZWriteEnable)
	{
		if (material.ZWriteEnable && (AllowZWriteOnTransparent || !material.isTransparent()))
		{
			glDepthMask(GL_TRUE);
		}
		else
			glDepthMask(GL_FALSE);
	}

	// back face culling
	if (resetAllRenderStates || (lastmaterial.FrontfaceCulling != material.FrontfaceCulling) || (lastmaterial.BackfaceCulling != material.BackfaceCulling))
	{
		if ((material.FrontfaceCulling) && (material.BackfaceCulling))
		{
			glCullFace(GL_FRONT_AND_BACK);
			glEnable(GL_CULL_FACE);
		}
		else
		if (material.BackfaceCulling)
		{
			glCullFace(GL_BACK);
			glEnable(GL_CULL_FACE);
		}
		else
		if (material.FrontfaceCulling)
		{
			glCullFace(GL_FRONT);
			glEnable(GL_CULL_FACE);
		}
		else
			glDisable(GL_CULL_FACE);
	}

	if (resetAllRenderStates || (lastmaterial.RasterizerDiscard != material.RasterizerDiscard))
    {
        if (material.RasterizerDiscard)
            glEnable(GL_RASTERIZER_DISCARD);
        else
            glDisable(GL_RASTERIZER_DISCARD);
    }

	// Color Mask
	if (resetAllRenderStates || lastmaterial.ColorMask != material.ColorMask)
	{
		glColorMask(
			(material.ColorMask & ECP_RED)?GL_TRUE:GL_FALSE,
			(material.ColorMask & ECP_GREEN)?GL_TRUE:GL_FALSE,
			(material.ColorMask & ECP_BLUE)?GL_TRUE:GL_FALSE,
			(material.ColorMask & ECP_ALPHA)?GL_TRUE:GL_FALSE);
	}

	if (resetAllRenderStates|| lastmaterial.BlendOperation != material.BlendOperation)
	{
		if (material.BlendOperation==EBO_NONE)
			glDisable(GL_BLEND);
		else
		{
			glEnable(GL_BLEND);
			switch (material.BlendOperation)
			{
			case EBO_SUBTRACT:
                extGlBlendEquation(GL_FUNC_SUBTRACT);
				break;
			case EBO_REVSUBTRACT:
                extGlBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
				break;
			case EBO_MIN:
                extGlBlendEquation(GL_MIN);
				break;
			case EBO_MAX:
                extGlBlendEquation(GL_MAX);
				break;
			default:
				extGlBlendEquation(GL_FUNC_ADD);
				break;
			}
		}
	}


	// thickness
	if (resetAllRenderStates || lastmaterial.Thickness != material.Thickness)
	{
        glPointSize(core::clamp(static_cast<GLfloat>(material.Thickness), DimAliasedPoint[0], DimAliasedPoint[1]));
        glLineWidth(core::clamp(static_cast<GLfloat>(material.Thickness), DimAliasedLine[0], DimAliasedLine[1]));
	}
}


//! Enable the 2d override material
void COpenGLDriver::enableMaterial2D(bool enable)
{
	if (!enable)
		CurrentRenderMode = ERM_NONE;
	CNullDriver::enableMaterial2D(enable);
}


//! \return Returns the name of the video driver.
const wchar_t* COpenGLDriver::getName() const
{
	return Name.c_str();
}



// this code was sent in by Oliver Klems, thank you! (I modified the glViewport
// method just a bit.
void COpenGLDriver::setViewPort(const core::rect<int32_t>& area)
{
	if (area == ViewPort)
		return;
	core::rect<int32_t> vp = area;
	core::rect<int32_t> rendert(0,0, getCurrentRenderTargetSize().Width, getCurrentRenderTargetSize().Height);
	vp.clipAgainst(rendert);

	if (vp.getHeight()>0 && vp.getWidth()>0)
	{
		glViewport(vp.UpperLeftCorner.X,
				vp.UpperLeftCorner.Y,
				vp.getWidth(), vp.getHeight());

		ViewPort = vp;
	}
}

ITexture* COpenGLDriver::createGPUTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, asset::E_FORMAT format)
{
    return createDeviceDependentTexture(type, size, mipmapLevels, "", format);
}

IMultisampleTexture* COpenGLDriver::addMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size, asset::E_FORMAT format, const bool& fixedSampleLocations)
{
    //check to implement later on  attachment of textures to FBO
    //if (!isFormatRenderable(glTex->getOpenGLInternalFormat()))
        //return nullptr;
    //! Vulkan and D3D only allow PoT sample counts
    if (core::isNPoT(samples))
        return nullptr;

	IMultisampleTexture* tex;
	switch (type)
	{
        case IMultisampleTexture::EMTT_2D:
            tex = new COpenGLMultisampleTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),samples,size,fixedSampleLocations);
            break;
        case IMultisampleTexture::EMTT_2D_ARRAY:
            tex = new COpenGLMultisampleTextureArray(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),samples,size,fixedSampleLocations);
            break;
        default:
            tex = nullptr;
            break;
	}

	if (tex)
        CNullDriver::addMultisampleTexture(tex);

	return tex;
}

ITextureBufferObject* COpenGLDriver::addTextureBufferObject(IGPUBuffer* buf, const ITextureBufferObject::E_TEXURE_BUFFER_OBJECT_FORMAT& format, const size_t& offset, const size_t& length)
{
    COpenGLBuffer* buffer = static_cast<COpenGLBuffer*>(buf);
    if (!buffer)
        return nullptr;

    ITextureBufferObject* tbo = new COpenGLTextureBufferObject(buffer,format,offset,length);
	CNullDriver::addTextureBufferObject(tbo);
    return tbo;
}

IFrameBuffer* COpenGLDriver::addFrameBuffer()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return nullptr;

	IFrameBuffer* fbo = new COpenGLFrameBuffer(this);
	auto it = std::lower_bound(found->FrameBuffers.begin(),found->FrameBuffers.end(),fbo);
    found->FrameBuffers.insert(it,fbo);
	return fbo;
}

void COpenGLDriver::removeFrameBuffer(IFrameBuffer* framebuf)
{
    if (!framebuf)
        return;

    _IRR_CHECK_OWNING_THREAD(framebuf,return;);

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

	auto it = std::lower_bound(found->FrameBuffers.begin(),found->FrameBuffers.end(),framebuf);
	if (it!=found->FrameBuffers.end() && !(framebuf<*it))
        found->FrameBuffers.erase(it);
    else
        return;

    framebuf->drop();
}

void COpenGLDriver::removeAllFrameBuffers()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

	for (auto fb : found->FrameBuffers)
		fb->drop();
    found->FrameBuffers.clear();
}


//! Returns type of video driver
E_DRIVER_TYPE COpenGLDriver::getDriverType() const
{
	return EDT_OPENGL;
}


//! returns color format
asset::E_FORMAT COpenGLDriver::getColorFormat() const
{
	return ColorFormat;
}


void COpenGLDriver::setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number)
{
	os::Printer::log("Error: Please call services->setShaderConstant(), not VideoDriver->setShaderConstant().");
}


int32_t COpenGLDriver::addHighLevelShaderMaterial(
    const char* vertexShaderProgram,
    const char* controlShaderProgram,
    const char* evaluationShaderProgram,
    const char* geometryShaderProgram,
    const char* pixelShaderProgram,
    uint32_t patchVertices,
    E_MATERIAL_TYPE baseMaterial,
    IShaderConstantSetCallBack* callback,
    const char** xformFeedbackOutputs,
    const uint32_t& xformFeedbackOutputCount,
    int32_t userData,
    const char* vertexShaderEntryPointName,
    const char* controlShaderEntryPointName,
    const char* evaluationShaderEntryPointName,
    const char* geometryShaderEntryPointName,
    const char* pixelShaderEntryPointName)
{
    int32_t nr = -1;

	COpenGLSLMaterialRenderer* r = new COpenGLSLMaterialRenderer(
		this, nr,
		vertexShaderProgram, vertexShaderEntryPointName,
		pixelShaderProgram, pixelShaderEntryPointName,
		geometryShaderProgram, geometryShaderEntryPointName,
		controlShaderProgram,controlShaderEntryPointName,
		evaluationShaderProgram,evaluationShaderEntryPointName,
		patchVertices,callback,baseMaterial,
		xformFeedbackOutputs, xformFeedbackOutputCount, userData);
	r->drop();
	return nr;
}


//! Returns a pointer to the IVideoDriver interface. (Implementation for
//! IMaterialRendererServices)
IVideoDriver* COpenGLDriver::getVideoDriver()
{
	return this;
}



void COpenGLDriver::blitRenderTargets(IFrameBuffer* in, IFrameBuffer* out,
                                        bool copyDepth, bool copyStencil,
                                        core::recti srcRect, core::recti dstRect,
                                        bool bilinearFilter)
{
	GLuint inFBOHandle = 0;
	GLuint outFBOHandle = 0;


	if (srcRect.getArea()==0)
	{
	    if (in)
        {
            if (!static_cast<COpenGLFrameBuffer*>(in)->rebindRevalidate())
                return;

            bool firstAttached = true;
            uint32_t width,height;
            for (size_t i=0; i<EFAP_MAX_ATTACHMENTS; i++)
            {
                const IRenderableVirtualTexture* rndrbl = in->getAttachment(i);
                if (!rndrbl)
                    continue;

                if (firstAttached)
                {
                    firstAttached = false;
                    width = rndrbl->getRenderableSize().Width;
                    height = rndrbl->getRenderableSize().Height;
                }
                else
                {
                    width = core::min_(rndrbl->getRenderableSize().Width,width);
                    height = core::min_(rndrbl->getRenderableSize().Height,height);
                }
            }
            if (firstAttached)
                return;

            srcRect = core::recti(0,0,width,height);
        }
        else
            srcRect = core::recti(0,0,ScreenSize.Width,ScreenSize.Height);
	}
	if (dstRect.getArea()==0)
	{
	    if (out)
        {
            if (!static_cast<COpenGLFrameBuffer*>(out)->rebindRevalidate())
                return;

            bool firstAttached = true;
            uint32_t width,height;
            for (size_t i=0; i<EFAP_MAX_ATTACHMENTS; i++)
            {
                const IRenderableVirtualTexture* rndrbl = out->getAttachment(i);
                if (!rndrbl)
                    continue;

                if (firstAttached)
                {
                    firstAttached = false;
                    width = rndrbl->getRenderableSize().Width;
                    height = rndrbl->getRenderableSize().Height;
                }
                else
                {
                    width = core::min_(rndrbl->getRenderableSize().Width,width);
                    height = core::min_(rndrbl->getRenderableSize().Height,height);
                }
            }
            if (firstAttached)
                return;

            dstRect = core::recti(0,0,width,height);
        }
        else
            dstRect = core::recti(0,0,ScreenSize.Width,ScreenSize.Height);
	}
	if (srcRect==dstRect||copyDepth||copyStencil) //and some checks for multisample
		bilinearFilter = false;

    setViewPort(dstRect);

    if (in)
        inFBOHandle = static_cast<COpenGLFrameBuffer*>(in)->getOpenGLName();
    if (out)
        outFBOHandle = static_cast<COpenGLFrameBuffer*>(out)->getOpenGLName();

    extGlBlitNamedFramebuffer(inFBOHandle,outFBOHandle,
                        srcRect.UpperLeftCorner.X,srcRect.UpperLeftCorner.Y,srcRect.LowerRightCorner.X,srcRect.LowerRightCorner.Y,
                        dstRect.UpperLeftCorner.X,dstRect.UpperLeftCorner.Y,dstRect.LowerRightCorner.X,dstRect.LowerRightCorner.Y,
						GL_COLOR_BUFFER_BIT|(copyDepth ? GL_DEPTH_BUFFER_BIT:0)|(copyStencil ? GL_STENCIL_BUFFER_BIT:0),
						bilinearFilter ? GL_LINEAR:GL_NEAREST);
}



//! Returns the maximum amount of primitives (mostly vertices) which
//! the device is able to render with one drawIndexedTriangleList
//! call.
uint32_t COpenGLDriver::getMaximalIndicesCount() const
{
	return MaxIndices;
}



//! Sets multiple render targets
bool COpenGLDriver::setRenderTarget(IFrameBuffer* frameBuffer, bool setNewViewport)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return false;

    if (frameBuffer==found->CurrentFBO)
        return true;

    if (!frameBuffer)
    {
        found->CurrentRendertargetSize = ScreenSize;
        extGlBindFramebuffer(GL_FRAMEBUFFER, 0);
        if (found->CurrentFBO)
            found->CurrentFBO->drop();
        found->CurrentFBO = NULL;

        if (setNewViewport)
            setViewPort(core::recti(0,0,ScreenSize.Width,ScreenSize.Height));

        return true;
    }

    _IRR_CHECK_OWNING_THREAD(frameBuffer,return false;);

    if (!frameBuffer->rebindRevalidate())
    {
        os::Printer::log("FBO revalidation failed!", ELL_ERROR);
        return false;
    }

    bool firstAttached = true;
    core::dimension2du newRTTSize;
    for (size_t i=0; i<EFAP_MAX_ATTACHMENTS; i++)
    {
        const IRenderableVirtualTexture* attachment = frameBuffer->getAttachment(i);
        if (!attachment)
            continue;

        if (firstAttached)
        {
            newRTTSize = attachment->getRenderableSize();
            firstAttached = false;
        }
        else
        {
            newRTTSize.Width = core::min_(newRTTSize.Width,attachment->getRenderableSize().Width);
            newRTTSize.Height = core::min_(newRTTSize.Height,attachment->getRenderableSize().Height);
        }
    }

    //! Get rid of this! OpenGL 4.3 is here!
    if (firstAttached)
    {
        os::Printer::log("FBO has no attachments! (We don't support that OpenGL 4.3 feature yet!).", ELL_ERROR);
        return false;
    }
    found->CurrentRendertargetSize = newRTTSize;


    extGlBindFramebuffer(GL_FRAMEBUFFER, static_cast<COpenGLFrameBuffer*>(frameBuffer)->getOpenGLName());
    if (setNewViewport)
        setViewPort(core::recti(0,0,newRTTSize.Width,newRTTSize.Height));


    frameBuffer->grab();
    if (found->CurrentFBO)
        found->CurrentFBO->drop();
    found->CurrentFBO = static_cast<COpenGLFrameBuffer*>(frameBuffer);
    ResetRenderStates=true; //! OPTIMIZE: Needed?


    return true;
}


// returns the current size of the screen or rendertarget
const core::dimension2d<uint32_t>& COpenGLDriver::getCurrentRenderTargetSize() const
{
    const SAuxContext* found = getThreadContext();
	if (!found || found->CurrentRendertargetSize.Width == 0)
		return ScreenSize;
	else
		return found->CurrentRendertargetSize;
}


//! Clears the ZBuffer.
void COpenGLDriver::clearZBuffer(const float &depth)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    glDepthMask(GL_TRUE);
    LastMaterial.ZWriteEnable=true;

    if (found->CurrentFBO)
        extGlClearNamedFramebufferfv(found->CurrentFBO->getOpenGLName(),GL_DEPTH,0,&depth);
    else
        extGlClearNamedFramebufferfv(0,GL_DEPTH,0,&depth);
}

void COpenGLDriver::clearStencilBuffer(const int32_t &stencil)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    if (found->CurrentFBO)
        extGlClearNamedFramebufferiv(found->CurrentFBO->getOpenGLName(),GL_STENCIL,0,&stencil);
    else
        extGlClearNamedFramebufferiv(0,GL_STENCIL,0,&stencil);
}

void COpenGLDriver::clearZStencilBuffers(const float &depth, const int32_t &stencil)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    if (found->CurrentFBO)
        extGlClearNamedFramebufferfi(found->CurrentFBO->getOpenGLName(),GL_DEPTH_STENCIL,0,depth,stencil);
    else
        extGlClearNamedFramebufferfi(0,GL_DEPTH_STENCIL,0,depth,stencil);
}

void COpenGLDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    if (attachment<EFAP_COLOR_ATTACHMENT0)
        return;

    if (found->CurrentFBO)
        extGlClearNamedFramebufferiv(found->CurrentFBO->getOpenGLName(),GL_COLOR,attachment-EFAP_COLOR_ATTACHMENT0,vals);
    else
        extGlClearNamedFramebufferiv(0,GL_COLOR,attachment-EFAP_COLOR_ATTACHMENT0,vals);
}
void COpenGLDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    if (attachment<EFAP_COLOR_ATTACHMENT0)
        return;

    if (found->CurrentFBO)
        extGlClearNamedFramebufferuiv(found->CurrentFBO->getOpenGLName(),GL_COLOR,attachment-EFAP_COLOR_ATTACHMENT0,vals);
    else
        extGlClearNamedFramebufferuiv(0,GL_COLOR,attachment-EFAP_COLOR_ATTACHMENT0,vals);
}
void COpenGLDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    if (attachment<EFAP_COLOR_ATTACHMENT0)
        return;

    if (found->CurrentFBO)
        extGlClearNamedFramebufferfv(found->CurrentFBO->getOpenGLName(),GL_COLOR,attachment-EFAP_COLOR_ATTACHMENT0,vals);
    else
        extGlClearNamedFramebufferfv(0,GL_COLOR,attachment-EFAP_COLOR_ATTACHMENT0,vals);
}

void COpenGLDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals)
{
    switch (buffer)
    {
        case ESB_BACK_LEFT:
            extGlClearNamedFramebufferfv(0,GL_COLOR,0,vals);
            break;
        case ESB_BACK_RIGHT:
            extGlClearNamedFramebufferfv(0,GL_COLOR,0,vals);
            break;
        case ESB_FRONT_LEFT:
            extGlClearNamedFramebufferfv(0,GL_COLOR,0,vals);
            break;
        case ESB_FRONT_RIGHT:
            extGlClearNamedFramebufferfv(0,GL_COLOR,0,vals);
            break;
    }
}
void COpenGLDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals)
{
    switch (buffer)
    {
        case ESB_BACK_LEFT:
            extGlClearNamedFramebufferuiv(0,GL_COLOR,0,vals);
            break;
        case ESB_BACK_RIGHT:
            extGlClearNamedFramebufferuiv(0,GL_COLOR,0,vals);
            break;
        case ESB_FRONT_LEFT:
            extGlClearNamedFramebufferuiv(0,GL_COLOR,0,vals);
            break;
        case ESB_FRONT_RIGHT:
            extGlClearNamedFramebufferuiv(0,GL_COLOR,0,vals);
            break;
    }
}


ITransformFeedback* COpenGLDriver::createTransformFeedback()
{
    return new COpenGLTransformFeedback();
}


void COpenGLDriver::bindTransformFeedback(ITransformFeedback* xformFeedback)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    bindTransformFeedback(xformFeedback,found);
}

void COpenGLDriver::bindTransformFeedback(ITransformFeedback* xformFeedback, SAuxContext* toContext)
{
    if (xformFeedback)
    {
        _IRR_CHECK_OWNING_THREAD(xformFeedback,return;);
    }

    if (toContext->CurrentXFormFeedback==xformFeedback)
        return;

    if (toContext->CurrentXFormFeedback)
    {
#ifdef _IRR_DEBUG
        if (!toContext->CurrentXFormFeedback->isEnded())
            os::Printer::log("FIDDLING WITH XFORM FEEDBACK BINDINGS WHILE THE BOUND XFORMFEEDBACK HASN't ENDED!\n",ELL_ERROR);
#endif // _IRR_DEBUG
        toContext->CurrentXFormFeedback->drop();
    }

    toContext->CurrentXFormFeedback = static_cast<COpenGLTransformFeedback*>(xformFeedback);

    if (!toContext->CurrentXFormFeedback)
    {
	    extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,0);
		toContext->CurrentXFormFeedback = NULL;
	}
    else
    {
#ifdef _IRR_DEBUG
        if (!toContext->CurrentXFormFeedback->isEnded())
            os::Printer::log("WHY IS A NOT PREVIOUSLY BOUND XFORM FEEDBACK STARTED!?\n",ELL_ERROR);
#endif // _IRR_DEBUG
        toContext->CurrentXFormFeedback->grab();
        extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,toContext->CurrentXFormFeedback->getOpenGLHandle());
    }
}

void COpenGLDriver::beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const asset::E_PRIMITIVE_TYPE& primType)
{
    if (xformFeedback)
    {
        _IRR_CHECK_OWNING_THREAD(xformFeedback,return;);
    }

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    //grabs a ref
    bindTransformFeedback(xformFeedback,found);
    if (!xformFeedback)
    {
        found->XFormFeedbackRunning = false;
        return;
    }

	switch (primType)
	{
		case asset::EPT_POINTS:
            found->CurrentXFormFeedback->setPrimitiveType(GL_POINTS);
            break;
		case asset::EPT_LINE_STRIP:
			_IRR_FALLTHROUGH;
		case asset::EPT_LINE_LOOP:
			os::Printer::log("Not using PROPER TRANSFORM FEEDBACK primitive type (only EPT_POINTS, EPT_LINES and EPT_TRIANGLES allowed!)!\n",ELL_ERROR);
            break;
		case asset::EPT_LINES:
            found->CurrentXFormFeedback->setPrimitiveType(GL_LINES);
            break;
		case asset::EPT_TRIANGLE_STRIP:
			_IRR_FALLTHROUGH;
		case asset::EPT_TRIANGLE_FAN:
			os::Printer::log("Not using PROPER TRANSFORM FEEDBACK primitive type (only EPT_POINTS, EPT_LINES and EPT_TRIANGLES allowed!)!\n",ELL_ERROR);
            break;
		case asset::EPT_TRIANGLES:
            found->CurrentXFormFeedback->setPrimitiveType(GL_TRIANGLES);
            break;
	}
	found->CurrentXFormFeedback->setMaterialType(xformFeedbackShader);

	found->XFormFeedbackRunning = true;

	if (Material.MaterialType==xformFeedbackShader)
	{
        if (LastMaterial.MaterialType!=xformFeedbackShader)
            setRenderStates3DMode();

		if (!found->CurrentXFormFeedback->isActive())
			found->CurrentXFormFeedback->beginResumeFeedback();
	}
}

void COpenGLDriver::pauseTransformFeedback()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


	found->XFormFeedbackRunning = false;
    found->CurrentXFormFeedback->pauseFeedback();
}

void COpenGLDriver::resumeTransformFeedback()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


	found->XFormFeedbackRunning = true;
    found->CurrentXFormFeedback->beginResumeFeedback();
}

void COpenGLDriver::endTransformFeedback()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


    if (!found->CurrentXFormFeedback)
    {
        os::Printer::log("No Transform Feedback Object bound, possible redundant glEndTransform...!\n",ELL_ERROR);
        return;
    }
#ifdef _IRR_DEBUG
    if (!found->CurrentXFormFeedback->isActive())
        os::Printer::log("Ending an already paused transform feedback, the pause call is redundant!\n",ELL_ERROR);
#endif // _IRR_DEBUG
    found->CurrentXFormFeedback->endFeedback();
	found->XFormFeedbackRunning = false;
    ///In the interest of binding speed we wont release the CurrentXFormFeedback
    //bindTransformFeedback(NULL,found);
}


//! Enable/disable a clipping plane.
void COpenGLDriver::enableClipPlane(uint32_t index, bool enable)
{
	if (index >= MaxUserClipPlanes)
		return;
	if (enable)
        glEnable(GL_CLIP_DISTANCE0 + index);
	else
		glDisable(GL_CLIP_DISTANCE0 + index);
}

/*
GLenum COpenGLDriver::getGLBlend(E_BLEND_FACTOR factor) const
{
	GLenum r = 0;
	switch (factor)
	{
		case EBF_ZERO:			r = GL_ZERO; break;
		case EBF_ONE:			r = GL_ONE; break;
		case EBF_DST_COLOR:		r = GL_DST_COLOR; break;
		case EBF_ONE_MINUS_DST_COLOR:	r = GL_ONE_MINUS_DST_COLOR; break;
		case EBF_SRC_COLOR:		r = GL_SRC_COLOR; break;
		case EBF_ONE_MINUS_SRC_COLOR:	r = GL_ONE_MINUS_SRC_COLOR; break;
		case EBF_SRC_ALPHA:		r = GL_SRC_ALPHA; break;
		case EBF_ONE_MINUS_SRC_ALPHA:	r = GL_ONE_MINUS_SRC_ALPHA; break;
		case EBF_DST_ALPHA:		r = GL_DST_ALPHA; break;
		case EBF_ONE_MINUS_DST_ALPHA:	r = GL_ONE_MINUS_DST_ALPHA; break;
		case EBF_SRC_ALPHA_SATURATE:	r = GL_SRC_ALPHA_SATURATE; break;
	}
	return r;
}*/


} // end namespace
} // end namespace

#endif // _IRR_COMPILE_WITH_OPENGL_

namespace irr
{
namespace video
{


// -----------------------------------
// WINDOWS VERSION
// -----------------------------------
#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
IVideoDriver* createOpenGLDriver(const SIrrlichtCreationParameters& params,
	io::IFileSystem* io, CIrrDeviceWin32* device)
{
#ifdef _IRR_COMPILE_WITH_OPENGL_
	COpenGLDriver* ogl =  new COpenGLDriver(params, io, device);
	if (!ogl->initDriver(device))
	{
		ogl->drop();
		ogl = 0;
	}
	return ogl;
#else
	return 0;
#endif // _IRR_COMPILE_WITH_OPENGL_
}
#endif // _IRR_COMPILE_WITH_WINDOWS_DEVICE_

// -----------------------------------
// MACOSX VERSION
// -----------------------------------
#if defined(_IRR_COMPILE_WITH_OSX_DEVICE_)
IVideoDriver* createOpenGLDriver(const SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceMacOSX *device)
{
#ifdef _IRR_COMPILE_WITH_OPENGL_
	return new COpenGLDriver(params, io, device);
#else
	return 0;
#endif //  _IRR_COMPILE_WITH_OPENGL_
}
#endif // _IRR_COMPILE_WITH_OSX_DEVICE_

// -----------------------------------
// X11 VERSION
// -----------------------------------
#ifdef _IRR_COMPILE_WITH_X11_DEVICE_
IVideoDriver* createOpenGLDriver(const SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceLinux* device
#ifdef _IRR_COMPILE_WITH_OPENGL_
		, COpenGLDriver::SAuxContext* auxCtxts
#endif // _IRR_COMPILE_WITH_OPENGL_
        )
{
#ifdef _IRR_COMPILE_WITH_OPENGL_
	COpenGLDriver* ogl =  new COpenGLDriver(params, io, device);
	if (!ogl->initDriver(device,auxCtxts))
	{
		ogl->drop();
		ogl = 0;
	}
	return ogl;
#else
	return 0;
#endif //  _IRR_COMPILE_WITH_OPENGL_
}
#endif // _IRR_COMPILE_WITH_X11_DEVICE_


// -----------------------------------
// SDL VERSION
// -----------------------------------
#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
IVideoDriver* createOpenGLDriver(const SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceSDL* device)
{
#ifdef _IRR_COMPILE_WITH_OPENGL_
	return new COpenGLDriver(params, io, device);
#else
	return 0;
#endif //  _IRR_COMPILE_WITH_OPENGL_
}
#endif // _IRR_COMPILE_WITH_SDL_DEVICE_

} // end namespace
} // end namespace


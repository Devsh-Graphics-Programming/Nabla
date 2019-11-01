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
#include "irr/video/COpenGLBufferView.h"

#include "irr/video/COpenGLShader.h"
#include "irr/video/COpenGLSpecializedShader.h"
#include "irr/asset/IGLSLCompiler.h"
#include "irr/asset/CShaderIntrospector.h"

#include "COpenGLBuffer.h"
#include "COpenGLFrameBuffer.h"
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
		io::IFileSystem* io, CIrrDeviceWin32* device, const asset::IGLSLCompiler* glslcomp)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
	runningInRenderDoc(false),  /*ResetRenderStates(true),*/ ColorFormat(asset::EF_R8G8B8_UNORM), Params(params),
	HDc(0), Window(static_cast<HWND>(params.WindowId)), Win32Device(device),
	AuxContexts(0), DerivativeMapCreator(nullptr), GLSLCompiler(glslcomp), DeviceType(EIDT_WIN32)
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
        AuxContexts[0].ID = 0u;
    }
	for (size_t i=1; i<=Params.AuxGLContexts; i++)
    {
        AuxContexts[i].threadId = std::thread::id(); //invalid ID
        AuxContexts[i].ctx = wglCreateContextAttribs_ARB(HDc, hrc, iAttribs);
        AuxContexts[i].ID = i;
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
		io::IFileSystem* io, CIrrDeviceMacOSX *device, const asset::IGLSLCompiler* glslcomp)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
    runningInRenderDoc(false), CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(asset::EF_R8G8B8_UNORM),
	Params(params),
	OSXDevice(device), DeviceType(EIDT_OSX), AuxContexts(0), GLSLCompiler(glslcomp)
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
		io::IFileSystem* io, CIrrDeviceLinux* device, const asset::IGLSLCompiler* glslcomp)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
	runningInRenderDoc(false),  CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(asset::EF_R8G8B8_UNORM),
	Params(params), X11Device(device), DeviceType(EIDT_X11), AuxContexts(0), GLSLCompiler(glslcomp)
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
		io::IFileSystem* io, CIrrDeviceSDL* device, const asset::IGLSLCompiler* glslcomp)
: CNullDriver(device, io, params.WindowSize), COpenGLExtensionHandler(),
    runningInRenderDoc(false), CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(EF_R8G8B8_UNORM),
	CurrentTarget(ERT_FRAME_BUFFER), Params(params),
	SDLDevice(device), DeviceType(EIDT_SDL), AuxContexts(0), GLSLCompiler(glslcomp)
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
    if (DerivativeMapCreator)
        DerivativeMapCreator->drop();

	if (!AuxContexts) //opengl dead and never initialized in the first place
		return;

    cleanUpContextBeforeDelete();

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

    removeAllFrameBuffers();

    extGlBindVertexArray(0);
    for (auto& vao : found->VAOMap)
    {
        extGlDeleteVertexArrays(1, &vao.second.GLname);
    }
    found->VAOMap.clear();

    extGlUseProgram(0);
    extGlBindProgramPipeline(0);
    for (auto& ppln : found->GraphicsPipelineMap)
        extGlDeleteProgramPipelines(1, &ppln.second.GLname);
    found->GraphicsPipelineMap.clear();

    //force drop of all all grabbed (through smart_refctd_ptr) resources (descriptor sets, buffers, program pipeline)
    found->currentState = SOpenGLState();
    found->nextState = SOpenGLState();
    for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        found->effectivelyBoundDescriptors.descSets[i] = SOpenGLState::SDescSetBnd();

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

	// adjust flat coloring scheme to DirectX version
	///extGlProvokingVertex(GL_FIRST_VERTEX_CONVENTION_EXT);

	// We need to reset once more at the beginning of the first rendering.
	// This fixes problems with intermediate changes to the material during texture load.
    SAuxContext* found = getThreadContext_helper(false);
    glEnable(GL_FRAMEBUFFER_SRGB);//once set and should never change (engine doesnt track it)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);//once set and should never change (engine doesnt track it)
    glDepthRange(1.0, 0.0);//once set and should never change (engine doesnt track it)
    found->nextState.rasterParams.multisampleEnable = 0;
    found->nextState.rasterParams.depthFunc = GL_GEQUAL;
    found->nextState.rasterParams.frontFace = GL_CCW;

	// down
	{
        auto reqs = getDownStreamingMemoryReqs();
        reqs.vulkanReqs.size = Params.StreamingDownloadBufferSize;
        reqs.vulkanReqs.alignment = 64u*1024u; // if you need larger alignments then you're not right in the head
        defaultDownloadBuffer = core::make_smart_refctd_ptr<video::StreamingTransientDataBufferMT<> >(this,reqs);
	}
	// up
	{
        auto reqs = getUpStreamingMemoryReqs();
        reqs.vulkanReqs.size = Params.StreamingUploadBufferSize;
        reqs.vulkanReqs.alignment = 64u*1024u; // if you need larger alignments then you're not right in the head
        defaultUploadBuffer = core::make_smart_refctd_ptr < video::StreamingTransientDataBufferMT<> >(this,reqs);
	}

    DerivativeMapCreator = new CDerivativeMapCreator(this);

	return true;
}
/*
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
*/
void COpenGLDriver::createMaterialRenderers()
{
    /*
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

    //SimpleDummyCallBack* sdCB = new SimpleDummyCallBack();

    COpenGLSLMaterialRenderer* rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_solid_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3,nullptr,EMT_SOLID);
    if (rdr)
        rdr->drop();

	rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_trans_add_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3, nullptr,EMT_TRANSPARENT_ADD_COLOR);
    if (rdr)
        rdr->drop();

	rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_trans_alpha_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3, nullptr,EMT_TRANSPARENT_ALPHA_CHANNEL);
    if (rdr)
        rdr->drop();

	rdr = new COpenGLSLMaterialRenderer(
		this, nr,
		std_vert, "main",
		std_trans_vertex_frag, "main",
		NULL, NULL, NULL, NULL, NULL, NULL,3, nullptr,EMT_TRANSPARENT_VERTEX_ALPHA);
    if (rdr)
        rdr->drop();

    //sdCB->drop();
    */
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

    auto ctx = getThreadContext_helper(false);
	ctx->freeUpVAOCache(false);
    ctx->freeUpGraphicsPipelineCache(false);

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
        clearScreen(Params.Doublebuffer ? ESB_BACK_LEFT:ESB_FRONT_LEFT,colorf.pointer);
    }
	return true;
}


const core::smart_refctd_dynamic_array<std::string> COpenGLDriver::getSupportedGLSLExtensions() const
{
    constexpr size_t GLSLcnt = std::extent<decltype(m_GLSLExtensions)>::value;
    if (!m_supportedGLSLExtsNames)
    {
        size_t cnt = 0ull;
        for (size_t i = 0ull; i < GLSLcnt; ++i)
            cnt += (FeatureAvailable[m_GLSLExtensions[i]]);
        m_supportedGLSLExtsNames = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(cnt);
        for (size_t i = 0ull; i < GLSLcnt; ++i)
            if (FeatureAvailable[m_GLSLExtensions[i]])
                (*m_supportedGLSLExtsNames)[i] = (OpenGLFeatureStrings[m_GLSLExtensions[i]]);
    }

    return m_supportedGLSLExtsNames;
}

bool COpenGLDriver::bindGraphicsPipeline(video::IGPURenderpassIndependentPipeline* _gpipeline)
{
    SAuxContext* ctx = getThreadContext_helper(false);
    if (!ctx)
        return false;

    ctx->updateNextState_pipelineAndRaster(_gpipeline);

    return true;
}

bool COpenGLDriver::bindComputePipeline(video::IGPUComputePipeline* _cpipeline)
{
    SAuxContext* ctx = getThreadContext_helper(false);
    if (!ctx)
        return false;

    const COpenGLComputePipeline* glppln = static_cast<COpenGLComputePipeline*>(_cpipeline);
    ctx->nextState.pipeline.compute.usedShader = glppln ? glppln->getGLnameForCtx(ctx->ID) : 0u;
    ctx->nextState.pipeline.compute.pipeline = core::smart_refctd_ptr<const COpenGLComputePipeline>(glppln);

    return true;
}

bool COpenGLDriver::bindDescriptorSets(E_PIPELINE_BIND_POINT _pipelineType, const IGPUPipelineLayout* _layout,
    uint32_t _first, uint32_t _count, const IGPUDescriptorSet** _descSets, core::smart_refctd_dynamic_array<uint32_t>* _dynamicOffsets)
{
    //TODO take into account `_pipelineType` param and adjust this function when compute pipelines come in
    if (_first + _count >= IGPUPipelineLayout::DESCRIPTOR_SET_COUNT)
        return false;

    SAuxContext* ctx = getThreadContext_helper(false);
    if (!ctx)
        return false;

    const IGPUPipelineLayout* layouts[IGPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
    for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        layouts[i] = ctx->nextState.descriptorsParams[_pipelineType].descSets[i].pplnLayout.get();
    bindDescriptorSets_generic(_layout, _first, _count, _descSets, layouts);

    for (uint32_t i = 0u; i < IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
        if (!layouts[i])
            ctx->nextState.descriptorsParams[_pipelineType].descSets[i] = { nullptr, nullptr, nullptr };

    for (uint32_t i = 0u; i < _count; i++)
    {
        ctx->nextState.descriptorsParams[_pipelineType].descSets[_first + i] =
        {
        core::smart_refctd_ptr<const COpenGLPipelineLayout>(static_cast<const COpenGLPipelineLayout*>(_layout)),
        core::smart_refctd_ptr<const COpenGLDescriptorSet>(static_cast<const COpenGLDescriptorSet*>(_descSets[i])),
        _dynamicOffsets[i] //intentionally copy, not move
        };
    }

    return true;
}

bool COpenGLDriver::dispatch(uint32_t _groupCountX, uint32_t _groupCountY, uint32_t _groupCountZ)
{
    SAuxContext* ctx = getThreadContext_helper(false);
    if (!ctx)
        return false;

    ctx->flushStateCompute(GSB_ALL);

    extGlDispatchCompute(_groupCountX, _groupCountY, _groupCountZ);

    return true;
}

core::smart_refctd_ptr<IGPUShader> COpenGLDriver::createGPUShader(const asset::ICPUShader* _cpushader)
{
    return core::make_smart_refctd_ptr<COpenGLShader>(_cpushader);
}

core::smart_refctd_ptr<IGPUSpecializedShader> COpenGLDriver::createGPUSpecializedShader(const IGPUShader* _unspecialized, const asset::ISpecializationInfo* _specInfo)
{
    const COpenGLShader* glUnspec = static_cast<const COpenGLShader*>(_unspecialized);
    const asset::ICPUShader* cpuUnspec = glUnspec->getCPUCounterpart();

    const std::string& EP = _specInfo->entryPoint;
    const asset::E_SHADER_STAGE stage = _specInfo->shaderStage;

    core::smart_refctd_ptr<const asset::ICPUShader> spvCPUShader = nullptr;
    if (cpuUnspec->containsGLSL()) {
        std::string glsl = reinterpret_cast<const char*>(cpuUnspec->getSPVorGLSL()->getPointer());
        asset::ICPUShader::insertGLSLExtensionsDefines(glsl, getSupportedGLSLExtensions().get());
        auto spvShader = core::smart_refctd_ptr<asset::ICPUShader>(
            GLSLCompiler->createSPIRVFromGLSL(
                glsl.c_str(),
                stage,
                EP.c_str(),
                "????"
            ), core::dont_grab);
        if (!spvShader)
            return nullptr;

        spvCPUShader = core::smart_refctd_ptr<const asset::ICPUShader>(spvShader.get());
    }
    else {
        spvCPUShader = core::smart_refctd_ptr<const asset::ICPUShader>(cpuUnspec);
    }

    asset::CShaderIntrospector::SEntryPoint_Stage_Extensions introspectionParams{_specInfo->entryPoint, _specInfo->shaderStage, getSupportedGLSLExtensions()};
    asset::CShaderIntrospector introspector(GLSLCompiler.get(), introspectionParams);
    const asset::CIntrospectionData* introspection = introspector.introspect(spvCPUShader.get());
    if (introspection->pushConstant.present && introspection->pushConstant.info.name.empty())
    {
        assert(false);//abort on debug build
        os::Printer::log("Attempted to create OpenGL GPU specialized shader from SPIR-V without debug info - unable to set push constants. Creation failed.", ELL_ERROR);
        return nullptr;//release build: maybe let it return the shader
    }

    auto ctx = getThreadContext_helper(false);
    COpenGLSpecializedShader* gpuSpecialized = new COpenGLSpecializedShader(Params.AuxGLContexts + 1u, ctx->ID, this->ShaderLanguageVersion, spvCPUShader->getSPVorGLSL(), _specInfo, introspection);
    return core::smart_refctd_ptr<COpenGLSpecializedShader>(gpuSpecialized, core::dont_grab);
}

core::smart_refctd_ptr<IGPUBufferView> COpenGLDriver::createGPUBufferView(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset, size_t _size)
{
    if (!_underlying)
        return nullptr;
    const size_t effectiveSize = (_size != IGPUBufferView::whole_buffer) ? (_underlying->getSize() - _offset) : _size;
    if ((_offset + effectiveSize) > _underlying->getSize())
        return nullptr;
    if (!core::is_aligned_to(_offset, reqTBOAlignment)) //offset must be aligned to GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT
        return nullptr;
    if (!isAllowedBufferViewFormat(_fmt))
        return nullptr;
    if (effectiveSize > (maxTBOSizeInTexels * asset::getTexelOrBlockBytesize(_fmt)))
        return nullptr;

    COpenGLBuffer* glbuf = static_cast<COpenGLBuffer*>(_underlying);
    return core::make_smart_refctd_ptr<COpenGLBufferView>(core::smart_refctd_ptr<COpenGLBuffer>(glbuf), _fmt, _offset, _size);
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> COpenGLDriver::createGPUDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
{
    return core::make_smart_refctd_ptr<IGPUDescriptorSetLayout>(_begin, _end);//there's no COpenGLDescriptorSetLayout (no need for such)
}

core::smart_refctd_ptr<IGPUSampler> COpenGLDriver::createGPUSampler(const IGPUSampler::SParams& _params)
{
    return core::make_smart_refctd_ptr<COpenGLSampler>(_params);
}

core::smart_refctd_ptr<IGPUPipelineLayout> COpenGLDriver::createGPUPipelineLayout(const asset::SPushConstantRange* const _pcRangesBegin, const asset::SPushConstantRange* const _pcRangesEnd, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3)
{
    return core::make_smart_refctd_ptr<COpenGLPipelineLayout>(
        _pcRangesBegin, _pcRangesEnd,
        std::move(_layout0), std::move(_layout1),
        std::move(_layout2), std::move(_layout3)
        );
}

core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> COpenGLDriver::createGPURenderpassIndependentPipeline(core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>&& _parent, core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout, IGPUSpecializedShader** _shadersBegin, IGPUSpecializedShader** _shadersEnd, const asset::SVertexInputParams& _vertexInputParams, const asset::SBlendParams& _blendParams, const asset::SPrimitiveAssemblyParams& _primAsmParams, const asset::SRasterizationParams& _rasterParams)
{
    //_parent parameter is ignored

    using GLPpln = COpenGLRenderpassIndependentPipeline;

    auto shaders = core::SRange<IGPUSpecializedShader*>(_shadersBegin, _shadersEnd);
    auto vsIsPresent = [&shaders] {
        return std::find_if(shaders.begin(), shaders.end(), [](IGPUSpecializedShader* shdr) {return shdr->getStage()==asset::ESS_VERTEX;}) != shaders.end();
    };

    if (!_layout || vsIsPresent())
        return nullptr;

    return core::make_smart_refctd_ptr<COpenGLRenderpassIndependentPipeline>(
        nullptr,
        std::move(_layout),
        _shadersBegin, _shadersEnd,
        _vertexInputParams, _blendParams, _primAsmParams, _rasterParams
        );
}

core::smart_refctd_ptr<IGPUComputePipeline> COpenGLDriver::createGPUComputePipeline(core::smart_refctd_ptr<IGPUComputePipeline>&& _parent, core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout, core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader)
{
    if (!_layout || !_shader)
        return nullptr;
    if (_shader->getStage() != asset::ESS_COMPUTE)
        return nullptr;

    return core::make_smart_refctd_ptr<COpenGLComputePipeline>(std::move(_parent), std::move(_layout), std::move(_shader));
}

core::smart_refctd_ptr<IGPUDescriptorSet> COpenGLDriver::createGPUDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout, core::smart_refctd_dynamic_array<IGPUDescriptorSet::SDescriptorBinding>&& _descriptors)
{
    if (!_layout || !_descriptors || !_descriptors->size())
        return nullptr;

    return core::make_smart_refctd_ptr<COpenGLDescriptorSet>(std::move(_layout), std::move(_descriptors));
}

core::smart_refctd_ptr<IGPUDescriptorSet> COpenGLDriver::createGPUDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout)
{
    if (!_layout)
        return nullptr;

    return core::make_smart_refctd_ptr<COpenGLDescriptorSet>(std::move(_layout));
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

    if (currentQuery[query->getQueryObjectType()])
        return; //error

    query->grab();
    currentQuery[query->getQueryObjectType()] = query;


    extGlBeginQuery(queryGL->getType(),queryGL->getGLHandle());
    queryGL->flagBegun();
}
void COpenGLDriver::endQuery(IQueryObject* query)
{
    if (!query)
        return; //error
    if (currentQuery[query->getQueryObjectType()]!=query)
        return; //error

    COpenGLQuery* queryGL = static_cast<COpenGLQuery*>(query);
    if (queryGL->getGLHandle()==0||!queryGL->isActive())
        return;

    if (currentQuery[query->getQueryObjectType()])
        currentQuery[query->getQueryObjectType()]->drop();
    currentQuery[query->getQueryObjectType()] = nullptr;


    extGlEndQuery(queryGL->getType());
    queryGL->flagEnded();
}

// small helper function to create vertex buffer object adress offsets
static inline uint8_t* buffer_offset(const long offset)
{
	return ((uint8_t*)0 + offset);
}

static GLenum getGLprimitiveType(asset::E_PRIMITIVE_TOPOLOGY pt)
{
    using namespace asset;
    switch (pt)
    {
    case EPT_POINT_LIST:
        return GL_POINTS;
    case EPT_LINE_LIST:
        return GL_LINES;
    case EPT_LINE_STRIP:
        return GL_LINE_STRIP;
    case EPT_TRIANGLE_LIST:
        return GL_TRIANGLES;
    case EPT_TRIANGLE_STRIP:
        return GL_TRIANGLE_STRIP;
    case EPT_TRIANGLE_FAN:
        return GL_TRIANGLE_FAN;
    case EPT_LINE_LIST_WITH_ADJACENCY:
        return GL_LINES_ADJACENCY;
    case EPT_LINE_STRIP_WITH_ADJACENCY:
        return GL_LINE_STRIP_ADJACENCY;
    case EPT_TRIANGLE_LIST_WITH_ADJACENCY:
        return GL_TRIANGLES_ADJACENCY;
    case EPT_TRIANGLE_STRIP_WITH_ADJACENCY:
        return GL_TRIANGLE_STRIP_ADJACENCY;
    case EPT_PATCH_LIST:
        return GL_PATCHES;
    }
}


void COpenGLDriver::drawMeshBuffer(const IGPUMeshBuffer* mb)
{
    if (mb && !mb->getInstanceCount())
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;
    if (!found->nextState.pipeline.graphics.pipeline)
        return;

    found->updateNextState_vertexInput(mb->getVertexBufferBindings(), mb->getIndexBufferBinding()->buffer.get(), nullptr, nullptr);

#ifdef _IRR_DEBUG
	if (mb->getIndexCount() > getMaximalIndicesCount())
	{
		char tmp[1024];
		sprintf(tmp,"Could not draw, too many indices(%u), maxium is %u.", mb->getIndexCount(), getMaximalIndicesCount());
		os::Printer::log(tmp, ELL_ERROR);
	}
#endif // _IRR_DEBUG

	CNullDriver::drawMeshBuffer(mb);

	GLenum indexSize=0;
    if (mb->getIndexBufferBinding()->buffer)
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


    found->flushStateGraphics(GSB_ALL);

    GLenum primType = getGLprimitiveType(found->currentState.pipeline.graphics.pipeline->getPrimitiveAssemblyParams().primitiveType);
    if (primType==GL_POINTS)
        extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);

    if (indexSize) {
        static_assert(sizeof(mb->getIndexBufferBinding()->offset) == sizeof(void*), "Might break without this requirement");
        const void* const idxBufOffset = reinterpret_cast<void*>(mb->getIndexBufferBinding()->offset);
        extGlDrawElementsInstancedBaseVertexBaseInstance(primType, mb->getIndexCount(), indexSize, idxBufOffset, mb->getInstanceCount(), mb->getBaseVertex(), mb->getBaseInstance());
    }
    else
		extGlDrawArraysInstancedBaseInstance(primType, mb->getBaseVertex(), mb->getIndexCount(), mb->getInstanceCount(), mb->getBaseInstance());
}


//! Indirect Draw
void COpenGLDriver::drawArraysIndirect(const IGPUMeshBuffer::SBufferBinding _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer, size_t countOffset)
{
    if (!indirectDrawBuff)
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;
    if (!found->nextState.pipeline.graphics.pipeline)
        return;

    if (countBuffer && !FeatureAvailable[IRR_ARB_indirect_parameters] && (Version < 460u))
    {
        os::Printer::log("OpenGL driver: glMultiDrawArraysIndirectCount() not supported!");
        return;
    }
    if (!core::is_aligned_to(countOffset, 4ull))
    {
        os::Printer::log("COpenGLDriver::drawArraysIndirect: countOffset must be aligned to 4!");
        return;
    }

    found->updateNextState_vertexInput(_vtxBindings, found->nextState.vertexInputParams.indexBuf.get(), indirectDrawBuff, countBuffer);

    GLenum primType = getGLprimitiveType(found->currentState.pipeline.graphics.pipeline->getPrimitiveAssemblyParams().primitiveType);
    if (primType == GL_POINTS)
        extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);

    //actual drawing
    if (countBuffer)
        extGlMultiDrawArraysIndirectCount(primType, (void*)offset, countOffset, maxCount, stride);
    else
        extGlMultiDrawArraysIndirect(primType, (void*)offset, maxCount, stride);
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
        case EDF_INPUT_ATTACHMENTS:
            return 
                COpenGLExtensionHandler::FeatureAvailable[IRR_EXT_shader_pixel_local_storage] || 
                COpenGLExtensionHandler::FeatureAvailable[IRR_EXT_shader_framebuffer_fetch] ||
                COpenGLExtensionHandler::FeatureAvailable[IRR_EXT_shader_framebuffer_fetch_non_coherent];
        default:
            break;
	};
	return false;
}

void COpenGLDriver::drawIndexedIndirect(const IGPUMeshBuffer::SBufferBinding _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT],
                                        asset::E_PRIMITIVE_TOPOLOGY mode,
                                        asset::E_INDEX_TYPE indexType, const IGPUBuffer* indexBuff,
                                        const IGPUBuffer* indirectDrawBuff,
                                        size_t offset, size_t maxCount, size_t stride,
                                        const IGPUBuffer* countBuffer, size_t countOffset)
{
    if (!indirectDrawBuff)
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;
    if (!found->nextState.pipeline.graphics.pipeline)
        return;

    if (countBuffer && !FeatureAvailable[IRR_ARB_indirect_parameters] && (Version < 460u))
    {
        os::Printer::log("OpenGL driver: glMultiDrawElementsIndirectCount() not supported!");
        return;
    }
    if (!core::is_aligned_to(countOffset, 4ull))
    {
        os::Printer::log("COpenGLDriver::drawIndexedIndirect: countOffset must be aligned to 4!");
        return;
    }

    found->updateNextState_vertexInput(_vtxBindings, found->nextState.vertexInputParams.indexBuf.get(), indirectDrawBuff, countBuffer);

	GLenum indexSize = (indexType!=asset::EIT_16BIT) ? GL_UNSIGNED_INT:GL_UNSIGNED_SHORT;
    GLenum primType = getGLprimitiveType(found->currentState.pipeline.graphics.pipeline->getPrimitiveAssemblyParams().primitiveType);
    if (primType == GL_POINTS)
        extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);

    //actual drawing
    if (countBuffer)
        extGlMultiDrawElementsIndirectCount(primType, indexSize, (void*)offset, countOffset, maxCount, stride);
    else
        extGlMultiDrawElementsIndirect(primType,indexSize,(void*)offset,maxCount,stride);
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

void COpenGLDriver::SAuxContext::flushState_descriptors(E_PIPELINE_BIND_POINT _pbp, const COpenGLPipelineLayout* _currentLayout, const COpenGLPipelineLayout* _prevLayout)
{
    //bind new descriptor sets
    uint32_t compatibilityLimitPlusOne = 0u;
    if (_prevLayout && _currentLayout)
        compatibilityLimitPlusOne = _prevLayout->isCompatibleForSet(IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u, _currentLayout) + 1u;
    if (compatibilityLimitPlusOne >= IGPUPipelineLayout::DESCRIPTOR_SET_COUNT)
        compatibilityLimitPlusOne = 0u;

    int64_t newUboCount = 0u, newSsboCount = 0u, newTexCount = 0u, newImgCount = 0u;
    for (uint32_t i = 0u; i < video::IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        if (!_currentLayout)
            break;

        const auto& first_count = _currentLayout->getMultibindParamsForDescSet(i);

        {
            GLsizei count{};

#define CLAMP_COUNT(resname,limit,printstr) \
count = (first_count.resname.count - std::max(0, static_cast<int32_t>(first_count.resname.first + first_count.resname.count)-static_cast<int32_t>(limit)))

            CLAMP_COUNT(ubos, COpenGLExtensionHandler::maxUBOBindings, UBO);
            newUboCount = first_count.ubos.first + count;
            CLAMP_COUNT(ssbos, COpenGLExtensionHandler::maxSSBOBindings, SSBO);
            newSsboCount = first_count.ssbos.first + count;
            CLAMP_COUNT(textures, COpenGLExtensionHandler::maxTextureBindings, texture); //TODO should use maxTextureBindingsCompute for compute
            newTexCount = first_count.textures.first + count;
            CLAMP_COUNT(textureImages, COpenGLExtensionHandler::maxImageBindings, image);
            newImgCount = first_count.textureImages.first + count;
#undef CLAMP_COUNT
        }

        //if prev and curr pipeline layouts are compatible for set N, currState.set[N]==nextState.set[N] and the sets were bound with same dynamic offsets, then binding set N would be redundant
        if ((i < compatibilityLimitPlusOne) &&
            (effectivelyBoundDescriptors.descSets[i].set == nextState.descriptorsParams[_pbp].descSets[i].set) &&
            (effectivelyBoundDescriptors.descSets[i].dynamicOffsets == nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets)
        ) 
        {
            continue;
        }

        const auto& multibind_params = nextState.descriptorsParams[_pbp].descSets[i].set ?
            nextState.descriptorsParams[_pbp].descSets[i].set->getMultibindParams() :
            COpenGLDescriptorSet::SMultibindParams{};//all nullptr

        const GLsizei localUboCount = (newUboCount - first_count.ubos.first);//"local" as in this DS
        const GLsizei localSsboCount = (newSsboCount - first_count.ssbos.first);//"local" as in this DS
        //not entirely sure those MAXes are right
        constexpr size_t MAX_UBO_COUNT = 96ull;
        constexpr size_t MAX_SSBO_COUNT = 91ull;
        GLintptr uboOffsets_array[MAX_UBO_COUNT]{};
        GLintptr* const uboOffsets = nextState.descriptorsParams[_pbp].descSets[i].set ? uboOffsets_array : nullptr;
        GLintptr ssboOffsets_array[MAX_SSBO_COUNT]{};
        GLintptr* const ssboOffsets = nextState.descriptorsParams[_pbp].descSets[i].set ? ssboOffsets_array : nullptr;

        if (uboOffsets)
            memcpy(uboOffsets, multibind_params.ubos.offsets, localUboCount * sizeof(GLintptr));
        //if the loop below crashes, it means that there are dynamic UBOs in the DS, but the DS was bound with no (or not enough) dynamic offsets
        //or for some weird reason (bug) descSets[i].set is nullptr, but descSets[i].dynamicOffsets is not
        for (GLsizei u = 0u; ((u < localUboCount) && nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets); ++u)
            if (multibind_params.ubos.dynOffsetIxs[u] < nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets->size())
                uboOffsets[u] += nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets->operator[](multibind_params.ubos.dynOffsetIxs[u]);

        if (ssboOffsets)
            memcpy(ssboOffsets, multibind_params.ssbos.offsets, localSsboCount * sizeof(GLintptr));
        //if the loop below crashes, it means that there are dynamic SSBOs in the DS, but the DS was bound with no (or not enough) dynamic offsets
        //or for some weird reason (bug) descSets[i].set is nullptr, but descSets[i].dynamicOffsets is not
        for (GLsizei s = 0u; ((s < localSsboCount) && nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets); ++s)
            if (multibind_params.ssbos.dynOffsetIxs[s] < nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets->size())
                ssboOffsets[s] += nextState.descriptorsParams[_pbp].descSets[i].dynamicOffsets->operator[](multibind_params.ssbos.dynOffsetIxs[s]);

        extGlBindBuffersRange(GL_UNIFORM_BUFFER, first_count.ubos.first, localUboCount, multibind_params.ubos.buffers, uboOffsets, multibind_params.ubos.sizes);
        extGlBindBuffersRange(GL_SHADER_STORAGE_BUFFER, first_count.ssbos.first, localSsboCount, multibind_params.ssbos.buffers, ssboOffsets, multibind_params.ssbos.sizes);
        extGlBindTextures(first_count.textures.first, (newTexCount - first_count.textures.first), multibind_params.textures.textures, nullptr); //targets=nullptr: assuming ARB_multi_bind (or GL>4.4) is always available
        extGlBindSamplers(first_count.textures.first, (newTexCount - first_count.textures.first), multibind_params.textures.samplers);
        extGlBindImageTextures(first_count.textureImages.first, (newImgCount - first_count.textureImages.first), multibind_params.textureImages.textures, nullptr); //formats=nullptr: assuming ARB_multi_bind (or GL>4.4) is always available
    }

    //unbind previous descriptors if needed (if bindings not replaced by new multibind calls)
    if (_prevLayout)//if previous pipeline was nullptr, then no descriptors were bound
    {
        int64_t prevUboCount = 0u, prevSsboCount = 0u, prevTexCount = 0u, prevImgCount = 0u;
        const auto& first_count = _prevLayout->getMultibindParamsForDescSet(video::IGPUPipelineLayout::DESCRIPTOR_SET_COUNT - 1u);

        prevUboCount = first_count.ubos.first + first_count.ubos.count;
        prevSsboCount = first_count.ssbos.first + first_count.ssbos.count;
        prevTexCount = first_count.textures.first + first_count.textures.count;
        prevImgCount = first_count.textureImages.first + first_count.textureImages.count;

        int64_t diff = 0LL;
        if ((diff = prevUboCount - newUboCount) > 0LL)
            extGlBindBuffersRange(GL_UNIFORM_BUFFER, newUboCount, diff, nullptr, nullptr, nullptr);
        if ((diff = prevSsboCount - newSsboCount) > 0LL)
            extGlBindBuffersRange(GL_SHADER_STORAGE_BUFFER, newSsboCount, diff, nullptr, nullptr, nullptr);
        if ((diff = prevTexCount - newTexCount) > 0LL) {
            extGlBindTextures(newTexCount, diff, nullptr, nullptr);
            extGlBindSamplers(newTexCount, diff, nullptr);
        }
        if ((diff = prevImgCount - newImgCount) > 0LL)
            extGlBindImageTextures(newImgCount, diff, nullptr, nullptr);
    }

    //update state in state tracker
    for (uint32_t i = 0u; i < video::IGPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++i)
    {
        currentState.descriptorsParams[_pbp].descSets[i] = nextState.descriptorsParams[_pbp].descSets[i];
        effectivelyBoundDescriptors.descSets[i] = nextState.descriptorsParams[_pbp].descSets[i];
    }
}

void COpenGLDriver::SAuxContext::flushStateGraphics(GL_STATE_BITS stateBits)
{
    core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline> prevPipeline = currentState.pipeline.graphics.pipeline;
    if (stateBits & GSB_PIPELINE_AND_RASTER_PARAMETERS)
    {
        if (nextState.pipeline.graphics.pipeline != currentState.pipeline.graphics.pipeline)
        {
            if (nextState.pipeline.graphics.usedShadersHash != currentState.pipeline.graphics.usedShadersHash)
            {
                GLuint GLname = 0u;

                constexpr SOpenGLState::SGraphicsPipelineHash NULL_HASH = {0u, 0u, 0u, 0u, 0u};

                HashPipelinePair lookingFor{nextState.pipeline.graphics.usedShadersHash, {}};
                if (lookingFor.first != NULL_HASH)
                {
                    auto found = std::lower_bound(GraphicsPipelineMap.begin(), GraphicsPipelineMap.end(), lookingFor);
                    if (found != GraphicsPipelineMap.end() && found->first==nextState.pipeline.graphics.usedShadersHash)
                    {
                        GLname = found->second.GLname;
                        found->second.lastValidated = CNullDriver::ReallocationCounter;
                    }
                    else
                    {
                        GLname = createGraphicsPipeline(nextState.pipeline.graphics.usedShadersHash);
                        lookingFor.second.GLname = GLname;
                        lookingFor.second.lastValidated = CNullDriver::ReallocationCounter;
                        lookingFor.second.object = nextState.pipeline.graphics.pipeline;
                        freeUpGraphicsPipelineCache(true);
                        GraphicsPipelineMap.insert(found, lookingFor);
                    }
                }

                if (GLname)
                {
                    currentState.pipeline.compute.pipeline = nullptr;
                    currentState.pipeline.compute.usedShader = 0u;
                    extGlUseProgram(0);
                }
                extGlBindProgramPipeline(GLname);

                currentState.pipeline.graphics.usedShadersHash = nextState.pipeline.graphics.usedShadersHash;
            }

            currentState.pipeline.graphics.pipeline = nextState.pipeline.graphics.pipeline;
        }


#define STATE_NEQ(member) (nextState.member != currentState.member)
#define UPDATE_STATE(member) (currentState.member = nextState.member)
        decltype(glEnable)* disable_enable_fptr[2]{ &glDisable, &glEnable };

        if (STATE_NEQ(rasterParams.polygonMode)) {
            glPolygonMode(GL_FRONT_AND_BACK, nextState.rasterParams.polygonMode);
            UPDATE_STATE(rasterParams.polygonMode);
        }
        if (STATE_NEQ(rasterParams.faceCullingEnable)) {
            disable_enable_fptr[nextState.rasterParams.faceCullingEnable](GL_CULL_FACE);
            UPDATE_STATE(rasterParams.faceCullingEnable);
        }
        if (STATE_NEQ(rasterParams.cullFace)) {
            glCullFace(nextState.rasterParams.cullFace);
            UPDATE_STATE(rasterParams.cullFace);
        }
        if (STATE_NEQ(rasterParams.stencilTestEnable)) {
            disable_enable_fptr[nextState.rasterParams.stencilTestEnable](GL_STENCIL_TEST);
            UPDATE_STATE(rasterParams.stencilTestEnable);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilOp_front)) {
            COpenGLExtensionHandler::extGlStencilOpSeparate(GL_FRONT, nextState.rasterParams.stencilOp_front.sfail, nextState.rasterParams.stencilOp_front.dpfail, nextState.rasterParams.stencilOp_front.dppass);
            UPDATE_STATE(rasterParams.stencilOp_front);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilOp_back)) {
            COpenGLExtensionHandler::extGlStencilOpSeparate(GL_BACK, nextState.rasterParams.stencilOp_back.sfail, nextState.rasterParams.stencilOp_back.dpfail, nextState.rasterParams.stencilOp_back.dppass);
            UPDATE_STATE(rasterParams.stencilOp_back);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilFunc_front)) {
            COpenGLExtensionHandler::extGlStencilFuncSeparate(GL_FRONT, nextState.rasterParams.stencilFunc_front.func, nextState.rasterParams.stencilFunc_front.ref, nextState.rasterParams.stencilFunc_front.mask);
            UPDATE_STATE(rasterParams.stencilFunc_front);
        }
        if (nextState.rasterParams.stencilTestEnable && STATE_NEQ(rasterParams.stencilFunc_back)) {
            COpenGLExtensionHandler::extGlStencilFuncSeparate(GL_FRONT, nextState.rasterParams.stencilFunc_back.func, nextState.rasterParams.stencilFunc_back.ref, nextState.rasterParams.stencilFunc_back.mask);
            UPDATE_STATE(rasterParams.stencilFunc_back);
        }
        if (STATE_NEQ(rasterParams.depthTestEnable)) {
            disable_enable_fptr[nextState.rasterParams.depthTestEnable](GL_DEPTH_TEST);
            UPDATE_STATE(rasterParams.depthTestEnable);
        }
        if (nextState.rasterParams.depthTestEnable && STATE_NEQ(rasterParams.depthFunc)) {
            glDepthFunc(nextState.rasterParams.depthFunc);
            UPDATE_STATE(rasterParams.depthFunc);
        }
        if (STATE_NEQ(rasterParams.frontFace)) {
            glFrontFace(nextState.rasterParams.frontFace);
            UPDATE_STATE(rasterParams.frontFace);
        }
        if (STATE_NEQ(rasterParams.depthClampEnable)) {
            disable_enable_fptr[nextState.rasterParams.depthClampEnable](GL_DEPTH_CLAMP);
            UPDATE_STATE(rasterParams.depthClampEnable);
        }
        if (STATE_NEQ(rasterParams.rasterizerDiscardEnable)) {
            disable_enable_fptr[nextState.rasterParams.rasterizerDiscardEnable](GL_RASTERIZER_DISCARD);
            UPDATE_STATE(rasterParams.rasterizerDiscardEnable);
        }
        if (STATE_NEQ(rasterParams.polygonOffsetEnable)) {
            disable_enable_fptr[nextState.rasterParams.polygonOffsetEnable](GL_POLYGON_OFFSET_POINT);
            disable_enable_fptr[nextState.rasterParams.polygonOffsetEnable](GL_POLYGON_OFFSET_LINE);
            disable_enable_fptr[nextState.rasterParams.polygonOffsetEnable](GL_POLYGON_OFFSET_FILL);
            UPDATE_STATE(rasterParams.polygonOffsetEnable);
        }
        if (STATE_NEQ(rasterParams.polygonOffset)) {
            glPolygonOffset(nextState.rasterParams.polygonOffset.factor, nextState.rasterParams.polygonOffset.units);
            UPDATE_STATE(rasterParams.polygonOffset);
        }
        if (STATE_NEQ(rasterParams.lineWidth)) {
            glLineWidth(nextState.rasterParams.lineWidth);
            UPDATE_STATE(rasterParams.lineWidth);
        }
        if (STATE_NEQ(rasterParams.sampleShadingEnable)) {
            disable_enable_fptr[nextState.rasterParams.sampleShadingEnable](GL_SAMPLE_SHADING);
            UPDATE_STATE(rasterParams.sampleShadingEnable);
        }
        if (nextState.rasterParams.sampleShadingEnable && STATE_NEQ(rasterParams.minSampleShading)) {
            COpenGLExtensionHandler::extGlMinSampleShading(nextState.rasterParams.minSampleShading);
            UPDATE_STATE(rasterParams.minSampleShading);
        }
        if (STATE_NEQ(rasterParams.sampleMaskEnable)) {
            disable_enable_fptr[nextState.rasterParams.sampleMaskEnable](GL_SAMPLE_MASK);
            UPDATE_STATE(rasterParams.sampleMaskEnable);
        }
        if (nextState.rasterParams.sampleMaskEnable && STATE_NEQ(rasterParams.sampleMask[0])) {
            COpenGLExtensionHandler::extGlSampleMaski(0u, nextState.rasterParams.sampleMask[0]);
            UPDATE_STATE(rasterParams.sampleMask[0]);
        }
        if (nextState.rasterParams.sampleMaskEnable && STATE_NEQ(rasterParams.sampleMask[1])) {
            COpenGLExtensionHandler::extGlSampleMaski(1u, nextState.rasterParams.sampleMask[1]);
            UPDATE_STATE(rasterParams.sampleMask[1]);
        }
        if (STATE_NEQ(rasterParams.depthWriteEnable)) {
            glDepthMask(nextState.rasterParams.depthWriteEnable);
            UPDATE_STATE(rasterParams.depthWriteEnable);
        }
        if (STATE_NEQ(rasterParams.multisampleEnable)) {
            disable_enable_fptr[nextState.rasterParams.multisampleEnable](GL_MULTISAMPLE);
            UPDATE_STATE(rasterParams.multisampleEnable);
        }
        if (STATE_NEQ(rasterParams.primitiveRestartEnable)) {
            disable_enable_fptr[nextState.rasterParams.primitiveRestartEnable](GL_PRIMITIVE_RESTART);
            UPDATE_STATE(rasterParams.primitiveRestartEnable);
        }


        if (STATE_NEQ(rasterParams.logicOpEnable)) {
            disable_enable_fptr[nextState.rasterParams.logicOpEnable](GL_COLOR_LOGIC_OP);
            UPDATE_STATE(rasterParams.logicOpEnable);
        }
        if (STATE_NEQ(rasterParams.logicOp)) {
            glLogicOp(nextState.rasterParams.logicOp);
            UPDATE_STATE(rasterParams.logicOp);
        }
        decltype(COpenGLExtensionHandler::extGlEnablei)* disable_enable_indexed_fptr[2]{ &COpenGLExtensionHandler::extGlDisablei, &COpenGLExtensionHandler::extGlEnablei };
        for (GLuint i = 0u; asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; ++i)
        {
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].blendEnable)) {
                disable_enable_indexed_fptr[nextState.rasterParams.drawbufferBlend[i].blendEnable](GL_BLEND, i);
                UPDATE_STATE(rasterParams.drawbufferBlend[i].blendEnable);
            }
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].blendFunc)) {
                COpenGLExtensionHandler::extGlBlendFuncSeparatei(i,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.srcRGB,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.dstRGB,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.srcAlpha,
                    nextState.rasterParams.drawbufferBlend[i].blendFunc.dstAlpha
                );
                UPDATE_STATE(rasterParams.drawbufferBlend[i].blendFunc);
            }
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].blendEquation)) {
                COpenGLExtensionHandler::extGlBlendEquationSeparatei(i,
                    nextState.rasterParams.drawbufferBlend[i].blendEquation.modeRGB,
                    nextState.rasterParams.drawbufferBlend[i].blendEquation.modeAlpha
                );
                UPDATE_STATE(rasterParams.drawbufferBlend[i].blendEquation);
            }
            if (STATE_NEQ(rasterParams.drawbufferBlend[i].colorMask)) {
                COpenGLExtensionHandler::extGlColorMaski(i,
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[0],
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[1],
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[2],
                    nextState.rasterParams.drawbufferBlend[i].colorMask.colorWritemask[3]
                );
                UPDATE_STATE(rasterParams.drawbufferBlend[i].colorMask);
            }
        }
    }
    if ((stateBits & GSB_VAO_AND_VERTEX_INPUT) && currentState.pipeline.graphics.pipeline)
    {
        if (nextState.vertexInputParams.vao.second.GLname == 0u)
        {
            extGlBindVertexArray(0u);
            freeUpVAOCache(true);
            currentState.vertexInputParams.vao = { COpenGLRenderpassIndependentPipeline::SVAOHash(), {0u, 0u} };
        }
        else if (STATE_NEQ(vertexInputParams.vao.first))
        {
            bool brandNewVAO = false;//if VAO is taken from cache we don't have to modify VAO state that is part of hashval (everything except index and vertex buf bindings)
            auto hashVal = nextState.vertexInputParams.vao.first;
            auto it = std::lower_bound(VAOMap.begin(), VAOMap.end(), SOpenGLState::HashVAOPair{hashVal, SOpenGLState::SVAO{}});
            if (it != VAOMap.end() && it->first == hashVal) {
                it->second.lastValidated = CNullDriver::ReallocationCounter;
                currentState.vertexInputParams.vao = *it;
            }
            else
            {
                GLuint GLvao;
                COpenGLExtensionHandler::extGlCreateVertexArrays(1u, &GLvao);
                SOpenGLState::SVAO vao;
                vao.GLname = GLvao;
                vao.lastValidated = CNullDriver::ReallocationCounter;
                currentState.vertexInputParams.vao = SOpenGLState::HashVAOPair{hashVal, vao};
                VAOMap.insert(it, currentState.vertexInputParams.vao);
                brandNewVAO = true;
            }
            GLuint vao = currentState.vertexInputParams.vao.second.GLname;
            COpenGLExtensionHandler::extGlBindVertexArray(vao);

            bool updatedBindings[16]{};
            for (uint32_t attr = 0u; attr < 16u; ++attr)
            {
                if (hashVal.attribFormatAndComponentCount[attr] != asset::EF_UNKNOWN) {
                    if (brandNewVAO)
                        extGlEnableVertexArrayAttrib(currentState.vertexInputParams.vao.second.GLname, attr);
                }
                else 
                    continue;

                const uint32_t bnd = hashVal.getBindingForAttrib(attr);

                if (brandNewVAO)
                {
                    extGlVertexArrayAttribBinding(vao, attr, bnd);

                    const asset::E_FORMAT format = static_cast<asset::E_FORMAT>(hashVal.attribFormatAndComponentCount[attr]);

                    if (isFloatingPointFormat(format) && getTexelOrBlockBytesize(format) == getFormatChannelCount(format) * sizeof(double))//DOUBLE
                        extGlVertexArrayAttribLFormat(vao, attr, getFormatChannelCount(format), GL_DOUBLE, hashVal.getRelativeOffsetForAttrib(attr));
                    else if (isFloatingPointFormat(format) || isScaledFormat(format) || isNormalizedFormat(format))//FLOATING-POINT, SCALED ("weak integer"), NORMALIZED
                        extGlVertexArrayAttribFormat(vao, attr, isBGRALayoutFormat(format) ? GL_BGRA : getFormatChannelCount(format), formatEnumToGLenum(format), isNormalizedFormat(format) ? GL_TRUE : GL_FALSE, hashVal.getRelativeOffsetForAttrib(attr));
                    else if (isIntegerFormat(format))//INTEGERS
                        extGlVertexArrayAttribIFormat(vao, attr, getFormatChannelCount(format), formatEnumToGLenum(format), hashVal.getRelativeOffsetForAttrib(attr));

                    if (!updatedBindings[bnd]) {
                        extGlVertexArrayBindingDivisor(vao, bnd, hashVal.getDivisorForBinding(bnd));
                        updatedBindings[bnd] = true;
                    }
                }

                if (STATE_NEQ(vertexInputParams.bindings[bnd]))//this if-statement also doesnt allow GlVertexArrayVertexBuffer be called multiple times for single binding
                {
                    assert(nextState.vertexInputParams.bindings[bnd].buf);//something went wrong
                    extGlVertexArrayVertexBuffer(vao, bnd, nextState.vertexInputParams.bindings[bnd].buf->getOpenGLName(), nextState.vertexInputParams.bindings[bnd].offset, hashVal.getStrideForBinding(bnd));
                    UPDATE_STATE(vertexInputParams.bindings[bnd]);
                }
            }
            if (STATE_NEQ(vertexInputParams.indexBuf))
            {
                extGlVertexArrayElementBuffer(vao, nextState.vertexInputParams.indexBuf ? nextState.vertexInputParams.indexBuf->getOpenGLName() : 0u);
                UPDATE_STATE(vertexInputParams.indexBuf);
            }
        }
        if (STATE_NEQ(vertexInputParams.indirectDrawBuf))
        {
            extGlBindBuffer(GL_DRAW_INDIRECT_BUFFER, nextState.vertexInputParams.indirectDrawBuf ? nextState.vertexInputParams.indirectDrawBuf->getOpenGLName() : 0u);
            UPDATE_STATE(vertexInputParams.indirectDrawBuf);
        }
        if (STATE_NEQ(vertexInputParams.parameterBuf))
        {
            extGlBindBuffer(GL_PARAMETER_BUFFER, nextState.vertexInputParams.parameterBuf ? nextState.vertexInputParams.parameterBuf->getOpenGLName() : 0u);
            UPDATE_STATE(vertexInputParams.parameterBuf);
        }
    }
#undef STATE_NEQ
#undef UPDATE_STATE
    if (stateBits & GSB_DESCRIPTOR_SETS)
    {
        const COpenGLPipelineLayout* currLayout = static_cast<const COpenGLPipelineLayout*>(currentState.pipeline.graphics.pipeline->getLayout());
        const COpenGLPipelineLayout* prevLayout = static_cast<const COpenGLPipelineLayout*>(prevPipeline->getLayout());
        flushState_descriptors(EPBP_GRAPHICS, currLayout, prevLayout);
    }
}

void COpenGLDriver::SAuxContext::flushStateCompute(GL_STATE_BITS stateBits)
{
    core::smart_refctd_ptr<const COpenGLComputePipeline> prevPipeline = currentState.pipeline.compute.pipeline;

    if (stateBits & GSB_PIPELINE_AND_RASTER_PARAMETERS) //obviously "_AND_RASTER_PARAMETERS" is ignored while flushing compute state. Or shouldnt it?
    {
        if (nextState.pipeline.compute.usedShader != currentState.pipeline.compute.usedShader)
        {
            const GLuint GLname = nextState.pipeline.compute.usedShader;
            extGlUseProgram(GLname);
            currentState.pipeline.compute.usedShader = GLname;
        }
        if (nextState.pipeline.compute.pipeline != currentState.pipeline.compute.pipeline)
        {
            currentState.pipeline.compute.pipeline = nextState.pipeline.compute.pipeline;
        }
    }
    else if (stateBits & GSB_DESCRIPTOR_SETS)
    {
        const COpenGLPipelineLayout* currLayout = static_cast<const COpenGLPipelineLayout*>(currentState.pipeline.compute.pipeline->getLayout());
        const COpenGLPipelineLayout* prevLayout = static_cast<const COpenGLPipelineLayout*>(prevPipeline->getLayout());
        flushState_descriptors(EPBP_COMPUTE, currLayout, prevLayout);
    }
}

static GLenum getGLpolygonMode(asset::E_POLYGON_MODE pm)
{
    const static GLenum glpm[3]{ GL_FILL, GL_LINE, GL_POINT };
    return glpm[pm];
}
static GLenum getGLcullFace(asset::E_FACE_CULL_MODE cf)
{
    const static GLenum glcf[4]{ 0, GL_FRONT, GL_BACK, GL_FRONT_AND_BACK };
    return glcf[cf];
}
static GLenum getGLstencilOp(asset::E_STENCIL_OP so)
{
    static const GLenum glso[]{ GL_KEEP, GL_ZERO, GL_REPLACE, GL_INCR, GL_DECR, GL_INVERT, GL_INCR_WRAP, GL_DECR_WRAP };
    return glso[so];
}
static GLenum getGLcmpFunc(asset::E_COMPARE_OP sf)
{
    static const GLenum glsf[]{ GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS };
    return glsf[sf];
}
static GLenum getGLlogicOp(asset::E_LOGIC_OP lo)
{
    static const GLenum gllo[]{ GL_CLEAR, GL_AND, GL_AND_REVERSE, GL_COPY, GL_AND_INVERTED, GL_NOOP, GL_XOR, GL_OR, GL_NOR, GL_EQUIV, GL_INVERT, GL_OR_REVERSE,
        GL_COPY_INVERTED, GL_OR_INVERTED, GL_NAND, GL_SET
    };
    return gllo[lo];
}
static GLenum getGLblendFunc(asset::E_BLEND_FACTOR bf)
{
    static const GLenum glbf[]{ GL_ZERO , GL_ONE, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA,
        GL_SRC_ALPHA_SATURATE, GL_SRC1_COLOR, GL_ONE_MINUS_SRC1_COLOR, GL_SRC1_ALPHA, GL_ONE_MINUS_SRC1_ALPHA
    };
    return glbf[bf];
}
static GLenum getGLblendEq(asset::E_BLEND_OP bo)
{
    GLenum glbo[]{ GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX };
    if (bo >= std::extent<decltype(glbo)>::value)
        return GL_INVALID_ENUM;
    return glbo[bo];
}

GLuint COpenGLDriver::SAuxContext::createGraphicsPipeline(const SOpenGLState::SGraphicsPipelineHash& _hash)
{
    constexpr size_t STAGE_CNT = COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT;
    static_assert(STAGE_CNT == 5u, "SHADER_STAGE_COUNT is expected to be 5");
    const GLenum stages[5]{ GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const GLenum stageFlags[5]{ GL_VERTEX_SHADER_BIT, GL_TESS_CONTROL_SHADER_BIT, GL_TESS_EVALUATION_SHADER_BIT, GL_GEOMETRY_SHADER_BIT, GL_FRAGMENT_SHADER_BIT };

    GLuint GLpipeline = 0u;
    COpenGLExtensionHandler::extGlCreateProgramPipelines(1u, &GLpipeline);

    for (uint32_t ix = 0u; ix < STAGE_CNT; ++ix) {
        GLuint progName = _hash[ix];

        if (progName)
            COpenGLExtensionHandler::extGlUseProgramStages(GLpipeline, stageFlags[ix], progName);
    }

    return GLpipeline;
}

void COpenGLDriver::SAuxContext::updateNextState_pipelineAndRaster(const IGPURenderpassIndependentPipeline* _pipeline)
{
    nextState.pipeline.graphics.pipeline = core::smart_refctd_ptr<const COpenGLRenderpassIndependentPipeline>(
        static_cast<const COpenGLRenderpassIndependentPipeline*>(_pipeline)
    );
    if (!_pipeline)
    {
        SOpenGLState::SGraphicsPipelineHash hash;
        std::fill(hash.begin(), hash.end(), 0u);
        nextState.pipeline.graphics.usedShadersHash = hash;
        return;
    }
    SOpenGLState::SGraphicsPipelineHash hash;
    for (uint32_t i = 0u; i < COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++i)
    {
        hash[i] = nextState.pipeline.graphics.pipeline->getShaderAtIndex(i) ?
            static_cast<const COpenGLSpecializedShader*>(nextState.pipeline.graphics.pipeline->getShaderAtIndex(i))->getGLnameForCtx(this->ID) :
            0u;
    }
    nextState.pipeline.graphics.usedShadersHash = hash;

    const auto& ppln = nextState.pipeline.graphics.pipeline;

    const auto& raster_src = ppln->getRasterizationParams();
    auto& raster_dst = nextState.rasterParams;

    raster_dst.polygonMode = getGLpolygonMode(raster_src.polygonMode);
    if (raster_src.faceCullingMode == asset::EFCM_NONE)
        raster_dst.faceCullingEnable = 0;
    else {
        raster_dst.faceCullingEnable = 1;
        raster_dst.cullFace = getGLcullFace(raster_src.faceCullingMode);
    }
    
    const asset::SStencilOpParams* stencil_src[2]{ &raster_src.frontStencilOps, &raster_src.backStencilOps };
    decltype(raster_dst.stencilOp_front)* stencilo_dst[2]{ &raster_dst.stencilOp_front, &raster_dst.stencilOp_back };
    for (uint32_t i = 0u; i < 2u; ++i) {
        stencilo_dst[i]->sfail = getGLstencilOp(stencil_src[i]->failOp);
        stencilo_dst[i]->dpfail = getGLstencilOp(stencil_src[i]->depthFailOp);
        stencilo_dst[i]->dppass = getGLstencilOp(stencil_src[i]->passOp);
    }

    decltype(raster_dst.stencilFunc_front)* stencilf_dst[2]{ &raster_dst.stencilFunc_front, &raster_dst.stencilFunc_back };
    for (uint32_t i = 0u; i < 2u; ++i) {
        stencilf_dst[i]->func = getGLcmpFunc(stencil_src[i]->compareOp);
        stencilf_dst[i]->ref = stencil_src[i]->reference;
        stencilf_dst[i]->mask = stencil_src[i]->writeMask;
    }

    raster_dst.depthFunc = getGLcmpFunc(raster_src.depthCompareOp);
    raster_dst.frontFace = raster_src.frontFaceIsCCW ? GL_CCW : GL_CW;
    raster_dst.depthClampEnable = raster_src.depthClampEnable;
    raster_dst.rasterizerDiscardEnable = raster_src.rasterizerDiscard;

    raster_dst.polygonOffsetEnable = raster_src.depthBiasEnable;
    raster_dst.polygonOffset.factor = raster_src.depthBiasSlopeFactor;
    raster_dst.polygonOffset.units = raster_src.depthBiasSlopeFactor;

    raster_dst.sampleShadingEnable = raster_src.sampleShadingEnable;
    raster_dst.minSampleShading = raster_src.minSampleShading;

    //raster_dst.sampleMaskEnable = ???
    raster_dst.sampleMask[0] = raster_src.sampleMask[0];
    raster_dst.sampleMask[1] = raster_src.sampleMask[1];

    raster_dst.sampleAlphaToCoverageEnable = raster_src.alphaToCoverageEnable;
    raster_dst.sampleAlphaToOneEnable = raster_src.alphaToOneEnable;

    raster_dst.depthTestEnable = raster_src.depthTestEnable;
    raster_dst.depthWriteEnable = raster_src.depthWriteEnable;
    raster_dst.stencilTestEnable = raster_src.stencilTestEnable;

    raster_dst.multisampleEnable = (raster_src.rasterizationSamplesHint > asset::ESC_1_BIT);

    const auto& blend_src = ppln->getBlendParams();
    raster_dst.logicOpEnable = blend_src.logicOpEnable;
    raster_dst.logicOp = getGLlogicOp(static_cast<asset::E_LOGIC_OP>(blend_src.logicOp));

    for (size_t i = 0ull; i < asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; ++i) {
        const auto& attach_src = blend_src.blendParams[i];
        auto& attach_dst = raster_dst.drawbufferBlend[i];

        attach_dst.blendEnable = attach_src.blendEnable;
        attach_dst.blendFunc.srcRGB = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.srcColorFactor));
        attach_dst.blendFunc.dstRGB = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.dstColorFactor));
        attach_dst.blendFunc.srcAlpha = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.srcAlphaFactor));
        attach_dst.blendFunc.dstAlpha = getGLblendFunc(static_cast<asset::E_BLEND_FACTOR>(attach_src.dstAlphaFactor));

        attach_dst.blendEquation.modeRGB = getGLblendEq(static_cast<asset::E_BLEND_OP>(attach_src.colorBlendOp));
        assert(attach_dst.blendEquation.modeRGB != GL_INVALID_ENUM);
        attach_dst.blendEquation.modeAlpha = getGLblendEq(static_cast<asset::E_BLEND_OP>(attach_src.alphaBlendOp));
        assert(attach_dst.blendEquation.modeAlpha != GL_INVALID_ENUM);

        for (uint32_t j = 0u; j < 4u; ++j)
            attach_dst.colorMask.colorWritemask[j] = (attach_src.colorWriteMask>>j)&1u;
    }
}

void COpenGLDriver::SAuxContext::updateNextState_vertexInput(const IGPUMeshBuffer::SBufferBinding _vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT], const IGPUBuffer* _indexBuffer, const IGPUBuffer* _indirectDrawBuffer, const IGPUBuffer* _paramBuffer)
{
    for (size_t i = 0ull; i < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++i)
    {
        const IGPUMeshBuffer::SBufferBinding& bnd = _vtxBindings[i];
        if (bnd.buffer) {
            const COpenGLBuffer* buf = static_cast<COpenGLBuffer*>(bnd.buffer.get());
            nextState.vertexInputParams.bindings[i] = {core::smart_refctd_ptr<const COpenGLBuffer>(buf), static_cast<GLintptr>(bnd.offset)};
        }
    }
    const COpenGLBuffer* buf = static_cast<const COpenGLBuffer*>(_indexBuffer);
    nextState.vertexInputParams.indexBuf = core::smart_refctd_ptr<const COpenGLBuffer>(buf);

    buf = static_cast<const COpenGLBuffer*>(_indirectDrawBuffer);
    nextState.vertexInputParams.indirectDrawBuf = core::smart_refctd_ptr<const COpenGLBuffer>(buf);

    if (FeatureAvailable[IRR_ARB_indirect_parameters] || (Version >= 460u))
    {
        buf = static_cast<const COpenGLBuffer*>(_paramBuffer);
        nextState.vertexInputParams.parameterBuf = core::smart_refctd_ptr<const COpenGLBuffer>(buf);
    }

    //nextState.pipeline is the one set in updateNextState_pipelineAndRaster() or is the same object as currentState.pipeline
    nextState.vertexInputParams.vao.first = nextState.pipeline.graphics.pipeline->getVAOHash();
}



bool orderByMip(asset::CImageData* a, asset::CImageData* b)
{
    return a->getSupposedMipLevel() < b->getSupposedMipLevel();
}


//! returns a device dependent texture
core::smart_refctd_ptr<video::ITexture> COpenGLDriver::createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size,
																					uint32_t mipmapLevels, const io::path& name, asset::E_FORMAT format)
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

	auto internalFormat = COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format);
	if (internalFormat==GL_INVALID_ENUM)
		return nullptr;

    switch (type)
    {
        case ITexture::ETT_1D:
            return core::make_smart_refctd_ptr<COpenGL1DTexture>(internalFormat, size, mipmapLevels, name);
            break;
        case ITexture::ETT_2D:
            return core::make_smart_refctd_ptr<COpenGL2DTexture>(internalFormat, size, mipmapLevels, name);
            break;
        case ITexture::ETT_3D:
            return core::make_smart_refctd_ptr<COpenGL3DTexture>(internalFormat,size,mipmapLevels,name);
            break;
        case ITexture::ETT_1D_ARRAY:
            return core::make_smart_refctd_ptr<COpenGL1DTextureArray>(internalFormat,size,mipmapLevels,name);
            break;
        case ITexture::ETT_2D_ARRAY:
            return core::make_smart_refctd_ptr<COpenGL2DTextureArray>(internalFormat,size,mipmapLevels,name);
            break;
        case ITexture::ETT_CUBE_MAP:
            return core::make_smart_refctd_ptr<COpenGLCubemapTexture>(internalFormat,size,mipmapLevels,name);
            break;
        case ITexture::ETT_CUBE_MAP_ARRAY:
            return core::make_smart_refctd_ptr<COpenGLCubemapArrayTexture>(internalFormat,size,mipmapLevels,name);
            break;
        default:// ETT_CUBE_MAP, ETT_CUBE_MAP_ARRAY, ETT_TEXTURE_BUFFER
            break;
    }

    return nullptr;
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

core::smart_refctd_ptr<ITexture> COpenGLDriver::createGPUTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels, asset::E_FORMAT format)
{
    return createDeviceDependentTexture(type, size, mipmapLevels, "", format);
}

IMultisampleTexture* COpenGLDriver::createMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size, asset::E_FORMAT format, const bool& fixedSampleLocations)
{
    //check to implement later on  attachment of textures to FBO
    //if (!isFormatRenderable(glTex->getOpenGLInternalFormat()))
        //return nullptr;
    //! Vulkan and D3D only allow PoT sample counts
    if (core::isNPoT(samples))
        return nullptr;

	auto internalFormat = COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format);
	if (internalFormat == GL_INVALID_ENUM)
		return nullptr;

	IMultisampleTexture* tex = nullptr;
	switch (type)
	{
        case IMultisampleTexture::EMTT_2D:
            tex = new COpenGLMultisampleTexture(internalFormat,samples,size,fixedSampleLocations);
            break;
        case IMultisampleTexture::EMTT_2D_ARRAY:
            tex = new COpenGLMultisampleTextureArray(internalFormat,samples,size,fixedSampleLocations);
            break;
        default:
            break;
	}

	return tex;
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

/*
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
*/

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

    SAuxContext * found = getThreadContext_helper(false);
    if (!found)
        return;


    GLboolean rasterDiscard = 0;
    GLboolean colormask[4]{};
    GLboolean depthWrite = found->nextState.rasterParams.depthWriteEnable;
    GLuint smask_back = found->nextState.rasterParams.stencilFunc_back.mask;
    GLuint smask_front = found->nextState.rasterParams.stencilFunc_front.mask;

    if (copyDepth)
        found->nextState.rasterParams.depthWriteEnable = 1;
    if (copyStencil)
    {
        found->nextState.rasterParams.stencilFunc_back.mask = ~0u;
        found->nextState.rasterParams.stencilFunc_front.mask = ~0u;
    }
    clearColor_gatherAndOverrideState(found, 0u, &rasterDiscard, colormask);

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
                    width = std::min(rndrbl->getRenderableSize().Width,width);
                    height = std::min(rndrbl->getRenderableSize().Height,height);
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
                    width = std::min(rndrbl->getRenderableSize().Width,width);
                    height = std::min(rndrbl->getRenderableSize().Height,height);
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

    if (copyDepth)
        found->nextState.rasterParams.depthWriteEnable = depthWrite;
    if (copyStencil)
    {
        found->nextState.rasterParams.stencilFunc_back.mask = smask_back;
        found->nextState.rasterParams.stencilFunc_front.mask = smask_front;
    }
    clearColor_bringbackState(found, 0u, rasterDiscard, colormask);
}

void COpenGLDriver::clearColor_gatherAndOverrideState(SAuxContext * found, uint32_t _attIx, GLboolean* _rasterDiscard, GLboolean* _colorWmask)
{
    _rasterDiscard[0] = found->nextState.rasterParams.rasterizerDiscardEnable;
    memcpy(_colorWmask, found->nextState.rasterParams.drawbufferBlend[_attIx].colorMask.colorWritemask, 4);

    found->nextState.rasterParams.rasterizerDiscardEnable = 0;
    const GLboolean newmask[4]{ 1,1,1,1 };
    memcpy(found->nextState.rasterParams.drawbufferBlend[_attIx].colorMask.colorWritemask, newmask, sizeof(newmask));
    found->flushStateGraphics(GSB_PIPELINE_AND_RASTER_PARAMETERS);
}

void COpenGLDriver::clearColor_bringbackState(SAuxContext * found, uint32_t _attIx, GLboolean _rasterDiscard, const GLboolean * _colorWmask)
{
    found->nextState.rasterParams.rasterizerDiscardEnable = _rasterDiscard;
    memcpy(found->nextState.rasterParams.drawbufferBlend[_attIx].colorMask.colorWritemask, _colorWmask, 4);
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
            newRTTSize.Width = std::min(newRTTSize.Width,attachment->getRenderableSize().Width);
            newRTTSize.Height = std::min(newRTTSize.Height,attachment->getRenderableSize().Height);
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
    //found->flushStateGraphics(GSB_ALL); //! OPTIMIZE: Needed?


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


    GLboolean depthWrite = found->nextState.rasterParams.depthWriteEnable;
    GLboolean rasterizerDiscard = found->nextState.rasterParams.rasterizerDiscardEnable;

    found->nextState.rasterParams.depthWriteEnable = 1;
    found->nextState.rasterParams.rasterizerDiscardEnable = 0;
    found->flushStateGraphics(GSB_PIPELINE_AND_RASTER_PARAMETERS);

    if (found->CurrentFBO)
        extGlClearNamedFramebufferfv(found->CurrentFBO->getOpenGLName(),GL_DEPTH,0,&depth);
    else
        extGlClearNamedFramebufferfv(0,GL_DEPTH,0,&depth);

    found->nextState.rasterParams.depthWriteEnable = depthWrite;
    found->nextState.rasterParams.rasterizerDiscardEnable = rasterizerDiscard;
}

void COpenGLDriver::clearStencilBuffer(const int32_t &stencil)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    GLuint smask_back = found->nextState.rasterParams.stencilFunc_back.mask;
    GLuint smask_front = found->nextState.rasterParams.stencilFunc_front.mask;
    GLboolean rasterizedDiscard = found->nextState.rasterParams.rasterizerDiscardEnable;

    found->nextState.rasterParams.stencilFunc_back.mask = ~0u;
    found->nextState.rasterParams.stencilFunc_front.mask = ~0u;
    found->nextState.rasterParams.rasterizerDiscardEnable = 0;
    found->flushStateGraphics(GSB_PIPELINE_AND_RASTER_PARAMETERS);

    if (found->CurrentFBO)
        extGlClearNamedFramebufferiv(found->CurrentFBO->getOpenGLName(),GL_STENCIL,0,&stencil);
    else
        extGlClearNamedFramebufferiv(0,GL_STENCIL,0,&stencil);

    found->nextState.rasterParams.stencilFunc_back.mask = smask_back;
    found->nextState.rasterParams.stencilFunc_front.mask = smask_front;
    found->nextState.rasterParams.rasterizerDiscardEnable = rasterizedDiscard;
}

void COpenGLDriver::clearZStencilBuffers(const float &depth, const int32_t &stencil)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    GLboolean depthWrite = found->nextState.rasterParams.depthWriteEnable;
    GLuint smask_back = found->nextState.rasterParams.stencilFunc_back.mask;
    GLuint smask_front = found->nextState.rasterParams.stencilFunc_front.mask;
    GLboolean rasterizedDiscard = found->nextState.rasterParams.rasterizerDiscardEnable;

    found->nextState.rasterParams.depthWriteEnable = 1;
    found->nextState.rasterParams.stencilFunc_back.mask = ~0u;
    found->nextState.rasterParams.stencilFunc_front.mask = ~0u;
    found->nextState.rasterParams.rasterizerDiscardEnable = 0;
    found->flushStateGraphics(GSB_PIPELINE_AND_RASTER_PARAMETERS);

    if (found->CurrentFBO)
        extGlClearNamedFramebufferfi(found->CurrentFBO->getOpenGLName(),GL_DEPTH_STENCIL,0,depth,stencil);
    else
        extGlClearNamedFramebufferfi(0,GL_DEPTH_STENCIL,0,depth,stencil);

    found->nextState.rasterParams.depthWriteEnable = depthWrite;
    found->nextState.rasterParams.stencilFunc_back.mask = smask_back;
    found->nextState.rasterParams.stencilFunc_front.mask = smask_front;
    found->nextState.rasterParams.rasterizerDiscardEnable = rasterizedDiscard;
}

void COpenGLDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const int32_t* vals)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    if (attachment<EFAP_COLOR_ATTACHMENT0)
        return;

    const uint32_t attIx = attachment - EFAP_COLOR_ATTACHMENT0;
    GLboolean rasterizerDiscard = found->nextState.rasterParams.rasterizerDiscardEnable;
    GLboolean colormask[4]{};
    clearColor_gatherAndOverrideState(found, attIx, &rasterizerDiscard, colormask);

    if (found->CurrentFBO)
        extGlClearNamedFramebufferiv(found->CurrentFBO->getOpenGLName(),GL_COLOR,attIx,vals);
    else
        extGlClearNamedFramebufferiv(0,GL_COLOR,attIx,vals);

    clearColor_bringbackState(found, attIx, rasterizerDiscard, colormask);
}
void COpenGLDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const uint32_t* vals)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    if (attachment<EFAP_COLOR_ATTACHMENT0)
        return;

    const uint32_t attIx = attachment - EFAP_COLOR_ATTACHMENT0;
    GLboolean rasterizerDiscard = found->nextState.rasterParams.rasterizerDiscardEnable;
    GLboolean colormask[4]{};
    clearColor_gatherAndOverrideState(found, attIx, &rasterizerDiscard, colormask);

    if (found->CurrentFBO)
        extGlClearNamedFramebufferuiv(found->CurrentFBO->getOpenGLName(),GL_COLOR,attIx,vals);
    else
        extGlClearNamedFramebufferuiv(0,GL_COLOR,attIx,vals);

    clearColor_bringbackState(found, attIx, rasterizerDiscard, colormask);
}
void COpenGLDriver::clearColorBuffer(const E_FBO_ATTACHMENT_POINT &attachment, const float* vals)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    if (attachment<EFAP_COLOR_ATTACHMENT0)
        return;

    const uint32_t attIx = attachment - EFAP_COLOR_ATTACHMENT0;
    GLboolean rasterizerDiscard = 0;
    GLboolean colormask[4]{};
    clearColor_gatherAndOverrideState(found, attIx, &rasterizerDiscard, colormask);

    if (found->CurrentFBO)
        extGlClearNamedFramebufferfv(found->CurrentFBO->getOpenGLName(),GL_COLOR,attIx,vals);
    else
        extGlClearNamedFramebufferfv(0,GL_COLOR,attIx,vals);

    clearColor_bringbackState(found, attIx, rasterizerDiscard, colormask);
}

void COpenGLDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const float* vals)
{
    auto ctx = getThreadContext_helper(false);
    if (!ctx)
        return;

    GLboolean rasterDiscard;
    GLboolean colorWmask[4];
    clearColor_gatherAndOverrideState(ctx, 0u, &rasterDiscard, colorWmask);
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
    clearColor_bringbackState(ctx, 0u, rasterDiscard, colorWmask);
}
void COpenGLDriver::clearScreen(const E_SCREEN_BUFFERS &buffer, const uint32_t* vals)
{
    auto ctx = getThreadContext_helper(false);
    if (!ctx)
        return;

    GLboolean rasterDiscard;
    GLboolean colorWmask[4];
    clearColor_gatherAndOverrideState(ctx, 0u, &rasterDiscard, colorWmask);
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
    clearColor_bringbackState(ctx, 0u, rasterDiscard, colorWmask);
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
	io::IFileSystem* io, CIrrDeviceWin32* device, const const asset::IGLSLCompiler* glslcomp)
{
#ifdef _IRR_COMPILE_WITH_OPENGL_
	COpenGLDriver* ogl =  new COpenGLDriver(params, io, device, glslcomp);
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


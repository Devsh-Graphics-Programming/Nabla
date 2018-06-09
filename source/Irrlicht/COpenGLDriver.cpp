// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "COpenGLDriver.h"
// needed here also because of the create methods' parameters
#include "CNullDriver.h"
#include "CSkinnedMesh.h"

#include "vectorSIMD.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_

#include "COpenGL2DTexture.h"
#include "COpenGL3DTexture.h"
#include "COpenGL2DTextureArray.h"
#include "COpenGLCubemapTexture.h"
#include "COpenGLMultisampleTexture.h"
#include "COpenGLMultisampleTextureArray.h"
#include "COpenGLTextureBufferObject.h"

#include "COpenGLRenderBuffer.h"
#include "COpenGLPersistentlyMappedBuffer.h"
#include "COpenGLFrameBuffer.h"
#include "COpenGLSLMaterialRenderer.h"
#include "COpenGLOcclusionQuery.h"
#include "COpenGLTimestampQuery.h"
#include "os.h"

#ifdef _IRR_COMPILE_WITH_OSX_DEVICE_
#include "MacOSX/CIrrDeviceMacOSX.h"
#endif

#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_
#include <SDL/SDL.h>
#endif

namespace irr
{
namespace video
{

// -----------------------------------------------------------------------
// WINDOWS CONSTRUCTOR
// -----------------------------------------------------------------------
#ifdef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
//! Windows constructor and init code
COpenGLDriver::COpenGLDriver(const irr::SIrrlichtCreationParameters& params,
		io::IFileSystem* io, CIrrDeviceWin32* device)
: CNullDriver(io, params.WindowSize), COpenGLExtensionHandler(),
	CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(ECF_R8G8B8), Params(params),
	HDc(0), Window(static_cast<HWND>(params.WindowId)), Win32Device(device),
	DeviceType(EIDT_WIN32), AuxContexts(0)
{
	#ifdef _DEBUG
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
#ifdef _DEBUG
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
			WGL_FRAMEBUFFER_SRGB_CAPABLE_ARB, Params.HandleSRGB ? 1:0,
//			WGL_DEPTH_FLOAT_EXT, 1,
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
		WGL_CONTEXT_MINOR_VERSION_ARB, 5,
		WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};
	// create rendering context
	hrc=wglCreateContextAttribs_ARB(HDc, 0, iAttribs);
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

    AuxContexts = new SAuxContext[Params.AuxGLContexts+1];
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
		return false;
	}


	int pf = GetPixelFormat(HDc);
	DescribePixelFormat(HDc, pf, sizeof(PIXELFORMATDESCRIPTOR), &pfd);
	if (pfd.cAlphaBits != 0)
	{
		if (pfd.cRedBits == 8)
			ColorFormat = ECF_A8R8G8B8;
		else
			ColorFormat = ECF_A1R5G5B5;
	}
	else
	{
		if (pfd.cRedBits == 8)
			ColorFormat = ECF_R8G8B8;
		else
			ColorFormat = ECF_R5G6B5;
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
        cleanUpContextBeforeDelete();
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
: CNullDriver(io, params.WindowSize), COpenGLExtensionHandler(),
	CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(ECF_R8G8B8),
	Params(params),
	OSXDevice(device), DeviceType(EIDT_OSX), AuxContexts(0)
{
	#ifdef _DEBUG
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
: CNullDriver(io, params.WindowSize), COpenGLExtensionHandler(),
	CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(ECF_R8G8B8),
	Params(params), X11Device(device), DeviceType(EIDT_X11), AuxContexts(0)
{
	#ifdef _DEBUG
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
    bool retval = false;
    glContextMutex->Get();
    SAuxContext* found = getThreadContext_helper(true);
    if (found)
    {
        cleanUpContextBeforeDelete();
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
: CNullDriver(io, params.WindowSize), COpenGLExtensionHandler(),
	CurrentRenderMode(ERM_NONE), ResetRenderStates(true), ColorFormat(ECF_R8G8B8),
	CurrentTarget(ERT_FRAME_BUFFER), Params(params),
	SDLDevice(device), DeviceType(EIDT_SDL), AuxContexts(0)
{
	#ifdef _DEBUG
	setDebugName("COpenGLDriver");
	#endif

	genericDriverInit();
}

#endif // _IRR_COMPILE_WITH_SDL_DEVICE_


//! destructor
COpenGLDriver::~COpenGLDriver()
{
    cleanUpContextBeforeDelete();

	deleteMaterialRenders();
    removeAllRenderBuffers();
	// I get a blue screen on my laptop, when I do not delete the
	// textures manually before releasing the dc. Oh how I love this.
	deleteAllTextures();

    glContextMutex->Get();
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
    delete [] AuxContexts;
    glContextMutex->Release();
    delete glContextMutex;
}


// -----------------------------------------------------------------------
// METHODS
// -----------------------------------------------------------------------

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
	for(std::unordered_map<COpenGLVAOSpec::HashAttribs,SAuxContext::COpenGLVAO*>::iterator it = found->VAOMap.begin(); it != found->VAOMap.end(); it++)
    {
        delete it->second;
    }
    found->VAOMap.clear();

	found->CurrentTexture.clear();

	for(std::unordered_map<uint64_t,GLuint>::iterator it = found->SamplerMap.begin(); it != found->SamplerMap.end(); it++)
    {
        extGlDeleteSamplers(1,&it->second);
    }
    found->SamplerMap.clear();

    glFinish();
}


bool COpenGLDriver::genericDriverInit()
{
    glContextMutex = new FW_Mutex();

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


    maxConcurrentShaderInvocations = 0;
    maxALUShaderInvocations = 0;
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
/*
	Params.HandleSRGB &= ((FeatureAvailable[IRR_ARB_framebuffer_sRGB] || FeatureAvailable[IRR_EXT_framebuffer_sRGB]) &&
		FeatureAvailable[IRR_EXT_texture_sRGB]);

	if (Params.HandleSRGB)
		glEnable(GL_FRAMEBUFFER_SRGB);
*/
    glDisable(GL_DITHER);
    glDisable(GL_MULTISAMPLE);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glClearDepth(0.0);
	///glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glDepthFunc(GL_GEQUAL);
	glDepthRange(1.0,0.0);
	glFrontFace(GL_CW);

	// adjust flat coloring scheme to DirectX version
	///extGlProvokingVertex(GL_FIRST_VERTEX_CONVENTION_EXT);

	// create material renderers
	createMaterialRenderers();

	// set the renderstates
	setRenderStates3DMode();

	// We need to reset once more at the beginning of the first rendering.
	// This fixes problems with intermediate changes to the material during texture load.
	ResetRenderStates = true;

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
    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE &materialType, const core::array<video::SConstantLocationNamePair> &constants)
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
    virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SMaterial &material, const video::SMaterial &lastMaterial)
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
    //"#version 430 core\n"
    "#version 400 core\n"
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
    //"#version 430 core\n"
    "#version 400 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   outColor = texture(tex0,tcCoord);"
    "}";
    const char* std_trans_add_frag =
    //"#version 430 core\n"
    "#version 400 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   outColor = texture(tex0,tcCoord);"
    "}";
    const char* std_trans_alpha_frag =
    //"#version 430 core\n"
    "#version 400 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "uniform sampler2D tex0;"
    "\n"
    "void main()\n"
    "{\n"
    "   vec4 tmp = texture(tex0,tcCoord)*vxCol;\n"
    "   if (tmp.a<0.00000000000000000000000000000000001)\n"
    "       discard;\n"
    "   outColor = tmp;"
    "}";
    const char* std_trans_vertex_frag =
    //"#version 430 core\n"
    "#version 400 core\n"
    "in vec4 vxCol;\n"
    "in vec2 tcCoord;\n"
    "\n"
    "layout(location = 0) out vec4 outColor;\n"
    "\n"
    "uniform sampler2D tex0;"
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

	//glFlush();

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

#if defined(_IRR_COMPILE_WITH_SDL_DEVICE_)
	if (DeviceType == EIDT_SDL)
	{
		// todo: SDL sets glFrontFace(GL_CCW) after driver creation,
		// it would be better if this was fixed elsewhere.
		glFrontFace(GL_CW);
	}
#endif

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


IGPUBuffer* COpenGLDriver::createGPUBuffer(const size_t &size, const void* data, const bool canModifySubData, const bool &inCPUMem, const E_GPU_BUFFER_ACCESS &usagePattern)
{
    switch (usagePattern)
    {
        case EGBA_READ:
            return new COpenGLBuffer(size,data,(canModifySubData ? GL_DYNAMIC_STORAGE_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_READ_BIT);
        case EGBA_WRITE:
            return new COpenGLBuffer(size,data,(canModifySubData ? GL_DYNAMIC_STORAGE_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_WRITE_BIT);
        case EGBA_READ_WRITE:
            return new COpenGLBuffer(size,data,(canModifySubData ? GL_DYNAMIC_STORAGE_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT);
        default:
            return new COpenGLBuffer(size,data,(canModifySubData ? GL_DYNAMIC_STORAGE_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0));
    }
}

IGPUMappedBuffer* COpenGLDriver::createPersistentlyMappedBuffer(const size_t &size, const void* data, const E_GPU_BUFFER_ACCESS &usagePattern, const bool &assumedCoherent, const bool &inCPUMem)
{
    switch (usagePattern)
    {
        case EGBA_READ:
            return new COpenGLPersistentlyMappedBuffer(size,data,GL_MAP_PERSISTENT_BIT|(assumedCoherent ? GL_MAP_COHERENT_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_READ_BIT,GL_MAP_PERSISTENT_BIT|(assumedCoherent ? GL_MAP_COHERENT_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_READ_BIT);
        case EGBA_WRITE:
            return new COpenGLPersistentlyMappedBuffer(size,data,GL_MAP_PERSISTENT_BIT|(assumedCoherent ? GL_MAP_COHERENT_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_WRITE_BIT,GL_MAP_PERSISTENT_BIT|(assumedCoherent ? GL_MAP_COHERENT_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_WRITE_BIT);
        case EGBA_READ_WRITE:
            return new COpenGLPersistentlyMappedBuffer(size,data,GL_MAP_PERSISTENT_BIT|(assumedCoherent ? GL_MAP_COHERENT_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT,GL_MAP_PERSISTENT_BIT|(assumedCoherent ? GL_MAP_COHERENT_BIT:0)|(inCPUMem ? GL_CLIENT_STORAGE_BIT:0)|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT);
        default:
            return NULL;
    }
}

void COpenGLDriver::bufferCopy(IGPUBuffer* readBuffer, IGPUBuffer* writeBuffer, const size_t& readOffset, const size_t& writeOffset, const size_t& length)
{
    COpenGLBuffer* readbuffer = static_cast<COpenGLBuffer*>(readBuffer);
    COpenGLBuffer* writebuffer = static_cast<COpenGLBuffer*>(writeBuffer);
    extGlCopyNamedBufferSubData(readbuffer->getOpenGLName(),writebuffer->getOpenGLName(),readOffset,writeOffset,length);
}

scene::IGPUMeshDataFormatDesc* COpenGLDriver::createGPUMeshDataFormatDesc(core::LeakDebugger* dbgr)
{
    return new COpenGLVAOSpec(dbgr);
}

scene::IGPUMesh* COpenGLDriver::createGPUMeshFromCPU(scene::ICPUMesh* mesh, const E_MESH_DESC_CONVERT_BEHAVIOUR& bufferOptions)
{
    scene::IGPUMesh* outmesh;
    switch (mesh->getMeshType())
    {
        case scene::EMT_ANIMATED_SKINNED:
            outmesh = new scene::CGPUSkinnedMesh(static_cast<scene::ICPUSkinnedMesh*>(mesh)->getBoneReferenceHierarchy());
            break;
        default:
            outmesh = new scene::SGPUMesh();
            break;
    }


    for (size_t i=0; i<mesh->getMeshBufferCount(); i++)
    {
        scene::ICPUMeshBuffer* origmeshbuf = mesh->getMeshBuffer(i);
        scene::ICPUMeshDataFormatDesc* origdesc = static_cast<scene::ICPUMeshDataFormatDesc*>(origmeshbuf->getMeshDataAndFormat());
        if (!origdesc)
            continue;

        bool success = true;
        bool noAttributes = true;
        const core::ICPUBuffer* oldbuffer[scene::EVAI_COUNT];
        scene::E_COMPONENTS_PER_ATTRIBUTE components[scene::EVAI_COUNT];
        scene::E_COMPONENT_TYPE componentTypes[scene::EVAI_COUNT];
        for (size_t j=0; j<scene::EVAI_COUNT; j++)
        {
            oldbuffer[j] = origdesc->getMappedBuffer((scene::E_VERTEX_ATTRIBUTE_ID)j);
            if (oldbuffer[j])
                noAttributes = false;

            scene::E_VERTEX_ATTRIBUTE_ID attrId = (scene::E_VERTEX_ATTRIBUTE_ID)j;
            components[attrId] = origdesc->getAttribComponentCount(attrId);
            componentTypes[attrId] = origdesc->getAttribType(attrId);
            if (scene::vertexAttrSize[componentTypes[attrId]][components[attrId]]>=0xdeadbeefu)
            {
                os::Printer::log("createGPUMeshFromCPU input ICPUMeshBuffer(s) have one or more invalid attribute specs!\n",ELL_ERROR);
                success = false;
            }
        }
        if (noAttributes||!success)
            continue;
        //
        int64_t oldBaseVertex;
        size_t indexBufferByteSize = 0;
        void* newIndexBuffer = NULL;
        uint32_t indexRange;
        //set indexCount
        scene::IGPUMeshBuffer* meshbuffer = new scene::IGPUMeshBuffer();
        meshbuffer->setIndexCount(origmeshbuf->getIndexCount());
        if (origdesc->getIndexBuffer())
        {
            //set indices
            uint32_t minIx = 0xffffffffu;
            uint32_t maxIx = 0;
            bool success = origmeshbuf->getIndexCount()>0;
            for (size_t j=0; success&&j<origmeshbuf->getIndexCount(); j++)
            {
                uint32_t ix;
                switch (origmeshbuf->getIndexType())
                {
                    case EIT_16BIT:
                        ix = ((uint16_t*)origmeshbuf->getIndices())[j];
                        break;
                    case EIT_32BIT:
                        ix = ((uint32_t*)origmeshbuf->getIndices())[j];
                        break;
                    default:
                        success = false;
                        break;
                }
                if (ix<minIx)
                    minIx = ix;
                if (ix>maxIx)
                    maxIx = ix;
            }

            if (int64_t(minIx)+origmeshbuf->getBaseVertex()<0)
            {
                meshbuffer->drop();
                continue;
            }

            //nothing will work if this is fucked
            for (size_t j=0; j<scene::EVAI_COUNT; j++)
            {
                if (!oldbuffer[j])
                    continue;

                scene::E_VERTEX_ATTRIBUTE_ID attrId = (scene::E_VERTEX_ATTRIBUTE_ID)j;

                size_t byteEnd = origdesc->getMappedBufferOffset(attrId);
                if (origdesc->getAttribDivisor(attrId))
                    byteEnd += (origmeshbuf->getInstanceCount()+origmeshbuf->getBaseInstance()-1)*origdesc->getMappedBufferStride(attrId);
                else
                    byteEnd += (int64_t(maxIx)+origmeshbuf->getBaseVertex())*origdesc->getMappedBufferStride(attrId);
                byteEnd += scene::vertexAttrSize[componentTypes[attrId]][components[attrId]];

                if (byteEnd>oldbuffer[j]->getSize())
                    success = false;
            }
            // kill MB
            if (!success)
            {
                meshbuffer->drop();
                continue;
            }
            oldBaseVertex = int64_t(minIx)+origmeshbuf->getBaseVertex();
            indexRange = maxIx-minIx;
            if (indexRange<0x10000u)
            {
                meshbuffer->setIndexType(EIT_16BIT);
                indexBufferByteSize = meshbuffer->getIndexCount()*2;
            }
            else
            {
                meshbuffer->setIndexType(EIT_32BIT);
                indexBufferByteSize = meshbuffer->getIndexCount()*4;
            }
            newIndexBuffer = malloc(indexBufferByteSize);
            //doesnt matter if shared VAO or not, range gets checked before baseVx

            if (origmeshbuf->getIndexType()==meshbuffer->getIndexType()&&minIx==0)
                memcpy(newIndexBuffer,origmeshbuf->getIndices(),indexBufferByteSize);
            else
            {
                for (size_t j=0; j<origmeshbuf->getIndexCount(); j++)
                {
                    uint32_t ix;
                    if (origmeshbuf->getIndexType()==EIT_16BIT)
                        ix = ((uint16_t*)origmeshbuf->getIndices())[j];
                    else
                        ix = ((uint32_t*)origmeshbuf->getIndices())[j];

                    ix -= minIx;
                    if (indexRange<0x10000u)
                        ((uint16_t*)newIndexBuffer)[j] = ix;
                    else
                        ((uint32_t*)newIndexBuffer)[j] = ix;
                }
            }
        }
        else
        {
            oldBaseVertex = origmeshbuf->getBaseVertex();
            //
            int64_t bigIx = origmeshbuf->getIndexCount();
            bool success = bigIx!=0;
            bigIx--;
            bigIx += oldBaseVertex;
            //check for overflow
            for (size_t j=0; success&&j<scene::EVAI_COUNT; j++)
            {
                if (!oldbuffer[j])
                    continue;

                scene::E_VERTEX_ATTRIBUTE_ID attrId = (scene::E_VERTEX_ATTRIBUTE_ID)j;

                int64_t byteEnd = origdesc->getMappedBufferOffset(attrId);
                if (origdesc->getAttribDivisor(attrId))
                    byteEnd += (origmeshbuf->getInstanceCount()+origmeshbuf->getBaseInstance()-1)*origdesc->getMappedBufferStride(attrId);
                else
                    byteEnd += bigIx*origdesc->getMappedBufferStride(attrId);
                byteEnd += scene::vertexAttrSize[componentTypes[attrId]][components[attrId]];

                if (byteEnd>oldbuffer[j]->getSize())
                    success = false;
            }
            // kill MB
            if (!success)
            {
                meshbuffer->drop();
                continue;
            }
            indexRange = origmeshbuf->getIndexCount()-1;
        }
        //set bbox
        core::aabbox3df oldBBox = origmeshbuf->getBoundingBox();
	if (mesh->getMeshType()!=scene::EMT_ANIMATED_SKINNED)
		origmeshbuf->recalculateBoundingBox();
        meshbuffer->setBoundingBox(origmeshbuf->getBoundingBox());
	if (mesh->getMeshType()!=scene::EMT_ANIMATED_SKINNED)
        	origmeshbuf->setBoundingBox(oldBBox);
        //set primitive type
        meshbuffer->setPrimitiveType(origmeshbuf->getPrimitiveType());
        //set material
        meshbuffer->getMaterial() = origmeshbuf->getMaterial();


        size_t bufferBindings[scene::EVAI_COUNT] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        int64_t attrStride[scene::EVAI_COUNT];
        size_t attrOffset[scene::EVAI_COUNT];
        size_t bufferMin[scene::EVAI_COUNT];
        size_t bufferMax[scene::EVAI_COUNT];
        for (size_t j=0; j<scene::EVAI_COUNT; j++)
        {
            if (!oldbuffer[j])
                continue;

            bool alternateBinding = false;
            for (size_t k=0; k<j; k++)
            {
                if (oldbuffer[j]==oldbuffer[k])
                {
                    alternateBinding = true;
                    bufferBindings[j] = k;
                    break;
                }
            }


            scene::E_VERTEX_ATTRIBUTE_ID attrId = (scene::E_VERTEX_ATTRIBUTE_ID)j;

            attrStride[j] = origdesc->getMappedBufferStride(attrId);
            attrOffset[j] = origdesc->getMappedBufferOffset(attrId);
            //
            size_t minMemPos = attrOffset[j];
            if (origdesc->getAttribDivisor(attrId))
                minMemPos += origmeshbuf->getBaseInstance()*attrStride[j];
            else
                minMemPos += oldBaseVertex*attrStride[j];

            if (!alternateBinding)
                bufferMin[j] = minMemPos;
            else if (minMemPos<bufferMin[bufferBindings[j]])
                bufferMin[bufferBindings[j]] = minMemPos;
            //
            size_t maxMemPos = minMemPos;
            if (origdesc->getAttribDivisor(attrId))
                maxMemPos += (origmeshbuf->getInstanceCount()-1)*attrStride[j];
            else
                maxMemPos += indexRange*attrStride[j];
            maxMemPos += scene::vertexAttrSize[componentTypes[attrId]][components[attrId]];

            if (!alternateBinding)
                bufferMax[j] = maxMemPos;
            else if (maxMemPos>bufferMax[bufferBindings[j]])
                bufferMax[bufferBindings[j]] = maxMemPos;
        }
        scene::IGPUMeshDataFormatDesc* desc = createGPUMeshDataFormatDesc();
        meshbuffer->setMeshDataAndFormat(desc);
        desc->drop();
        ///since we only copied relevant shit over
        //meshbuffer->setBaseVertex(0);
        //if (newIndexBuffer)
            //meshbuffer->setIndexBufferOffset(0);
        switch (bufferOptions)
        {
            //! It would be beneficial if this function compacted subdata ranges of used buffers to eliminate unused bytes
            //! But for This maybe a ICPUMeshBuffer "isolate" function is needed outside of this API, so we dont bother here?
            case EMDCB_CLONE_AND_MIRROR_LAYOUT:
                {
                    if (newIndexBuffer)
                    {
                        IGPUBuffer* indexBuf = createGPUBuffer(indexBufferByteSize,newIndexBuffer);
                        desc->mapIndexBuffer(indexBuf);
                        indexBuf->drop();
                    }

                    size_t allocatedGPUBuffers = 0;
                    IGPUBuffer* attrBuf[scene::EVAI_COUNT] = {NULL};
                    for (size_t j=0; j<scene::EVAI_COUNT; j++)
                    {
                        if (!oldbuffer[j])
                            continue;

                        if (bufferBindings[j]==j)
                            attrBuf[j] = createGPUBuffer(bufferMax[j]-bufferMin[j],((uint8_t*)oldbuffer[j]->getPointer())+bufferMin[j]);

                        scene::E_VERTEX_ATTRIBUTE_ID attrId = (scene::E_VERTEX_ATTRIBUTE_ID)j;
                        desc->mapVertexAttrBuffer(attrBuf[bufferBindings[j]],attrId,components[attrId],componentTypes[attrId],attrStride[attrId],attrOffset[attrId]+oldBaseVertex*attrStride[j]-bufferMin[bufferBindings[j]],origdesc->getAttribDivisor((scene::E_VERTEX_ATTRIBUTE_ID)j));
                        if (bufferBindings[j]==j)
                            attrBuf[bufferBindings[j]]->drop();
                    }
                }
                break;
            /**
            These conversion functions need to take into account the empty space (unused data) in buffers to avoid duplication
            This is why they are unfinished
            case EMDCB_PACK_ATTRIBUTES_SINGLE_BUFFER:
                {
                    if (newIndexBuffer)
                    {
                        IGPUBuffer* indexBuf = createGPUBuffer(indexBufferByteSize,newIndexBuffer);
                        desc->mapIndexBuffer(indexBuf);
                        indexBuf->drop();
                    }

                    size_t allocatedGPUBuffers = 0;
                    for (size_t j=0; j<scene::EVAI_COUNT; j++)
                    {
                        if (!oldbuffer[j])
                            continue;

                        if (bufferBindings[j]==j)
                            attrBuf[j] = createGPUBuffer(bufferMax[j]-bufferMin[j],((uint8_t*)oldbuffer[j]->getPointer())+bufferMin[j]);

                        scene::E_VERTEX_ATTRIBUTE_ID attrId = (scene::E_VERTEX_ATTRIBUTE_ID)j;
                        desc->mapVertexAttrBuffer(attrBuf[bufferBindings[j]],attrId,components[attrId],componentTypes[attrId],attrStride[attrId],attrOffset[attrId]+oldBaseVertex*attrStride[j]-bufferMin[bufferBindings[j]]);
                        if (bufferBindings[j]==j)
                            attrBuf[bufferBindings[j]]->drop();
                    }
                }
                break;
            case EMDCB_PACK_ALL_SINGLE_BUFFER:
                break;**/
            case EMDCB_INTERLEAVED_PACK_ATTRIBUTES_SINGLE_BUFFER:
            case EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER:
                {
                    size_t vertexSize = 0;
                    uint8_t* inPtr[scene::EVAI_COUNT] = {NULL};
                    for (size_t j=0; j<scene::EVAI_COUNT; j++)
                    {
                        if (!oldbuffer[j])
                            continue;

                        inPtr[j] = (uint8_t*)oldbuffer[j]->getPointer();
                        inPtr[j] += attrOffset[j]+oldBaseVertex*attrStride[j];

                        vertexSize += scene::vertexAttrSize[componentTypes[j]][components[j]];
                    }

                    size_t vertexBufferSize = vertexSize*(indexRange+1);
                    void* mem = malloc(vertexBufferSize+indexBufferByteSize);
                    uint8_t* memPtr = (uint8_t*)mem;
                    for (uint8_t* memPtrLimit = memPtr+vertexBufferSize; memPtr<memPtrLimit; )
                    {
                        for (size_t j=0; j<scene::EVAI_COUNT; j++)
                        {
                            if (!oldbuffer[j])
                                continue;

                            switch (scene::vertexAttrSize[componentTypes[j]][components[j]])
                            {
                                case 1:
                                    ((uint8_t*)memPtr)[0] = ((uint8_t*)inPtr[j])[0];
                                    break;
                                case 2:
                                    ((uint16_t*)memPtr)[0] = ((uint16_t*)inPtr[j])[0];
                                    break;
                                case 3:
                                    ((uint16_t*)memPtr)[0] = ((uint16_t*)inPtr[j])[0];
                                    ((uint8_t*)memPtr)[2] = ((uint8_t*)inPtr[j])[2];
                                    break;
                                case 4:
                                    ((uint32_t*)memPtr)[0] = ((uint32_t*)inPtr[j])[0];
                                    break;
                                case 6:
                                    ((uint32_t*)memPtr)[0] = ((uint32_t*)inPtr[j])[0];
                                    ((uint16_t*)memPtr)[2] = ((uint16_t*)inPtr[j])[2];
                                    break;
                                case 8:
                                    ((uint64_t*)memPtr)[0] = ((uint64_t*)inPtr[j])[0];
                                    break;
                                case 12:
                                    ((uint64_t*)memPtr)[0] = ((uint64_t*)inPtr[j])[0];
                                    ((uint32_t*)memPtr)[2] = ((uint32_t*)inPtr[j])[2];
                                    break;
                                case 16:
                                    ((uint64_t*)memPtr)[0] = ((uint64_t*)inPtr[j])[0];
                                    ((uint64_t*)memPtr)[1] = ((uint64_t*)inPtr[j])[1];
                                    break;
                                case 24:
                                    ((uint64_t*)memPtr)[0] = ((uint64_t*)inPtr[j])[0];
                                    ((uint64_t*)memPtr)[1] = ((uint64_t*)inPtr[j])[1];
                                    ((uint64_t*)memPtr)[2] = ((uint64_t*)inPtr[j])[2];
                                    break;
                                case 32:
                                    ((uint64_t*)memPtr)[0] = ((uint64_t*)inPtr[j])[0];
                                    ((uint64_t*)memPtr)[1] = ((uint64_t*)inPtr[j])[1];
                                    ((uint64_t*)memPtr)[2] = ((uint64_t*)inPtr[j])[2];
                                    ((uint64_t*)memPtr)[3] = ((uint64_t*)inPtr[j])[3];
                                    break;
                            }
                            memPtr += scene::vertexAttrSize[componentTypes[j]][components[j]];

                            inPtr[j] += attrStride[j];
                        }
                    }
                    IGPUBuffer* vertexbuffer;
                    if (newIndexBuffer&&bufferOptions==EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER)
                    {
                        memcpy(memPtr,newIndexBuffer,indexBufferByteSize);
                        vertexbuffer = createGPUBuffer(vertexBufferSize+indexBufferByteSize,mem);
                    }
                    else
                        vertexbuffer = createGPUBuffer(vertexBufferSize,mem);
                    free(mem);

                    size_t offset = 0;
                    for (size_t j=0; j<scene::EVAI_COUNT; j++)
                    {
                        if (!oldbuffer[j])
                            continue;

                        desc->mapVertexAttrBuffer(vertexbuffer,(scene::E_VERTEX_ATTRIBUTE_ID)j,components[j],componentTypes[j],vertexSize,offset);
                        offset += scene::vertexAttrSize[componentTypes[j]][components[j]];
                    }
                    vertexbuffer->drop();

                    if (newIndexBuffer)
                    {
                        if (bufferOptions==EMDCB_INTERLEAVED_PACK_ALL_SINGLE_BUFFER)
                        {
                            desc->mapIndexBuffer(vertexbuffer);
                            meshbuffer->setIndexBufferOffset(vertexBufferSize);
                        }
                        else
                        {
                            IGPUBuffer* indexBuf = createGPUBuffer(indexBufferByteSize,newIndexBuffer);
                            desc->mapIndexBuffer(indexBuf);
                            indexBuf->drop();
                        }
                    }
                }
                break;
            default:
                os::Printer::log("THIS CPU to GPU Mesh CONVERSION NOT SUPPORTED YET!\n",ELL_ERROR);
                if (newIndexBuffer)
                    free(newIndexBuffer);
                meshbuffer->drop();
                outmesh->drop();
                return NULL;
                break;
        }

        if (newIndexBuffer)
            free(newIndexBuffer);

        switch (mesh->getMeshType())
        {
            case scene::EMT_ANIMATED_SKINNED:
                static_cast<scene::CGPUSkinnedMesh*>(outmesh)->addMeshBuffer(meshbuffer,static_cast<scene::SCPUSkinMeshBuffer*>(origmeshbuf)->getMaxVertexBoneInfluences());
                break;
            default:
                static_cast<scene::SGPUMesh*>(outmesh)->addMeshBuffer(meshbuffer);
                break;
        }
        meshbuffer->drop();
    }
    outmesh->recalculateBoundingBox();

    return outmesh;
}


IOcclusionQuery* COpenGLDriver::createOcclusionQuery(const E_OCCLUSION_QUERY_TYPE& heuristic)
{
    return new COpenGLOcclusionQuery(heuristic);
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

    COpenGLQuery* queryGL = dynamic_cast<COpenGLQuery*>(query);
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

    COpenGLQuery* queryGL = dynamic_cast<COpenGLQuery*>(query);
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

    COpenGLQuery* queryGL = dynamic_cast<COpenGLQuery*>(query);
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

    COpenGLQuery* queryGL = dynamic_cast<COpenGLQuery*>(query);
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



void COpenGLDriver::drawMeshBuffer(const scene::IGPUMeshBuffer* mb, IOcclusionQuery* query)
{
    if (mb && !mb->getInstanceCount())
        return;

    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

    const COpenGLVAOSpec* meshLayoutVAO = static_cast<const COpenGLVAOSpec*>(mb->getMeshDataAndFormat());
    if (!found->setActiveVAO(meshLayoutVAO,mb->isIndexCountGivenByXFormFeedback() ? mb:NULL))
        return;

#ifdef _DEBUG
	if (mb->getIndexCount() > getMaximalIndicesCount())
	{
		char tmp[1024];
		sprintf(tmp,"Could not draw, too many indices(%u), maxium is %u.", mb->getIndexCount(), getMaximalIndicesCount());
		os::Printer::log(tmp, ELL_ERROR);
	}
#endif // _DEBUG

	CNullDriver::drawMeshBuffer(mb,query);

	// draw everything
	setRenderStates3DMode();

    COpenGLOcclusionQuery* queryGL = (static_cast<COpenGLOcclusionQuery*>(query));

    bool didConditional = false;
    if (queryGL&&(queryGL->getGLHandle()!=0))
    {
        extGlBeginConditionalRender(queryGL->getGLHandle(),queryGL->getCondWaitModeGL());
        didConditional = true;
    }


	GLenum indexSize=0;
    if (meshLayoutVAO->getIndexBuffer())
    {
        switch (mb->getIndexType())
        {
            case EIT_16BIT:
            {
                indexSize=GL_UNSIGNED_SHORT;
                break;
            }
            case EIT_32BIT:
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
		case scene::EPT_POINTS:
		{
			// prepare size and attenuation (where supported)
			GLfloat particleSize=Material.Thickness;
			extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);
			glPointSize(particleSize);

		}
			break;
		case scene::EPT_TRIANGLES:
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
#ifdef _DEBUG
        if (xfmFb->isEnded())
            os::Printer::log("Trying To DrawTransformFeedback which hasn't ended yet (call glEndTransformFeedback() on the damn thing)!\n",ELL_ERROR);
        if (mb->getXFormFeedbackStream()>=MaxVertexStreams)
            os::Printer::log("Trying to use more than GL_MAX_VERTEX_STREAMS vertex streams in transform feedback!\n",ELL_ERROR);
#endif // _DEBUG
        extGlDrawTransformFeedbackStreamInstanced(primType,xfmFb->getOpenGLHandle(),mb->getXFormFeedbackStream(),mb->getInstanceCount());
    }
    else
        extGlDrawArraysInstancedBaseInstance(primType, mb->getBaseVertex(), mb->getIndexCount(), mb->getInstanceCount(), mb->getBaseInstance());

    if (didConditional)
        extGlEndConditionalRender();
}


//! Indirect Draw
void COpenGLDriver::drawArraysIndirect(  const scene::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                         const scene::E_PRIMITIVE_TYPE& mode,
                                         const IGPUBuffer* indirectDrawBuff,
                                         const size_t& offset, const size_t& count, const size_t& stride,
                                         IOcclusionQuery* query)
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

    COpenGLOcclusionQuery* queryGL = (static_cast<COpenGLOcclusionQuery*>(query));

    bool didConditional = false;
    if (queryGL&&(queryGL->getGLHandle()!=0)&&(!queryGL->isActive()))
    {
        extGlBeginConditionalRender(queryGL->getGLHandle(),queryGL->getCondWaitModeGL());
        didConditional = true;
    }

    GLenum primType = primitiveTypeToGL(mode);
	switch (mode)
	{
		case scene::EPT_POINTS:
		{
			// prepare size and attenuation (where supported)
			GLfloat particleSize=Material.Thickness;
			extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);
			glPointSize(particleSize);
		}
			break;
		case scene::EPT_TRIANGLES:
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


    if (didConditional)
        extGlEndConditionalRender();
}

void COpenGLDriver::drawIndexedIndirect(const scene::IMeshDataFormatDesc<video::IGPUBuffer>* vao,
                                        const scene::E_PRIMITIVE_TYPE& mode,
                                        const E_INDEX_TYPE& type, const IGPUBuffer* indirectDrawBuff,
                                        const size_t& offset, const size_t& count, const size_t& stride,
                                        IOcclusionQuery* query)
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

    COpenGLOcclusionQuery* queryGL = (static_cast<COpenGLOcclusionQuery*>(query));

    bool didConditional = false;
    if (queryGL&&(queryGL->getGLHandle()!=0)&&(!queryGL->isActive()))
    {
        extGlBeginConditionalRender(queryGL->getGLHandle(),queryGL->getCondWaitModeGL());
        didConditional = true;
    }

	GLenum indexSize = type!=EIT_16BIT ? GL_UNSIGNED_INT:GL_UNSIGNED_SHORT;

    GLenum primType = primitiveTypeToGL(mode);
	switch (mode)
	{
		case scene::EPT_POINTS:
		{
			// prepare size and attenuation (where supported)
			GLfloat particleSize=Material.Thickness;
			extGlPointParameterf(GL_POINT_FADE_THRESHOLD_SIZE, 1.0f);
			glPointSize(particleSize);
		}
			break;
		case scene::EPT_TRIANGLES:
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



    if (didConditional)
        extGlEndConditionalRender();
}

/*
template<GLenum bindType>
void COpenGLDriver::BoundIndexedBuffer::set(const uint32_t& first, const uint32_t& count, const COpenGLBuffer** buffers)
{
    GLuint toBind[BIND_POINTS];

    for (uint32_t i=0; i<count; i++)
    {
        uint32_t actualIx = i+first;

        if (!buffers || !buffers[i])
        {
            if (boundBuffer[actualIx])
            {
                boundBuffer[actualIx]->drop();
                boundBuffer[actualIx] = NULL;
            }

            if (buffers)
                toBind[i] = 0;
        }
        else
        {

        }
    }

    if (!buffers)
    {
        extGlBindBuffersBase(bindType,first,count,NULL);
        return;
    }

    //
}
*/

template<GLenum BIND_POINT>
void COpenGLDriver::SAuxContext::BoundBuffer<BIND_POINT>::set(const COpenGLBuffer* buff)
{
    if (!buff)
    {
        if (boundBuffer)
        {
            boundBuffer->drop();
            boundBuffer = NULL;
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

        extGlBindBuffer(BIND_POINT,boundBuffer->getOpenGLName());
        lastValidatedBuffer = boundBuffer->getLastTimeReallocated();
    }
    else if (lastValidatedBuffer>boundBuffer->getLastTimeReallocated())
    {
        extGlBindBuffer(BIND_POINT,boundBuffer->getOpenGLName());
        lastValidatedBuffer = boundBuffer->getLastTimeReallocated();
    }
}


COpenGLDriver::SAuxContext::COpenGLVAO::COpenGLVAO(const COpenGLVAOSpec* spec)
        : vao(0), lastValidated(0)
#ifdef _DEBUG
            ,debugHash(spec->getHash())
#endif // _DEBUG
{
    extGlCreateVertexArrays(1,&vao);

    memcpy(attrOffset,&spec->getMappedBufferOffset(scene::EVAI_ATTR0),sizeof(attrOffset));
    for (scene::E_VERTEX_ATTRIBUTE_ID attrId=scene::EVAI_ATTR0; attrId<scene::EVAI_COUNT; attrId = static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(attrId+1))
    {
        const IGPUBuffer* buf = spec->getMappedBuffer(attrId);
        mappedAttrBuf[attrId] = static_cast<const COpenGLBuffer*>(buf);
        if (mappedAttrBuf[attrId])
        {
            mappedAttrBuf[attrId]->grab();
            attrStride[attrId] = spec->getMappedBufferStride(attrId);

            extGlEnableVertexArrayAttrib(vao,attrId);
            extGlVertexArrayAttribBinding(vao,attrId,attrId);

            scene::E_COMPONENTS_PER_ATTRIBUTE components = spec->getAttribComponentCount(attrId);
            scene::E_COMPONENT_TYPE type = spec->getAttribType(attrId);
            switch (type)
            {
                case scene::ECT_FLOAT:
                case scene::ECT_HALF_FLOAT:
                case scene::ECT_DOUBLE_IN_FLOAT_OUT:
                case scene::ECT_UNSIGNED_INT_10F_11F_11F_REV:
                //INTEGER FORMS
                case scene::ECT_NORMALIZED_INT_2_10_10_10_REV:
                case scene::ECT_NORMALIZED_UNSIGNED_INT_2_10_10_10_REV:
                case scene::ECT_NORMALIZED_BYTE:
                case scene::ECT_NORMALIZED_UNSIGNED_BYTE:
                case scene::ECT_NORMALIZED_SHORT:
                case scene::ECT_NORMALIZED_UNSIGNED_SHORT:
                case scene::ECT_NORMALIZED_INT:
                case scene::ECT_NORMALIZED_UNSIGNED_INT:
                case scene::ECT_INT_2_10_10_10_REV:
                case scene::ECT_UNSIGNED_INT_2_10_10_10_REV:
                case scene::ECT_BYTE:
                case scene::ECT_UNSIGNED_BYTE:
                case scene::ECT_SHORT:
                case scene::ECT_UNSIGNED_SHORT:
                case scene::ECT_INT:
                case scene::ECT_UNSIGNED_INT:
                    extGlVertexArrayAttribFormat(vao,attrId,eComponentsPerAttributeToGLint[components],eComponentTypeToGLenum[type],scene::isNormalized(type) ? GL_TRUE:GL_FALSE,0);
                    break;
                case scene::ECT_INTEGER_INT_2_10_10_10_REV:
                case scene::ECT_INTEGER_UNSIGNED_INT_2_10_10_10_REV:
                case scene::ECT_INTEGER_BYTE:
                case scene::ECT_INTEGER_UNSIGNED_BYTE:
                case scene::ECT_INTEGER_SHORT:
                case scene::ECT_INTEGER_UNSIGNED_SHORT:
                case scene::ECT_INTEGER_INT:
                case scene::ECT_INTEGER_UNSIGNED_INT:
                    extGlVertexArrayAttribIFormat(vao,attrId,eComponentsPerAttributeToGLint[components],eComponentTypeToGLenum[type],0);
                    break;
            //special
                case scene::ECT_DOUBLE_IN_DOUBLE_OUT:
                    extGlVertexArrayAttribLFormat(vao,attrId,eComponentsPerAttributeToGLint[components],GL_DOUBLE,0);
                    break;
            }

            extGlVertexArrayBindingDivisor(vao,attrId,spec->getAttribDivisor(attrId));

            extGlVertexArrayVertexBuffer(vao,attrId,mappedAttrBuf[attrId]->getOpenGLName(),attrOffset[attrId],attrStride[attrId]);
        }
        else
        {
            mappedAttrBuf[attrId] = NULL;
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
    extGlDeleteVertexArrays(1,&vao);

    for (scene::E_VERTEX_ATTRIBUTE_ID attrId=scene::EVAI_ATTR0; attrId<scene::EVAI_COUNT; attrId = static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(attrId+1))
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
                                                            const size_t offsets[scene::EVAI_COUNT],
                                                            const size_t strides[scene::EVAI_COUNT])
{
    uint64_t beginStamp = CNullDriver::ReallocationCounter;

    for (scene::E_VERTEX_ATTRIBUTE_ID attrId=scene::EVAI_ATTR0; attrId<scene::EVAI_COUNT; attrId = static_cast<scene::E_VERTEX_ATTRIBUTE_ID>(attrId+1))
    {
#ifdef _DEBUG
        assert( (mappedAttrBuf[attrId]==NULL && attribBufs[attrId]==NULL)||
                (mappedAttrBuf[attrId]!=NULL && attribBufs[attrId]!=NULL));
#endif // _DEBUG
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

bool COpenGLDriver::SAuxContext::setActiveVAO(const COpenGLVAOSpec* spec, const scene::IGPUMeshBuffer* correctOffsetsForXFormDraw)
{
    if (!spec)
    {
        CurrentVAO = std::pair<COpenGLVAOSpec::HashAttribs,COpenGLVAO*>(COpenGLVAOSpec::HashAttribs(),NULL);
        extGlBindVertexArray(0);
        freeUpVAOCache(true);
        return false;
    }

    const COpenGLVAOSpec::HashAttribs& hashVal = spec->getHash();
	if (CurrentVAO.first!=hashVal)
    {
        std::unordered_map<COpenGLVAOSpec::HashAttribs,COpenGLVAO*>::iterator it = VAOMap.find(hashVal);
        if (it != VAOMap.end())
            CurrentVAO = *it;
        else
        {
            COpenGLVAO* vao = new COpenGLVAO(spec);
            VAOMap[hashVal] = vao;
            CurrentVAO = std::pair<COpenGLVAOSpec::HashAttribs,COpenGLVAO*>(hashVal,vao);
        }

        #ifdef _DEBUG
            assert(!(CurrentVAO.second->getDebugHash()!=hashVal));
        #endif // _DEBUG

        extGlBindVertexArray(CurrentVAO.second->getOpenGLName());
    }

    if (correctOffsetsForXFormDraw)
    {
        size_t offsets[scene::EVAI_COUNT] = {0};
        memcpy(offsets,&spec->getMappedBufferOffset(scene::EVAI_ATTR0),sizeof(offsets));
        for (size_t i=0; i<scene::EVAI_COUNT; i++)
        {
            if (!spec->getMappedBuffer((scene::E_VERTEX_ATTRIBUTE_ID)i))
                continue;

            if (spec->getAttribDivisor((scene::E_VERTEX_ATTRIBUTE_ID)i))
            {
                if (correctOffsetsForXFormDraw->getBaseInstance())
                    offsets[i] += spec->getMappedBufferStride((scene::E_VERTEX_ATTRIBUTE_ID)i)*correctOffsetsForXFormDraw->getBaseInstance();
            }
            else
            {
                if (correctOffsetsForXFormDraw->getBaseVertex())
                    offsets[i] = int64_t(offsets[i])+int64_t(spec->getMappedBufferStride((scene::E_VERTEX_ATTRIBUTE_ID)i))*correctOffsetsForXFormDraw->getBaseVertex();
            }
        }
        CurrentVAO.second->bindBuffers(static_cast<const COpenGLBuffer*>(spec->getIndexBuffer()),reinterpret_cast<const COpenGLBuffer* const*>(spec->getMappedBuffers()),offsets,&spec->getMappedBufferStride(scene::EVAI_ATTR0));
    }
    else
        CurrentVAO.second->bindBuffers(static_cast<const COpenGLBuffer*>(spec->getIndexBuffer()),reinterpret_cast<const COpenGLBuffer* const*>(spec->getMappedBuffers()),&spec->getMappedBufferOffset(scene::EVAI_ATTR0),&spec->getMappedBufferStride(scene::EVAI_ATTR0));

    return true;
}

//! Get native wrap mode value
inline GLint getTextureWrapMode(const uint8_t &clamp)
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
                std::unordered_map<uint64_t,GLuint>::iterator it = SamplerMap.find(hashVal);
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


bool orderByMip(CImageData* a, CImageData* b)
{
    return a->getSupposedMipLevel() < b->getSupposedMipLevel();
}


//! returns a device dependent texture from a software surface (IImage)
video::ITexture* COpenGLDriver::createDeviceDependentTexture(const ITexture::E_TEXTURE_TYPE& type, const uint32_t* size, uint32_t mipmapLevels,
			const io::path& name, ECOLOR_FORMAT format)
{
#ifdef _DEBUG
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
#endif // _DEBUG
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
        ///case ITexture::ETT_1D:
            ///break;
        case ITexture::ETT_2D:
            return new COpenGL2DTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format), size, mipmapLevels, name);
            break;
        case ITexture::ETT_3D:
            return new COpenGL3DTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        ///case ITexture::ETT_1D_ARRAY:
            ///break;
        case ITexture::ETT_2D_ARRAY:
            return new COpenGL2DTextureArray(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        case ITexture::ETT_CUBE_MAP:
            return new COpenGLCubemapTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            break;
        ///case ITexture::ETT_CUBE_MAP_ARRAY:
            ///return new COpenGLCubemapArrayTexture(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,mipmapLevels,name);
            ///break;
        default:// ETT_CUBE_MAP, ETT_CUBE_MAP_ARRAY, ETT_TEXTURE_BUFFER
            break;
    }

    return NULL;
}


//! Sets a material. All 3d drawing functions draw geometry now using this material.
void COpenGLDriver::setMaterial(const SMaterial& material)
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;


	Material = material;
	OverrideMaterial.apply(Material);

	for (int32_t i = MaxTextureUnits-1; i>= 0; --i)
	{
		found->setActiveTexture(i, material.getTexture(i), material.TextureLayer[i].SamplingParams);
	}
}


//! prints error if an error happened.
bool COpenGLDriver::testGLError()
{
#ifdef _DEBUG
	GLenum g = glGetError();
	switch (g)
	{
	case GL_NO_ERROR:
		return false;
	case GL_INVALID_ENUM:
		os::Printer::log("GL_INVALID_ENUM", ELL_ERROR); break;
	case GL_INVALID_VALUE:
		os::Printer::log("GL_INVALID_VALUE", ELL_ERROR); break;
	case GL_INVALID_OPERATION:
		os::Printer::log("GL_INVALID_OPERATION", ELL_ERROR); break;
	case GL_STACK_OVERFLOW:
		os::Printer::log("GL_STACK_OVERFLOW", ELL_ERROR); break;
	case GL_STACK_UNDERFLOW:
		os::Printer::log("GL_STACK_UNDERFLOW", ELL_ERROR); break;
	case GL_OUT_OF_MEMORY:
		os::Printer::log("GL_OUT_OF_MEMORY", ELL_ERROR); break;
	case GL_TABLE_TOO_LARGE:
		os::Printer::log("GL_TABLE_TOO_LARGE", ELL_ERROR); break;
	case GL_INVALID_FRAMEBUFFER_OPERATION_EXT:
		os::Printer::log("GL_INVALID_FRAMEBUFFER_OPERATION", ELL_ERROR); break;
	};
//	_IRR_DEBUG_BREAK_IF(true);
	return true;
#else
	return false;
#endif
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
void COpenGLDriver::setBasicRenderStates(const SMaterial& material, const SMaterial& lastmaterial,
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

ITexture* COpenGLDriver::addTexture(const ITexture::E_TEXTURE_TYPE& type, const std::vector<CImageData*>& images, const io::path& name, ECOLOR_FORMAT format)
{
    if (!images.size())
        return NULL;

    //validate a bit
    uint32_t initialMaxCoord[3] = {1,1,1};
    uint32_t highestMip = 0;
    ECOLOR_FORMAT candidateFormat = format;
    for (std::vector<CImageData*>::const_iterator it=images.begin(); it!=images.end(); it++)
    {
        CImageData* img = *it;
        if (!img||img->getColorFormat()==ECF_UNKNOWN)
        {
#ifdef _DEBUG
            os::Printer::log("Very invalid mip-chain!", ELL_ERROR);
#endif // _DEBUG
            return NULL;
        }

        for (size_t i=0; i<3; i++)
        {
            const uint32_t& sideSize = img->getSliceMax()[i];
            if (initialMaxCoord[i] < sideSize)
                initialMaxCoord[i] = sideSize;
        }
        if (highestMip < img->getSupposedMipLevel())
            highestMip = img->getSupposedMipLevel();

        //figure out the format
        if (format==ECF_UNKNOWN)
        {
            if (candidateFormat==ECF_UNKNOWN)
                candidateFormat = img->getColorFormat();
            else if (candidateFormat!=img->getColorFormat())
            {
#ifdef _DEBUG
                os::Printer::log("Can't pick a default texture format if the mip-chain doesn't have a consistent one!", ELL_ERROR);
#endif // _DEBUG
                return NULL;
            }
        }
    }
    //haven't figured best format out
    if (format==ECF_UNKNOWN)
    {
        if (candidateFormat==ECF_UNKNOWN)
        {
    #ifdef _DEBUG
            os::Printer::log("Couldn't pick a texture format, entire mip-chain doesn't know!", ELL_ERROR);
    #endif // _DEBUG
            return NULL;
        }
        //else
            //candidateFormat = candidateFormat;
    }

    //! Sort the mipchain!!!
    std::vector<CImageData*> sortedMipchain(images);
    std::sort(sortedMipchain.begin(),sortedMipchain.end(),orderByMip);

    //figure out the texture type if not provided
    ITexture::E_TEXTURE_TYPE actualType = type;
    if (type>=ITexture::ETT_COUNT)
    {
        if (initialMaxCoord[2]>1)
        {
            //! with this little info I literally can't guess if you want a cubemap!
            if (sortedMipchain.size()>1&&sortedMipchain.front()->getSliceMax()[2]==sortedMipchain.back()->getSliceMax()[2])
                actualType = ITexture::ETT_2D_ARRAY;
            else
                actualType = ITexture::ETT_3D;
        }
        else if (initialMaxCoord[1]>1)
        {
            if (sortedMipchain.size()>1&&sortedMipchain.front()->getSliceMax()[1]==sortedMipchain.back()->getSliceMax()[1])
                actualType = ITexture::ETT_1D_ARRAY;
            else
                actualType = ITexture::ETT_2D;
        }
        else
        {
            actualType = ITexture::ETT_2D; //should be ETT_1D but 2D is default since forever
        }
    }

    //get out max texture size
    uint32_t maxCoord[3] = {initialMaxCoord[0],initialMaxCoord[1],initialMaxCoord[2]};
    for (std::vector<CImageData*>::const_iterator it=sortedMipchain.begin(); it!=sortedMipchain.end(); it++)
    {
        CImageData* img = *it;
        if (img->getSliceMax()[0]>getMaxTextureSize(actualType)[0]||
            (actualType==ITexture::ETT_2D||actualType==ITexture::ETT_1D_ARRAY||actualType==ITexture::ETT_CUBE_MAP)
                &&img->getSliceMax()[1]>getMaxTextureSize(actualType)[1]||
            (actualType==ITexture::ETT_3D||actualType==ITexture::ETT_2D_ARRAY||actualType==ITexture::ETT_CUBE_MAP_ARRAY)
                &&img->getSliceMax()[2]>getMaxTextureSize(actualType)[2])
        {
#ifdef _DEBUG
            os::Printer::log("Attemped to create a larger texture than supported (we should implement mip-chain dropping)!", ELL_ERROR);
#endif // _DEBUG
            return NULL;
        }
    }

    video::ITexture* texture = createDeviceDependentTexture(actualType,maxCoord,highestMip ? (highestMip+1):0,name,candidateFormat);
	addToTextureCache(texture);
	if (texture)
		texture->drop();

    for (std::vector<CImageData*>::const_iterator it=sortedMipchain.begin(); it!=sortedMipchain.end(); it++)
    {
        CImageData* img = *it;
        if (!img)
            continue;

        texture->updateSubRegion(img->getColorFormat(),img->getData(),img->getSliceMin(),img->getSliceMax(),img->getSupposedMipLevel(),img->getUnpackAlignment());
    }

    //has mipmap but no explicit chain
    if (highestMip==0&&texture->hasMipMaps())
        texture->regenerateMipMapLevels();

    return texture;
}

IMultisampleTexture* COpenGLDriver::addMultisampleTexture(const IMultisampleTexture::E_MULTISAMPLE_TEXTURE_TYPE& type, const uint32_t& samples, const uint32_t* size, ECOLOR_FORMAT format, const bool& fixedSampleLocations)
{
    //check to implement later on renderbuffer creation and attachment of textures to FBO
    //if (!isFormatRenderable(glTex->getOpenGLInternalFormat()))
        //return NULL;

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
            tex = NULL;
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
        return NULL;

    ITextureBufferObject* tbo = new COpenGLTextureBufferObject(buffer,format,offset,length);
	CNullDriver::addTextureBufferObject(tbo);
    return tbo;
}

IRenderBuffer* COpenGLDriver::addRenderBuffer(const core::dimension2d<uint32_t>& size, const ECOLOR_FORMAT format)
{
	IRenderBuffer* buffer = new COpenGLRenderBuffer(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size);
	CNullDriver::addRenderBuffer(buffer);
	return buffer;
}

IRenderBuffer* COpenGLDriver::addMultisampleRenderBuffer(const uint32_t& samples, const core::dimension2d<uint32_t>& size, const ECOLOR_FORMAT format)
{
	IRenderBuffer* buffer = new COpenGLMultisampleRenderBuffer(COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(format),size,samples);
	CNullDriver::addRenderBuffer(buffer);
	return buffer;
}

IFrameBuffer* COpenGLDriver::addFrameBuffer()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return NULL;

	IFrameBuffer* fbo = new COpenGLFrameBuffer(this);
    found->FrameBuffers.push_back(fbo);
    found->FrameBuffers.sort();
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

    int32_t ix = found->FrameBuffers.binary_search(framebuf);
    if (ix<0)
        return;
    found->FrameBuffers.erase(ix);

    framebuf->drop();
}

void COpenGLDriver::removeAllFrameBuffers()
{
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

	for (uint32_t i=0; i<found->FrameBuffers.size(); ++i)
		found->FrameBuffers[i]->drop();
    found->FrameBuffers.clear();
}


//! Removes a texture from the texture cache and deletes it, freeing lot of memory.
void COpenGLDriver::removeTexture(ITexture* texture)
{
	if (!texture)
		return;

	CNullDriver::removeTexture(texture);
	// Remove this texture from CurrentTexture as well
    SAuxContext* found = getThreadContext_helper(false);
    if (!found)
        return;

	found->CurrentTexture.remove(texture);
}

//! Only used by the internal engine. Used to notify the driver that
//! the window was resized.
void COpenGLDriver::OnResize(const core::dimension2d<uint32_t>& size)
{
	CNullDriver::OnResize(size);
	glViewport(0, 0, size.Width, size.Height);
}


//! Returns type of video driver
E_DRIVER_TYPE COpenGLDriver::getDriverType() const
{
	return EDT_OPENGL;
}


//! returns color format
ECOLOR_FORMAT COpenGLDriver::getColorFormat() const
{
	return ColorFormat;
}


void COpenGLDriver::setShaderConstant(const void* data, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number)
{
	os::Printer::log("Error: Please call services->setShaderConstant(), not VideoDriver->setShaderConstant().");
}

void COpenGLDriver::setShaderTextures(const int32_t* textureIndices, int32_t location, E_SHADER_CONSTANT_TYPE type, uint32_t number)
{
	os::Printer::log("Error: Please call services->setShaderTextures(), not VideoDriver->setShaderTextures().");
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
                const IRenderable* rndrbl = in->getAttachment(i);
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
                const IRenderable* rndrbl = out->getAttachment(i);
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
        const IRenderable* attachment = frameBuffer->getAttachment(i);
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
#ifdef _DEBUG
        if (!toContext->CurrentXFormFeedback->isEnded())
            os::Printer::log("FIDDLING WITH XFORM FEEDBACK BINDINGS WHILE THE BOUND XFORMFEEDBACK HASN't ENDED!\n",ELL_ERROR);
#endif // _DEBUG
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
#ifdef _DEBUG
        if (!toContext->CurrentXFormFeedback->isEnded())
            os::Printer::log("WHY IS A NOT PREVIOUSLY BOUND XFORM FEEDBACK STARTED!?\n",ELL_ERROR);
#endif // _DEBUG
        toContext->CurrentXFormFeedback->grab();
        extGlBindTransformFeedback(GL_TRANSFORM_FEEDBACK,toContext->CurrentXFormFeedback->getOpenGLHandle());
    }
}

void COpenGLDriver::beginTransformFeedback(ITransformFeedback* xformFeedback, const E_MATERIAL_TYPE& xformFeedbackShader, const scene::E_PRIMITIVE_TYPE& primType)
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
		case scene::EPT_POINTS:
            found->CurrentXFormFeedback->setPrimitiveType(GL_POINTS);
            break;
		case scene::EPT_LINE_STRIP:
		case scene::EPT_LINE_LOOP:
			os::Printer::log("Not using PROPER TRANSFORM FEEDBACK primitive type (only EPT_POINTS, EPT_LINES and EPT_TRIANGLES allowed!)!\n",ELL_ERROR);
		case scene::EPT_LINES:
            found->CurrentXFormFeedback->setPrimitiveType(GL_LINES);
            break;
		case scene::EPT_TRIANGLE_STRIP:
		case scene::EPT_TRIANGLE_FAN:
			os::Printer::log("Not using PROPER TRANSFORM FEEDBACK primitive type (only EPT_POINTS, EPT_LINES and EPT_TRIANGLES allowed!)!\n",ELL_ERROR);
		case scene::EPT_TRIANGLES:
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
#ifdef _DEBUG
    if (!found->CurrentXFormFeedback->isActive())
        os::Printer::log("Ending an already paused transform feedback, the pause call is redundant!\n",ELL_ERROR);
#endif // _DEBUG
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



//! Convert E_PRIMITIVE_TYPE to OpenGL equivalent
GLenum COpenGLDriver::primitiveTypeToGL(scene::E_PRIMITIVE_TYPE type) const
{
	switch (type)
	{
		case scene::EPT_POINTS:
			return GL_POINTS;
		case scene::EPT_LINE_STRIP:
			return GL_LINE_STRIP;
		case scene::EPT_LINE_LOOP:
			return GL_LINE_LOOP;
		case scene::EPT_LINES:
			return GL_LINES;
		case scene::EPT_TRIANGLE_STRIP:
			return GL_TRIANGLE_STRIP;
		case scene::EPT_TRIANGLE_FAN:
			return GL_TRIANGLE_FAN;
		case scene::EPT_TRIANGLES:
			return GL_TRIANGLES;
	}
	return GL_TRIANGLES;
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

GLenum COpenGLDriver::getZBufferBits() const
{
	GLenum bits = 0;
	switch (Params.ZBufferBits)
	{
	case 16:
		bits = GL_DEPTH_COMPONENT16;
		break;
	case 24:
		bits = GL_DEPTH_COMPONENT24;
		break;
	case 32:
		bits = GL_DEPTH_COMPONENT32;
		break;
	default:
		bits = GL_DEPTH_COMPONENT;
		break;
	}
	return bits;
}


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



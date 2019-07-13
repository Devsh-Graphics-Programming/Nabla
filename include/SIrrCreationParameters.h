// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_IRRLICHT_CREATION_PARAMETERS_H_INCLUDED__
#define __I_IRRLICHT_CREATION_PARAMETERS_H_INCLUDED__

#include "EDriverTypes.h"
#include "EDeviceTypes.h"
#include "dimension2d.h"
#include "ILogger.h"

namespace irr
{
	class IEventReceiver;

	//! Structure for holding Irrlicht Device creation parameters.
	/** This structure is used in the createDeviceEx() function. */
	struct SIrrlichtCreationParameters
	{
		//! Constructs a SIrrlichtCreationParameters structure with default values.
		SIrrlichtCreationParameters() :
			DeviceType(EIDT_BEST),
			DriverType(video::EDT_BURNINGSVIDEO),
			WindowSize(core::dimension2d<uint32_t>(800, 600)),
			Bits(16),
			ZBufferBits(16),
			Fullscreen(false),
			Stencilbuffer(false),
			Vsync(false),
			WithAlphaChannel(false),
			Doublebuffer(true),
			IgnoreInput(false),
			Stereobuffer(false),
			StreamingDownloadBufferSize(0x4000000u), // 64MB should be enough to stream one 4K image in 64bit HDR without breaking it into chunks
			StreamingUploadBufferSize(0x4000000u), // 64MB should be enough to stream one 4K image in 64bit HDR without breaking it into chunks
			EventReceiver(0),
			WindowId(0),
#ifdef _IRR_DEBUG
			LoggingLevel(ELL_DEBUG),
#else
			LoggingLevel(ELL_INFORMATION),
#endif
			AuxGLContexts(0),
			SDK_version_do_not_use(IRRLICHTBAW_SDK_VERSION)
		{
		}

		SIrrlichtCreationParameters(const SIrrlichtCreationParameters& other) :
			SDK_version_do_not_use(IRRLICHTBAW_SDK_VERSION)
		{*this = other;}

		SIrrlichtCreationParameters& operator=(const SIrrlichtCreationParameters& other)
		{
			DeviceType = other.DeviceType;
			DriverType = other.DriverType;
			WindowSize = other.WindowSize;
			Bits = other.Bits;
			ZBufferBits = other.ZBufferBits;
			Fullscreen = other.Fullscreen;
			Stencilbuffer = other.Stencilbuffer;
			Vsync = other.Vsync;
			WithAlphaChannel = other.WithAlphaChannel;
			Doublebuffer = other.Doublebuffer;
			IgnoreInput = other.IgnoreInput;
			Stereobuffer = other.Stereobuffer;
			StreamingDownloadBufferSize = other.StreamingDownloadBufferSize;
			StreamingUploadBufferSize = other.StreamingUploadBufferSize;
			EventReceiver = other.EventReceiver;
			WindowId = other.WindowId;
			LoggingLevel = other.LoggingLevel;
			AuxGLContexts = other.AuxGLContexts;
			return *this;
		}

		//! Type of the device.
		/** This setting decides the windowing system used by the device, most device types are native
		to a specific operating system and so may not be available.
		EIDT_WIN32 is only available on Windows desktops,
		EIDT_WINCE is only available on Windows mobile devices,
		EIDT_COCOA is only available on Mac OSX,
		EIDT_X11 is available on Linux, Solaris, BSD and other operating systems which use X11,
		EIDT_SDL is available on most systems if compiled in,
		EIDT_CONSOLE is usually available but can only render to text,
		EIDT_BEST will select the best available device for your operating system.
		Default: EIDT_BEST. */
		E_DEVICE_TYPE DeviceType;

		//! Type of video driver used to render graphics.
		/** This can currently be video::EDT_NULL, video::EDT_BURNINGSVIDEO, and video::EDT_OPENGL.
		Default: Burnings. */
		video::E_DRIVER_TYPE DriverType;

		//! Size of the window or the video mode in fullscreen mode. Default: 800x600
		core::dimension2d<uint32_t> WindowSize;

		//! Minimum Bits per pixel of the color buffer in fullscreen mode. Ignored if windowed mode. Default: 16.
		uint8_t Bits;

		//! Minimum Bits per pixel of the depth buffer. Default: 16.
		uint8_t ZBufferBits;

		//! Should be set to true if the device should run in fullscreen.
		/** Otherwise the device runs in windowed mode. Default: false. */
		bool Fullscreen;

		//! Specifies if the stencil buffer should be enabled.
		/** Set this to true, if you want the engine be able to draw
		stencil buffer shadows. Note that not all drivers are able to
		use the stencil buffer, hence it can be ignored during device
		creation. Without the stencil buffer no shadows will be drawn.
		Default: false. */
		bool Stencilbuffer;

		//! Specifies vertical syncronisation.
		/** If set to true, the driver will wait for the vertical
		retrace period, otherwise not. May be silently ignored.
		Default: false */
		bool Vsync;

		//! Whether the main framebuffer uses an alpha channel.
		/** In some situations it might be desireable to get a color
		buffer with an alpha channel, e.g. when rendering into a
		transparent window or overlay. If this flag is set the device
		tries to create a framebuffer with alpha channel.
		If this flag is set, only color buffers with alpha channel
		are considered. Otherwise, it depends on the actual hardware
		if the colorbuffer has an alpha channel or not.
		Default value: false */
		bool WithAlphaChannel;

		//! Whether the main framebuffer uses doublebuffering.
		/** This should be usually enabled, in order to avoid render
		artifacts on the visible framebuffer. However, it might be
		useful to use only one buffer on very small devices. If no
		doublebuffering is available, the drivers will fall back to
		single buffers. Default value: true */
		bool Doublebuffer;

		//! Specifies if the device should ignore input events
		/** This is only relevant when using external I/O handlers.
		External windows need to take care of this themselves.
		Currently only supported by X11.
		Default value: false */
		bool IgnoreInput;

		//! Specifies if the device should use stereo buffers
		/** Some high-end gfx cards support two framebuffers for direct
		support of stereoscopic output devices. If this flag is set the
		device tries to create a stereo context.
		Currently only supported by OpenGL.
		Default value: false */
		bool Stereobuffer;

		//! Default download buffer size
		uint32_t StreamingDownloadBufferSize;

		//! Default upload buffer size
		uint32_t StreamingUploadBufferSize;

		//! A user created event receiver.
		IEventReceiver* EventReceiver;

		//! Window Id.
		/** If this is set to a value other than 0, the Irrlicht Engine
		will be created in an already existing window. For windows, set
		this to the HWND of the window you want. The windowSize and
		FullScreen options will be ignored when using the WindowId
		parameter. Default this is set to 0.
		To make Irrlicht run inside the custom window, you still will
		have to draw Irrlicht on your own. You can use this loop, as
		usual:
		\code
		while (device->run())
		{
			driver->beginScene(true, true, 0);
			smgr->drawAll();
			driver->endScene();
		}
		\endcode
		Instead of this, you can also simply use your own message loop
		using GetMessage, DispatchMessage and whatever. Calling
		IrrlichtDevice::run() will cause Irrlicht to dispatch messages
		internally too.  You need not call Device->run() if you want to
		do your own message dispatching loop, but Irrlicht will not be
		able to fetch user input then and you have to do it on your own
		using the window messages, DirectInput, or whatever. Also,
		you'll have to increment the Irrlicht timer.
		An alternative, own message dispatching loop without
		device->run() would look like this:
		\code
		MSG msg;
		while (true)
		{
			if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);

				if (msg.message == WM_QUIT)
					break;
			}

			// increase virtual timer time
			device->getTimer()->tick();

			// draw engine picture
			driver->beginScene(true, true, 0);
			smgr->drawAll();
			driver->endScene();
		}
		\endcode
		However, there is no need to draw the picture this often. Just
		do it how you like. */
		void* WindowId;

		//! Specifies the logging level used in the logging interface.
		/** The default value is ELL_INFORMATION. You can access the ILogger interface
		later on from the IrrlichtDevice with getLogger() and set another level.
		But if you need more or less logging information already from device creation,
		then you have to change it here.
		*/
		ELOG_LEVEL LoggingLevel;

		//!
		uint8_t AuxGLContexts;

		//! Don't use or change this parameter.
		/** Always set it to IRRLICHTBAW_SDK_VERSION, which is done by default.
		This is needed for sdk version checks. */
		const char* const SDK_version_do_not_use;
	};


} // end namespace irr

#endif


// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "CIrrDeviceConsole.h"
#include "CSceneManager.h"
#include <sstream>

#include "os.h"

// to close the device on terminate signal
irr::CIrrDeviceConsole *DeviceToClose;

#ifdef _IRR_WINDOWS_NT_CONSOLE_
// Callback for Windows
BOOL WINAPI ConsoleHandler(DWORD CEvent)
{
    switch(CEvent)
    {
    case CTRL_C_EVENT:
		irr::os::Printer::log("Closing console device", "CTRL+C");
		break;
	case CTRL_BREAK_EVENT:
		irr::os::Printer::log("Closing console device", "CTRL+Break");
		break;
    case CTRL_CLOSE_EVENT:
		irr::os::Printer::log("Closing console device", "User closed console");
		break;
    case CTRL_LOGOFF_EVENT:
		irr::os::Printer::log("Closing console device", "User is logging off");
		break;
    case CTRL_SHUTDOWN_EVENT:
		irr::os::Printer::log("Closing console device", "Computer shutting down");
		break;
    }
	DeviceToClose->closeDevice();
    return TRUE;
}
#elif defined(_IRR_POSIX_API_)
// sigterm handler
#include <signal.h>

void sighandler(int sig)
{
    std::ostringstream code("Signal ");
    code.seekp(0,std::ios_base::end);
	code << sig << " received";
	irr::os::Printer::log("Closing console device", code.str().c_str());

	DeviceToClose->closeDevice();
}
#endif

namespace irr
{

const int8_t ASCIIArtChars[] = " .,'~:;!+>=icopjtJY56SB8XDQKHNWM"; //MWNHKQDX8BS65YJtjpoci=+>!;:~',. ";
const uint16_t ASCIIArtCharsCount = 32;

//const int8_t ASCIIArtChars[] = " \xb0\xb1\xf9\xb2\xdb";
//const uint16_t ASCIIArtCharsCount = 5;

//! constructor
CIrrDeviceConsole::CIrrDeviceConsole(const SIrrlichtCreationParameters& params)
  : CIrrDeviceStub(params), IsWindowFocused(true), OutFile(stdout)
{
	DeviceToClose = this;

#ifdef _IRR_WINDOWS_NT_CONSOLE_
	MouseButtonStates = 0;

	WindowsSTDIn  = GetStdHandle(STD_INPUT_HANDLE);
	WindowsSTDOut = GetStdHandle(STD_OUTPUT_HANDLE);
	PCOORD Dimensions = 0;

	if (CreationParams.Fullscreen)
	{
// Some mingw versions lack this define, so avoid it in case it does not exist
#if (_WIN32_WINNT >= 0x0501) && defined(CONSOLE_FULLSCREEN_MODE)
		if (SetConsoleDisplayMode(WindowsSTDOut, CONSOLE_FULLSCREEN_MODE, Dimensions))
		{
			CreationParams.WindowSize.Width = Dimensions->X;
			CreationParams.WindowSize.Width = Dimensions->Y;
		}
#endif
	}
	else
	{
		COORD ConsoleSize;
		ConsoleSize.X = CreationParams.WindowSize.Width;
		ConsoleSize.X = CreationParams.WindowSize.Height;
		SetConsoleScreenBufferSize(WindowsSTDOut, ConsoleSize);
	}

	// catch windows close/break signals
	SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE);

#elif defined(_IRR_POSIX_API_)
	// catch other signals
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT,  &sighandler);

	// set output stream
	if (params.WindowId)
		OutFile = (FILE*)(params.WindowId);
#endif

#ifdef _IRR_VT100_CONSOLE_
	// reset terminal
	fprintf(OutFile, "%cc", 27);
	// disable line wrapping
	fprintf(OutFile, "%c[7l", 27);
#endif

	switch (params.DriverType)
	{

	case video::EDT_BURNINGSVIDEO:
		#ifdef _IRR_COMPILE_WITH_BURNINGSVIDEO_
		VideoDriver = video::createBurningVideoDriver(this, CreationParams, FileSystem, this);
		#else
		os::Printer::log("Burning's Video driver was not compiled in.", ELL_ERROR);
		#endif
		break;
	case video::EDT_OPENGL:
		os::Printer::log("The console device cannot use hardware drivers yet.", ELL_ERROR);
		break;
	case video::EDT_NULL:
		VideoDriver = video::createNullDriver(this, FileSystem, CreationParams.WindowSize);
		break;
	default:
		break;
	}

	// set up output buffer
	for (uint32_t y=0; y<CreationParams.WindowSize.Height; ++y)
	{
		core::stringc str;
		str.reserve(CreationParams.WindowSize.Width);
		for (uint32_t x=0; x<CreationParams.WindowSize.Width; ++x)
			str += " ";
		OutputBuffer.push_back(str);
	}


#ifdef _IRR_WINDOWS_NT_CONSOLE_
	CursorControl = new CCursorControl(CreationParams.WindowSize);
#endif

	if (VideoDriver)
	{
		createGUIAndScene();
	}
}

//! destructor
CIrrDeviceConsole::~CIrrDeviceConsole()
{
	if (SceneManager)
		SceneManager->drop();

	if (InputReceivingSceneManager)
		InputReceivingSceneManager->drop();

	if (CursorControl)
		CursorControl->drop();

	if (VideoDriver)
		VideoDriver->drop();

#ifdef _IRR_VT100_CONSOLE_
	// reset terminal
	fprintf(OutFile, "%cc", 27);
#endif
}

//! runs the device. Returns false if device wants to be deleted
bool CIrrDeviceConsole::run()
{
	// increment timer
	Timer->tick();

	// process Windows console input
#ifdef _IRR_WINDOWS_NT_CONSOLE_

	INPUT_RECORD in;
	DWORD        oldMode;
	DWORD        count, waste;

	// get old input mode
	GetConsoleMode(WindowsSTDIn, &oldMode);
	SetConsoleMode(WindowsSTDIn, ENABLE_WINDOW_INPUT | ENABLE_MOUSE_INPUT);

	GetNumberOfConsoleInputEvents(WindowsSTDIn, &count);

	// read keyboard and mouse input
	while (count)
	{
		ReadConsoleInput(WindowsSTDIn, &in, 1, &waste );
		switch(in.EventType)
		{
		case KEY_EVENT:
		{
			SEvent e;
			e.EventType            = EET_KEY_INPUT_EVENT;
			e.KeyInput.PressedDown = (in.Event.KeyEvent.bKeyDown == TRUE);
			e.KeyInput.Control     = (in.Event.KeyEvent.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0;
			e.KeyInput.Shift       = (in.Event.KeyEvent.dwControlKeyState & SHIFT_PRESSED) != 0;
			e.KeyInput.Key         = EKEY_CODE(in.Event.KeyEvent.wVirtualKeyCode);
			e.KeyInput.Char        = in.Event.KeyEvent.uChar.UnicodeChar;
			postEventFromUser(e);
			break;
		}
		case MOUSE_EVENT:
		{
			SEvent e;
			e.EventType        = EET_MOUSE_INPUT_EVENT;
			e.MouseInput.X     = in.Event.MouseEvent.dwMousePosition.X;
			e.MouseInput.Y     = in.Event.MouseEvent.dwMousePosition.Y;
			e.MouseInput.Wheel = 0.f;
			e.MouseInput.ButtonStates =
				( (in.Event.MouseEvent.dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED) ? EMBSM_LEFT   : 0 ) |
				( (in.Event.MouseEvent.dwButtonState & RIGHTMOST_BUTTON_PRESSED)     ? EMBSM_RIGHT  : 0 ) |
				( (in.Event.MouseEvent.dwButtonState & FROM_LEFT_2ND_BUTTON_PRESSED) ? EMBSM_MIDDLE : 0 ) |
				( (in.Event.MouseEvent.dwButtonState & FROM_LEFT_3RD_BUTTON_PRESSED) ? EMBSM_EXTRA1 : 0 ) |
				( (in.Event.MouseEvent.dwButtonState & FROM_LEFT_4TH_BUTTON_PRESSED) ? EMBSM_EXTRA2 : 0 );

			if (in.Event.MouseEvent.dwEventFlags & MOUSE_MOVED)
			{
				CursorControl->setPosition(core::position2di(e.MouseInput.X, e.MouseInput.Y));

				// create mouse moved event
				e.MouseInput.Event = EMIE_MOUSE_MOVED;
				postEventFromUser(e);
			}

			if (in.Event.MouseEvent.dwEventFlags & MOUSE_WHEELED)
			{
				e.MouseInput.Event = EMIE_MOUSE_WHEEL;
				e.MouseInput.Wheel = (in.Event.MouseEvent.dwButtonState & 0xFF000000) ? -1.0f : 1.0f;
				postEventFromUser(e);
			}

			if ( (MouseButtonStates & EMBSM_LEFT) != (e.MouseInput.ButtonStates & EMBSM_LEFT) )
			{
				e.MouseInput.Event = (e.MouseInput.ButtonStates & EMBSM_LEFT) ? EMIE_LMOUSE_PRESSED_DOWN : EMIE_LMOUSE_LEFT_UP;
				postEventFromUser(e);
			}

			if ( (MouseButtonStates & EMBSM_RIGHT) != (e.MouseInput.ButtonStates & EMBSM_RIGHT) )
			{
				e.MouseInput.Event = (e.MouseInput.ButtonStates & EMBSM_RIGHT) ? EMIE_RMOUSE_PRESSED_DOWN : EMIE_RMOUSE_LEFT_UP;
				postEventFromUser(e);
			}

			if ( (MouseButtonStates & EMBSM_MIDDLE) != (e.MouseInput.ButtonStates & EMBSM_MIDDLE) )
			{
				e.MouseInput.Event = (e.MouseInput.ButtonStates & EMBSM_MIDDLE) ? EMIE_MMOUSE_PRESSED_DOWN : EMIE_MMOUSE_LEFT_UP;
				postEventFromUser(e);
			}

			// save current button states
			MouseButtonStates = e.MouseInput.ButtonStates;

			break;
		}
		case WINDOW_BUFFER_SIZE_EVENT:
			VideoDriver->OnResize(
				core::dimension2d<uint32_t>(in.Event.WindowBufferSizeEvent.dwSize.X,
				                       in.Event.WindowBufferSizeEvent.dwSize.Y));
			break;
		case FOCUS_EVENT:
			IsWindowFocused = (in.Event.FocusEvent.bSetFocus == TRUE);
			break;
		default:
			break;
		}
		GetNumberOfConsoleInputEvents(WindowsSTDIn, &count);
	}

	// set input mode
	SetConsoleMode(WindowsSTDIn, oldMode);
#else
	// todo: keyboard input from terminal in raw mode
#endif

	return !Close;
}

//! Cause the device to temporarily pause execution and let other processes to run
// This should bring down processor usage without major performance loss for Irrlicht
void CIrrDeviceConsole::yield()
{
#ifdef _IRR_WINDOWS_API_
	Sleep(1);
#else
	struct timespec ts = {0,0};
	nanosleep(&ts, NULL);
#endif
}

//! Pause execution and let other processes to run for a specified amount of time.
void CIrrDeviceConsole::sleep(uint32_t timeMs, bool pauseTimer)
{
	const bool wasStopped = Timer ? Timer->isStopped() : true;

#ifdef _IRR_WINDOWS_API_
	Sleep(timeMs);
#else
	struct timespec ts;
	ts.tv_sec = (time_t) (timeMs / 1000);
	ts.tv_nsec = (long) (timeMs % 1000) * 1000000;

	if (pauseTimer && !wasStopped)
		Timer->stop();

	nanosleep(&ts, NULL);
#endif

	if (pauseTimer && !wasStopped)
		Timer->start();
}

//! sets the caption of the window
void CIrrDeviceConsole::setWindowCaption(const std::wstring& text)
{
#ifdef _IRR_WINDOWS_NT_CONSOLE_
	SetConsoleTitleW(text.c_str());
#endif
}

//! returns if window is active. if not, nothing need to be drawn
bool CIrrDeviceConsole::isWindowActive() const
{
	// there is no window, but we always assume it is active
	return true;
}

//! returns if window has focus
bool CIrrDeviceConsole::isWindowFocused() const
{
	return IsWindowFocused;
}

//! returns if window is minimized
bool CIrrDeviceConsole::isWindowMinimized() const
{
	return false;
}

//! presents a surface in the client area
bool CIrrDeviceConsole::present(video::IImage* surface, void* windowId, core::rect<int32_t>* src)
{

	if (surface)
	{
		for (uint32_t y=0; y < surface->getDimension().Height; ++y)
		{
			for (uint32_t x=0; x< surface->getDimension().Width; ++x)
			{
				// get average pixel
				uint32_t avg = surface->getPixel(x,y).getAverage() * (ASCIIArtCharsCount-1);
				avg /= 255;
				OutputBuffer[y] [x] = ASCIIArtChars[avg];
			}
		}
	}

	// draw output
	for (uint32_t y=0; y<OutputBuffer.size(); ++y)
	{
		setTextCursorPos(0,y);
		fprintf(OutFile, "%s", OutputBuffer[y].c_str());
	}
	return surface != 0;
}

//! notifies the device that it should close itself
void CIrrDeviceConsole::closeDevice()
{
	// return false next time we run()
	Close = true;
}


//! Sets if the window should be resizable in windowed mode.
void CIrrDeviceConsole::setResizable(bool resize)
{
	// do nothing
}


//! Minimize the window.
void CIrrDeviceConsole::minimizeWindow()
{
	// do nothing
}


//! Maximize window
void CIrrDeviceConsole::maximizeWindow()
{
	// do nothing
}


//! Restore original window size
void CIrrDeviceConsole::restoreWindow()
{
	// do nothing
}


void CIrrDeviceConsole::setTextCursorPos(int16_t x, int16_t y)
{
#ifdef _IRR_WINDOWS_NT_CONSOLE_
	// move WinNT cursor
	COORD Position;
    Position.X = x;
    Position.Y = y;
    SetConsoleCursorPosition(WindowsSTDOut, Position);
#elif defined(_IRR_VT100_CONSOLE_)
	// send escape code
	fprintf(OutFile, "%c[%d;%dH", 27, y, x);
#else
	// not implemented
#endif
}

void CIrrDeviceConsole::addPostPresentText(int16_t X, int16_t Y, const wchar_t *text)
{
	SPostPresentText p;
	p.Text = text;
	p.Pos.X = X;
	p.Pos.Y = Y;
	Text.push_back(p);
}

} // end namespace irr

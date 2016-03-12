// Copyright (C) 2009-2012 Gaz Davidson
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_IRR_DEVICE_CONSOLE_H_INCLUDED__
#define __C_IRR_DEVICE_CONSOLE_H_INCLUDED__

#include "IrrCompileConfig.h"
#ifdef _IRR_COMPILE_WITH_CONSOLE_DEVICE_


#include "SIrrCreationParameters.h"
#include "CIrrDeviceStub.h"
#include "IImagePresenter.h"
// for console font

#ifdef _IRR_WINDOWS_API_
#define WIN32_LEAN_AND_MEAN
#if !defined(_IRR_XBOX_PLATFORM_)
	#include <windows.h>
#endif
#if(_WIN32_WINNT >= 0x0500)
#define _IRR_WINDOWS_NT_CONSOLE_
#endif
#else
#include <time.h>
#endif

// for now we assume all other terminal types are VT100
#ifndef _IRR_WINDOWS_NT_CONSOLE_
#define _IRR_VT100_CONSOLE_
#endif

namespace irr
{

	class CIrrDeviceConsole : public CIrrDeviceStub, video::IImagePresenter
	{
	public:

		//! constructor
		CIrrDeviceConsole(const SIrrlichtCreationParameters& params);

		//! destructor
		virtual ~CIrrDeviceConsole();

		//! runs the device. Returns false if device wants to be deleted
		virtual bool run();

		//! Cause the device to temporarily pause execution and let other processes to run
		// This should bring down processor usage without major performance loss for Irrlicht
		virtual void yield();

		//! Pause execution and let other processes to run for a specified amount of time.
		virtual void sleep(u32 timeMs, bool pauseTimer);

		//! sets the caption of the window
		virtual void setWindowCaption(const wchar_t* text);

		//! returns if window is active. if not, nothing need to be drawn
		virtual bool isWindowActive() const;

		//! returns if window has focus
		virtual bool isWindowFocused() const;

		//! returns if window is minimized
		virtual bool isWindowMinimized() const;

		//! presents a surface in the client area
		virtual bool present(video::IImage* surface, void* windowId=0, core::rect<s32>* src=0);

		//! notifies the device that it should close itself
		virtual void closeDevice();

		//! Sets if the window should be resizable in windowed mode.
		virtual void setResizable(bool resize=false);

		//! Minimizes the window.
		virtual void minimizeWindow();

		//! Maximizes the window.
		virtual void maximizeWindow();

		//! Restores the window size.
		virtual void restoreWindow();

		//! Get the device type
		virtual E_DEVICE_TYPE getType() const
		{
				return EIDT_CONSOLE;
		}

		void addPostPresentText(s16 X, s16 Y, const wchar_t *text);

		//! Implementation of the win32 console mouse cursor
		class CCursorControl : public gui::ICursorControl
		{
		public:

			CCursorControl(const core::dimension2d<u32>& wsize)
				: WindowSize(wsize), InvWindowSize(0.0f, 0.0f), IsVisible(true), UseReferenceRect(false)
			{
				if (WindowSize.Width!=0)
					InvWindowSize.Width = 1.0f / WindowSize.Width;

				if (WindowSize.Height!=0)
					InvWindowSize.Height = 1.0f / WindowSize.Height;
			}

			//! Changes the visible state of the mouse cursor.
			virtual void setVisible(bool visible)
			{
				if(visible != IsVisible)
				{
					IsVisible = visible;
					setPosition(CursorPos.X, CursorPos.Y);
				}
			}

			//! Returns if the cursor is currently visible.
			virtual bool isVisible() const
			{
				return IsVisible;
			}

			//! Sets the new position of the cursor.
			virtual void setPosition(const core::position2d<f32> &pos)
			{
				setPosition(pos.X, pos.Y);
			}

			//! Sets the new position of the cursor.
			virtual void setPosition(f32 x, f32 y)
			{
				if (!UseReferenceRect)
					setPosition((s32)(x*WindowSize.Width), (s32)(y*WindowSize.Height));
				else
					setPosition((s32)(x*ReferenceRect.getWidth()), (s32)(y*ReferenceRect.getHeight()));
			}

			//! Sets the new position of the cursor.
			virtual void setPosition(const core::position2d<s32> &pos)
			{
				setPosition(pos.X, pos.Y);
			}

			//! Sets the new position of the cursor.
			virtual void setPosition(s32 x, s32 y)
			{
				setInternalCursorPosition(core::position2di(x,y));
			}

			//! Returns the current position of the mouse cursor.
			virtual const core::position2d<s32>& getPosition()
			{
				return CursorPos;
			}

			//! Returns the current position of the mouse cursor.
			virtual core::position2d<f32> getRelativePosition()
			{
				if (!UseReferenceRect)
				{
					return core::position2d<f32>(CursorPos.X * InvWindowSize.Width,
						CursorPos.Y * InvWindowSize.Height);
				}

				return core::position2d<f32>(CursorPos.X / (f32)ReferenceRect.getWidth(),
						CursorPos.Y / (f32)ReferenceRect.getHeight());
			}

			//! Sets an absolute reference rect for calculating the cursor position.
			virtual void setReferenceRect(core::rect<s32>* rect=0)
			{
				if (rect)
				{
					ReferenceRect = *rect;
					UseReferenceRect = true;

					// prevent division through zero and uneven sizes

					if (!ReferenceRect.getHeight() || ReferenceRect.getHeight()%2)
						ReferenceRect.LowerRightCorner.Y += 1;

					if (!ReferenceRect.getWidth() || ReferenceRect.getWidth()%2)
						ReferenceRect.LowerRightCorner.X += 1;
				}
				else
					UseReferenceRect = false;
			}


			//! Updates the internal cursor position
			void setInternalCursorPosition(const core::position2di &pos)
			{
				CursorPos = pos;

				if (UseReferenceRect)
					CursorPos -= ReferenceRect.UpperLeftCorner;
			}

		private:

			core::position2d<s32>  CursorPos;
			core::dimension2d<u32> WindowSize;
			core::dimension2d<f32> InvWindowSize;
			bool                   IsVisible,
			                       UseReferenceRect;
			core::rect<s32>        ReferenceRect;
		};

	private:

		//! Set the position of the text caret
		void setTextCursorPos(s16 x, s16 y);

		// text to be added after drawing the screen
		struct SPostPresentText
		{
			core::position2d<s16> Pos;
			core::stringc         Text;
		};

		bool IsWindowFocused;

		core::array<core::stringc> OutputBuffer;
		core::array<SPostPresentText> Text;

		FILE *OutFile;

#ifdef _IRR_WINDOWS_NT_CONSOLE_
		HANDLE WindowsSTDIn, WindowsSTDOut;
		u32 MouseButtonStates;
#endif
	};


} // end namespace irr

#endif // _IRR_COMPILE_WITH_CONSOLE_DEVICE_
#endif // __C_IRR_DEVICE_CONSOLE_H_INCLUDED__


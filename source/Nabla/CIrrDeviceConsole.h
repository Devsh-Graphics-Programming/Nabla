// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_NBL_DEVICE_CONSOLE_H_INCLUDED__
#define __NBL_C_NBL_DEVICE_CONSOLE_H_INCLUDED__

#include "nbl/core/compile_config.h"

#include "SIrrCreationParameters.h"
#include "CIrrDeviceStub.h"
// for console font

#ifdef _NBL_WINDOWS_API_
#define WIN32_LEAN_AND_MEAN

#include <windows.h>

#if(_WIN32_WINNT >= 0x0500)
#define _NBL_WINDOWS_NT_CONSOLE_
#endif
#else
#include <time.h>
#endif

// for now we assume all other terminal types are VT100
#ifndef _NBL_WINDOWS_NT_CONSOLE_
#define _NBL_VT100_CONSOLE_
#endif

namespace nbl
{

	class CIrrDeviceConsole : public CIrrDeviceStub
	{
        protected:
            //! destructor
            virtual ~CIrrDeviceConsole();

        public:
            //! constructor
            CIrrDeviceConsole(const SIrrlichtCreationParameters& params);

            //! runs the device. Returns false if device wants to be deleted
            virtual bool run();

            //! Cause the device to temporarily pause execution and let other processes to run
            // This should bring down processor usage without major performance loss for Irrlicht
            virtual void yield();

            //! Pause execution and let other processes to run for a specified amount of time.
            virtual void sleep(uint32_t timeMs, bool pauseTimer);

            //! sets the caption of the window
            virtual void setWindowCaption(const std::wstring& text);

            //! returns if window is active. if not, nothing need to be drawn
            virtual bool isWindowActive() const;

            //! returns if window has focus
            virtual bool isWindowFocused() const;

            //! returns if window is minimized
            virtual bool isWindowMinimized() const;

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

            bool switchToFullscreen(bool) override { return false; }

            void addPostPresentText(int16_t X, int16_t Y, const wchar_t *text);

            //! Implementation of the win32 console mouse cursor
            class CCursorControl : public gui::ICursorControl
            {
            public:

                CCursorControl(const core::dimension2d<uint32_t>& wsize)
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
                virtual void setPosition(const core::position2d<float> &pos)
                {
                    setPosition(pos.X, pos.Y);
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(float x, float y)
                {
                    if (!UseReferenceRect)
                        setPosition((int32_t)(x*WindowSize.Width), (int32_t)(y*WindowSize.Height));
                    else
                        setPosition((int32_t)(x*ReferenceRect.getWidth()), (int32_t)(y*ReferenceRect.getHeight()));
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(const core::position2d<int32_t> &pos)
                {
                    setPosition(pos.X, pos.Y);
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(int32_t x, int32_t y)
                {
                    setInternalCursorPosition(core::position2di(x,y));
                }

                //! Returns the current position of the mouse cursor.
                virtual const core::position2d<int32_t>& getPosition()
                {
                    return CursorPos;
                }

                //! Returns the current position of the mouse cursor.
                virtual core::position2d<float> getRelativePosition()
                {
                    if (!UseReferenceRect)
                    {
                        return core::position2d<float>(CursorPos.X * InvWindowSize.Width,
                            CursorPos.Y * InvWindowSize.Height);
                    }

                    return core::position2d<float>(CursorPos.X / (float)ReferenceRect.getWidth(),
                            CursorPos.Y / (float)ReferenceRect.getHeight());
                }

                //! Sets an absolute reference rect for calculating the cursor position.
                virtual void setReferenceRect(core::rect<int32_t>* rect=0)
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

                core::position2d<int32_t>  CursorPos;
                core::dimension2d<uint32_t> WindowSize;
                core::dimension2d<float> InvWindowSize;
                bool                   IsVisible,
                                       UseReferenceRect;
                core::rect<int32_t>        ReferenceRect;
            };

        private:

            //! Set the position of the text caret
            void setTextCursorPos(int16_t x, int16_t y);

            // text to be added after drawing the screen
            struct SPostPresentText
            {
                core::position2d<int16_t> Pos;
                core::stringc         Text;
            };

            bool IsWindowFocused;

            core::vector<core::stringc> OutputBuffer;
            core::vector<SPostPresentText> Text;

            FILE *OutFile;

    #ifdef _NBL_WINDOWS_NT_CONSOLE_
            HANDLE WindowsSTDIn, WindowsSTDOut;
            uint32_t MouseButtonStates;
    #endif
	};


} // end namespace nbl

#endif


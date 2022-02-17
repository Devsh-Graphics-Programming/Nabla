// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_NBL_DEVICE_WIN32_H_INCLUDED__
#define __NBL_C_NBL_DEVICE_WIN32_H_INCLUDED__

#ifdef _NBL_COMPILE_WITH_WINDOWS_DEVICE_

namespace nbl
{
	
	class CIrrDeviceWin32 : public CIrrDeviceStub
	{

        protected:
            //! destructor
            virtual ~CIrrDeviceWin32();

        public:
            //! constructor
            CIrrDeviceWin32(const SIrrlichtCreationParameters& params);

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

            //! Remove all messages pending in the system message loop
            virtual void clearSystemMessages();


            //! Compares to the last call of this function to return double and triple clicks.
            //! \return Returns only 1,2 or 3. A 4th click will start with 1 again.
            virtual uint32_t checkSuccessiveClicks(int32_t mouseX, int32_t mouseY, EMOUSE_INPUT_EVENT inputEvent )
            {
                // we just have to make it public
                return CIrrDeviceStub::checkSuccessiveClicks(mouseX, mouseY, inputEvent );
            }

            //! switchs to fullscreen
            bool switchToFullscreen(bool reset=false) override;

            //! Check for and show last Windows API error to help internal debugging.
            //! Does call GetLastError and on errors formats the errortext and displays it in a messagebox.
            static void ReportLastWinApiError();

            //! Implementation of the win32 cursor control
            class CCursorControl : public gui::ICursorControl
            {
            public:

                CCursorControl(CIrrDeviceWin32* device, const core::dimension2d<uint32_t>& wsize, HWND hwnd, bool fullscreen);
                ~CCursorControl();

                //! Changes the visible state of the mouse cursor.
                virtual void setVisible(bool visible)
                {
                    CURSORINFO info;
                    info.cbSize = sizeof(CURSORINFO);
                    BOOL gotCursorInfo = GetCursorInfo(&info);
                    while ( gotCursorInfo )
                    {
    #ifdef CURSOR_SUPPRESSED
                        // new flag for Windows 8, where cursor
                        // might be suppressed for touch interface
                        if (info.flags == CURSOR_SUPPRESSED)
                        {
                            visible=false;
                            break;
                        }
    #endif
                        if ( (visible && info.flags == CURSOR_SHOWING) || // visible
                            (!visible && info.flags == 0 ) ) // hidden
                        {
                            break;
                        }
                        // this only increases an internal
                        // display counter in windows, so it
                        // might have to be called some more
                        const int showResult = ShowCursor(visible);
                        // if result has correct sign we can
                        // stop here as well
                        if (( !visible && showResult < 0 ) ||
                            (visible && showResult >= 0))
                            break;
                        // yes, it really must be set each time
                        info.cbSize = sizeof(CURSORINFO);
                        gotCursorInfo = GetCursorInfo(&info);
                    }
                    IsVisible = visible;
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
                        setPosition(core::round<float,int32_t>(x*WindowSize.Width), core::round<float,int32_t>(y*WindowSize.Height));
                    else
                        setPosition(core::round<float,int32_t>(x*ReferenceRect.getWidth()), core::round<float,int32_t>(y*ReferenceRect.getHeight()));
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(const core::position2d<int32_t> &pos)
                {
                    setPosition(pos.X, pos.Y);
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(int32_t x, int32_t y)
                {
                    if (UseReferenceRect)
                    {
                        SetCursorPos(ReferenceRect.UpperLeftCorner.X + x,
                                     ReferenceRect.UpperLeftCorner.Y + y);
                    }
                    else
                    {
                        RECT rect;
                        if (GetWindowRect(HWnd, &rect))
                            SetCursorPos(x + rect.left + BorderX, y + rect.top + BorderY);
                    }

                    CursorPos.X = x;
                    CursorPos.Y = y;
                }

                //! Returns the current position of the mouse cursor.
                virtual const core::position2d<int32_t>& getPosition()
                {
                    updateInternalCursorPosition();
                    return CursorPos;
                }

                //! Returns the current position of the mouse cursor.
                virtual core::position2d<float> getRelativePosition()
                {
                    updateInternalCursorPosition();

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

                /** Used to notify the cursor that the window was resized. */
                virtual void OnResize(const core::dimension2d<uint32_t>& size)
                {
                    WindowSize = size;
                    if (size.Width!=0)
                        InvWindowSize.Width = 1.0f / size.Width;
                    else
                        InvWindowSize.Width = 0.f;

                    if (size.Height!=0)
                        InvWindowSize.Height = 1.0f / size.Height;
                    else
                        InvWindowSize.Height = 0.f;
                }

                /** Used to notify the cursor that the window resizable settings changed. */
                void updateBorderSize(bool fullscreen, bool resizable)
                {
                   if (!fullscreen)
                   {
                      if (resizable)
                      {
                         BorderX = GetSystemMetrics(SM_CXSIZEFRAME);
                         BorderY = GetSystemMetrics(SM_CYCAPTION) + GetSystemMetrics(SM_CYSIZEFRAME);
                      }
                      else
                      {
                         BorderX = GetSystemMetrics(SM_CXDLGFRAME);
                         BorderY = GetSystemMetrics(SM_CYCAPTION) + GetSystemMetrics(SM_CYDLGFRAME);
                      }
                   }
                   else
                   {
                      BorderX = BorderY = 0;
                   }
                }


                //! Sets the active cursor icon
                virtual void setActiveIcon(gui::ECURSOR_ICON iconId);

                //! Gets the currently active icon
                virtual gui::ECURSOR_ICON getActiveIcon() const
                {
                    return ActiveIcon;
                }

                //! Add a custom sprite as cursor icon.
                virtual gui::ECURSOR_ICON addIcon(const gui::SCursorSprite& icon);

                //! replace the given cursor icon.
                virtual void changeIcon(gui::ECURSOR_ICON iconId, const gui::SCursorSprite& icon);

                //! Return a system-specific size which is supported for cursors. Larger icons will fail, smaller icons might work.
                virtual core::dimension2di getSupportedIconSize() const;

                void update();

            private:

                //! Updates the internal cursor position
                void updateInternalCursorPosition()
                {
                    POINT p;
                    if (!GetCursorPos(&p))
                    {
                        DWORD xy = GetMessagePos();
                        p.x = GET_X_LPARAM(xy);
                        p.y = GET_Y_LPARAM(xy);
                    }

                    if (UseReferenceRect)
                    {
                        CursorPos.X = p.x - ReferenceRect.UpperLeftCorner.X;
                        CursorPos.Y = p.y - ReferenceRect.UpperLeftCorner.Y;
                    }
                    else
                    {
                        RECT rect;
                        if (GetWindowRect(HWnd, &rect))
                        {
                            CursorPos.X = p.x-rect.left-BorderX;
                            CursorPos.Y = p.y-rect.top-BorderY;
                        }
                        else
                        {
                            // window seems not to be existent, so set cursor to
                            // a negative value
                            CursorPos.X = -1;
                            CursorPos.Y = -1;
                        }
                    }
                }

                CIrrDeviceWin32* Device;
                core::position2d<int32_t> CursorPos;
                core::dimension2d<uint32_t> WindowSize;
                core::dimension2d<float> InvWindowSize;
                HWND HWnd;

                int32_t BorderX, BorderY;
                core::rect<int32_t> ReferenceRect;
                bool UseReferenceRect;
                bool IsVisible;


                struct CursorFrameW32
                {
                    CursorFrameW32() : IconHW(0) {}
                    CursorFrameW32(HCURSOR icon) : IconHW(icon) {}

                    HCURSOR IconHW;	// hardware cursor
                };

                struct CursorW32
                {
                    CursorW32() {}
                    explicit CursorW32(HCURSOR iconHw, uint32_t frameTime=0) : FrameTime(frameTime)
                    {
                        Frames.push_back( CursorFrameW32(iconHw) );
                    }
                    core::vector<CursorFrameW32> Frames;
                    uint32_t FrameTime;
                };

                core::vector<CursorW32> Cursors;
                gui::ECURSOR_ICON ActiveIcon;
                uint32_t ActiveIconStartTime;

                void initCursors();
            };

            //! returns the win32 cursor control
            CCursorControl* getWin32CursorControl();

        private:

            //! Process system events
            void handleSystemMessages();

            void getWindowsVersion(core::stringc& version);

            void resizeIfNecessary();

            HWND HWnd;

            bool ChangedToFullScreen;
            bool Resized;
            bool ExternalWindow;
            CCursorControl* Win32CursorControl;
#if 0
            DEVMODE DesktopMode;
#endif
	};

} // end namespace nbl

#endif // _NBL_COMPILE_WITH_WINDOWS_DEVICE_
#endif

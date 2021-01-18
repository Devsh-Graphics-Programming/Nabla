// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_NBL_DEVICE_LINUX_H_INCLUDED__
#define __NBL_C_NBL_DEVICE_LINUX_H_INCLUDED__

#include "BuildConfigOptions.h"
#include "nbl/system/compile_config.h"

#ifdef _NBL_COMPILE_WITH_X11_DEVICE_

#include "CIrrDeviceStub.h"
#include "IrrlichtDevice.h"
#include "ICursorControl.h"
#include "os.h"

#ifdef _NBL_COMPILE_WITH_X11_

#include "COpenGLStateManager.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#ifdef _NBL_LINUX_X11_VIDMODE_
    #include <X11/extensions/xf86vmode.h>
#endif
#ifdef _NBL_LINUX_X11_RANDR_
    #include <X11/extensions/Xrandr.h>
#endif
#include <X11/keysym.h>

#ifdef _NBL_COMPILE_WITH_OPENGL_
    #include "GL/glx.h"
    #include "../src/3rdparty/GL/glxext.h"
#endif

#else
#define KeySym int32_t
#endif

namespace nbl
{

	class CIrrDeviceLinux : public CIrrDeviceStub
	{
        protected:
            //! destructor
            virtual ~CIrrDeviceLinux();

        public:
            //! constructor
            CIrrDeviceLinux(const SIrrlichtCreationParameters& param);

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

            //! returns if window has focus.
            virtual bool isWindowFocused() const;

            //! returns if window is minimized.
            virtual bool isWindowMinimized() const;

            //! returns color format of the window.
            virtual asset::E_FORMAT getColorFormat() const;

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

            //! Activate any joysticks, and generate events for them.
            virtual bool activateJoysticks(core::vector<SJoystickInfo> & joystickInfo);

            //! gets text from the clipboard
            //! \return Returns 0 if no string is in there.
            virtual const char* getTextFromClipboard() const;

            //! copies text to the clipboard
            //! This sets the clipboard selection and _not_ the primary selection which you have on X on the middle mouse button.
            virtual void copyToClipboard(const char* text) const;

            //! Remove all messages pending in the system message loop
            virtual void clearSystemMessages();

            //! Get the device type
            virtual E_DEVICE_TYPE getType() const
            {
                    return EIDT_X11;
            }


        private:

            //! create the driver
            void createDriver();

            bool createWindow();

            void createKeyMap();

            void pollJoysticks();

            void initXAtoms();

            bool switchToFullscreen(bool reset=false);

#ifdef _NBL_COMPILE_WITH_X11_
            bool createInputContext();
            void destroyInputContext();
            EKEY_CODE getKeyCode(const uint32_t& xEventKey);
#endif

            //! Implementation of the linux cursor control
            class CCursorControl : public gui::ICursorControl
            {
            public:

                CCursorControl(CIrrDeviceLinux* dev, bool null);

                ~CCursorControl();

                //! Changes the visible state of the mouse cursor.
                virtual void setVisible(bool visible)
                {
                    if (visible==IsVisible)
                        return;
                    IsVisible = visible;
    #ifdef _NBL_COMPILE_WITH_X11_
                    if (!Null)
                    {
                        if ( !IsVisible )
                            XDefineCursor( Device->display, Device->window, invisCursor );
                        else
                            XUndefineCursor( Device->display, Device->window );
                    }
    #endif
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
                    setPosition((int32_t)(x*Device->Width), (int32_t)(y*Device->Height));
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(const core::position2d<int32_t> &pos)
                {
                    setPosition(pos.X, pos.Y);
                }

                //! Sets the new position of the cursor.
                virtual void setPosition(int32_t x, int32_t y)
                {
    #ifdef _NBL_COMPILE_WITH_X11_

                    if (!Null)
                    {
                        if (UseReferenceRect)
                        {
                            XWarpPointer(Device->display,
                                None,
                                Device->window, 0, 0,
                                Device->Width,
                                Device->Height,
                                ReferenceRect.UpperLeftCorner.X + x,
                                ReferenceRect.UpperLeftCorner.Y + y);

                        }
                        else
                        {
                            XWarpPointer(Device->display,
                                None,
                                Device->window, 0, 0,
                                Device->Width,
                                Device->Height, x, y);
                        }
                        XFlush(Device->display);
                    }
    #endif
                    CursorPos.X = x;
                    CursorPos.Y = y;
                }

                //! Returns the current position of the mouse cursor.
                virtual const core::position2d<int32_t>& getPosition()
                {
                    updateCursorPos();
                    return CursorPos;
                }

                //! Returns the current position of the mouse cursor.
                virtual core::position2d<float> getRelativePosition()
                {
                    updateCursorPos();

                    if (!UseReferenceRect)
                    {
                        return core::position2d<float>(CursorPos.X / (float)Device->Width,
                            CursorPos.Y / (float)Device->Height);
                    }

                    return core::position2d<float>(CursorPos.X / (float)ReferenceRect.getWidth(),
                            CursorPos.Y / (float)ReferenceRect.getHeight());
                }

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

    #ifdef _NBL_COMPILE_WITH_X11_
                //! Set platform specific behavior flags.
                virtual void setPlatformBehavior(gui::ECURSOR_PLATFORM_BEHAVIOR behavior) {PlatformBehavior = behavior; }

                //! Return platform specific behavior.
                virtual gui::ECURSOR_PLATFORM_BEHAVIOR getPlatformBehavior() const { return PlatformBehavior; }

                void update();
                void clearCursors();
    #endif
            private:

                void updateCursorPos()
                {
    #ifdef _NBL_COMPILE_WITH_X11_
                    if (Null)
                        return;

                    if ( PlatformBehavior&gui::ECPB_X11_CACHE_UPDATES && !Device->getTimer()->isStopped() )
                    {
                        auto now = Device->getTimer()->getTime();
                        if (now <= lastQuery)
                            return;
                        lastQuery = std::chrono::duration_cast<std::chrono::milliseconds>(now);
                    }

                    Window tmp;
                    int itmp1, itmp2;
                    unsigned  int maskreturn;
                    XQueryPointer(Device->display, Device->window,
                        &tmp, &tmp,
                        &itmp1, &itmp2,
                        &CursorPos.X, &CursorPos.Y, &maskreturn);

                    if (CursorPos.X < 0)
                        CursorPos.X = 0;
                    if (CursorPos.X > (int32_t) Device->Width)
                        CursorPos.X = Device->Width;
                    if (CursorPos.Y < 0)
                        CursorPos.Y = 0;
                    if (CursorPos.Y > (int32_t) Device->Height)
                        CursorPos.Y = Device->Height;
    #endif
                }

                CIrrDeviceLinux* Device;
                core::position2d<int32_t> CursorPos;
                core::rect<int32_t> ReferenceRect;
    #ifdef _NBL_COMPILE_WITH_X11_
                gui::ECURSOR_PLATFORM_BEHAVIOR PlatformBehavior;
                std::chrono::milliseconds lastQuery;
                Cursor invisCursor;

                struct CursorFrameX11
                {
                    CursorFrameX11() : IconHW(0) {}
                    CursorFrameX11(Cursor icon) : IconHW(icon) {}

                    Cursor IconHW;	// hardware cursor
                };

                struct CursorX11
                {
                    CursorX11() {}
                    explicit CursorX11(Cursor iconHw, uint32_t frameTime=0) : FrameTime(frameTime)
                    {
                        Frames.push_back( CursorFrameX11(iconHw) );
                    }
                    core::vector<CursorFrameX11> Frames;
                    uint32_t FrameTime;
                };

                core::vector<CursorX11> Cursors;

                void initCursors();
    #endif
                bool IsVisible;
                bool Null;
                bool UseReferenceRect;
                gui::ECURSOR_ICON ActiveIcon;
                uint32_t ActiveIconStartTime;
            };

            friend class CCursorControl;

    #ifdef _NBL_COMPILE_WITH_X11_
            friend class COpenGLDriver;

            Display *display;
            XVisualInfo* visual;
            int screennr;
            Window window;
            XSetWindowAttributes attributes;
            XSizeHints* StdHints;
            XIM XInputMethod;
            XIC XInputContext;
            mutable core::stringc Clipboard;
            #ifdef _NBL_LINUX_X11_VIDMODE_
            XF86VidModeModeInfo oldVideoMode;
            #endif
            #ifdef _NBL_LINUX_X11_RANDR_
            SizeID oldRandrMode;
            Rotation oldRandrRotation;
            #endif
            #ifdef _NBL_COMPILE_WITH_OPENGL_
            GLXWindow glxWin;
            GLXContext Context;
            void* AuxContexts;
            #endif
    #endif
            uint32_t Width, Height;
            bool WindowHasFocus;
            bool WindowMinimized;
            bool UseXVidMode;
            bool UseXRandR;
            bool ExternalWindow;
            int AutorepeatSupport;

            core::unordered_map<KeySym,int32_t> KeyMap;

    #if defined(_NBL_COMPILE_WITH_JOYSTICK_EVENTS_)
            struct JoystickInfo
            {
                int	fd;
                int	axes;
                int	buttons;

                SEvent persistentData;

                JoystickInfo() : fd(-1), axes(0), buttons(0) { }
            };
            core::vector<JoystickInfo> ActiveJoysticks;
    #endif
	};


} // end namespace nbl

#endif // _NBL_COMPILE_WITH_X11_DEVICE_
#endif


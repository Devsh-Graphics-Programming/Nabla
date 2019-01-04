// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// This device code is based on the original SDL device implementation
// contributed by Shane Parker (sirshane).

#ifndef __C_IRR_DEVICE_SDL_H_INCLUDED__
#define __C_IRR_DEVICE_SDL_H_INCLUDED__

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_SDL_DEVICE_

#include "IrrlichtDevice.h"
#include "CIrrDeviceStub.h"
#include "IImagePresenter.h"
#include "ICursorControl.h"

#include <SDL/SDL.h>
#include <SDL/SDL_syswm.h>

namespace irr
{

	class CIrrDeviceSDL : public CIrrDeviceStub, video::IImagePresenter
	{
        protected:
            //! destructor
            virtual ~CIrrDeviceSDL();

        public:
            //! constructor
            CIrrDeviceSDL(const SIrrlichtCreationParameters& param);

            //! runs the device. Returns false if device wants to be deleted
            virtual bool run();

            //! pause execution temporarily
            virtual void yield();

            //! pause execution for a specified time
            virtual void sleep(uint32_t timeMs, bool pauseTimer);

            //! sets the caption of the window
            virtual void setWindowCaption(const std::wstring& text);

            //! returns if window is active. if not, nothing need to be drawn
            virtual bool isWindowActive() const;

            //! returns if window has focus.
            bool isWindowFocused() const;

            //! returns if window is minimized.
            bool isWindowMinimized() const;

            //! returns color format of the window.
            asset::E_FORMAT getColorFormat() const;

            //! presents a surface in the client area
            virtual bool present(video::IImage* surface, void* windowId=0, core::rect<int32_t>* src=0);

            //! notifies the device that it should close itself
            virtual void closeDevice();

            //! \return Returns a pointer to a list with all video modes supported
            video::IVideoModeList* getVideoModeList();

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

            //! Get the device type
            virtual E_DEVICE_TYPE getType() const
            {
                    return EIDT_SDL;
            }

            //! Implementation of the linux cursor control
            class CCursorControl : public gui::ICursorControl
            {
            public:

                CCursorControl(CIrrDeviceSDL* dev)
                    : Device(dev), IsVisible(true)
                {
                }

                //! Changes the visible state of the mouse cursor.
                virtual void setVisible(bool visible)
                {
                    IsVisible = visible;
                    if ( visible )
                        SDL_ShowCursor( SDL_ENABLE );
                    else
                        SDL_ShowCursor( SDL_DISABLE );
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
                    SDL_WarpMouse( x, y );
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
                    return core::position2d<float>(CursorPos.X / (float)Device->Width,
                        CursorPos.Y / (float)Device->Height);
                }

                virtual void setReferenceRect(core::rect<int32_t>* rect=0)
                {
                }

            private:

                void updateCursorPos()
                {
                    CursorPos.X = Device->MouseX;
                    CursorPos.Y = Device->MouseY;

                    if (CursorPos.X < 0)
                        CursorPos.X = 0;
                    if (CursorPos.X > (int32_t)Device->Width)
                        CursorPos.X = Device->Width;
                    if (CursorPos.Y < 0)
                        CursorPos.Y = 0;
                    if (CursorPos.Y > (int32_t)Device->Height)
                        CursorPos.Y = Device->Height;
                }

                CIrrDeviceSDL* Device;
                core::position2d<int32_t> CursorPos;
                bool IsVisible;
            };

        private:

            //! create the driver
            void createDriver();

            bool createWindow();

            void createKeyMap();

            SDL_Surface* Screen;
            int SDL_Flags;
    #if defined(_IRR_COMPILE_WITH_JOYSTICK_EVENTS_)
            core::vector<SDL_Joystick*> Joysticks;
    #endif

            int32_t MouseX, MouseY;
            uint32_t MouseButtonStates;

            uint32_t Width, Height;

            bool Resizable;
            bool WindowHasFocus;
            bool WindowMinimized;

            struct SKeyMap
            {
                SKeyMap() {}
                SKeyMap(int32_t x11, int32_t win32)
                    : SDLKey(x11), Win32Key(win32)
                {
                }

                int32_t SDLKey;
                int32_t Win32Key;

                bool operator<(const SKeyMap& o) const
                {
                    return SDLKey<o.SDLKey;
                }
            };

            core::vector<SKeyMap> KeyMap;
            SDL_SysWMinfo Info;
	};

} // end namespace irr

#endif // _IRR_COMPILE_WITH_SDL_DEVICE_
#endif // __C_IRR_DEVICE_SDL_H_INCLUDED__


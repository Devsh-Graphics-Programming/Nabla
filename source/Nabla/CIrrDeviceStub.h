// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_NBL_DEVICE_STUB_H_INCLUDED__
#define __NBL_C_NBL_DEVICE_STUB_H_INCLUDED__


#include "IrrlichtDevice.h"
#include "SIrrCreationParameters.h"


#include "nbl/asset/utils/IIncludeHandler.h"

#include "nbl/video/CCUDAHandler.h"
//#include "nbl/video/COpenCLHandler.h" move this
#include "COpenCLHandler.h"


namespace nbl
{
	// lots of prototypes:
	class ILogger;
	class CLogger;

	namespace scene
	{
		ISceneManager* createSceneManager(IrrlichtDevice* device, video::IVideoDriver* driver,
			io::IFileSystem* fs, gui::ICursorControl* cc);
	}

	namespace io
	{
		IFileSystem* createFileSystem();
	}


	//! Stub for an Irrlicht Device implementation
	class CIrrDeviceStub : public IrrlichtDevice
	{
	    protected:
            //! destructor
            virtual ~CIrrDeviceStub();

        public:
            //! constructor
            CIrrDeviceStub(const SIrrlichtCreationParameters& param);

            //! returns the video driver
            virtual video::IVideoDriver* getVideoDriver();

            //! return file system
            virtual io::IFileSystem* getFileSystem() { return FileSystem.get(); }

            //! returns the scene manager
            virtual scene::ISceneManager* getSceneManager();

            //! \return Returns a pointer to the mouse cursor control interface.
            virtual gui::ICursorControl* getCursorControl();

            //! Returns a pointer to the ITimer object. With it the current Time can be received.
            virtual ITimer* getTimer();

            //! Returns the version of the engine.
            virtual const char* getVersion() const;

            //! send the event to the right receiver
            virtual bool postEventFromUser(const SEvent& event);

            //! Sets a new event receiver to receive events
            virtual void setEventReceiver(IEventReceiver* receiver);

            //! Returns pointer to the current event receiver. Returns 0 if there is none.
            virtual IEventReceiver* getEventReceiver();

            //! Sets the input receiving scene manager.
            /** If set to null, the main scene manager (returned by GetSceneManager()) will receive the input */
            virtual void setInputReceivingSceneManager(scene::ISceneManager* sceneManager);

            //! Returns a pointer to the logger.
            virtual ILogger* getLogger();

            //! Returns the operation system opertator object.
            virtual IOSOperator* getOSOperator();

            //! Checks if the window is running in fullscreen mode.
            virtual bool isFullscreen() const;

            //! get color format of the current window
            virtual asset::E_FORMAT getColorFormat() const;

            //! Activate any joysticks, and generate events for them.
            virtual bool activateJoysticks(core::vector<SJoystickInfo> & joystickInfo);

            //! Set the maximal elapsed time between 2 clicks to generate doubleclicks for the mouse. It also affects tripleclick behavior.
            //! When set to 0 no double- and tripleclicks will be generated.
            virtual void setDoubleClickTime( uint32_t timeMs );

            //! Get the maximal elapsed time between 2 clicks to generate double- and tripleclicks for the mouse.
            virtual uint32_t getDoubleClickTime() const;

            //! Remove all messages pending in the system message loop
            virtual void clearSystemMessages();

        protected:

            void createGUIAndScene();

            //! checks version of SDK and prints warning if there might be a problem
            bool checkVersion(const char* version);

            //! Compares to the last call of this function to return double and triple clicks.
            //! \return Returns only 1,2 or 3. A 4th click will start with 1 again.
            virtual uint32_t checkSuccessiveClicks(int32_t mouseX, int32_t mouseY, EMOUSE_INPUT_EVENT inputEvent );

            video::IVideoDriver* VideoDriver;
            scene::ISceneManager* SceneManager;
            nbl::ITimer* Timer;
            gui::ICursorControl* CursorControl;
            IEventReceiver* UserReceiver;
            CLogger* Logger;
            IOSOperator* Operator;
            core::smart_refctd_ptr<io::IFileSystem> FileSystem;
            scene::ISceneManager* InputReceivingSceneManager;

            struct SMouseMultiClicks
            {
                SMouseMultiClicks()
                    : DoubleClickTime(500), CountSuccessiveClicks(0), LastClickTime(0), LastMouseInputEvent(EMIE_COUNT)
                {}

                uint32_t DoubleClickTime;
                uint32_t CountSuccessiveClicks;
                uint32_t LastClickTime;
                core::position2di LastClick;
                EMOUSE_INPUT_EVENT LastMouseInputEvent;
            };
            SMouseMultiClicks MouseMultiClicks;
            SIrrlichtCreationParameters CreationParams;
            bool Close;
	};

} // end namespace nbl

#endif


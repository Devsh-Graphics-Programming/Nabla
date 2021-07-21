// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_I_EVENT_RECEIVER_H_INCLUDED__
#define __NBL_I_EVENT_RECEIVER_H_INCLUDED__

#include "nbl/system/ILogger.h"
#include "Keycodes.h"
#include "irrString.h"

namespace nbl
{
	//! Enumeration for all event types there are.
	enum EEVENT_TYPE
	{
		//! An event of the graphical user interface.
		/** GUI events are created by the GUI environment or the GUI elements in response
		to mouse or keyboard events. When a GUI element receives an event it will either
		process it and return true, or pass the event to its parent. If an event is not absorbed
		before it reaches the root element then it will then be passed to the user receiver. */
		EET_GUI_EVENT = 0,

		//! A mouse input event.
		/** Mouse events are created by the device and passed to IrrlichtDevice::postEventFromUser
		in response to mouse input received from the operating system.
		Mouse events are first passed to the user receiver, then to the GUI environment and its elements,
		then finally the input receiving scene manager where it is passed to the active camera.
		*/
		EET_MOUSE_INPUT_EVENT,

		//! A key input event.
		/** Like mouse events, keyboard events are created by the device and passed to
		IrrlichtDevice::postEventFromUser. They take the same path as mouse events. */
		EET_KEY_INPUT_EVENT,

		//! A joystick (joypad, gamepad) input event.
		/** Joystick events are created by polling all connected joysticks once per
		device run() and then passing the events to IrrlichtDevice::postEventFromUser.
		They take the same path as mouse events.
		Windows, SDL: Implemented.
		Linux: Implemented, with POV hat issues.
		MacOS / Other: Not yet implemented.
		*/
		EET_JOYSTICK_INPUT_EVENT,

		//! A log event
		/** Log events are only passed to the user receiver if there is one. If they are absorbed by the
		user receiver then no text will be sent to the console. */
		EET_LOG_TEXT_EVENT,

		//! A user event with user data.
		/** This is not used by Irrlicht and can be used to send user
		specific data though the system. The Irrlicht 'window handle'
		can be obtained from IrrlichtDevice::getExposedVideoData()
		The usage and behavior depends on the operating system:
		Windows: send a WM_USER message to the Irrlicht Window; the
			wParam and lParam will be used to populate the
			UserData1 and UserData2 members of the SUserEvent.
		Linux: send a ClientMessage via XSendEvent to the Irrlicht
			Window; the data.l[0] and data.l[1] members will be
			casted to int32_t and used as UserData1 and UserData2.
		MacOS: Not yet implemented
		*/
		EET_USER_EVENT,

		//! This enum is never used, it only forces the compiler to
		//! compile these enumeration values to 32 bit.
		EGUIET_FORCE_32_BIT = 0x7fffffff

	};

	//! Enumeration for all mouse input events
	enum EMOUSE_INPUT_EVENT
	{
		//! Left mouse button was pressed down.
		EMIE_LMOUSE_PRESSED_DOWN = 0,

		//! Right mouse button was pressed down.
		EMIE_RMOUSE_PRESSED_DOWN,

		//! Middle mouse button was pressed down.
		EMIE_MMOUSE_PRESSED_DOWN,

		//! Left mouse button was left up.
		EMIE_LMOUSE_LEFT_UP,

		//! Right mouse button was left up.
		EMIE_RMOUSE_LEFT_UP,

		//! Middle mouse button was left up.
		EMIE_MMOUSE_LEFT_UP,

		//! The mouse cursor changed its position.
		EMIE_MOUSE_MOVED,

		//! The mouse wheel was moved. Use Wheel value in event data to find out
		//! in what direction and how fast.
		EMIE_MOUSE_WHEEL,

		//! Left mouse button double click.
		//! This event is generated after the second EMIE_LMOUSE_PRESSED_DOWN event.
		EMIE_LMOUSE_DOUBLE_CLICK,

		//! Right mouse button double click.
		//! This event is generated after the second EMIE_RMOUSE_PRESSED_DOWN event.
		EMIE_RMOUSE_DOUBLE_CLICK,

		//! Middle mouse button double click.
		//! This event is generated after the second EMIE_MMOUSE_PRESSED_DOWN event.
		EMIE_MMOUSE_DOUBLE_CLICK,

		//! Left mouse button triple click.
		//! This event is generated after the third EMIE_LMOUSE_PRESSED_DOWN event.
		EMIE_LMOUSE_TRIPLE_CLICK,

		//! Right mouse button triple click.
		//! This event is generated after the third EMIE_RMOUSE_PRESSED_DOWN event.
		EMIE_RMOUSE_TRIPLE_CLICK,

		//! Middle mouse button triple click.
		//! This event is generated after the third EMIE_MMOUSE_PRESSED_DOWN event.
		EMIE_MMOUSE_TRIPLE_CLICK,

		//! No real event. Just for convenience to get number of events
		EMIE_COUNT
	};

	//! Masks for mouse button states
	enum E_MOUSE_BUTTON_STATE_MASK
	{
		EMBSM_LEFT    = 0x01,
		EMBSM_RIGHT   = 0x02,
		EMBSM_MIDDLE  = 0x04,

		//! currently only on windows
		EMBSM_EXTRA1  = 0x08,

		//! currently only on windows
		EMBSM_EXTRA2  = 0x10,

		EMBSM_FORCE_32_BIT = 0x7fffffff
	};


//! SEvents hold information about an event. See nbl::IEventReceiver for details on event handling.
struct SEvent
{

	//! Any kind of mouse event.
	struct SMouseInput
	{
		//! X position of mouse cursor
		int32_t X;

		//! Y position of mouse cursor
		int32_t Y;

		//! mouse wheel delta, often 1.0 or -1.0, but can have other values < 0.f or > 0.f;
		/** Only valid if event was EMIE_MOUSE_WHEEL */
		float Wheel;

		//! True if shift was also pressed
		bool Shift:1;

		//! True if ctrl was also pressed
		bool Control:1;

		//! A bitmap of button states. You can use isButtonPressed() to determine
		//! if a button is pressed or not.
		//! Currently only valid if the event was EMIE_MOUSE_MOVED
		uint32_t ButtonStates;

		//! Is the left button pressed down?
		bool isLeftPressed() const { return 0 != ( ButtonStates & EMBSM_LEFT ); }

		//! Is the right button pressed down?
		bool isRightPressed() const { return 0 != ( ButtonStates & EMBSM_RIGHT ); }

		//! Is the middle button pressed down?
		bool isMiddlePressed() const { return 0 != ( ButtonStates & EMBSM_MIDDLE ); }

		//! Type of mouse event
		EMOUSE_INPUT_EVENT Event;
	};

	//! Any kind of keyboard event.
	struct SKeyInput
	{
		//! Character corresponding to the key (0, if not a character, value undefined in key releases)
		wchar_t Char;

		//! Key which has been pressed or released
		EKEY_CODE Key;

		//! If not true, then the key was left up
		bool PressedDown:1;

		//! True if shift was also pressed
		bool Shift:1;

		//! True if ctrl was also pressed
		bool Control:1;
	};

	//! A joystick event.
	/** Unlike other events, joystick events represent the result of polling
	 * each connected joystick once per run() of the device. Joystick events will
	 * not be generated by default.  If joystick support is available for the
	 * active device, _NBL_COMPILE_WITH_JOYSTICK_EVENTS_ is defined, and
	 * @ref nbl::IrrlichtDevice::activateJoysticks() has been called, an event of
	 * this type will be generated once per joystick per @ref IrrlichtDevice::run()
	 * regardless of whether the state of the joystick has actually changed. */
	struct SJoystickEvent
	{
		enum
		{
			NUMBER_OF_BUTTONS = 32,

			AXIS_X = 0, // e.g. analog stick 1 left to right
			AXIS_Y,		// e.g. analog stick 1 top to bottom
			AXIS_Z,		// e.g. throttle, or analog 2 stick 2 left to right
			AXIS_R,		// e.g. rudder, or analog 2 stick 2 top to bottom
			AXIS_U,
			AXIS_V,
			AXIS_6,
			AXIS_7,
			NUMBER_OF_AXES
		};

		/** A bitmap of button states.  You can use IsButtonPressed() to
		 ( check the state of each button from 0 to (NUMBER_OF_BUTTONS - 1) */
		uint32_t ButtonStates;

		/** For AXIS_X, AXIS_Y, AXIS_Z, AXIS_R, AXIS_U and AXIS_V
		 * Values are in the range -32768 to 32767, with 0 representing
		 * the center position.  You will receive the raw value from the
		 * joystick, and so will usually want to implement a dead zone around
		 * the center of the range. Axes not supported by this joystick will
		 * always have a value of 0. On Linux, POV hats are represented as axes,
		 * usually the last two active axis.
		 */
		int16_t Axis[NUMBER_OF_AXES];

		/** The POV represents the angle of the POV hat in degrees * 100,
		 * from 0 to 35,900.  A value of 65535 indicates that the POV hat
		 * is centered (or not present).
		 * This value is only supported on Windows.  On Linux, the POV hat
		 * will be sent as 2 axes instead. */
		uint16_t POV;

		//! The ID of the joystick which generated this event.
		/** This is an internal Irrlicht index; it does not map directly
		 * to any particular hardware joystick. */
		uint8_t Joystick;

		//! A helper function to check if a button is pressed.
		bool IsButtonPressed(uint32_t button) const
		{
			if(button >= (uint32_t)NUMBER_OF_BUTTONS)
				return false;

			return (ButtonStates & (1 << button)) ? true : false;
		}
	};


	//! Any kind of log event.
	struct SLogEvent
	{
		//! Pointer to text which has been logged
		const char* Text;

		//! Log level in which the text has been logged
		system::ILogger::E_LOG_LEVEL Level;
	};

	//! Any kind of user event.
	struct SUserEvent
	{
		//! Some user specified data as int
		int32_t UserData1;

		//! Another user specified data as int
		int32_t UserData2;
	};

	EEVENT_TYPE EventType;
	union
	{
		struct SMouseInput MouseInput;
		struct SKeyInput KeyInput;
		struct SJoystickEvent JoystickEvent;
		struct SLogEvent LogEvent;
		struct SUserEvent UserEvent;
	};

};

//! Interface of an object which can receive events.
/** Many of the engine's classes inherit IEventReceiver so they are able to
process events. Events usually start at a postEventFromUser function and are
passed down through a chain of event receivers until OnEvent returns true. See
nbl::EEVENT_TYPE for a description of where each type of event starts, and the
path it takes through the system. */
class NBL_FORCE_EBO IEventReceiver
{
public:
	//! Called if an event happened.
	/** Please take care that you should only return 'true' when you want to _prevent_ Irrlicht
	* from processing the event any further. So 'true' does mean that an event is completely done.
	* Therefore your return value for all unprocessed events should be 'false'.
	\return True if the event was processed.
	*/
	virtual bool OnEvent(const SEvent& event) = 0;
};


//! Information on a joystick, returned from @ref nbl::IrrlichtDevice::activateJoysticks()
struct SJoystickInfo
{
	//! The ID of the joystick
	/** This is an internal Irrlicht index; it does not map directly
	 * to any particular hardware joystick. It corresponds to the
	 * nbl::SJoystickEvent Joystick ID. */
	uint8_t				Joystick;

	//! The name that the joystick uses to identify itself.
	core::stringc	Name;

	//! The number of buttons that the joystick has.
	uint32_t				Buttons;

	//! The number of axes that the joystick has, i.e. X, Y, Z, R, U, V.
	/** Note: with a Linux device, the POV hat (if any) will use two axes. These
	 *  will be included in this count. */
	uint32_t				Axes;

	//! An indication of whether the joystick has a POV hat.
	/** A Windows device will identify the presence or absence or the POV hat.  A
	 *  Linux device cannot, and will always return POV_HAT_UNKNOWN. */
	enum
	{
		//! A hat is definitely present.
		POV_HAT_PRESENT,

		//! A hat is definitely not present.
		POV_HAT_ABSENT,

		//! The presence or absence of a hat cannot be determined.
		POV_HAT_UNKNOWN
	} PovHat;
}; // struct SJoystickInfo


} // end namespace nbl

#endif


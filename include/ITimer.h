// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_TIMER_H_INCLUDED__
#define __I_TIMER_H_INCLUDED__

#include "IReferenceCounted.h"

namespace irr
{

//! Interface for getting and manipulating the virtual time
class ITimer : public virtual IReferenceCounted
{
public:
	//! Returns current real time in milliseconds of the system.
	/** This value does not start with 0 when the application starts.
	For example in one implementation the value returned could be the
	amount of milliseconds which have elapsed since the system was started.
	*/
	virtual uint32_t getRealTime() const = 0;
	virtual uint64_t getRealTime64() const = 0;

	enum EWeekday
	{
		EWD_SUNDAY=0,
		EWD_MONDAY,
		EWD_TUESDAY,
		EWD_WEDNESDAY,
		EWD_THURSDAY,
		EWD_FRIDAY,
		EWD_SATURDAY
	};

	struct RealTimeDate
	{
		// Hour of the day, from 0 to 23
		uint32_t Hour;
		// Minute of the hour, from 0 to 59
		uint32_t Minute;
		// Second of the minute, due to extra seconds from 0 to 61
		uint32_t Second;
		// Year of the gregorian calender
		int32_t Year;
		// Month of the year, from 1 to 12
		uint32_t Month;
		// Day of the month, from 1 to 31
		uint32_t Day;
		// Weekday for the current day
		EWeekday Weekday;
		// Day of the year, from 1 to 366
		uint32_t Yearday;
		// Whether daylight saving is on
		bool IsDST;
	};

	virtual RealTimeDate getRealTimeAndDate() const = 0;

	//! Returns current virtual time in milliseconds.
	/** This value starts with 0 and can be manipulated using setTime(),
	stopTimer(), startTimer(), etc. This value depends on the set speed of
	the timer if the timer is stopped, etc. If you need the system time,
	use getRealTime() */
	virtual uint32_t getTime() const = 0;

	//! sets current virtual time
	virtual void setTime(uint32_t time) = 0;

	//! Stops the virtual timer.
	/** The timer is reference counted, which means everything which calls
	stop() will also have to call start(), otherwise the timer may not
	start/stop correctly again. */
	virtual void stop() = 0;

	//! Starts the virtual timer.
	/** The timer is reference counted, which means everything which calls
	stop() will also have to call start(), otherwise the timer may not
	start/stop correctly again. */
	virtual void start() = 0;

	//! Sets the speed of the timer
	/** The speed is the factor with which the time is running faster or
	slower then the real system time. */
	virtual void setSpeed(float speed = 1.0f) = 0;

	//! Returns current speed of the timer
	/** The speed is the factor with which the time is running faster or
	slower then the real system time. */
	virtual float getSpeed() const = 0;

	//! Returns if the virtual timer is currently stopped
	virtual bool isStopped() const = 0;

	//! Advances the virtual time
	/** Makes the virtual timer update the time value based on the real
	time. This is called automatically when calling IrrlichtDevice::run(),
	but you can call it manually if you don't use this method. */
	virtual void tick() = 0;
};

} // end namespace irr

#endif

// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_TIMER_H_INCLUDED__
#define __I_TIMER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

#include <chrono>
#include <ctime>

namespace irr
{

//! Interface for getting and manipulating the virtual time
class ITimer final : public core::IReferenceCounted
{
		using clock_type = std::chrono::high_resolution_clock;
	public:
		//! Returns current real time in milliseconds of the system.
		/** This value does not start with 0 when the application starts.
		For example in one implementation the value returned could be the
		amount of milliseconds which have elapsed since the system was started.
		*/
		static inline uint32_t getRealTime()
		{
			return static_cast<uint32_t>(getRealTime64());
		}
		static inline uint64_t getRealTime64()
		{
			return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		}

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

		static inline RealTimeDate getRealTimeAndDate()
		{
			std::time_t rawtime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

			std::tm* timeinfo = std::localtime(&rawtime);

			// init with all 0 to indicate error
			ITimer::RealTimeDate date = { 0u,0u,0u,0,0u,0u,EWD_SUNDAY,0u,false };
			// at least Windows returns NULL on some illegal dates
			if (timeinfo)
			{
				// set useful values if succeeded
				date.Hour = (uint32_t)timeinfo->tm_hour;
				date.Minute = (uint32_t)timeinfo->tm_min;
				date.Second = (uint32_t)timeinfo->tm_sec;
				date.Day = (uint32_t)timeinfo->tm_mday;
				date.Month = (uint32_t)timeinfo->tm_mon + 1;
				date.Year = (uint32_t)timeinfo->tm_year + 1900;
				date.Weekday = (ITimer::EWeekday)timeinfo->tm_wday;
				date.Yearday = (uint32_t)timeinfo->tm_yday + 1;
				date.IsDST = timeinfo->tm_isdst != 0;
			}
			return date;
		}

		ITimer() : VirtualTimerSpeed(1.f), VirtualTimerStopCounter(0)
		{
			setTime(clock_type::duration::zero());
		}

		//! Returns current virtual time in milliseconds.
		/** This value starts with 0 and can be manipulated using setTime(),
		stopTimer(), startTimer(), etc. This value depends on the set speed of
		the timer if the timer is stopped, etc. If you need the system time,
		use getRealTime() */
		inline clock_type::duration getTime() const
		{
			if (isStopped())
				return LastVirtualTime;

			std::chrono::duration<double,clock_type::duration::period> delta = StaticTime - StartRealTime;
			delta *= double(VirtualTimerSpeed);
			return LastVirtualTime + std::chrono::duration_cast<clock_type::duration>(delta);
		}

		//! sets current virtual time
		inline void setTime(const clock_type::duration& time)
		{
			tick();
			StartRealTime = StaticTime;
			LastVirtualTime = time;
		}

		//! Stops the virtual timer.
		/** The timer is reference counted, which means everything which calls
		stop() will also have to call start(), otherwise the timer may not
		start/stop correctly again. */
		inline void stop()
		{
			if (!isStopped())
			{
				// stop the virtual timer
				LastVirtualTime = getTime();
			}

			--VirtualTimerStopCounter;
		}

		//! Starts the virtual timer.
		/** The timer is reference counted, which means everything which calls
		stop() will also have to call start(), otherwise the timer may not
		start/stop correctly again. */
		inline void start()
		{
			++VirtualTimerStopCounter;

			if (!isStopped())
			{
				// restart virtual timer
				setTime(LastVirtualTime);
			}
		}

		//! Sets the speed of the timer
		/** The speed is the factor with which the time is running faster or
		slower then the real system time. */
		inline void setSpeed(float speed = 1.0f)
		{
			assert(speed >= 0.f);

			setTime(getTime());
			VirtualTimerSpeed = speed;
		}

		//! Returns current speed of the timer
		/** The speed is the factor with which the time is running faster or
		slower then the real system time. */
		inline float getSpeed() const { return VirtualTimerSpeed; }

		//! Returns if the virtual timer is currently stopped
		inline bool isStopped() const { return VirtualTimerStopCounter < 0; }

		//! Advances the virtual time
		/** Makes the virtual timer update the time value based on the real
		time. This is called automatically when calling IrrlichtDevice::run(),
		but you can call it manually if you don't use this method. */
		inline void tick() {StaticTime = clock_type::now();}
	protected:
		virtual ~ITimer() {}

		float VirtualTimerSpeed;
		int32_t VirtualTimerStopCounter;
		clock_type::time_point StaticTime;
		clock_type::time_point StartRealTime;
		clock_type::duration LastVirtualTime;
};

} // end namespace irr

#endif

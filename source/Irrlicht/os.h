// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_OS_H_INCLUDED__
#define __IRR_OS_H_INCLUDED__

#include "IrrCompileConfig.h" // for endian check
#include "irr/core/Types.h"
#include "irr/core/irrString.h"
#include "path.h"
#include "ILogger.h"
#include "ITimer.h"

namespace irr
{

namespace os
{
	class Byteswap
	{
	    Byteswap() = delete;
	public:
		static uint16_t byteswap(uint16_t num);
		static int16_t byteswap(int16_t num);
		static uint32_t byteswap(uint32_t num);
		static int32_t byteswap(int32_t num);
		static float byteswap(float num);
		// prevent accidental swapping of chars
		static uint8_t  byteswap(uint8_t  num);
		static int8_t  byteswap(int8_t  num);
	};

	class Printer
	{
	    Printer() = delete;
	public:
		// prints out a string to the console out stdout or debug log or whatever
		static void print(const std::string& message);
		static void log(const std::string& message, ELOG_LEVEL ll = ELL_INFORMATION);
		static void log(const std::wstring& message, ELOG_LEVEL ll = ELL_INFORMATION);
		static void log(const std::string& message, const std::string& hint, ELOG_LEVEL ll = ELL_INFORMATION);
		static ILogger* Logger;
	};



	class Timer
	{
	    Timer() = delete;
	public:
/*	    Timer()
	    {
            VirtualTimerSpeed = 1.0f;
            VirtualTimerStopCounter = 0;
            LastVirtualTime = 0;
            StartRealTime = 0;
            StaticTime = 0;
	    }
*/
		//! returns the current time in milliseconds
		static uint32_t getTime();

		//! get current time and date in calendar form
		static ITimer::RealTimeDate getRealTimeAndDate();

		//! initializes the real timer
		static void initTimer(bool usePerformanceTimer=true);

		//! sets the current virtual (game) time
		static void setTime(uint32_t time);

		//! stops the virtual (game) timer
		static void stopTimer();

		//! starts the game timer
		static void startTimer();

		//! sets the speed of the virtual timer
		static void setSpeed(float speed);

		//! gets the speed of the virtual timer
		static float getSpeed();

		//! returns if the timer currently is stopped
		static bool isStopped();

		//! makes the virtual timer update the time value based on the real time
		static void tick();

		//! returns the current real time in milliseconds
		static uint32_t getRealTime();
		static uint64_t getRealTime64();

	private:

		static void initVirtualTimer();

		static float VirtualTimerSpeed;
		static int32_t VirtualTimerStopCounter;
		static uint32_t StartRealTime;
		static uint32_t LastVirtualTime;
		static uint32_t StaticTime;
	};

} // end namespace os
} // end namespace irr


#endif


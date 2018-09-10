// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "os.h"

#include "IrrCompileConfig.h"
#include "irr/core/math/irrMath.h"

#if defined(_IRR_COMPILE_WITH_SDL_DEVICE_)
	#include <SDL/SDL_endian.h>
	#define bswap_16(X) SDL_Swap16(X)
	#define bswap_32(X) SDL_Swap32(X)
#elif defined(_IRR_WINDOWS_API_) && defined(_MSC_VER) && (_MSC_VER > 1298)
	#include <stdlib.h>
	#define bswap_16(X) _byteswap_ushort(X)
	#define bswap_32(X) _byteswap_ulong(X)
#if (_MSC_VER >= 1400)
	#define localtime _localtime_s
#endif
#elif defined(_IRR_OSX_PLATFORM_)
	#include <libkern/OSByteOrder.h>
	#define bswap_16(X) OSReadSwapInt16(&X,0)
	#define bswap_32(X) OSReadSwapInt32(&X,0)
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
	#include <sys/endian.h>
	#define bswap_16(X) bswap16(X)
	#define bswap_32(X) bswap32(X)
#elif !defined(_IRR_SOLARIS_PLATFORM_) && !defined(__PPC__) && !defined(_IRR_WINDOWS_API_)
	#include <byteswap.h>
#else
	#define bswap_16(X) ((((X)&0xFF) << 8) | (((X)&0xFF00) >> 8))
	#define bswap_32(X) ( (((X)&0x000000FF)<<24) | (((X)&0xFF000000) >> 24) | (((X)&0x0000FF00) << 8) | (((X) &0x00FF0000) >> 8))
#endif

namespace irr
{
namespace os
{
	uint16_t Byteswap::byteswap(uint16_t num) {return bswap_16(num);}
	int16_t Byteswap::byteswap(int16_t num) {return bswap_16(num);}
	uint32_t Byteswap::byteswap(uint32_t num) {return bswap_32(num);}
	int32_t Byteswap::byteswap(int32_t num) {return bswap_32(num);}
	float Byteswap::byteswap(float num) {uint32_t tmp=IR(num); tmp=bswap_32(tmp); return (FR(tmp));}
	// prevent accidental byte swapping of chars
	uint8_t  Byteswap::byteswap(uint8_t num)  {return num;}
	int8_t  Byteswap::byteswap(int8_t num)  {return num;}
}
}

#if defined(_IRR_WINDOWS_API_)
// ----------------------------------------------------------------
// Windows specific functions
// ----------------------------------------------------------------

#ifdef _IRR_XBOX_PLATFORM_
#include <xtl.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <time.h>
#endif

namespace irr
{
namespace os
{
	//! prints a debuginfo string
	void Printer::print(const std::string& message)
	{
#if defined (_WIN32_WCE )
		core::stringw tmp(message);
		tmp += L"\n";
		OutputDebugStringW(tmp.c_str());
#else
		std::string tmp(message);
		tmp += "\n";
		OutputDebugStringA(tmp.c_str());
		printf("%s", tmp.c_str());
#endif
	}

	static LARGE_INTEGER HighPerformanceFreq;
	static BOOL HighPerformanceTimerSupport = FALSE;
	static BOOL MultiCore = FALSE;

	void Timer::initTimer(bool usePerformanceTimer)
	{
#if !defined(_WIN32_WCE) && !defined (_IRR_XBOX_PLATFORM_)
		// workaround for hires timer on multiple core systems, bios bugs result in bad hires timers.
		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		MultiCore = (sysinfo.dwNumberOfProcessors > 1);
#endif
		if (usePerformanceTimer)
			HighPerformanceTimerSupport = QueryPerformanceFrequency(&HighPerformanceFreq);
		else
			HighPerformanceTimerSupport = FALSE;
		initVirtualTimer();
	}

	uint32_t Timer::getRealTime()
	{
		if (HighPerformanceTimerSupport)
		{
#if !defined(_WIN32_WCE) && !defined (_IRR_XBOX_PLATFORM_)
			// Avoid potential timing inaccuracies across multiple cores by
			// temporarily setting the affinity of this process to one core.
//			DWORD_PTR affinityMask=0;
//			if(MultiCore)
//				affinityMask = SetThreadAffinityMask(GetCurrentThread(), 1);
#endif
			LARGE_INTEGER nTime;
			BOOL queriedOK = QueryPerformanceCounter(&nTime);

#if !defined(_WIN32_WCE)  && !defined (_IRR_XBOX_PLATFORM_)
			// Restore the true affinity.
//			if(MultiCore)
//				(void)SetThreadAffinityMask(GetCurrentThread(), affinityMask);
#endif
			if(queriedOK)
				return uint32_t((nTime.QuadPart) * 1000 / HighPerformanceFreq.QuadPart);

		}

		return GetTickCount();
	}

	uint64_t Timer::getRealTime64()
	{
		if (HighPerformanceTimerSupport)
		{
#if !defined(_WIN32_WCE) && !defined (_IRR_XBOX_PLATFORM_)
			// Avoid potential timing inaccuracies across multiple cores by
			// temporarily setting the affinity of this process to one core.
			DWORD_PTR affinityMask=0;
//			if(MultiCore)
//				affinityMask = SetThreadAffinityMask(GetCurrentThread(), 1);
#endif
			LARGE_INTEGER nTime;
			BOOL queriedOK = QueryPerformanceCounter(&nTime);

#if !defined(_WIN32_WCE)  && !defined (_IRR_XBOX_PLATFORM_)
			// Restore the true affinity.
//			if(MultiCore)
//				(void)SetThreadAffinityMask(GetCurrentThread(), affinityMask);
#endif
			if(queriedOK)
			{
				uint64_t r = nTime.QuadPart;
				r *= 1000;
				r /= HighPerformanceFreq.QuadPart;
				return r;
//				return uint64_t((nTime.QuadPart) * 1000 / HighPerformanceFreq.QuadPart);
			}

		}

		return GetTickCount();
	}

} // end namespace os


#else

// ----------------------------------------------------------------
// linux/ansi version
// ----------------------------------------------------------------

#include <stdio.h>
#include <time.h>
#include <sys/time.h>

namespace irr
{
namespace os
{

	//! prints a debuginfo string
	void Printer::print(const std::string& message)
	{
		printf("%s\n", message.c_str());
	}

	void Timer::initTimer(bool usePerformanceTimer)
	{
		initVirtualTimer();
	}

	uint32_t Timer::getRealTime()
	{
		timeval tv;
		gettimeofday(&tv, 0);
		return (uint32_t)(tv.tv_sec * 1000) + (tv.tv_usec / 1000);
	}
	uint64_t Timer::getRealTime64()
	{
		timeval tv;
		gettimeofday(&tv, 0);
		return (uint64_t)(tv.tv_sec)*1000ull + (uint64_t)(tv.tv_usec / 1000);
	}
} // end namespace os

#endif // end linux / windows

namespace os
{
	// The platform independent implementation of the printer
	ILogger* Printer::Logger = 0;

	void Printer::log(const std::string& message, ELOG_LEVEL ll)
	{
		if (Logger)
			Logger->log(message, ll);
	}

	void Printer::log(const std::wstring& message, ELOG_LEVEL ll)
	{
		if (Logger)
			Logger->log(message, ll);
	}

	void Printer::log(const std::string& message, const std::string& hint, ELOG_LEVEL ll)
	{
		if (Logger)
			Logger->log(message, hint, ll);
	}


	// ------------------------------------------------------
	// virtual timer implementation

// this shit aint here, then win32 wont compile
	float Timer::VirtualTimerSpeed = 1.0f;
	int32_t Timer::VirtualTimerStopCounter = 0;
	uint32_t Timer::LastVirtualTime = 0;
	uint32_t Timer::StartRealTime = 0;
	uint32_t Timer::StaticTime = 0;

	//! Get real time and date in calendar form
	ITimer::RealTimeDate Timer::getRealTimeAndDate()
	{
		time_t rawtime;
		time(&rawtime);

		struct tm * timeinfo;
		timeinfo = localtime(&rawtime);

		// init with all 0 to indicate error
		ITimer::RealTimeDate date={0};
		// at least Windows returns NULL on some illegal dates
		if (timeinfo)
		{
			// set useful values if succeeded
			date.Hour=(uint32_t)timeinfo->tm_hour;
			date.Minute=(uint32_t)timeinfo->tm_min;
			date.Second=(uint32_t)timeinfo->tm_sec;
			date.Day=(uint32_t)timeinfo->tm_mday;
			date.Month=(uint32_t)timeinfo->tm_mon+1;
			date.Year=(uint32_t)timeinfo->tm_year+1900;
			date.Weekday=(ITimer::EWeekday)timeinfo->tm_wday;
			date.Yearday=(uint32_t)timeinfo->tm_yday+1;
			date.IsDST=timeinfo->tm_isdst != 0;
		}
		return date;
	}

	//! returns current virtual time
	uint32_t Timer::getTime()
	{
		if (isStopped())
			return LastVirtualTime;

		return LastVirtualTime + (uint32_t)((StaticTime - StartRealTime) * VirtualTimerSpeed);
	}

	//! ticks, advances the virtual timer
	void Timer::tick()
	{
		StaticTime = getRealTime();
	}

	//! sets the current virtual time
	void Timer::setTime(uint32_t time)
	{
		StaticTime = getRealTime();
		LastVirtualTime = time;
		StartRealTime = StaticTime;
	}

	//! stops the virtual timer
	void Timer::stopTimer()
	{
		if (!isStopped())
		{
			// stop the virtual timer
			LastVirtualTime = getTime();
		}

		--VirtualTimerStopCounter;
	}

	//! starts the virtual timer
	void Timer::startTimer()
	{
		++VirtualTimerStopCounter;

		if (!isStopped())
		{
			// restart virtual timer
			setTime(LastVirtualTime);
		}
	}

	//! sets the speed of the virtual timer
	void Timer::setSpeed(float speed)
	{
		setTime(getTime());

		VirtualTimerSpeed = speed;
		if (VirtualTimerSpeed < 0.0f)
			VirtualTimerSpeed = 0.0f;
	}

	//! gets the speed of the virtual timer
	float Timer::getSpeed()
	{
		return VirtualTimerSpeed;
	}

	//! returns if the timer currently is stopped
	bool Timer::isStopped()
	{
		return VirtualTimerStopCounter < 0;
	}

	void Timer::initVirtualTimer()
	{
		StaticTime = getRealTime();
		StartRealTime = StaticTime;
	}

} // end namespace os
} // end namespace irr



// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// Released under Apache 2.0 license
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

//#include "stdafx.h"

#include "FW_Mutex.h"
#include "os.h"

#if _MSC_VER && !__INTEL_COMPILER
//
#else
#include "errno.h"
#endif // _MSC_VER

using namespace irr;

// ---------------------------------------------------------------------------
//		* FW_Mutex
// ---------------------------------------------------------------------------

FW_Mutex::FW_Mutex()
{
	bool fail = 0;
#ifndef FW_MUTEX_H_CXX11_IMPL
#if _MSC_VER && !__INTEL_COMPILER
    //maybe have more spins? - currently 1024 before Kernel scheduler sleep
	fail = !InitializeCriticalSectionAndSpinCount(&hMutex,100);
#else
    fail = pthread_mutex_init(&hMutex, NULL) != 0;
#endif
#endif // FW_MUTEX_H_CXX11_IMPL

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("CreateMutex failed!\n",ELL_ERROR);
#endif // _IRR_DEBUG
}


// ---------------------------------------------------------------------------
//		* ~FW_Mutex
// ---------------------------------------------------------------------------

FW_Mutex::~FW_Mutex()
{
#ifndef FW_MUTEX_H_CXX11_IMPL
#if _MSC_VER && !__INTEL_COMPILER
//	if (hMutex)
		DeleteCriticalSection(&hMutex);
#else
    pthread_mutex_destroy(&hMutex);
#endif
#endif // FW_MUTEX_H_CXX11_IMPL
}





//
//
//
FW_ConditionVariable::FW_ConditionVariable(FW_Mutex *mutex)
{
#ifdef _IRR_DEBUG
    mutexAttachedTo = mutex;
#endif // _IRR_DEBUG
    bool fail = false;

#ifndef FW_MUTEX_H_CXX11_IMPL
#if _MSC_VER && !__INTEL_COMPILER
    InitializeConditionVariable(&conditionVar);
#else
    fail = pthread_cond_init(&conditionVar,NULL);
#endif // _MSC_VER
#endif // FW_MUTEX_H_CXX11_IMPL

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("FW_ConditionVariable constructor failed!\n",ELL_ERROR);
#endif
}

//
//
//
FW_ConditionVariable::~FW_ConditionVariable()
{
#if (_MSC_VER && !__INTEL_COMPILER) || defined(FW_MUTEX_H_CXX11_IMPL)
    //no need to delete cond var on windows nor using c++11
    bool fail = false;
#else
    bool fail = pthread_cond_destroy(&conditionVar);
#endif // _MSC_VER || FW_MUTEX_H_CXX11_IMPL

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("FW_ConditionVariable destructor failed!\n",ELL_ERROR);
#endif // _IRR_DEBUG
}

//
//
//
void FW_ConditionVariable::WaitForCondition(FW_Mutex *mutex)
{
#ifdef _IRR_DEBUG
    if (mutexAttachedTo!=mutex)
    {
        os::Printer::log("Tried to wait on condition bound to a different mutex!\n",ELL_ERROR);
        exit(-69);
    }
#endif // _IRR_DEBUG

	bool fail = 0;
#if defined(FW_MUTEX_H_CXX11_IMPL)
	std::unique_lock<std::mutex> ul(mutex->hMutex, std::defer_lock);
	conditionVar.wait(ul);
#elif _MSC_VER && !__INTEL_COMPILER
    fail = !SleepConditionVariableCS(&conditionVar,&mutex->hMutex,INFINITE);
#else
    fail = pthread_cond_wait(&conditionVar,&mutex->hMutex);
#endif // _MSC_VER

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("WaitForCondition system call returned error for unknown reason!\n",ELL_ERROR);
#endif // _IRR_DEBUG
}



void FW_ConditionVariable::TimedWaitForCondition(FW_Mutex *mutex, const uint64_t& nanosec)
{
#ifdef _IRR_DEBUG
    if (mutexAttachedTo!=mutex)
    {
        os::Printer::log("Tried to wait on condition bound to a different mutex!\n",ELL_ERROR);
        exit(-69);
    }
#endif // _IRR_DEBUG

#if defined(FW_MUTEX_H_CXX11_IMPL)
	std::unique_lock<std::mutex> ul(mutex->hMutex, std::defer_lock);
	bool fail = conditionVar.wait_for(ul, std::chrono::duration<uint64_t, std::nano>(nanosec)) == std::cv_status::no_timeout;
#elif _MSC_VER && !__INTEL_COMPILER
    bool fail = !SleepConditionVariableCS(&conditionVar,&mutex->hMutex,(nanosec+999999ull)/1000000ull);
	if (fail)
	{
		DWORD er = GetLastError();
		if (er == ERROR_TIMEOUT)
			fail = false;
	}
#else
    struct timespec ts;
    ts.tv_sec = (time_t) (nanosec / 1000000000ull);
    ts.tv_nsec = (long) (nanosec % 1000000000ull);

    int retval = pthread_cond_timedwait(&conditionVar,&mutex->hMutex,&ts);
    bool fail = retval&&retval!=ETIMEDOUT;
#endif // _MSC_VER

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("WaitForCondition system call returned error for unknown reason!\n",ELL_ERROR);
#endif // _IRR_DEBUG
}


//
//
//
void FW_ConditionVariable::SignalConditionOnce()
{
    bool fail = false;
#if defined(FW_MUTEX_H_CXX11_IMPL)
	conditionVar.notify_one();
#elif _MSC_VER && !__INTEL_COMPILER
    WakeConditionVariable(&conditionVar);
#else
    fail = pthread_cond_signal(&conditionVar);
#endif // _MSC_VER

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("SignalConditionOnce system call returned error for unknown reason!\n",ELL_ERROR);
#endif
}

//
//
//
void FW_ConditionVariable::SignalConditionToAll()
{
    bool fail = false;
#if defined(FW_MUTEX_H_CXX11_IMPL)
	conditionVar.notify_all();
#elif _MSC_VER && !__INTEL_COMPILER
    WakeAllConditionVariable(&conditionVar);
#else
    fail = pthread_cond_broadcast(&conditionVar);
#endif // _MSC_VER

#ifdef _IRR_DEBUG
    if (fail)
        os::Printer::log("SignalConditionToAll system call returned error for unknown reason!\n",ELL_ERROR);
#endif
}

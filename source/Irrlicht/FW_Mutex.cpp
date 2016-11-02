// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// Released under Apache 2.0 license
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#include "stdafx.h"

#include "FW_Mutex.h"
#include "os.h"

using namespace irr;

// ---------------------------------------------------------------------------
//		* FW_Mutex
// ---------------------------------------------------------------------------

FW_Mutex::FW_Mutex()
{
#if _MSC_VER && !__INTEL_COMPILER
    //maybe have more spins? - currently 1024 before Kernel scheduler sleep
	bool fail = !InitializeCriticalSectionAndSpinCount(&hMutex,100);
#else
    bool fail = pthread_mutex_init(&hMutex, NULL) != 0;
#endif

#ifdef _DEBUG
    if (fail)
        os::Printer::log("CreateMutex failed!\n",ELL_ERROR);
#endif // _DEBUG
}


// ---------------------------------------------------------------------------
//		* ~FW_Mutex
// ---------------------------------------------------------------------------

FW_Mutex::~FW_Mutex()
{
#if _MSC_VER && !__INTEL_COMPILER
//	if (hMutex)
		DeleteCriticalSection(&hMutex);
#else
    pthread_mutex_destroy(&hMutex);
#endif
}





//
//
//
FW_ConditionVariable::FW_ConditionVariable(FW_Mutex *mutex)
{
#ifdef _DEBUG
    mutexAttachedTo = mutex;
#endif // _DEBUG

#if _MSC_VER && !__INTEL_COMPILER
    InitializeConditionVariable(&conditionVar);
#else
    if (pthread_cond_init(&conditionVar,NULL))
        os::Printer::log("FW_ConditionVariable constructor failed!\n",ELL_ERROR);
#endif
}

//
//
//
FW_ConditionVariable::~FW_ConditionVariable()
{
#if _MSC_VER && !__INTEL_COMPILER
    //no need to delete cond var on windows
#else
    if (pthread_cond_destroy(&conditionVar))
        os::Printer::log("FW_ConditionVariable destructor failed!\n",ELL_ERROR);
#endif
}

//
//
//
void FW_ConditionVariable::WaitForCondition(FW_Mutex *mutex)
{
#ifdef _DEBUG
    if (mutexAttachedTo!=mutex)
    {
        os::Printer::log("Tried to wait on condition bound to a different mutex!\n",ELL_ERROR);
        exit(-69);
    }
#endif // _DEBUG

#if _MSC_VER && !__INTEL_COMPILER
    if (!SleepConditionVariableCS(&conditionVar,&mutex->hMutex,INFINITE))
#else
    if (pthread_cond_wait(&conditionVar,&mutex->hMutex))
#endif
        os::Printer::log("WaitForCondition system call returned error for unknown reason!\n",ELL_ERROR);
}




/*
//
//
bool FW_ConditionVariable::TimedWaitForCondition(FW_Mutex *mutex, uint32_t milliseconds)
{
#ifdef _DEBUG
    if (mutexAttachedTo!=mutex)
    {
        os::Printer::log("Tried to wait on condition bound to a different mutex!\n");
        exit(-1)
    }
#endif // _DEBUG

#ifndef LINUX
    if (!SleepConditionVariableCS(&conditionVar,&mutex->hMutex,INFINITE))
    {
        if (GetLastError()!=ERROR_TIMEOUT)
            os::Printer::log("TimedWaitForCondition system call returned error for unknown reason!\n");
        return false;
    }
#else
    #error "Havent figured out how to sleep for a relative time yet"

    int waitResult = pthread_cond_timedwait(&conditionVar,&mutex->hMutex,timespec);
    if (waitResult)
    {
        if (waitResult!=ETIMEDOUT)
            os::Printer::log("TimedWaitForCondition system call returned error for unknown reason!\n");
        return false;
    }
#endif

    return true;
}
*/


//
//
//
void FW_ConditionVariable::SignalConditionOnce()
{
#if _MSC_VER && !__INTEL_COMPILER
    WakeConditionVariable(&conditionVar);
#else
    if (pthread_cond_signal(&conditionVar))
        os::Printer::log("SignalConditionOnce system call returned error for unknown reason!\n",ELL_ERROR);
#endif
}

//
//
//
void FW_ConditionVariable::SignalConditionToAll()
{
#if _MSC_VER && !__INTEL_COMPILER
    WakeAllConditionVariable(&conditionVar);
#else
    if (pthread_cond_broadcast(&conditionVar))
        os::Printer::log("SignalConditionToAll system call returned error for unknown reason!\n",ELL_ERROR);
#endif
}

// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// Released under Apache 2.0 license
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672
#include "stdint.h"

#ifndef _FW_MUTEX_H_
#define _FW_MUTEX_H_

//#define FW_MUTEX_H_CXX11_IMPL

#if defined(FW_MUTEX_H_CXX11_IMPL)

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#endif


#ifdef __GNUC__
#include <pthread.h>
#endif

#if (defined(WIN32) || defined(_MSC_VER))
#include "Windows.h"
#endif


inline void FW_SleepMs(const uint64_t &milliseconds)
{
#if defined(FW_MUTEX_H_CXX11_IMPL)
	if (!milliseconds)
		std::this_thread::yield();
	else
		std::this_thread::sleep_for(std::chrono::duration<uint64_t, std::milli>(milliseconds));
#elif (defined(WIN32) || defined(_MSC_VER))
    if (!milliseconds)
        SwitchToThread();
	else
        Sleep(milliseconds);
#else
	if (!milliseconds)
		pthread_yield();
	else
	{
		struct timespec ts;
		ts.tv_sec = (time_t) (milliseconds / 1000ull);
		ts.tv_nsec = (long) (milliseconds % 1000ull) * 1000000ull;
		nanosleep(&ts, NULL);
	}
#endif
}

//nanoseconds :D
inline void FW_SleepNano(const uint64_t &nanoseconds)
{
#if defined(FW_MUTEX_H_CXX11_IMPL)
	if (!nanoseconds)
		std::this_thread::yield();
	else
		std::this_thread::sleep_for(std::chrono::duration<uint64_t, std::nano>(nanoseconds));
#elif (defined(WIN32) || defined(_MSC_VER))
    if (!nanoseconds)
        SwitchToThread();
    else
    {
        __int64 time1 = 0, time2 = 0, freq = 0;

        QueryPerformanceCounter((LARGE_INTEGER*)&time1);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        time1 -= freq/__int64(20000); //add a 50us guard

        do {
            SwitchToThread();
            QueryPerformanceCounter((LARGE_INTEGER *) &time2);
        } while((time2-time1) < nanoseconds*freq/__int64(1000000000));
    }
#else
	if (!nanoseconds)
		pthread_yield();
	else
	{
		struct timespec ts;
		ts.tv_sec = (time_t) (nanoseconds / 1000000000ull);
		ts.tv_nsec = (long) (nanoseconds % 1000000000ull);
		nanosleep(&ts, NULL);
	}
#endif
}

inline uint64_t FW_GetTimestampNs()
{
#if defined(FW_MUTEX_H_CXX11_IMPL)
	return std::chrono::steady_clock::now().time_since_epoch().count();
#elif (defined(WIN32) || defined(_MSC_VER))
    __int64 time1 = 0, freq = 0;

    QueryPerformanceCounter((LARGE_INTEGER*)&time1);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    return time1*__int64(1000000000)/freq;
#else
    struct timespec ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts1); //maybe CLOCK_PROCESS_CPUTIME_ID?
    return uint64_t(ts1.tv_sec)*1000000000ull+uint64_t(ts1.tv_nsec);
#endif
}


class FW_ConditionVariable;

class	FW_Mutex
{
public:
    FW_Mutex();
    ~FW_Mutex();

	inline  void	Get(void)
    {
	#if defined(FW_MUTEX_H_CXX11_IMPL)
		hMutex.lock();
    #elif _MSC_VER && !__INTEL_COMPILER
        EnterCriticalSection(&hMutex);
    #else
        pthread_mutex_lock(&hMutex);
    #endif
    }
	inline  void	Release(void)
    {
	#if defined(FW_MUTEX_H_CXX11_IMPL)
		hMutex.unlock();
    #elif _MSC_VER && !__INTEL_COMPILER
        LeaveCriticalSection(&hMutex);
    #else
        pthread_mutex_unlock(&hMutex);
    #endif
    }
	inline  bool    TryLock(void)
	{
	#if defined(FW_MUTEX_H_CXX11_IMPL)
		return hMutex.try_lock();
    #elif _MSC_VER && !__INTEL_COMPILER
        return TryEnterCriticalSection(&hMutex);
    #else
        return pthread_mutex_trylock(&hMutex)==0;
    #endif
    }

private:
    friend class FW_ConditionVariable;
    FW_Mutex(const FW_Mutex&); // no implementation
    FW_Mutex& operator=(const FW_Mutex&); // no implementation
#if defined(FW_MUTEX_H_CXX11_IMPL)
	std::mutex hMutex;
#elif _MSC_VER && !__INTEL_COMPILER
	CRITICAL_SECTION hMutex;
#elif defined(_PTHREAD_H)
    pthread_mutex_t hMutex;
#else
#error "No Threading Lib Used!"
#endif
};


class FW_ConditionVariable
{
    public:
        FW_ConditionVariable(FW_Mutex *mutex);
        ~FW_ConditionVariable();

        //! YOU NEED TO LOCK THE ASSOCIATED MUTEX AROUND THE WAITS
        // don't worry the mutex is atomically released during the wait and reclaimed after
        /**
        YOU NEED TO RE-CHECK THE PREDICATE (the boolean condition such as queue non-emptyness)
        BECAUSE A DIFFERENT THREAD MAY CUT INBETWEEN THE WAKEUP AND MUTEX CLAIM
        AS WELL AS THE WAIT MAY AWAKE WITHOUT SIGNAL!

        CHECK USAGE EXAMPLES of pthread_cond_wait on the internet!

        Proper use:

        mutex->Get();

        while (!predicate) //something like !queue.empty()
        {
            condition->WaitForCondition(mutex);
        }
        //now we've woken up and predicate definitely true

        mutex->Release();
        **/
        void    WaitForCondition(FW_Mutex *mutex);

        void    TimedWaitForCondition(FW_Mutex* mutex, const uint64_t& nanosec);


        /**
        You can signal a condition outside a mutex lock,
        BUT ITS ILL ADVISED. READ THE EXPLANATION (applies to Windows too)

        http://fixunix.com/linux/345366-pthread_cond_signal-mutex-lock-unlock-needed.html
        **/
        void    SignalConditionOnce();
        void    SignalConditionToAll();

    private:
        FW_ConditionVariable(const FW_ConditionVariable&); // no implementation
        FW_ConditionVariable&   operator=(const FW_ConditionVariable&); // no implementation
#ifdef _IRR_DEBUG
        FW_Mutex*      mutexAttachedTo;
#endif // _IRR_DEBUG
#if defined(FW_MUTEX_H_CXX11_IMPL)
		std::condition_variable conditionVar;
#elif _MSC_VER && !__INTEL_COMPILER
        CONDITION_VARIABLE      conditionVar;
#elif defined(_PTHREAD_H)
        pthread_cond_t conditionVar;
#else
#error "No Threading Lib Used!"
#endif
};


/**
FW_AtomicCounterBlock:
+ waits for lock to be 0 - basically not used by anyone
+ sets the counter to some magical value (unattainable by just incrementing counter)
not using add because we atomically wait for value to be 0 before swapping
**/
#define FW_AtomicCounterMagicBlockVal 0x1000000

#if defined(FW_MUTEX_H_CXX11_IMPL)

#define FW_FastLock(lock) std::atomic_int lock; std::atomic_init(&lock, 0)
#define FW_FastLockGet(lock) { int zero = 0;\
	while (!lock.compare_exchange_weak(zero, 1, std::memory_order_acq_rel)) { \
	zero = 0;\
	std::this_thread::yield();\
	}\
}
#define FW_FastLockRelease(lock) lock.store(0, std::memory_order_release)

#define FW_AtomicCounter std::atomic_int

//! DO WE NEED std::memory_order_seq_cst on THESE? (would really like to avoid)
inline void FW_AtomicCounterIncr(FW_AtomicCounter &lock)
{
	if (lock.fetch_add(1, std::memory_order_acq_rel) >= FW_AtomicCounterMagicBlockVal)
    {
		while (lock >= FW_AtomicCounterMagicBlockVal)
        {
            std::this_thread::yield();
        }
    }
}
inline void FW_AtomicCounterDecr(FW_AtomicCounter &lock)
{
	lock.fetch_sub(1, std::memory_order_acq_rel);
}
inline void FW_AtomicCounterBlock(FW_AtomicCounter &lock)
{
	int32_t zero = 0;

	for (size_t i = 0; (!lock.compare_exchange_weak(zero, FW_AtomicCounterMagicBlockVal, std::memory_order_acq_rel) && i < 512); ++i)
		zero = 0;

	while (!lock.compare_exchange_weak(zero, FW_AtomicCounterMagicBlockVal, std::memory_order_acq_rel))
	{
		zero = 0;
		std::this_thread::yield();
	}
}
inline void FW_AtomicCounterDecrBlock(FW_AtomicCounter &lock)
{
	//to make sure we get lock first, and dont keep waiting forever because every read op has priority
	//but we check if someone else has a read or write block, in that case we're not the only ones having the lock
	//so lock value is more than the intended FW_AtomicCounterMagicBlockVal
	if (lock.fetch_add(FW_AtomicCounterMagicBlockVal - 1, std::memory_order_acq_rel) > 1)
	{
		//someone has a read or write block before we tried to swap access types
		//so we release our lock completely and wait
		lock.fetch_sub(FW_AtomicCounterMagicBlockVal, std::memory_order_relaxed);
		//we now own no locks and wait for a free gap
		FW_AtomicCounterBlock(lock);
	}
}
inline void FW_AtomicCounterUnBlock(FW_AtomicCounter &lock)
{
	lock.fetch_sub(FW_AtomicCounterMagicBlockVal, std::memory_order_acq_rel);
}
inline void FW_AtomicCounterUnBlockIncr(FW_AtomicCounter &lock)
{
	//we had the lock value at >=FW_AtomicCounterMagicBlockVal
	//no other write thread could have gotten out of FW_AtomicCounter***Block()
	//because we got there before and value of lock was never 0 between this thread
	//grabbing the write lock and now - trying to release it
	//so we don't need to wait for write lock to be released to grab the read lock
	lock.fetch_sub(FW_AtomicCounterMagicBlockVal-1, std::memory_order_acq_rel);
}

#elif _MSC_VER && !__INTEL_COMPILER

#define FW_FastLock(lock) volatile long lock = 0
#define FW_FastLockGet(lock) while(InterlockedCompareExchange(&lock, 1, 0)) \
SwitchToThread()
#define FW_FastLockRelease(lock) InterlockedExchange(&lock, 0)


#define FW_AtomicCounter volatile long
inline void FW_AtomicCounterIncr(FW_AtomicCounter &lock)
{
    //potential problem: so lets say a thread already has a write lock
    //and a second thread is waiting to get a write lock (waiting for value to be 0 - now fixed)
    //and this increments by 1 anyway and waits for write lock to be released by the first thread
    //well really we have no problem because the first thread will unblock no problem (no wait loop there)
    //then all these threads will stop waiting and either decrement (no problem) or wait for write block
    //after all the previous read threads change state to either decremented or write block wait
    //anyhow the value of the counter will fall to 0 (value is decremented anyway in *DecrBlock)
    //either the original second thread will grab the write lock or one of the previous read threads which called DecrBlock
	if (InterlockedIncrement(&lock)>FW_AtomicCounterMagicBlockVal)
	{
		while (lock>=FW_AtomicCounterMagicBlockVal)
		{
			SwitchToThread();
		}
	}

    //! Alternative possibly safer version
    /*
	tryReadLockAgain:
	if (InterlockedIncrement(&lock)>=FW_AtomicCounterMagicBlockVal)
	{
        InterlockedDecrement(&lock);

        //a loop
        checkLock:
        if (lock>=FW_AtomicCounterMagicBlockVal)
        {
			Sleep(0);
            goto checkLock;
        }
        else
            goto tryReadLockAgain;
	}*/
}
inline void FW_AtomicCounterDecr(FW_AtomicCounter &lock)
{
    InterlockedDecrement(&lock);
}
inline void FW_AtomicCounterBlock(FW_AtomicCounter &lock)
{
    while(InterlockedCompareExchange(&lock, FW_AtomicCounterMagicBlockVal, 0))
    {
        SwitchToThread();
    }
}

//! can loose the lock and have another thread cut in the middle of this function
inline void FW_AtomicCounterDecrBlock(FW_AtomicCounter &lock)
{
    //to make sure we get lock first, and dont keep waiting forever because every read op has priority
    //but we check if someone else has a read or write block, in that case we're not the only ones having the lock
    //so lock value is more than the intended FW_AtomicCounterMagicBlockVal
    if (InterlockedExchangeAdd(&lock,long(FW_AtomicCounterMagicBlockVal-1))>1)
    {
        //someone has a read or write block before we tried to swap access types
        //so we release our lock completely and wait
        InterlockedExchangeAdd(&lock,long(-FW_AtomicCounterMagicBlockVal));
        //we now own no locks and wait for a free gap
        FW_AtomicCounterBlock(lock);
    }
}
inline void FW_AtomicCounterUnBlock(FW_AtomicCounter &lock)
{
	InterlockedExchangeAdd(&lock,long(-FW_AtomicCounterMagicBlockVal));
}
inline void FW_AtomicCounterUnBlockIncr(FW_AtomicCounter &lock)
{
    //we had the lock value at >=FW_AtomicCounterMagicBlockVal
    //no other write thread could have gotten out of FW_AtomicCounter***Block()
    //because we got there before and value of lock was never 0 between this thread
    //grabbing the write lock and now - trying to release it
    //so we don't need to wait for write lock to be released to grab the read lock
	InterlockedExchangeAdd(&lock,long(1-FW_AtomicCounterMagicBlockVal));
}

#elif defined(__GNUC__)

#define FW_FastLock(lock) volatile int32_t lock = 0
#define FW_FastLockGet(lock) while(__sync_val_compare_and_swap(&lock, 0, 1)) \
pthread_yield()
#define FW_FastLockRelease(lock) __sync_fetch_and_add(&lock,int32_t(-1));


#define FW_AtomicCounter volatile int32_t
inline void FW_AtomicCounterIncr(FW_AtomicCounter &lock)
{
    //potential problem: so lets say a thread already has a write lock
    //and a second thread is waiting to get a write lock (waiting for value to be 0)
    //and this increments by 1 anyway and waits for write lock to be released by the first thread
    //well really we have no problem because the first thread will unblock no problem
    //then all these threads will stop waiting and either decrement (no problem) or wait for write block
    //after all the previous read threads change state to either decremented or write block wait
    //anyhow the value of the counter will fall to 0 (value is decremented anyway in *DecrBlock)
    //either the original second thread will grab the write lock or one of the previous read threads which called DecrBlock
	if (__sync_fetch_and_add(&lock,int32_t(1))>=FW_AtomicCounterMagicBlockVal) // value before incr being compared on Linux
	{
        while (lock>=FW_AtomicCounterMagicBlockVal)
        {
            pthread_yield();
        }
	}

    //! Alternative possibly safer version
	///tryReadLockAgain:
	///if (__sync_fetch_and_add(&lock,int32_t(1))>=FW_AtomicCounterMagicBlockVal) // value before incr being compared on Linux
	///{
        ///__sync_fetch_and_add(&lock,int32_t(-1));

        //a loop
        ///checkLock:
        ///if (lock>=FW_AtomicCounterMagicBlockVal)
        ///{
            ///pthread_yield();
            ///goto checkLock;
        ///}
        ///else
            ///goto tryReadLockAgain;
	///}
}
inline void FW_AtomicCounterDecr(FW_AtomicCounter &lock)
{
    __sync_fetch_and_add(&lock,int32_t(-1));
}
inline void FW_AtomicCounterBlock(FW_AtomicCounter &lock)
{
    while(__sync_val_compare_and_swap(&lock, 0, FW_AtomicCounterMagicBlockVal))
    {
        pthread_yield();
    }
}

//! can loose the lock and have another thread cut in the middle of this function
//can turn this into a bool function that returns if lock was not lost during upgrade
inline void FW_AtomicCounterDecrBlock(FW_AtomicCounter &lock)
{
    //to make sure we get lock first, and dont keep waiting forever because every read op has priority
    //but we check if someone else has a read or write block, in that case we're not the only ones having the lock
    //so lock value is more than the intended FW_AtomicCounterMagicBlockVal
    if (__sync_fetch_and_add(&lock,int32_t(FW_AtomicCounterMagicBlockVal-1))>1) // value before incr being compared on Linux
    {
        //someone has a read or write block before we tried to swap access types
        //so we release our lock completely and wait
        __sync_fetch_and_add(&lock,int32_t(-FW_AtomicCounterMagicBlockVal));
        //we now own no locks and wait for a free gap
        FW_AtomicCounterBlock(lock);

        ///return false;
    }

    ///return true;
}
inline void FW_AtomicCounterUnBlock(FW_AtomicCounter &lock)
{
    __sync_fetch_and_add(&lock,int32_t(-FW_AtomicCounterMagicBlockVal));
}
inline void FW_AtomicCounterUnBlockIncr(FW_AtomicCounter &lock)
{
    //we had the lock value at >=FW_AtomicCounterMagicBlockVal
    //no other write thread could have gotten out of FW_AtomicCounter***Block()
    //because we got there before and value of lock was never 0 between this thread
    //grabbing the write lock and now - trying to release it
    //so we don't need to wait for write lock to be released to grab the read lock
    __sync_fetch_and_add(&lock,int32_t(1-FW_AtomicCounterMagicBlockVal));
}

#else
#error "Atomic Counters NOT SUPPORTED on target compiler/platform. Can't build User-Space Readers Writers Locks!"
#endif

#define FW_AtomicCounterInit(lock) FW_AtomicCounter lock = 0;
#if defined(FW_MUTEX_H_CXX11_IMPL)
    #undef FW_AtomicCounterInit
#endif // defined


#endif // _FW_MUTEX_H_

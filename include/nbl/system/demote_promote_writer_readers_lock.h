#ifndef __NBL_DEMOTE_PROMOTE_WRITER_READERS_LOCK_H_INCLUDED__
#define __NBL_DEMOTE_PROMOTE_WRITER_READERS_LOCK_H_INCLUDED__

#include <atomic>
#include <thread>
#include <mutex> // for std::adopt_lock_t

// TODO: Reset spins if have to yield?
// TODO atomic_unsigned_lock_free? since C++20 (from SReadWriteSpinLock)

namespace nbl::system
{

class demote_promote_writer_readers_lock
{
	constexpr static inline uint32_t MaxActors = 1023;
	constexpr static inline uint32_t SpinsBeforeYield = 5000;
	struct StateSemantics
	{
		explicit inline operator uint32_t() const { return std::bit_cast<uint32_t>(*this); }

		uint32_t currentReaders : 10 = 0;
		uint32_t pendingWriters : 10 = 0;
		uint32_t pendingUpgrades : 10 = 0;
		uint32_t writing : 1 = 0;
		uint32_t stateLocked : 1 = 0; 
	};
	std::atomic_uint32_t state;
	
	constexpr uint32_t flipLock = StateSemantics{ .currentReaders = 0,.pendingWriters = 0,.pendingUpgrades = 0,.writing = false,.stateLocked = true };
	constexpr uint32_t writingMask = StateSemantics{ .currentReaders = 0,.pendingWriters = 0,.pendingUpgrades = 0,.writing = true,.stateLocked = false };

	/**
	* @brief Acquires lock for reading. This is the operation with the lowest priority, thread will be blocked until there are no writers and no readers waiting for a writer upgrade
	*/
	void read_lock()
	{
		constexpr uint32_t preemptedMask = StateSemantics{ .currentReaders = 0,.pendingWriters = MaxActors,.pendingUpgrades = MaxActors,.writing = true,.stateLocked = false };
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			uint32_t actual = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (actual & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// If there are any pending writers, upgrades or a thread is currently writing, cannot acquire lock (`preemptedMask`)
			// Must release flipLock and yield for other threads to progress
			if (actual & preemptedMask)
			{
				state.store(actual);
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			actual += static_cast<uint32_t>(StateSemantics{ .currentReaders = 1 });
			// release the flipLock, increment currentReaders 
			state.store(actual);
			reading = true;
			break;
		}
	}

	/**
	* @brief Release lock after reading. 
	*/
	void read_unlock()
	{
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			uint32_t actual = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (actual & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			actual -= static_cast<uint32_t>(StateSemantics{ .currentReaders = 1 });
			// release the flipLock, decrement currentReaders
			state.store(actual);
			break;
		}
	}

	/**
	* @brief Acquires lock for writing. Is superseded by reader upgrades: if any thread with a reader's lock is trying to upgrade, then this thread will be blocked until that reader becomes a writer
	*/
	void write_lock()
	{
		constexpr uint32_t preemptedMask = StateSemantics{ .currentReaders = MaxActors,.pendingWriters = 0,.pendingUpgrades = MaxActors,.writing = true,.stateLocked = false };
		bool registeredPending = false;
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			uint32_t actual = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (actual & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// If there are any pending upgrades or a thread is currently reading or writing, cannot acquire lock (`preemptedMask`), so release flipLock and yield for other threads to progress
			// Also registers this thread as a pending writer if not done so already
			if (actual & preemptedMask)
			{
				if (!registeredPending)
				{
					registeredPending = true;
					actual += static_cast<uint32_t>(StateSemantics{ .pendingWriters = 1 });
				}
				state.store(actual);
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			if (registeredPending)
				actual -= static_cast<uint32_t>(StateSemantics{ .pendingWriters = 1 });
			// release the flipLock, declare that this thread will be writing
			actual |= writingMask;
			state.store(actual);
			break;
		}
	}

	/**
	* @brief Releases lock for writing.
	*/
	void write_unlock()
	{
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			uint32_t actual = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (actual & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			actual &= ~writingMask;
			// release the flipLock, decrement currentReaders
			state.store(actual);
			break;
		}
	}

	/**
	* @brief Upgrades a thread with a reader's lock to a writer's lock. If a thread already has a reader's lock, use this instead of `read_unlock()` -> `write_lock()` 
	*/
	void upgrade()
	{
		bool registeredPending = false;
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			uint32_t actual = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (actual & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// If no one's writing, take lock and release the flipLock
			if (!(actual & writingMask))
			{
				actual |= writingMask;
				// If had been registered earlier
				if (registeredPending)
				{
					actual -= static_cast<uint32_t>(StateSemantics{ .pendingUpgrades = 1 });
				}
				state.store(actual);
				break;
			}
			// Register pending if not done so earlier
			if (!registeredPending)
			{
				registeredPending = true;
				actual += static_cast<uint32_t>(StateSemantics{ .pendingUpgrades = 1 });
			}
			// Release flipLock
			state.store(actual);
			if (spinCount > SpinsBeforeYield)
				std::this_thread::yield();
			}
		}
	}

	/**
	* @brief Downgrades a thread with a writer's lock to a reader's lock. If a thread already has a writer's lock, use this instead of `write_unlock()` -> `read_lock()`
	*/
	void downgrade()
	{
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			uint32_t actual = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (actual & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			actual &= ~writingMask;
			actual += static_cast<uint32_t>(StateSemantics{ .currentReaders = 1 });
			// Release flipLock, declare that we're no longer writing and that we're reading
			state.store(actual);
		}
	}
};

}

#endif

#ifndef __NBL_DEMOTE_PROMOTE_WRITER_READERS_LOCK_H_INCLUDED__
#define __NBL_DEMOTE_PROMOTE_WRITER_READERS_LOCK_H_INCLUDED__

#include <thread>

// TODO: Bring back proper memory semantics on fetch/store

namespace nbl::system
{

// Ugly workaround, structs with constevals can't be nested
namespace impl
{
	template<typename U>
	struct DPWRLstate_lock
	{
		using type = std::atomic<U>;
	};

	// will fail if `atomic_unsiged_lock_free` doesn't exist or its value type has wrong size
	template<typename U> requires (sizeof(std::atomic_unsigned_lock_free::value_type) >= sizeof(U))
	struct DPWRLstate_lock<U>
	{
		using type = std::atomic_unsigned_lock_free;
	};

	constexpr inline DPWRLstate_lock<uint32_t>::type::value_type DPWRLMaxActors = 1023;

	struct DPWRLStateSemantics
	{
		using state_lock_value_t = typename DPWRLstate_lock<uint32_t>::type::value_type;
		uint32_t currentReaders : 10 = 0;
		uint32_t pendingWriters : 10 = 0;
		uint32_t pendingUpgrades : 10 = 0;
		uint32_t writing : 1 = 0;
		uint32_t stateLocked : 1 = 0;

		consteval inline operator state_lock_value_t() const
		{ 
			assert(currentReaders <= DPWRLMaxActors && pendingWriters <= DPWRLMaxActors && pendingUpgrades <= DPWRLMaxActors);
			return static_cast<state_lock_value_t>((currentReaders << 22) | (pendingWriters << 12) | (pendingUpgrades << 2) | (writing << 1) | stateLocked);
		}
	};
} //namespace impl

class demote_promote_writer_readers_lock
{
public:
	using state_lock_t = impl::DPWRLstate_lock<uint32_t>::type;
	using state_lock_value_t = impl::DPWRLStateSemantics::state_lock_value_t;
	// Limit on how many threads can be launched concurrently that try to read/write from/to the same file
	constexpr static inline state_lock_value_t MaxActors = impl::DPWRLMaxActors;
	
	friend class dp_read_lock_guard;
	friend class dp_write_lock_guard;

	/**
	* @brief Acquires lock for reading. This thread will be blocked until there are no writers (writing or pending) and no readers waiting for a writer upgrade
	*/
	void read_lock()
	{
		constexpr state_lock_value_t preemptedMask = pendingWritersMask | pendingUpgradesMask | writingMask;
		for (state_lock_value_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// If there are any pending writers, upgrades or a thread is currently writing, cannot acquire lock (`preemptedMask`)
			// Must release flipLock and yield for other threads to progress
			state_lock_value_t newState = oldState;
			if (newState & preemptedMask)
			{
				state.store(newState);
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// release the flipLock, increment currentReaders 
			newState += impl::DPWRLStateSemantics{ .currentReaders = 1,.pendingWriters = 0,.pendingUpgrades = 0,.writing = false,.stateLocked = false };
			state.store(newState);
			break;
		}
	}

	/**
	* @brief Release lock after reading. 
	*/
	void read_unlock()
	{
		for (state_lock_value_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			
			// Sanity check: ensure no one's writing if we had a reader's lock
			assert(!(oldState & writingMask));
			// Sanity check: if this thread is reading, then `currentReaders` can't be 0
			assert(oldState & currentReadersMask);
			
			// release the flipLock, decrement currentReaders
			state_lock_value_t newState = oldState - impl::DPWRLStateSemantics{ .currentReaders = 1, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
			state.store(newState);
			break;
		}
	}

	/**
	* @brief Acquires lock for writing. This thread will be blocked until all previous readers and writers (prior to the first state lock acquire) 
	* release their locks, but it will preempt any further readers from acquiring lock between first state lock acquire and releasing the lock it will 
	* acquire when this method returns
	*/
	void write_lock()
	{
		constexpr state_lock_value_t preemptedMask = currentReadersMask | pendingUpgradesMask | writingMask;
		bool registeredPending = false;
		for (state_lock_value_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// If there are any pending upgrades or a thread is currently reading or writing, cannot acquire lock (`preemptedMask`), so release flipLock and yield for other threads to progress
			// Also registers this thread as a pending writer if not done so already
			state_lock_value_t newState = oldState;
			if (newState & preemptedMask)
			{
				if (!registeredPending)
				{
					registeredPending = true;
					newState += impl::DPWRLStateSemantics{ .currentReaders = 0,.pendingWriters = 1,.pendingUpgrades = 0,.writing = false,.stateLocked = false };
				}
				state.store(newState);
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			// release the flipLock, declare that this thread will be writing, mark that this thread is no longer pending IF had been registered as such
			if (registeredPending)
				newState -= impl::DPWRLStateSemantics{ .currentReaders = 0,.pendingWriters = 1, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
			newState |= writingMask;
			state.store(newState);
			break;
		}
	}

	/**
	* @brief Releases lock for writing.
	*/
	void write_unlock()
	{
		for (state_lock_value_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}

			// Sanity check: writing flag is actually on
			assert(oldState & writingMask);
			//Sanity check: no one should be reading if this thread was just writing
			assert(!(oldState & currentReadersMask));

			// release the flipLock, decrement currentReaders
			state_lock_value_t newState = oldState & ~writingMask;
			state.store(newState);
			break;
		}
	}

	/**
	* @brief Upgrades a thread with a reader's lock to a writer's lock. If a thread already has a reader's lock, use this instead of `read_unlock()` -> `write_lock()`
	*        This thread will be blocked until there are no more readers and there's no thread currently writing. It will also preempt new readers and writers from 
	*        acquiring the lock.
	*/
	void upgrade()
	{
		constexpr state_lock_value_t preemptedMask = currentReadersMask;
		bool registeredPending = false;
		for (state_lock_value_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}

			// Sanity check: ensure no one's writing if we had a reader's lock
			assert(!(oldState & writingMask));
			// Sanity check: if this thread is reading, then `currentReaders` can't be 0
			assert(oldState & currentReadersMask);

			state_lock_value_t nextState = oldState;
			// If there are threads currently reading or writing, must release flipLock for other threads to progress
			if (nextState & preemptedMask)
			{
				// Register pending if not done so earlier
				if (!registeredPending)
				{
					registeredPending = true;
					nextState += impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 1, .writing = false, .stateLocked = false };
				}
				// Release flipLock
				state.store(nextState);
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
			}
			// Release flipLock, declare that this thread is writing, mark that this thread is no longer pending IF had been registered as such
			if (registeredPending)
				nextState -= impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 1, .writing = false, .stateLocked = false };
			nextState |= writingMask;
			state.store(nextState);
			break;
		}
	}

	/**
	* @brief Downgrades a thread with a writer's lock to a reader's lock. If a thread already has a writer's lock, use this instead of `write_unlock()` -> `read_lock()`
	*/
	void downgrade()
	{
		for (state_lock_value_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			// stateLocked: some other thread is doing work and expects to update state
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}

			// Sanity check: writing flag is actually on
			assert(oldState & writingMask);
			//Sanity check: no one should be reading if this thread was just writing
			assert(!(oldState & currentReadersMask));

			// Release flipLock, declare that this thread is no longer writing and that it's reading
			state_lock_value_t nextState = oldState & ~writingMask;
			nextState += impl::DPWRLStateSemantics{ .currentReaders = 1, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
			state.store(nextState);
			break;
		}
	}

private:
	state_lock_t state;

	constexpr static inline state_lock_value_t flipLock = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = true };
	constexpr static inline state_lock_value_t writingMask = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 0, .writing = true, .stateLocked = false };
	constexpr static inline state_lock_value_t pendingUpgradesMask = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = MaxActors, .writing = false, .stateLocked = false };
	constexpr static inline state_lock_value_t pendingWritersMask = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = MaxActors, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
	constexpr static inline state_lock_value_t currentReadersMask = impl::DPWRLStateSemantics{ .currentReaders = MaxActors, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
	constexpr static inline state_lock_value_t SpinsBeforeYield = 5000;
};

namespace impl
{
	class dpwr_lock_guard_base
	{
		dpwr_lock_guard_base() : m_lock(nullptr) {}

	public:
		dpwr_lock_guard_base& operator=(const dpwr_lock_guard_base&) = delete;
		dpwr_lock_guard_base(const dpwr_lock_guard_base&) = delete;

		dpwr_lock_guard_base& operator=(dpwr_lock_guard_base&& rhs) noexcept
		{
			m_lock = rhs.m_lock;
			rhs.m_lock = nullptr;
			return *this;
		}
		dpwr_lock_guard_base(dpwr_lock_guard_base&& rhs) noexcept : dpwr_lock_guard_base()
		{
			operator=(std::move(rhs));
		}

	protected:
		dpwr_lock_guard_base(demote_promote_writer_readers_lock& lk) noexcept : m_lock(&lk) {}

		demote_promote_writer_readers_lock* m_lock;
	};
} // namespace impl

class dp_read_lock_guard : public impl::dpwr_lock_guard_base
{
public:
	dp_read_lock_guard(demote_promote_writer_readers_lock& lk, std::adopt_lock_t) : impl::dpwr_lock_guard_base(lk) {}
	explicit dp_read_lock_guard(demote_promote_writer_readers_lock& lk) : dp_read_lock_guard(lk, std::adopt_lock_t())
	{
		m_lock->read_lock();
	}
	explicit dp_read_lock_guard(dp_write_lock_guard&& wl);

	~dp_read_lock_guard()
	{
		if (m_lock)
			m_lock->read_unlock();
	}
};

class dp_write_lock_guard : public impl::dpwr_lock_guard_base
{
public:
	dp_write_lock_guard(demote_promote_writer_readers_lock& lk, std::adopt_lock_t) : impl::dpwr_lock_guard_base(lk) {}
	explicit dp_write_lock_guard(demote_promote_writer_readers_lock& lk) : dp_write_lock_guard(lk, std::adopt_lock_t())
	{
		m_lock->write_lock();
	}
	explicit dp_write_lock_guard(dp_read_lock_guard&& rl);

	~dp_write_lock_guard()
	{
		if (m_lock)
			m_lock->write_unlock();
	}
};

inline explicit dp_read_lock_guard::dp_read_lock_guard(dp_write_lock_guard&& wl) : impl::dpwr_lock_guard_base(std::move(wl))
{
	m_lock->downgrade();
}

inline explicit dp_write_lock_guard::dp_write_lock_guard(dp_read_lock_guard&& rl) : impl::dpwr_lock_guard_base(std::move(rl))
{
	m_lock->upgrade();
}

} // namespace nbl::system

#endif

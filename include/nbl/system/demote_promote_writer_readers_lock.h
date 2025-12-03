#ifndef __NBL_DEMOTE_PROMOTE_WRITER_READERS_LOCK_H_INCLUDED__
#define __NBL_DEMOTE_PROMOTE_WRITER_READERS_LOCK_H_INCLUDED__

#include <thread>
#include <concepts>

// TODO: Bring back proper memory semantics on fetch/store
// TODO: CRTP/F-Bound on `perform_under_locked_state`?

namespace nbl::system
{

// Ugly workaround, structs with constevals can't be nested
namespace impl
{

template<typename U>
struct DPWRLStateLock
{
	using type = std::atomic<U>;
};

// will fail if `atomic_unsiged_lock_free` doesn't exist or its value type has wrong size
template<typename U> requires (sizeof(std::atomic_unsigned_lock_free::value_type) >= sizeof(U))
struct DPWRLStateLock<U>
{
	using type = std::atomic_unsigned_lock_free;
};

struct DPWRLStateSemantics
{
	using state_lock_value_t = typename DPWRLStateLock<uint32_t>::type::value_type;
	uint32_t currentReaders : 10 = 0;
	uint32_t pendingWriters : 10 = 0;
	uint32_t pendingUpgrades : 10 = 0;
	uint32_t writing : 1 = 0;
	uint32_t stateLocked : 1 = 0;

	consteval inline operator state_lock_value_t() const
	{ 
		return static_cast<state_lock_value_t>((currentReaders << 22) | (pendingWriters << 12) | (pendingUpgrades << 2) | (writing << 1) | stateLocked);
	}
};

enum DPWR_LOCK_DEBUG_STAGE
{
	// To ensure correctness during debug, the callback at `BEFORE_STATE_UPDATE` MUST make the virtual thread that called it run again. This is so that during debug
	// only the virtual thread holding the flipLock progresses
	BEFORE_STATE_UPDATE,
	AFTER_STATE_UPDATE,
	PREEMPTED
};

struct DPWRLVoidDebugCallback
{
};

template<typename T>
/**
* @brief A valid debug callback overrides `void operator()(DPWR_LOCK_DEBUG_STAGE stage)`
*/
concept DPWRLDebugCallback = requires(T t, DPWR_LOCK_DEBUG_STAGE stage) {
	{ t(stage) } -> std::same_as<void>;
} || std::is_same_v<T, DPWRLVoidDebugCallback>;

} //namespace impl



template <impl::DPWRLDebugCallback DebugCallback = impl::DPWRLVoidDebugCallback>
/**
* @brief By default it has no debug callback. You can provide a debug callback for every stage of the lock's loop (see `perform_under_locked_state` and `DPWR_LOCK_DEBUG_STAGE`)
*        as a functional struct that overloads `void operator(DPWR_LOCK_DEBUG_STAGE)`
*/
class demote_promote_writer_readers_lock_debug
{
public:
	using state_lock_t = impl::DPWRLStateLock<uint32_t>::type;
	using state_lock_value_t = impl::DPWRLStateSemantics::state_lock_value_t;
	// Limit on how many threads can be launched concurrently that try to read/write from/to the resource behind this lock
	constexpr static inline state_lock_value_t MaxActors = 1023;
	
	template <impl::DPWRLDebugCallback>
	friend class dpwr_read_lock_guard_debug;
	template <impl::DPWRLDebugCallback>
	friend class dpwr_write_lock_guard_debug;

	/**
	* @brief Acquires lock for reading. This thread will be blocked until there are no writers (writing or pending) and no readers waiting for a writer upgrade.
	*		 It will preempt any further readers from acquiring lock between first state lock acquire and releasing the lock it will acquire when this method returns
	*/
	void read_lock()
	{
		const auto success = [](const state_lock_value_t oldState) -> state_lock_value_t
			{
				// increase currentReaders 
				return oldState + impl::DPWRLStateSemantics{ .currentReaders = 1,.pendingWriters = 0,.pendingUpgrades = 0,.writing = false,.stateLocked = false };
			};

		const auto preemptionCheck = [](const state_lock_value_t oldState) -> bool 
			{
				// Can't acquire lock if there's threads writing or otherwise waiting to write
				constexpr state_lock_value_t preemptedMask = pendingWritersMask | pendingUpgradesMask | writingMask;
				return oldState & preemptedMask;
			};

		perform_under_locked_state(success, preemptionCheck);
	}

	/**
	* @brief Release lock after reading. 
	*/
	void read_unlock()
	{
		const auto success = [](const state_lock_value_t oldState) -> state_lock_value_t
			{
				// decrease currentReaders 
				return oldState - impl::DPWRLStateSemantics{ .currentReaders = 1,.pendingWriters = 0,.pendingUpgrades = 0,.writing = false,.stateLocked = false };
			};

		const auto sanityChecks = [](const state_lock_value_t oldState) -> bool
			{
				// If this thread is reading, then `currentReaders` can't be 0
				// If we had a reader's lock, then no thread can be writing
				return (oldState & currentReadersMask) && !(oldState & writingMask);
			};

		perform_under_locked_state(success, DefaultPreemptionCheck(), sanityChecks);
	}

	/**
	* @brief Acquires lock for writing. This thread will be blocked until all previous readers and writers (prior to the first state lock acquire) 
	* release their locks, but it will preempt any further readers from acquiring lock between first state lock acquire and releasing the lock it will 
	* acquire when this method returns
	*/
	void write_lock()
	{
		bool registeredPending = false;

		const auto success = [&](const state_lock_value_t oldState) -> state_lock_value_t
			{
				// Declare this thread as writer
				state_lock_value_t newState = oldState | writingMask;
				// If this thread had been registered as a pending writer, unregister
				if (registeredPending)
					newState -= impl::DPWRLStateSemantics{ .currentReaders = 0,.pendingWriters = 1, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
				return newState;
			};

		const auto preemptionCheck = [](const state_lock_value_t oldState) -> bool
			{
				// Can't acquire lock if there's threads still reading, writing, or waiting to upgrade
				constexpr state_lock_value_t preemptedMask = currentReadersMask | pendingUpgradesMask | writingMask;
				return oldState & preemptedMask;
			};

		const auto preempted = [&](const state_lock_value_t oldState) -> state_lock_value_t
			{
				state_lock_value_t preemptedState = oldState;
				// Register this thread as a pending writer if it hadn't done so already
				if (!registeredPending)
				{
					registeredPending = true;
					preemptedState += impl::DPWRLStateSemantics{ .currentReaders = 0,.pendingWriters = 1,.pendingUpgrades = 0,.writing = false,.stateLocked = false };
				}
				return preemptedState;
			};

		perform_under_locked_state(success, preemptionCheck, defaultSanityChecks, preempted);
	}

	/**
	* @brief Releases lock for writing.
	*/
	void write_unlock()
	{
		const auto success = [](const state_lock_value_t oldState) -> state_lock_value_t
			{
				// Declare that this thread is no longer writing
				return oldState & ~writingMask;
			};

		const auto sanityChecks = [](const state_lock_value_t oldState) -> bool
			{
				// If this thread was writing, then the writing mask must have been set to 1 when we acquired the state lock
				// If this thread was writing, then there can't be any threads currently reading
				return (oldState & writingMask) && !(oldState & currentReadersMask);
			};

		perform_under_locked_state(success, DefaultPreemptionCheck(), sanityChecks);
	}

	/**
	* @brief Upgrades a thread with a reader's lock to a writer's lock. If a thread already has a reader's lock, use this instead of `read_unlock()` -> `write_lock()`
	*        This thread will be blocked until there are no more readers and there's no thread currently writing. It will also preempt new readers and writers from 
	*        acquiring the lock.
	*/
	void upgrade()
	{
		bool registeredPending = false;

		const auto success = [&](const state_lock_value_t oldState) -> state_lock_value_t
			{
				// Declare this thread as writer
				state_lock_value_t newState = oldState | writingMask;
				// If this thread had been registered as a pending upgrade, unregister
				if (registeredPending)
					newState -= impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 1, .writing = false, .stateLocked = false };
				// If not, then we didn't unregister as a reader yet, so do that as well
				else
					newState -= impl::DPWRLStateSemantics{ .currentReaders = 1, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
				return newState;
			};

		const auto preemptionCheck = [&](const state_lock_value_t oldState) -> bool
			{
				// If there's another reader, can't upgrade
				return (oldState & currentReadersMask) > (state_lock_value_t(registeredPending ? 0 : 1) << 22);
			};

		const auto sanityChecks = [&](const state_lock_value_t oldState) -> bool
			{
				// If we had a reader's lock, no thread should be writing
				// Also if this thread has not registered as a pending upgrade, then the count of current readers should be at least 1
				return !(oldState & writingMask) && (registeredPending || (oldState & currentReadersMask)) ;
			};

		const auto preempted = [&](const state_lock_value_t oldState) -> state_lock_value_t
			{
				constexpr state_lock_value_t preemptedMask = currentReadersMask;
				state_lock_value_t preemptedState = oldState;
				// Register as pending upgrade if not done so earlier, and unregister this thread as a reader
				if (!registeredPending)
				{
					registeredPending = true;
					preemptedState += impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 1, .writing = false, .stateLocked = false };
					preemptedState -= impl::DPWRLStateSemantics{ .currentReaders = 1, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
				}
				return preemptedState;
			};

		perform_under_locked_state(success, preemptionCheck, sanityChecks, preempted);
	}

	/**
	* @brief Downgrades a thread with a writer's lock to a reader's lock. If a thread already has a writer's lock, use this instead of `write_unlock()` -> `read_lock()`
	*/
	void downgrade()
	{
		const auto success = [](const state_lock_value_t oldState) -> state_lock_value_t
			{
				// Declare that thread is no longer writing and that it will be reading
				state_lock_value_t newState = oldState & ~writingMask;
				newState += impl::DPWRLStateSemantics{ .currentReaders = 1, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
				return newState;
			};

		const auto sanityChecks = [&](const state_lock_value_t oldState) -> bool
			{
				// If we had a writer's lock, the writing flag should be on
				// If we had a writer's lock, no thread should be currently reading
				return (oldState & writingMask) && !(oldState & currentReadersMask);
			};

		perform_under_locked_state(success, DefaultPreemptionCheck(), sanityChecks);
	}

private:

	constexpr static inline bool usingDebugCallback = ! std::is_same_v<DebugCallback, impl::DPWRLVoidDebugCallback>;

	struct DefaultPreemptionCheck
	{
		bool operator()(const state_lock_value_t oldState)
		{
			return false;
		}
	};

	constexpr static auto defaultPreempted = [](const state_lock_value_t oldState)->state_lock_value_t {return oldState; };
	constexpr static auto defaultSanityChecks = [](const state_lock_value_t oldState)->bool {return true; };

	template<typename Success, typename PreemptionCheck = DefaultPreemptionCheck, typename SanityChecks = decltype(defaultSanityChecks), typename Preempted = decltype(defaultPreempted)>
	inline void perform_under_locked_state(
		Success&& success,
		PreemptionCheck&& preemptionCheck = DefaultPreemptionCheck(),
		SanityChecks& sanityChecks = defaultSanityChecks,
		Preempted& preempted = defaultPreempted
		)
	{
		DebugCallback callback = {};
		for (uint32_t spinCount = 0; true; spinCount++)
		{
			const state_lock_value_t oldState = state.fetch_or(flipLock);
			if (oldState & flipLock)
			{
				if (spinCount > SpinsBeforeYield)
					std::this_thread::yield();
				continue;
			}
			assert(sanityChecks(oldState));
			const bool wasPreempted = preemptionCheck(oldState);
			// thankfully `?` operator actually short circuits
			const state_lock_value_t newState = wasPreempted ? preempted(oldState) : success(oldState);
			// new state must unlock the state lock
			assert(!(newState & flipLock));
			if constexpr (usingDebugCallback) callback(impl::DPWR_LOCK_DEBUG_STAGE::BEFORE_STATE_UPDATE);
			state.store(newState);
			if constexpr (usingDebugCallback) callback(impl::DPWR_LOCK_DEBUG_STAGE::AFTER_STATE_UPDATE);
			if (wasPreempted)
			{
				if constexpr (usingDebugCallback)
				{
					callback(impl::DPWR_LOCK_DEBUG_STAGE::PREEMPTED);
				}
				else
				{
					if (spinCount > SpinsBeforeYield)
						std::this_thread::yield();
				}
				continue;
			}
			break;
		}
	}

	state_lock_t state = {};

	constexpr static inline state_lock_value_t flipLock = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = true };
	constexpr static inline state_lock_value_t writingMask = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = 0, .writing = true, .stateLocked = false };
	constexpr static inline state_lock_value_t pendingUpgradesMask = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = 0, .pendingUpgrades = MaxActors, .writing = false, .stateLocked = false };
	constexpr static inline state_lock_value_t pendingWritersMask = impl::DPWRLStateSemantics{ .currentReaders = 0, .pendingWriters = MaxActors, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
	constexpr static inline state_lock_value_t currentReadersMask = impl::DPWRLStateSemantics{ .currentReaders = MaxActors, .pendingWriters = 0, .pendingUpgrades = 0, .writing = false, .stateLocked = false };
	constexpr static inline uint32_t SpinsBeforeYield = 5000;
};

namespace impl
{

template <impl::DPWRLDebugCallback DebugCallback = impl::DPWRLVoidDebugCallback>
class dpwr_lock_guard_base
{
	dpwr_lock_guard_base() : m_lock(nullptr) {}

public:
	using dpwr_lock_t = demote_promote_writer_readers_lock_debug<DebugCallback>;
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

	/**
	* @brief Checks whether this guard is currently locking the lock `lk`
	*/
	bool hasLocked(const dpwr_lock_t& lk) const
	{
		return m_lock == &lk;
	}

protected:
	dpwr_lock_guard_base(const dpwr_lock_t& lk) noexcept : m_lock(&lk) {}

	dpwr_lock_t* m_lock;
};

} // namespace impl

// Forward declaration required by GCC
template <impl::DPWRLDebugCallback DebugCallback>
class dpwr_write_lock_guard_debug;

template <impl::DPWRLDebugCallback DebugCallback = impl::DPWRLVoidDebugCallback>
class dpwr_read_lock_guard_debug : public impl::dpwr_lock_guard_base<DebugCallback>
{
	using base_t = impl::dpwr_lock_guard_base<DebugCallback>;
public:
	using dpwr_lock_t = demote_promote_writer_readers_lock_debug<DebugCallback>;
	using dpwr_write_lock_guard_debug_t = dpwr_write_lock_guard_debug<DebugCallback>;
	dpwr_read_lock_guard_debug(const dpwr_lock_t& lk, std::adopt_lock_t) : base_t(lk) {}
	explicit dpwr_read_lock_guard_debug(dpwr_lock_t& lk) : dpwr_read_lock_guard_debug(lk, std::adopt_lock_t())
	{
		this->m_lock->read_lock();
	}
	explicit dpwr_read_lock_guard_debug(dpwr_write_lock_guard_debug_t&& wl);

	~dpwr_read_lock_guard_debug()
	{
		if (this->m_lock)
			this->m_lock->read_unlock();
	}
};

template <impl::DPWRLDebugCallback DebugCallback = impl::DPWRLVoidDebugCallback>
class dpwr_write_lock_guard_debug : public impl::dpwr_lock_guard_base<DebugCallback>
{
	using base_t = impl::dpwr_lock_guard_base<DebugCallback>;
public:
	using dpwr_lock_t = demote_promote_writer_readers_lock_debug<DebugCallback>;
	using dpwr_read_lock_guard_debug_t = dpwr_read_lock_guard_debug<DebugCallback>;
	dpwr_write_lock_guard_debug(const dpwr_lock_t& lk, std::adopt_lock_t) : base_t(lk) {}
	explicit dpwr_write_lock_guard_debug(dpwr_lock_t& lk) : dpwr_write_lock_guard_debug(lk, std::adopt_lock_t())
	{
		this->m_lock->write_lock();
	}
	explicit dpwr_write_lock_guard_debug(dpwr_read_lock_guard_debug_t&& rl);

	~dpwr_write_lock_guard_debug()
	{
		if (this->m_lock)
			this->m_lock->write_unlock();
	}
};

template <impl::DPWRLDebugCallback DebugCallback>
inline dpwr_read_lock_guard_debug<DebugCallback>::dpwr_read_lock_guard_debug(dpwr_write_lock_guard_debug<DebugCallback>&& wl) : impl::dpwr_lock_guard_base<DebugCallback>(std::move(wl))
{
	this->m_lock->downgrade();
}

template <impl::DPWRLDebugCallback DebugCallback>
inline dpwr_write_lock_guard_debug<DebugCallback>::dpwr_write_lock_guard_debug(dpwr_read_lock_guard_debug<DebugCallback>&& rl) : impl::dpwr_lock_guard_base<DebugCallback>(std::move(rl))
{
	this->m_lock->upgrade();
}

using demote_promote_writer_readers_lock = demote_promote_writer_readers_lock_debug<impl::DPWRLVoidDebugCallback>;
using dpwr_read_lock_guard = dpwr_read_lock_guard_debug<impl::DPWRLVoidDebugCallback>;
using dpwr_write_lock_guard = dpwr_write_lock_guard_debug<impl::DPWRLVoidDebugCallback>;

} // namespace nbl::system

#endif

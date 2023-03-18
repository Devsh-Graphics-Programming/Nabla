// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_UNLOCK_GUARD_H_INCLUDED__
#define __NBL_CORE_UNLOCK_GUARD_H_INCLUDED__

namespace nbl
{
namespace core
{

	template <class BasicLockable>
	class unlock_guard
	{
	public:
		typedef BasicLockable mutex_type;

		inline explicit unlock_guard(BasicLockable& mutex) : mutex_(mutex)
		{
			mutex_.unlock();
		}
		inline unlock_guard(BasicLockable& mutex, std::adopt_lock_t t) : mutex_(mutex) {}
		inline ~unlock_guard() {
			mutex_.lock();
		}

		unlock_guard(const unlock_guard&) = delete;
		unlock_guard& operator=(const unlock_guard&) = delete;
	private:
		BasicLockable& mutex_;
	};
}
}

#endif

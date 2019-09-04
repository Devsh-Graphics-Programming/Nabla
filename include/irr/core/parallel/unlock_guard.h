// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_UNLOCK_GUARD_H_INCLUDED__
#define __IRR_UNLOCK_GUARD_H_INCLUDED__

namespace irr
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

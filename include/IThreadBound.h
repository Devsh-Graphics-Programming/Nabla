// Copyright (C) 2018
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_THREAD_BOUND_H_INCLUDED__
#define __I_THREAD_BOUND_H_INCLUDED__

#include <thread>

namespace irr
{
namespace core
{

//! Base class for things that cannot be shared between threads
class IThreadBound
{
        std::thread::id tid;
    protected:
        //! Class not intended to be instantiated by itself, always an ADT
        IThreadBound() : tid(std::this_thread::get_id()) {}
    public:
		//! Returns if the calling thread is the same as the one which created the object
		/**
		@returns If the calling thread is the same thread as the one which created the object.
		*/
        bool belongsToCurrentThread() const
        {
            return tid==std::this_thread::get_id();
        }

		//! Returns the ThreadID of the thread which created the object.
		/**
		@returns The ThreadID of the thread which created the object.
		*/
        std::thread::id getCreationThreadID() const
        {
            return std::this_thread::get_id();
        }
};

} // end namespace core
} // end namespace irr

#endif




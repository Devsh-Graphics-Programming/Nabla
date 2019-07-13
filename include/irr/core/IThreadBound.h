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

#ifdef _IRR_DEBUG
    #define _IRR_CHECK_OWNING_THREAD(_obj, EXTRA_BODY_TO_EXEC) \
        if (!_obj->belongsToCurrentThread()) \
        { \
            os::Printer::log("Attemped to use an IThreadBound object from a thread that did not create it!", ELL_ERROR);\
            EXTRA_BODY_TO_EXEC \
        }
#else
    #define _IRR_CHECK_OWNING_THREAD(_obj, EXTRA_BODY_TO_EXEC) \
        if (!_obj->belongsToCurrentThread()) \
        { \
            EXTRA_BODY_TO_EXEC \
        }
#endif

//! Base class for things that cannot be shared between threads
class IRR_FORCE_EBO IThreadBound
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




// Copyright (C) 2016 Mateusz "DeVsh" Kielan
// This file is part of the "Irrlicht Engine" and "Build A World".
// For conditions of distribution and use, see copyright notice in irrlicht.h
// and on http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=49672

#ifndef __I_DRIVER_FENCE_H_INCLUDED__
#define __I_DRIVER_FENCE_H_INCLUDED__


namespace irr
{
namespace video
{

enum E_DRIVER_FENCE_RETVAL
{
    //! Indicates that an error occurred. Additionally, an OpenGL error will be generated.
    EDFR_FAIL=0,
    //! If it returns GL_TIMEOUT_EXPIRED, then the sync object did not signal within the given timeout period (includes before us calling the func).
    EDFR_TIMEOUT_EXPIRED,
    //! Indicates that sync​ was signaled before the timeout expired.
    EDFR_CONDITION_SATISFIED,
    //! GPU already completed work before we even asked == THIS IS WHAT WE WANT
    EDFR_ALREADY_SIGNALED
};

//! Persistently Mapped buffer
class IDriverFence : public IReferenceCounted
{
    public:
        //! If timeout​ is zero, the function will simply check to see if the sync object is signaled and return immediately.
        /** \param timeout in nanoseconds.
        \param whether to perform a special implicit flush in OpenGL (quite useless).
        IMPORTANT: In OpenGL you need to glFlush at some point in the thread (context) AFTER you create this fence with
        IVideoDriver::placeFence and BEFORE you waitCPU on it in the same thread or another.
        https://www.khronos.org/opengl/wiki/Sync_Object#Flushing_and_contexts
        If you don't THE FENCE MAY NEVER BE SIGNALLED because the signalling command has never been flushed to the device queue.
        The `flush` parameter can do this glFlush for you just before the wait, BUT ONLY IF you've placed the fence in the same
        thread as the one you are waiting on the fence. So if you want to use IDriverFence for inter-context coordination you
        are screwed and must call glFlush manually.*/
        virtual E_DRIVER_FENCE_RETVAL waitCPU(const uint64_t &timeout, const bool &flush=false) = 0;

        //! This makes the GPU pause executing commands in the current context until commands before the fence in the context which created it, have completed
        /** You may be shocked to learn that OpenGL allows for commands in the same context to execute simultaneously or out of order as long as the result
        of these commands is the same as if they have been executed strictly in-order (except the memory effects on the objects following the incoherent memory model).
        For solving the above within a context you want to issue memory barriers, however for ensuring the order of commands between contexts you want to use waitGPU.*/
        virtual void waitGPU() = 0;
};

} // end namespace scene
} // end namespace irr

#endif


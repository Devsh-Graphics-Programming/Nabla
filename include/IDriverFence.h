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
        virtual E_DRIVER_FENCE_RETVAL waitCPU(const uint64_t &timeout, const bool &flush=false) = 0;

        virtual void waitGPU() = 0;
};

} // end namespace scene
} // end namespace irr

#endif


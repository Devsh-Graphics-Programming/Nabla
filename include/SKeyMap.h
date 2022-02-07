// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_S_KEY_MAP_H_INCLUDED__
#define __NBL_S_KEY_MAP_H_INCLUDED__

#include "Keycodes.h"

namespace nbl
{
//! enumeration for key actions. Used for example in the FPS Camera.
enum EKEY_ACTION
{
    EKA_MOVE_FORWARD = 0,
    EKA_MOVE_BACKWARD,
    EKA_STRAFE_LEFT,
    EKA_STRAFE_RIGHT,
    EKA_JUMP_UP,
    EKA_CROUCH,
    EKA_COUNT,

    //! This value is not used. It only forces this enumeration to compile in 32 bit.
    EKA_FORCE_32BIT = 0x7fffffff
};

//! Struct storing which key belongs to which action.
struct SKeyMap
{
    SKeyMap() {}
    SKeyMap(EKEY_ACTION action, EKEY_CODE keyCode)
        : Action(action), KeyCode(keyCode) {}

    EKEY_ACTION Action;
    EKEY_CODE KeyCode;
};

}  // end namespace nbl

#endif

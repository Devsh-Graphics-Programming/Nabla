// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_BULLET_I_MOTION_STATE_BASE_INCLUDED_
#define _NBL_EXT_BULLET_I_MOTION_STATE_BASE_INCLUDED_

#include "nabla.h"

#include "btBulletDynamicsCommon.h"

namespace nbl::ext::Bullet3
{
class CPhysicsWorld;
class IMotionStateBase : public btMotionState
{
public:
    IMotionStateBase(const btTransform& startTrans = btTransform::getIdentity())
        : m_startWorldTrans(startTrans),
          m_userPointer(0)
    {
    }

    virtual void getWorldTransform(btTransform& worldTrans) const = 0;
    virtual void setWorldTransform(const btTransform& worldTrans) = 0;

    btTransform m_startWorldTrans;
    void* m_userPointer;

protected:
    friend class CPhysicsWorld;
};

}

#endif
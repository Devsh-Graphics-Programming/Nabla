// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_BULLET_C_INSTANCED_MOTION_STATE_INCLUDED_
#define _NBL_EXT_BULLET_C_INSTANCED_MOTION_STATE_INCLUDED_

#include <cstdint>
#include "nabla.h"
#include "nbl/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "BulletUtility.h"
#include "IMotionStateBase.h"

namespace nbl
{
namespace ext
{
namespace Bullet3
{

class CInstancedMotionState : public IMotionStateBase{
public:
    inline CInstancedMotionState() {}
    inline CInstancedMotionState(scene::IMeshSceneNodeInstanced *node, uint32_t index)
        : m_node(node), 
          m_index(index),
          IMotionStateBase(convertMatrixSIMD(node->getInstanceTransform(index)))
    {
       

        m_node->grab();
    }

    inline ~CInstancedMotionState() {
        m_node->drop();
    }
    

    virtual void getWorldTransform(btTransform &worldTrans) const;
    virtual void setWorldTransform(const btTransform &worldTrans);
protected:

    scene::IMeshSceneNodeInstanced *m_node;
    uint32_t m_index;



};

}
}
}


#endif
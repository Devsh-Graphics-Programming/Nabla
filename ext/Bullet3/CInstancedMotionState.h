#ifndef _IRR_EXT_BULLET_C_INSTANCED_MOTION_STATE_INCLUDED_
#define _IRR_EXT_BULLET_C_INSTANCED_MOTION_STATE_INCLUDED_

#include <cstdint>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "BulletUtility.h"
#include "IMotionStateBase.h"

namespace irr
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
          IMotionStateBase(convertRetardedMatrix(node->getInstanceTransform(index)))
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
#ifndef _IRR_EXT_BULLET_I_MOTION_STATE_BASE_INCLUDED_
#define _IRR_EXT_BULLET_I_MOTION_STATE_BASE_INCLUDED_

#include <cstdint>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

namespace irr
{
namespace ext
{
namespace Bullet3
{


class CPhysicsWorld;
class IMotionStateBase : public btMotionState {
public:

    IMotionStateBase(const btTransform &startTrans = btTransform::getIdentity())
        : m_startWorldTrans(startTrans),
          m_userPointer(0)
    {
    }

    virtual void getWorldTransform(btTransform &worldTrans) const = 0;
    virtual void setWorldTransform(const btTransform &worldTrans) = 0;

    btTransform m_startWorldTrans;
    void *m_userPointer;

protected:


    friend class CPhysicsWorld;


};

}
}
}


#endif
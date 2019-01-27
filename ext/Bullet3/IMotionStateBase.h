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

    virtual void getWorldTransform(btTransform &worldTrans) const = 0;
    virtual void setWorldTransform(const btTransform &worldTrans) = 0;



protected:


    friend class CPhysicsWorld;


};

}
}
}


#endif
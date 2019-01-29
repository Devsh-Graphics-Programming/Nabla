#ifndef _IRR_EXT_BULLET_C_DEFAULT_MOTION_STATE_INCLUDED_
#define _IRR_EXT_BULLET_C_DEFAULT_MOTION_STATE_INCLUDED_

#include <cstdint>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "IMotionStateBase.h"

namespace irr
{
namespace ext
{
namespace Bullet3
{

class CDefaultMotionState : public IMotionStateBase {
public:
    CDefaultMotionState();
    virtual void getWorldTransform(btTransform &worldTrans) const;
    virtual void setWorldTransform(const btTransform &worldTrans);
protected:

};


}
}
}

#endif
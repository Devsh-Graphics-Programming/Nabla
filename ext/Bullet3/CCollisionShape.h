#ifndef _IRR_EXT_BULLET_C_COLLISION_SHAPE_INCLUDED_
#define _IRR_EXT_BULLET_C_COLLISION_SHAPE_INCLUDED_

#include <cstdint>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"

class btCollisionShape;

namespace irr
{
namespace ext
{
namespace Bullet3
{
class CCollisionShape : public core::IReferenceCounted {
public:
    CCollisionShape();

    virtual void setMargin(float margin);
    virtual void setBox(core::vector3df halfExts);
    virtual void setSphere(float radius);

protected:
    virtual ~CCollisionShape();

    virtual void clear();
    virtual void dirty();

    btCollisionShape *m_collisionShape;

};

}
}
}


#endif
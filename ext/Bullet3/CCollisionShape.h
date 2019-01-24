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
protected:
    virtual ~CCollisionShape();

    btCollisionShape *m_collisionShape;

};

}
}
}


#endif
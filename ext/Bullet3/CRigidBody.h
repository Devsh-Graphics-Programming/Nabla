#ifndef _IRR_EXT_BULLET_C_RIGIDBODY_INCLUDED_
#define _IRR_EXT_BULLET_C_RIGIDBODY_INCLUDED_

#include <cstdint>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"

class btRigidBody;

namespace irr
{
namespace ext
{
namespace Bullet3
{

class CCollisionShape;
class CRigidBody : public core::IReferenceCounted {

    struct InitParams {
        CCollisionShape *shape = nullptr;
        float mass;
        float friction;
        core::matrix4x3 transform;
    };

public:
    CRigidBody();
protected:
    virtual ~CRigidBody();

    btRigidBody *m_rigidBody;

};

}
}
}


#endif
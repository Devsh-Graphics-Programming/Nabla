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
class CRigidBody : public core::IReferenceCounted {
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
#ifndef _IRR_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_
#define _IRR_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_

#include <cstdint>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"

class btDiscreteDynamicsWorld;
class btDefaultCollisionConfiguration;
class btCollisionDispatcher;
class btBroadphaseInterface;
class btSequentialImpulseConstraintSolver;

namespace irr
{
namespace ext 
{
namespace Bullet3 
{

class CPhysicsWorld : public core::IReferenceCounted {
public:
    CPhysicsWorld();

    void step(float dt);

    void setGravity(core::vector3df gravity);
    core::vector3df getGravity() const;


protected:
    virtual ~CPhysicsWorld();

private:
    btDiscreteDynamicsWorld *m_physicsWorld;

    btDefaultCollisionConfiguration *m_collisionCfg;

    btCollisionDispatcher *m_dispatcher;
    btBroadphaseInterface *m_overlappingPairCache;
    btSequentialImpulseConstraintSolver *m_solver;

};


}
}
}


#endif 
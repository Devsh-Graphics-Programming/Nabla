#ifndef _IRR_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_
#define _IRR_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_

#include <cstdint>
#include <type_traits>
#include "irrlicht.h"
#include "irr/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "IMotionStateBase.h"
#include "CDefaultMotionState.h"

namespace irr
{
namespace ext 
{
namespace Bullet3 
{


class CPhysicsWorld : public core::IReferenceCounted {
public:

    struct RigidBodyData {
        btCollisionShape *shape;
        float mass;
    };

    CPhysicsWorld();
    ~CPhysicsWorld();

    btRigidBody *createRigidBody(RigidBodyData data);

    template<class state, typename... Args>
    inline state *bindRigidBody(btRigidBody *body, Args&&... args) {
       // static_assert(!std::is_base_of<IMotionStateBase, state>, "Motionstate being binded not inherited from IMotionStateBase!");
        free(body->getMotionState());
        
        void *mem = _IRR_ALIGNED_MALLOC(sizeof(state), 32u);
        state *motionState = new(mem) state(std::forward<Args>(args)...);

        body->setMotionState(motionState);

        m_physicsWorld->addRigidBody(body);

        return motionState;
    }

    void unbindRigidBody(btRigidBody *body);


    btDiscreteDynamicsWorld *getWorld();

protected:
    

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
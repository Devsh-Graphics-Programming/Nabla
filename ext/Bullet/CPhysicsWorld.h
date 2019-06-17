#ifndef _IRR_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_
#define _IRR_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_

#include <cstdint>
#include <type_traits>
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


class CPhysicsWorld : public core::IReferenceCounted {
public:

    struct RigidBodyData {
        btCollisionShape *shape;
        core::matrix3x4SIMD trans;
        core::vectorSIMDf inertia;
        float mass;
    };

    CPhysicsWorld();
    ~CPhysicsWorld();

    template<class btObject, typename... Args>
    inline btObject *createbtObject(Args&&... args) {
        void *mem = _IRR_ALIGNED_MALLOC(sizeof(btObject), 32u);
        return new(mem) btObject(std::forward<Args>(args)...);
    }

    template<class btObject>
    inline void deletebtObject(btObject *obj) {
        obj->~btObject();
        _IRR_ALIGNED_FREE(obj);
    }



    btRigidBody *createRigidBody(RigidBodyData data);
    void deleteRigidBody(btRigidBody *body);

    inline void bindRigidBody(btRigidBody *body) {
        m_physicsWorld->addRigidBody(body);
    }

    template<class state, typename... Args>
    inline state *bindRigidBody(btRigidBody *body, Args... args) {
        assert(!body->getMotionState());

        state *motionState = createbtObject<state>(args...);
        body->setMotionState(motionState);

        m_physicsWorld->addRigidBody(body);

        return motionState;
    }

    inline void unbindRigidBody(btRigidBody *body, bool free = true) {
        m_physicsWorld->removeRigidBody(body);
        if (free) {
            deletebtObject(body->getMotionState());
        }
    }


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

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_
#define _NBL_EXT_BULLET_C_PHYSICS_WORLD_INCLUDED_

#include <cstdint>
#include <type_traits>
#include "nabla.h"
#include "nbl/core/IReferenceCounted.h"
#include "btBulletDynamicsCommon.h"

#include "IMotionStateBase.h"

namespace nbl
{
namespace ext
{
namespace Bullet3
{
class CPhysicsWorld : public core::IReferenceCounted
{
public:
    struct RigidBodyData
    {
        btCollisionShape* shape;
        core::matrix3x4SIMD trans;
        core::vectorSIMDf inertia;
        float mass;
    };

    CPhysicsWorld();
    ~CPhysicsWorld();

    template<class btObject, typename... Args>
    inline btObject* createbtObject(Args&&... args)
    {
        void* mem = _NBL_ALIGNED_MALLOC(sizeof(btObject), 32u);
        return new(mem) btObject(std::forward<Args>(args)...);
    }

    template<class btObject>
    inline void deletebtObject(btObject* obj)
    {
        obj->~btObject();
        _NBL_ALIGNED_FREE(obj);
    }

    btRigidBody* createRigidBody(RigidBodyData data);
    void deleteRigidBody(btRigidBody* body);

    inline void bindRigidBody(btRigidBody* body)
    {
        m_physicsWorld->addRigidBody(body);
    }

    template<class state, typename... Args>
    inline state* bindRigidBody(btRigidBody* body, Args... args)
    {
        assert(!body->getMotionState());

        state* motionState = createbtObject<state>(args...);
        body->setMotionState(motionState);

        m_physicsWorld->addRigidBody(body);

        return motionState;
    }

    inline void unbindRigidBody(btRigidBody* body, bool free = true)
    {
        m_physicsWorld->removeRigidBody(body);
        if(free)
        {
            deletebtObject(body->getMotionState());
        }
    }

    btDiscreteDynamicsWorld* getWorld();

protected:
private:
    btDiscreteDynamicsWorld* m_physicsWorld;

    btDefaultCollisionConfiguration* m_collisionCfg;

    btCollisionDispatcher* m_dispatcher;
    btBroadphaseInterface* m_overlappingPairCache;
    btSequentialImpulseConstraintSolver* m_solver;
};

}
}
}

#endif

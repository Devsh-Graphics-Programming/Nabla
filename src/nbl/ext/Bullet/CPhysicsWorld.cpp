// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/Bullet/CPhysicsWorld.h"
#include "nbl/ext/Bullet/BulletUtility.h"

using namespace nbl;
using namespace ext;
using namespace Bullet3;

core::smart_refctd_ptr<CPhysicsWorld> CPhysicsWorld::create()
{
    auto world = new CPhysicsWorld();
    return core::smart_refctd_ptr<CPhysicsWorld>(world,core::dont_grab);
}

CPhysicsWorld::CPhysicsWorld()
{
    m_collisionCfg = createbtObject<btDefaultCollisionConfiguration>();
    m_dispatcher = createbtObject<btCollisionDispatcher>(m_collisionCfg);
    m_overlappingPairCache = createbtObject<btDbvtBroadphase>();
    m_solver = createbtObject<btSequentialImpulseConstraintSolver>();

    m_physicsWorld = createbtObject<btDiscreteDynamicsWorld>(
        m_dispatcher,
        m_overlappingPairCache,
        m_solver,
        m_collisionCfg
    );   
}

CPhysicsWorld::~CPhysicsWorld()
{
    deletebtObject(m_collisionCfg);
    deletebtObject(m_dispatcher);
    deletebtObject(m_overlappingPairCache);
    deletebtObject(m_solver);
    deletebtObject(m_physicsWorld);    
}


btRigidBody *CPhysicsWorld::createRigidBody(RigidBodyData data)
{
    btRigidBody *rigidBody = createbtObject<btRigidBody>(data.mass, nullptr, data.shape, tobtVec3(data.inertia));
    
    btTransform trans = convertMatrixSIMD(data.trans);
    rigidBody->setWorldTransform(trans);

    return rigidBody;
}
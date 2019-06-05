#include "CPhysicsWorld.h"
#include "BulletUtility.h"

using namespace irr;
using namespace ext;
using namespace Bullet3;

CPhysicsWorld::CPhysicsWorld() {

    m_collisionCfg = createbtObject<btDefaultCollisionConfiguration>();
    m_dispatcher = createbtObject<btCollisionDispatcher>(m_collisionCfg);
    m_overlappingPairCache = createbtObject<btDbvtBroadphase>();
    m_solver = createbtObject<btSequentialImpulseConstraintSolver>();

    m_physicsWorld = createbtObject <btDiscreteDynamicsWorld>(
        m_dispatcher,
        m_overlappingPairCache,
        m_solver,
        m_collisionCfg
    );   
}

CPhysicsWorld::~CPhysicsWorld() {
    deletebtObject(m_collisionCfg);
    deletebtObject(m_dispatcher);
    deletebtObject(m_overlappingPairCache);
    deletebtObject(m_solver);
    deletebtObject(m_physicsWorld);    
}


btRigidBody *CPhysicsWorld::createRigidBody(RigidBodyData data) {
    btRigidBody *rigidBody = createbtObject<btRigidBody>(data.mass, nullptr, data.shape, tobtVec3(data.inertia));
    
    btTransform trans = convertMatrixSIMD(data.trans);
    rigidBody->setWorldTransform(trans);

    return rigidBody;
}

void CPhysicsWorld::deleteRigidBody(btRigidBody *body) {
    deletebtObject(body);
}



btDiscreteDynamicsWorld *CPhysicsWorld::getWorld() {
    return m_physicsWorld;
}
#include "../../ext/Bullet3/CPhysicsWorld.h"
#include "../../ext/Bullet3/BulletUtility.h"
#include "btBulletDynamicsCommon.h"


using namespace irr;
using namespace ext;
using namespace Bullet3;

CPhysicsWorld::CPhysicsWorld() {
    m_collisionCfg = _IRR_NEW(btDefaultCollisionConfiguration);

    m_dispatcher = _IRR_NEW(btCollisionDispatcher, m_collisionCfg);
    m_overlappingPairCache = _IRR_NEW(btDbvtBroadphase);
    m_solver = _IRR_NEW(btSequentialImpulseConstraintSolver);
    
    m_physicsWorld = _IRR_NEW(btDiscreteDynamicsWorld,
        m_dispatcher, m_overlappingPairCache, m_solver, m_collisionCfg);
   
}

CPhysicsWorld::~CPhysicsWorld() {
    _IRR_DELETE(m_collisionCfg);
    _IRR_DELETE(m_dispatcher);
    _IRR_DELETE(m_overlappingPairCache);
    _IRR_DELETE(m_solver);
    _IRR_DELETE(m_physicsWorld);
    
}



btRigidBody *CPhysicsWorld::createRigidBody(RigidBodyData data) {
    void *mem = _IRR_ALIGNED_MALLOC(sizeof(btDefaultMotionState), 32u);
    btDefaultMotionState *state = new(mem) btDefaultMotionState();

    btRigidBody *rigidBody = _IRR_NEW(btRigidBody, data.mass, state, data.shape);
    
    btTransform trans = convertMatrixSIMD(data.trans);
    rigidBody->setWorldTransform(trans);

    return rigidBody;
}

void CPhysicsWorld::unbindRigidBody(btRigidBody *body, bool free) {
    m_physicsWorld->removeRigidBody(body);
    if (free) {
        body->getMotionState()->~btMotionState();
        _IRR_ALIGNED_FREE(body->getMotionState());
    }
}

btDiscreteDynamicsWorld *CPhysicsWorld::getWorld() {
    return m_physicsWorld;
}
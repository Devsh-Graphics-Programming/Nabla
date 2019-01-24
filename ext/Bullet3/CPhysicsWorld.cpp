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

void CPhysicsWorld::step(float dt) {
    m_physicsWorld->stepSimulation(dt);
}

void CPhysicsWorld::setGravity(core::vector3df gravity) {
    m_physicsWorld->setGravity(irrVec3Convert(gravity));
}

core::vector3df CPhysicsWorld::getGravity() const {
    return btVec3Convert(m_physicsWorld->getGravity());
}
#include "CRigidBody.h"
#include "BulletUtility.h"
#include "CCollisionShape.h"
#include "btBulletDynamicsCommon.h"

using namespace irr;
using namespace ext;
using namespace Bullet3;

CRigidBody::CRigidBody() {
    
}

CRigidBody::~CRigidBody() {
    _IRR_DELETE(m_rigidBody);
}

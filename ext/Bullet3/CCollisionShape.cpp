#include "CCollisionShape.h"
#include "BulletUtility.h"
#include "btBulletDynamicsCommon.h"


using namespace irr;
using namespace ext;
using namespace Bullet3;

CCollisionShape::CCollisionShape() {
    m_collisionShape = _IRR_NEW(btEmptyShape);
}

CCollisionShape::~CCollisionShape() {
    clear();
}

void CCollisionShape::setMargin(float margin) {
    m_collisionShape->setMargin(margin);
}

void CCollisionShape::setBox(core::vector3df halfExts) {
    clear();
    m_collisionShape = _IRR_NEW(btBoxShape, irrVec3Convert(halfExts));
    dirty();
}

void CCollisionShape::setSphere(float radius) {
    clear();
    m_collisionShape = _IRR_NEW(btSphereShape, radius);
    dirty();
}

void CCollisionShape::clear() {
    _IRR_DELETE(m_collisionShape);
}

void CCollisionShape::dirty() {
    //TODO: Find way to report to rigidbody collisionshape change!
}












#include "CDefaultMotionState.h"

using namespace irr;
using namespace ext;
using namespace Bullet3;

void CDefaultMotionState::getWorldTransform(btTransform &worldTrans) const {
    
}

void CDefaultMotionState::setWorldTransform(const btTransform &worldTrans) {
   btVector3 a = worldTrans.getOrigin();
}
#include "CDefaultMotionState.h"
#include <iostream>
using namespace irr;
using namespace ext;
using namespace Bullet3;

CDefaultMotionState::CDefaultMotionState() {
}

void CDefaultMotionState::getWorldTransform(btTransform &worldTrans) const {
    
}

void CDefaultMotionState::setWorldTransform(const btTransform &worldTrans) {
   btVector3 a = worldTrans.getOrigin();
   std::cout << a.x() << " " << a.y() << " " << a.z() << std::endl;
}
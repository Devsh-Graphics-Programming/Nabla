#include "CInstancedMotionState.h"

#include "../../ext/DebugDraw/CDraw3DLine.h"
using namespace irr;
using namespace ext;
using namespace Bullet3;


void CInstancedMotionState::getWorldTransform(btTransform &worldTrans) const {    
    worldTrans = convertMatrixSIMD(m_node->getInstanceTransform(m_index));
}

void CInstancedMotionState::setWorldTransform(const btTransform &worldTrans) {
    m_node->setInstanceTransform(m_index, convertbtTransform(worldTrans));
}
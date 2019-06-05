#include "CInstancedMotionState.h"

#include "../../ext/DebugDraw/CDraw3DLine.h"
using namespace irr;
using namespace ext;
using namespace Bullet3;


void CInstancedMotionState::getWorldTransform(btTransform &worldTrans) const {
    core::matrix3x4SIMD mat;
    mat.set(m_node->getInstanceTransform(m_index));
    
    worldTrans = convertMatrixSIMD(mat);
}

void CInstancedMotionState::setWorldTransform(const btTransform &worldTrans) {
    

    m_node->setInstanceTransform(m_index, convertbtTransform(worldTrans).getAsRetardedIrrlichtMatrix());
}
#include "CInstancedMotionState.h"
#include "BulletUtility.h"

using namespace irr;
using namespace ext;
using namespace Bullet3;

CInstancedMotionState::CInstancedMotionState() {}

CInstancedMotionState::CInstancedMotionState(scene::IMeshSceneNodeInstanced *node, uint32_t index):
    m_node(node), m_index(index){
}

void CInstancedMotionState::getWorldTransform(btTransform &worldTrans) const {
    worldTrans = convertMatrixSIMD(m_transform);
}

void CInstancedMotionState::setWorldTransform(const btTransform &worldTrans) {
    m_transform = convertbtTransform(worldTrans);

    m_node->setInstanceTransform(m_index, m_transform.getAsRetardedIrrlichtMatrix());
}
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

}

void CInstancedMotionState::setWorldTransform(const btTransform &worldTrans) {
    core::matrix3x4SIMD irrTrans;

    for (uint32_t i = 0; i < 3u; ++i) {
        irrTrans.rows[i] = btVec3Convert(worldTrans.getBasis().getRow(i));
    }
    irrTrans.setTranslation(btVec3Convert(worldTrans.getOrigin()));

   
    m_node->setInstanceTransform(m_index, irrTrans.getAsRetardedIrrlichtMatrix());
}
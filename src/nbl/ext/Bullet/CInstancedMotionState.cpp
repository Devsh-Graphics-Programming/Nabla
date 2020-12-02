// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "irr/ext/Bullet/CInstancedMotionState.h"

#include "irr/ext/DebugDraw/CDraw3DLine.h"
using namespace irr;
using namespace ext;
using namespace Bullet3;


void CInstancedMotionState::getWorldTransform(btTransform &worldTrans) const {    
    worldTrans = convertMatrixSIMD(m_node->getInstanceTransform(m_index));
}

void CInstancedMotionState::setWorldTransform(const btTransform &worldTrans) {
    m_node->setInstanceTransform(m_index, convertbtTransform(worldTrans));
}
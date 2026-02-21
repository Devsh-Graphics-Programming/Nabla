// Copyright (C) 2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/scene/ITransformTree.h"


using namespace nbl;
using namespace scene;

#if 0 // legacy and unported
ITransformTree::~ITransformTree()
{
}


ITransformTreeWithoutNormalMatrices::ITransformTreeWithoutNormalMatrices(
	core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS
) : ITransformTree(std::move(_transformHierarchyDS),std::move(_renderDS)), m_nodeStorage(std::move(_nodeStorage))
{
	m_nodeStorage->getPropertyMemoryBlock(parent_prop_ix).buffer->setObjectDebugName("ITransformTreeWithoutNormalMatrices::parent_t");
	m_nodeStorage->getPropertyMemoryBlock(relative_transform_prop_ix).buffer->setObjectDebugName("ITransformTreeWithoutNormalMatrices::relative_transform_t");
	m_nodeStorage->getPropertyMemoryBlock(modified_stamp_prop_ix).buffer->setObjectDebugName("ITransformTreeWithoutNormalMatrices::modified_stamp_t");
	m_nodeStorage->getPropertyMemoryBlock(global_transform_prop_ix).buffer->setObjectDebugName("ITransformTreeWithoutNormalMatrices::global_transform_t");
	m_nodeStorage->getPropertyMemoryBlock(recomputed_stamp_prop_ix).buffer->setObjectDebugName("ITransformTreeWithoutNormalMatrices::recomputed_stamp_t");
}


ITransformTreeWithNormalMatrices::ITransformTreeWithNormalMatrices(
	core::smart_refctd_ptr<property_pool_t>&& _nodeStorage, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _transformHierarchyDS, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& _renderDS
) : ITransformTree(std::move(_transformHierarchyDS),std::move(_renderDS)), m_nodeStorage(std::move(_nodeStorage))
{
	m_nodeStorage->getPropertyMemoryBlock(parent_prop_ix).buffer->setObjectDebugName("ITransformTreeWithNormalMatrices::parent_t");
	m_nodeStorage->getPropertyMemoryBlock(relative_transform_prop_ix).buffer->setObjectDebugName("ITransformTreeWithNormalMatrices::relative_transform_t");
	m_nodeStorage->getPropertyMemoryBlock(modified_stamp_prop_ix).buffer->setObjectDebugName("ITransformTreeWithNormalMatrices::modified_stamp_t");
	m_nodeStorage->getPropertyMemoryBlock(global_transform_prop_ix).buffer->setObjectDebugName("ITransformTreeWithNormalMatrices::global_transform_t");
	m_nodeStorage->getPropertyMemoryBlock(recomputed_stamp_prop_ix).buffer->setObjectDebugName("ITransformTreeWithNormalMatrices::recomputed_stamp_t");
	m_nodeStorage->getPropertyMemoryBlock(normal_matrix_prop_ix).buffer->setObjectDebugName("ITransformTreeWithNormalMatrices::normal_matrix_t");
}
#endif
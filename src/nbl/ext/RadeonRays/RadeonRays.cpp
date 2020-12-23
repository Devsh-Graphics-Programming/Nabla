// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <numeric>

#include "nbl/ext/RadeonRays/RadeonRays.h"

#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#include "../source/Nabla/COpenGLDriver.h"

using namespace nbl;
using namespace asset;
using namespace video;

using namespace nbl::ext::RadeonRays;



core::smart_refctd_ptr<Manager> Manager::create(video::IVideoDriver* _driver)
{
	if (!_driver)
		return nullptr;

	auto* glDriver = static_cast<COpenGLDriver*>(_driver);
	cl_context context = clCreateContext(glDriver->getOpenCLAssociatedContextProperties(),1u,&glDriver->getOpenCLAssociatedDevice(),nullptr,nullptr,nullptr);
	
	if (!context)
		return nullptr;

	auto manager = new Manager(_driver,context,ocl::COpenCLHandler::getPlatformInfo(glDriver->getOpenCLAssociatedPlatformID()).FeatureAvailable[ocl::COpenCLHandler::SOpenCLPlatformInfo::NBL_KHR_GL_EVENT]);
	return core::smart_refctd_ptr<Manager>(manager,core::dont_grab);
}

Manager::Manager(video::IVideoDriver* _driver, cl_context context, bool automaticOpenCLSync) : driver(_driver), rr(nullptr), m_context(context), commandQueue(nullptr), m_automaticOpenCLSync(automaticOpenCLSync)
{
	auto* glDriver = static_cast<COpenGLDriver*>(driver);
	commandQueue = clCreateCommandQueue(m_context, glDriver->getOpenCLAssociatedDevice(),/*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/0,nullptr);
	rr = ::RadeonRays::CreateFromOpenClContext(m_context, glDriver->getOpenCLAssociatedDevice(), commandQueue);
}

Manager::~Manager()
{
	clFinish(commandQueue);

	::RadeonRays::IntersectionApi::Delete(rr);
	rr = nullptr;

	clReleaseCommandQueue(commandQueue);
	commandQueue = nullptr;

	clReleaseContext(m_context);
	m_context = nullptr;
}



std::pair<::RadeonRays::Buffer*,cl_mem> Manager::linkBuffer(const video::IGPUBuffer* buffer, cl_mem_flags access)
{
	if (!buffer)
		return {nullptr,nullptr};

	cl_int error = CL_SUCCESS;
	cl_mem clbuff = clCreateFromGLBuffer(m_context,access,static_cast<const COpenGLBuffer*>(buffer)->getOpenGLName(),&error);
	switch (error)
	{
		case CL_SUCCESS:
			return {::RadeonRays::CreateFromOpenClBuffer(rr,clbuff),clbuff};
			break;
		default:
			break;
	}
	return {nullptr,nullptr};
}

void Manager::makeShape(MeshBufferRRShapeCache& shapeCache, const asset::ICPUMeshBuffer* mb)
{
	auto found = shapeCache.find(mb);
	if (found==shapeCache.end())
		return;

	auto pType = mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType;
	const auto* indices = reinterpret_cast<const int32_t*>(mb->getIndices());
	const auto indexCount = mb->getIndexCount();
	if (indexCount<3)
		return;

	const int32_t vertexCount = mb->calcVertexCount();

	constexpr int32_t IndicesPerTriangle = 3;
	const auto posAttrID = mb->getPositionAttributeIx();
	found->second = rr->CreateMesh(	reinterpret_cast<const float*>(	mb->getAttribPointer(posAttrID)),vertexCount,
																	mb->getAttribStride(posAttrID),
																	indices,sizeof(int32_t)*IndicesPerTriangle, // radeon rays understands index stride differently to me
																	nullptr,indexCount/IndicesPerTriangle);
}

#ifdef TODO
void Manager::makeInstance(	MeshNodeRRInstanceCache& instanceCache,
							const core::unordered_map<const video::IGPUMeshBuffer*,MeshBufferRRShapeCache::value_type>& GPU2CPUTable,
							scene::IMeshSceneNode* node, const int32_t* id_it)
{
	if (instanceCache.find(node)!=instanceCache.end())
		return; // already cached

	const auto* mesh = node->getMesh();
	auto mbCount = mesh->getMeshBufferCount();
	if (mbCount==0u)
		return;

	auto output = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<::RadeonRays::Shape*>>(mbCount);
	for (auto i=0; i<mbCount; i++)
	{
		const auto* mb = mesh->getMeshBuffer(i);
		auto found = GPU2CPUTable.find(mb);
		if (found==GPU2CPUTable.end())
			continue;

		auto* instance = rr->CreateInstance(found->second.second);
		if (id_it)
			instance->SetId(*id_it);
		output->operator[](i) = instance;
	}
	instanceCache.insert({node,output});
}
#endif
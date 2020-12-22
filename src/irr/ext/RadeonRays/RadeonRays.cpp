#include <numeric>

#include "irr/ext/RadeonRays/RadeonRays.h"

#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace asset;
using namespace video;

using namespace irr::ext::RadeonRays;



core::smart_refctd_ptr<Manager> Manager::create(video::IVideoDriver* _driver)
{
	if (!_driver)
		return nullptr;

	auto* glDriver = static_cast<COpenGLDriver*>(_driver);
	cl_context context = clCreateContext(glDriver->getOpenCLAssociatedContextProperties(),1u,&glDriver->getOpenCLAssociatedDevice(),nullptr,nullptr,nullptr);
	
	if (!context)
		return nullptr;

	auto manager = new Manager(_driver,context,ocl::COpenCLHandler::getPlatformInfo(glDriver->getOpenCLAssociatedPlatformID()).FeatureAvailable[ocl::COpenCLHandler::SOpenCLPlatformInfo::IRR_KHR_GL_EVENT]);
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

void Manager::makeShape(core::unordered_map<const asset::ICPUMeshBuffer*,::RadeonRays::Shape*>& cpuCache, const asset::ICPUMeshBuffer* mb)
{
	auto found = cpuCache.find(mb);
	if (found==cpuCache.end())
		return;

	auto pType = mb->getPipeline()->getPrimitiveAssemblyParams().primitiveType;
	const auto* indices = reinterpret_cast<const int32_t*>(mb->getIndices());
	const auto indexCount = mb->getIndexCount();
	assert(indexCount>=3);

	const int32_t vertexCount = mb->calcVertexCount();

	constexpr int32_t IndicesPerTriangle = 3;
	const auto posAttrID = mb->getPositionAttributeIx();
	found->second = rr->CreateMesh(	reinterpret_cast<const float*>(	mb->getAttribPointer(posAttrID)),vertexCount,
																	mb->getAttribStride(posAttrID),
																	indices,sizeof(int32_t)*IndicesPerTriangle, // radeon rays understands index stride differently to me
																	nullptr,indexCount/IndicesPerTriangle);
}

void Manager::makeInstance(	NblInstanceRRInstanceCache& instanceCache, MockSceneManager* mock_smgr,
							MeshBufferRRShapeCache& shapeCache, asset::IAssetManager* _assetManager,
							MockSceneManager::ObjectGUID objectID)
{
	if (instanceCache.find(objectID)!=instanceCache.end())
		return; // already cached

	const auto& objectData = mock_smgr->getObjectData(objectID);
	const auto& mesh = objectData.mesh;
	const auto mbCount = mesh->getMeshBufferCount();
	assert(mbCount==objectData.instanceGUIDPerMeshBuffer.size());
	assert(mbCount>0u);

	auto& gpuCache = shapeCache.m_gpuAssociative;
	bool notRefreshedAlready = true;

	auto output = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<::RadeonRays::Shape*>>(mbCount);
	for (auto i=0; i<mbCount; i++)
	{
		const auto* mb = mesh->getMeshBuffer(i);
		auto found = gpuCache.find(mb);
		// refresh cpu cache from cpu cache if not found
		if (notRefreshedAlready && found==gpuCache.end())
		{
			for (auto cpuAssociation : shapeCache.m_cpuAssociative)
			{
				const auto* gpumeshbuffer = dynamic_cast<video::IGPUMeshBuffer*>(_assetManager->findGPUObject(cpuAssociation.first).get());
				if (!gpumeshbuffer)
					continue;
				::RadeonRays::Shape* shape = cpuAssociation.second;
				auto successAndLocation = gpuCache.emplace(gpumeshbuffer,shape);
				if (gpumeshbuffer==mb)
					found = successAndLocation.first;
			}
			notRefreshedAlready = false;
		}
		if (found==gpuCache.end())
			continue;

		auto* instance = rr->CreateInstance(found->second);
		instance->SetId(objectData.instanceGUIDPerMeshBuffer[i]);
		output->operator[](i) = instance;
	}
	instanceCache.insert({objectID,output});
}
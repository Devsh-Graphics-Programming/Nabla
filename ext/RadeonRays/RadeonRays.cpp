#include <numeric>

#include "../../ext/RadeonRays/RadeonRays.h"

#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace asset;
using namespace video;

using namespace irr::ext::RadeonRays;



cl_context Manager::context = nullptr;
core::smart_refctd_ptr<RadeonRaysIncludeLoader> Manager::radeonRaysIncludes = nullptr;
bool Manager::automaticOpenCLSync = false;

core::smart_refctd_ptr<Manager> Manager::create(video::IVideoDriver* _driver)
{
	if (!_driver || context)
		return nullptr;

	auto* glDriver = static_cast<COpenGLDriver*>(_driver);
	context = clCreateContext(glDriver->getOpenCLAssociatedContextProperties(),1u,&glDriver->getOpenCLAssociatedDevice(),nullptr,nullptr,nullptr);
	
	if (!context)
		return nullptr;

	if (!radeonRaysIncludes)
		radeonRaysIncludes = core::make_smart_refctd_ptr<RadeonRaysIncludeLoader>();

	automaticOpenCLSync = ocl::COpenCLHandler::getPlatformInfo(glDriver->getOpenCLAssociatedPlatformID()).FeatureAvailable[ocl::COpenCLHandler::SOpenCLPlatformInfo::IRR_KHR_GL_EVENT];

	auto manager = new Manager(_driver);
	return core::smart_refctd_ptr<Manager>(manager,core::dont_grab);
}

Manager::Manager(video::IVideoDriver* _driver) : driver(_driver), commandQueue(nullptr), rr(nullptr)
{
	auto* glDriver = static_cast<COpenGLDriver*>(driver);
	commandQueue = clCreateCommandQueue(context, glDriver->getOpenCLAssociatedDevice(),/*CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/0,nullptr);
	rr = ::RadeonRays::CreateFromOpenClContext(context, glDriver->getOpenCLAssociatedDevice(), commandQueue);
}

Manager::~Manager()
{
	clFinish(commandQueue);

	::RadeonRays::IntersectionApi::Delete(rr);
	rr = nullptr;

	clReleaseCommandQueue(commandQueue);
	commandQueue = nullptr;

	clReleaseContext(context);
	context = nullptr;
}



std::pair<::RadeonRays::Buffer*,cl_mem> Manager::linkBuffer(const video::IGPUBuffer* buffer, cl_mem_flags access)
{
	if (!buffer)
		return {nullptr,nullptr};

	cl_int error = CL_SUCCESS;
	cl_mem clbuff = clCreateFromGLBuffer(context,access,static_cast<const COpenGLBuffer*>(buffer)->getOpenGLName(),&error);
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


void Manager::makeShape(MeshBufferRRShapeCache& shapeCache, const asset::ICPUMeshBuffer* mb, int32_t* indices)
{
	auto found = shapeCache.find(mb);
	if (found==shapeCache.end())
		return;

	int32_t vertexCount = 0;
	const int32_t* theseIndices = indices;
	int32_t indexCount = 0;

	auto pType = mb->getPrimitiveType();
	const void* meshIndices = mb->getIndices();
	const auto meshIndexCount = mb->getIndexCount();
	auto setIndex = [&](int32_t& index, auto orig)
	{
		index = orig;
		if (index > vertexCount)
			vertexCount = index;
	};
	if (pType==asset::EPT_TRIANGLE_STRIP)
	{
		if (meshIndexCount<3u)
			return;

		indexCount = (meshIndexCount-2u)*3u;
		auto strips2tris = [&](auto* optr, const auto* iptr)
		{
			for (int32_t i=0, j=0; i<indexCount; j += 2)
			{
				setIndex(optr[i++],iptr[j + 0]);
				setIndex(optr[i++],iptr[j + 1]);
				setIndex(optr[i++],iptr[j + 2]);
				if (i == indexCount)
					break;
				setIndex(optr[i++],iptr[j + 2]);
				setIndex(optr[i++],iptr[j + 1]);
				setIndex(optr[i++],iptr[j + 3]);
			}
			vertexCount++;
		};
		switch (mb->getIndexType())
		{
			case EIT_32BIT:
				strips2tris(indices,reinterpret_cast<const uint32_t*>(meshIndices));
				break;
			case EIT_16BIT:
				strips2tris(indices,reinterpret_cast<const uint16_t*>(meshIndices));
				break;
			default:
				vertexCount = meshIndexCount;
				for (int32_t i=0, j=0; i<indexCount; j += 2)
				{
					indices[i++] = j + 0;
					indices[i++] = j + 1;
					indices[i++] = j + 2;
					if (i == indexCount)
						break;
					indices[i++] = j + 2;
					indices[i++] = j + 1;
					indices[i++] = j + 3;
				}
				break;
		}
	}
	else if (pType==asset::EPT_TRIANGLE_FAN)
	{
		if (meshIndexCount<3)
			return;

		indexCount = ((meshIndexCount-1u)/2u)*3u;
		auto fan2tris = [&](auto* optr, const auto* iptr)
		{
			for (int32_t i=0, j=1; i<indexCount; j += 2)
			{
				setIndex(optr[i++],iptr[0]);
				setIndex(optr[i++],iptr[j]);
				setIndex(optr[i++],iptr[j+1]);
			}
			vertexCount++;
		};
		switch (mb->getIndexType())
		{
			case EIT_32BIT:
				fan2tris(indices,reinterpret_cast<const uint32_t*>(meshIndices));
				break;
			case EIT_16BIT:
				fan2tris(indices,reinterpret_cast<const uint16_t*>(meshIndices));
				break;
			default:
				vertexCount = meshIndexCount;
				for (int32_t i=0, j=1; i<indexCount; j += 2)
				{
					indices[i++] = 0;
					indices[i++] = j;
					indices[i++] = j+1;
				}
				break;
		}
	}
	else// if (pType==asset::EPT_TRIANGLES)
	{
		if (meshIndexCount<3)
			return;

		indexCount = meshIndexCount;
		switch (mb->getIndexType())
		{
			case EIT_32BIT:
				theseIndices = reinterpret_cast<const int32_t*>(meshIndices);
				for (uint32_t i=0; i<mb->getIndexCount(); i++)
				{
					int32_t index;
					setIndex(index,theseIndices[i]);
				}
				vertexCount++;
				break;
			case EIT_16BIT:
				for (uint32_t i=0; i<mb->getIndexCount(); i++)
					setIndex(indices[i],reinterpret_cast<const uint16_t*>(meshIndices)[i]);
				vertexCount++;
				break;
			default:
				vertexCount = meshIndexCount;
				std::iota(indices,indices+vertexCount,0);
				break;
		}
	}
		
	auto posAttrID = mb->getPositionAttributeIx();
	constexpr int32_t IndicesPerTriangle = 3;
	found->second = rr->CreateMesh(	reinterpret_cast<const float*>(	mb->getAttribPointer(posAttrID)),vertexCount,
																	mb->getMeshDataAndFormat()->getMappedBufferStride(posAttrID),
																	theseIndices,sizeof(int32_t)*IndicesPerTriangle, // radeon rays understands index stride differently to me
																	nullptr,indexCount/IndicesPerTriangle);
}

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

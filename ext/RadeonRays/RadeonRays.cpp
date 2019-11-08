#include <numeric>

#include "../../ext/RadeonRays/RadeonRays.h"

#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace asset;
using namespace video;

using namespace irr::ext::RadeonRays;



cl_context Manager::context = nullptr;
core::smart_refctd_ptr<RadeonRaysIncludeLoader> Manager::radeonRaysIncludes = nullptr;


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



::RadeonRays::Buffer* Manager::linkBuffer(const video::IGPUBuffer* buffer, cl_mem_flags access)
{
	if (!buffer)
		return nullptr;

	cl_int error = CL_SUCCESS;
	cl_mem clbuff = clCreateFromGLBuffer(context,access,static_cast<const COpenGLBuffer*>(buffer)->getOpenGLName(),&error);
	switch (error)
	{
		case CL_SUCCESS:
			return ::RadeonRays::CreateFromOpenClBuffer(rr,clbuff);
			break;
		default:
			break;
	}
	return nullptr;
}


void Manager::makeRRShapes(MeshBufferRRShapeCache& shapeCache, const asset::ICPUMeshBuffer** _begin, const asset::ICPUMeshBuffer** _end)
{
	shapeCache.reserve(std::distance(_begin,_end));

	uint32_t maxIndexCount = 0u;
	for (auto it=_begin; it!=_end; it++)
	{
		auto* mb = *it;
		auto found = shapeCache.find(mb);
		if (found!=shapeCache.end())
			continue;
		shapeCache.insert({mb,nullptr});


		auto posAttrID = mb->getPositionAttributeIx();
		auto format = mb->getMeshDataAndFormat()->getAttribFormat(posAttrID);
		assert(format!=asset::EF_R32G32B32A32_SFLOAT&&format!=asset::EF_R32G32B32_SFLOAT);

		auto pType = mb->getPrimitiveType();
		switch (pType)
		{
			case asset::EPT_TRIANGLE_STRIP:
				maxIndexCount = core::max((mb->getIndexCount()-2u)/3u, maxIndexCount);
				break;
			case asset::EPT_TRIANGLE_FAN:
				maxIndexCount = core::max(((mb->getIndexCount()-1u)/2u)*3u, maxIndexCount);
				break;
			case asset::EPT_TRIANGLES:
				maxIndexCount = core::max(mb->getIndexCount(), maxIndexCount);
				break;
			default:
				assert(false);
		}
	}

	if (maxIndexCount ==0u)
		return;


	constexpr int32_t VerticesInTriangle = 3;
	auto* mem = new int32_t[maxIndexCount*(VerticesInTriangle+1)/VerticesInTriangle];

	auto* const indices = mem;
	auto* const vertsPerFace = mem+maxIndexCount;
	std::fill(vertsPerFace, vertsPerFace+maxIndexCount/VerticesInTriangle, VerticesInTriangle);
	for (auto it=_begin; it!=_end; it++)
	{
		auto* mb = *it;
		auto found = shapeCache.find(mb);
		if (found==shapeCache.end())
			continue;

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
				continue;

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
				continue;

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
				continue;

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
		found->second = rr->CreateMesh(	reinterpret_cast<const float*>(	mb->getAttribPointer(posAttrID)),vertexCount,
																		mb->getMeshDataAndFormat()->getMappedBufferStride(posAttrID),
																		theseIndices,sizeof(int32_t),
																		vertsPerFace,indexCount/3);
	}

	delete[] mem;
}

void Manager::makeRRInstances(	MeshNodeRRInstanceCache& instanceCache, const MeshBufferRRShapeCache& shapeCache, asset::IAssetManager* _assetManager,
								scene::IMeshSceneNode** _begin, scene::IMeshSceneNode** _end, const int32_t* _id_begin)
{
	core::unordered_map<const video::IGPUMeshBuffer*,MeshBufferRRShapeCache::value_type> GPU2CPUTable;
	GPU2CPUTable.reserve(shapeCache.size());
	for (auto record : shapeCache)
	{
		auto gpumesh = dynamic_cast<video::IGPUMeshBuffer*>(_assetManager->findGPUObject(record.first).get());
		if (!gpumesh)
			continue;

		GPU2CPUTable.insert({gpumesh,record});
	}

	auto* id_it = _id_begin;
	for (auto it=_begin; it!=_end; it++,id_it++)
	{
		auto* node = *it;
		if (instanceCache.find(node)!=instanceCache.end())
			continue; // already cached

		const auto* mesh = node->getMesh();
		auto mbCount = mesh->getMeshBufferCount();
		if (mbCount==0u)
			continue;

		auto output = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<::RadeonRays::Shape*>>(mbCount);
		for (auto i=0; i<mbCount; i++)
		{
			const auto* mb = mesh->getMeshBuffer(i);
			auto found = GPU2CPUTable.find(mb);
			if (found==GPU2CPUTable.end())
				continue;

			auto* instance = rr->CreateInstance(found->second.second);
			if (_id_begin)
				instance->SetId(*id_it);
			output->operator[](i) = instance;
		}
		instanceCache.insert({node,output});
	}
}

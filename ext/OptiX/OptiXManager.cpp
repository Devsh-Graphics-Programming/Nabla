#include <numeric>

#include "../../ext/OptiX/OptiXManager.h"

#include "optix_function_table_definition.h"

//#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace asset;
using namespace video;

using namespace irr::ext::OptiX;


core::smart_refctd_ptr<Manager> Manager::create(video::IVideoDriver* _driver, io::IFileSystem* _filesystem)
{
	if (!_driver)
		return nullptr;

	cuda::CCUDAHandler::init();

	int32_t version = 0;
	if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuDriverGetVersion(&version)) || version<7000)
		return nullptr;

	// find device
	uint32_t foundDeviceCount = 0u;
	CUdevice devices[MaxSLI] = {};
	cuda::CCUDAHandler::getDefaultGLDevices(&foundDeviceCount, devices, MaxSLI);

	// create context
	CUcontext contexts[MaxSLI] = {};
	bool ownContext[MaxSLI] = {};
	uint32_t suitableDevices = 0u;
	for (uint32_t i=0u; i<foundDeviceCount; i++)
	{
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxCreate_v2(contexts+suitableDevices, CU_CTX_SCHED_YIELD|CU_CTX_MAP_HOST|CU_CTX_LMEM_RESIZE_TO_MAX, devices[suitableDevices])))
			continue;

		uint32_t version = 0u;
		cuda::CCUDAHandler::cuda.pcuCtxGetApiVersion(contexts[suitableDevices],&version);
		if (version<3020)
		{
			cuda::CCUDAHandler::cuda.pcuCtxDestroy_v2(contexts[suitableDevices]);
			continue;
		}
		cuda::CCUDAHandler::cuda.pcuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
		ownContext[suitableDevices++] = true;
	}

	if (!suitableDevices)
		return nullptr;

	auto manager = new Manager(_driver,_filesystem,suitableDevices,contexts,ownContext);
	return core::smart_refctd_ptr<Manager>(manager,core::dont_grab);
}

Manager::Manager(video::IVideoDriver* _driver, io::IFileSystem* _filesystem, uint32_t _contextCount, CUcontext* _contexts, bool* _ownContexts) : driver(_driver), contextCount(_contextCount)
{
	assert(contextCount<=MaxSLI);

	// Initialize the OptiX API, loading all API entry points 
	optixInit();

	for (uint32_t i=0u; i<contextCount; i++)
	{
		context[i] = _contexts[i];
		ownContext[i] = _ownContexts ? _ownContexts[i]:false;
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuStreamCreate(stream+i,CU_STREAM_NON_BLOCKING)))
		{
			i--;
			contextCount--;
			continue;
		}

		OptixDeviceContextOptions options = {};
		optixDeviceContextCreate(context[i], &options, optixContext+i);
		optixDeviceContextSetLogCallback(optixContext[i], &defaultCallback, reinterpret_cast<void*>(i), 3u);
	}

	// TODO: This cannot stay like that, we need a resource compiler to "build-in" the optix CUDA/device headers into irr::ext::OptiX so that we can retrieve them.
	auto sdkDir = io::path(OPTIX_INCLUDE_DIR)+"/";
	auto addHeader = [&](const char* subpath) -> void
	{
		auto file = _filesystem->createAndOpenFile(sdkDir+subpath);
		if (!file)
			return;

		auto namelen = strlen(subpath);
		char* name = new char[namelen+1ull];
		memcpy(name,subpath,namelen);
		name[namelen] = 0;
		optixHeaderNames.push_back(name);
		char* data = new char[file->getSize()+1ull];
		file->read(data,file->getSize());
		data[file->getSize()] = 0;
		optixHeaders.push_back(data);

		file->drop();
	};
	addHeader("optix.h");
	addHeader("optix_device.h");
	addHeader("optix_7_device.h");
	addHeader("optix_7_types.h");
	addHeader("internal/optix_7_device_impl.h");
	addHeader("internal/optix_7_device_impl_exception.h");
	addHeader("internal/optix_7_device_impl_transformations.h");
	headersCreated = optixHeaders.size();

    optixHeaderNames.insert(optixHeaderNames.end(),cuda::CCUDAHandler::getCUDASTDHeaderNames().begin(),cuda::CCUDAHandler::getCUDASTDHeaderNames().end());
    optixHeaders.insert(optixHeaders.end(),cuda::CCUDAHandler::getCUDASTDHeaders().begin(),cuda::CCUDAHandler::getCUDASTDHeaders().end());
}

Manager::~Manager()
{
	for (uint32_t i=0u; i<headersCreated; i++)
		delete[] optixHeaderNames[i];
	for (uint32_t i=0u; i<headersCreated; i++)
		delete[] optixHeaders[i];

	for (uint32_t i=0u; i<contextCount; i++)
	{
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::cuda.pcuCtxPushCurrent_v2(context[i])))
			continue;
		cuda::CCUDAHandler::cuda.pcuCtxSynchronize();

		cuda::CCUDAHandler::cuda.pcuStreamDestroy_v2(stream[i]);
		if (ownContext[i])
			cuda::CCUDAHandler::cuda.pcuCtxDestroy_v2(context[i]);
	}
}


void Manager::defaultCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
	uint32_t contextID = reinterpret_cast<const uint32_t&>(cbdata);
	printf("irr::ext::OptiX Context:%d [%s]: %s\n", contextID, tag, message);
}


/*
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
*/
#include <numeric>
#include <filesystem>

#include "Renderer.h"

#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/asset/filters/CFillImageFilter.h"
#include "../source/Nabla/COpenCLHandler.h"
#include "COpenGLDriver.h"


#ifndef _NBL_BUILD_OPTIX_
	#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#endif

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;


constexpr uint32_t kOptiXPixelSize = sizeof(uint16_t)*3u;

core::smart_refctd_ptr<ICPUSpecializedShader> specializedShaderFromFile(IAssetManager* assetManager, const char* path)
{
	auto bundle = assetManager->getAsset(path, {});
	return core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(*bundle.getContents().begin());
}
core::smart_refctd_ptr<IGPUSpecializedShader> gpuSpecializedShaderFromFile(IAssetManager* assetManager, IVideoDriver* driver, const char* path)
{
	auto shader = specializedShaderFromFile(assetManager,path);
	// TODO: @Crisspl find a way to stop the user from such insanity as moving from the bundle's dynamic array
	//return std::move(driver->getGPUObjectsFromAssets<ICPUSpecializedShader>(&shader,&shader+1u)->operator[](0));
	return driver->getGPUObjectsFromAssets<ICPUSpecializedShader>(&shader,&shader+1u)->operator[](0);
}
// TODO: make these util function in `IDescriptorSetLayout` -> Assign: @Vib
auto fillIotaDescriptorBindingDeclarations = [](auto* outBindings, uint32_t accessFlags, uint32_t count, asset::E_DESCRIPTOR_TYPE descType=asset::EDT_INVALID, uint32_t startIndex=0u) -> void
{
	for (auto i=0u; i<count; i++)
	{
		outBindings[i].binding = i+startIndex;
		outBindings[i].type = descType;
		outBindings[i].count = 1u;
		outBindings[i].stageFlags = static_cast<ISpecializedShader::E_SHADER_STAGE>(accessFlags);
		outBindings[i].samplers = nullptr;
	}
};

Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, scene::ISceneManager* _smgr, bool useDenoiser) :
		m_useDenoiser(useDenoiser),	m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager),
		m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
		m_prevView(), m_prevCamTform(), m_sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
		m_framesDispatched(0u), m_rcpPixelSize{0.f,0.f},
		m_staticViewData{{0u,0u},0u,0u}, m_raytraceCommonData{core::matrix4SIMD(), vec3(),0.f,0u,0u,0u,0.f},
		m_indirectDrawBuffers{nullptr},m_cullPushConstants{core::matrix4SIMD(),1.f,0u,0u,0u},m_cullWorkGroups(0u),
		m_raygenWorkGroups{0u,0u},m_visibilityBuffer(nullptr),m_colorBuffer(nullptr)
{
	// TODO: reimplement
	m_useDenoiser = false;

	// set up raycount buffers
	{
		const uint32_t zeros[RAYCOUNT_N_BUFFERING] = { 0u };
		m_rayCountBuffer = m_driver->createFilledDeviceLocalBufferOnDedMem(sizeof(uint32_t)*RAYCOUNT_N_BUFFERING,zeros);
		IDeviceMemoryBacked::SDeviceMemoryRequirements reqs;
		reqs.vulkanReqs.size = sizeof(uint32_t);
		reqs.vulkanReqs.alignment = alignof(uint32_t);
		reqs.vulkanReqs.memoryTypeBits = ~0u;
		reqs.memoryHeapLocation = IDeviceMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
		reqs.mappingCapability = IDeviceMemoryAllocation::EMCF_COHERENT|IDeviceMemoryAllocation::EMCF_CAN_MAP_FOR_READ;
		reqs.prefersDedicatedAllocation = 0u;
		reqs.requiresDedicatedAllocation = 0u;
		m_littleDownloadBuffer = m_driver->createGPUBufferOnDedMem(reqs);
		m_littleDownloadBuffer->getBoundMemory()->mapMemoryRange(IDeviceMemoryAllocation::EMCAF_READ,{0,sizeof(uint32_t)});
	}

	// set up Visibility Buffer pipeline
	{
		IGPUDescriptorSetLayout::SBinding binding;
		fillIotaDescriptorBindingDeclarations(&binding,ISpecializedShader::ESS_VERTEX|ISpecializedShader::ESS_FRAGMENT,1u,asset::EDT_STORAGE_BUFFER);

		m_rasterInstanceDataDSLayout = m_driver->createDescriptorSetLayout(&binding,&binding+1u);
	}
	{
		constexpr auto additionalGlobalDescriptorCount = 5u;
		IGPUDescriptorSetLayout::SBinding bindings[additionalGlobalDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE|ISpecializedShader::ESS_VERTEX|ISpecializedShader::ESS_FRAGMENT,additionalGlobalDescriptorCount,asset::EDT_STORAGE_BUFFER);

		m_additionalGlobalDSLayout = m_driver->createDescriptorSetLayout(bindings,bindings+additionalGlobalDescriptorCount);
	}
	{
		constexpr auto cullingDescriptorCount = 3u;
		IGPUDescriptorSetLayout::SBinding bindings[cullingDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE|ISpecializedShader::ESS_VERTEX,cullingDescriptorCount,asset::EDT_STORAGE_BUFFER);
		bindings[2u].count = 2u;

		m_cullDSLayout = m_driver->createDescriptorSetLayout(bindings,bindings+cullingDescriptorCount);
	}
	m_perCameraRasterDSLayout = core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(m_cullDSLayout);
	{
		core::smart_refctd_ptr<IGPUSpecializedShader> shaders[] = {gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../fillVisBuffer.vert"),gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../fillVisBuffer.frag")};
		SPrimitiveAssemblyParams primitiveAssembly;
		primitiveAssembly.primitiveType = EPT_TRIANGLE_LIST;
		SRasterizationParams raster;
		raster.faceCullingMode = EFCM_NONE;
		auto _visibilityBufferFillPipelineLayout = m_driver->createPipelineLayout(
			nullptr,nullptr,
			core::smart_refctd_ptr(m_rasterInstanceDataDSLayout),
			core::smart_refctd_ptr(m_additionalGlobalDSLayout),
			core::smart_refctd_ptr(m_cullDSLayout)
		);
		m_visibilityBufferFillPipeline = m_driver->createGPURenderpassIndependentPipeline(
			nullptr,std::move(_visibilityBufferFillPipelineLayout),&shaders->get(),&shaders->get()+2u,
			SVertexInputParams{},SBlendParams{},primitiveAssembly,raster
		);
	}
	
	{
		constexpr auto raytracingCommonDescriptorCount = 8u;
		IGPUDescriptorSetLayout::SBinding bindings[raytracingCommonDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,raytracingCommonDescriptorCount);
		bindings[0].type = asset::EDT_UNIFORM_BUFFER;
		bindings[1].type = asset::EDT_UNIFORM_TEXEL_BUFFER;
		bindings[2].type = asset::EDT_STORAGE_IMAGE;
		bindings[3].type = asset::EDT_STORAGE_BUFFER;
		bindings[4].type = asset::EDT_STORAGE_BUFFER;
		bindings[5].type = asset::EDT_STORAGE_IMAGE;
		bindings[6].type = asset::EDT_STORAGE_IMAGE;
		bindings[7].type = asset::EDT_COMBINED_IMAGE_SAMPLER;

		m_commonRaytracingDSLayout = m_driver->createDescriptorSetLayout(bindings,bindings+raytracingCommonDescriptorCount);
	}

	ISampler::SParams samplerParams;
	samplerParams.TextureWrapU = samplerParams.TextureWrapV = samplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;
	samplerParams.MinFilter = samplerParams.MaxFilter = ISampler::ETF_NEAREST;
	samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
	samplerParams.AnisotropicFilter = 0u;
	samplerParams.CompareEnable = false;
	auto sampler = m_driver->createSampler(samplerParams);
	{
		constexpr auto raygenDescriptorCount = 3u;
		IGPUDescriptorSetLayout::SBinding bindings[raygenDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,raygenDescriptorCount,EDT_COMBINED_IMAGE_SAMPLER);
		bindings[0].samplers = &sampler;
		bindings[1].samplers = &sampler;
		bindings[2].type = asset::EDT_STORAGE_IMAGE;

		m_raygenDSLayout = m_driver->createDescriptorSetLayout(bindings,bindings+raygenDescriptorCount);
	}
	{
		constexpr auto closestHitDescriptorCount = 2u;
		IGPUDescriptorSetLayout::SBinding bindings[2];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,closestHitDescriptorCount,EDT_STORAGE_BUFFER);

		m_closestHitDSLayout = m_driver->createDescriptorSetLayout(bindings,bindings+closestHitDescriptorCount);
	}
	{
		constexpr auto resolveDescriptorCount = 7u;
		IGPUDescriptorSetLayout::SBinding bindings[resolveDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,resolveDescriptorCount);
		bindings[0].type = asset::EDT_UNIFORM_BUFFER;
		bindings[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[1].samplers = &sampler;
		bindings[2].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[2].samplers = &sampler;
		bindings[3].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[3].samplers = &sampler;
		bindings[4].type = asset::EDT_STORAGE_IMAGE;
		bindings[5].type = asset::EDT_STORAGE_IMAGE;
		bindings[6].type = asset::EDT_STORAGE_IMAGE;

		m_resolveDSLayout = m_driver->createDescriptorSetLayout(bindings,bindings+resolveDescriptorCount);
	}
}

Renderer::~Renderer()
{
	deinitSceneResources();
}


Renderer::InitializationData Renderer::initSceneObjects(const SAssetBundle& meshes)
{
	constexpr bool meshPackerUsesSSBO = true;
	using CPUMeshPacker = CCPUMeshPackerV2<DrawElementsIndirectCommand_t>;
	using GPUMeshPacker = CGPUMeshPackerV2<DrawElementsIndirectCommand_t>;

	// get primary (texture and material) global DS
	InitializationData retval;
	m_globalMeta  = meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>();
	assert(m_globalMeta );

	//
	{
		// extract integrator parameters
		std::stack<const ext::MitsubaLoader::CElementIntegrator*> integratorStack;
		integratorStack.push(&m_globalMeta->m_global.m_integrator);
		while (!integratorStack.empty())
		{
			auto integrator = integratorStack.top();
			integratorStack.pop();
			using Enum = ext::MitsubaLoader::CElementIntegrator::Type;
			switch (integrator->type)
			{
				case Enum::DIRECT:
					pathDepth = 2u;
					break;
				case Enum::PATH:
				case Enum::VOL_PATH_SIMPLE:
				case Enum::VOL_PATH:
				case Enum::BDPT:
					pathDepth = integrator->bdpt.maxPathDepth;
					noRussianRouletteDepth = integrator->bdpt.russianRouletteDepth-1u;
					break;
				case Enum::ADAPTIVE:
					for (size_t i=0u; i<integrator->multichannel.childCount; i++)
						integratorStack.push(integrator->multichannel.children[i]);
					break;
				case Enum::IRR_CACHE:
					assert(false);
					break;
				case Enum::MULTI_CHANNEL:
					for (size_t i=0u; i<integrator->multichannel.childCount; i++)
						integratorStack.push(integrator->multichannel.children[i]);
					break;
				default:
					break;
			};
		}

		//
		for (const auto& sensor : m_globalMeta->m_global.m_sensors)
		{
			if (maxSensorSamples<sensor.sampler.sampleCount)
				maxSensorSamples = sensor.sampler.sampleCount;
		}
	}

	//
	auto* _globalBackendDataDS = m_globalMeta ->m_global.m_ds0.get();

	auto* instanceDataDescPtr = _globalBackendDataDS->getDescriptors(5u).begin();
	assert(instanceDataDescPtr->desc->getTypeCategory()==IDescriptor::EC_BUFFER);
	auto* origInstanceData = reinterpret_cast<const ext::MitsubaLoader::instance_data_t*>(static_cast<ICPUBuffer*>(instanceDataDescPtr->desc.get())->getPointer());

	IGPUDescriptorSet::SDescriptorInfo infos[4];
	auto recordInfoBuffer = [](IGPUDescriptorSet::SDescriptorInfo& info, core::smart_refctd_ptr<IGPUBuffer>&& buf) -> void
	{
		info.buffer.size = buf->getSize();
		info.buffer.offset = 0u;
		info.desc = std::move(buf);
	};
	constexpr uint32_t writeBound = 3u;
	IGPUDescriptorSet::SWriteDescriptorSet writes[writeBound];
	auto recordSSBOWrite = [](IGPUDescriptorSet::SWriteDescriptorSet& write, IGPUDescriptorSet::SDescriptorInfo* infos, uint32_t binding, uint32_t count=1u) -> void
	{
		write.binding = binding;
		write.arrayElement = 0u;
		write.count = count;
		write.descriptorType = EDT_STORAGE_BUFFER;
		write.info = infos;
	};
	auto setDstSetOnAllWrites = [&writes,writeBound](IGPUDescriptorSet* dstSet) -> void
	{
		for (auto i=0u; i<writeBound; i++)
			writes[i].dstSet = dstSet;
	};
	// make secondary (geometry) DS
	m_additionalGlobalDS = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_additionalGlobalDSLayout));

	// one cull data per instace of a batch
	core::vector<CullData_t> cullData;
	{
		auto* rr = m_rrManager->getRadeonRaysAPI();
		// set up batches/meshlets, lights and culling data
		{
			auto contents = meshes.getContents();

			core::vector<IMeshPackerBase::PackedMeshBufferData> pmbd;
			// split into packed batches
			{
				// one instance data per instance of a batch
				core::smart_refctd_ptr<ICPUBuffer> newInstanceDataBuffer;

				constexpr uint16_t minTrisBatch = MAX_TRIANGLES_IN_BATCH>>1u;
				constexpr uint16_t maxTrisBatch = MAX_TRIANGLES_IN_BATCH;
				constexpr uint8_t minVertexSize = 
					asset::getTexelOrBlockBytesize<asset::EF_R32G32B32_SFLOAT>()+
					asset::getTexelOrBlockBytesize<asset::EF_A2R10G10B10_SNORM_PACK32>()+
					asset::getTexelOrBlockBytesize<asset::EF_R32G32_SFLOAT>();

				constexpr uint8_t kIndicesPerTriangle = 3u;
				constexpr uint16_t minIndicesBatch = minTrisBatch*kIndicesPerTriangle;

				CPUMeshPacker::AllocationParams allocParams;
				allocParams.vertexBuffSupportedByteSize = 1u<<31u;
				allocParams.vertexBufferMinAllocByteSize = minTrisBatch*minVertexSize;
				allocParams.indexBuffSupportedCnt = (allocParams.vertexBuffSupportedByteSize/allocParams.vertexBufferMinAllocByteSize)*minIndicesBatch;
				allocParams.indexBufferMinAllocCnt = minIndicesBatch;
				allocParams.MDIDataBuffSupportedCnt = allocParams.indexBuffSupportedCnt/minIndicesBatch;
				allocParams.MDIDataBuffMinAllocCnt = 1u; //so structs from different meshbuffers are adjacent in memory


				// TODO: after position moves to RGB21, need to split up normal from UV
				constexpr auto combinedNormalUVAttributeIx = 1;
				constexpr auto newEnabledAttributeMask = (0x1u<<combinedNormalUVAttributeIx)|0b1;

				IMeshPackerV2Base::SupportedFormatsContainer formats;
				formats.insert(EF_R32G32B32_SFLOAT);
				formats.insert(EF_R32G32_UINT);
				auto cpump = core::make_smart_refctd_ptr<CCPUMeshPackerV2<>>(allocParams,formats,minTrisBatch,maxTrisBatch);
				uint32_t mdiBoundMax=0u,batchInstanceBoundTotal=0u;
				core::vector<CPUMeshPacker::ReservedAllocationMeshBuffers> allocData;
				// virtually allocate and size the storage
				{
					core::vector<const ICPUMeshBuffer*> meshBuffersToProcess;
					meshBuffersToProcess.reserve(contents.size());
					// TODO: Optimize! Check which triangles need normals, bin into two separate meshbuffers, dont have normals for meshbuffers where all(abs(transpose(normals)*cross(pos1-pos0,pos2-pos0))~=1.f) 
					// TODO: Optimize! Check which materials use any textures, if meshbuffer doens't use any textures, its pipeline doesn't need UV coordinates
					// TODO: separate pipeline for stuff without UVs and separate out the barycentric derivative FBO attachment 
					for (const auto& asset : contents)
					{
						auto cpumesh = static_cast<asset::ICPUMesh*>(asset.get());
						auto meshBuffers = cpumesh->getMeshBuffers();

						assert(!meshBuffers.empty());
						const uint32_t instanceCount = (*meshBuffers.begin())->getInstanceCount();
						for (auto mbIt=meshBuffers.begin(); mbIt!=meshBuffers.end(); mbIt++)
						{
							auto meshBuffer = *mbIt;
							assert(meshBuffer->getInstanceCount()==instanceCount);
							// We'll disable certain attributes to ensure we only copy position, normal and uv attribute
							SVertexInputParams& vertexInput = meshBuffer->getPipeline()->getVertexInputParams();
							// but we'll pack normals and UVs together to save one SSBO binding (and quantize UVs to half floats)
							constexpr auto freeBinding = 15u;
							vertexInput.attributes[combinedNormalUVAttributeIx].binding = freeBinding;
							vertexInput.attributes[combinedNormalUVAttributeIx].format = EF_R32G32_UINT;
							vertexInput.attributes[combinedNormalUVAttributeIx].relativeOffset = 0u;
							vertexInput.enabledBindingFlags |= 0x1u<<freeBinding;
							vertexInput.bindings[freeBinding].inputRate = EVIR_PER_VERTEX;
							vertexInput.bindings[freeBinding].stride = 0u;
							const auto approxVxCount = IMeshManipulator::upperBoundVertexID(meshBuffer)+meshBuffer->getBaseVertex();
							struct CombinedNormalUV
							{
								uint32_t nml;
								uint16_t u,v;
							};
							auto newBuff = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(CombinedNormalUV)*approxVxCount);
							auto* dst = reinterpret_cast<CombinedNormalUV*>(newBuff->getPointer())+meshBuffer->getBaseVertex();
							meshBuffer->setVertexBufferBinding({0u,newBuff},freeBinding);
							// copy and pack data
							const auto normalAttr = meshBuffer->getNormalAttributeIx();
							vertexInput.attributes[normalAttr].format = EF_R32_UINT;
							for (auto i=0u; i<approxVxCount; i++)
							{
								meshBuffer->getAttribute(&dst[i].nml,normalAttr,i);
								core::vectorSIMDf uv;
								meshBuffer->getAttribute(uv,2u,i);
								dst[i].u = core::Float16Compressor::compress(uv.x);
								dst[i].v = core::Float16Compressor::compress(uv.y);
							}
						}

						const uint32_t mdiBound = cpump->calcMDIStructMaxCount(meshBuffers.begin(),meshBuffers.end());
						mdiBoundMax = core::max(mdiBound,mdiBoundMax);
						batchInstanceBoundTotal += mdiBound*instanceCount;

						meshBuffersToProcess.insert(meshBuffersToProcess.end(),meshBuffers.begin(),meshBuffers.end());
					}
					for (auto meshBuffer : meshBuffersToProcess)
						const_cast<ICPUMeshBuffer*>(meshBuffer)->getPipeline()->getVertexInputParams().enabledAttribFlags = newEnabledAttributeMask;

					allocData.resize(meshBuffersToProcess.size());

					cpump->alloc(allocData.data(),meshBuffersToProcess.begin(),meshBuffersToProcess.end());
					cpump->shrinkOutputBuffersSize();
					cpump->instantiateDataStorage();

					pmbd.resize(meshBuffersToProcess.size());
					cullData.reserve(batchInstanceBoundTotal);

					newInstanceDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ext::MitsubaLoader::instance_data_t)*batchInstanceBoundTotal);
				}
				// actually commit the physical memory, compute batches and set up instance data
				{
					auto allocDataIt = allocData.begin();
					auto pmbdIt = pmbd.begin();
					auto* indexPtr = reinterpret_cast<const uint16_t*>(cpump->getPackerDataStore().indexBuffer->getPointer());
					auto* vertexPtr = reinterpret_cast<const float*>(cpump->getPackerDataStore().vertexBuffer->getPointer());
					auto* mdiPtr = reinterpret_cast<DrawElementsIndirectCommand_t*>(cpump->getPackerDataStore().MDIDataBuffer->getPointer());
					auto* newInstanceData = reinterpret_cast<ext::MitsubaLoader::instance_data_t*>(newInstanceDataBuffer->getPointer());

					constexpr uint32_t kIndicesPerTriangle = 3u;
					core::vector<CPUMeshPacker::CombinedDataOffsetTable> cdot(mdiBoundMax);
					core::vector<core::aabbox3df> aabbs(mdiBoundMax);
					MDICall* mdiCall = nullptr;
					core::vector<int32_t> fatIndicesForRR(maxTrisBatch*kIndicesPerTriangle);
					for (const auto& asset : contents)
					{
						auto cpumesh = static_cast<asset::ICPUMesh*>(asset.get());
						const auto* meta = m_globalMeta ->getAssetSpecificMetadata(cpumesh);
						const auto& instanceData = meta->m_instances;
						const auto& instanceAuxData = meta->m_instanceAuxData;

						auto meshBuffers = cpumesh->getMeshBuffers();
						const uint32_t actualMdiCnt = cpump->commit(&*pmbdIt,cdot.data(),aabbs.data(),&*allocDataIt,meshBuffers.begin(),meshBuffers.end());
						allocDataIt += meshBuffers.size();
						if (actualMdiCnt==0u)
						{
							std::cout << "Commit failed" << std::endl;
							_NBL_DEBUG_BREAK_IF(true);
							pmbdIt += meshBuffers.size();
							continue;
						}

						const auto aabbMesh = cpumesh->getBoundingBox();
						// meshbuffers
						auto cdotIt = cdot.begin();
						auto aabbsIt = aabbs.begin();
						for (auto mb : meshBuffers)
						{
							assert(mb->getInstanceCount()==instanceData.size());
							const auto posAttrID = mb->getPositionAttributeIx();
							const auto* mbInstanceData = origInstanceData+mb->getBaseInstance();
							const bool frontFaceIsCCW = mb->getPipeline()->getRasterizationParams().frontFaceIsCCW;
							// batches/meshlets
							for (auto i=0u; i<pmbdIt->mdiParameterCount; i++)
							{
								const uint32_t drawCommandGUID = pmbdIt->mdiParameterOffset+i;
								auto& mdi = mdiPtr[drawCommandGUID];
								mdi.baseInstance = cullData.size();
								mdi.instanceCount = 0; // needs to be cleared, will be set by compute culling

								const uint32_t firstIndex = mdi.firstIndex;
								// set up BLAS
								const auto indexCount = mdi.count;
								std::copy_n(indexPtr+firstIndex,indexCount,fatIndicesForRR.data());
								rrShapes.emplace_back() = rr->CreateMesh(
									vertexPtr+cdotIt->attribInfo[posAttrID].getOffset()*sizeof(vec3)/sizeof(float),
									mdi.count, // could be improved if mesh packer returned the `usedVertices.size()` for every batch in the cdot
									asset::getTexelOrBlockBytesize<asset::EF_R32G32B32_SFLOAT>(),
									fatIndicesForRR.data(),
									sizeof(uint32_t)*kIndicesPerTriangle,nullptr, // radeon rays understands index stride differently to me
									indexCount/kIndicesPerTriangle
								);

								const auto thisShapeInstancesBeginIx = rrInstances.size();
								const auto& batchAABB = *aabbsIt;
								for (auto auxIt=instanceAuxData.begin(); auxIt!=instanceAuxData.end(); auxIt++)
								{
									const auto batchInstanceGUID = cullData.size();

									const auto instanceID = std::distance(instanceAuxData.begin(),auxIt);
									*newInstanceData = mbInstanceData[instanceID];
									//assert(instanceData.begin()[instanceID].worldTform==newInstanceData->tform); TODO: later
									newInstanceData->padding0 = firstIndex;
									newInstanceData->padding1 = reinterpret_cast<const uint32_t&>(cdotIt->attribInfo[posAttrID]);
									newInstanceData->determinantSignBit = core::bitfieldInsert(
										newInstanceData->determinantSignBit,
										reinterpret_cast<const uint32_t&>(cdotIt->attribInfo[combinedNormalUVAttributeIx]),
										0u,31u
									);
									if (frontFaceIsCCW) // compensate for Nabla's default camera being left handed
										newInstanceData->determinantSignBit ^= 0x80000000u;

									auto& c = cullData.emplace_back();
									c.aabbMinEdge.x = batchAABB.MinEdge.X;
									c.aabbMinEdge.y = batchAABB.MinEdge.Y;
									c.aabbMinEdge.z = batchAABB.MinEdge.Z;
									c.batchInstanceGUID = batchInstanceGUID;
									c.aabbMaxEdge.x = batchAABB.MaxEdge.X;
									c.aabbMaxEdge.y = batchAABB.MaxEdge.Y;
									c.aabbMaxEdge.z = batchAABB.MaxEdge.Z;
									c.drawCommandGUID = drawCommandGUID;

									rrInstances.emplace_back() = rr->CreateInstance(rrShapes.back());
									rrInstances.back()->SetId(batchInstanceGUID);
									ext::RadeonRays::Manager::shapeSetTransform(rrInstances.back(),newInstanceData->tform);

									// set up scene bounds and lights
									if (i==0u)
									{
										if (mb==*meshBuffers.begin())
											m_sceneBound.addInternalBox(core::transformBoxEx(aabbMesh,newInstanceData->tform));
										const auto& emitter = auxIt->frontEmitter;
										if (emitter.type!=ext::MitsubaLoader::CElementEmitter::Type::INVALID)
										{
											assert(emitter.type==ext::MitsubaLoader::CElementEmitter::Type::AREA);

											SLight newLight(aabbMesh,newInstanceData->tform); // TODO: should be an OBB

											const float weight = newLight.computeFluxBound(emitter.area.radiance)*emitter.area.samplingWeight;
											if (weight<=FLT_MIN)
												continue;

											retval.lights.emplace_back(std::move(newLight));
											retval.lightPDF.push_back(weight);
										}
									}

									newInstanceData++;
								}
								for (auto j=thisShapeInstancesBeginIx; j!=rrInstances.size(); j++)
									rr->AttachShape(rrInstances[j]);
								cdotIt++;
								aabbsIt++;
							}
							//
							if (!mdiCall || pmbdIt->mdiParameterOffset!=mdiCall->mdiOffset+mdiCall->mdiCount)
							{
								mdiCall = &m_mdiDrawCalls.emplace_back();
								mdiCall->mdiOffset = pmbdIt->mdiParameterOffset;
								mdiCall->mdiCount = 0u;
							}
							mdiCall->mdiCount += pmbdIt->mdiParameterCount;
							//
							pmbdIt++;
						}
					}
				}
				printf("Scene Bound: %f,%f,%f -> %f,%f,%f\n",
					m_sceneBound.MinEdge.X,
					m_sceneBound.MinEdge.Y,
					m_sceneBound.MinEdge.Z,
					m_sceneBound.MaxEdge.X,
					m_sceneBound.MaxEdge.Y,
					m_sceneBound.MaxEdge.Z
				);
				instanceDataDescPtr->buffer = {0u,cullData.size()*sizeof(ext::MitsubaLoader::instance_data_t)};
				instanceDataDescPtr->desc = std::move(newInstanceDataBuffer); // TODO: trim the buffer
				{
					auto gpump = core::make_smart_refctd_ptr<GPUMeshPacker>(m_driver,cpump.get());
					const auto& dataStore = gpump->getPackerDataStore();
					m_indexBuffer = dataStore.indexBuffer;
					// set up descriptor set for the inputs
					{
						for (auto i=0u; i<writeBound; i++)
						{
							recordInfoBuffer(infos[i],core::smart_refctd_ptr(dataStore.vertexBuffer));
							recordSSBOWrite(writes[i],infos+i,i);
						}
						recordInfoBuffer(infos[1],core::smart_refctd_ptr(m_indexBuffer));

						setDstSetOnAllWrites(m_additionalGlobalDS.get());
						m_driver->updateDescriptorSets(writeBound,writes,0u,nullptr);
					}
					// set up double buffering of MDI command buffers
					{
						m_indirectDrawBuffers[0] = dataStore.MDIDataBuffer;
						const auto mdiBufferSize = m_indirectDrawBuffers[0]->getSize();
						m_indirectDrawBuffers[1] = m_driver->createDeviceLocalGPUBufferOnDedMem(mdiBufferSize);
						m_driver->copyBuffer(m_indirectDrawBuffers[0].get(),m_indirectDrawBuffers[1].get(),0u,0u,mdiBufferSize);
					}
				}
			}
			m_cullPushConstants.maxDrawCommandCount = pmbd.back().mdiParameterOffset+pmbd.back().mdiParameterCount;
			m_cullPushConstants.maxGlobalInstanceCount = cullData.size();
		}

		// build TLAS with up to date transformations of instances
		rr->SetOption("bvh.sah.use_splits",1.f);
		rr->SetOption("bvh.builder","sah");
		// deinstance everything for great perf
		rr->SetOption("bvh.forceflat",1.f);
		rr->SetOption("acc.type","fatbvh");
		rr->Commit();
	}

	m_cullPushConstants.currentCommandBufferIx = 0x0u;
	m_cullWorkGroups = (m_cullPushConstants.maxGlobalInstanceCount-1u)/WORKGROUP_SIZE+1u;

	m_cullDS = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_cullDSLayout));
	m_perCameraRasterDS = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_perCameraRasterDSLayout));
	{
		recordInfoBuffer(infos[3],core::smart_refctd_ptr(m_indirectDrawBuffers[1]));
		recordInfoBuffer(infos[2],core::smart_refctd_ptr(m_indirectDrawBuffers[0]));
		recordInfoBuffer(infos[1],m_driver->createFilledDeviceLocalBufferOnDedMem(m_cullPushConstants.maxGlobalInstanceCount*sizeof(CullData_t),cullData.data()));
		cullData.clear();
		recordInfoBuffer(infos[0],m_driver->createDeviceLocalGPUBufferOnDedMem(m_cullPushConstants.maxGlobalInstanceCount*sizeof(DrawData_t)));
		
		recordSSBOWrite(writes[0],infos+0,0u);
		recordSSBOWrite(writes[1],infos+1,1u);
		recordSSBOWrite(writes[2],infos+2,2u,2u);

		setDstSetOnAllWrites(m_perCameraRasterDS.get());
		m_driver->updateDescriptorSets(1u,writes,0u,nullptr);
		setDstSetOnAllWrites(m_cullDS.get());
		m_driver->updateDescriptorSets(3u,writes,0u,nullptr);
	}
	
	// TODO: after port to new API, use a converter which does not generate mip maps
	m_globalBackendDataDS = m_driver->getGPUObjectsFromAssets(&_globalBackendDataDS,&_globalBackendDataDS+1)->front();
	// make a shortened version of the globalBackendDataDS
	m_rasterInstanceDataDS = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_rasterInstanceDataDSLayout));
	{
		IGPUDescriptorSet::SCopyDescriptorSet copy = {};
		copy.dstSet = m_rasterInstanceDataDS.get();
		copy.srcSet = m_globalBackendDataDS.get();
		copy.srcBinding = 5u;
		copy.srcArrayElement = 0u;
		copy.dstBinding = 0u;
		copy.dstArrayElement = 0u;
		copy.count = 1u;
		m_driver->updateDescriptorSets(0u,nullptr,1u,&copy);
	}
	return retval;
}

void Renderer::initSceneNonAreaLights(Renderer::InitializationData& initData)
{
	core::vectorSIMDf _envmapBaseColor;
	_envmapBaseColor.set(0.0f,0.0f,0.0f,1.f);

	for (const auto& emitter : m_globalMeta->m_global.m_emitters)
	{
		float weight = 0.f;
		switch (emitter.type)
		{
			case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
			{
				_envmapBaseColor += emitter.constant.radiance;
			}
				break;
			case ext::MitsubaLoader::CElementEmitter::Type::ENVMAP:
			{
				std::cout << "ENVMAP FOUND = " << std::endl;
				std::cout << "\tScale = " << emitter.envmap.scale << std::endl;
				std::cout << "\tGamma = " << emitter.envmap.gamma << std::endl;
				std::cout << "\tSamplingWeight = " << emitter.envmap.samplingWeight << std::endl;
				std::cout << "\tFileName = " << emitter.envmap.filename.svalue << std::endl;
				// LOAD file relative to the XML
			}
				break;
			case ext::MitsubaLoader::CElementEmitter::Type::INVALID:
				break;
			default:
			#ifdef _DEBUG
				assert(false);
			#endif
				// let's implement a new emitter type!
				//weight = emitter.unionType.samplingWeight;
				break;
		}
		if (weight==0.f)
			continue;
			
		//weight *= light.computeFlux(NAN);
		if (weight <= FLT_MIN)
			continue;

		//initData.lightPDF.push_back(weight);
		//initData.lights.push_back(light);
	}

	// Initialize Pipeline and Resources for EnvMap Blending
	auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(m_assetManager, m_driver);

	IGPUDescriptorSetLayout::SBinding binding{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	auto blendEnvDescriptorSetLayout = m_driver->createDescriptorSetLayout(&binding, &binding + 1u);
	
	IAssetLoader::SAssetLoadParams lp;
	auto fs_bundle = m_assetManager->getAsset("nbl/builtin/material/lambertian/singletexture/specialized_shader.frag",lp);
	auto fs_contents = fs_bundle.getContents();
	assert(!fs_contents.empty());
	
	ICPUSpecializedShader* fs = static_cast<ICPUSpecializedShader*>(fs_contents.begin()->get());
	
	auto fragShader = m_driver->getGPUObjectsFromAssets(&fs, &fs + 1)->front();
	if (!fragShader)
		std::cout << "[ERROR] Couldn't get fragShader." << std::endl;
	
	IGPUSpecializedShader* shaders[2] = { std::get<0>(fullScreenTriangle).get(), fragShader.get() };
	SBlendParams blendParams = {};
	blendParams.logicOpEnable = false;
	blendParams.logicOp = ELO_NO_OP;
	blendParams.blendParams[0].blendEnable = true;
	blendParams.blendParams[0].srcColorFactor = asset::EBF_ONE;
	blendParams.blendParams[0].dstColorFactor = asset::EBF_ONE;
	blendParams.blendParams[0].colorBlendOp = asset::EBO_ADD;
	blendParams.blendParams[0].srcAlphaFactor = asset::EBF_ONE;
	blendParams.blendParams[0].dstAlphaFactor = asset::EBF_ONE;
	blendParams.blendParams[0].alphaBlendOp = asset::EBO_ADD;
	blendParams.blendParams[0].colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);

	SRasterizationParams rasterParams = {};
	rasterParams.faceCullingMode = EFCM_NONE;
	rasterParams.depthCompareOp = ECO_ALWAYS;
	rasterParams.minSampleShading = 1.f;
	rasterParams.depthWriteEnable = false;
	rasterParams.depthTestEnable = false;

	auto gpuPipelineLayout = m_driver->createPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(blendEnvDescriptorSetLayout));

	blendEnvPipeline = m_driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), shaders, shaders + 2,
		std::get<SVertexInputParams>(fullScreenTriangle), blendParams,
		std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);
	
	SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
	blendEnvMeshBuffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
	blendEnvMeshBuffer->setIndexCount(3u);
	blendEnvMeshBuffer->setInstanceCount(1u);
	
	video::IFrameBuffer* finalEnvFramebuffer = nullptr;
	{
		const auto colorFormat = asset::EF_R16G16B16A16_SFLOAT;
		const auto mipCount = 13u;
		const auto resolution = 0x1u<<(mipCount-1u);

		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = resolution;
		imgInfo.extent.height = resolution/2;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = mipCount;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

		auto image = m_driver->createGPUImageOnDedMem(std::move(imgInfo), m_driver->getDeviceLocalGPUMemoryReqs());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = mipCount;

		m_finalEnvmap = m_driver->createImageView(std::move(imgViewInfo));

		finalEnvFramebuffer = m_driver->addFrameBuffer();
		finalEnvFramebuffer->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(m_finalEnvmap));
	}

	auto blendToFinalEnvMap = [&](nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView) -> void
	{
		auto blendEnvDescriptorSet = m_driver->createDescriptorSet(std::move(blendEnvDescriptorSetLayout));
		
		IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuImageView;
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			info.image.sampler = m_driver->createSampler(samplerParams);
			info.image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		IGPUDescriptorSet::SWriteDescriptorSet write;
		write.dstSet = blendEnvDescriptorSet.get();
		write.binding = 0u;
		write.arrayElement = 0u;
		write.count = 1u;
		write.descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		write.info = &info;

		m_driver->updateDescriptorSets(1u, &write, 0u, nullptr);
		m_driver->bindGraphicsPipeline(blendEnvPipeline.get());
		m_driver->bindDescriptorSets(EPBP_GRAPHICS, blendEnvPipeline->getLayout(), 3u, 1u, &blendEnvDescriptorSet.get(), nullptr);
		m_driver->drawMeshBuffer(blendEnvMeshBuffer.get());
	};
	
	m_driver->setRenderTarget(finalEnvFramebuffer, true);
	float colorClearValues[] = { _envmapBaseColor.x, _envmapBaseColor.y, _envmapBaseColor.z, _envmapBaseColor.w };
	m_driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, colorClearValues);
	for(uint32_t i = 0u; i < m_globalMeta->m_global.m_envMapImages.size(); ++i)
	{
		auto envmapCpuImage = m_globalMeta->m_global.m_envMapImages[i];
		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = envmapCpuImage;
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.viewType = IImageView<ICPUImage>::ET_2D;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;
		auto cpuEnvmapImageView = ICPUImageView::create(std::move(viewParams));
		auto envMapImageView = m_driver->getGPUObjectsFromAssets(&cpuEnvmapImageView.get(), &cpuEnvmapImageView.get() + 1u)->front();
		blendToFinalEnvMap(envMapImageView);
	}
	m_driver->setRenderTarget(nullptr, true);
	// always needs doing after rendering
	// TODO: better filter and GPU accelerated
	m_finalEnvmap->regenerateMipMapLevels();
}

void Renderer::finalizeScene(Renderer::InitializationData& initData)
{
	if (initData.lights.empty())
		return;
	m_staticViewData.lightCount = initData.lights.size();

	const double weightSum = std::accumulate(initData.lightPDF.begin(),initData.lightPDF.end(),0.0);
	assert(weightSum>FLT_MIN);

	constexpr double UINT_MAX_DOUBLE = double(0x1ull<<32ull);
	const double weightSumRcp = UINT_MAX_DOUBLE/weightSum;

	auto outCDF = initData.lightCDF.begin();

	auto inPDF = initData.lightPDF.begin();
	double partialSum = *inPDF;

	auto computeCDF = [UINT_MAX_DOUBLE,weightSumRcp,&partialSum,&outCDF](uint32_t prevCDF) -> void
	{
		const double exactCDF = weightSumRcp*partialSum+double(FLT_MIN);
		if (exactCDF<UINT_MAX_DOUBLE)
			*outCDF = static_cast<uint32_t>(exactCDF);
		else
		{
			assert(exactCDF<UINT_MAX_DOUBLE+1.0);
			*outCDF = 0xdeadbeefu;
		}
	};

	computeCDF(0u);
	for (auto prevCDF=outCDF++; outCDF!=initData.lightCDF.end(); prevCDF=outCDF++)
	{
		partialSum += double(*(++inPDF));

		computeCDF(*prevCDF);
	}
}

core::smart_refctd_ptr<IGPUImageView> Renderer::createScreenSizedTexture(E_FORMAT format, uint32_t layers)
{
	const auto real_layers = layers ? layers:1u;

	IGPUImage::SCreationParams imgparams;
	imgparams.extent = {m_staticViewData.imageDimensions.x,m_staticViewData.imageDimensions.y,1u};
	imgparams.arrayLayers = real_layers;
	imgparams.flags = static_cast<IImage::E_CREATE_FLAGS>(0);
	imgparams.format = format;
	imgparams.mipLevels = 1u;
	imgparams.samples = IImage::ESCF_1_BIT;
	imgparams.type = IImage::ET_2D;

	IGPUImageView::SCreationParams viewparams;
	viewparams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0);
	viewparams.format = format;
	viewparams.image = m_driver->createDeviceLocalGPUImageOnDedMem(std::move(imgparams));
	viewparams.viewType = layers ? IGPUImageView::ET_2D_ARRAY:IGPUImageView::ET_2D;
	viewparams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
	viewparams.subresourceRange.baseArrayLayer = 0u;
	viewparams.subresourceRange.layerCount = real_layers;
	viewparams.subresourceRange.baseMipLevel = 0u;
	viewparams.subresourceRange.levelCount = 1u;

	return m_driver->createImageView(std::move(viewparams));
}


core::smart_refctd_ptr<asset::ICPUBuffer> Renderer::SampleSequence::createCPUBuffer(uint32_t quantizedDimensions, uint32_t sampleCount)
{
	const size_t bytesize = SampleSequence::QuantizedDimensionsBytesize*quantizedDimensions*sampleCount;
	if (bytesize)
		return core::make_smart_refctd_ptr<asset::ICPUBuffer>(bytesize);
	else
		return nullptr;
}
void Renderer::SampleSequence::createBufferView(IVideoDriver* driver, core::smart_refctd_ptr<asset::ICPUBuffer>&& buff)
{
	auto gpubuf = driver->createFilledDeviceLocalBufferOnDedMem(buff->getSize(),buff->getPointer());
	bufferView = driver->createBufferView(gpubuf.get(),asset::EF_R32G32_UINT);
}
core::smart_refctd_ptr<ICPUBuffer> Renderer::SampleSequence::createBufferView(IVideoDriver* driver, uint32_t quantizedDimensions, uint32_t sampleCount)
{
	constexpr auto DimensionsPerQuanta = 3u;
	const auto dimensions = quantizedDimensions*DimensionsPerQuanta;
	core::OwenSampler sampler(dimensions,0xdeadbeefu);

	// Memory Order: 3 Dimensions, then multiple of sampling stragies per vertex, then depth, then sample ID
	auto buff = createCPUBuffer(quantizedDimensions,sampleCount);
	uint32_t(&pout)[][2] = *reinterpret_cast<uint32_t(*)[][2]>(buff->getPointer());
	// the horrible order of iteration over output memory is caused by the fact that certain samplers like the 
	// Owen Scramble sampler, have a large cache which needs to be generated separately for each dimension.
	for (auto metadim=0u; metadim<quantizedDimensions; metadim++)
	{
		const auto trudim = metadim*DimensionsPerQuanta;
		for (uint32_t i=0; i<sampleCount; i++)
			pout[i*quantizedDimensions+metadim][0] = sampler.sample(trudim+0u,i);
		for (uint32_t i=0; i<sampleCount; i++)
			pout[i*quantizedDimensions+metadim][1] = sampler.sample(trudim+1u,i);
		for (uint32_t i=0; i<sampleCount; i++)
		{
			const auto sample = sampler.sample(trudim+2u,i);
			const auto out = pout[i*quantizedDimensions+metadim];
			out[0] &= 0xFFFFF800u;
			out[0] |= sample>>21;
			out[1] &= 0xFFFFF800u;
			out[1] |= (sample>>10)&0x07FFu;
		}
	}
	// upload sequence to GPU
	createBufferView(driver,core::smart_refctd_ptr(buff));
	// return for caching
	return buff;
}

//

// TODO: be able to fail
void Renderer::initSceneResources(SAssetBundle& meshes, nbl::io::path&& _sampleSequenceCachePath)
{
	deinitSceneResources();


	// set up Descriptor Sets
	{
		// captures m_globalBackendDataDS, creates m_indirectDrawBuffers, sets up m_mdiDrawCalls ranges
		// creates m_additionalGlobalDS and m_cullDS, sets m_cullPushConstants and m_cullWorkgroups, creates m_perCameraRasterDS
		auto initData = initSceneObjects(meshes);
		{
			initSceneNonAreaLights(initData);
			finalizeScene(initData);
		}

		//
		{
			// i know what I'm doing
			auto globalBackendDataDSLayout = core::smart_refctd_ptr<IGPUDescriptorSetLayout>(const_cast<IGPUDescriptorSetLayout*>(m_globalBackendDataDS->getLayout()));

			// cull
			{
				SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t)};
				m_cullPipelineLayout = m_driver->createPipelineLayout(&range,&range+1u,core::smart_refctd_ptr(globalBackendDataDSLayout),core::smart_refctd_ptr(m_cullDSLayout),nullptr,nullptr);
			}

			SPushConstantRange raytracingCommonPCRange{ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t)};
			// raygen
			{
				m_raygenPipelineLayout = m_driver->createPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_additionalGlobalDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_raygenDSLayout)
				);

				m_raygenDS = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_raygenDSLayout));
			}

			// closest hit
			{
				m_closestHitPipelineLayout = m_driver->createPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_additionalGlobalDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_closestHitDSLayout)
				);
			}

			// resolve
			{
				m_resolvePipelineLayout = m_driver->createPipelineLayout(nullptr,nullptr,core::smart_refctd_ptr(m_resolveDSLayout));
				m_resolveDS = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_resolveDSLayout));
			}
			
			//
			auto setBufferInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, const core::smart_refctd_ptr<IGPUBuffer>& buffer) -> void
			{
				info->buffer.size = buffer->getSize();
				info->buffer.offset = 0u;
				info->desc = core::smart_refctd_ptr(buffer);
			};
			auto createFilledBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, size_t size, const void* data)
			{
				auto buf = m_driver->createFilledDeviceLocalBufferOnDedMem(size,data);
				setBufferInfo(info,core::smart_refctd_ptr(buf));
				return buf;
			};
			auto createFilledBufferAndSetUpInfoFromVector = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& vector)
			{
				return createFilledBufferAndSetUpInfo(info,vector.size()*sizeof(decltype(*vector.data())),vector.data());
			};
			auto setDstSetAndDescTypesOnWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, IGPUDescriptorSet::SDescriptorInfo* _infos, const std::initializer_list<asset::E_DESCRIPTOR_TYPE>& list, uint32_t baseBinding=0u)
			{
				auto typeIt = list.begin();
				for (auto i=0u; i<list.size(); i++)
				{
					writes[i].dstSet = dstSet;
					writes[i].binding = baseBinding+i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = *(typeIt++);
					writes[i].info = _infos+i;
				}
			};
			
			constexpr uint32_t MaxDescritorUpdates = 2u;
			IGPUDescriptorSet::SDescriptorInfo infos[MaxDescritorUpdates];
			IGPUDescriptorSet::SWriteDescriptorSet writes[MaxDescritorUpdates];

			size_t lightCDF_BufferSize = 0u;
			size_t lights_BufferSize = 0u;

			// set up rest of m_additionalGlobalDS
			if(initData.lights.empty())
			{
				std::cout << "\n[ERROR] No supported lights found in the scene.";
			}
			else
			{
				auto lightCDFBuffer = createFilledBufferAndSetUpInfoFromVector(infos+0,initData.lightCDF);
				auto lightsBuffer = createFilledBufferAndSetUpInfoFromVector(infos+1,initData.lights);
				lightCDF_BufferSize = lightCDFBuffer->getSize();
				lights_BufferSize = lightsBuffer->getSize();
				setDstSetAndDescTypesOnWrites(m_additionalGlobalDS.get(),writes,infos,{EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER},3u);
				m_driver->updateDescriptorSets(2u,writes,0u,nullptr);
			}

			std::cout << "\nScene Resources Initialized:" << std::endl;
			std::cout << "\tlightCDF = " << lightCDF_BufferSize << " bytes" << std::endl;
			std::cout << "\tlights = " << lights_BufferSize << " bytes" << std::endl;
			std::cout << "\tindexBuffer = " << m_indexBuffer->getSize() << " bytes" << std::endl;
			for (auto i=0u; i<2u; i++)
				std::cout << "\tIndirect Draw Buffers[" << i << "] = " << m_indirectDrawBuffers[i]->getSize() << " bytes" << std::endl;
		}
		
		// load sample cache
		{
			core::smart_refctd_ptr<ICPUBuffer> cachebuff;
			uint32_t cachedQuantizedDimensions=0u,cachedSampleCount=0u;
			{
				sampleSequenceCachePath = std::move(_sampleSequenceCachePath);
				io::IReadFile* cacheFile = m_assetManager->getFileSystem()->createAndOpenFile(sampleSequenceCachePath);
				if (cacheFile)
				{
					cacheFile->read(&cachedQuantizedDimensions,sizeof(cachedQuantizedDimensions));
					if (cachedQuantizedDimensions)
					{
						cachedSampleCount = (cacheFile->getSize()-cacheFile->getPos())/(cachedQuantizedDimensions*SampleSequence::QuantizedDimensionsBytesize);
						cachebuff = sampleSequence.createCPUBuffer(cachedQuantizedDimensions,cachedSampleCount);
						if (cachebuff)
							cacheFile->read(cachebuff->getPointer(),cachebuff->getSize());
					}
					cacheFile->drop();
				}
			}
			// lets keep path length within bounds of sanity
			constexpr auto MaxPathDepth = 255u;
			if (pathDepth==0)
			{
				printf("[ERROR] No suppoerted Integrator found in the Mitsuba XML, setting default.\n");
				pathDepth = DefaultPathDepth;
			}
			else if (pathDepth>MaxPathDepth)
			{
				printf("[WARNING] Path Depth %d greater than maximum supported, clamping to %d\n",pathDepth,MaxPathDepth);
				pathDepth = MaxPathDepth;
			}
			const uint32_t quantizedDimensions = SampleSequence::computeQuantizedDimensions(pathDepth);
			// The primary limiting factor is the precision of turning a fixed point grid sample to IEEE754 32bit float in the [0,1] range.
			// Mantissa is only 23 bits, and primary sample space low discrepancy sequence will start to produce duplicates
			// near 1.0 with exponent -1 after the sample count passes 2^24 elements.
			// Another limiting factor is our encoding of sample sequences, we only use 21bits per channel, so no duplicates till 2^21 samples.
			maxSensorSamples = core::min(0x1<<21,maxSensorSamples);
			if (cachedQuantizedDimensions>=quantizedDimensions && cachedSampleCount>=maxSensorSamples)
				sampleSequence.createBufferView(m_driver,std::move(cachebuff));
			else
			{
				printf("[INFO] Generating Low Discrepancy Sample Sequence Cache, please wait...\n");
				cachebuff = sampleSequence.createBufferView(m_driver,quantizedDimensions,maxSensorSamples);
				// save sequence
				io::IWriteFile* cacheFile = m_assetManager->getFileSystem()->createAndWriteFile(sampleSequenceCachePath);
				if (cacheFile)
				{
					cacheFile->write(&quantizedDimensions,sizeof(quantizedDimensions));
					cacheFile->write(cachebuff->getPointer(),cachebuff->getSize());
					cacheFile->drop();
				}
			}
			std::cout << "\tpathDepth = " << pathDepth << std::endl;
			std::cout << "\tnoRussianRouletteDepth = " << noRussianRouletteDepth << std::endl;
			std::cout << "\tmaxSamples = " << maxSensorSamples << std::endl;
		}
	}
	std::cout << std::endl;
}

void Renderer::deinitSceneResources()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	m_resolveDS = nullptr;
	m_raygenDS = nullptr;
	m_additionalGlobalDS = nullptr;
	m_rasterInstanceDataDS = nullptr;
	m_globalBackendDataDS = nullptr;

	m_perCameraRasterDS = nullptr;
	
	m_cullPipelineLayout = nullptr;
	m_raygenPipelineLayout = nullptr;
	m_closestHitPipelineLayout = nullptr;
	m_resolvePipelineLayout = nullptr;

	m_cullWorkGroups = 0u;
	m_cullPushConstants = {core::matrix4SIMD(),1.f,0u,0u,0u};
	m_cullDS = nullptr;
	m_mdiDrawCalls.clear();
	m_indirectDrawBuffers[1] = m_indirectDrawBuffers[0] = nullptr;
	m_indexBuffer = nullptr;

	m_raytraceCommonData = {core::matrix4SIMD(),vec3(),0.f,0,0,0,0.f};
	m_sceneBound = core::aabbox3df(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	
	m_finalEnvmap = nullptr;
	m_staticViewData = {{0u,0u},0u,0u};

	auto rr = m_rrManager->getRadeonRaysAPI();
	rr->DetachAll();
	for (auto instance : rrInstances)
	{
		rr->DeleteShape(instance);
	}
	rrInstances.clear();

	for (auto shape : rrShapes)
		rr->DeleteShape(shape);
	rrShapes.clear();

	pathDepth = DefaultPathDepth;
	noRussianRouletteDepth = 5u;
	maxSensorSamples = MaxFreeviewSamples;
}

void Renderer::initScreenSizedResources(uint32_t width, uint32_t height)
{
	m_staticViewData.imageDimensions = {width, height};
	m_rcpPixelSize = { 2.f/float(m_staticViewData.imageDimensions.x),-2.f/float(m_staticViewData.imageDimensions.y) };

	// figure out dispatch sizes
	m_raygenWorkGroups[0] = (m_staticViewData.imageDimensions.x-1u)/WORKGROUP_DIM+1u;
	m_raygenWorkGroups[1] = (m_staticViewData.imageDimensions.y-1u)/WORKGROUP_DIM+1u;

	const auto renderPixelCount = m_staticViewData.imageDimensions.x*m_staticViewData.imageDimensions.y;
	// figure out how much Samples Per Pixel Per Dispatch we can afford
	size_t scrambleBufferSize=0u;
	size_t raygenBufferSize=0u,intersectionBufferSize=0u;
	{
		m_staticViewData.pathDepth = pathDepth;
		m_staticViewData.noRussianRouletteDepth = noRussianRouletteDepth;

		uint32_t _maxRaysPerDispatch = 0u;
		auto setRayBufferSizes = [renderPixelCount,this,&_maxRaysPerDispatch,&raygenBufferSize,&intersectionBufferSize](uint32_t sampleMultiplier) -> void
		{
			m_staticViewData.samplesPerPixelPerDispatch = SAMPLING_STRATEGY_COUNT*sampleMultiplier;

			const size_t minimumSampleCountPerDispatch = static_cast<size_t>(renderPixelCount)*getSamplesPerPixelPerDispatch();
			_maxRaysPerDispatch = static_cast<uint32_t>(minimumSampleCountPerDispatch);
			const auto doubleBufferSampleCountPerDispatch = minimumSampleCountPerDispatch*2ull;

			raygenBufferSize = doubleBufferSampleCountPerDispatch*sizeof(::RadeonRays::ray);
			intersectionBufferSize = doubleBufferSampleCountPerDispatch*sizeof(::RadeonRays::Intersection);
		};
		// see how much we can bump the sample count per raster pass
		{
			uint32_t sampleMultiplier = 0u;
			const auto maxSSBOSize = core::min(m_driver->getMaxSSBOSize(),256u<<20);
			while (sampleMultiplier<0x10000u && raygenBufferSize<=maxSSBOSize && intersectionBufferSize<=maxSSBOSize)
				setRayBufferSizes(++sampleMultiplier);
			if (sampleMultiplier==1u)
				setRayBufferSizes(sampleMultiplier);
			printf("[INFO] Using %d samples (per pixel) per dispatch\n",getSamplesPerPixelPerDispatch());
		}
	}
	
	(std::ofstream("runtime_defines.glsl")
		<< "#define _NBL_EXT_MITSUBA_LOADER_VT_STORAGE_VIEW_COUNT " << m_globalMeta->m_global.getVTStorageViewCount() << "\n"
		<< m_globalMeta->m_global.m_materialCompilerGLSL_declarations
		<< "#define SAMPLE_SEQUENCE_STRIDE " << SampleSequence::computeQuantizedDimensions(pathDepth) << "\n"
		<< "#ifndef MAX_RAYS_GENERATED\n"
		<< "#	define MAX_RAYS_GENERATED " << getSamplesPerPixelPerDispatch() << "\n"
		<< "#endif\n"
	).close();
	
	compileShadersFuture = std::async(std::launch::async, [&]()
	{
		// cull
		m_cullGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../cull.comp");

		// raygen
		m_raygenGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../raygen.comp");

		// closest hit
		m_closestHitGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../closestHit.comp");

		// resolve
		m_resolveGPUShader = gpuSpecializedShaderFromFile(m_assetManager,m_driver,m_useDenoiser ? "../resolveForDenoiser.comp":"../resolve.comp");
		
		bool success = m_cullGPUShader && m_raygenGPUShader && m_closestHitGPUShader && m_resolveGPUShader;
		return success;
	});

	auto setBufferInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, const core::smart_refctd_ptr<IGPUBuffer>& buffer) -> void
	{
		info->buffer.size = buffer->getSize();
		info->buffer.offset = 0u;
		info->desc = core::smart_refctd_ptr(buffer);
	};

	auto createFilledBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, size_t size, const void* data)
	{
		auto buf = m_driver->createFilledDeviceLocalBufferOnDedMem(size,data);
		setBufferInfo(info,core::smart_refctd_ptr(buf));
		return buf;
	};
	auto createFilledBufferAndSetUpInfoFromStruct = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& _struct)
	{
		return createFilledBufferAndSetUpInfo(info,sizeof(_struct),&_struct);
	};
	auto createFilledBufferAndSetUpInfoFromVector = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& vector)
	{
		return createFilledBufferAndSetUpInfo(info,vector.size()*sizeof(decltype(*vector.data())),vector.data());
	};
	auto setImageInfo = [](IGPUDescriptorSet::SDescriptorInfo* info, const asset::E_IMAGE_LAYOUT imageLayout, core::smart_refctd_ptr<IGPUImageView>&& imageView) -> void
	{
		info->image.imageLayout = imageLayout;
		info->image.sampler = nullptr; // storage image dont have samplers, and the combined sampler image views we have all use immutable samplers
		info->desc = std::move(imageView);
	};
	auto createEmptyInteropBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, InteropBuffer& interopBuffer, size_t size) -> void
	{
		if (static_cast<COpenGLDriver*>(m_driver)->runningInRenderdoc()) // makes Renderdoc capture the modifications done by OpenCL
		{
			interopBuffer.buffer = m_driver->createUpStreamingGPUBufferOnDedMem(size);
			//interopBuffer.buffer->getBoundMemory()->mapMemoryRange(IDeviceMemoryAllocation::EMCAF_WRITE,{0u,size})
		}
		else
			interopBuffer.buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(size);
		interopBuffer.asRRBuffer = m_rrManager->linkBuffer(interopBuffer.buffer.get(), CL_MEM_READ_ONLY);

		info->buffer.size = size;
		info->buffer.offset = 0u;
		info->desc = core::smart_refctd_ptr(interopBuffer.buffer);
	};
	auto setDstSetAndDescTypesOnWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, IGPUDescriptorSet::SDescriptorInfo* _infos, const std::initializer_list<asset::E_DESCRIPTOR_TYPE>& list, uint32_t baseBinding=0u)
	{
		auto typeIt = list.begin();
		for (auto i=0u; i<list.size(); i++)
		{
			writes[i].dstSet = dstSet;
			writes[i].binding = baseBinding+i;
			writes[i].arrayElement = 0u;
			writes[i].count = 1u;
			writes[i].descriptorType = *(typeIt++);
			writes[i].info = _infos+i;
		}
	};

	// create out screen-sized textures
	m_accumulation = createScreenSizedTexture(EF_R32G32_UINT,m_staticViewData.samplesPerPixelPerDispatch);
	m_albedoAcc = createScreenSizedTexture(EF_R32_UINT,m_staticViewData.samplesPerPixelPerDispatch);
	m_normalAcc = createScreenSizedTexture(EF_R32_UINT,m_staticViewData.samplesPerPixelPerDispatch);
	m_tonemapOutput = createScreenSizedTexture(EF_R16G16B16A16_SFLOAT);
	m_albedoRslv = createScreenSizedTexture(EF_A2B10G10R10_UNORM_PACK32);
	m_normalRslv = createScreenSizedTexture(EF_R16G16B16A16_SFLOAT);

	constexpr uint32_t MaxDescritorUpdates = 8u;
	IGPUDescriptorSet::SDescriptorInfo infos[MaxDescritorUpdates];
	IGPUDescriptorSet::SWriteDescriptorSet writes[MaxDescritorUpdates];

	// set up m_commonRaytracingDS
	core::smart_refctd_ptr<IGPUBuffer> _staticViewDataBuffer;
	size_t staticViewDataBufferSize=0u;
	{
		_staticViewDataBuffer = createFilledBufferAndSetUpInfoFromStruct(infos+0,m_staticViewData);
		staticViewDataBufferSize = _staticViewDataBuffer->getSize();
		infos[1].desc = sampleSequence.getBufferView();
		setImageInfo(infos+2,asset::EIL_GENERAL,core::smart_refctd_ptr(m_accumulation));
		setImageInfo(infos+5,asset::EIL_GENERAL,core::smart_refctd_ptr(m_albedoAcc));
		setImageInfo(infos+6,asset::EIL_GENERAL,core::smart_refctd_ptr(m_normalAcc));

		// envmap
		setImageInfo(infos+7,asset::EIL_GENERAL,core::smart_refctd_ptr(m_finalEnvmap));
		ISampler::SParams samplerParams = { ISampler::ETC_REPEAT, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		infos[7].image.sampler = m_driver->createSampler(samplerParams);
		infos[7].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;

		createEmptyInteropBufferAndSetUpInfo(infos+3,m_rayBuffer[0],raygenBufferSize);
		setBufferInfo(infos+4,m_rayCountBuffer);
			
		for (auto i=0u; i<2u; i++)
			m_commonRaytracingDS[i] = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_commonRaytracingDSLayout));

		constexpr auto descriptorUpdateCount = 8u;
		setDstSetAndDescTypesOnWrites(m_commonRaytracingDS[0].get(),writes,infos,{
			EDT_UNIFORM_BUFFER,
			EDT_UNIFORM_TEXEL_BUFFER,
			EDT_STORAGE_IMAGE,
			EDT_STORAGE_BUFFER,
			EDT_STORAGE_BUFFER,
			EDT_STORAGE_IMAGE,
			EDT_STORAGE_IMAGE,
			EDT_COMBINED_IMAGE_SAMPLER,
		});
		m_driver->updateDescriptorSets(descriptorUpdateCount,writes,0u,nullptr);
		// set up second DS
		createEmptyInteropBufferAndSetUpInfo(infos+3,m_rayBuffer[1],raygenBufferSize);
		for (auto i=0u; i<descriptorUpdateCount; i++)
			writes[i].dstSet = m_commonRaytracingDS[1].get();
		m_driver->updateDescriptorSets(descriptorUpdateCount,writes,0u,nullptr);
	}

	// set up m_raygenDS
	core::smart_refctd_ptr<IGPUImageView> visibilityBuffer = createScreenSizedTexture(EF_R32G32B32A32_UINT);
	{
		{
			constexpr auto ScrambleStateChannels = 2u;
			auto tmpBuff = m_driver->createCPUSideGPUVisibleGPUBufferOnDedMem(sizeof(uint32_t)*ScrambleStateChannels*renderPixelCount);
			// generate (maybe let's improve the scramble key beginning distribution)
			{
				core::RandomSampler rng(0xbadc0ffeu);
				auto it = reinterpret_cast<uint32_t*>(tmpBuff->getBoundMemory()->mapMemoryRange(
					IDeviceMemoryAllocation::EMCAF_WRITE,
					IDeviceMemoryAllocation::MemoryRange(0u,tmpBuff->getSize())
				));
				for (auto end=it+ScrambleStateChannels*renderPixelCount; it!=end; it++)
					*it = rng.nextSample();
				tmpBuff->getBoundMemory()->unmapMemory();
			}
			scrambleBufferSize = tmpBuff->getSize();
			// upload
			IGPUImage::SBufferCopy region;
			//region.imageSubresource.aspectMask = ;
			region.imageSubresource.baseArrayLayer = 0u;
			region.imageSubresource.layerCount = 1u;
			region.imageExtent = {m_staticViewData.imageDimensions.x,m_staticViewData.imageDimensions.y,0u};
			auto scrambleKeys = createScreenSizedTexture(EF_R32G32_UINT);
			m_driver->copyBufferToImage(tmpBuff.get(),scrambleKeys->getCreationParameters().image.get(),1u,&region);
			setImageInfo(infos+0,asset::EIL_SHADER_READ_ONLY_OPTIMAL,std::move(scrambleKeys));
		}
		setImageInfo(infos+1,asset::EIL_SHADER_READ_ONLY_OPTIMAL,core::smart_refctd_ptr(visibilityBuffer));
		setImageInfo(infos+2,asset::EIL_GENERAL,core::smart_refctd_ptr(m_tonemapOutput));

		setDstSetAndDescTypesOnWrites(m_raygenDS.get(),writes,infos,{
			EDT_COMBINED_IMAGE_SAMPLER,
			EDT_COMBINED_IMAGE_SAMPLER,
			EDT_STORAGE_IMAGE
		});
	}
	m_driver->updateDescriptorSets(3u,writes,0u,nullptr);

	// set up m_closestHitDS
	for (auto i=0u; i<2u; i++)
	{
		const auto other = i^0x1u;
		infos[0u].desc = m_rayBuffer[other].buffer;
		infos[0u].buffer.offset = 0u;
		infos[0u].buffer.size = m_rayBuffer[other].buffer->getSize();
		createEmptyInteropBufferAndSetUpInfo(infos+1,m_intersectionBuffer[other],intersectionBufferSize);
				
		m_closestHitDS[i] = m_driver->createDescriptorSet(core::smart_refctd_ptr(m_closestHitDSLayout));

		setDstSetAndDescTypesOnWrites(m_closestHitDS[i].get(),writes,infos,{EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER});
		m_driver->updateDescriptorSets(2u,writes,0u,nullptr);
	}

	// set up m_resolveDS
	{
		infos[0].buffer = {0u,_staticViewDataBuffer->getSize()};
		infos[0].desc = std::move(_staticViewDataBuffer);
		setImageInfo(infos+1,asset::EIL_GENERAL,core::smart_refctd_ptr(m_accumulation));
		core::smart_refctd_ptr<IGPUImageView> albedoSamplerView;
		{
			IGPUImageView::SCreationParams viewparams = m_albedoAcc->getCreationParameters();
			viewparams.format = EF_A2B10G10R10_UNORM_PACK32;
			albedoSamplerView = m_driver->createImageView(std::move(viewparams));
		}
		setImageInfo(infos+2,asset::EIL_GENERAL,std::move(albedoSamplerView));
		setImageInfo(infos+3,asset::EIL_GENERAL,core::smart_refctd_ptr(m_normalAcc));
		setImageInfo(infos+4,asset::EIL_GENERAL,core::smart_refctd_ptr(m_tonemapOutput));
		core::smart_refctd_ptr<IGPUImageView> albedoStorageView;
		{
			IGPUImageView::SCreationParams viewparams = m_albedoRslv->getCreationParameters();
			viewparams.format = EF_R32_UINT;
			albedoStorageView = m_driver->createImageView(std::move(viewparams));
		}
		setImageInfo(infos+5,asset::EIL_GENERAL,std::move(albedoStorageView));
		setImageInfo(infos+6,asset::EIL_GENERAL,core::smart_refctd_ptr(m_normalRslv));
				
		setDstSetAndDescTypesOnWrites(m_resolveDS.get(),writes,infos,{
			EDT_UNIFORM_BUFFER,
			EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,
			EDT_STORAGE_IMAGE,EDT_STORAGE_IMAGE,EDT_STORAGE_IMAGE
		});
	}
	m_driver->updateDescriptorSets(7u,writes,0u,nullptr);

	m_visibilityBuffer = m_driver->addFrameBuffer();
	m_visibilityBuffer->attach(EFAP_DEPTH_ATTACHMENT,createScreenSizedTexture(EF_D32_SFLOAT));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT0,std::move(visibilityBuffer));

	m_colorBuffer = m_driver->addFrameBuffer();
	m_colorBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_tonemapOutput));

	std::cout << "\nScreen Sized Resources have been initialized (" << width << "x" << height << ")" << std::endl;
	std::cout << "\tStaticViewData = " << staticViewDataBufferSize << " bytes" << std::endl;
	std::cout << "\tScrambleBuffer = " << scrambleBufferSize << " bytes" << std::endl;
	std::cout << "\tSampleSequence = " << sampleSequence.getBufferView()->getByteSize() << " bytes" << std::endl;
	std::cout << "\tRayCount Buffer = " << m_rayCountBuffer->getSize() << " bytes" << std::endl;
	for (auto i=0u; i<2u; i++)
		std::cout << "\tIntersection Buffer[" << i << "] = " << m_intersectionBuffer[i].buffer->getSize() << " bytes" << std::endl;
	for (auto i=0u; i<2u; i++)
		std::cout << "\tRay Buffer[" << i << "] = " << m_rayBuffer[i].buffer->getSize() << " bytes" << std::endl;
	std::cout << std::endl;
}

void Renderer::deinitScreenSizedResources()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	// make sure descriptor sets dont dangle
	//m_driver->bindDescriptorSets(video::EPBP_COMPUTE,nullptr,0u,4u,nullptr);
	m_closestHitDS[0] = m_closestHitDS[1] = nullptr;
	m_commonRaytracingDS[0] = m_commonRaytracingDS[1] = nullptr;
	
	// unset the framebuffer (dangling smartpointer in state cache can prevent the framebuffer from being dropped until the next framebuffer set)
	m_driver->setRenderTarget(nullptr,false);
	if (m_visibilityBuffer)
	{
		m_driver->removeFrameBuffer(m_visibilityBuffer);
		m_visibilityBuffer = nullptr;
	}
	if (m_colorBuffer)
	{
		m_driver->removeFrameBuffer(m_colorBuffer);
		m_colorBuffer = nullptr;
	}
	m_accumulation = m_tonemapOutput = nullptr;
	m_albedoAcc = m_albedoRslv = nullptr;
	m_normalAcc = m_normalRslv = nullptr;

	glFinish();
	
	// wait for OpenCL to finish
	ocl::COpenCLHandler::ocl.pclFlush(commandQueue);
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);
	for (auto i=0; i<2u; i++)
	{
		auto deleteInteropBuffer = [&](InteropBuffer& buffer) -> void
		{
			m_rrManager->unlinkBuffer(std::move(buffer.asRRBuffer));
			buffer = {};
		};
		deleteInteropBuffer(m_intersectionBuffer[i]);
		deleteInteropBuffer(m_rayBuffer[i]);
	}

	m_raygenWorkGroups[0] = m_raygenWorkGroups[1] = 0u;
	
	m_cullPipeline = nullptr;
	m_raygenPipeline = nullptr;
	m_closestHitPipeline = nullptr;
	m_resolvePipeline = nullptr;

	m_staticViewData.imageDimensions = {0u, 0u};
	m_staticViewData.pathDepth = DefaultPathDepth;
	m_staticViewData.noRussianRouletteDepth = 5u;
	m_staticViewData.samplesPerPixelPerDispatch = 1u;
	m_totalRaysCast = 0ull;
	m_rcpPixelSize = {0.f,0.f};
	m_framesDispatched = 0u;
	std::fill_n(m_prevView.pointer(),12u,0.f);
	m_prevCamTform = nbl::core::matrix4x3();
}

void Renderer::resetSampleAndFrameCounters()
{
	m_totalRaysCast = 0ull;
	m_framesDispatched = 0u;
	std::fill_n(m_prevView.pointer(),12u,0.f);
	m_prevCamTform = nbl::core::matrix4x3();
}

void Renderer::takeAndSaveScreenShot(const std::filesystem::path& screenshotFilePath, bool denoise, const DenoiserArgs& denoiserArgs)
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	// we always decode to 16bit HDR because thats what the denoiser takes
	// if we save to PNG instead of EXR, it will be converted and clamped once more automagically
	const asset::E_FORMAT format = asset::EF_R16G16B16A16_SFLOAT;

	auto filename_wo_ext = screenshotFilePath;
	filename_wo_ext.replace_extension();
	if (m_tonemapOutput)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_tonemapOutput.get(),filename_wo_ext.string()+".exr",format);
	if (m_albedoRslv)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_albedoRslv.get(),filename_wo_ext.string()+"_albedo.exr",format);
	if (m_normalRslv)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_normalRslv.get(),filename_wo_ext.string()+"_normal.exr",format);

	if(denoise)
	{
		const std::string defaultBloomFile = "../../media/kernels/physical_flare_512.exr";
		const std::string defaultTonemapperArgs = "ACES=0.4,0.8";
		constexpr auto defaultBloomScale = 0.1f;
		constexpr auto defaultBloomIntensity = 0.1f;
		auto bloomFilePathStr = (denoiserArgs.bloomFilePath.string().empty()) ? defaultBloomFile : denoiserArgs.bloomFilePath.string();
		auto bloomScale = (denoiserArgs.bloomScale == 0.0f) ? defaultBloomScale : denoiserArgs.bloomScale;
		auto bloomIntensity = (denoiserArgs.bloomIntensity == 0.0f) ? defaultBloomIntensity : denoiserArgs.bloomIntensity;
		auto tonemapperArgs = (denoiserArgs.tonemapperArgs.empty()) ? defaultTonemapperArgs : denoiserArgs.tonemapperArgs;

		std::ostringstream denoiserCmd;
		// 1.ColorFile 2.AlbedoFile 3.NormalFile 4.BloomPsfFilePath(STRING) 5.BloomScale(FLOAT) 6.BloomIntensity(FLOAT) 7.TonemapperArgs(STRING)
		denoiserCmd << "call ../denoiser_hook.bat";
		denoiserCmd << " \"" << filename_wo_ext.string() << ".exr" << "\"";
		denoiserCmd << " \"" << filename_wo_ext.string() << "_albedo.exr" << "\"";
		denoiserCmd << " \"" << filename_wo_ext.string() << "_normal.exr" << "\"";
		denoiserCmd << " \"" << bloomFilePathStr << "\"";
		denoiserCmd << " " << bloomScale;
		denoiserCmd << " " << bloomIntensity;
		denoiserCmd << " " << "\"" << tonemapperArgs << "\"";
		// NOTE/TODO/FIXME : Do as I say, not as I do
		// https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177
		std::cout << "\n---[DENOISER_BEGIN]---" << std::endl;
		std::system(denoiserCmd.str().c_str());
		std::cout << "\n---[DENOISER_END]---" << std::endl;
	}
}

void Renderer::denoiseCubemapFaces(
	std::filesystem::path filePaths[6],
	const std::string& mergedFileName,
	int borderPixels,
	const DenoiserArgs& denoiserArgs)
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

	std::string renderFilePaths[6] = {};
	std::string albedoFilePaths[6] = {};
	std::string normalFilePaths[6] = {};
	for(uint32_t i = 0; i < 6; ++i)
		renderFilePaths[i] = filePaths[i].replace_extension().string() + ".exr";
	for(uint32_t i = 0; i < 6; ++i)
		albedoFilePaths[i] = filePaths[i].replace_extension().string() + "_albedo.exr";
	for(uint32_t i = 0; i < 6; ++i)
		normalFilePaths[i] = filePaths[i].replace_extension().string() + "_normal.exr";
	
	std::string mergedRenderFilePath = mergedFileName + ".exr";
	std::string mergedAlbedoFilePath = mergedFileName + "_albedo.exr";
	std::string mergedNormalFilePath = mergedFileName + "_normal.exr";
	std::string mergedDenoisedFilePath = mergedFileName + "_denoised.exr";
	
	std::ostringstream mergeRendersCmd;
	mergeRendersCmd << "call ../mergeCubemap.bat";
	for(uint32_t i = 0; i < 6; ++i)
		mergeRendersCmd << " " << renderFilePaths[i];
	mergeRendersCmd << " " << mergedRenderFilePath;
	std::system(mergeRendersCmd.str().c_str());

	std::ostringstream mergeAlbedosCmd;
	mergeAlbedosCmd << "call ../mergeCubemap.bat ";
	for(uint32_t i = 0; i < 6; ++i)
		mergeAlbedosCmd << " " << albedoFilePaths[i];
	mergeAlbedosCmd << " " << mergedAlbedoFilePath;
	std::system(mergeAlbedosCmd.str().c_str());
	
	std::ostringstream mergeNormalsCmd;
	mergeNormalsCmd << "call ../mergeCubemap.bat ";
	for(uint32_t i = 0; i < 6; ++i)
		mergeNormalsCmd << " " << normalFilePaths[i];
	mergeNormalsCmd << " " << mergedNormalFilePath;
	std::system(mergeNormalsCmd.str().c_str());

	const std::string defaultBloomFile = "../../media/kernels/physical_flare_512.exr";
	const std::string defaultTonemapperArgs = "ACES=0.4,0.8";
	constexpr auto defaultBloomScale = 0.1f;
	constexpr auto defaultBloomIntensity = 0.1f;
	auto bloomFilePathStr = (denoiserArgs.bloomFilePath.string().empty()) ? defaultBloomFile : denoiserArgs.bloomFilePath.string();
	auto bloomScale = (denoiserArgs.bloomScale == 0.0f) ? defaultBloomScale : denoiserArgs.bloomScale;
	auto bloomIntensity = (denoiserArgs.bloomIntensity == 0.0f) ? defaultBloomIntensity : denoiserArgs.bloomIntensity;
	auto tonemapperArgs = (denoiserArgs.tonemapperArgs.empty()) ? defaultTonemapperArgs : denoiserArgs.tonemapperArgs;
	
	std::ostringstream denoiserCmd;
	// 1.ColorFile 2.AlbedoFile 3.NormalFile 4.BloomPsfFilePath(STRING) 5.BloomScale(FLOAT) 6.BloomIntensity(FLOAT) 7.TonemapperArgs(STRING)
	denoiserCmd << "call ../denoiser_hook.bat";
	denoiserCmd << " \"" << mergedRenderFilePath << "\"";
	denoiserCmd << " \"" << mergedAlbedoFilePath << "\"";
	denoiserCmd << " \"" << mergedNormalFilePath << "\"";
	denoiserCmd << " \"" << bloomFilePathStr << "\"";
	denoiserCmd << " " << bloomScale;
	denoiserCmd << " " << bloomIntensity;
	denoiserCmd << " " << "\"" << tonemapperArgs << "\"";
	// NOTE/TODO/FIXME : Do as I say, not as I do
	// https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152177
	std::cout << "\n---[DENOISER_BEGIN]---" << std::endl;
	std::system(denoiserCmd.str().c_str());
	std::cout << "\n---[DENOISER_END]---" << std::endl;
	
	auto extractCubemapFaces = [&](const std::string& extension) -> void
	{
		std::ostringstream extractImagesCmd;
		auto mergedDenoisedWithoutExtension = std::filesystem::path(mergedDenoisedFilePath).replace_extension().string();
		extractImagesCmd << "call ../extractCubemap.bat ";
		extractImagesCmd << " " << std::to_string(borderPixels);
		extractImagesCmd << " " << mergedDenoisedWithoutExtension + extension;
		for(uint32_t i = 0; i < 6; ++i)
		{
			auto renderFilePathWithoutExtension = std::filesystem::path(renderFilePaths[i]).replace_extension().string();
			extractImagesCmd << " " << renderFilePathWithoutExtension + "_denoised" + extension;
		}
		std::system(extractImagesCmd.str().c_str());
	};

	extractCubemapFaces(".exr");
	extractCubemapFaces(".png");
	extractCubemapFaces(".jpg");
}

// one day it will just work like that
//#include <nbl/builtin/glsl/sampling/box_muller_transform.glsl>

bool Renderer::render(nbl::ITimer* timer, const bool transformNormals, const bool beauty)
{
	if (m_cullPushConstants.maxGlobalInstanceCount==0u)
		return true;

	auto camera = m_smgr->getActiveCamera();
	camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(timer->getTime()).count());
	camera->render();

	// check if camera moved
	{
		auto properEquals = [](const core::matrix4x3& lhs, const core::matrix4x3& rhs) -> bool
		{
			const float rotationTolerance = 1.01f;
			const float positionTolerance = 1.005f;
			for (auto r=0; r<3u; r++)
			for (auto c=0; c<4u; c++)
			{
				const float ratio = core::abs((&rhs.getColumn(c).X)[r]/(&lhs.getColumn(c).X)[r]);
				// TODO: do by ULP
				if (core::isnan(ratio) || core::isinf(ratio))
					continue;
				const float tolerance = c!=3u ? rotationTolerance:positionTolerance;
				if (ratio>tolerance || ratio*tolerance<1.f)
					return false;
			}
			return true;
		};
		auto tform = camera->getRelativeTransformationMatrix();
		if (!properEquals(tform,m_prevCamTform))
		{
			m_framesDispatched = 0u;		
			m_prevView = camera->getViewMatrix();
			m_prevCamTform = tform;
		}
		else // need this to stop mouse cursor drift
			camera->setRelativeTransformationMatrix(m_prevCamTform);
	}

	// TODO: update positions and rr->Commit() if stuff starts to move

	if(compileShadersFuture.valid())
	{
		bool compiledShaders = compileShadersFuture.get();
		if(compiledShaders)
		{
			m_cullPipeline = m_driver->createComputePipeline(nullptr,core::smart_refctd_ptr(m_cullPipelineLayout), core::smart_refctd_ptr(m_cullGPUShader));
			m_raygenPipeline = m_driver->createComputePipeline(nullptr,core::smart_refctd_ptr(m_raygenPipelineLayout), core::smart_refctd_ptr(m_raygenGPUShader));
			m_closestHitPipeline = m_driver->createComputePipeline(nullptr,core::smart_refctd_ptr(m_closestHitPipelineLayout), core::smart_refctd_ptr(m_closestHitGPUShader));
			m_resolvePipeline = m_driver->createComputePipeline(nullptr,core::smart_refctd_ptr(m_resolvePipelineLayout), core::smart_refctd_ptr(m_resolveGPUShader));
			bool createPipelinesSuceess = m_cullPipeline && m_raygenPipeline && m_closestHitPipeline && m_resolvePipeline;
			if(!createPipelinesSuceess)
				std::cout << "Pipeline Compilation Failed." << std::endl;
		}
		else
			std::cout << "Shader Compilation Failed." << std::endl;
	}

	// only advance frame if rendering a beauty
	if (beauty)
		m_framesDispatched++;

	// raster jittered frame
	{
		// jitter with AA AntiAliasingSequence
		const auto modifiedViewProj = [&](uint32_t frameID)
		{
			const float stddev = 0.5f;
			const float* sample = AntiAliasingSequence[frameID%AntiAliasingSequenceLength];
			const float phi = core::PI<float>()*(2.f*sample[1]-1.f);
			const float sinPhi = sinf(phi);
			const float cosPhi = cosf(phi);
			const float truncated = sample[0]*0.99999f+0.00001f;
			const float r = sqrtf(-2.f*logf(truncated))*stddev;
			core::matrix4SIMD jitterMatrix;
			jitterMatrix.rows[0][3] = cosPhi*r*m_rcpPixelSize.x;
			jitterMatrix.rows[1][3] = sinPhi*r*m_rcpPixelSize.y;
			return core::concatenateBFollowedByA(jitterMatrix,core::concatenateBFollowedByA(camera->getProjectionMatrix(),m_prevView));
		}(m_framesDispatched);
		m_raytraceCommonData.rcpFramesDispatched = 1.f/float(m_framesDispatched);
		m_raytraceCommonData.textureFootprintFactor = core::inversesqrt(core::min<float>(m_framesDispatched,Renderer::AntiAliasingSequenceLength));
		if(!modifiedViewProj.getInverseTransform<core::matrix4SIMD::E_MATRIX_INVERSE_PRECISION::EMIP_64BBIT>(m_raytraceCommonData.viewProjMatrixInverse))
			std::cout << "Couldn't calculate viewProjection matrix's inverse. something is wrong." << std::endl;
		// for (auto i=0u; i<3u; i++)
		// 	m_raytraceCommonData.ndcToV.rows[i] = inverseMVP.rows[3]*cameraPosition[i]-inverseMVP.rows[i];
		// cull batches
		m_driver->bindComputePipeline(m_cullPipeline.get());
		{
			const auto* _cullPipelineLayout = m_cullPipeline->getLayout();

			IGPUDescriptorSet* descriptorSets[] = { m_globalBackendDataDS.get(),m_cullDS.get() };
			m_driver->bindDescriptorSets(EPBP_COMPUTE,_cullPipelineLayout,0u,2u,descriptorSets,nullptr);
			
			m_cullPushConstants.viewProjMatrix = modifiedViewProj;
			m_cullPushConstants.viewProjDeterminant = core::determinant(modifiedViewProj);
			m_driver->pushConstants(_cullPipelineLayout,ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t),&m_cullPushConstants);
		}
		// TODO: Occlusion Culling against HiZ Buffer
		m_driver->dispatch(m_cullWorkGroups, 1u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

		m_driver->setRenderTarget(m_visibilityBuffer);
		{ // clear
			m_driver->clearZBuffer();
			uint32_t clearTriangleID[4] = {0xffffffffu,0,0,0};
			m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, clearTriangleID);
		}
		// all batches draw with the same pipeline
		m_driver->bindGraphicsPipeline(m_visibilityBufferFillPipeline.get());
		{
			IGPUDescriptorSet* descriptorSets[] = { m_rasterInstanceDataDS.get(),m_additionalGlobalDS.get(),m_cullDS.get() };
			m_driver->bindDescriptorSets(EPBP_GRAPHICS,m_visibilityBufferFillPipeline->getLayout(),0u,3u,descriptorSets,nullptr);
		}
		for (const auto& call : m_mdiDrawCalls)
		{
			const asset::SBufferBinding<IGPUBuffer> nullBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT] = {};
			m_driver->drawIndexedIndirect(
				nullBindings,EPT_TRIANGLE_LIST,EIT_16BIT,m_indexBuffer.get(),
				m_indirectDrawBuffers[m_cullPushConstants.currentCommandBufferIx].get(),
				call.mdiOffset*sizeof(DrawElementsIndirectCommand_t),call.mdiCount,sizeof(DrawElementsIndirectCommand_t)
			);
		}
		// flip MDI buffers
		m_cullPushConstants.currentCommandBufferIx ^= 0x01u;

		// prepare camera data for raytracing
		const auto cameraPosition = core::vectorSIMDf().set(camera->getAbsolutePosition());
		m_raytraceCommonData.camPos.x = cameraPosition.x;
		m_raytraceCommonData.camPos.y = cameraPosition.y;
		m_raytraceCommonData.camPos.z = cameraPosition.z;
	}
	// raygen
	{
		// vertex 0 is camera
		m_raytraceCommonData.depth = beauty ? 0u:(~0u);

		//
		video::IGPUDescriptorSet* sameDS[2] = {m_raygenDS.get(),m_raygenDS.get()};
		preDispatch(m_raygenPipeline->getLayout(),sameDS);

		//
		m_driver->bindComputePipeline(m_raygenPipeline.get());
		m_driver->dispatch(m_raygenWorkGroups[0],m_raygenWorkGroups[1],1);
	}
	// path trace
	if (beauty)
	{
		while (m_raytraceCommonData.depth!=m_staticViewData.pathDepth)
		{
			uint32_t raycount;
			 if(!traceBounce(raycount))
				 return false;
			 if (raycount==0u)
				 break;
		}
	}
	// mostly writes to accumulation buffers and SSBO clears
	// probably wise to flush all caches (in the future can optimize to texture_fetch|shader_image_access|shader_storage_buffer|blit|texture_download|...)
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);

	// resolve pseudo-MSAA
	if (beauty)
	{
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_resolvePipeline->getLayout(),0u,1u,&m_resolveDS.get(),nullptr);
		m_driver->bindComputePipeline(m_resolvePipeline.get());
		if (transformNormals)
			m_driver->pushConstants(m_resolvePipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(m_prevView),&m_prevView);
		else
		{
			decltype(m_prevView) identity;
			m_driver->pushConstants(m_resolvePipeline->getLayout(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(identity),&identity);
		}
		m_driver->dispatch(m_raygenWorkGroups[0],m_raygenWorkGroups[1],1);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
			// because of direct to screen resolve
			|GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT
		);
		m_raytraceCommonData.samplesComputed = (m_raytraceCommonData.samplesComputed+getSamplesPerPixelPerDispatch())%maxSensorSamples;
	}

	// TODO: autoexpose properly
	return true;
}

void Renderer::preDispatch(const video::IGPUPipelineLayout* pipelineLayout, video::IGPUDescriptorSet*const *const lastDS)
{
	// increment depth
	const uint32_t descSetIx = (++m_raytraceCommonData.depth)&0x1u;
	m_driver->pushConstants(pipelineLayout,ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t),&m_raytraceCommonData);

	// advance
	static_assert(core::isPoT(RAYCOUNT_N_BUFFERING),"Raycount Buffer needs to be PoT sized!");
	m_raytraceCommonData.rayCountWriteIx = (++m_raytraceCommonData.rayCountWriteIx)&RAYCOUNT_N_BUFFERING_MASK;
	
	IGPUDescriptorSet* descriptorSets[4] = {m_globalBackendDataDS.get(),m_additionalGlobalDS.get(),m_commonRaytracingDS[descSetIx].get(),lastDS[descSetIx]};
	m_driver->bindDescriptorSets(EPBP_COMPUTE,pipelineLayout,0u,4u,descriptorSets,nullptr);
}

bool Renderer::traceBounce(uint32_t& raycount)
{
	// probably wise to flush all caches (in the future can optimize to texture_fetch|shader_image_access|shader_storage_buffer|blit|texture_download|...)
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);
	{
		const auto rayCountReadIx = (m_raytraceCommonData.rayCountWriteIx-1u)&RAYCOUNT_N_BUFFERING_MASK;
		m_driver->copyBuffer(m_rayCountBuffer.get(),m_littleDownloadBuffer.get(),sizeof(uint32_t)*rayCountReadIx,0u,sizeof(uint32_t));
	}
	glFinish(); // sync CPU to GL
	raycount = *reinterpret_cast<uint32_t*>(m_littleDownloadBuffer->getBoundMemory()->getMappedPointer());

	if (raycount)
	{
		// trace rays
		m_totalRaysCast += raycount;
		{
			const uint32_t descSetIx = m_raytraceCommonData.depth&0x1u;

			auto commandQueue = m_rrManager->getCLCommandQueue();
			const cl_mem clObjects[] = {m_rayBuffer[descSetIx].asRRBuffer.second,m_intersectionBuffer[descSetIx].asRRBuffer.second};
			const auto objCount = sizeof(clObjects)/sizeof(cl_mem);
			cl_event acquired=nullptr, raycastDone=nullptr;
			// run the raytrace queries
			{
				ocl::COpenCLHandler::ocl.pclEnqueueAcquireGLObjects(commandQueue,objCount,clObjects,0u,nullptr,&acquired);

				clEnqueueWaitForEvents(commandQueue,1u,&acquired);
				m_rrManager->getRadeonRaysAPI()->QueryIntersection(
					m_rayBuffer[descSetIx].asRRBuffer.first,raycount,
					m_intersectionBuffer[descSetIx].asRRBuffer.first,nullptr,nullptr
				);
				clEnqueueMarker(commandQueue,&raycastDone);
			}

			// sync CPU to CL
			cl_event released;
			ocl::COpenCLHandler::ocl.pclEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, &released);
			ocl::COpenCLHandler::ocl.pclFlush(commandQueue);

			cl_int retval = -1;
			auto startWait = std::chrono::steady_clock::now();
			constexpr auto timeoutInSeconds = 20ull;
			bool timedOut = false;
			do {
				const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-startWait).count();
				if (elapsed > timeoutInSeconds * 1'000'000ull)
				{
					timedOut = true;
					break;
				}

				std::this_thread::yield();
				ocl::COpenCLHandler::ocl.pclGetEventInfo(released, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &retval, nullptr);
			} while(retval != CL_COMPLETE);
		
			if(timedOut)
			{
				std::cout << "[ERROR] RadeonRays Timed Out" << std::endl;
				return false;
			}
		}
	
		// compute bounce (accumulate contributions and optionally generate rays)
		{
			preDispatch(m_closestHitPipeline->getLayout(),&m_closestHitDS->get());

			m_driver->bindComputePipeline(m_closestHitPipeline.get());
			m_driver->dispatch((raycount-1u)/WORKGROUP_SIZE+1u,1u,1u);
		}
	}

	return true;
}


const float Renderer::AntiAliasingSequence[Renderer::AntiAliasingSequenceLength][2] =
{
{0.229027962000000, 0.100901043000000},
{0.934988661250000, 0.900492937500000},
{0.693936740750000, 0.477888665000000},
{0.396013875250000, 0.867381653000000},
{0.151208663250000, 0.331649132250000},
{0.919338615000000, 0.306386117750000},
{0.454737456500000, 0.597940860250000},
{0.911951413000000, 0.584874565000000},
{0.471331207500000, 0.117509299250000},
{0.724981748000000, 0.988645892000000},
{0.227727943750000, 0.553082892250000},
{0.927148254750000, 0.059077206250000},
{0.170420940250000, 0.853803466500000},
{0.369496963250000, 0.372492160250000},
{0.709055501500000, 0.719526612750000},
{0.708593019750000, 0.236308825250000},
{0.053515783250000, 0.244794542562500},
{0.759417624125000, 0.846532545187500},
{0.572365454937500, 0.341559262437500},
{0.269128942562500, 0.962581831375000},
{0.246508261687500, 0.286661635812500},
{0.819542439062500, 0.459099133812500},
{0.411348913687500, 0.737420359250000},
{0.896647944437500, 0.717554343125000},
{0.358057598000000, 0.050206801437500},
{0.605871046250000, 0.779868041500000},
{0.036816445812500, 0.506511135625000},
{0.806931985937500, 0.138270723062500},
{0.045020470000000, 0.818334270875000},
{0.433264399500000, 0.254739200375000},
{0.556258709500000, 0.559776624000000},
{0.611048395312500, 0.162518625750000},
{0.028918631812500, 0.053438072375000},
{0.856252533125000, 0.916712681500000},
{0.580344816187500, 0.463534157062500},
{0.291334488000000, 0.774756179000000},
{0.157847279187500, 0.464948199125000},
{0.775478249937500, 0.320623736250000},
{0.306258709500000, 0.653526624000000},
{0.798533046937500, 0.552896543187500},
{0.349953270437500, 0.123764825500000},
{0.534027961437500, 0.969931745937500},
{0.122488661312500, 0.681742937625000},
{0.849003468812500, 0.216845413250000},
{0.145343900750000, 0.962506045625000},
{0.395929912437500, 0.488477370312500},
{0.675219736437500, 0.601237158875000},
{0.728921568625000, 0.053308823500000},
{0.153721825125000, 0.145597505062500},
{0.852763510375000, 0.797682223125000},
{0.644595719312500, 0.367380713687500},
{0.475934665312500, 0.787623234375000},
{0.037670496437500, 0.386130180750000},
{0.916111850937500, 0.403604173437500},
{0.307256453062500, 0.518207928812500},
{0.836158139312500, 0.677526975812500},
{0.291525812500000, 0.197831715312500},
{0.632543215125000, 0.896220934750000},
{0.039235045687500, 0.629605464812500},
{0.927263875375000, 0.179881653187500},
{0.036335975187500, 0.990626511375000},
{0.458406617875000, 0.372877193062500},
{0.545614665812500, 0.676662283062500},
{0.606815968812500, 0.044970413250000},
{0.031533697125000, 0.184836288625000},
{0.943869562500000, 0.830155934062500},
{0.607026984312500, 0.286243495000000},
{0.385468447812500, 0.923477959062500},
{0.211591778000000, 0.432717372437500},
{0.959561740812500, 0.477888665062500},
{0.340921091062500, 0.599871303750000},
{0.770926812125000, 0.740443845937500},
{0.492972183312500, 0.243769330562500},
{0.520086204062500, 0.865883539250000},
{0.194132187625000, 0.711586172812500},
{0.867832801875000, 0.029377324812500},
{0.018898352500000, 0.755166315812500},
{0.294110519250000, 0.340476317312500},
{0.645436781125000, 0.669120978187500},
{0.537010584750000, 0.070669853500000},
{0.161951413000000, 0.209874565062500},
{0.786335975187500, 0.990626511375000},
{0.525681985937500, 0.419520723062500},
{0.287619562500000, 0.834550465312500},
{0.100299557750000, 0.367542953000000},
{0.787670496437500, 0.386130180750000},
{0.425010132750000, 0.666850725937500},
{0.959417841312500, 0.712724761625000},
{0.259027114250000, 0.027505482375000},
{0.706747124500000, 0.863983912687500},
{0.118758709500000, 0.559776624000000},
{0.979834653750000, 0.076596529437500},
{0.076814113250000, 0.879551982187500},
{0.458038607062500, 0.495297691687500},
{0.676899749875000, 0.533654791000000},
{0.739509651750000, 0.162886922875000},
{0.130635833000000, 0.032884578937500},
{0.995486845875000, 0.879726983937500},
{0.681683761437500, 0.415213866187500},
{0.471888733500000, 0.975077322375000},
{0.002080578437500, 0.292317740812500},
{0.982026984312500, 0.286243495000000},
{0.291525812500000, 0.713456715312500},
{0.803515783250000, 0.619794542562500},
{0.363736251000000, 0.241491573500000},
{0.581375603187500, 0.850024182625000},
{0.126134788437500, 0.739345154625000},
{0.807256990625000, 0.025225260812500},
{0.214063133312500, 0.979178170312500},
{0.279120068187500, 0.455460706437500},
{0.521614411125000, 0.748128257250000},
{0.541375661500000, 0.191916865812500},
{0.092374240812500, 0.093123040062500},
{0.819780017000000, 0.863865176562500},
{0.723535390937500, 0.290673655562500},
{0.333626471625000, 0.991508772375000},
{0.180081879937500, 0.273337083437500},
{0.884853249937500, 0.353826861250000},
{0.486489450437500, 0.649922456187500},
{0.970355124125000, 0.588720045187500},
{0.411054041562500, 0.190728892687500},
{0.670557598000000, 0.782628676437500},
{0.176686781125000, 0.590995978187500},
{0.923484185187500, 0.119472166250000},
{0.229834653750000, 0.826596529437500},
{0.402229645500000, 0.427815757250000},
{0.614887300500000, 0.582390020187500},
{0.721331207625000, 0.117509299250000},
{0.221780261562500, 0.160787322875000},
{0.980871046250000, 0.779868041500000},
{0.521614411125000, 0.498128257250000},
{0.462698109750000, 0.855009158437500},
{0.102148981812500, 0.485181351500000},
{0.790505847500000, 0.272359588500000},
{0.357263913000000, 0.553624565062500},
{0.852875617687500, 0.518271589687500},
{0.412788910312500, 0.072860258937500},
{0.739509651750000, 0.912886922875000},
{0.244715387500000, 0.610882883562500},
{0.931245437000000, 0.247161473000000},
{0.118495619500000, 0.827404835625000},
{0.356241537562500, 0.307793951312500},
{0.739954645312500, 0.601971750750000},
{0.652229645500000, 0.240315757250000},
{0.085230272750000, 0.149967825937500},
{0.790487853250000, 0.802468641250000},
{0.742972183312500, 0.431269330562500},
{0.338023546687500, 0.864140358375000},
{0.161359195250000, 0.386030244500000},
{0.979622565375000, 0.415764143437500},
{0.344324410875000, 0.743490102812500},
{0.850000234687500, 0.588036936812500},
{0.478921568625000, 0.053308823500000},
{0.575878945812500, 0.904948635625000},
{0.066809364125000, 0.711985215062500},
{0.842374240812500, 0.093123040062500},
{0.072833622000000, 0.943689044750000},
{0.473982478062500, 0.309619342562500},
{0.643468702500000, 0.727011596187500},
{0.661784804062500, 0.096504548000000},
{0.075593410125000, 0.020665263437500},
{0.846367111937500, 0.980869489750000},
{0.584417841312500, 0.402177886625000},
{0.419264650000000, 0.807665176000000},
{0.108911798812500, 0.274823750687500},
{0.842214949125000, 0.395649388625000},
{0.424460011250000, 0.541515061937500},
{0.914875915625000, 0.525088448500000},
{0.276815978250000, 0.138406141250000},
{0.682946765937500, 0.941192325375000},
{0.243922631125000, 0.674414353500000},
{0.983747165312500, 0.225123234375000},
{0.209039534812500, 0.919381743937500},
{0.317979460312500, 0.396931190375000},
{0.595789254625000, 0.645833852812500},
{0.589063133312500, 0.229178170312500},
{0.201732996437500, 0.034567680750000},
{0.911951413000000, 0.959874565062500},
{0.669465614812500, 0.307270670687500},
{0.442773254937500, 0.918452206312500},
{0.228659227750000, 0.498372525687500},
{0.864786425062500, 0.258916160937500},
{0.366015783250000, 0.682294542562500},
{0.832368054687500, 0.749523853312500},
{0.475148582250000, 0.180782790250000},
{0.543804934062500, 0.806559289687500},
{0.041831345187500, 0.574164114062500},
{0.981787063687500, 0.014769301562500},
{0.167325380812500, 0.796656456375000},
{0.305883847687500, 0.260168413875000},
{0.736593646187500, 0.544303438875000},
{0.595631980687500, 0.113338942562500},
{0.233747165312500, 0.225123234375000},
{0.881420496437500, 0.854880180750000},
{0.514120916562500, 0.361726452125000},
{0.262088408375000, 0.897305108937500},
{0.040764352812500, 0.448613484812500},
{0.882527922875000, 0.453355770312500},
{0.486593646187500, 0.544303438875000},
{0.944193581437500, 0.650074503000000},
{0.403764392000000, 0.003513614062500},
{0.647805652187500, 0.839498260375000},
{0.004346402437500, 0.700568695812500},
{0.863684364125000, 0.149485215062500},
{0.075593410125000, 0.770665263437500},
{0.260573230937500, 0.378374937125000},
{0.606947383687500, 0.518907935937500},
{0.522543332062500, 0.131538315687500},
{0.115674527437500, 0.213752289562500},
{0.978861550187500, 0.943531534250000},
{0.716222608750000, 0.357993041500000},
{0.396123640562500, 0.988911053812500},
{0.116417439062500, 0.427849133812500},
{0.960375986437500, 0.355143408875000},
{0.396123640562500, 0.613911053812500},
{0.771872079375000, 0.679610301562500},
{0.407651999187500, 0.129979780250000},
{0.610967390562500, 0.957538983500000},
{0.099030910125000, 0.622227763437500},
{0.792605235062500, 0.213450866625000},
{0.231787063687500, 0.764769301562500},
{0.345102996500000, 0.451097146437500},
{0.537639116937500, 0.609792796562500},
{0.670557598000000, 0.032628676437500},
{0.161148929812500, 0.091845178375000},
{0.915446201000000, 0.774126136937500},
{0.542495165812500, 0.275647345250000},
{0.316935423562500, 0.923529210750000},
{0.209068937250000, 0.340693571625000},
{0.770926812125000, 0.490443845937500},
{0.462732614000000, 0.712224844000000},
{0.887121046250000, 0.643149291500000},
{0.302373640562500, 0.082661053812500},
{0.728921568625000, 0.803308823500000},
{0.181258709500000, 0.653526624000000},
{0.977987853250000, 0.146218641250000},
{0.016702341062500, 0.927996303750000},
{0.467374240812500, 0.431013665062500},
{0.706158315437500, 0.659556033875000},
{0.669301523312500, 0.186625825562500},
{0.039461819812500, 0.116237049750000},
{0.798533046937500, 0.927896543187500},
{0.636245165812500, 0.463147345250000},
{0.358057598000000, 0.800206801437500},
{0.057953992187500, 0.323742603312500},
{0.838196808062500, 0.318240323625000},
{0.288441098000000, 0.563378403500000},
{0.981947383687500, 0.518907935937500},
{0.348181288187500, 0.183212688250000},
{0.506420496437500, 0.917380180750000},
{0.164875915625000, 0.525088448500000},
{0.787802165812500, 0.082912283062500},
{0.134409779187500, 0.902448199125000},
{0.408376747562500, 0.326245734125000},
{0.584561740812500, 0.727888665062500},
{0.534266910125000, 0.009900787187500},
{0.192731703031250, 0.122610961484375},
{0.886403666453125, 0.919165570765625},
{0.740362459437500, 0.479082682703125},
{0.381143881328125, 0.825372076343750},
{0.177058600328125, 0.355660703968750},
{0.880411986109375, 0.304388585781250},
{0.489954645359375, 0.601971750750000},
{0.931245437015625, 0.622161472968750},
{0.495691442609375, 0.089534956375000},
{0.748617226093750, 0.964032599359375},
{0.200716102781250, 0.538594564312500},
{0.902762098828125, 0.040629656437500},
{0.167366403703125, 0.826817667671875},
{0.326986691484375, 0.360298081343750},
{0.725793624375000, 0.689620323765625},
{0.725003755000000, 0.197653489734375},
{0.019203528312500, 0.219887995546875},
{0.789461819796875, 0.866237049781250},
{0.602987853250000, 0.364968641265625},
{0.286530910140625, 0.997227763453125},
{0.197691088203125, 0.299653371203125},
{0.864509651765625, 0.475386922921875},
{0.409509265921875, 0.695098575390625},
{0.924966150343750, 0.731175805390625},
{0.366549151562500, 0.016182260046875},
{0.620446765921875, 0.753692325390625},
{0.004126035234375, 0.512365679515625},
{0.759204111453125, 0.126627783906250},
{0.039461819796875, 0.866237049781250},
{0.385324830531250, 0.282537197156250},
{0.529500805046875, 0.539659192578125},
{0.620486845921875, 0.129726983984375},
{0.040287232453125, 0.022961294593750},
{0.871648411546875, 0.886075859718750},
{0.567731703031250, 0.497610961484375},
{0.271219649968750, 0.785940731265625},
{0.149813081953125, 0.495059302156250},
{0.752354406031250, 0.336633136296875},
{0.267950605953125, 0.630717656421875},
{0.763064970921875, 0.526211358984375},
{0.350302165859375, 0.082912283093750},
{0.541375661546875, 0.941916865828125},
{0.067888875390625, 0.648631653203125},
{0.817282235640625, 0.240645457843750},
{0.176686781125000, 0.965995978171875},
{0.414407018781250, 0.458335411421875},
{0.635381359671875, 0.622395865796875},
{0.696580598671875, 0.010305563015625},
{0.146140435203125, 0.181972166265625},
{0.853197227578125, 0.768215064734375},
{0.631158461921875, 0.330667103468750},
{0.443098689046875, 0.770526325000000},
{0.008860189078125, 0.404241883828125},
{0.920499240812500, 0.436873040078125},
{0.274931749687500, 0.517395920968750},
{0.872488661328125, 0.681742937687500},
{0.273658139312500, 0.240026975812500},
{0.686178846875000, 0.902720720890625},
{0.022328994562500, 0.659601535312500},
{0.889064677375000, 0.139944156000000},
{0.041831345203125, 0.949164114093750},
{0.443262299140625, 0.313206182765625},
{0.553107937421875, 0.637161480234375},
{0.576361266343750, 0.010049207671875},
{0.024757727531250, 0.155556940859375},
{0.954885609765625, 0.864774783453125},
{0.576988498046875, 0.268435650828125},
{0.378272153750000, 0.889096529468750},
{0.243922631171875, 0.424414353515625},
{0.993449504812500, 0.487462829328125},
{0.315047772046875, 0.590538342781250},
{0.757436479140625, 0.715431613031250},
{0.454737456671875, 0.222940860375000},
{0.506538910328125, 0.822860258968750},
{0.223798019234375, 0.699851317375000},
{0.839514399500000, 0.012551700359375},
{0.013378945812500, 0.811198635640625},
{0.259404890625000, 0.333637616328125},
{0.674460011265625, 0.635265061968750},
{0.552179912484375, 0.113477370312500},
{0.133506990359375, 0.242482936484375},
{0.792605235078125, 0.963450866640625},
{0.556245437015625, 0.434661472968750},
{0.302640369765625, 0.866357903265625},
{0.104025812484375, 0.322831715312500},
{0.788292439093750, 0.420036633796875},
{0.383947288453125, 0.645595154640625},
{0.987679025703125, 0.720586141078125},
{0.310166960328125, 0.049274940406250},
{0.692634651765625, 0.826949422921875},
{0.066739180750000, 0.551367061812500},
{0.954885609765625, 0.114774783453125},
{0.106252533187500, 0.916712681484375},
{0.490362459437500, 0.479082682703125},
{0.646028602781250, 0.509297689312500},
{0.696508847734375, 0.182043413890625},
{0.167639399500000, 0.008157169109375},
{0.942731703031250, 0.935110961484375},
{0.682010510203125, 0.383364961312500},
{0.444750986453125, 0.993815283906250},
{0.012293182515625, 0.265019109265625},
{0.943520700468750, 0.285664643703125},
{0.256436740812500, 0.727888665078125},
{0.792605235078125, 0.588450866640625},
{0.323828669343750, 0.228345414000000},
{0.589727949703125, 0.818705937671875},
{0.146647944500000, 0.717554343109375},
{0.763378945812500, 0.061198635640625},
{0.245217761562500, 0.944967010375000},
{0.302009651765625, 0.475386922921875},
{0.508157018781250, 0.708335411421875},
{0.552058600328125, 0.230660703968750},
{0.076470961921875, 0.065042103468750},
{0.839060384390625, 0.826948487828125},
{0.743383847734375, 0.260168413890625},
{0.361729406031250, 0.961633136296875},
{0.130411986109375, 0.304388585781250},
{0.913057411375000, 0.372578939312500},
{0.450791960328125, 0.635700721656250},
{0.994715387546875, 0.610882883562500},
{0.396123640625000, 0.238911053828125},
{0.635171568625000, 0.803308823531250},
{0.134436274500000, 0.588235294109375},
{0.893091363734375, 0.085389815609375},
{0.204885609765625, 0.864774783453125},
{0.419763913046875, 0.397374565093750},
{0.589063133281250, 0.604178170375000},
{0.692634651765625, 0.076949422921875},
{0.192731703031250, 0.185110961484375},
{0.951814247656250, 0.756306315203125},
{0.506689186515625, 0.439765218421875},
{0.456461826015625, 0.821001137000000},
{0.083707248625000, 0.461775383828125},
{0.764249240812500, 0.280623040078125},
{0.323579912671875, 0.557221730578125},
{0.818079645359375, 0.508221750750000},
{0.435674328421875, 0.095052115796875},
{0.725148582281250, 0.930782790234375},
{0.211591777984375, 0.620217372437500},
{0.901467761562500, 0.194967010375000},
{0.114509651765625, 0.873824422921875},
{0.350302165859375, 0.270412283093750},
{0.713196808109375, 0.568240323640625},
{0.631158461921875, 0.205667103468750},
{0.121648411546875, 0.136075859718750},
{0.807256990687500, 0.775225260828125},
{0.748441255000000, 0.385153489734375},
{0.322881990687500, 0.822100260828125},
{0.170499240812500, 0.436873040078125},
{0.976735045687500, 0.379605464812500},
{0.326988498046875, 0.705935650828125},
{0.849030910140625, 0.622227763453125},
{0.456628942609375, 0.025081831375000},
{0.603718978906250, 0.881272112125000},
{0.087671365468750, 0.733711546609375},
{0.858316099875000, 0.063684800093750},
{0.079233855890625, 0.980882302687500},
{0.498831558375000, 0.275019330593750},
{0.683006746078125, 0.696560873750000},
{0.634421685203125, 0.070155760015625},
{0.100941098000000, 0.000878403515625},
{0.868983666328125, 0.946905808593750},
{0.622823333015625, 0.407884578921875},
{0.380701413046875, 0.772374565093750},
{0.070966777984375, 0.276467372437500},
{0.869730392156250, 0.411764705875000},
{0.390973727828125, 0.533716984406250},
{0.934919978109375, 0.561328326953125},
{0.267725152796875, 0.170350753109375},
{0.650556467859375, 0.939171186375000},
{0.208092013671875, 0.656130963328125},
{0.939854406031250, 0.211633136296875},
{0.227987853250000, 0.896218641265625},
{0.362037827187500, 0.411778179156250},
{0.575564970921875, 0.682461358984375},
{0.603861550250000, 0.193531534234375},
{0.245936791328125, 0.056280808593750},
{0.931245437015625, 0.997161472968750},
{0.674341363734375, 0.272889815609375},
{0.475148582281250, 0.930782790234375},
{0.196690920375000, 0.490305748390625},
{0.823073230953125, 0.290484312156250},
{0.349801672015625, 0.643614082703125},
{0.816809364156250, 0.711985215093750},
{0.442773254953125, 0.168452206343750},
{0.559175124515625, 0.768645847968750},
{0.012608312281250, 0.564660480046875},
{0.951732996484375, 0.034567680765625},
{0.130635833015625, 0.782884578921875},
{0.295859197828125, 0.295320202140625},
{0.712431749687500, 0.517395920968750},
{0.572626461875000, 0.068089897125000},
{0.211591777984375, 0.245217372437500},
{0.901756746078125, 0.821560873750000},
{0.512364601359375, 0.315328221531250},
{0.275838593406250, 0.932598880093750},
{0.007956102718750, 0.451497525703125},
{0.924966150343750, 0.481175805390625},
{0.454495282953125, 0.559257955593750},
{0.978187903312500, 0.673257136171875},
{0.416103249937500, 0.041326861265625},
{0.664447403859375, 0.864416693968750},
{0.033521988046875, 0.696631179015625},
{0.852837228421875, 0.184355089812500},
{0.090934062750000, 0.810372893375000},
{0.275746963312500, 0.411554660312500},
{0.588037245453125, 0.558795337875000},
{0.554922654015625, 0.160357524531250},
{0.072833622000000, 0.193689044750000},
{0.964063133281250, 0.979178170375000},
{0.708517234312500, 0.319548392906250},
{0.432256990687500, 0.962725260828125},
{0.068079645359375, 0.414471750750000},
{0.963190877593750, 0.324420555781250},
{0.411502470921875, 0.573086358984375},
{0.800162124781250, 0.669611615750000},
{0.387554934109375, 0.150309289718750},
{0.579945004250000, 0.965966294140625},
{0.065522102093750, 0.599326277234375},
{0.761255117500000, 0.204583567718750},
{0.196950541453125, 0.770728070765625},
{0.344324410937500, 0.493490102843750},
{0.510111550250000, 0.568531534234375},
{0.636389399500000, 0.008157169109375},
{0.128530229140625, 0.090431613031250},
{0.883566727531250, 0.752475196796875},
{0.552206093281250, 0.309003302484375},
{0.348181288187500, 0.933212688265625},
{0.227987853250000, 0.364968641265625},
{0.771924527437500, 0.448127289609375},
{0.489679912484375, 0.745313307812500},
{0.927148254953125, 0.684077206343750},
{0.264066255000000, 0.103903489734375},
{0.740057568187500, 0.764054456484375},
{0.148947313656250, 0.630208463203125},
{0.974161986109375, 0.179388585781250},
{0.010457836296875, 0.893541028515625},
{0.498441255000000, 0.385153489734375},
{0.744468904421875, 0.637380397421875},
{0.678301531546875, 0.136423458078125},
{0.010191088203125, 0.112153371203125},
{0.774757727531250, 0.905556940859375},
{0.674229406031250, 0.446008136296875},
{0.319922419343750, 0.784986039000000},
{0.011042923812500, 0.349437248625000},
{0.821379264859375, 0.354136628500000},
{0.257237357453125, 0.579287821171875},
{0.948151308765625, 0.522112716656250},
{0.318520700468750, 0.160664643703125},
{0.543804934109375, 0.900309289718750},
{0.130607996843750, 0.519919121093750},
{0.811627065421875, 0.071665408953125},
{0.160867175625000, 0.931752899046875},
{0.428297719390625, 0.362355138953125},
{0.609505036140625, 0.690144858781250},
{0.504804041609375, 0.003228892687500},
{0.216196606265625, 0.064729040234375},
{0.901736845921875, 0.879726983984375},
{0.708719649968750, 0.453909481265625},
{0.415337952218750, 0.849024716109375},
{0.134853249937500, 0.353826861265625},
{0.903787097500000, 0.267080047484375},
{0.479025812484375, 0.572831715312500},
{0.876605124109375, 0.604345045187500},
{0.456461826015625, 0.071001137000000},
{0.709494562484375, 0.955644215312500},
{0.231947383703125, 0.518907935984375},
{0.932230392156250, 0.013327205875000},
{0.145086204046875, 0.865883539265625},
{0.350930456281250, 0.348899376265625},
{0.725034838203125, 0.739106496203125},
{0.739954645359375, 0.226971750750000},
{0.042605235078125, 0.213450866640625},
{0.811627065421875, 0.821665408953125},
{0.587735274500000, 0.312719600875000},
{0.307230392156250, 0.950827205875000},
{0.217487057734375, 0.273792953031250},
{0.854401308765625, 0.440081466656250},
{0.384747995484375, 0.706561433531250},
{0.899206688062500, 0.699456330843750},
{0.334068937265625, 0.028193571656250},
{0.576864665859375, 0.801662283093750},
{0.041360180171875, 0.539240991515625},
{0.780622165328125, 0.170435734406250},
{0.025074889437500, 0.841885738250000},
{0.412686740812500, 0.274763665078125},
{0.551686781125000, 0.512870978171875},
{0.574633261734375, 0.138224135796875},
{0.058436791328125, 0.056280808593750},
{0.822110274500000, 0.890844600875000},
{0.618449504812500, 0.487462829328125},
{0.264066255000000, 0.807028489734375},
{0.132527922859375, 0.453355770328125},
{0.807953992203125, 0.323742603312500},
{0.302148254953125, 0.684077206343750},
{0.794171695281250, 0.522748994546875},
{0.372686140625000, 0.110004803828125},
{0.507741981953125, 0.951245734125000},
{0.107097304093750, 0.651192048015625},
{0.822833622000000, 0.193689044750000},
{0.181245437015625, 0.997161472968750},
{0.384747995484375, 0.456561433531250},
{0.662226998781250, 0.569583683906250},
{0.727197083078125, 0.021632137984375},
{0.184988661328125, 0.150492937687500},
{0.873243045828125, 0.810942332640625},
{0.684963615078125, 0.357045297593750},
{0.461525611203125, 0.759528012437500},
{0.025718904421875, 0.410817897421875},
{0.897009311359375, 0.420948834468750},
{0.263037063734375, 0.546019301578125},
{0.857097304093750, 0.651192048015625},
{0.252866199843750, 0.205957512640625},
{0.665602115484375, 0.895166775859375},
{0.056996963312500, 0.684992160312500},
{0.918804934109375, 0.150309289718750},
{0.019203528312500, 0.969887995546875},
{0.485853633703125, 0.339220435984375},
{0.509352115484375, 0.676416775859375},
{0.564952190359375, 0.039350341546875},
{0.061178846875000, 0.152720720890625},
{0.979027962734375, 0.850901043359375},
{0.618986693593750, 0.251067502968750},
{0.416788412578125, 0.890801763843750},
{0.191935423609375, 0.392279210781250},
{0.959424527437500, 0.448127289609375},
{0.360967390625000, 0.582538983515625},
{0.802390435203125, 0.744472166265625},
{0.498617226093750, 0.214032599359375},
{0.542495165859375, 0.838147345281250},
{0.211457644609375, 0.742513028953125},
{0.837488317609375, 0.030941206375000},
{0.038430456281250, 0.786399376265625},
{0.290324504812500, 0.370275329328125},
{0.678301531546875, 0.667673458078125},
{0.510613942796875, 0.106955928328125},
{0.170736691484375, 0.235298081343750},
{0.759083993796875, 0.997656627843750},
{0.539572313656250, 0.380208463203125},
{0.255388875390625, 0.867381653203125},
{0.071379264859375, 0.354136628500000},
{0.758860189078125, 0.404241883828125},
{0.420321786390625, 0.636298924375000},
{0.978659227718750, 0.748372525703125},
{0.287503755000000, 0.010153489734375},
{0.739679912484375, 0.870313307812500},
{0.089315978250000, 0.513406141265625},
{0.943869562484375, 0.080155934062500},
{0.076564677375000, 0.913381656000000},
{0.444542439093750, 0.459099133796875},
{0.633162999796875, 0.540307445062500},
{0.709818581500000, 0.157887002984375},
{0.167325380828125, 0.046656456390625},
{0.977987853250000, 0.896218641265625},
{0.652229645515625, 0.427815757218750},
{0.492972183375000, 0.993769330593750},
{0.040505847500000, 0.272359588500000},
{0.962978249937500, 0.260076861265625},
{0.302640369765625, 0.741357903265625},
{0.768208405500000, 0.610922261187500},
{0.318273390046875, 0.193320190000000},
{0.619905641343750, 0.853941035859375},
{0.181258709546875, 0.747276624031250},
{0.757229657953125, 0.013359518093750},
{0.244715387546875, 0.985882883562500},
{0.259008847734375, 0.494543413890625},
{0.554055652187500, 0.714498260375000},
{0.534027961468750, 0.219931745984375},
{0.069780017046875, 0.113865176609375},
{0.848982478109375, 0.872119342578125},
{0.713196808109375, 0.271365323640625},
{0.363736251062500, 0.991491573531250},
{0.150433761421875, 0.282401366203125},
{0.901208663437500, 0.331649132359375},
{0.463675308375000, 0.681269330593750},
{0.949633261734375, 0.606974135796875},
{0.384851754687500, 0.208333852843750},
{0.668614601359375, 0.752828221531250},
{0.181245437015625, 0.622161472968750},
{0.895086204046875, 0.115883539265625},
{0.193869562484375, 0.830155934062500},
{0.387335340921875, 0.396347453890625},
{0.564854406031250, 0.586633136296875},
{0.745691442609375, 0.089534956375000},
{0.245486845921875, 0.129726983984375},
{0.982811359250000, 0.811790368250000},
{0.536503468843750, 0.474657913296875},
{0.489679912484375, 0.870313307812500},
{0.067888875390625, 0.492381653203125},
{0.751398662484375, 0.307813307812500},
{0.352987853250000, 0.521218641265625},
{0.869468904421875, 0.543630397421875},
{0.400000938734375, 0.102147747421875},
{0.716287097500000, 0.892080047484375},
{0.204945004250000, 0.590966294140625},
{0.883506990359375, 0.242482936484375},
{0.065143307734375, 0.844593734281250},
{0.327001686515625, 0.299140218421875},
{0.748446786390625, 0.573798924375000},
{0.665686274500000, 0.213235294109375},
{0.115683153500000, 0.178056211000000},
{0.757883424281250, 0.796209072156250},
{0.713222390562500, 0.397222857359375},
{0.362679025703125, 0.845586141078125},
{0.147009311359375, 0.420948834468750},
{0.954472218843750, 0.404345413296875},
{0.363836204046875, 0.740883539265625},
{0.821904890625000, 0.583637616328125},
{0.492832801906250, 0.029377324812500},
{0.599161986109375, 0.929388585781250},
{0.096331207625000, 0.695634299281250},
{0.848982478109375, 0.122119342578125},
{0.099003468843750, 0.966845413296875},
{0.462431749687500, 0.267395920968750},
{0.677354460328125, 0.725544471656250},
{0.650008709546875, 0.122276624031250},
{0.123243045828125, 0.060942332640625},
{0.822833622000000, 0.943689044750000},
{0.586591777984375, 0.432717372437500},
{0.408903485687500, 0.776738982078125},
{0.080479406031250, 0.305383136296875},
{0.818079645359375, 0.414471750750000},
{0.431931985968750, 0.513270723078125},
{0.903169756421875, 0.555146535265625},
{0.285517059468750, 0.166546514046875},
{0.675219736453125, 0.976237158906250},
{0.238942076937500, 0.643155269500000},
{0.964063133281250, 0.229178170375000},
{0.192731703031250, 0.935110961484375},
{0.348181288187500, 0.401962688265625},
{0.618003220203125, 0.658636770343750},
{0.588190877593750, 0.199420555781250},
{0.216958200468750, 0.007344331203125},
{0.883506990359375, 0.992482936484375},
{0.636042923812500, 0.294749748625000},
{0.481304934109375, 0.900309289718750},
{0.237679025703125, 0.470586141078125},
{0.831750421625000, 0.262285054609375},
{0.341264392046875, 0.675388614109375},
{0.864509651765625, 0.725386922921875},
{0.481304934109375, 0.150309289718750},
{0.507180456281250, 0.786399376265625},
{0.033602444796875, 0.600612049781250},
{0.962250867203125, 0.054211353312500},
{0.151703992203125, 0.761242603312500},
{0.261464799453125, 0.261330050531250},
{0.740667841375000, 0.511552886640625},
{0.615093996609375, 0.088785852218750},
{0.228861550250000, 0.193531534234375},
{0.920420940359375, 0.853803466546875},
{0.541111850968750, 0.345010423484375},
{0.305472183375000, 0.900019330593750},
{0.044319650703125, 0.478886922328125},
{0.892725152796875, 0.482850753109375},
{0.490667841375000, 0.511552886640625},
{0.971780261562500, 0.629537322875000},
{0.387554934109375, 0.056559289718750},
{0.662939186515625, 0.814765218421875},
{0.052390435203125, 0.744472166265625},
{0.826564677375000, 0.163381656000000},
{0.103197227578125, 0.768215064734375},
{0.306265020015625, 0.415613958984375},
{0.575564970921875, 0.526211358984375},
{0.505411986109375, 0.183294835781250},
{0.079233855890625, 0.230882302687500},
{0.939854406031250, 0.961633136296875},
{0.747488661328125, 0.369242937687500},
{0.399926992500000, 0.954583567718750},
{0.122446606265625, 0.392854040234375},
{0.987719736453125, 0.351237158906250},
{0.425219736453125, 0.601237158906250},
{0.774082801906250, 0.642658574812500},
{0.416299322109375, 0.161398788656250},
{0.602987853250000, 0.989968641265625},
{0.089514399500000, 0.575051700359375},
{0.786335975187500, 0.240626511406250},
{0.231815968843750, 0.794970413296875},
{0.318750014656250, 0.443002308546875},
{0.505566445812500, 0.623698635640625},
{0.637067640531250, 0.039603148984375},
{0.180883847734375, 0.072668413890625},
{0.917325380828125, 0.796656456390625},
{0.526756746078125, 0.259060873750000},
{0.334878599875000, 0.893762925093750},
{0.225425998046875, 0.315310650828125},
{0.794319650703125, 0.478886922328125},
{0.445957801906250, 0.724689824812500},
{0.915440720703125, 0.656963966125000},
{0.302640369765625, 0.116357903265625},
{0.698976641187500, 0.774455388406250},
{0.177148254953125, 0.684077206343750},
{0.949633261734375, 0.138224135796875},
{0.045314677375000, 0.913381656000000},
{0.458038607125000, 0.401547691687500},
{0.713980484156250, 0.625879926296875},
{0.643255797593750, 0.155068738921875},
{0.025074889437500, 0.091885738250000},
{0.766702341031250, 0.927996303765625},
{0.664447403859375, 0.489416693968750},
{0.352787232453125, 0.772961294593750},
{0.025478249937500, 0.320623736265625},
{0.854074830531250, 0.352849697156250},
{0.286530910140625, 0.622227763453125},
{0.977727943875000, 0.553082892328125},
{0.350685461906250, 0.153596186453125},
{0.557230392156250, 0.880514705875000},
{0.153169756421875, 0.555146535265625},
{0.789461819796875, 0.116237049781250},
{0.168804934109375, 0.900309289718750},
{0.377355421296875, 0.330038940000000},
{0.612679025703125, 0.720586141078125},
{0.508506990359375, 0.054982936484375},
{0.196917624109375, 0.096532545187500},
{0.910867175625000, 0.931752899046875},
{0.728171685203125, 0.443690916265625},
{0.435674328421875, 0.845052115796875},
{0.182230392156250, 0.325827205875000},
{0.884087952218750, 0.286524716109375},
{0.448834987281250, 0.579381097156250},
{0.884436274500000, 0.588235294109375},
{0.439752211921875, 0.123635853468750},
{0.688559795171875, 0.981591765453125},
{0.198151308765625, 0.522112716656250},
{0.901703992203125, 0.011242603312500},
{0.128530229140625, 0.840431613031250},
{0.317826892046875, 0.321872989109375},
{0.696508847734375, 0.744543413890625},
{0.688847218843750, 0.216845413296875},
{0.004724588125000, 0.188791578953125},
{0.787802165859375, 0.832912283093750},
{0.567052101359375, 0.371480565281250},
{0.302148254953125, 0.996577206343750},
{0.243986693593750, 0.251067502968750},
{0.825795788375000, 0.474201552750000},
{0.430456688062500, 0.699456330843750},
{0.931258709546875, 0.747276624031250},
{0.325564970921875, 0.057461358984375},
{0.575724486109375, 0.777044835781250},
{0.058436791328125, 0.525030808593750},
{0.759808761421875, 0.157401366203125},
{0.010191088203125, 0.862153371203125},
{0.411476654468750, 0.296344298265625},
{0.505607996843750, 0.519919121093750},
{0.563898662484375, 0.182813307812500},
{0.007229657953125, 0.013359518093750},
{0.826564677375000, 0.913381656000000},
{0.615744562484375, 0.440019215312500},
{0.273049196593750, 0.752824731078125},
{0.126134788453125, 0.489345154640625},
{0.776528750687500, 0.346344691359375},
{0.271093019843750, 0.673808825390625},
{0.766504140937500, 0.549462718109375},
{0.321690920375000, 0.115305748390625},
{0.552058600328125, 0.980660703968750},
{0.086158139312500, 0.677526975812500},
{0.868983666328125, 0.196905808593750},
{0.133506990359375, 0.992482936484375},
{0.428389216390625, 0.479768192062500},
{0.642183234343750, 0.594837245046875},
{0.693414534828125, 0.060006743953125},
{0.177263875390625, 0.179881653203125},
{0.822881453125000, 0.799457928828125},
{0.657602996515625, 0.342698708937500},
{0.447833622000000, 0.795251544750000},
{0.049908315421875, 0.432993533953125},
{0.884744019140625, 0.384203634359375},
{0.303495121031250, 0.531469981562500},
{0.829156508796875, 0.635903919875000},
{0.279003945812500, 0.225261135640625},
{0.626154293906250, 0.912625724750000},
{0.009546365468750, 0.639961546609375},
{0.886403666453125, 0.169165570765625},
{0.052259883703125, 0.979845435984375},
{0.459494562484375, 0.330644215312500},
{0.524813081953125, 0.651309302156250},
{0.606787063734375, 0.014769301578125},
{0.010457836296875, 0.143541028515625},
{0.947626461875000, 0.818089897125000},
{0.571917624109375, 0.284032545187500},
{0.416299322109375, 0.911398788656250},
{0.236033046906250, 0.396646543203125},
{0.990797719390625, 0.456105138953125},
{0.333626471656250, 0.616508772390625},
{0.787209695250000, 0.716482502812500},
{0.439756778562500, 0.194376370593750},
{0.552179912484375, 0.863477370312500},
{0.243449504812500, 0.737462829328125},
{0.822881453125000, 0.049457928828125},
{0.058436791328125, 0.806280808593750},
{0.267237456671875, 0.347940860375000},
{0.659100916609375, 0.628228892687500},
{0.525109796625000, 0.082597554609375},
{0.135951233515625, 0.201639822421875},
{0.761255117500000, 0.954583567718750},
{0.556490320328125, 0.399823343937500},
{0.258466777984375, 0.838967372437500},
{0.104074830531250, 0.352849697156250},
{0.803389216390625, 0.386018192062500},
{0.392068237890625, 0.666186056234375},
{0.984505036140625, 0.690144858781250},
{0.277932351500000, 0.052908284062500},
{0.713196808109375, 0.833865323640625},
{0.102915496515625, 0.537034646437500},
{0.979027962734375, 0.100901043359375},
{0.106115002812500, 0.885378765484375},
{0.473302101359375, 0.455953221531250},
{0.685199429843750, 0.551526284734375},
{0.706158315421875, 0.128306033953125},
{0.147265783328125, 0.057294542578125},
{0.959039534828125, 0.919381743953125},
{0.646013875390625, 0.398631653203125},
{0.458517234312500, 0.944548392906250},
{0.056758487734375, 0.295253452109375},
{0.988836204046875, 0.303383539265625},
{0.254309364156250, 0.711985215093750},
{0.775799306890625, 0.567053766171875},
{0.340921091031250, 0.224871303765625},
{0.620506746078125, 0.821560873750000},
{0.160012464359375, 0.690720392781250},
{0.788430456281250, 0.036399376265625},
{0.220355124109375, 0.963720045187500},
{0.287648582281250, 0.493282790234375},
{0.536503468843750, 0.724657913296875},
{0.552148254953125, 0.246577206343750},
{0.116843560203125, 0.107753416265625},
{0.842374240812500, 0.843123040078125},
{0.696912145562500, 0.310820697562500},
{0.330881907437500, 0.958779769828125},
{0.169338615078125, 0.306386117906250},
{0.927058600328125, 0.355660703968750},
{0.477915496515625, 0.630784646437500},
{0.947107614062500, 0.571599844062500},
{0.411298019234375, 0.231101317375000},
{0.634196201015625, 0.774126137000000},
{0.133506990359375, 0.617482936484375},
{0.911148929828125, 0.091845178421875},
{0.229027962734375, 0.850901043359375},
{0.433436791328125, 0.431280808593750},
{0.570900611203125, 0.618903012437500},
{0.691994851500000, 0.099783284062500},
{0.212500058671875, 0.147009234203125},
{0.951732996484375, 0.784567680765625},
{0.538479240078125, 0.442006235093750},
{0.495691442609375, 0.839534956375000},
{0.092295940359375, 0.445600341546875},
{0.806758487734375, 0.295253452109375},
{0.374367956281250, 0.505149376265625},
{0.822584307093750, 0.538276272453125},
{0.381143881328125, 0.075372076343750},
{0.746648411546875, 0.886075859718750},
{0.225690524703125, 0.572657414093750},
{0.908135803781250, 0.224055233687500},
{0.100557411375000, 0.856953939312500},
{0.326749240812500, 0.257185540078125},
{0.703382877203125, 0.594522854968750},
{0.635381359671875, 0.247395865796875},
{0.091503945812500, 0.170573635640625},
{0.773093560203125, 0.773280760015625},
{0.699864601359375, 0.424703221531250},
{0.337978249937500, 0.822576861265625},
{0.177278602781250, 0.415547689312500},
{0.950716102781250, 0.382344564312500},
{0.345102996515625, 0.701097146437500},
{0.850941098000000, 0.563378403515625},
{0.447833622000000, 0.045251544750000},
{0.584417841375000, 0.933427886640625},
{0.117581880000000, 0.710837083484375},
{0.864601654468750, 0.093219298265625},
{0.115674527437500, 0.963752289609375},
{0.438901999203125, 0.285253217765625},
{0.641702341031250, 0.693621303765625},
{0.662939186515625, 0.064765218421875},
{0.102763510390625, 0.047682223171875},
{0.821904890625000, 0.958637616328125},
{0.617697313656250, 0.380208463203125},
{0.387554934109375, 0.806559289718750},
{0.096212938656250, 0.263020963203125},
{0.867422654015625, 0.379107524531250},
{0.383386910937500, 0.555990102843750},
{0.880607996843750, 0.519919121093750},
{0.252252211921875, 0.162698353468750},
{0.642183234343750, 0.969837245046875},
{0.196583993796875, 0.685156627843750},
{0.963190877593750, 0.199420555781250},
{0.199633261734375, 0.888224135796875},
{0.350685461906250, 0.434846186453125},
{0.614399749906250, 0.627404791062500},
{0.608747165328125, 0.225123234406250},
{0.244958663437500, 0.019149132359375},
{0.880021551015625, 0.966470884812500},
{0.686627065421875, 0.259165408953125},
{0.463246963312500, 0.880304660312500},
{0.209424527437500, 0.448127289609375},
{0.850148582281250, 0.305782790234375},
{0.321802423796875, 0.647870625703125},
{0.853921568625000, 0.709558823531250},
{0.496648411546875, 0.136075859718750},
{0.534266910125000, 0.759900787234375},
{0.018208405500000, 0.610922261187500},
{0.981815968843750, 0.044970413296875},
{0.147265783328125, 0.807294542578125},
{0.278764594718750, 0.293912076453125},
{0.697168203437500, 0.508447093109375},
{0.569936479140625, 0.090431613031250},
{0.199633261734375, 0.231974135796875},
{0.878530229140625, 0.840431613031250},
{0.530111691484375, 0.313423081343750},
{0.287503755000000, 0.877340989734375},
{0.020926812156250, 0.490443845953125},
{0.925118529859375, 0.463551966546875},
{0.460281853234375, 0.514086682671875},
{0.946583993796875, 0.685156627843750},
{0.432953992203125, 0.011242603312500},
{0.681996963312500, 0.841242160312500},
{0.035440756328125, 0.741967535328125},
{0.826814113265625, 0.129551982203125},
{0.102763510390625, 0.797682223171875},
{0.257250986453125, 0.431315283906250},
{0.601735045687500, 0.535855464812500},
{0.523947313656250, 0.161458463203125},
{0.089514399500000, 0.200051700359375},
{0.998871711468750, 0.969931745984375},
{0.725557411375000, 0.325703939312500},
{0.411054041609375, 0.940728892687500},
{0.092214949156250, 0.395649388656250},
{0.947571808109375, 0.318240323640625},
{0.385150528859375, 0.598791693968750},
{0.760340045031250, 0.666060247875000},
{0.385468447859375, 0.173477959078125},
{0.619715387546875, 0.985882883562500},
{0.110693313734375, 0.604613051578125},
{0.759083993796875, 0.247656627843750},
{0.198151308765625, 0.803362716656250},
{0.368295322046875, 0.475490672062500},
{0.522431987421875, 0.579676484406250},
{0.668614601359375, 0.002828221531250},
{0.159061291453125, 0.115786836312500},
{0.885343915375000, 0.797979216453125},
{0.502049350968750, 0.290322923484375},
{0.361960011265625, 0.916515061968750},
{0.212257727531250, 0.374306940859375},
{0.808006746078125, 0.446560873750000},
{0.489786425109375, 0.696416160921875},
{0.914775229140625, 0.625807223171875},
{0.272394756234375, 0.082105300000000},
{0.697833622000000, 0.795251544750000},
{0.146614411140625, 0.654378257218750},
{0.959039534828125, 0.169381743953125},
{0.056931985968750, 0.888270723078125},
{0.493765020015625, 0.415613958984375},
{0.727875617718750, 0.674521589734375},
{0.646232996843750, 0.172262871093750},
{0.052267234312500, 0.085173392906250},
{0.794171695281250, 0.897748994546875},
{0.664407018781250, 0.458335411421875},
{0.318273390046875, 0.755820190000000},
{0.046247165328125, 0.350123234406250},
{0.865799202015625, 0.333466330906250},
{0.285498339406250, 0.583855022421875},
{0.956147102093750, 0.505576277234375},
{0.318949826718750, 0.141763441546875},
{0.510429611953125, 0.887119489765625},
{0.184919978109375, 0.561328326953125},
{0.759417624109375, 0.096532545187500},
{0.130403602781250, 0.937032064312500},
{0.394595719296875, 0.367380713734375},
{0.567731703031250, 0.747610961484375},
{0.538716806515625, 0.039836274843750}
};
#include <numeric>

#include "Renderer.h"

#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../source/Nabla/COpenCLHandler.h"


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
	//return std::move(m_driver->getGPUObjectsFromAssets<ICPUSpecializedShader>(&shader,&shader+1u)->operator[](0));
	return driver->getGPUObjectsFromAssets<ICPUSpecializedShader>(&shader,&shader+1u)->operator[](0);
}
// TODO: make these util function in `IDescriptorSetLayout` -> Assign: @Hazardu
auto fillIotaDescriptorBindingDeclarations = [](auto* outBindings, ISpecializedShader::E_SHADER_STAGE accessFlags, uint32_t count, asset::E_DESCRIPTOR_TYPE descType=asset::EDT_INVALID, uint32_t startIndex=0u) -> void
{
	for (auto i=0u; i<count; i++)
	{
		outBindings[i].binding = i+startIndex;
		outBindings[i].type = descType;
		outBindings[i].count = 1u;
		outBindings[i].stageFlags = accessFlags;
		outBindings[i].samplers = nullptr;
	}
};


Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, scene::ISceneManager* _smgr, bool useDenoiser) :
		m_useDenoiser(useDenoiser),	m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager),
		m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
	#ifdef _IRR_BUILD_OPTIX_
		m_optixManager(), m_cudaStream(nullptr), m_optixContext(),
	#endif
		rrShapeCache(), rrInstances(), m_prevView(), m_sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
		m_maxRaysPerDispatch(0), m_staticViewData{{0.f,0.f,0.f},0u,{0.f,0.f},{0.f,0.f},{0u,0u},0u,0u},
		m_raytraceCommonData{core::matrix4SIMD(),core::matrix3x4SIMD(),0,0,0},
		m_indirectDrawBuffers{nullptr},m_cullPushConstants{core::matrix4SIMD(),1.f,0u,0u,0u},m_cullWorkGroups(0u),
		m_raygenWorkGroups{0u,0u},m_resolveWorkGroups{0u,0u},
		m_visibilityBuffer(nullptr),tmpTonemapBuffer(nullptr),m_colorBuffer(nullptr)
{
	while (m_useDenoiser)
	{
		m_useDenoiser = false;
#ifdef _IRR_BUILD_OPTIX_
		m_optixManager = ext::OptiX::Manager::create(m_driver, m_assetManager->getFileSystem());
		if (!m_optixManager)
			break;
		m_cudaStream = m_optixManager->getDeviceStream(0);
		if (!m_cudaStream)
			break;
		m_optixContext = m_optixManager->createContext(0);
		if (!m_optixContext)
			break;
		OptixDenoiserOptions opts = {OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL,OPTIX_PIXEL_FORMAT_HALF3};
		m_denoiser = m_optixContext->createDenoiser(&opts);
		if (!m_denoiser)
			break;

		m_useDenoiser = true;
#endif
		break;
	}

	{
		constexpr auto cullingDescriptorCount = 3u;
		IGPUDescriptorSetLayout::SBinding bindings[cullingDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,cullingDescriptorCount,asset::EDT_STORAGE_BUFFER);
		bindings[2u].count = 2u;
		m_cullDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+3u);
	}

	m_visibilityBufferFillShaders[0] = specializedShaderFromFile(m_assetManager,"../fillVisBuffer.vert");
	m_visibilityBufferFillShaders[1] = specializedShaderFromFile(m_assetManager,"../fillVisBuffer.frag");
	{
		ICPUDescriptorSetLayout::SBinding binding;
		fillIotaDescriptorBindingDeclarations(&binding,ISpecializedShader::ESS_VERTEX,1u,asset::EDT_STORAGE_BUFFER);

		auto dsLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(&binding,&binding+1u);
		m_visibilityBufferFillPipelineLayoutCPU = core::make_smart_refctd_ptr<ICPUPipelineLayout>(nullptr,nullptr,nullptr,core::smart_refctd_ptr(dsLayout),nullptr,nullptr);

		// TODO: @Crisspl find a way to stop the user from such insanity as moving from the bundle's dynamic array
		//m_visibilityBufferFillPipelineLayoutGPU = std::move(m_driver->getGPUObjectsFromAssets<ICPUPipelineLayout>(&m_visibilityBufferFillPipelineLayoutCPU,&m_visibilityBufferFillPipelineLayoutCPU+1u)->operator[](0));
		m_visibilityBufferFillPipelineLayoutGPU = core::smart_refctd_ptr(m_driver->getGPUObjectsFromAssets<ICPUPipelineLayout>(&m_visibilityBufferFillPipelineLayoutCPU,&m_visibilityBufferFillPipelineLayoutCPU+1u)->operator[](0));
		// TODO: @Crisspl make it so that I can create GPU pipeline with `const` smartpointer layouts as arguments!
		m_perCameraRasterDSLayout = core::smart_refctd_ptr<IGPUDescriptorSetLayout>(const_cast<video::IGPUDescriptorSetLayout*>(m_visibilityBufferFillPipelineLayoutGPU->getDescriptorSetLayout(1u)));
	}
	
	{
		constexpr auto raytracingCommonDescriptorCount = 6u;
		IGPUDescriptorSetLayout::SBinding bindings[raytracingCommonDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,raytracingCommonDescriptorCount);
		bindings[0].type = asset::EDT_UNIFORM_BUFFER;
		bindings[1].type = asset::EDT_STORAGE_IMAGE;
		bindings[2].type = asset::EDT_STORAGE_BUFFER;
		bindings[3].type = asset::EDT_STORAGE_BUFFER;
		bindings[4].type = asset::EDT_STORAGE_BUFFER;
		bindings[5].type = asset::EDT_STORAGE_BUFFER;

		m_commonRaytracingDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+raytracingCommonDescriptorCount);
	}
	{
		ISampler::SParams samplerParams;
		samplerParams.TextureWrapU = samplerParams.TextureWrapV = samplerParams.TextureWrapW = ISampler::ETC_CLAMP_TO_EDGE;
		samplerParams.MinFilter = samplerParams.MaxFilter = ISampler::ETF_NEAREST;
		samplerParams.MipmapMode = ISampler::ESMM_NEAREST;
		samplerParams.AnisotropicFilter = 0u;
		samplerParams.CompareEnable = false;
		auto sampler = m_driver->createGPUSampler(samplerParams);

		constexpr auto raygenDescriptorCount = 6u;
		IGPUDescriptorSetLayout::SBinding bindings[raygenDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,raygenDescriptorCount);
		bindings[0].type = asset::EDT_UNIFORM_TEXEL_BUFFER;
		bindings[1].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[1].samplers = &sampler;
		bindings[2].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[2].samplers = &sampler;
		bindings[3].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[3].samplers = &sampler;
		bindings[4].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[4].samplers = &sampler;
		bindings[5].type = asset::EDT_COMBINED_IMAGE_SAMPLER;
		bindings[5].samplers = &sampler;

		m_raygenDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+raygenDescriptorCount);
	}
	{
		constexpr auto resolveDescriptorCount = 2u;
		IGPUDescriptorSetLayout::SBinding bindings[resolveDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,resolveDescriptorCount);
		bindings[0].type = asset::EDT_STORAGE_BUFFER;
		bindings[1].type = asset::EDT_STORAGE_IMAGE;

		m_resolveDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+resolveDescriptorCount);
	}
}

Renderer::~Renderer()
{
	deinit();
}


Renderer::InitializationData Renderer::initSceneObjects(const SAssetBundle& meshes)
{
	InitializationData retval;
	retval.globalMeta = meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>();
	assert(retval.globalMeta);
	{
		auto* glslMaterialBackendGlobalDS = retval.globalMeta->m_global.m_ds0.get();
		m_globalBackendDataDS = m_driver->getGPUObjectsFromAssets(&glslMaterialBackendGlobalDS,&glslMaterialBackendGlobalDS+1)->front();
	}
	
	core::vector<CullData_t> cullData;
	core::vector<std::pair<core::smart_refctd_ptr<IGPUMeshBuffer>,uint32_t>> gpuMeshBuffers;
	{
		core::vector<ICPUMesh*> meshesToProcess;
		core::vector<ICPUMeshBuffer*> meshBuffersToProcess;
		{
			auto contents = meshes.getContents();

			ICPUSpecializedShader* shaders[] = { m_visibilityBufferFillShaders[0].get(),m_visibilityBufferFillShaders[1].get() };

			struct VisibilityBufferPipelineKey
			{
				inline bool operator==(const VisibilityBufferPipelineKey& other) const
				{
					return vertexParams == other.vertexParams && frontFaceIsCCW == other.frontFaceIsCCW;
				}

				SVertexInputParams vertexParams;
				uint8_t frontFaceIsCCW;
			};
			struct VisibilityBufferPipelineKeyHash
			{
				inline std::size_t operator()(const VisibilityBufferPipelineKey& key) const
				{
					std::basic_string_view view(reinterpret_cast<const char*>(&key), sizeof(key));
					return std::hash<decltype(view)>()(view);
				}
			};
			core::unordered_map<VisibilityBufferPipelineKey, core::smart_refctd_ptr<ICPURenderpassIndependentPipeline>, VisibilityBufferPipelineKeyHash> visibilityBufferFillPipelines;

			{
				meshesToProcess.reserve(contents.size());

				uint32_t drawableCount = 0u;
				for (const auto& asset : contents)
				{
					auto cpumesh = static_cast<asset::ICPUMesh*>(asset.get());
					auto meshBuffers = cpumesh->getMeshBuffers();
					assert(!meshBuffers.empty());

					meshesToProcess.push_back(cpumesh);

					drawableCount += meshBuffers.size()*retval.globalMeta->getAssetSpecificMetadata(cpumesh)->m_instances.size();
				}
				cullData.resize(drawableCount);
			}
			auto cullDataIt = cullData.begin();
			for (const auto& asset : contents)
			{
				auto cpumesh = static_cast<asset::ICPUMesh*>(asset.get());
				const auto* meta = retval.globalMeta->getAssetSpecificMetadata(cpumesh);
				const auto& instanceData = meta->m_instances;
				const auto& instanceAuxData = meta->m_instanceAuxData;

				auto cullDataBaseBegin = cullDataIt;
				auto meshBuffers = cpumesh->getMeshBuffers();
				for (auto cpumb : meshBuffers)
				{
					assert(cpumb->getInstanceCount()==instanceData.size());

					// set up Visibility Buffer pipelines
					{
						auto oldPipeline = cpumb->getPipeline();
						auto vertexInputParams = oldPipeline->getVertexInputParams();
						const bool frontFaceIsCCW = oldPipeline->getRasterizationParams().frontFaceIsCCW;
						auto found = visibilityBufferFillPipelines.find(VisibilityBufferPipelineKey{ vertexInputParams,frontFaceIsCCW });

						core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> newPipeline;
						if (found != visibilityBufferFillPipelines.end())
							newPipeline = core::smart_refctd_ptr(found->second);
						else
						{
							vertexInputParams.enabledAttribFlags &= 0b1101u;
							asset::SPrimitiveAssemblyParams assemblyParams;
							assemblyParams.primitiveType = oldPipeline->getPrimitiveAssemblyParams().primitiveType;
							asset::SRasterizationParams rasterParams;
							rasterParams.faceCullingMode = EFCM_NONE;
							rasterParams.frontFaceIsCCW = !frontFaceIsCCW; // compensate for Nabla's default camer being left handed
							newPipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
								core::smart_refctd_ptr(m_visibilityBufferFillPipelineLayoutCPU), shaders, shaders + 2u,
								vertexInputParams, asset::SBlendParams{}, assemblyParams, rasterParams
								);
							visibilityBufferFillPipelines.emplace(VisibilityBufferPipelineKey{ vertexInputParams,frontFaceIsCCW }, core::smart_refctd_ptr(newPipeline));
						}
						cpumb->setPipeline(std::move(newPipeline));
					}

					CullData_t& baseCullData = *(cullDataIt++);
					{
						const auto aabbOriginal = cpumb->getBoundingBox();
						baseCullData.aabbMinEdge.x = aabbOriginal.MinEdge.X;
						baseCullData.aabbMinEdge.y = aabbOriginal.MinEdge.Y;
						baseCullData.aabbMinEdge.z = aabbOriginal.MinEdge.Z;
						baseCullData.globalObjectID = cpumb->getBaseInstance();
						baseCullData.aabbMaxEdge.x = aabbOriginal.MaxEdge.X;
						baseCullData.aabbMaxEdge.y = aabbOriginal.MaxEdge.Y;
						baseCullData.aabbMaxEdge.z = aabbOriginal.MaxEdge.Z;
						baseCullData.drawID = meshBuffersToProcess.size();
					}

					meshBuffersToProcess.push_back(cpumb);
				}

				// set up scene bounds and lights
				const auto aabbOriginal = cpumesh->getBoundingBox();
				for (auto j=0u; j<instanceData.size(); j++)
				{
					const auto& worldTform = instanceData.begin()[j].worldTform;
					const auto& aux = instanceAuxData.begin()[j];

					ext::RadeonRays::MockSceneManager::ObjectData objectData;
					{
						objectData.tform = worldTform;
						objectData.mesh = nullptr;
						objectData.instanceGUIDPerMeshBuffer.reserve(meshBuffers.size());
						for (auto src=cullDataBaseBegin; src!=cullDataIt; src++)
						{
							auto dst = src+j*meshBuffers.size();
							*dst = *src;
							dst->globalObjectID += j;
							objectData.instanceGUIDPerMeshBuffer.push_back(dst->globalObjectID);
						}
					}
					m_mock_smgr.m_objectData.push_back(std::move(objectData));

					m_sceneBound.addInternalBox(core::transformBoxEx(aabbOriginal,worldTform));

					auto emitter = aux.frontEmitter;
					if (emitter.type != ext::MitsubaLoader::CElementEmitter::Type::INVALID)
					{
						assert(emitter.type == ext::MitsubaLoader::CElementEmitter::Type::AREA);

						SLight newLight(aabbOriginal,worldTform); // TODO: should be an OBB

						const float weight = newLight.computeFluxBound(emitter.area.radiance)*emitter.area.samplingWeight;
						if (weight <= FLT_MIN)
							continue;

						retval.lights.emplace_back(std::move(newLight));
						retval.lightRadiances.push_back(emitter.area.radiance);
						retval.lightPDF.push_back(weight);
					}
				}
			}
		}

		// this wont get rid of identical pipelines
		IMeshManipulator::homogenizePrimitiveTypeAndIndices(meshBuffersToProcess.begin(), meshBuffersToProcess.end(), EPT_TRIANGLE_LIST);
		// set up BLAS
		m_rrManager->makeRRShapes(rrShapeCache, meshBuffersToProcess.begin(), meshBuffersToProcess.end());

		// convert to GPU objects
		auto gpuMeshes = m_driver->getGPUObjectsFromAssets(meshesToProcess.data(),meshesToProcess.data()+meshesToProcess.size());
		{
			auto objectDataIt = m_mock_smgr.m_objectData.begin();
			for (auto i=0u; i<gpuMeshes->size(); i++)
			{
				const auto instanceCount = retval.globalMeta->getAssetSpecificMetadata(meshesToProcess[i])->m_instances.size();
				for (size_t j=0u; j<instanceCount; j++)
					(objectDataIt++)->mesh = gpuMeshes->operator[](i);
			}
		}
		// there should be usually no conversion going on here, just cache retrieval, we just do it to later sort by pipeline
		auto gpuObjs = m_driver->getGPUObjectsFromAssets(meshBuffersToProcess.data(),meshBuffersToProcess.data()+meshBuffersToProcess.size());
		gpuMeshBuffers.resize(gpuObjs->size());
		for (auto i=0u; i<gpuObjs->size(); i++)
			gpuMeshBuffers[i] = {core::smart_refctd_ptr(gpuObjs->operator[](i)),i};
		std::sort(	gpuMeshBuffers.begin(), gpuMeshBuffers.end(),
					[](const auto& lhs, const auto& rhs) -> bool
					{
						return std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(lhs)->getPipeline()<std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(rhs)->getPipeline();
					}
		);

		// set up Radeon Rays instances and TLAS
		{
			core::vector<ext::RadeonRays::MockSceneManager::ObjectGUID> ids(m_mock_smgr.m_objectData.size());
			std::iota(ids.begin(),ids.end(),0u);
			m_rrManager->makeRRInstances(rrInstances, &m_mock_smgr, rrShapeCache, m_assetManager, ids.begin(), ids.end());
			m_rrManager->attachInstances(rrInstances.begin(), rrInstances.end());
		}
	}

	core::vector<DrawElementsIndirectCommand_t> mdiData;
	{
		core::vector<uint32_t> meshbufferIDToDrawID(gpuMeshBuffers.size());
		{
			MDICall call;
			auto initNewMDI = [&call](const core::smart_refctd_ptr<IGPUMeshBuffer>& gpumb) -> void
			{
				std::copy(gpumb->getVertexBufferBindings(),gpumb->getVertexBufferBindings()+IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT,call.vertexBindings);
				call.indexBuffer = core::smart_refctd_ptr(gpumb->getIndexBufferBinding().buffer);
				call.pipeline = core::smart_refctd_ptr<const IGPURenderpassIndependentPipeline>(gpumb->getPipeline());
			};
			initNewMDI(std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(gpuMeshBuffers.front()));
			call.mdiOffset = 0u;
			call.mdiCount = 0u;
			auto queueUpMDI = [&](const MDICall& call) -> void
			{
				m_mdiDrawCalls.emplace_back(call);
			};
			uint32_t drawBaseInstance = 0u;
			for (const auto& item : gpuMeshBuffers)
			{
				const auto gpumb = std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(item);

				const uint32_t drawID = mdiData.size();
				meshbufferIDToDrawID[std::get<uint32_t>(item)] = drawID;

				mdiData.emplace_back(DrawElementsIndirectCommand_t{
					static_cast<uint32_t>(gpumb->getIndexCount()), // pretty sure index count should be a uint32_t
					0u,
					static_cast<uint32_t>(gpumb->getIndexBufferBinding().offset/sizeof(uint32_t)),
					static_cast<uint32_t>(gpumb->getBaseVertex()), // pretty sure base vertex should be a uint32_t
					drawBaseInstance
				});
				drawBaseInstance += gpumb->getInstanceCount();

				bool haveToBreakMDI = false;
				if (gpumb->getPipeline()!=call.pipeline.get())
					haveToBreakMDI = true;
				if (!std::equal(call.vertexBindings,call.vertexBindings+IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT,gpumb->getVertexBufferBindings()))
					haveToBreakMDI = true;
				if (gpumb->getIndexBufferBinding().buffer!=call.indexBuffer)
					haveToBreakMDI = true;
				if (haveToBreakMDI)
				{
					queueUpMDI(call);
					initNewMDI(gpumb);
					call.mdiOffset = drawID*sizeof(DrawElementsIndirectCommand_t);
					call.mdiCount = 1u;
				}
				else
					call.mdiCount++;
			}
			queueUpMDI(call);
		}
		for (auto& cull : cullData)
			cull.drawID = meshbufferIDToDrawID[cull.drawID];
	}
	m_cullPushConstants.currentCommandBufferIx = 0x0u;
	m_cullPushConstants.maxDrawCount = mdiData.size();
	m_cullPushConstants.maxObjectCount = cullData.size();
	m_cullWorkGroups = (m_cullPushConstants.maxObjectCount-1u)/WORKGROUP_SIZE+1u;

	m_cullDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_cullDSLayout));
	m_perCameraRasterDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_perCameraRasterDSLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo infos[4];
		
		infos[0].buffer.size = m_cullPushConstants.maxObjectCount*sizeof(DrawData_t);
		infos[0].buffer.offset = 0u;
		infos[0].desc = m_driver->createDeviceLocalGPUBufferOnDedMem(infos[0].buffer.size);
		
		infos[1].buffer.size = m_cullPushConstants.maxObjectCount*sizeof(CullData_t);
		infos[1].buffer.offset = 0u;
		infos[1].desc = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(infos[1].buffer.size,cullData.data());
		cullData.clear();
		
		for (auto offset=2u,i=0u; i<2u; i++)
		{
			auto j = i+offset;
			infos[j].buffer.size = mdiData.size()*sizeof(DrawElementsIndirectCommand_t);
			infos[j].buffer.offset = 0u;
			infos[j].desc = core::smart_refctd_ptr(m_indirectDrawBuffers[i] = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(infos[j].buffer.size,mdiData.data()));
		}
		mdiData.clear();
		
		IGPUDescriptorSet::SWriteDescriptorSet commonWrites[3];
		for (auto i=0u; i<3u; i++)
		{
			commonWrites[i].binding = i;
			commonWrites[i].arrayElement = 0u;
			commonWrites[i].count = 1u;
			commonWrites[i].descriptorType = EDT_STORAGE_BUFFER;
			commonWrites[i].info = infos+i;
		}
		commonWrites[2u].count = 2u;

		auto setDstSetOnAllWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, uint32_t count)
		{
			for (auto i=0u; i<count; i++)
				writes[i].dstSet = dstSet;
		};
		setDstSetOnAllWrites(m_perCameraRasterDS.get(),commonWrites,1u);
		m_driver->updateDescriptorSets(1u,commonWrites,0u,nullptr);
		setDstSetOnAllWrites(m_cullDS.get(),commonWrites,3u);
		m_driver->updateDescriptorSets(3u,commonWrites,0u,nullptr);
	}

	return retval;
}

void Renderer::initSceneNonAreaLights(Renderer::InitializationData& initData)
{
	core::vectorSIMDf _envmapBaseColor;
	_envmapBaseColor.set(0.f,0.f,0.f,1.f);
	for (const auto& emitter : initData.globalMeta->m_global.m_emitters)
	{
		float weight = 0.f;
		switch (emitter.type)
		{
			case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
				_envmapBaseColor += emitter.constant.radiance;
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
	m_staticViewData.envmapBaseColor.x = _envmapBaseColor.x;
	m_staticViewData.envmapBaseColor.y = _envmapBaseColor.y;
	m_staticViewData.envmapBaseColor.z = _envmapBaseColor.z;
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

	auto radianceIn = initData.lightRadiances.begin();
	core::vector<uint64_t> compressedRadiance(m_staticViewData.lightCount,0ull);
	auto radianceOut = compressedRadiance.begin();
	auto divideRadianceByPDF = [UINT_MAX_DOUBLE,weightSumRcp,&partialSum,&outCDF,&radianceIn,&radianceOut](uint32_t prevCDF) -> void
	{
		double inv_prob = NAN;
		const double exactCDF = weightSumRcp*partialSum+double(FLT_MIN);
		if (exactCDF<UINT_MAX_DOUBLE)
		{
			uint32_t thisCDF = *outCDF = static_cast<uint32_t>(exactCDF);
			inv_prob = UINT_MAX_DOUBLE/double(thisCDF-prevCDF);
		}
		else
		{
			assert(exactCDF<UINT_MAX_DOUBLE+1.0);
			*outCDF = 0xdeadbeefu;
			inv_prob = 1.0/(1.0-double(prevCDF)/UINT_MAX_DOUBLE);
		}
		auto tmp = (radianceIn++)->operator*(inv_prob);
		*(radianceOut++) = core::rgb32f_to_rgb19e7(tmp.pointer);
	};

	divideRadianceByPDF(0u);
	for (auto prevCDF=outCDF++; outCDF!=initData.lightCDF.end(); prevCDF=outCDF++)
	{
		partialSum += double(*(++inPDF));

		divideRadianceByPDF(*prevCDF);
	}
}

core::smart_refctd_ptr<IGPUImageView> Renderer::createScreenSizedTexture(E_FORMAT format)
{
	IGPUImage::SCreationParams imgparams;
	imgparams.extent = {m_staticViewData.imageDimensions.x,m_staticViewData.imageDimensions.y,1u};
	imgparams.arrayLayers = 1u;
	imgparams.flags = static_cast<IImage::E_CREATE_FLAGS>(0);
	imgparams.format = format;
	imgparams.mipLevels = 1u;
	imgparams.samples = IImage::ESCF_1_BIT;
	imgparams.type = IImage::ET_2D;

	IGPUImageView::SCreationParams viewparams;
	viewparams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0);
	viewparams.format = format;
	viewparams.image = m_driver->createDeviceLocalGPUImageOnDedMem(std::move(imgparams));
	viewparams.viewType = IGPUImageView::ET_2D;
	viewparams.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
	viewparams.subresourceRange.baseArrayLayer = 0u;
	viewparams.subresourceRange.layerCount = 1u;
	viewparams.subresourceRange.baseMipLevel = 0u;
	viewparams.subresourceRange.levelCount = 1u;

	return m_driver->createGPUImageView(std::move(viewparams));
}

enum E_VISIBILITY_BUFFER_ATTACHMENT
{
	EVBA_DEPTH,
	EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING,
	// TODO: Once we get geometry packer V2 (virtual geometry) no need for these buffers actually (might want/need a barycentric buffer)
	EVBA_NORMALS,
	EVBA_UV_COORDINATES,
	EVBA_COUNT
};

void Renderer::init(const SAssetBundle& meshes,
					core::smart_refctd_ptr<ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize)
{
	deinit();

	core::smart_refctd_ptr<IGPUImageView> visibilityBufferAttachments[EVBA_COUNT];
	// set up Descriptor Sets
	{
		// captures creates m_indirectDrawBuffers, sets up m_mdiDrawCalls ranges, creates m_cullDS, sets m_cullPushConstants and m_cullWorkgroups, creates m_perCameraRasterDS, and captures m_globalBackendDataDS
		auto initData = initSceneObjects(meshes);
		{
			initSceneNonAreaLights(initData);
			finalizeScene(initData);
		}
		
		// figure out the renderable size
		{
			m_staticViewData.imageDimensions = {m_driver->getScreenSize().Width,m_driver->getScreenSize().Height};
			const auto& sensors = initData.globalMeta->m_global.m_sensors;
			if (sensors.size())
			{
				// just grab the first sensor
				const auto& sensor = sensors.front();
				const auto& film = sensor.film;
				assert(film.cropOffsetX == 0);
				assert(film.cropOffsetY == 0);
				m_staticViewData.imageDimensions = {static_cast<uint32_t>(film.cropWidth),static_cast<uint32_t>(film.cropHeight)};
			}
			m_staticViewData.rcpPixelSize = { 2.f/float(m_staticViewData.imageDimensions.x),-2.f/float(m_staticViewData.imageDimensions.y) };
			m_staticViewData.rcpHalfPixelSize = { 1.f/float(m_staticViewData.imageDimensions.x)-1.f,1.f-1.f/float(m_staticViewData.imageDimensions.y) };
		}

		// figure out dispatch sizes
		m_raygenWorkGroups[0] = (m_staticViewData.imageDimensions.x-1u)/WORKGROUP_DIM+1u;
		m_raygenWorkGroups[1] = (m_staticViewData.imageDimensions.y-1u)/WORKGROUP_DIM+1u;
		m_resolveWorkGroups[0] = (m_staticViewData.imageDimensions.x-1u)/WORKGROUP_DIM+1u;
		m_resolveWorkGroups[1] = (m_staticViewData.imageDimensions.y-1u)/WORKGROUP_DIM+1u;

		const auto renderPixelCount = m_staticViewData.imageDimensions.x*m_staticViewData.imageDimensions.y;
		// figure out how much Samples Per Pixel Per Dispatch we can afford
		size_t raygenBufferSize, intersectionBufferSize;
		{
			const auto misSamples = 2u;
			const auto minimumSampleCountPerDispatch = renderPixelCount*misSamples;

			const auto raygenBufferSizePerSample = static_cast<size_t>(minimumSampleCountPerDispatch)*sizeof(::RadeonRays::ray);
			assert(raygenBufferSizePerSample<=rayBufferSize);
			const auto intersectionBufferSizePerSample = static_cast<size_t>(minimumSampleCountPerDispatch)*sizeof(::RadeonRays::Intersection);
			assert(intersectionBufferSizePerSample<=rayBufferSize);
			m_staticViewData.samplesPerPixelPerDispatch = rayBufferSize/(raygenBufferSizePerSample+intersectionBufferSizePerSample);
			assert(m_staticViewData.samplesPerPixelPerDispatch >= 1u);
			printf("Using %d samples\n", m_staticViewData.samplesPerPixelPerDispatch);

			m_staticViewData.samplesPerRowPerDispatch = m_staticViewData.imageDimensions.x*m_staticViewData.samplesPerPixelPerDispatch;

			m_maxRaysPerDispatch = minimumSampleCountPerDispatch*m_staticViewData.samplesPerPixelPerDispatch;
			raygenBufferSize = raygenBufferSizePerSample*m_staticViewData.samplesPerPixelPerDispatch;
			intersectionBufferSize = intersectionBufferSizePerSample*m_staticViewData.samplesPerPixelPerDispatch;
		}

		// set up raycount buffer for RR
		{
			m_rayCountBuffer.buffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t),&m_maxRaysPerDispatch);
			m_rayCountBuffer.asRRBuffer = m_rrManager->linkBuffer(m_rayCountBuffer.buffer.get(), CL_MEM_READ_ONLY);

			ocl::COpenCLHandler::ocl.pclEnqueueAcquireGLObjects(m_rrManager->getCLCommandQueue(), 1u, &m_rayCountBuffer.asRRBuffer.second, 0u, nullptr, nullptr);
		}

		// create out screen-sized textures
		m_accumulation = createScreenSizedTexture(EF_R32G32_UINT);
		m_tonemapOutput = createScreenSizedTexture(EF_A2B10G10R10_UNORM_PACK32);

		//
		{
			// i know what I'm doing
			auto globalBackendDataDSLayout = core::smart_refctd_ptr<IGPUDescriptorSetLayout>(const_cast<IGPUDescriptorSetLayout*>(m_globalBackendDataDS->getLayout()));


			// cull
			{
				SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t)};
				m_cullPipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1u,core::smart_refctd_ptr(globalBackendDataDSLayout),core::smart_refctd_ptr(m_cullDSLayout),nullptr,nullptr);
				m_cullPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_cullPipelineLayout),gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../cull.comp"));
			}
			

			SPushConstantRange raytracingCommonPCRange{ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t)};
			m_commonRaytracingDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_commonRaytracingDSLayout));

			// raygen
			{
				m_raygenPipelineLayout = m_driver->createGPUPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_raygenDSLayout),
					nullptr
				);
				(std::ofstream("material_declarations.glsl") << "#define _NBL_EXT_MITSUBA_LOADER_VT_STORAGE_VIEW_COUNT " << initData.globalMeta->m_global.getVTStorageViewCount() << "\n" << initData.globalMeta->m_global.m_materialCompilerGLSL_declarations).close();
				m_raygenPipeline = m_driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_raygenPipelineLayout),gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../raygen.comp"));

				m_raygenDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_raygenDSLayout));
			}

			// resolve
			{
				constexpr auto resolveDescriptorCount = 2u;
				IGPUDescriptorSetLayout::SBinding bindings[resolveDescriptorCount];
				fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,resolveDescriptorCount);
				bindings[0].type = asset::EDT_STORAGE_BUFFER;
				bindings[1].type = asset::EDT_STORAGE_IMAGE;

				m_resolveDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+resolveDescriptorCount);

				m_resolvePipelineLayout = m_driver->createGPUPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_resolveDSLayout),
					nullptr
				);
				m_resolvePipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_resolvePipelineLayout),gpuSpecializedShaderFromFile(m_assetManager,m_driver,m_useDenoiser ? "../resolveForDenoiser.comp":"../resolve.comp"));

				m_resolveDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_resolveDSLayout));
			}


			//
			constexpr uint32_t descriptorCountInSet[3] = { 6u,6u,2u };
			constexpr uint32_t descriptorExclScanSum[4] = { 0u,descriptorCountInSet[0],descriptorCountInSet[0]+descriptorCountInSet[1],descriptorCountInSet[0]+descriptorCountInSet[1]+descriptorCountInSet[2] };


			auto createEmptyInteropBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, InteropBuffer& interopBuffer, size_t size) -> void
			{
				interopBuffer.buffer = m_driver->createDeviceLocalGPUBufferOnDedMem(size);
				interopBuffer.asRRBuffer = m_rrManager->linkBuffer(interopBuffer.buffer.get(), CL_MEM_READ_ONLY);

				info->buffer.size = size;
				info->buffer.offset = 0u;
				info->desc = core::smart_refctd_ptr(interopBuffer.buffer);
			};
			auto createFilledBufferAndSetUpInfo = [&](IGPUDescriptorSet::SDescriptorInfo* info, size_t size, const void* data) -> void
			{
				info->buffer.size = size;
				info->buffer.offset = 0u;
				info->desc = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(size,data);
			};
			auto createFilledBufferAndSetUpInfoFromStruct = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& _struct) -> void
			{
				createFilledBufferAndSetUpInfo(info,sizeof(_struct),&_struct);
			};
			auto createFilledBufferAndSetUpInfoFromVector = [createFilledBufferAndSetUpInfo](IGPUDescriptorSet::SDescriptorInfo* info, const auto& vector) -> void
			{
				createFilledBufferAndSetUpInfo(info,vector.size()*sizeof(decltype(*vector.data())),vector.data());
			};

			auto setImageInfo = [](IGPUDescriptorSet::SDescriptorInfo* info, const asset::E_IMAGE_LAYOUT imageLayout, core::smart_refctd_ptr<IGPUImageView>&& imageView) -> void
			{
				info->image.imageLayout = imageLayout;
				info->image.sampler = nullptr; // storage image dont have samplers, and the combined sampler image views we have all use immutable samplers
				info->desc = std::move(imageView);
			};

			IGPUDescriptorSet::SDescriptorInfo infos[descriptorExclScanSum[3]];


			auto setDstSetAndDescTypesOnWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, IGPUDescriptorSet::SDescriptorInfo* _infos, const std::initializer_list<asset::E_DESCRIPTOR_TYPE>& list)
			{
				auto typeIt = list.begin();
				for (auto i=0u; i<list.size(); i++)
				{
					writes[i].dstSet = dstSet;
					writes[i].binding = i;
					writes[i].arrayElement = 0u;
					writes[i].count = 1u;
					writes[i].descriptorType = *(typeIt++);
					writes[i].info = _infos+i;
				}
			};
			IGPUDescriptorSet::SWriteDescriptorSet writes[descriptorExclScanSum[3]];


			// set up m_commonRaytracingDS
			{
				auto commonInfos = infos+descriptorExclScanSum[0];
				auto commonWrites = writes+descriptorExclScanSum[0];
				createFilledBufferAndSetUpInfoFromStruct(commonInfos+0,m_staticViewData);
				setImageInfo(commonInfos+1,asset::EIL_GENERAL,core::smart_refctd_ptr(m_accumulation));
				createEmptyInteropBufferAndSetUpInfo(commonInfos+2,m_rayBuffer,raygenBufferSize);
				createFilledBufferAndSetUpInfoFromVector(commonInfos+3,initData.lightCDF);
				createFilledBufferAndSetUpInfoFromVector(commonInfos+4,initData.lights);
				createFilledBufferAndSetUpInfoFromVector(commonInfos+5,initData.lightRadiances);
				initData = {}; // reclaim some memory

				setDstSetAndDescTypesOnWrites(m_commonRaytracingDS.get(),commonWrites,commonInfos,{EDT_UNIFORM_BUFFER,EDT_STORAGE_IMAGE,EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER,EDT_STORAGE_BUFFER});
			}
			// set up m_raygenDS
			{
				auto scrambleTexture = createScreenSizedTexture(EF_R32G32_UINT);
				{
					constexpr auto ScrambleStateChannels = 2u;
					core::vector<uint32_t> random(renderPixelCount*ScrambleStateChannels);
					// generate
					{
						core::RandomSampler rng(0xbadc0ffeu);
						for (auto& pixel : random)
							pixel = rng.nextSample();
					}
					// upload
					auto gpuBuff = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(random.size()*sizeof(uint32_t),random.data());
					IGPUImage::SBufferCopy region;
					//region.imageSubresource.aspectMask = ;
					region.imageSubresource.layerCount = 1u;
					region.imageExtent = {m_staticViewData.imageDimensions.x,m_staticViewData.imageDimensions.y,1u};
					m_driver->copyBufferToImage(gpuBuff.get(), scrambleTexture->getCreationParameters().image.get(), 1u, &region);
				}
				visibilityBufferAttachments[EVBA_DEPTH] = createScreenSizedTexture(EF_D32_SFLOAT);
				visibilityBufferAttachments[EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING] = createScreenSizedTexture(EF_R32G32_UINT);
				visibilityBufferAttachments[EVBA_NORMALS] = createScreenSizedTexture(EF_R16G16_SNORM);
				visibilityBufferAttachments[EVBA_UV_COORDINATES] = createScreenSizedTexture(EF_R16G16_SFLOAT);

				auto raygenInfos = infos+descriptorExclScanSum[1];
				auto raygenWrites = writes+descriptorExclScanSum[1];
				//! set up GPU sampler
				{
					// TODO: maybe use in the future to stop a converged render
					const auto maxSamples = sampleSequence->getSize()/(sizeof(uint32_t)*MaxDimensions);
					assert(maxSamples==MAX_ACCUMULATED_SAMPLES);
					// upload sequence to GPU
					auto gpubuf = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
					raygenInfos[0].desc = m_driver->createGPUBufferView(gpubuf.get(), asset::EF_R32G32B32_UINT);
				}
				setImageInfo(raygenInfos+1,asset::EIL_SHADER_READ_ONLY_OPTIMAL,std::move(scrambleTexture));
				for (auto i=0u; i<EVBA_COUNT; i++)
					setImageInfo(raygenInfos+2+i,asset::EIL_SHADER_READ_ONLY_OPTIMAL,core::smart_refctd_ptr(visibilityBufferAttachments[i]));
				

				setDstSetAndDescTypesOnWrites(m_raygenDS.get(),raygenWrites,raygenInfos,{EDT_UNIFORM_TEXEL_BUFFER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER,EDT_COMBINED_IMAGE_SAMPLER });
			}
			// set up m_resolveDS
			{
				auto resolveInfos = infos+descriptorExclScanSum[2];
				auto resolveWrites = writes+descriptorExclScanSum[2];
				createEmptyInteropBufferAndSetUpInfo(resolveInfos+0,m_intersectionBuffer,intersectionBufferSize);
				core::smart_refctd_ptr<IGPUImageView> tonemapOutputStorageView;
				{
					IGPUImageView::SCreationParams viewparams = m_tonemapOutput->getCreationParameters();
					viewparams.format = EF_R32_UINT;
					tonemapOutputStorageView = m_driver->createGPUImageView(std::move(viewparams));
				}
				setImageInfo(resolveInfos+1,asset::EIL_GENERAL,std::move(tonemapOutputStorageView));
				

				setDstSetAndDescTypesOnWrites(m_resolveDS.get(),resolveWrites,resolveInfos,{EDT_STORAGE_BUFFER,EDT_STORAGE_IMAGE});
			}

			m_driver->updateDescriptorSets(descriptorExclScanSum[3], writes, 0u, nullptr);
		}
	}


	m_visibilityBuffer = m_driver->addFrameBuffer();
	m_visibilityBuffer->attach(EFAP_DEPTH_ATTACHMENT, core::smart_refctd_ptr(visibilityBufferAttachments[EVBA_DEPTH]));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(visibilityBufferAttachments[EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING]));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT1, core::smart_refctd_ptr(visibilityBufferAttachments[EVBA_NORMALS]));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT2, core::smart_refctd_ptr(visibilityBufferAttachments[EVBA_UV_COORDINATES]));

	tmpTonemapBuffer = m_driver->addFrameBuffer();
	tmpTonemapBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_accumulation));

	m_colorBuffer = m_driver->addFrameBuffer();
	m_colorBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_tonemapOutput));

#ifdef _NBL_BUILD_OPTIX_
	while (m_denoiser)
	{
		m_denoiser->computeMemoryResources(&m_denoiserMemReqs,&m_staticViewData.imageDimensions.x);
#if TODO
		auto inputBuffSz = (kOptiXPixelSize*EDI_COUNT)*renderPixelCount;
		m_denoiserInputBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(inputBuffSz),core::dont_grab);
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&m_denoiserInputBuffer, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)))
			break;
		m_denoiserStateBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(m_denoiserMemReqs.stateSizeInBytes),core::dont_grab);
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&m_denoiserStateBuffer)))
			break;
		m_denoisedBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(kOptiXPixelSize*renderPixelCount), core::dont_grab);
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&m_denoisedBuffer, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)))
			break;
		if (m_rayBuffer->getSize()<m_denoiserMemReqs.recommendedScratchSizeInBytes)
			break;
		m_denoiserScratchBuffer = core::smart_refctd_ptr(m_rayBuffer); // could alias the denoised output to this as well
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&m_denoiserScratchBuffer, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)))
			break;

		auto setUpOptiXImage2D = [&](OptixImage2D& img, uint32_t pixelSize) -> void
		{
			img = {};
			img.width = m_staticViewData.imageDimensions.x;
			img.height = m_staticViewData.imageDimensions.y;
			img.pixelStrideInBytes = pixelSize;
			img.rowStrideInBytes = img.width*img.pixelStrideInBytes;
		};

		setUpOptiXImage2D(m_denoiserInputs[EDI_COLOR],kOptiXPixelSize);
		m_denoiserInputs[EDI_COLOR].data = 0;
		m_denoiserInputs[EDI_COLOR].format = OPTIX_PIXEL_FORMAT_HALF3;
		setUpOptiXImage2D(m_denoiserInputs[EDI_ALBEDO],kOptiXPixelSize);
		m_denoiserInputs[EDI_ALBEDO].data = m_denoiserInputs[EDI_COLOR].rowStrideInBytes*m_denoiserInputs[EDI_COLOR].height;
		m_denoiserInputs[EDI_ALBEDO].format = OPTIX_PIXEL_FORMAT_HALF3;
		setUpOptiXImage2D(m_denoiserInputs[EDI_NORMAL],kOptiXPixelSize);
		m_denoiserInputs[EDI_NORMAL].data = m_denoiserInputs[EDI_ALBEDO].data+m_denoiserInputs[EDI_ALBEDO].rowStrideInBytes*m_denoiserInputs[EDI_ALBEDO].height;;
		m_denoiserInputs[EDI_NORMAL].format = OPTIX_PIXEL_FORMAT_HALF3;

		setUpOptiXImage2D(m_denoiserOutput,kOptiXPixelSize);
		m_denoiserOutput.format = OPTIX_PIXEL_FORMAT_HALF3;
#endif
		break;
	}
#endif
}


void Renderer::deinit()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);

	glFinish();

#ifdef _IRR_BUILD_OPTIX_
	if (m_cudaStream)
		cuda::CCUDAHandler::cuda.pcuStreamSynchronize(m_cudaStream);
	m_denoiserInputBuffer = {};
	m_denoiserScratchBuffer = {};
	m_denoisedBuffer = {};
	m_denoiserStateBuffer = {};
	m_denoiserInputs[EDI_COLOR] = {};
	m_denoiserInputs[EDI_ALBEDO] = {};
	m_denoiserInputs[EDI_NORMAL] = {};
	m_denoiserOutput = {};
#endif

	// TODO: @Crisspl When we finally make a EF_RGB19E7 format enum and appropriate encode/decode functions (and finally finish the driver hardware support for formats queries)
	//if (m_accumulation)
		//ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_accumulation.get(),"sceneReferred.exr",asset::EF_R16G16B16A16_SFLOAT);
	if (m_tonemapOutput)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_tonemapOutput.get(),"tonemapped.png",asset::EF_R8G8B8_SRGB);
	if (m_visibilityBuffer)
	{
		m_driver->removeFrameBuffer(m_visibilityBuffer);
		m_visibilityBuffer = nullptr;
	}
	if (tmpTonemapBuffer)
	{
		m_driver->removeFrameBuffer(tmpTonemapBuffer);
		tmpTonemapBuffer = nullptr;
	}
	if (m_colorBuffer)
	{
		m_driver->removeFrameBuffer(m_colorBuffer);
		m_colorBuffer = nullptr;
	}
	m_accumulation = m_tonemapOutput = nullptr;
	
	auto deleteInteropBuffer = [&](InteropBuffer& buffer) -> void
	{
		if (buffer.asRRBuffer.first)
			m_rrManager->deleteRRBuffer(buffer.asRRBuffer.first);
		buffer = {};
	};
	deleteInteropBuffer(m_intersectionBuffer);
	deleteInteropBuffer(m_rayBuffer);
	// release the last OpenCL object and wait for OpenCL to finish
	ocl::COpenCLHandler::ocl.pclEnqueueReleaseGLObjects(commandQueue, 1u, &m_rayCountBuffer.asRRBuffer.second, 1u, nullptr, nullptr);
	ocl::COpenCLHandler::ocl.pclFlush(commandQueue);
	ocl::COpenCLHandler::ocl.pclFinish(commandQueue);
	deleteInteropBuffer(m_rayCountBuffer);

	m_resolveWorkGroups[0] = m_resolveWorkGroups[1] = 0u;
	m_resolveDS = nullptr;

	m_raygenWorkGroups[0] = m_raygenWorkGroups[1] = 0u;
	m_raygenDS = nullptr;
	m_commonRaytracingDS = nullptr;
	m_globalBackendDataDS = nullptr;

	m_cullPipelineLayout = nullptr;
	m_raygenPipelineLayout = nullptr;
	m_resolvePipelineLayout = nullptr;
	m_cullPipeline = nullptr;
	m_raygenPipeline = nullptr;
	m_resolvePipeline = nullptr;

	m_perCameraRasterDS = nullptr;

	m_cullWorkGroups = 0u;
	m_cullPushConstants = {core::matrix4SIMD(),1.f,0u,0u,0u};
	m_cullDS = nullptr;
	m_mdiDrawCalls.clear();
	m_indirectDrawBuffers[1] = m_indirectDrawBuffers[0] = nullptr;

	m_raytraceCommonData = {core::matrix4SIMD(),core::matrix3x4SIMD(),0,0,0};
	m_staticViewData = {{0.f,0.f,0.f},0u,{0.f,0.f},{0.f,0.f},{0u,0u},0u,0u};
	m_maxRaysPerDispatch = 0u;
	std::fill_n(m_prevView.pointer(),12u,0.f);
	m_sceneBound = core::aabbox3df(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

	m_rrManager->detachInstances(rrInstances.begin(),rrInstances.end());
	m_rrManager->deleteInstances(rrInstances.begin(),rrInstances.end());
	rrInstances.clear();
	m_mock_smgr = {};

	m_rrManager->deleteShapes(rrShapeCache.m_gpuAssociative.begin(), rrShapeCache.m_gpuAssociative.end()); // or CPU assoc?
	rrShapeCache = {};
}

// one day it will just work like that
//#include <nbl/builtin/glsl/sampling/box_muller_transform.glsl>

void Renderer::render(nbl::ITimer* timer)
{
	if (m_cullPushConstants.maxObjectCount==0u)
		return;


	auto camera = m_smgr->getActiveCamera();
	camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(timer->getTime()).count());
	camera->render();

	// check if camera moved
	{
		const auto currentView = camera->getViewMatrix();
		auto properEquals = [](const auto& lhs, const auto& rhs) -> bool
		{
			const float rotationTolerance = 1.01f;
			const float positionTolerance = 1.005f;
			for (auto r=0; r<3u; r++)
			for (auto c=0; c<4u; c++)
			{
				const float ratio = core::abs(rhs.rows[r][c]/lhs.rows[r][c]);
				// TODO: do by ULP
				if (core::isnan(ratio) || core::isinf(ratio))
					continue;
				const float tolerance = c!=3u ? rotationTolerance:positionTolerance;
				if (ratio>tolerance || ratio*tolerance<1.f)
					return false;
			}
			return true;
		};
		if (!properEquals(currentView,m_prevView))
		{
			m_raytraceCommonData.framesDispatched = 0u;

			m_driver->setRenderTarget(tmpTonemapBuffer);
			{
				uint32_t clearAccumulation[4] = { 0,0,0,0 };
				m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, clearAccumulation);
			}
		
			m_prevView = currentView;
		}
		else // need this to stop mouse cursor drift
		{
			core::matrix3x4SIMD invView;
			m_prevView.getInverse(invView);
			camera->setRelativeTransformationMatrix(invView.getAsRetardedIrrlichtMatrix());
		}
	}
	// draw jittered frame
	{
		// jitter with AA AntiAliasingSequence
		const auto modifiedViewProj = [&](uint32_t frameID)
		{
			const float stddev = 0.707f;
			const float* sample = AntiAliasingSequence[frameID];
			const float phi = core::PI<float>()*(2.f*sample[1]-1.f);
			const float sinPhi = sinf(phi);
			const float cosPhi = cosf(phi);
			const float truncated = sample[0]*0.99999f+0.00001f;
			const float r = sqrtf(-2.f*logf(truncated))*stddev;
			core::matrix4SIMD jitterMatrix;
			jitterMatrix.rows[0][3] = cosPhi*r*m_staticViewData.rcpPixelSize.x;
			jitterMatrix.rows[1][3] = sinPhi*r*m_staticViewData.rcpPixelSize.y;
			return core::concatenateBFollowedByA(jitterMatrix,core::concatenateBFollowedByA(camera->getProjectionMatrix(),m_prevView));
		}(m_raytraceCommonData.framesDispatched++);
		m_raytraceCommonData.rcpFramesDispatched = 1.f/float(m_raytraceCommonData.framesDispatched);

		IGPUDescriptorSet* descriptors[3] = { m_globalBackendDataDS.get(),m_cullDS.get(),nullptr };

		m_driver->bindComputePipeline(m_cullPipeline.get());
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_cullPipelineLayout.get(),0u,2u,descriptors,nullptr);
		{
			m_cullPushConstants.viewProjMatrix = modifiedViewProj;
			m_cullPushConstants.viewProjDeterminant = core::determinant(modifiedViewProj);
			m_driver->pushConstants(m_cullPipelineLayout.get(),ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t),&m_cullPushConstants);
		}
		// TODO: Occlusion Culling against HiZ Buffer
		m_driver->dispatch(m_cullWorkGroups, 1u, 1u);
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

		m_driver->setRenderTarget(m_visibilityBuffer);
		{ // clear
			m_driver->clearZBuffer();
			uint32_t clearTriangleID[4] = {0xffffffffu,0,0,0};
			m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, clearTriangleID);
			float zero[4] = { 0.f,0.f,0.f,0.f };
			m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT1, zero);
			m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT2, zero);
		}
		m_driver->bindDescriptorSets(EPBP_GRAPHICS,m_visibilityBufferFillPipelineLayoutGPU.get(),1u,1u,descriptors+1u,nullptr);
		for (const auto& call : m_mdiDrawCalls)
		{
			m_driver->bindGraphicsPipeline(call.pipeline.get());
			m_driver->drawIndexedIndirect(call.vertexBindings,EPT_TRIANGLE_LIST,EIT_32BIT,call.indexBuffer.get(),m_indirectDrawBuffers[m_cullPushConstants.currentCommandBufferIx].get(),call.mdiOffset,call.mdiCount,sizeof(DrawElementsIndirectCommand_t));
		}
		// flip MDI buffers
		m_cullPushConstants.currentCommandBufferIx ^= 0x01u;

		// prepare camera data for raytracing
		modifiedViewProj.getInverseTransform(m_raytraceCommonData.inverseMVP);
		const auto cameraPosition = core::vectorSIMDf().set(camera->getAbsolutePosition());
		for (auto i=0u; i<3u; i++)
			m_raytraceCommonData.ndcToV.rows[i] = m_raytraceCommonData.inverseMVP.rows[3]*cameraPosition[i]-m_raytraceCommonData.inverseMVP.rows[i];
	}

	// generate rays
	{
		IGPUDescriptorSet* descriptorSets[] = {m_commonRaytracingDS.get(),m_raygenDS.get()};
		m_driver->bindDescriptorSets(EPBP_COMPUTE, m_raygenPipelineLayout.get(), 1, 2, descriptorSets, nullptr);
		m_driver->bindComputePipeline(m_raygenPipeline.get());
		m_driver->pushConstants(m_raygenPipelineLayout.get(),ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t),&m_raytraceCommonData);
		m_driver->dispatch(m_raygenWorkGroups[0], m_raygenWorkGroups[1], 1);
		// probably wise to flush all caches
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);
	}

	// do radeon rays
	m_rrManager->update(&m_mock_smgr,rrInstances.begin(),rrInstances.end());
	if (m_rrManager->hasImplicitCL2GLSync())
		glFlush();
	else
		glFinish();

	{
		auto commandQueue = m_rrManager->getCLCommandQueue();

		const cl_mem clObjects[] = {m_rayBuffer.asRRBuffer.second,m_intersectionBuffer.asRRBuffer.second};
		const auto objCount = sizeof(clObjects)/sizeof(cl_mem);

		cl_event acquired = nullptr;
		ocl::COpenCLHandler::ocl.pclEnqueueAcquireGLObjects(commandQueue,objCount,clObjects,0u,nullptr,&acquired);

		clEnqueueWaitForEvents(commandQueue,1u,&acquired);
		m_rrManager->getRadeonRaysAPI()->QueryIntersection(m_rayBuffer.asRRBuffer.first,m_rayCountBuffer.asRRBuffer.first,m_maxRaysPerDispatch,m_intersectionBuffer.asRRBuffer.first,nullptr,nullptr);
		cl_event raycastDone = nullptr;
		clEnqueueMarker(commandQueue,&raycastDone);

		if (m_rrManager->hasImplicitCL2GLSync())
		{
			ocl::COpenCLHandler::ocl.pclEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, nullptr);
			ocl::COpenCLHandler::ocl.pclFlush(commandQueue);
		}
		else
		{
			cl_event released;
			ocl::COpenCLHandler::ocl.pclEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, &released);
			ocl::COpenCLHandler::ocl.pclFlush(commandQueue);
			ocl::COpenCLHandler::ocl.pclWaitForEvents(1u,&released);
		}
	}

	// use raycast results
	{
		m_driver->bindDescriptorSets(EPBP_COMPUTE, m_resolvePipelineLayout.get(), 2, 1, &m_resolveDS.get(), nullptr);
		m_driver->bindComputePipeline(m_resolvePipeline.get());
		m_driver->dispatch(m_resolveWorkGroups[0], m_resolveWorkGroups[1], 1);

		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
#ifndef _NBL_BUILD_OPTIX_
			|GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT
#else
			|(m_denoisedBuffer.getObject() ? (GL_PIXEL_BUFFER_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT):(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT))
#endif
		);
	}

	m_raytraceCommonData.samplesComputedPerPixel += m_staticViewData.samplesPerPixelPerDispatch;

	// TODO: tonemap properly
#ifdef _NBL_BUILD_OPTIX_
	if (m_denoisedBuffer.getObject())
	{
#if TODO
		cuda::CCUDAHandler::acquireAndGetPointers(&m_denoiserInputBuffer,&m_denoiserScratchBuffer+1,m_cudaStream);

		if (m_denoiser->getLastSetupResult()!=OPTIX_SUCCESS)
		{
			m_denoiser->setup(	m_cudaStream,&rSize[0],m_denoiserStateBuffer,m_denoiserStateBuffer.getObject()->getSize(),
								m_denoiserScratchBuffer,m_denoiserMemReqs.recommendedScratchSizeInBytes);
		}

		OptixImage2D denoiserInputs[EDI_COUNT];
		for (auto i=0; i<EDI_COUNT; i++)
		{
			denoiserInputs[i] = m_denoiserInputs[i];
			denoiserInputs[i].data = m_denoiserInputs[i].data+m_denoiserInputBuffer.asBuffer.pointer;
		}
		m_denoiser->computeIntensity(	m_cudaStream,denoiserInputs+0,m_denoiserScratchBuffer,m_denoiserScratchBuffer,
										m_denoiserMemReqs.recommendedScratchSizeInBytes,m_denoiserMemReqs.recommendedScratchSizeInBytes);

		OptixDenoiserParams m_denoiserParams = {};
		volatile float kConstant = 0.0001f;
		m_denoiserParams.blendFactor = core::min(1.f-1.f/core::max(kConstant*float(m_framesDone*m_samplesPerPixelPerDispatch),1.f),0.25f);
		m_denoiserParams.denoiseAlpha = 0u;
		m_denoiserParams.hdrIntensity = m_denoiserScratchBuffer.asBuffer.pointer+m_denoiserMemReqs.recommendedScratchSizeInBytes;
		m_denoiserOutput.data = m_denoisedBuffer.asBuffer.pointer;
		m_denoiser->invoke(	m_cudaStream,&m_denoiserParams,denoiserInputs,denoiserInputs+EDI_COUNT,&m_denoiserOutput,
							m_denoiserScratchBuffer,m_denoiserMemReqs.recommendedScratchSizeInBytes);

		void* scratch[16];
		cuda::CCUDAHandler::releaseResourcesToGraphics(scratch,&m_denoiserInputBuffer,&m_denoiserScratchBuffer+1,m_cudaStream);

		auto glbuf = static_cast<COpenGLBuffer*>(m_denoisedBuffer.getObject());
		auto gltex = static_cast<COpenGLFilterableTexture*>(m_tonemapOutput.get());
		COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_UNPACK_BUFFER,glbuf->getOpenGLName());
		COpenGLExtensionHandler::extGlTextureSubImage2D(gltex->getOpenGLName(),gltex->getOpenGLTextureType(),0,0,0,rSize[0],rSize[1],GL_RGB,GL_HALF_FLOAT,nullptr);
		COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
#endif
	}
	else
#endif
	{
		auto oldVP = m_driver->getViewPort();
		m_driver->setViewPort(core::recti(0u,0u,m_staticViewData.imageDimensions.x,m_staticViewData.imageDimensions.y));
		m_driver->blitRenderTargets(tmpTonemapBuffer, m_colorBuffer, false, false, {}, {}, true);
		m_driver->setViewPort(oldVP);
	}
}


const float Renderer::AntiAliasingSequence[4096][2] =
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
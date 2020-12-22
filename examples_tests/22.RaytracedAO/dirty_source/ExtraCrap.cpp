#include <numeric>

#include "ExtraCrap.h"

#include "irr/ext/ScreenShot/ScreenShot.h"

#include "../source/Irrlicht/COpenCLHandler.h"


#ifndef _IRR_BUILD_OPTIX_
	#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#endif

using namespace irr;
using namespace irr::asset;
using namespace irr::video;


constexpr uint32_t kOptiXPixelSize = sizeof(uint16_t)*3u;

core::smart_refctd_ptr<ICPUSpecializedShader> specializedShaderFromFile(IAssetManager* assetManager, const char* path)
{
	auto bundle = assetManager->getAsset(path, {});
	return core::move_and_static_cast<ICPUSpecializedShader>(*bundle.getContents().begin());
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
		rrShapeCache(), rrInstances(), m_sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX),
		m_maxRaysPerDispatch(0), m_staticViewData{{0.f,0.f,0.f},0u,{0.f,0.f},{0.f,0.f},{0u,0u},0u,0u},
		m_raytraceCommonData{core::matrix3x4SIMD(),core::matrix3x4SIMD(),0.f,0u,0u,0.f},
		m_indirectDrawBuffers{nullptr},m_cullPushConstants{core::matrix4SIMD(),0u,0u,1.f,0xdeadbeefu},m_cullWorkGroups(0u),
		m_raygenWorkGroups{0u,0u},m_resolveWorkGroups{0u,0u},
		m_visibilityBuffer(nullptr),tmpTonemapBuffer(nullptr),m_colorBuffer(nullptr)
{
#ifdef _IRR_BUILD_OPTIX_
	while (useDenoiser)
	{
		useDenoiser = false;
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

		useDenoiser = true;
		break;
	}
#endif

	constexpr auto cullingOutputDescriptorCount = 2u;
	{
		constexpr auto cullingDescriptorCount = cullingOutputDescriptorCount+2u;
		IGPUDescriptorSetLayout::SBinding bindings[cullingDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,cullingDescriptorCount,asset::EDT_STORAGE_BUFFER);
		bindings[3u].count = 2u;
		m_cullDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+4u);
		SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t)};
		m_cullPipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1u,nullptr,core::smart_refctd_ptr(m_cullDSLayout),nullptr,nullptr);
		m_cullPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_cullPipelineLayout),gpuSpecializedShaderFromFile(m_assetManager,m_driver,"../cull.comp"));
	}

	m_visibilityBufferFillShaders[0] = specializedShaderFromFile(m_assetManager,"../fillVisBuffer.vert");
	m_visibilityBufferFillShaders[1] = specializedShaderFromFile(m_assetManager,"../fillVisBuffer.frag");
	{
		ICPUDescriptorSetLayout::SBinding bindings[cullingOutputDescriptorCount];
		fillIotaDescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_VERTEX,cullingOutputDescriptorCount,asset::EDT_STORAGE_BUFFER);

		auto dsLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings,bindings+2u);
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
	
	auto getInstances = [&retval](const auto& cpumesh) -> core::vector<ext::MitsubaLoader::IMeshMetadata::Instance>
	{
		auto* meta = cpumesh->getMetadata();
		assert(meta && core::strcmpi(meta->getLoaderName(), ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
		const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
		retval.globalMeta = meshmeta->globalMetadata.get();
		assert(retval.globalMeta);

		// WARNING !!!
		// all this instance-related things is a rework candidate since mitsuba loader supports instances
		// (all this metadata should be global, but meshbuffers has instanceCount correctly set
		// and globalDS with per-instance data (transform, normal matrix, instructions offsets, etc)
		return meshmeta->getInstances();
	};

	core::vector<std::tuple<core::smart_refctd_ptr<IGPUMeshBuffer>,core::smart_refctd_ptr<ICPUMesh>,ext::RadeonRays::MockSceneManager::ObjectGUID>> gpuMeshBuffers_CPUMesh_Object;
	{
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
			for (auto* it=contents.begin(); it!=contents.end(); it++)
			{
				auto cpumesh = static_cast<asset::ICPUMesh*>(it->get());
				const auto& instances = getInstances(cpumesh);
				assert(cpumesh->getInstanceCount()==instances.size());

				auto meshBufferCount = cpumesh->getMeshBufferCount();
				for (auto i = 0u; i < meshBufferCount; i++)
				{
					// TODO: get rid of `getMeshBuffer` and `getMeshBufferCount`, just return a range as `getMeshBuffers`
					auto cpumb = cpumesh->getMeshBuffer(i);

					// set up Visibility Buffer pipelines
					{
						auto oldPipeline = cpumb->getPipeline();

						// if global SSBO with instruction streams not captured yet
						if (!m_globalBackendDataDS)
						{
							// a bit roundabout but oh well what can we do (global metadata needs to be more useful)
							auto* pipelinemeta = static_cast<ext::MitsubaLoader::CMitsubaPipelineMetadata*>(oldPipeline->getMetadata());
							auto* glslMaterialBackendGlobalDS = pipelinemeta->getDescriptorSet();
							m_globalBackendDataDS = m_driver->getGPUObjectsFromAssets(&glslMaterialBackendGlobalDS,&glslMaterialBackendGlobalDS+1)->front();
						}

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
					meshBuffersToProcess.push_back(cpumb);
					gpuMeshBuffers_CPUMesh_Object.emplace_back(nullptr,core::smart_refctd_ptr<ICPUMesh>(cpumesh),m_mock_smgr.m_objectData.size());
				}

				// set up scene bounds and lights
				const auto aabbOriginal = cpumesh->getBoundingBox();
				for (auto instance : instances)
				{
					m_mock_smgr.m_objectData.push_back({instance.tform,nullptr,{}});

					m_sceneBound.addInternalBox(core::transformBoxEx(aabbOriginal,instance.tform));

					if (instance.emitter.type != ext::MitsubaLoader::CElementEmitter::Type::INVALID)
					{
						assert(instance.emitter.type == ext::MitsubaLoader::CElementEmitter::Type::AREA);

						SLight newLight(cpumesh->getBoundingBox(), instance.tform);

						const float weight = newLight.computeFluxBound(instance.emitter.area.radiance) * instance.emitter.area.samplingWeight;
						if (weight <= FLT_MIN)
							continue;

						retval.lights.emplace_back(std::move(newLight));
						retval.lightRadiances.push_back(instance.emitter.area.radiance);
						retval.lightPDF.push_back(weight);
					}
				}
			}
		}

		// set up BVH
		IMeshManipulator::homogenizePrimitiveTypeAndIndices(meshBuffersToProcess.begin(), meshBuffersToProcess.end(), EPT_TRIANGLE_LIST);
		m_rrManager->makeRRShapes(rrShapeCache, meshBuffersToProcess.begin(), meshBuffersToProcess.end());

		// convert to GPU objects and sort so they're ordered by pipeline
		auto gpuObjs = m_driver->getGPUObjectsFromAssets(meshBuffersToProcess.data(),meshBuffersToProcess.data()+meshBuffersToProcess.size());
		for (auto i=0u; i<gpuObjs->size(); i++)
		{
			std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(gpuMeshBuffers_CPUMesh_Object[i]) = core::smart_refctd_ptr(gpuObjs->operator[](i));
			// there should be usually no conversion going on here, just cache retrieval
			auto cpumesh = std::get<core::smart_refctd_ptr<ICPUMesh>>(gpuMeshBuffers_CPUMesh_Object[i]).get();
			auto gpuMesh = m_driver->getGPUObjectsFromAssets(&cpumesh,&cpumesh+1);
			m_mock_smgr.m_objectData[std::get<ext::RadeonRays::MockSceneManager::ObjectGUID>(gpuMeshBuffers_CPUMesh_Object[i])].mesh = core::smart_refctd_ptr(gpuMesh->operator[](0));
		}
		std::sort(	gpuMeshBuffers_CPUMesh_Object.begin(), gpuMeshBuffers_CPUMesh_Object.end(),
					[](const auto& lhs, const auto& rhs) -> bool
					{
						return std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(lhs)->getPipeline()<std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(rhs)->getPipeline();
					}
		);
	}

	core::vector<DrawElementsIndirectCommand_t> mdiData;
	core::vector<ObjectStaticData_t> objectStaticData;
	core::vector<CullData_t> cullData;
	{
		MDICall call;
		auto initNewMDI = [&call](const core::smart_refctd_ptr<IGPUMeshBuffer>& gpumb) -> void
		{
			std::copy(gpumb->getVertexBufferBindings(),gpumb->getVertexBufferBindings()+IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT,call.vertexBindings);
			call.indexBuffer = core::smart_refctd_ptr(gpumb->getIndexBufferBinding().buffer);
			call.pipeline = core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>(const_cast<IGPURenderpassIndependentPipeline*>(gpumb->getPipeline()));
		};
		initNewMDI(std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(gpuMeshBuffers_CPUMesh_Object.front()));
		call.mdiOffset = 0u;
		call.mdiCount = 0u;
		auto queueUpMDI = [&](const MDICall& call) -> void
		{
			m_mdiDrawCalls.emplace_back(call);
		};
		for (auto gpuMeshBuffer_cpuMesh_object : gpuMeshBuffers_CPUMesh_Object)
		{
			auto gpumb = std::get<core::smart_refctd_ptr<IGPUMeshBuffer>>(gpuMeshBuffer_cpuMesh_object);

			const uint32_t baseInstance = objectStaticData.size();
			gpumb->setBaseInstance(baseInstance);

			const uint32_t drawID = mdiData.size();
			const auto aabb = gpumb->getBoundingBox();
			const auto& instances = getInstances(std::get<core::smart_refctd_ptr<ICPUMesh>>(gpuMeshBuffer_cpuMesh_object));
			for (auto j=0u; j<gpumb->getInstanceCount(); j++)
			{
				core::matrix3x4SIMD worldMatrix,normalMatrix;
				worldMatrix = instances[j].tform;
				worldMatrix.getSub3x3InverseTranspose(normalMatrix);

				m_mock_smgr.m_objectData[std::get<ext::RadeonRays::MockSceneManager::ObjectGUID>(gpuMeshBuffer_cpuMesh_object)+j].instanceGUIDPerMeshBuffer.push_back(objectStaticData.size());
				objectStaticData.emplace_back(ObjectStaticData_t{
					{normalMatrix.rows[0].x,normalMatrix.rows[0].y,normalMatrix.rows[0].z},worldMatrix.getPseudoDeterminant()[0],
					{normalMatrix.rows[1].x,normalMatrix.rows[1].y,normalMatrix.rows[1].z},0xdeadbeefu,
					{normalMatrix.rows[2].x,normalMatrix.rows[2].y,normalMatrix.rows[2].z},0xdeadbeefu
				});
				cullData.emplace_back(CullData_t{
					worldMatrix,
					{aabb.MinEdge.X,aabb.MinEdge.Y,aabb.MinEdge.Z},drawID,
					{aabb.MaxEdge.X,aabb.MaxEdge.Y,aabb.MaxEdge.Z},baseInstance
				});
			}
			mdiData.emplace_back(DrawElementsIndirectCommand_t{
				static_cast<uint32_t>(gpumb->getIndexCount()), // pretty sure index count should be a uint32_t
				0u,
				static_cast<uint32_t>(gpumb->getIndexBufferBinding().offset/sizeof(uint32_t)),
				static_cast<uint32_t>(gpumb->getBaseVertex()), // pretty sure base vertex should be a uint32_t
				baseInstance
			});

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

	// set up Radeon Rays instances
	{
		core::vector<ext::RadeonRays::MockSceneManager::ObjectGUID> ids(m_mock_smgr.m_objectData.size());
		std::iota(ids.begin(),ids.end(),0u);
		m_rrManager->makeRRInstances(rrInstances, &m_mock_smgr, rrShapeCache, m_assetManager, ids.begin(), ids.end());
		m_rrManager->attachInstances(rrInstances.begin(), rrInstances.end());
	}


	m_cullPushConstants.maxObjectCount = objectStaticData.size();
	m_cullPushConstants.currentCommandBufferIx = 0x0u;
	m_cullWorkGroups = (m_cullPushConstants.maxObjectCount-1u)/WORKGROUP_SIZE+1u;

	m_cullDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_cullDSLayout));
	m_perCameraRasterDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_perCameraRasterDSLayout));
	{
		IGPUDescriptorSet::SDescriptorInfo infos[5];

		infos[0].buffer.size = m_cullPushConstants.maxObjectCount*sizeof(ObjectStaticData_t);
		infos[0].buffer.offset = 0u;
		infos[0].desc = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(infos[0].buffer.size,objectStaticData.data());
		objectStaticData.clear();
		
		infos[1].buffer.size = m_cullPushConstants.maxObjectCount*sizeof(DrawData_t);
		infos[1].buffer.offset = 0u;
		infos[1].desc = m_driver->createDeviceLocalGPUBufferOnDedMem(infos[1].buffer.size);
		
		infos[2].buffer.size = m_cullPushConstants.maxObjectCount*sizeof(CullData_t);
		infos[2].buffer.offset = 0u;
		infos[2].desc = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(infos[2].buffer.size,cullData.data());
		cullData.clear();
		
		for (auto i=3u; i<=4u; i++)
		{
			infos[i].buffer.size = mdiData.size()*sizeof(DrawElementsIndirectCommand_t);
			infos[i].buffer.offset = 0u;
			infos[i].desc = core::smart_refctd_ptr(m_indirectDrawBuffers[i-3u] = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(infos[i].buffer.size,mdiData.data()));
		}
		mdiData.clear();
		
		IGPUDescriptorSet::SWriteDescriptorSet commonWrites[4];
		for (auto i=0u; i<4u; i++)
		{
			commonWrites[i].binding = i;
			commonWrites[i].arrayElement = 0u;
			commonWrites[i].count = 1u;
			commonWrites[i].descriptorType = EDT_STORAGE_BUFFER;
			commonWrites[i].info = infos+i;
		}
		commonWrites[3u].count = 2u;

		auto setDstSetOnAllWrites = [](IGPUDescriptorSet* dstSet, IGPUDescriptorSet::SWriteDescriptorSet* writes, uint32_t count)
		{
			for (auto i=0u; i<count; i++)
				writes[i].dstSet = dstSet;
		};
		setDstSetOnAllWrites(m_perCameraRasterDS.get(),commonWrites,2u);
		m_driver->updateDescriptorSets(2u,commonWrites,0u,nullptr);
		setDstSetOnAllWrites(m_cullDS.get(),commonWrites,4u);
		m_driver->updateDescriptorSets(4u,commonWrites,0u,nullptr);
	}

	return retval;
}

void Renderer::initSceneNonAreaLights(Renderer::InitializationData& initData)
{
	core::vectorSIMDf _envmapBaseColor;
	_envmapBaseColor.set(0.f,0.f,0.f,1.f);
	for (auto emitter : initData.globalMeta->emitters)
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
			const auto& sensors = initData.globalMeta->sensors;
			if (sensors.size())
			{
				// just grab the first sensor
				const auto& sensor = sensors.front();
				const auto& film = sensor.film;
				assert(film.cropOffsetX == 0);
				assert(film.cropOffsetY == 0);
				m_staticViewData.imageDimensions = {static_cast<uint32_t>(film.cropWidth),static_cast<uint32_t>(film.cropHeight)};
			}
			m_staticViewData.rcpPixelSize = { 1.f/float(m_staticViewData.imageDimensions.x),1.f/float(m_staticViewData.imageDimensions.y) };
			m_staticViewData.rcpHalfPixelSize = { 0.5f/float(m_staticViewData.imageDimensions.x),0.5f/float(m_staticViewData.imageDimensions.y) };
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
			m_rayCountBuffer.buffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t),&raygenBufferSize);
			m_rayCountBuffer.asRRBuffer = m_rrManager->linkBuffer(m_rayCountBuffer.buffer.get(), CL_MEM_READ_ONLY);

			ocl::COpenCLHandler::ocl.pclEnqueueAcquireGLObjects(m_rrManager->getCLCommandQueue(), 1u, &m_rayCountBuffer.asRRBuffer.second, 0u, nullptr, nullptr);
		}

		// create out screen-sized textures
		m_accumulation = createScreenSizedTexture(EF_R32G32_UINT);
		m_tonemapOutput = createScreenSizedTexture(EF_A2B10G10R10_UNORM_PACK32);

		//
		{
			SPushConstantRange raytracingCommonPCRange{ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaytraceShaderCommonData_t)};
			m_commonRaytracingDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_commonRaytracingDSLayout));

			// i know what I'm doing
			auto globalBackendDataDSLayout = core::smart_refctd_ptr<IGPUDescriptorSetLayout>(const_cast<IGPUDescriptorSetLayout*>(m_globalBackendDataDS->getLayout()));

			// raygen
			{
				m_raygenPipelineLayout = m_driver->createGPUPipelineLayout(
					&raytracingCommonPCRange,&raytracingCommonPCRange+1u,
					core::smart_refctd_ptr(globalBackendDataDSLayout),
					core::smart_refctd_ptr(m_commonRaytracingDSLayout),
					core::smart_refctd_ptr(m_raygenDSLayout),
					nullptr
				);
#ifdef TODO
	{
		std::string glsl = "raygen.comp" +
			globalMeta->materialCompilerGLSL_declarations +
			// TODO ds0 descriptors and user-defined functions required by material compiler
			globalMeta->materialCompilerGLSL_source;
		
		auto shader = m_driver->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str()));
		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE);
		auto spec = m_driver->createGPUSpecializedShader(shader.get(), info);
		m_raygenPipeline = m_driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_raygenLayout), std::move(spec));
	}
#endif
				(std::ofstream("material_declarations.glsl") << initData.globalMeta->materialCompilerGLSL_declarations).close();
				(std::ofstream("material_source.glsl") << initData.globalMeta->materialCompilerGLSL_source).close();
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
				auto scrambleTexture = createScreenSizedTexture(EF_R32_UINT);
				{
					core::vector<uint32_t> random(renderPixelCount);
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
				setImageInfo(resolveInfos+1,asset::EIL_GENERAL,core::smart_refctd_ptr(m_tonemapOutput));
				

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

#ifdef _IRR_BUILD_OPTIX_
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

	if (m_accumulation)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_accumulation.get(),"sceneReferred.exr");
	if (m_tonemapOutput)
		ext::ScreenShot::createScreenShot(m_driver,m_assetManager,m_tonemapOutput.get(),"tonemapped.png");
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

	m_raygenPipelineLayout = nullptr;
	m_resolvePipelineLayout = nullptr;
	m_raygenPipeline = nullptr;
	m_resolvePipeline = nullptr;

	m_perCameraRasterDS = nullptr;

	m_cullWorkGroups = 0u;
	m_cullPushConstants = {core::matrix4SIMD(),0u,0u,1.f,0xdeadbeefu};
	m_cullDS = nullptr;
	m_mdiDrawCalls.clear();
	m_indirectDrawBuffers[1] = m_indirectDrawBuffers[0] = nullptr;

	m_raytraceCommonData = {core::matrix3x4SIMD(),core::matrix3x4SIMD(),0.f,0u,0u,0.f};
	m_staticViewData = {{0.f,0.f,0.f},0u,{0.f,0.f},{0.f,0.f},{0u,0u},0u,0u};
	m_maxRaysPerDispatch = 0u;
	m_sceneBound = core::aabbox3df(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

	m_rrManager->detachInstances(rrInstances.begin(),rrInstances.end());
	m_rrManager->deleteInstances(rrInstances.begin(),rrInstances.end());
	rrInstances.clear();
	m_mock_smgr = {};

	m_rrManager->deleteShapes(rrShapeCache.m_gpuAssociative.begin(), rrShapeCache.m_gpuAssociative.end());
	rrShapeCache = {};
}


void Renderer::render(irr::ITimer* timer)
{
	if (m_cullPushConstants.maxObjectCount==0u)
		return;


	auto camera = m_smgr->getActiveCamera();
	auto prevViewProj = camera->getConcatenatedMatrix();

	camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(timer->getTime()).count());
	camera->render();

	const auto currentViewProj = camera->getConcatenatedMatrix();
	// TODO: instead of rasterizing vis-buffer only once, subpixel jitter it to obtain AA
	if (!core::equals(prevViewProj,currentViewProj,core::ROUNDING_ERROR<core::matrix4SIMD>()*1000.0))
	{
		m_raytraceCommonData.framesDispatched = 0u;

		IGPUDescriptorSet* descriptors[3] = { nullptr };
		descriptors[1] = m_cullDS.get();

		m_driver->bindComputePipeline(m_cullPipeline.get());
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_cullPipelineLayout.get(),1u,1u,descriptors+1u,nullptr);
		{
			m_cullPushConstants.viewProjMatrix = currentViewProj;
			m_cullPushConstants.viewProjDeterminant = core::determinant(currentViewProj);
			m_driver->pushConstants(m_cullPipelineLayout.get(),ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t),&m_cullPushConstants);
		}
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
		m_driver->bindDescriptorSets(EPBP_GRAPHICS,m_visibilityBufferFillPipelineLayoutGPU.get(),0u,2u,descriptors,nullptr);
		for (auto call : m_mdiDrawCalls)
		{
			m_driver->bindGraphicsPipeline(call.pipeline.get());
			m_driver->drawIndexedIndirect(call.vertexBindings,EPT_TRIANGLE_LIST,EIT_32BIT,call.indexBuffer.get(),m_indirectDrawBuffers[m_cullPushConstants.currentCommandBufferIx].get(),call.mdiOffset,call.mdiCount,sizeof(DrawElementsIndirectCommand_t));
		}

		// flip MDI buffers
		m_cullPushConstants.currentCommandBufferIx ^= 0x01u;
	}

	// generate rays
	{
		const auto cameraPosition = core::vectorSIMDf().set(camera->getAbsolutePosition());
		{
			auto frustum = camera->getViewFrustum();
			m_raytraceCommonData.frustumCorners.rows[1] = frustum->getFarLeftDown();
			m_raytraceCommonData.frustumCorners.rows[0] = frustum->getFarRightDown()-m_raytraceCommonData.frustumCorners.rows[1];
			m_raytraceCommonData.frustumCorners.rows[1] -= cameraPosition;
			m_raytraceCommonData.frustumCorners.rows[3] = frustum->getFarLeftUp();
			m_raytraceCommonData.frustumCorners.rows[2] = frustum->getFarRightUp()-m_raytraceCommonData.frustumCorners.rows[3];
			m_raytraceCommonData.frustumCorners.rows[3] -= cameraPosition;
		}
		camera->getViewMatrix().getSub3x3InverseTranspose(m_raytraceCommonData.normalMatrixAndCameraPos);
		m_raytraceCommonData.normalMatrixAndCameraPos.setTranslation(cameraPosition);
		{
			auto projMat = camera->getProjectionMatrix();
			auto* row = projMat.rows;
			m_raytraceCommonData.depthLinearizationConstant = -row[3][2]/(row[3][2]-row[2][2]);
		}
		m_raytraceCommonData.framesDispatched++;
		m_raytraceCommonData.rcpFramesDispatched = 1.f/float(m_raytraceCommonData.framesDispatched);

		IGPUDescriptorSet* descriptorSets[] = {m_globalBackendDataDS.get(),m_commonRaytracingDS.get(),m_raygenDS.get()};
		m_driver->bindDescriptorSets(EPBP_COMPUTE, m_raygenPipelineLayout.get(), 0, 3, descriptorSets, nullptr);
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

	if (false) // TODO
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

		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT
#ifndef _IRR_BUILD_OPTIX_
			|GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT
#else
			|(m_denoisedBuffer.getObject() ? (GL_PIXEL_BUFFER_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT):(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT))
#endif
		);
	}

	m_raytraceCommonData.samplesComputedPerPixel += m_staticViewData.samplesPerPixelPerDispatch;

	// TODO: tonemap properly
#ifdef _IRR_BUILD_OPTIX_
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

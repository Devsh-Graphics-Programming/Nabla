#include <numeric>

#include "ExtraCrap.h"

#include "irr/ext/ScreenShot/ScreenShot.h"


#ifndef _IRR_BUILD_OPTIX_
	#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#endif

using namespace irr;
using namespace irr::asset;
using namespace irr::video;


constexpr uint32_t kOptiXPixelSize = sizeof(uint16_t)*3u;


Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, scene::ISceneManager* _smgr, core::smart_refctd_ptr<IGPUDescriptorSet>&& globalBackendDataDS, bool useDenoiser) :
		m_useDenoiser(useDenoiser),	m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager), m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
		m_sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), /*m_renderSize{0u,0u}, */m_rightHanded(false),
		m_globalBackendDataDS(std::move(globalBackendDataDS)), // TODO: review this member
		rrShapeCache(),
#if TODO
		m_raygenWorkGroups{0u,0u}, m_resolveWorkGroups{0u,0u},
		m_rayBufferAsRR(nullptr,nullptr), m_intersectionBufferAsRR(nullptr,nullptr), m_rayCountBufferAsRR(nullptr,nullptr),
		rrInstances(),
#endif
		m_indirectDrawBuffers{nullptr}, m_cullPushConstants{core::matrix4SIMD(),0u,0u,1.f,0xdeadbeefu}, m_lightCount(0u),
		m_visibilityBuffer(nullptr), tmpTonemapBuffer(nullptr), m_colorBuffer(nullptr),
		m_visibilityBufferAttachments{nullptr}, m_maxSamples(0u), m_samplesPerPixelPerDispatch(0u), m_rayCountPerDispatch(0u), m_framesDone(0u), m_samplesComputedPerPixel(0u)
	#ifdef _IRR_BUILD_OPTIX_
		,m_cudaStream(nullptr)
	#endif
{
	auto specializedShaderFromFile = [&](const char* path) -> core::smart_refctd_ptr<ICPUSpecializedShader>
	{
		auto bundle = m_assetManager->getAsset(path, {});
		return core::move_and_static_cast<ICPUSpecializedShader>(*bundle.getContents().begin());
	};
	m_visibilityBufferFillShaders[0] = specializedShaderFromFile("../fillVisBuffer.vert");
	m_visibilityBufferFillShaders[1] = specializedShaderFromFile("../fillVisBuffer.frag");

	// TODO: make these util functions for every descriptor type (maybe template with specs?) in `IDescriptorSetLayout` -> Assign: @Hazardu
	auto fillSSBODescriptorBindingDeclarations = [](auto* outBindings, ISpecializedShader::E_SHADER_STAGE accessFlags, uint32_t count, uint32_t startIndex=0u) -> void
	{
		for (auto i=0u; i<count; i++)
		{
			outBindings[i].binding = i+startIndex;
			outBindings[i].type = asset::EDT_STORAGE_BUFFER;
			outBindings[i].count = 1u;
			outBindings[i].stageFlags = accessFlags;
			outBindings[i].samplers = nullptr;
		}
	};

	constexpr auto cullingOutputDescriptorCount = 2u;
	{
		ICPUDescriptorSetLayout::SBinding bindings[cullingOutputDescriptorCount];
		fillSSBODescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_VERTEX,cullingOutputDescriptorCount);

		auto dsLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(bindings,bindings+2u);
		m_visibilityBufferFillPipelineLayoutCPU = core::make_smart_refctd_ptr<ICPUPipelineLayout>(nullptr,nullptr,nullptr,core::smart_refctd_ptr(dsLayout),nullptr,nullptr);

		// TODO: @Crisspl find a way to stop the user from such insanity as moving from the bundle's dynamic array
		//m_visibilityBufferFillPipelineLayoutGPU = std::move(m_driver->getGPUObjectsFromAssets<ICPUPipelineLayout>(&m_visibilityBufferFillPipelineLayoutCPU,&m_visibilityBufferFillPipelineLayoutCPU+1u)->operator[](0));
		m_visibilityBufferFillPipelineLayoutGPU = core::smart_refctd_ptr(m_driver->getGPUObjectsFromAssets<ICPUPipelineLayout>(&m_visibilityBufferFillPipelineLayoutCPU,&m_visibilityBufferFillPipelineLayoutCPU+1u)->operator[](0));
		m_perCameraRasterDSLayout = core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(m_visibilityBufferFillPipelineLayoutGPU->getDescriptorSetLayout(1u));
	}


	auto gpuSpecializedShaderFromFile = [&](const char* path) -> core::smart_refctd_ptr<IGPUSpecializedShader>
	{
		auto shader = specializedShaderFromFile(path);
		// TODO: @Crisspl find a way to stop the user from such insanity as moving from the bundle's dynamic array
		//return std::move(m_driver->getGPUObjectsFromAssets<ICPUSpecializedShader>(&shader,&shader+1u)->operator[](0));
		return m_driver->getGPUObjectsFromAssets<ICPUSpecializedShader>(&shader,&shader+1u)->operator[](0);
	};

	{
		constexpr auto cullingDescriptorCount = cullingOutputDescriptorCount+2u;
		IGPUDescriptorSetLayout::SBinding bindings[cullingDescriptorCount];
		fillSSBODescriptorBindingDeclarations(bindings,ISpecializedShader::ESS_COMPUTE,cullingDescriptorCount);
		bindings[3u].count = 2u;
		m_cullDSLayout = m_driver->createGPUDescriptorSetLayout(bindings,bindings+4u);
		SPushConstantRange range{ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t)};
		m_cullPipelineLayout = m_driver->createGPUPipelineLayout(&range,&range+1u,nullptr,core::smart_refctd_ptr(m_cullDSLayout),nullptr,nullptr);
		m_cullPipeline = m_driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(m_cullPipelineLayout),gpuSpecializedShaderFromFile("../cull.comp"));
	}

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
}


#if TODO
core::smart_refctd_ptr<IGPUDescriptorSetLayout> Renderer::createDS2layoutCompost(bool useDenoiser, core::smart_refctd_ptr<IGPUSampler>& nearestSampler)
{
	constexpr uint32_t MAX_COUNT = 10u;
	IGPUDescriptorSetLayout::SBinding bindings[MAX_COUNT];

	auto makeBinding = [](uint32_t bnd, uint32_t cnt, ISpecializedShader::E_SHADER_STAGE stage, E_DESCRIPTOR_TYPE dt, core::smart_refctd_ptr<IGPUSampler>* samplers)
	{
		IGPUDescriptorSetLayout::SBinding b;
		b.binding = bnd;
		b.count = cnt;
		b.samplers = samplers;
		b.stageFlags = stage;
		b.type = dt;

		return b;
	};

	const auto stage = ISpecializedShader::ESS_COMPUTE;

	bindings[0] = makeBinding(0u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler); // lightIndex
	bindings[1] = makeBinding(1u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler); // albedo, TODO delete
	bindings[2] = makeBinding(2u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler); // normal
	bindings[3] = makeBinding(3u, 1u, stage, EDT_STORAGE_IMAGE, nullptr); // framebuffer
	bindings[4] = makeBinding(4u, 1u, stage, EDT_STORAGE_BUFFER, nullptr); // rays ssbo
	bindings[5] = makeBinding(5u, 1u, stage, EDT_STORAGE_BUFFER, nullptr); // queries ssbo
	bindings[6] = makeBinding(6u, 1u, stage, EDT_STORAGE_BUFFER, nullptr); // light radiances
	if (useDenoiser)
	{
		bindings[7] = makeBinding(7u, 1u, stage, EDT_STORAGE_BUFFER, nullptr); // color output ssbo
		bindings[8] = makeBinding(8u, 1u, stage, EDT_STORAGE_BUFFER, nullptr); // albedo output ssbo
		bindings[9] = makeBinding(9u, 1u, stage, EDT_STORAGE_BUFFER, nullptr); // normal ouput ssbo
	}

	const uint32_t realCount = useDenoiser ? MAX_COUNT : (MAX_COUNT - 3u);

	return m_driver->createGPUDescriptorSetLayout(bindings, bindings + realCount);
}

core::smart_refctd_ptr<IGPUDescriptorSet> Renderer::createDS2Compost(bool useDenoiser, core::smart_refctd_ptr<IGPUSampler>& nearestSampler)
{
	constexpr uint32_t MAX_COUNT = 10u;
	constexpr uint32_t DENOISER_COUNT = 3u;
	constexpr uint32_t COUNT_WO_DENOISER = MAX_COUNT - DENOISER_COUNT;

	auto layout = createDS2layoutCompost(useDenoiser, nearestSampler);

#ifdef _IRR_BUILD_OPTIX
	auto resolveBuffer = core::smart_refctd_ptr<IDescriptor>(m_denoiserInputBuffer.getObject());
	uint32_t denoiserOffsets[DENOISER_COUNT]{ m_denoiserInputs[EDI_COLOR].data,m_denoiserInputs[EDI_ALBEDO].data,m_denoiserInputs[EDI_NORMAL].data };
	uint32_t denoiserSizes[DENOISER_COUNT]{ getDenoiserBufferSize(m_denoiserInputs[EDI_COLOR])
								,getDenoiserBufferSize(m_denoiserInputs[EDI_ALBEDO])
								,getDenoiserBufferSize(m_denoiserInputs[EDI_NORMAL]) };
#endif

	core::smart_refctd_ptr<IDescriptor> descriptors[MAX_COUNT]{
		m_lightIndex,
		m_albedo,
		m_visibilityBufferAttachments[EVBA_NORMALS],
		m_accumulation,
		m_rayBuffer,
		m_intersectionBuffer
#ifdef _IRR_BUILD_OPTIX
		,
		resolveBuffer,
		resolveBuffer,
		resolveBuffer
#endif
	};

	const uint32_t count = useDenoiser ? MAX_COUNT : COUNT_WO_DENOISER;
	IGPUDescriptorSet::SWriteDescriptorSet write[MAX_COUNT];
	write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[1].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[3].descriptorType = EDT_STORAGE_IMAGE;
	for (uint32_t i = 4u; i < MAX_COUNT; ++i)
		write[i].descriptorType = EDT_STORAGE_BUFFER;
	IGPUDescriptorSet::SDescriptorInfo info[MAX_COUNT];

	auto ds = m_driver->createGPUDescriptorSet(std::move(layout));

	for (uint32_t i = 0u; i < count; ++i)
	{
		auto& w = write[i];
		w.arrayElement = 0u;
		w.binding = i;
		w.count = 1u;
		w.dstSet = ds.get();
		w.info = info + i;

		info[i].desc = std::move(descriptors[i]);
		if (w.descriptorType == EDT_STORAGE_BUFFER)
		{
			auto* buf = static_cast<IGPUBuffer*>(info[i].desc.get());
#ifdef _IRR_BUILD_OPTIX
			info[i].buffer.offset = (i >= COUNT_WO_DENOISER) ? denoiserOffsets[i-COUNT_WO_DENOISER] : 0u;
#else
			info[i].buffer.offset = 0u;
#endif
#ifdef _IRR_BUILD_OPTIX
			info[i].buffer.size = (i >= COUNT_WO_DENOISER) ? denoiserSizes[i-COUNT_WO_DENOISER] : buf->getSize();
#else
			info[i].buffer.size = buf->getSize();
#endif
		}
		else
		{
			info[i].image.imageLayout = EIL_UNDEFINED;
			info[i].image.sampler = nullptr; // using immutable samplers
		}
	}

	m_driver->updateDescriptorSets(count, write, 0u, nullptr);

	return ds;
}

core::smart_refctd_ptr<IGPUPipelineLayout> Renderer::createLayoutCompost()
{
	auto& ds2 = m_compostDS2;
	auto* layout2 = const_cast<IGPUDescriptorSetLayout*>(ds2->getLayout());

	//TODO push constants
	return m_driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>(layout2), nullptr);
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> Renderer::createDS2layoutRaygen(core::smart_refctd_ptr<IGPUSampler>& nearestSampler)
{
	constexpr uint32_t count = 6u;

	auto makeBinding = [](uint32_t bnd, uint32_t cnt, ISpecializedShader::E_SHADER_STAGE stage, E_DESCRIPTOR_TYPE dt, core::smart_refctd_ptr<IGPUSampler>* samplers)
	{
		IGPUDescriptorSetLayout::SBinding b;
		b.binding = bnd;
		b.count = cnt;
		b.samplers = samplers;
		b.stageFlags = stage;
		b.type = dt;

		return b;
	};

	const auto stage = ISpecializedShader::ESS_COMPUTE;
	IGPUDescriptorSetLayout::SBinding bindings[count];
	bindings[0] = makeBinding(0u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler);
	bindings[1] = makeBinding(1u, 1u, stage, EDT_UNIFORM_TEXEL_BUFFER, nullptr);
	bindings[2] = makeBinding(2u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler);
	bindings[3] = makeBinding(3u, 1u, stage, EDT_STORAGE_BUFFER, nullptr);
	bindings[4] = makeBinding(4u, 1u, stage, EDT_STORAGE_BUFFER, nullptr);
	bindings[5] = makeBinding(5u, 1u, stage, EDT_STORAGE_BUFFER, nullptr);

	return m_driver->createGPUDescriptorSetLayout(bindings, bindings + count);
}

core::smart_refctd_ptr<IGPUDescriptorSet> Renderer::createDS2Raygen(core::smart_refctd_ptr<IGPUSampler>& nearstSampler)
{
	constexpr uint32_t count = 6u;
	core::smart_refctd_ptr<IDescriptor> descriptors[count]
	{
		m_visibilityBufferAttachments[EVBA_DEPTH],
		m_sampleSequence,
		m_scrambleTexture,
		m_rayBuffer,
		m_lightCDFBuffer,
		m_lightBuffer
	};

	auto layout = createDS2layoutRaygen(nearstSampler);

	auto ds = m_driver->createGPUDescriptorSet(std::move(layout));

	IGPUDescriptorSet::SWriteDescriptorSet write[count];
	write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
	write[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[3].descriptorType = EDT_STORAGE_BUFFER;
	write[4].descriptorType = EDT_STORAGE_BUFFER;
	write[5].descriptorType = EDT_STORAGE_BUFFER;
	IGPUDescriptorSet::SDescriptorInfo info[count];

	for (uint32_t i = 0u; i < count; ++i)
	{
		auto& w = write[i];
		w.arrayElement = 0u;
		w.binding = i;
		w.count = 1u;
		w.dstSet = ds.get();
		w.info = info + i;

		info[i].desc = std::move(descriptors[i]);
		if (w.descriptorType == EDT_STORAGE_BUFFER)
		{
			info[i].buffer.offset = 0u;
			auto* buf = static_cast<IGPUBuffer*>(info[i].desc.get());
			info[i].buffer.size = buf->getSize();
		}
		else
		{
			info[i].image.imageLayout = EIL_UNDEFINED;
			info[i].image.sampler = nullptr; // immutable samplers
		}
	}

	m_driver->updateDescriptorSets(count, write, 0u, nullptr);

	return ds;
}

core::smart_refctd_ptr<IGPUPipelineLayout> Renderer::createLayoutRaygen()
{
	//ds from mitsuba loader (VT stuff, material compiler data, instance data)
	//will be needed for gbuffer pass as well (at least to spit out instruction offsets/counts and maybe offset into instance data buffer)
	auto& ds0 = m_globalBackendDataDS;
	auto* layout0 = const_cast<IGPUDescriptorSetLayout*>(ds0->getLayout());

	auto& ds2 = m_raygenDS2;
	auto* layout2 = const_cast<IGPUDescriptorSetLayout*>(ds2->getLayout());

	//TODO push constants
	return m_driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>(layout0), nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>(layout2), nullptr);
}
#endif

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

	core::vector<std::pair<core::smart_refctd_ptr<IGPUMeshBuffer>,core::smart_refctd_ptr<ICPUMesh>>> gpuMeshBuffersAndMeta;
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
			for (auto& cpumesh_ : contents)
			{
				auto cpumesh = static_cast<asset::ICPUMesh*>(cpumesh_.get());
				const auto& instances = getInstances(cpumesh);

				auto meshBufferCount = cpumesh->getMeshBufferCount();
				for (auto i = 0u; i < meshBufferCount; i++)
				{
					// TODO: get rid of `getMeshBuffer` and `getMeshBufferCount`, just return a range as `getMeshBuffers`
					auto cpumb = cpumesh->getMeshBuffer(i);

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
					meshBuffersToProcess.push_back(cpumb);
					gpuMeshBuffersAndMeta.emplace_back(nullptr,core::smart_refctd_ptr<ICPUMesh>(cpumesh));
				}

				// set up lights
				for (auto instance : instances)
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

		// set up BVH
		IMeshManipulator::homogenizePrimitiveTypeAndIndices(meshBuffersToProcess.begin(), meshBuffersToProcess.end(), EPT_TRIANGLE_LIST);
		m_rrManager->makeRRShapes(rrShapeCache, meshBuffersToProcess.begin(), meshBuffersToProcess.end());

		// convert to GPU objects (and sort so they're ordered by pipeline)
		auto gpuObjs = m_driver->getGPUObjectsFromAssets(meshBuffersToProcess.data(),meshBuffersToProcess.data()+meshBuffersToProcess.size());
		for (auto i=0u; i<gpuObjs->size(); i++)
			gpuMeshBuffersAndMeta[i].first = core::smart_refctd_ptr(gpuObjs->operator[](i));
		std::sort(gpuMeshBuffersAndMeta.begin(), gpuMeshBuffersAndMeta.end(), [](const auto& lhs,const auto& rhs)->bool{return lhs.first->getPipeline()<rhs.first->getPipeline();});
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
		initNewMDI(gpuMeshBuffersAndMeta.front().first);
		call.mdiOffset = 0u;
		call.mdiCount = 0u;
		auto queueUpMDI = [&](const MDICall& call) -> void
		{
			m_mdiDrawCalls.emplace_back(call);
		};
		for (auto gpuAndMeta : gpuMeshBuffersAndMeta)
		{
			auto gpumb = gpuAndMeta.first;

			const uint32_t baseInstance = objectStaticData.size();
			gpumb->setBaseInstance(baseInstance);

			const uint32_t drawID = mdiData.size();
			const auto aabb = gpumb->getBoundingBox();
			const auto& instances = getInstances(gpuAndMeta.second);
			for (auto j=0u; j<gpumb->getInstanceCount(); j++)
			{
				core::matrix3x4SIMD worldMatrix,normalMatrix;
				worldMatrix = instances[j].tform;
				worldMatrix.getSub3x3InverseTranspose(normalMatrix);

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


	m_cullPushConstants.maxObjectCount = objectStaticData.size();
	m_cullPushConstants.currentCommandBufferIx = 0x0u;

	m_cullDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_cullDSLayout));
	m_perCameraRasterDS = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr(m_perCameraRasterDSLayout));
	{
		const uint32_t instanceCount = objectStaticData.size();

		IGPUDescriptorSet::SDescriptorInfo infos[5];

		infos[0].buffer.size = instanceCount*sizeof(ObjectStaticData_t);
		infos[0].buffer.offset = 0u;
		infos[0].desc = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(infos[0].buffer.size,objectStaticData.data());
		objectStaticData.clear();
		
		infos[1].buffer.size = instanceCount*sizeof(DrawData_t);
		infos[1].buffer.offset = 0u;
		infos[1].desc = m_driver->createDeviceLocalGPUBufferOnDedMem(infos[1].buffer.size);
		
		infos[2].buffer.size = instanceCount*sizeof(CullData_t);
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
	baseEnvColor.set(0.f,0.f,0.f,1.f);
	for (auto emitter : initData.globalMeta->emitters)
	{
		float weight = 0.f;
		switch (emitter.type)
		{
			case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
				baseEnvColor += emitter.constant.radiance;
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
}

void Renderer::finalizeScene(Renderer::InitializationData& initData)
{
	if (initData.lights.empty())
		return;
	m_lightCount = initData.lights.size();

	const double weightSum = std::accumulate(initData.lightPDF.begin(),initData.lightPDF.end(),0.0);
	assert(weightSum>FLT_MIN);

	constexpr double UINT_MAX_DOUBLE = double(0x1ull<<32ull);
	const double weightSumRcp = UINT_MAX_DOUBLE/weightSum;

	auto outCDF = initData.lightCDF.begin();

	auto inPDF = initData.lightPDF.begin();
	double partialSum = *inPDF;

	auto radianceIn = initData.lightRadiances.begin();
	core::vector<uint64_t> compressedRadiance(m_lightCount,0ull);
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

	m_lightCDFBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(initData.lightCDF.size()*sizeof(uint32_t),initData.lightCDF.data());
	m_lightBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(initData.lights.size()*sizeof(SLight),initData.lights.data());
	m_lightRadianceBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(initData.lightRadiances.size()*sizeof(uint64_t),initData.lightRadiances.data());
}

core::smart_refctd_ptr<IGPUImageView> Renderer::createScreenSizedTexture(E_FORMAT format)
{
	IGPUImage::SCreationParams imgparams;
	imgparams.extent = { m_renderSize[0], m_renderSize[1], 1u };
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

void Renderer::init(const SAssetBundle& meshes,
					bool isCameraRightHanded,
					core::smart_refctd_ptr<ICPUBuffer>&& sampleSequence,
					uint32_t rayBufferSize)
{
	deinit();

	m_rightHanded = isCameraRightHanded;

	//! set up GPU sampler 
	{
		m_maxSamples = sampleSequence->getSize()/(sizeof(uint32_t)*MaxDimensions);
		assert(m_maxSamples==MAX_ACCUMULATED_SAMPLES);

		// upload sequence to GPU
		auto gpubuf = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
		m_sampleSequence = m_driver->createGPUBufferView(gpubuf.get(), asset::EF_R32G32B32_UINT);
	}

	// initialize scene
	{
		auto initData = initSceneObjects(meshes);

		initSceneNonAreaLights(initData);
		finalizeScene(initData);
		{
	#if TODO
			auto gpumeshes = m_driver->getGPUObjectsFromAssets<ICPUMesh>(contents.first, contents.second);
			auto cpuit = contents.first;
			for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
			{
				auto* meta = cpuit->get()->getMetadata();

				assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
				const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);

				const auto& instances = meshmeta->getInstances();

				const auto& gpumesh = *gpuit;
				for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
					gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = nonInstanced;

				for (auto instance : instances)
				{
					auto node = core::smart_refctd_ptr<IMeshSceneNode>(m_smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh)));
					node->setRelativeTransformationMatrix(instance.tform.getAsRetardedIrrlichtMatrix());
					node->updateAbsolutePosition();
					m_sceneBound.addInternalBox(node->getTransformedBoundingBox());

					nodes.push_back(std::move(node));
				}
			}
			core::vector<int32_t> ids(nodes.size());
			std::iota(ids.begin(), ids.end(), 0);
			auto nodesBegin = &nodes.data()->get();
			m_rrManager->makeRRInstances(rrInstances, rrShapeCache, m_assetManager, nodesBegin, nodesBegin+nodes.size(), ids.data());
			m_rrManager->attachInstances(rrInstances.begin(), rrInstances.end());
	#endif
		}

		// figure out the renderable size
		m_renderSize[0] = m_driver->getScreenSize().Width;
		m_renderSize[1] = m_driver->getScreenSize().Height;
		const auto& sensors = initData.globalMeta->sensors;
		if (sensors.size())
		{
			// just grab the first sensor
			const auto& sensor = sensors.front();
			const auto& film = sensor.film;
			assert(film.cropOffsetX == 0);
			assert(film.cropOffsetY == 0);
			m_renderSize[0] = film.cropWidth;
			m_renderSize[1] = film.cropHeight;
		}
	}
	const auto renderPixelCount = m_renderSize[0]*m_renderSize[1];

	const auto raygenBufferSize = static_cast<size_t>(renderPixelCount)*sizeof(::RadeonRays::ray);
	assert(raygenBufferSize<=rayBufferSize);
	const auto shadowBufferSize = static_cast<size_t>(renderPixelCount)*sizeof(int32_t);
	assert(shadowBufferSize<=rayBufferSize);
	m_samplesPerPixelPerDispatch = rayBufferSize/(raygenBufferSize+shadowBufferSize);
	assert(m_samplesPerPixelPerDispatch >= 1u);
	printf("Using %d samples\n", m_samplesPerPixelPerDispatch);
	
	m_rayCountPerDispatch = m_samplesPerPixelPerDispatch*renderPixelCount;
	// create scramble texture
	m_scrambleTexture = createScreenSizedTexture(EF_R32_UINT);
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
		region.imageExtent = {m_renderSize[0],m_renderSize[1],1u};
		m_driver->copyBufferToImage(gpuBuff.get(), m_scrambleTexture->getCreationParameters().image.get(), 1u, &region);
	}


	core::smart_refctd_ptr<IGPUSampler> smplr_nearest;
	{
		IGPUSampler::SParams params;
		params.AnisotropicFilter = 0;
		params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
		params.CompareEnable = 0;
		params.CompareFunc = ISampler::ECO_ALWAYS;
		params.LodBias = 0;
		params.MaxFilter = ISampler::ETF_NEAREST;
		params.MinFilter = ISampler::ETF_NEAREST;
		params.MaxLod = 10000;
		params.MinLod = 0;
		params.MipmapMode = ISampler::ESMM_NEAREST;
		smplr_nearest = m_driver->createGPUSampler(params);
	}


#ifdef TODO
	m_raygenDS2 = createDS2Raygen(smplr_nearest);
	m_compostDS2 = createDS2Compost(m_useDenoiser, smplr_nearest);
	m_raygenLayout = createLayoutRaygen();
	m_compostLayout = createLayoutCompost();

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
	{
		auto glsl = 
			std::string(m_useDenoiser ? "#version 430 core\n#define USE_OPTIX_DENOISER\n" : "#version 430 core\n") +
			//"irr/builtin/glsl/ext/RadeonRays/"
			rr_includes->getBuiltinInclude("ray.glsl") +
			lightStruct +
			compostShader;

		auto shader = m_driver->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(glsl.c_str()));
		asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE);
		auto spec = m_driver->createGPUSpecializedShader(shader.get(), info);
		m_compostPipeline = m_driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(m_compostLayout), std::move(spec));
	}

	//
	constexpr auto RAYGEN_WORK_GROUP_DIM = 16u;
	m_raygenWorkGroups[0] = (renderSize[0]+RAYGEN_WORK_GROUP_DIM-1)/RAYGEN_WORK_GROUP_DIM;
	m_raygenWorkGroups[1] = (renderSize[1]+RAYGEN_WORK_GROUP_DIM-1)/RAYGEN_WORK_GROUP_DIM;
	constexpr auto RESOLVE_WORK_GROUP_DIM = 32u;
	m_resolveWorkGroups[0] = (renderSize[0]+RESOLVE_WORK_GROUP_DIM-1)/RESOLVE_WORK_GROUP_DIM;
	m_resolveWorkGroups[1] = (renderSize[1]+RESOLVE_WORK_GROUP_DIM-1)/RESOLVE_WORK_GROUP_DIM;

	raygenBufferSize *= m_samplesPerPixelPerDispatch;
	m_rayBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(raygenBufferSize);

	shadowBufferSize *= m_samplesPerPixelPerDispatch;
	m_intersectionBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(shadowBufferSize);

	m_rayCountBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t),&m_rayCountPerDispatch);

	m_rayBufferAsRR = m_rrManager->linkBuffer(m_rayBuffer.get(), CL_MEM_READ_WRITE);
	// TODO: clear hit buffer to -1 before usage
	m_intersectionBufferAsRR = m_rrManager->linkBuffer(m_intersectionBuffer.get(), CL_MEM_READ_WRITE);
	m_rayCountBufferAsRR = m_rrManager->linkBuffer(m_rayCountBuffer.get(), CL_MEM_READ_ONLY);

	const cl_mem clObjects[] = { m_rayCountBufferAsRR.second };
	auto objCount = sizeof(clObjects)/sizeof(cl_mem);
	clEnqueueAcquireGLObjects(m_rrManager->getCLCommandQueue(), objCount, clObjects, 0u, nullptr, nullptr);
#endif

#ifdef _IRR_BUILD_OPTIX_
	while (m_denoiser)
	{
		m_denoiser->computeMemoryResources(&m_denoiserMemReqs,renderSize);

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

		auto setUpOptiXImage2D = [&m_renderSize](OptixImage2D& img, uint32_t pixelSize) -> void
		{
			img = {};
			img.width = m_renderSize[0];
			img.height = m_renderSize[1];
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

		break;
	}
#endif
	
	m_visibilityBufferAttachments[EVBA_DEPTH] = createScreenSizedTexture(EF_D32_SFLOAT);
	m_visibilityBufferAttachments[EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING] = createScreenSizedTexture(EF_R32G32_UINT);
	m_visibilityBufferAttachments[EVBA_NORMALS] = createScreenSizedTexture(EF_R16G16_SNORM);
	m_visibilityBufferAttachments[EVBA_UV_COORDINATES] = createScreenSizedTexture(EF_R16G16_SFLOAT);
	m_visibilityBuffer = m_driver->addFrameBuffer();
	m_visibilityBuffer->attach(EFAP_DEPTH_ATTACHMENT, core::smart_refctd_ptr(m_visibilityBufferAttachments[EVBA_DEPTH]));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_visibilityBufferAttachments[EVBA_OBJECTID_AND_TRIANGLEID_AND_FRONTFACING]));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT1, core::smart_refctd_ptr(m_visibilityBufferAttachments[EVBA_NORMALS]));
	m_visibilityBuffer->attach(EFAP_COLOR_ATTACHMENT2, core::smart_refctd_ptr(m_visibilityBufferAttachments[EVBA_UV_COORDINATES]));

	m_accumulation = createScreenSizedTexture(EF_R32G32B32A32_SFLOAT);
	tmpTonemapBuffer = m_driver->addFrameBuffer();
	tmpTonemapBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_accumulation));

	//m_tonemapOutput = createScreenSizedTexture(m_denoiserScratchBuffer.getObject() ? EF_A2B10G10R10_UNORM_PACK32:EF_R8G8B8_SRGB);
	m_tonemapOutput = createScreenSizedTexture(EF_R8G8B8_SRGB);
	//m_tonemapOutput = createScreenSizedTexture(EF_A2B10G10R10_UNORM_PACK32);
	m_colorBuffer = m_driver->addFrameBuffer();
	m_colorBuffer->attach(EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(m_tonemapOutput));
}


void Renderer::deinit()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	clFinish(commandQueue);

	glFinish();

#if TODO
	// create a screenshot (TODO: create OpenEXR @Anastazluk)
	if (m_tonemapOutput)
		ext::ScreenShot::dirtyCPUStallingScreenshot(m_driver, m_assetManager, "screenshot.png", m_tonemapOutput.get(),0u,true,asset::EF_R8G8B8_SRGB);

	// release OpenCL objects and wait for OpenCL to finish
	const cl_mem clObjects[] = { m_rayCountBufferAsRR.second };
	auto objCount = sizeof(clObjects) / sizeof(cl_mem);
	clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, nullptr, nullptr);
	clFlush(commandQueue);
	clFinish(commandQueue);

	if (m_rayBufferAsRR.first)
	{
		m_rrManager->deleteRRBuffer(m_rayBufferAsRR.first);
		m_rayBufferAsRR = {nullptr,nullptr};
	}
	if (m_intersectionBufferAsRR.first)
	{
		m_rrManager->deleteRRBuffer(m_intersectionBufferAsRR.first);
		m_intersectionBufferAsRR = {nullptr,nullptr};
	}
	if (m_rayCountBufferAsRR.first)
	{
		m_rrManager->deleteRRBuffer(m_rayCountBufferAsRR.first);
		m_rayCountBufferAsRR = {nullptr,nullptr};
	}
	m_rayBuffer = m_intersectionBuffer = m_rayCountBuffer = nullptr;
	m_raygenWorkGroups[0] = m_raygenWorkGroups[1] = 0u;
	m_resolveWorkGroups[0] = m_resolveWorkGroups[1] = 0u;

	m_rrManager->detachInstances(rrInstances.begin(),rrInstances.end());
	m_rrManager->deleteInstances(rrInstances.begin(),rrInstances.end());
	rrInstances.clear();

	m_lightCDFBuffer = m_lightBuffer = m_lightRadianceBuffer = nullptr;
#endif


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

	m_maxSamples = m_samplesPerPixelPerDispatch = m_rayCountPerDispatch = 0u;
	m_framesDone = m_samplesComputedPerPixel = 0u;
	m_sampleSequence = nullptr;
	m_scrambleTexture = nullptr;

	for (uint32_t i = 0u; i < EVBA_COUNT; i++)
		m_visibilityBufferAttachments[i] = nullptr;


	m_mdiDrawCalls.clear();
	m_indirectDrawBuffers[1] = m_indirectDrawBuffers[0] = nullptr;
	m_cullPushConstants.maxObjectCount = 0u;

	m_lightCount = 0u;


	m_rrManager->deleteShapes(rrShapeCache.begin(), rrShapeCache.end());
	rrShapeCache.clear();

	m_perCameraRasterDS = nullptr;

	m_globalBackendDataDS = nullptr;
	m_rightHanded = false;
	m_renderSize[0u] = m_renderSize[1u] = 0u;
	m_sceneBound = core::aabbox3df(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
	baseEnvColor.set(0.f,0.f,0.f,1.f);

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
	if (!core::equals(prevViewProj,currentViewProj,core::ROUNDING_ERROR<core::matrix4SIMD>()*1000.0))
	{
		m_framesDone = 0u;

		IGPUDescriptorSet* descriptors[3] = { nullptr };
		descriptors[1] = m_cullDS.get();

		m_driver->bindComputePipeline(m_cullPipeline.get());
		m_driver->bindDescriptorSets(EPBP_COMPUTE,m_cullPipelineLayout.get(),1u,1u,descriptors+1u,nullptr);
		{
			m_cullPushConstants.viewProjMatrix = currentViewProj;
			m_cullPushConstants.viewProjDeterminant = core::determinant(currentViewProj);
			m_driver->pushConstants(m_cullPipelineLayout.get(),ISpecializedShader::ESS_COMPUTE,0u,sizeof(CullShaderData_t),&m_cullPushConstants);
		}
		m_driver->dispatch((m_cullPushConstants.maxObjectCount-1u)/WORKGROUP_SIZE+1u, 1u, 1u);
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


	uint32_t uImageWidth_ImageArea_TotalImageSamples_Samples[4] = { m_renderSize[0],m_renderSize[0]*m_renderSize[1],m_renderSize[0]*m_renderSize[1]*m_samplesPerPixelPerDispatch,m_samplesPerPixelPerDispatch};

	// generate rays
	{
#if TODO
		// TODO set push constants (i've commented out uniforms setting below)
		/*{
			auto camPos = core::vectorSIMDf().set(camera->getAbsolutePosition());
			COpenGLExtensionHandler::pGlProgramUniform3fv(m_raygenProgram, 0, 1, camPos.pointer);

			float uDepthLinearizationConstant;
			{
				auto projMat = camera->getProjectionMatrix();
				auto* row = projMat.rows;
				uDepthLinearizationConstant = -row[3][2]/(row[3][2]-row[2][2]);
			}
			COpenGLExtensionHandler::pGlProgramUniform1fv(m_raygenProgram, 1, 1, &uDepthLinearizationConstant);

			auto frustum = camera->getViewFrustum();
			core::matrix4SIMD uFrustumCorners;
			uFrustumCorners.rows[1] = frustum->getFarLeftDown();
			uFrustumCorners.rows[0] = frustum->getFarRightDown()-uFrustumCorners.rows[1];
			uFrustumCorners.rows[1] -= camPos;
			uFrustumCorners.rows[3] = frustum->getFarLeftUp();
			uFrustumCorners.rows[2] = frustum->getFarRightUp()-uFrustumCorners.rows[3];
			uFrustumCorners.rows[3] -= camPos;
			COpenGLExtensionHandler::pGlProgramUniformMatrix4fv(m_raygenProgram, 2, 1, false, uFrustumCorners.pointer()); // important to say no to transpose

			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_raygenProgram, 3, 1, m_renderSize);

			COpenGLExtensionHandler::pGlProgramUniform4uiv(m_raygenProgram, 4, 1, uImageWidth_ImageArea_TotalImageSamples_Samples);

			COpenGLExtensionHandler::pGlProgramUniform1uiv(m_raygenProgram, 5, 1, &m_samplesComputedPerPixel);
			m_samplesComputedPerPixel += m_samplesPerPixelPerDispatch;

			float uImageSize2Rcp[4] = {1.f/static_cast<float>(m_renderSize[0]),1.f/static_cast<float>(m_renderSize[1]),0.5f/static_cast<float>(m_renderSize[0]),0.5f/static_cast<float>(m_renderSize[1])};
			COpenGLExtensionHandler::pGlProgramUniform4fv(m_raygenProgram, 6, 1, uImageSize2Rcp);
		}*/
#endif		
		IGPUDescriptorSet* descriptorSets[] = {m_globalBackendDataDS.get(),m_raygenDS.get()};
		m_driver->bindDescriptorSets(EPBP_COMPUTE, m_raygenPipelineLayout.get(), 0, 2, descriptorSets, nullptr);
		m_driver->bindComputePipeline(m_raygenPipeline.get());
		m_driver->pushConstants(m_raygenPipelineLayout.get(),ISpecializedShader::ESS_COMPUTE,0u,sizeof(RaygenShaderData_t),&m_raygenShaderData);
		m_driver->dispatch(m_raygenWorkGroups[0], m_raygenWorkGroups[1], 1);
		// probably wise to flush all caches
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);
	}

	// do radeon rays
#if TODO
	m_rrManager->update(rrInstances);
#endif
	if (m_rrManager->hasImplicitCL2GLSync())
		glFlush();
	else
		glFinish();

	auto commandQueue = m_rrManager->getCLCommandQueue();
	{
#if TODO
		const cl_mem clObjects[] = {m_rayBufferAsRR.second,m_intersectionBufferAsRR.second};
		auto objCount = sizeof(clObjects)/sizeof(cl_mem);

		cl_event acquired = nullptr;
		clEnqueueAcquireGLObjects(commandQueue,objCount,clObjects,0u,nullptr,&acquired);

		clEnqueueWaitForEvents(commandQueue,1u,&acquired);
		m_rrManager->getRadeonRaysAPI()->QueryOcclusion(m_rayBufferAsRR.first,m_rayCountBufferAsRR.first,m_rayCountPerDispatch,m_intersectionBufferAsRR.first,nullptr,nullptr);
		cl_event raycastDone = nullptr;
		clEnqueueMarker(commandQueue,&raycastDone);

		if (m_rrManager->hasImplicitCL2GLSync())
		{
			clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, nullptr);
			clFlush(commandQueue);
		}
		else
		{
			cl_event released;
			clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, &released);
			clFlush(commandQueue);
			clWaitForEvents(1u, &released);
		}
#endif
	}

	// use raycast results
	{
#if TODO
		m_driver->bindDescriptorSets(EPBP_COMPUTE, m_compostLayout.get(), 2, 1, &m_raygenDS2.get(), nullptr);
		m_driver->bindComputePipeline(m_compostPipeline.get());

		// TODO push constants
		// commented out uniforms below
		/*{
			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_compostProgram, 0, 1, m_renderSize);
			
			COpenGLExtensionHandler::pGlProgramUniform4uiv(m_compostProgram, 1, 1, uImageWidth_ImageArea_TotalImageSamples_Samples);

			m_framesDone++;
			float uRcpFramesDone = 1.0/double(m_framesDone);
			COpenGLExtensionHandler::pGlProgramUniform1fv(m_compostProgram, 2, 1, &uRcpFramesDone);

			float tmp[9];
			camera->getViewMatrix().getSub3x3InverseTransposePacked(tmp);
			COpenGLExtensionHandler::pGlProgramUniformMatrix3fv(m_compostProgram, 3, 1, true, tmp);
		}*/

		m_driver->dispatch(m_resolveWorkGroups[0], m_resolveWorkGroups[1], 1);
#endif

		COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT
#ifndef _IRR_BUILD_OPTIX_
			|GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT
#else
			|(m_denoisedBuffer.getObject() ? (GL_PIXEL_BUFFER_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT):(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_UPDATE_BARRIER_BIT))
#endif
		);
	}

	// TODO: tonemap properly
#ifdef _IRR_BUILD_OPTIX_
	if (m_denoisedBuffer.getObject())
	{
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
	}
	else
#endif
	{
		auto oldVP = m_driver->getViewPort();
		m_driver->setViewPort(core::recti(0u,0u,m_renderSize[0u],m_renderSize[1u]));
		m_driver->blitRenderTargets(tmpTonemapBuffer, m_colorBuffer, false, false, {}, {}, true);
		m_driver->setViewPort(oldVP);
	}
}

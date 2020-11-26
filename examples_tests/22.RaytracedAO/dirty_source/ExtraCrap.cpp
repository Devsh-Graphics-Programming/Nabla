#include <numeric>

#include "ExtraCrap.h"

#include "irr/ext/ScreenShot/ScreenShot.h"
#include "../common.glsl"
//#include "irr/ext/MitsubaLoader/CMitsubaLoader.h"


#ifndef _IRR_BUILD_OPTIX_
	#define __C_CUDA_HANDLER_H__ // don't want CUDA declarations and defines to pollute here
#endif

using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace irr::scene;


const std::string lightStruct = R"======(
#define SLight_ET_ELLIPSOID	0u
#define SLight_ET_TRIANGLE	1u
#define SLight_ET_COUNT		2u
struct SLight
{
	vec3 factor;
	uint data0;
	mat4x3 transform; // needs row_major qualifier
	mat3 transformCofactors;
};
uint SLight_extractType(in SLight light)
{
	return bitfieldExtract(light.data0,0,findMSB(SLight_ET_COUNT)+1);
}
)======";


const std::string compostShader = R"======(
#define WORK_GROUP_DIM 32u
layout(local_size_x = WORK_GROUP_DIM, local_size_y = WORK_GROUP_DIM) in;
#define WORK_GROUP_SIZE (WORK_GROUP_DIM*WORK_GROUP_DIM)

// TODO translate into push contants
// uniforms
layout(location = 0) uniform uvec2 uImageSize;
layout(location = 1) uniform uvec4 uImageWidth_ImageArea_TotalImageSamples_Samples;
layout(location = 2) uniform float uRcpFramesDone;
layout(location = 3) uniform mat3 uNormalMatrix;

// image views
layout(set = 2, binding = 0) uniform usampler2D lightIndex;
layout(set = 2, binding = 1) uniform sampler2D albedobuf;
layout(set = 2, binding = 2) uniform sampler2D normalbuf;
layout(set = 2, binding = 3, rgba32f) restrict uniform image2D framebuffer;

// SSBOs
layout(set = 2, binding = 4, std430) restrict readonly buffer Rays
{
	RadeonRays_ray rays[];
};
layout(set = 2, binding = 5, std430) restrict buffer Queries
{
	int hit[];
};

layout(set = 2, binding = 6, std430, row_major) restrict readonly buffer LightRadiances
{
	vec3 lightRadiance[]; // Watts / steriadian / steradian
};

#ifdef USE_OPTIX_DENOISER
layout(set = 2, binding = 7, std430) restrict writeonly buffer DenoiserColorInput
{
	float16_t colorOutput[];
};
layout(set = 2, binding = 8, std430) restrict writeonly buffer DenoiserAlbedoInput
{
	float16_t albedoOutput[];
};
layout(set = 2, binding = 9, std430) restrict writeonly buffer DenoiserNormalInput
{
	float16_t normalOutput[];
};
#endif





#define RAYS_IN_CACHE 1u
#define KERNEL_HALF_SIZE 0u
#define CACHE_DIM (WORK_GROUP_DIM+2u*KERNEL_HALF_SIZE)
#define CACHE_SIZE (CACHE_DIM*CACHE_DIM*RAYS_IN_CACHE)
shared uint rayScratch0[CACHE_SIZE];
shared uint rayScratch1[CACHE_SIZE];
shared float rayScratch2[CACHE_SIZE];
shared float rayScratch3[CACHE_SIZE];
shared float rayScratch4[CACHE_SIZE];

void main()
{
	ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
	uint baseID = gl_GlobalInvocationID.x+uImageWidth_ImageArea_TotalImageSamples_Samples.x*gl_GlobalInvocationID.y;
	bool alive = all(lessThan(gl_GlobalInvocationID.xy,uImageSize));

	vec3 normal;
	if (alive)
		normal = irr_glsl_NormalDecode_signedSpherical(texelFetch(normalbuf,pixelCoord,0).rg);

	vec4 acc = vec4(0.0);
	if (uRcpFramesDone<1.0 && alive)
		acc = imageLoad(framebuffer,pixelCoord);

	vec3 color = vec3(0.0);
	uvec2 groupLocation = gl_WorkGroupID.xy*WORK_GROUP_DIM;
	for (uint i=0u; i<uImageWidth_ImageArea_TotalImageSamples_Samples.w; i+=RAYS_IN_CACHE)
	{
		for (uint lid=gl_LocalInvocationIndex; lid<CACHE_SIZE; lid+=WORK_GROUP_SIZE)
		{
			ivec3 coord;
			coord.z = int(lid/(CACHE_DIM*CACHE_DIM));
			coord.y = int(lid/CACHE_DIM)-int(coord.z*CACHE_DIM+KERNEL_HALF_SIZE);
			coord.x = int(lid)-int(coord.y*CACHE_DIM+KERNEL_HALF_SIZE);
			coord.z += int(i);
			coord.y += int(groupLocation.y);
			coord.x += int(groupLocation.x);
			bool invalidRay = any(lessThan(coord.xy,ivec2(0,0))) || any(greaterThan(coord,ivec3(uImageSize,uImageWidth_ImageArea_TotalImageSamples_Samples.w)));
			int rayID = coord.x+coord.y*int(uImageWidth_ImageArea_TotalImageSamples_Samples.x)+coord.z*int(uImageWidth_ImageArea_TotalImageSamples_Samples.y);
			invalidRay = invalidRay ? false:(hit[rayID]>=0);
			rayScratch0[lid] = invalidRay ? 0u:rays[rayID].useless_padding;
			rayScratch1[lid] = invalidRay ? 0u:rays[rayID].backfaceCulling;
			rayScratch2[lid] = invalidRay ? 0.0:rays[rayID].direction[0];
			rayScratch3[lid] = invalidRay ? 0.0:rays[rayID].direction[1];
			rayScratch4[lid] = invalidRay ? 0.0:rays[rayID].direction[2];
		}
		barrier();
		memoryBarrierShared();

		uint localID = (gl_LocalInvocationID.x+KERNEL_HALF_SIZE)+(gl_LocalInvocationID.y+KERNEL_HALF_SIZE)*CACHE_DIM;
		for (uint j=localID; j<CACHE_SIZE; j+=CACHE_DIM*CACHE_DIM)
		{
			vec3 raydiance = vec4(unpackHalf2x16(rayScratch0[j]),unpackHalf2x16(rayScratch1[j])).gra;
			// TODO: sophisticated BSDF eval
			raydiance *= max(dot(vec3(rayScratch2[j],rayScratch3[j],rayScratch4[j]),normal),0.0)/kPI;
			color += raydiance;
		}

		// hit buffer needs clearing
		for (uint j=i; j<min(uImageWidth_ImageArea_TotalImageSamples_Samples.w,i+RAYS_IN_CACHE); j++)
		{
			uint rayID = baseID+j*uImageWidth_ImageArea_TotalImageSamples_Samples.y;
			hit[rayID] = -1;
		}
	}

	if (alive)
	{
		// TODO: sophisticated BSDF eval
		vec3 albedo = texelFetch(albedobuf,pixelCoord,0).rgb;
		color *= albedo;

		// TODO: move  ray gen, for fractional sampling
		color *= 1.0/float(uImageWidth_ImageArea_TotalImageSamples_Samples.w);

		uint lightID = texelFetch(lightIndex,pixelCoord,0)[0];
		if (lightID!=0xdeadbeefu)
			color += lightRadiance[lightID];

		// TODO: optimize the color storage (RGB9E5/RGB19E7 anyone?)
		acc.rgb += (color-acc.rgb)*uRcpFramesDone;
		imageStore(framebuffer,pixelCoord,acc);
#ifdef USE_OPTIX_DENOISER
		for (uint i=0u; i<3u; i++)
			colorOutput[baseID*3+i] = float16_t(acc[i]);
			//colorOutput[baseID*3+i] = float16_t(clamp(acc[i],0.0001,10000.0));
		for (uint i=0u; i<3u; i++)
			albedoOutput[baseID*3+i] = float16_t(albedo[i]);
		normal = uNormalMatrix*normal;
		for (uint i=0u; i<3u; i++)
			normalOutput[baseID*3+i] = float16_t(normal[i]);
#endif
	}
}

)======";


constexpr uint32_t kOptiXPixelSize = sizeof(uint16_t)*3u;


Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, irr::scene::ISceneManager* _smgr, core::smart_refctd_ptr<video::IGPUDescriptorSet>&& globalBackendDataDS, bool useDenoiser) :
		m_useDenoiser(useDenoiser),	m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager), m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
		m_sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), /*m_renderSize{0u,0u}, */m_rightHanded(false),
		m_globalBackendDataDS(std::move(globalBackendDataDS)), rrShapeCache(),
#if TODO
		m_raygenProgram(0u), m_compostProgram(0u),
		m_raygenWorkGroups{0u,0u}, m_resolveWorkGroups{0u,0u},
		m_rayBuffer(), m_intersectionBuffer(), m_rayCountBuffer(),
		m_rayBufferAsRR(nullptr,nullptr), m_intersectionBufferAsRR(nullptr,nullptr), m_rayCountBufferAsRR(nullptr,nullptr),
		nodes(), rrInstances(),
#endif
		m_lightCount(0u),
		m_visibilityBufferAttachments{nullptr}, m_maxSamples(0u), m_samplesPerPixelPerDispatch(0u), m_rayCountPerDispatch(0u), m_framesDone(0u), m_samplesComputedPerPixel(0u),
		m_sampleSequence(), m_scrambleTexture(), m_accumulation(), m_tonemapOutput(), m_visibilityBuffer(nullptr), tmpTonemapBuffer(nullptr), m_colorBuffer(nullptr)
	#ifdef _IRR_BUILD_OPTIX_
		,m_cudaStream(nullptr)
	#endif
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
}


core::smart_refctd_ptr<video::IGPUImageView> Renderer::createScreenSizedTexture(E_FORMAT format)
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

#if TODO
core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> Renderer::createDS2layoutCompost(bool useDenoiser, core::smart_refctd_ptr<IGPUSampler>& nearestSampler)
{
	constexpr uint32_t MAX_COUNT = 10u;
	video::IGPUDescriptorSetLayout::SBinding bindings[MAX_COUNT];

	auto makeBinding = [](uint32_t bnd, uint32_t cnt, ISpecializedShader::E_SHADER_STAGE stage, E_DESCRIPTOR_TYPE dt, core::smart_refctd_ptr<video::IGPUSampler>* samplers)
	{
		video::IGPUDescriptorSetLayout::SBinding b;
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

core::smart_refctd_ptr<video::IGPUDescriptorSet> Renderer::createDS2Compost(bool useDenoiser, core::smart_refctd_ptr<IGPUSampler>& nearestSampler)
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
	video::IGPUDescriptorSet::SWriteDescriptorSet write[MAX_COUNT];
	write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[1].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[3].descriptorType = EDT_STORAGE_IMAGE;
	for (uint32_t i = 4u; i < MAX_COUNT; ++i)
		write[i].descriptorType = EDT_STORAGE_BUFFER;
	video::IGPUDescriptorSet::SDescriptorInfo info[MAX_COUNT];

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

core::smart_refctd_ptr<video::IGPUPipelineLayout> Renderer::createLayoutCompost()
{
	auto& ds2 = m_compostDS2;
	auto* layout2 = const_cast<video::IGPUDescriptorSetLayout*>(ds2->getLayout());

	//TODO push constants
	return m_driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(layout2), nullptr);
}

core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> Renderer::createDS2layoutRaygen(core::smart_refctd_ptr<IGPUSampler>& nearestSampler)
{
	constexpr uint32_t count = 6u;

	auto makeBinding = [](uint32_t bnd, uint32_t cnt, ISpecializedShader::E_SHADER_STAGE stage, E_DESCRIPTOR_TYPE dt, core::smart_refctd_ptr<video::IGPUSampler>* samplers)
	{
		video::IGPUDescriptorSetLayout::SBinding b;
		b.binding = bnd;
		b.count = cnt;
		b.samplers = samplers;
		b.stageFlags = stage;
		b.type = dt;

		return b;
	};

	const auto stage = ISpecializedShader::ESS_COMPUTE;
	video::IGPUDescriptorSetLayout::SBinding bindings[count];
	bindings[0] = makeBinding(0u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler);
	bindings[1] = makeBinding(1u, 1u, stage, EDT_UNIFORM_TEXEL_BUFFER, nullptr);
	bindings[2] = makeBinding(2u, 1u, stage, EDT_COMBINED_IMAGE_SAMPLER, &nearestSampler);
	bindings[3] = makeBinding(3u, 1u, stage, EDT_STORAGE_BUFFER, nullptr);
	bindings[4] = makeBinding(4u, 1u, stage, EDT_STORAGE_BUFFER, nullptr);
	bindings[5] = makeBinding(5u, 1u, stage, EDT_STORAGE_BUFFER, nullptr);

	return m_driver->createGPUDescriptorSetLayout(bindings, bindings + count);
}

core::smart_refctd_ptr<video::IGPUDescriptorSet> Renderer::createDS2Raygen(core::smart_refctd_ptr<IGPUSampler>& nearstSampler)
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

	video::IGPUDescriptorSet::SWriteDescriptorSet write[count];
	write[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
	write[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write[3].descriptorType = EDT_STORAGE_BUFFER;
	write[4].descriptorType = EDT_STORAGE_BUFFER;
	write[5].descriptorType = EDT_STORAGE_BUFFER;
	video::IGPUDescriptorSet::SDescriptorInfo info[count];

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

core::smart_refctd_ptr<video::IGPUPipelineLayout> Renderer::createLayoutRaygen()
{
	//ds from mitsuba loader (VT stuff, material compiler data, instance data)
	//will be needed for gbuffer pass as well (at least to spit out instruction offsets/counts and maybe offset into instance data buffer)
	auto& ds0 = m_globalBackendDataDS;
	auto* layout0 = const_cast<video::IGPUDescriptorSetLayout*>(ds0->getLayout());

	auto& ds2 = m_raygenDS2;
	auto* layout2 = const_cast<video::IGPUDescriptorSetLayout*>(ds2->getLayout());

	//TODO push constants
	return m_driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(layout0), nullptr, core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(layout2), nullptr);
}
#endif

Renderer::~Renderer()
{
	deinit();
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

#if TODO
	core::vector<SLight> lights;
	core::vector<uint32_t> lightCDF;
	core::vector<core::vectorSIMDf> lightRadiances;
	auto& lightPDF = reinterpret_cast<core::vector<float>&>(lightCDF); // save on memory
#endif
	const ext::MitsubaLoader::CGlobalMitsubaMetadata* globalMeta = nullptr;
	{
		auto contents = meshes.getContents();
		for (auto& cpumesh_ : contents)
		{
			auto cpumesh = static_cast<asset::ICPUMesh*>(cpumesh_.get());

			auto* meta = cpumesh->getMetadata();
			assert(meta && core::strcmpi(meta->getLoaderName(), ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
			globalMeta = meshmeta->globalMetadata.get();

			// WARNING!!!
			// all this instance-related things is a rework candidate since mitsuba loader supports instances
			// (all this metadata should be global, but meshbuffers has instanceCount correctly set
			// and globalDS with per-instance data (transform, normal matrix, instructions offsets, etc)
			const auto& instances = meshmeta->getInstances();

			auto meshBufferCount = cpumesh->getMeshBufferCount();
			for (auto i=0u; i<meshBufferCount; i++)
			{
				// TODO: get rid of `getMeshBuffer` and `getMeshBufferCount`, just return a range as `getMeshBuffers`
				auto cpumb = cpumesh->getMeshBuffer(i);
				m_rrManager->makeRRShapes(rrShapeCache, &cpumb, (&cpumb)+1);
			}

			// set up lights
			for (auto instance : instances)
			if (instance.emitter.type!=ext::MitsubaLoader::CElementEmitter::Type::INVALID)
			{
				assert(instance.emitter.type==ext::MitsubaLoader::CElementEmitter::Type::AREA);
#ifdef TODO
				SLight light;
				light.setFactor(instance.emitter.area.radiance);
				light.analytical = SLight::CachedTransform(instance.tform);

				// TODO: OBB stuffs
				auto addLight = [&instance,&cpumesh,&lightPDF,&lights,&lightRadiances](auto& newLight,float approxArea) -> void
				{
					float weight = newLight.computeFlux(approxArea) * instance.emitter.area.samplingWeight;
					if (weight <= FLT_MIN)
						return;

					lightPDF.push_back(weight);
					lights.push_back(newLight);
					lightRadiances.push_back(newLight.strengthFactor);
				};
				auto areaFromTriangulationAndMakeMeshLight = [&]() -> float
				{
					double totalSurfaceArea = 0.0;

					uint32_t runningTriangleCount = 0u;
					for (auto i=0u; i<meshBufferCount; i++)
					{
						auto cpumb = cpumesh->getMeshBuffer(i);
						reinterpret_cast<uint32_t&>(cpumb->getMaterial().userData) = lights.size();

						uint32_t triangleCount = 0u;
						asset::IMeshManipulator::getPolyCount(triangleCount,cpumb);
						for (auto triID=0u; triID<triangleCount; triID++)
						{
							auto triangle = asset::IMeshManipulator::getTriangleIndices(cpumb,triID);

							core::vectorSIMDf v[3];
							for (auto k=0u; k<3u; k++)
								v[k] = cpumb->getPosition(triangle[k]);

							float triangleArea = NAN;
							auto triLight = SLight::createFromTriangle(instance.emitter.area.radiance, light.analytical, v, &triangleArea);
							if (light.type==SLight::ET_TRIANGLE)
								addLight(triLight,triangleArea);
							else
								totalSurfaceArea += triangleArea;
						}
					}
					return totalSurfaceArea;
				};

				auto totalArea = areaFromTriangulationAndMakeMeshLight();
				if (light.type!=SLight::ET_TRIANGLE)
					addLight(light,totalArea);
#endif
			}
		}

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
	m_renderSize[0] = m_driver->getScreenSize().Width;
	m_renderSize[1] = m_driver->getScreenSize().Height;
	if (globalMeta)
	{
#ifdef TODO
		constantClearColor.set(0.f, 0.f, 0.f, 1.f);
		float constantCombinedWeight = 0.f;
		for (auto emitter : globalMeta->emitters)
		{
			SLight light;

			float weight = 0.f;
			switch (emitter.type)
			{
				case ext::MitsubaLoader::CElementEmitter::Type::CONSTANT:
					constantClearColor += emitter.constant.radiance;
					constantCombinedWeight += emitter.constant.samplingWeight;
					break;
				case ext::MitsubaLoader::CElementEmitter::Type::INVALID:
					break;
				default:
				#ifdef _DEBUG
					assert(false);
				#endif
					//weight = emitter..samplingWeight;
					//light.type = SLight::ET_;
					//light.setFactor(emitter..radiance);
					break;
			}
			if (weight==0.f)
				continue;
			
			weight *= light.computeFlux(NAN);
			if (weight <= FLT_MIN)
				continue;

			lightPDF.push_back(weight);
			lights.push_back(light);
		}
		// add constant light
		if (constantCombinedWeight>FLT_MIN)
		{
			core::matrix3x4SIMD tform;
			tform.setScale(core::vectorSIMDf().set(m_sceneBound.getExtent())*core::sqrt(3.f/4.f));
			tform.setTranslation(core::vectorSIMDf().set(m_sceneBound.getCenter()));
			SLight::CachedTransform cachedT(tform);

			const float areaOfSphere = 4.f * core::PI<float>();

			auto startIx = lights.size();
			float triangulationArea = 0.f;
			for (const auto& tri : m_precomputedGeodesic)
			{
				// make light, correct radiance parameter for domain of integration
				float area = NAN;
				auto triLight = SLight::createFromTriangle(constantClearColor,cachedT,&tri[0],&area);
				// compute flux, abuse of notation on area parameter, we've got the wrong units on strength factor so need to cancel
				float flux = triLight.computeFlux(1.f);
				// add to light list
				float weight = constantCombinedWeight*flux;
				if (weight>FLT_MIN)
				{
					// needs to be weighted be contribution to sphere area
					lightPDF.push_back(weight*area);
					lights.push_back(triLight);
					lightRadiances.push_back(constantClearColor);
				}
				triangulationArea += area;
			}

			for (auto i=startIx; i<lights.size(); i++)
				lightPDF[i] *= areaOfSphere/triangulationArea;
		}
#endif
		if (globalMeta->sensors.size())
		{
			// just grab the first sensor
			const auto& sensor = globalMeta->sensors.front();
			const auto& film = sensor.film;
			assert(film.cropOffsetX == 0);
			assert(film.cropOffsetY == 0);
			m_renderSize[0] = film.cropWidth;
			m_renderSize[1] = film.cropHeight;
		}
	}

#ifdef TODO
	//! TODO: move out into a function `finalizeLights`
	if (lights.size())
	{
		double weightSum = 0.0;
		for (auto i=0u; i<lightPDF.size(); i++)
			weightSum += lightPDF[i];
		assert(weightSum>FLT_MIN);

		constexpr double UINT_MAX_DOUBLE = double(0x1ull<<32ull);

		double weightSumRcp = UINT_MAX_DOUBLE/weightSum;
		double partialSum = 0.0;
		for (auto i=0u; i<lightCDF.size(); i++)
		{
			uint32_t prevCDF = i ? lightCDF[i-1u]:0u;

			double pdf = lightPDF[i];
			partialSum += pdf;

			double inv_prob = NAN;
			double exactCDF = partialSum*weightSumRcp+double(FLT_MIN);
			if (exactCDF<UINT_MAX_DOUBLE)
			{
				lightCDF[i] = static_cast<uint32_t>(exactCDF);
				inv_prob = UINT_MAX_DOUBLE/(lightCDF[i]-prevCDF);
			}
			else
			{
				assert(exactCDF<UINT_MAX_DOUBLE+1.0);
				lightCDF[i] = 0xdeadbeefu;
				inv_prob = 1.0/(1.0-double(prevCDF)/UINT_MAX_DOUBLE);
			}
			lights[i].setFactor(core::vectorSIMDf(lights[i].strengthFactor)*inv_prob);
		}
		lightCDF.back() = 0xdeadbeefu;

		m_lightCount = lights.size();
		m_lightCDFBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lightCDF.size()*sizeof(uint32_t),lightCDF.data());
		m_lightBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lights.size()*sizeof(SLight),lights.data());
		m_lightRadianceBuffer = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(lightRadiances.size()*sizeof(core::vectorSIMDf),lightRadiances.data());
	}
#endif
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


	core::smart_refctd_ptr<video::IGPUSampler> smplr_nearest;
	{
		video::IGPUSampler::SParams params;
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

	auto rr_includes = m_rrManager->getRadeonRaysGLSLIncludes();
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
		m_denoiserInputBuffer = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(inputBuffSz),core::dont_grab);
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&m_denoiserInputBuffer, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY)))
			break;
		m_denoiserStateBuffer = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(m_denoiserMemReqs.stateSizeInBytes),core::dont_grab);
		if (!cuda::CCUDAHandler::defaultHandleResult(cuda::CCUDAHandler::registerBuffer(&m_denoiserStateBuffer)))
			break;
		m_denoisedBuffer = core::smart_refctd_ptr<video::IGPUBuffer>(m_driver->createDeviceLocalGPUBufferOnDedMem(kOptiXPixelSize*renderPixelCount), core::dont_grab);
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

	for (auto& node : nodes)
		node->remove();
	nodes.clear();
	m_rrManager->detachInstances(rrInstances.begin(),rrInstances.end());
	m_rrManager->deleteInstances(rrInstances.begin(),rrInstances.end());
	rrInstances.clear();

	constantClearColor.set(0.f,0.f,0.f,1.f);
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


	m_lightCount = 0u;


	m_rrManager->deleteShapes(rrShapeCache.begin(), rrShapeCache.end());
	rrShapeCache.clear();

	m_globalBackendDataDS = nullptr;
	m_renderSize[0u] = m_renderSize[1u] = 0u;
	m_rightHanded = false;
	m_sceneBound = core::aabbox3df(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

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
	m_driver->setRenderTarget(m_visibilityBuffer);
	{ // clear
		m_driver->clearZBuffer();
		uint32_t clearTriangleID[4] = {0xffffffffu,0,0,0};
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT2, clearTriangleID);
		float zero[4] = { 0.f,0.f,0.f,0.f };
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT1, zero);
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT2, zero);
	}


	auto camera = m_smgr->getActiveCamera();
	auto prevViewProj = camera->getConcatenatedMatrix();

	camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(timer->getTime()).count());
	camera->render();

	// TODO: fill visibility buffer
	{
	}

	auto currentViewProj = camera->getConcatenatedMatrix();
	if (!core::equals(prevViewProj,currentViewProj,core::ROUNDING_ERROR<core::matrix4SIMD>()*1000.0))
		m_framesDone = 0u;

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

		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_raygenLayout.get(), 0, 1, &m_globalBackendDataDS.get(), nullptr);
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_raygenLayout.get(), 2, 1, &m_raygenDS2.get(), nullptr);
		m_driver->bindComputePipeline(m_raygenPipeline.get());
		m_driver->dispatch(m_raygenWorkGroups[0], m_raygenWorkGroups[1], 1);
#endif		
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
#endif

		if (m_rrManager->hasImplicitCL2GLSync())
		{
			//clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, nullptr);
			clFlush(commandQueue);
		}
		else
		{
			cl_event released;
			//clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, &raycastDone, &released);
			clFlush(commandQueue);
			clWaitForEvents(1u, &released);
		}
	}

	// use raycast results
	{
#if TODO
		m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_compostLayout.get(), 2, 1, &m_raygenDS2.get(), nullptr);
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
		video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_UNPACK_BUFFER,glbuf->getOpenGLName());
		video::COpenGLExtensionHandler::extGlTextureSubImage2D(gltex->getOpenGLName(),gltex->getOpenGLTextureType(),0,0,0,rSize[0],rSize[1],GL_RGB,GL_HALF_FLOAT,nullptr);
		video::COpenGLExtensionHandler::extGlBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
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

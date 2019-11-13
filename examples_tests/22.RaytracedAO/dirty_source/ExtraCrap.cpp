#include <numeric>

#include "ExtraCrap.h"

#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace irr::scene;


const std::string raygenShaderExtensions = R"======(
#version 430 core
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_ARB_shader_ballot : require
)======";

/*
const std::string raygenShader = R"======(
#define WORK_GROUP_DIM 16u
layout(local_size_x = WORK_GROUP_DIM, local_size_y = WORK_GROUP_DIM) in;
#define WORK_GROUP_SIZE (WORK_GROUP_DIM*WORK_GROUP_DIM)

// image views
layout(binding = 0) uniform usampler2D depth;
layout(binding = 1) uniform usampler2D normals;

// temporary debug
layout(binding = 0, rgba32f) uniform writeonly image2D out_img;


#define MAX_RAY_BUFFER_SZ 1024u
shared uint sharedData[MAX_RAY_BUFFER_SZ];

#define IMAGE_DATA_SZ (WORK_GROUP_SIZE*2u)

const uint SUBGROUP_COUNT = WORK_GROUP_SIZE/gl_SubGroupSizeARB;
#define RAY_SIZE 6u
const uint MAX_RAYS = ((MAX_RAY_BUFFER_SZ-SUBGROUP_COUNT)/RAY_SIZE);

#if MAX_RAY_BUFFER_SZ<(WORK_GROUP_SIZE*8u)
#error "Shared Memory too small for texture data"
#endif


vec3 decode(in vec2 enc)
{
	const float kPI = 3.14159265358979323846;
	float ang = enc.x*kPI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}


uvec2 morton2xy_4bit(in uint id)
{
	return XXX;
}
uint xy2morton_4bit(in uvec2 coord)
{
	return XXX;
}

// TODO: do this properly
uint getGlobalOffset(in bool cond)
{
	if (gl_LocalInvocationIndex<=gl_SubGroupSizeARB)
		sharedData[gl_LocalInvocationIndex] = 0u;
	barrier();
	memoryBarrierShared();
	
	uint raysBefore = countBits(ballotARB(cond)&gl_SubGroupLtMaskARB);

	uint macroInvocation = gl_LocalInvocationIndex/gl_SubGroupSizeARB;
	uint addr = IMAGE_DATA_SZ+macroInvocation;
	if (gl_SubGroupInvocationARB==gl_SubGroupSizeARB-1u)
		sharedData[addr] = atomicAdd(sharedData[gl_SubGroupSizeARB],raysBefore+1);
	barrier();
	memoryBarrierShared();

	if (gl_LocalInvocationIndex==0u)
		sharedData[gl_SubGroupSizeARB] = atomicAdd(rayCount,sharedData[gl_SubGroupSizeARB]);
	barrier();
	memoryBarrierShared();

	return sharedData[gl_SubGroupSizeARB]+sharedData[addr]+raysBefore;
}


void main()
{
	uint mortonIndex = xy2morton_4bit();
	uint subgroupInvocationIndex = gl_LocalInvocationIndex&63u;
	uint wideSubgroupID = gl_LocalInvocationIndex>>6u;

	uvec2 workGroupCoord = gl_WorkGroupID.uv*gl_WorkGroupSize.uv;
	vec2 sharedUV = (vec2(workGroupCoord+uvec2(8,8))+vec2(0.5))/textureSize();
	ivec2 offset = (ivec2(subgroupInvocationIndex&7u,subgroupInvocationIndex>>3u)<<1)-ivec2(8,8);

	uvec4 data;
	switch (wideSubgroupID)
	{
		case 0u:
			data = textureGatherOffset(depth,sharedUV,offset);
			break;
		case 1u:
			data = textureGatherOffset(normals,sharedUV,offset);
			break;
		default:
			break;
	}
	if (wideSubgroupID<2u)
	for (uint i=0; i<4u; i++)
		sharedData[wideSubgroupID*WORK_GROUP_SIZE+i*(WORK_GROUP_SIZE/4u)+subgroupInvocationIndex] = data[i];
	barrier();
	memoryBarrierShared();


	// get depth to check if alive
	float depth = uintBitsToFloat(sharedData[gl_LocalInvocationIndex]);
	ivec2 outputLocation = ivec2(workGroupCoord)+gl_LocationInvocationID.xy;
	//outputLocation =

	bool createRay = all(lessThan(outputLocation,textureSize(depth,0))) && depth>0.0;
	uint storageOffset = getGlobalOffset(createRay);

//	if (createRay)
//	{
		// reconstruct
		vec2 encnorm = unpackSnorm2x16(sharedData[gl_LocalInvocationIndex+WORK_GROUP_SIZE]);
//		vec3 position = reconstructFromNonLinearZ(depth);
		vec3 normal = decode(theta,phi);
		// debug
		imageStore(out_img,outputLocation,vec4(normal,1.0));
	
		// compute rays
		float fresnel = 0.5;
		RadeonRays_ray newray;
		newray.origin = position;
		newray.maxT = uMaxLen;
		newray.direction = reflect(position-uCameraPos);
		newray.time = 0.0;
		newray.mask = -1;
		newray.active = 1;
		newray.backfaceCulling = floatBitsToInt(fresnel);
		newray.useless_padding = outputLocation.x|(outputLocation.y<<13));

		// store rays
		rays[getGlobalOffset()] = newray;
	
//	}

	// future ray compaction system
	{
		// compute per-pixel data and store in smem (24 bytes)
		// decide how many rays to spawn
		// cumulative histogram of ray counts (4 bytes)
		// increment global atomic counter
		// binary search for pixel in histogram
		// fetch pixel-data
		// compute rays for pixel-data
		// store rays in buffer
	}
}

)======";
*/
const std::string raygenShader = R"======(
#define WORK_GROUP_DIM 16u
layout(local_size_x = WORK_GROUP_DIM, local_size_y = WORK_GROUP_DIM) in;
#define WORK_GROUP_SIZE (WORK_GROUP_DIM*WORK_GROUP_DIM)

// uniforms
layout(location = 0) uniform vec3 uCameraPos;
layout(location = 1) uniform mat4 uViewProjInverse;
layout(location = 2) uniform uvec2 uImageSize;
layout(location = 3) uniform uvec2 uSamples_ImageWidthSamples;
layout(location = 4) uniform vec4 uImageSize2Rcp;

// image views
layout(binding = 0) uniform sampler2D depthbuf;
layout(binding = 1) uniform usampler2D normalbuf;

// SSBOs
layout(binding = 0, std430) restrict writeonly buffer Rays
{
	RadeonRays_ray rays[];
};


vec3 decode(in vec2 enc)
{
	const float kPI = 3.14159265358979323846;
	float ang = enc.x*kPI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}


#define FLT_MAX 3.402823466e+38

void main()
{
	uvec2 outputLocation = gl_GlobalInvocationID.xy;
	bool alive = all(lessThan(outputLocation,uImageSize));
	if (alive)
	{
		float revdepth = texelFetch(depthbuf,ivec2(outputLocation),0).r;

		uint outputID = uSamples_ImageWidthSamples.x*outputLocation.x+uSamples_ImageWidthSamples.y*outputLocation.y;

		// unproject
		vec3 position;
		{
			vec2 NDC = vec2(outputLocation)*uImageSize2Rcp.xy+uImageSize2Rcp.zw;
			// reorder for precision
			vec4 tmp = uViewProjInverse[0]*NDC.x+uViewProjInverse[1]*NDC.y-uViewProjInverse[2]*revdepth;
			tmp += uViewProjInverse[2]+uViewProjInverse[3];

			position = tmp.xyz/tmp.www;
		}

		vec4 bsdf = vec4(0.5,0.5,0.5,-1.0);

		RadeonRays_ray newray;
		//newray.origin = position;
		newray.origin = uCameraPos;
		newray.maxT = alive ? FLT_MAX:0.0;
		//newray.direction = reflect(position-uCameraPos,normal);
		newray.direction = position-uCameraPos;
		newray.time = 0.0;
		newray.mask = -1;
		newray._active = alive ? 1:0;
		newray.backfaceCulling = int(packHalf2x16(bsdf.ab));
		newray.useless_padding = int(packHalf2x16(bsdf.gr));

		// store rays
		rays[outputID] = newray;
	}
}

)======";

inline GLuint createComputeShader(const std::string& source)
{
    GLuint program = COpenGLExtensionHandler::extGlCreateProgram();
	GLuint cs = COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

	const char* tmp = source.c_str();
	COpenGLExtensionHandler::extGlShaderSource(cs, 1, const_cast<const char**>(&tmp), NULL);
	COpenGLExtensionHandler::extGlCompileShader(cs);

	// check for compilation errors
    GLint success;
    GLchar infoLog[0x200];
    COpenGLExtensionHandler::extGlGetShaderiv(cs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        COpenGLExtensionHandler::extGlGetShaderInfoLog(cs, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS COMPILATION ERROR:\n", infoLog,ELL_ERROR);
        COpenGLExtensionHandler::extGlDeleteShader(cs);
        COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
	}

	COpenGLExtensionHandler::extGlAttachShader(program, cs);
	COpenGLExtensionHandler::extGlLinkProgram(program);

	//check linking errors
	success = 0;
    COpenGLExtensionHandler::extGlGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        COpenGLExtensionHandler::extGlGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS LINK ERROR:\n", infoLog,ELL_ERROR);
        COpenGLExtensionHandler::extGlDeleteShader(cs);
        COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
    }

	return program;
}



class SimpleCallBack : public video::IShaderConstantSetCallBack
{
		video::E_MATERIAL_TYPE currentMat;

		int32_t mvpUniformLocation[video::EMT_COUNT+2];
		int32_t nUniformLocation[video::EMT_COUNT+2];
		int32_t colorUniformLocation[video::EMT_COUNT+2];
		int32_t nastyUniformLocation[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE mvpUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE colorUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nastyUniformType[video::EMT_COUNT+2];
	public:
		SimpleCallBack() : currentMat(video::EMT_SOLID)
		{
			std::fill(mvpUniformLocation, reinterpret_cast<int32_t*>(mvpUniformType), -1);
		}

		virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
		{
			for (size_t i=0; i<constants.size(); i++)
			{
				if (constants[i].name == "MVP")
				{
					mvpUniformLocation[materialType] = constants[i].location;
					mvpUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "NormalMatrix")
				{
					nUniformLocation[materialType] = constants[i].location;
					nUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "color")
				{
					colorUniformLocation[materialType] = constants[i].location;
					colorUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "nasty")
				{
					nastyUniformLocation[materialType] = constants[i].location;
					nastyUniformType[materialType] = constants[i].type;
				}
			}
		}

		virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
		{
			currentMat = material.MaterialType;
			services->setShaderConstant(&material.AmbientColor, colorUniformLocation[currentMat], colorUniformType[currentMat], 1);
			services->setShaderConstant(&material.MaterialTypeParam, nastyUniformLocation[currentMat], nastyUniformType[currentMat], 1);
		}

		virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
		{
			if (nUniformLocation[currentMat]>=0)
			{
				float tmp[9];
				services->getVideoDriver()->getTransform(video::E4X3TS_VIEW).getSub3x3InverseTransposePacked(tmp);
				services->setShaderConstant(tmp, nUniformLocation[currentMat], nUniformType[currentMat], 1);
			}
			if (mvpUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation[currentMat], mvpUniformType[currentMat], 1);
		}

		virtual void OnUnsetMaterial() {}
};


Renderer::Renderer(IVideoDriver* _driver, IAssetManager* _assetManager, ISceneManager* _smgr) :
		m_driver(_driver), m_smgr(_smgr), m_assetManager(_assetManager),
		nonInstanced(static_cast<E_MATERIAL_TYPE>(-1)), m_raygenProgram(0u), m_rrManager(ext::RadeonRays::Manager::create(m_driver)),
		m_depth(), m_albedo(), m_normals(), m_colorBuffer(nullptr), m_gbuffer(nullptr),
		m_workGroupCount{0u,0u}, m_samplesPerDispatch(0u), m_rayCount(0u),
		m_rayBuffer(), m_intersectionBuffer(), m_rayCountBuffer(),
		m_rayBufferAsRR(nullptr,nullptr), m_intersectionBufferAsRR(nullptr,nullptr), m_rayCountBufferAsRR(nullptr,nullptr),
		nodes(), sceneBound(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), rrInstances()
{
	SimpleCallBack* cb = new SimpleCallBack();
	nonInstanced = (E_MATERIAL_TYPE)m_driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	auto includes = m_rrManager->getRadeonRaysGLSLIncludes();
	m_raygenProgram = createComputeShader(
		raygenShaderExtensions+
		//"irr/builtin/glsl/ext/RadeonRays/"
		includes->getBuiltinInclude("ray.glsl")+
		includes->getBuiltinInclude("intersection.glsl")+
		raygenShader
	);
}

Renderer::~Renderer()
{
	if (m_raygenProgram)
		COpenGLExtensionHandler::extGlDeleteProgram(m_raygenProgram);
}


void Renderer::init(const SAssetBundle& meshes, uint32_t rayBufferSize)
{
	deinit();

	const ext::MitsubaLoader::CGlobalMitsubaMetadata* globalMeta = nullptr;
	{
		auto contents = meshes.getContents();
		for (auto cpuit = contents.first; cpuit!=contents.second; cpuit++)
		{
			auto cpumesh = static_cast<asset::ICPUMesh*>(cpuit->get());
			for (auto i=0u; i<cpumesh->getMeshBufferCount(); i++)
			{
				// TODO: get rid of `getMeshBuffer` and `getMeshBufferCount`, just return a range as `getMeshBuffers`
				auto cpumb = cpumesh->getMeshBuffer(i);
				m_rrManager->makeRRShapes(rrShapeCache, &cpumb, (&cpumb)+1);
			}
		}

		auto gpumeshes = m_driver->getGPUObjectsFromAssets<ICPUMesh>(contents.first, contents.second);
		auto cpuit = contents.first;
		for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
		{
			auto* meta = cpuit->get()->getMetadata();

			assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
			globalMeta = meshmeta->globalMetadata.get();
			const auto& instances = meshmeta->getInstances();

			const auto& gpumesh = *gpuit;
			for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
				gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = nonInstanced;

			for (auto instance : instances)
			{
				auto node = core::smart_refctd_ptr<IMeshSceneNode>(m_smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh)));
				node->setRelativeTransformationMatrix(instance.getAsRetardedIrrlichtMatrix());
				sceneBound.addInternalBox(node->getTransformedBoundingBox());
				nodes.push_back(std::move(node));
			}
		}
		core::vector<int32_t> ids(nodes.size());
		std::iota(ids.begin(), ids.end(), 0);
		auto nodesBegin = &nodes.data()->get();
		m_rrManager->makeRRInstances(rrInstances, rrShapeCache, m_assetManager, nodesBegin, nodesBegin+nodes.size(), ids.data());
		m_rrManager->attachInstances(rrInstances.begin(), rrInstances.end());
	}
	
	auto renderSize = m_driver->getScreenSize();
	if (globalMeta && globalMeta->sensors.size())
	{
		const auto& sensor = globalMeta->sensors.front();
		const auto& film = sensor.film;
		renderSize.set(film.cropWidth,film.cropHeight);
	}

	m_depth = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_D32_SFLOAT);
	m_albedo = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_A2B10G10R10_UNORM_PACK32);
	m_normals = m_driver->createGPUTexture(ITexture::ETT_2D, &renderSize.Width, 1, EF_R16G16_SNORM);

	m_colorBuffer = m_driver->addFrameBuffer();
	m_colorBuffer->attach(EFAP_COLOR_ATTACHMENT0, m_albedo.get());

	m_gbuffer = m_driver->addFrameBuffer();
	m_gbuffer->attach(EFAP_DEPTH_ATTACHMENT, m_depth.get());
	m_gbuffer->attach(EFAP_COLOR_ATTACHMENT0, m_albedo.get());
	m_gbuffer->attach(EFAP_COLOR_ATTACHMENT1, m_normals.get());

	//
#define WORK_GROUP_DIM 16u
	m_workGroupCount[0] = (renderSize.Width+WORK_GROUP_DIM-1)/WORK_GROUP_DIM;
	m_workGroupCount[1] = (renderSize.Height+WORK_GROUP_DIM-1)/WORK_GROUP_DIM;
	uint32_t pixelCount = renderSize.Width*renderSize.Height;

	auto raygenBufferSize = static_cast<size_t>(pixelCount)*sizeof(::RadeonRays::ray);
	assert(raygenBufferSize<=rayBufferSize);
	auto shadowBufferSize = static_cast<size_t>(pixelCount)*sizeof(int32_t);
	assert(shadowBufferSize<=rayBufferSize);
	m_samplesPerDispatch = rayBufferSize/(raygenBufferSize+shadowBufferSize);
	assert(m_samplesPerDispatch >= 2u);

	raygenBufferSize *= m_samplesPerDispatch;
	m_rayBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDownStreamingGPUBufferOnDedMem(raygenBufferSize), core::dont_grab);

	shadowBufferSize *= m_samplesPerDispatch;
	m_intersectionBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createDownStreamingGPUBufferOnDedMem(shadowBufferSize),core::dont_grab);

	m_rayCount = m_samplesPerDispatch*pixelCount;
	m_rayCountBuffer = core::smart_refctd_ptr<IGPUBuffer>(m_driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uint32_t),&m_rayCount), core::dont_grab);

	m_rayBufferAsRR = m_rrManager->linkBuffer(m_rayBuffer.get(), CL_MEM_READ_WRITE);
	m_intersectionBufferAsRR = m_rrManager->linkBuffer(m_intersectionBuffer.get(), CL_MEM_READ_WRITE); // should it be write only?
	m_rayCountBufferAsRR = m_rrManager->linkBuffer(m_rayCountBuffer.get(), CL_MEM_READ_ONLY);

	const cl_mem clObjects[] = { m_rayCountBufferAsRR.second };
	auto objCount = sizeof(clObjects)/sizeof(cl_mem);
	clEnqueueAcquireGLObjects(m_rrManager->getCLCommandQueue(), objCount, clObjects, 0u, nullptr, nullptr);
}


void Renderer::deinit()
{
	auto commandQueue = m_rrManager->getCLCommandQueue();
	clFinish(commandQueue);

	glFinish();

	// release OpenCL objects and wait for OpenCL to finish
	const cl_mem clObjects[] = { m_rayCountBufferAsRR.second };
	auto objCount = sizeof(clObjects) / sizeof(cl_mem);
	clEnqueueReleaseGLObjects(commandQueue, objCount, clObjects, 1u, nullptr, nullptr);
	clFlush(commandQueue);
	clFinish(commandQueue);

	// start deleting objects
	if (m_colorBuffer)
	{
		m_driver->removeFrameBuffer(m_colorBuffer);
		m_colorBuffer = nullptr;
	}
	if (m_gbuffer)
	{
		m_driver->removeFrameBuffer(m_gbuffer);
		m_gbuffer = nullptr;
	}

	m_depth = m_albedo = m_normals = nullptr;

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
	m_samplesPerDispatch = 0u;
	m_rayCount = 0u;
	m_workGroupCount[0] = m_workGroupCount[1] = 0u;

	for (auto& node : nodes)
		node->remove();
	nodes.clear();
	sceneBound = core::aabbox3df(FLT_MAX,FLT_MAX,FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX);
	m_rrManager->deleteShapes(rrShapeCache.begin(),rrShapeCache.end());
	rrShapeCache.clear();
	m_rrManager->detachInstances(rrInstances.begin(),rrInstances.end());
	m_rrManager->deleteInstances(rrInstances.begin(),rrInstances.end());
	rrInstances.clear();
}


void Renderer::render()
{
	m_driver->setRenderTarget(m_gbuffer);
	{ // clear
		m_driver->clearZBuffer();
		float zero[4] = { 0.f,0.f,0.f,0.f };
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT0, zero);
		m_driver->clearColorBuffer(EFAP_COLOR_ATTACHMENT1, zero);
	}

	//! This animates (moves) the camera and sets the transforms
	//! Also draws the meshbuffer
	m_smgr->drawAll();

	auto rSize = m_depth->getSize();
	{
		GLint prevProgram;
		glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);

		STextureSamplingParams params;
		params.MaxFilter = ETFT_NEAREST_NEARESTMIP;
		params.MinFilter = ETFT_NEAREST_NEARESTMIP;
		params.UseMipmaps = 0;

		const COpenGLDriver::SAuxContext* foundConst = static_cast<COpenGLDriver*>(m_driver)->getThreadContext();
		COpenGLDriver::SAuxContext* found = const_cast<COpenGLDriver::SAuxContext*>(foundConst);
		found->setActiveTexture(0, core::smart_refctd_ptr(m_depth), params);
		found->setActiveTexture(1, core::smart_refctd_ptr(m_normals), params);


		COpenGLExtensionHandler::extGlUseProgram(m_raygenProgram);

		const COpenGLBuffer* buffers[] = { static_cast<const COpenGLBuffer*>(m_rayBuffer.get()) };
		ptrdiff_t offsets[] = { 0 };
		ptrdiff_t sizes[] = { m_rayBuffer->getSize() };
		found->setActiveSSBO(0, 1, buffers, offsets, sizes);

		{
			auto camera = m_smgr->getActiveCamera();
			auto camPos = camera->getAbsolutePosition();
			COpenGLExtensionHandler::pGlProgramUniform3fv(m_raygenProgram, 0, 1, &camPos.X);

			auto invViewProj = m_driver->getTransform(video::EPTS_PROJ_VIEW_INVERSE);
			COpenGLExtensionHandler::pGlProgramUniformMatrix4fv(m_raygenProgram, 1, 1, true, invViewProj.pointer());

			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_raygenProgram, 2, 1, rSize);

			uint32_t uSamples_ImageWidthSamples[2] = {m_samplesPerDispatch,rSize[0]*m_samplesPerDispatch};
			COpenGLExtensionHandler::pGlProgramUniform2uiv(m_raygenProgram, 3, 1, uSamples_ImageWidthSamples);

			float uImageSize2Rcp[4] = {2.f/static_cast<float>(rSize[0]),-2.f/static_cast<float>(rSize[1]),0.5f/static_cast<float>(rSize[0])-1.f,1.f-0.5f/static_cast<float>(rSize[1])};
			COpenGLExtensionHandler::pGlProgramUniform4fv(m_raygenProgram, 4, 1, uImageSize2Rcp);
		}

		COpenGLExtensionHandler::pGlDispatchCompute(m_workGroupCount[0], m_workGroupCount[1], 1);

		COpenGLExtensionHandler::extGlUseProgram(prevProgram);
		
		// probably wise to flush all caches
		COpenGLExtensionHandler::pGlMemoryBarrier(GL_ALL_BARRIER_BITS);
	}

	m_rrManager->update(rrInstances);
	if (m_rrManager->hasImplicitCL2GLSync())
		glFlush();
	else
		glFinish();

	auto commandQueue = m_rrManager->getCLCommandQueue();
	{
		const cl_mem clObjects[] = {m_rayBufferAsRR.second,m_intersectionBufferAsRR.second};
		auto objCount = sizeof(clObjects)/sizeof(cl_mem);

		cl_event acquired = nullptr;
		clEnqueueAcquireGLObjects(commandQueue,objCount,clObjects,0u,nullptr,&acquired);

		clEnqueueWaitForEvents(commandQueue,1u,&acquired);
		m_rrManager->getRadeonRaysAPI()->QueryOcclusion(m_rayBufferAsRR.first,m_rayCountBufferAsRR.first,m_rayCount,m_intersectionBufferAsRR.first,nullptr,nullptr);
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


		auto memory = m_intersectionBuffer->getBoundMemory();

		IDriverMemoryAllocation::MappedMemoryRange range(memory,0,rSize[0]*rSize[1]*sizeof(int32_t));
		auto intersections = reinterpret_cast<int32_t*>(memory->mapMemoryRange(IDriverMemoryAllocation::EMCAF_READ, range.range));

		constexpr uint32_t subsample = 2u;
		core::vector<uint32_t> data;
		for (int32_t y=0u; y<rSize[1]; y+=subsample)
		for (int32_t x=0u; x<rSize[0]; x+=subsample)
		{
			auto& intersection = *intersections;
			data.push_back(intersection<0 ? 0xffffffffu:0u);
			intersections += m_samplesPerDispatch*subsample*subsample;
		}
		uint32_t mini[3] = { 0,0,0 };
		uint32_t maxi[3] = { rSize[0]/subsample,rSize[1]/subsample,1 };
		m_albedo->updateSubRegion(asset::EF_R8G8B8A8_SRGB, data.data(), mini, maxi);
		memory->unmapMemory();
	}
}



#if 0
bool CToneMapper::CalculateFrameExposureFactors(IGPUBuffer* outBuffer, IGPUBuffer* uniformBuffer, core::smart_refctd_ptr<ITexture>&& inputTexture)
{
    bool highRes = false;
    if (!inputTexture)
        return false;

    COpenGLTexture* asGlTex = dynamic_cast<COpenGLTexture*>(inputTexture.get());
    if (asGlTex->getOpenGLTextureType()!=GL_TEXTURE_2D)
        return false;

    GLint prevProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProgram);


#ifdef PROFILE_TONEMAPPER
    IQueryObject* timeQuery = m_driver->createElapsedTimeQuery();
    m_driver->beginQuery(timeQuery);
#endif // PROFILE_TONEMAPPER

    STextureSamplingParams params;
    params.MaxFilter = ETFT_LINEAR_NO_MIP;
    params.MinFilter = ETFT_LINEAR_NO_MIP;
    params.UseMipmaps = 0;

    const COpenGLDriver::SAuxContext* foundConst = static_cast<COpenGLDriver*>(m_driver)->getThreadContext();
    COpenGLDriver::SAuxContext* found = const_cast<COpenGLDriver::SAuxContext*>(foundConst);
    found->setActiveTexture(0,std::move(inputTexture),params);


    COpenGLExtensionHandler::extGlUseProgram(m_histogramProgram);

    const COpenGLBuffer* buffers[2] = {static_cast<const COpenGLBuffer*>(m_histogramBuffer),static_cast<const COpenGLBuffer*>(outBuffer)};
    ptrdiff_t offsets[2] = {0,0};
    ptrdiff_t sizes[2] = {m_histogramBuffer->getSize(),outBuffer->getSize()};
    found->setActiveSSBO(0,2,buffers,offsets,sizes);

    buffers[0] = static_cast<const COpenGLBuffer*>(uniformBuffer);
    sizes[0] = uniformBuffer->getSize();
    found->setActiveUBO(0,1,buffers,offsets,sizes);

    COpenGLExtensionHandler::pGlDispatchCompute(m_workGroupCount[0],m_workGroupCount[1],1);
    COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    COpenGLExtensionHandler::extGlUseProgram(m_autoExpParamProgram);
    COpenGLExtensionHandler::pGlDispatchCompute(1,1, 1);


    COpenGLExtensionHandler::extGlUseProgram(prevProgram);
    COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

#ifdef PROFILE_TONEMAPPER
    m_driver->endQuery(timeQuery);
    uint32_t timeTaken=0;
    timeQuery->getQueryResult(&timeTaken);
    os::Printer::log("irr::ext::AutoExposure CS Time Taken:", std::to_string(timeTaken).c_str(),ELL_ERROR);
#endif // PROFILE_TONEMAPPER

    return true;
}

#endif
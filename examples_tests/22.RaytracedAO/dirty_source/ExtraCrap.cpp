#include "ExtraCrap.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;


const std::string raygenShaderExtensions = R"======(
#version 430 core
#extension GL_ARB_gpu_shader_int64 : require
#extension ARB_ballot : require
)======";

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
	/*
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
	*/
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


inline GLuint createComputeShader(const std::string& source)
{
    GLuint program = video::COpenGLExtensionHandler::extGlCreateProgram();
	GLuint cs = video::COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

	const char* tmp = source.c_str();
	video::COpenGLExtensionHandler::extGlShaderSource(cs, 1, const_cast<const char**>(&tmp), NULL);
	video::COpenGLExtensionHandler::extGlCompileShader(cs);

	// check for compilation errors
    GLint success;
    GLchar infoLog[0x200];
    video::COpenGLExtensionHandler::extGlGetShaderiv(cs, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        video::COpenGLExtensionHandler::extGlGetShaderInfoLog(cs, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS COMPILATION ERROR:\n", infoLog,ELL_ERROR);
        video::COpenGLExtensionHandler::extGlDeleteShader(cs);
        video::COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
	}

	video::COpenGLExtensionHandler::extGlAttachShader(program, cs);
	video::COpenGLExtensionHandler::extGlLinkProgram(program);

	//check linking errors
	success = 0;
    video::COpenGLExtensionHandler::extGlGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        video::COpenGLExtensionHandler::extGlGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
        os::Printer::log("CS LINK ERROR:\n", infoLog,ELL_ERROR);
        video::COpenGLExtensionHandler::extGlDeleteShader(cs);
        video::COpenGLExtensionHandler::extGlDeleteProgram(program);
        return 0;
    }

	return program;
}


Renderer::Renderer(video::IVideoDriver* _driver)
{
	m_raygenProgram = createComputeShader(
		raygenShaderExtensions+
	);
}
#if 0
CToneMapper* CToneMapper::instantiateTonemapper(video::IVideoDriver* _driver,
                                                const io::path& firstPassShaderFileName,
                                                const io::path& secondPassShaderFileName,
                                                const size_t& inputTexScaleOff, const size_t& percentilesOff, const size_t& outputOff)
{
    //! For Vulkan http://vulkan-spec-chunked.ahcox.com/ch09s07.html
    char* header = new char[sizeof(shaderHeaderDefines)+9*10+1];
    sprintf(header,shaderHeaderDefines,MIN_HISTOGRAM_RAW16F_AS_UINT,MAX_HISTOGRAM_RAW16F_AS_UINT,
            HISTOGRAM_POT2_RAW16F_BIN_SIZE,_BIN_COUNT_,_BIN_COUNT_,SUBCELL_SIZE,GLOBAL_REPLICATION,
            inputTexScaleOff/sizeof(core::vectorSIMDf),percentilesOff/sizeof(float),outputOff/sizeof(float));


    GLuint histoProgram = createComputeShader(firstPassShaderFileName,header);
    if (!histoProgram)
    {
        return NULL;
    }
    GLuint aexpPProgram = createComputeShader(secondPassShaderFileName,header);
    if (!aexpPProgram)
    {
        return NULL;
    }
    delete [] header;

    return new CToneMapper(_driver,histoProgram,aexpPProgram);
}


CToneMapper::CToneMapper(video::IVideoDriver* _driver, const uint32_t& _histoProgram, const uint32_t& _autoExpProgram)
                        : m_driver(_driver), m_histogramProgram(_histoProgram), m_autoExpParamProgram(_autoExpProgram)
{
    m_totalThreadCount[0] = 512;
    m_totalThreadCount[1] = 512;
    m_workGroupCount[0] = m_totalThreadCount[0]/SUBCELL_SIZE;
    m_workGroupCount[1] = m_totalThreadCount[1]/SUBCELL_SIZE;

    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.size = GLOBAL_REPLICATION*_BIN_COUNT_*sizeof(uint32_t);
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    m_histogramBuffer = m_driver->createGPUBufferOnDedMem(reqs);
}

CToneMapper::~CToneMapper()
{
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_histogramProgram);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_autoExpParamProgram);

    m_histogramBuffer->drop();
}

//#define PROFILE_TONEMAPPER

bool CToneMapper::CalculateFrameExposureFactors(video::IGPUBuffer* outBuffer, video::IGPUBuffer* uniformBuffer, core::smart_refctd_ptr<video::ITexture>&& inputTexture)
{
    bool highRes = false;
    if (!inputTexture)
        return false;

    video::COpenGLTexture* asGlTex = dynamic_cast<video::COpenGLTexture*>(inputTexture.get());
    if (asGlTex->getOpenGLTextureType()!=GL_TEXTURE_2D)
        return false;

    GLint prevProgram;
    glGetIntegerv(GL_CURRENT_PROGRAM,&prevProgram);


#ifdef PROFILE_TONEMAPPER
    video::IQueryObject* timeQuery = m_driver->createElapsedTimeQuery();
    m_driver->beginQuery(timeQuery);
#endif // PROFILE_TONEMAPPER

    video::STextureSamplingParams params;
    params.MaxFilter = video::ETFT_LINEAR_NO_MIP;
    params.MinFilter = video::ETFT_LINEAR_NO_MIP;
    params.UseMipmaps = 0;

    const video::COpenGLDriver::SAuxContext* foundConst = static_cast<video::COpenGLDriver*>(m_driver)->getThreadContext();
    video::COpenGLDriver::SAuxContext* found = const_cast<video::COpenGLDriver::SAuxContext*>(foundConst);
    found->setActiveTexture(0,std::move(inputTexture),params);


    video::COpenGLExtensionHandler::extGlUseProgram(m_histogramProgram);

    const video::COpenGLBuffer* buffers[2] = {static_cast<const video::COpenGLBuffer*>(m_histogramBuffer),static_cast<const video::COpenGLBuffer*>(outBuffer)};
    ptrdiff_t offsets[2] = {0,0};
    ptrdiff_t sizes[2] = {m_histogramBuffer->getSize(),outBuffer->getSize()};
    found->setActiveSSBO(0,2,buffers,offsets,sizes);

    buffers[0] = static_cast<const video::COpenGLBuffer*>(uniformBuffer);
    sizes[0] = uniformBuffer->getSize();
    found->setActiveUBO(0,1,buffers,offsets,sizes);

    video::COpenGLExtensionHandler::pGlDispatchCompute(m_workGroupCount[0],m_workGroupCount[1],1);
    video::COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


    video::COpenGLExtensionHandler::extGlUseProgram(m_autoExpParamProgram);
    video::COpenGLExtensionHandler::pGlDispatchCompute(1,1, 1);


    video::COpenGLExtensionHandler::extGlUseProgram(prevProgram);
    video::COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

#ifdef PROFILE_TONEMAPPER
    m_driver->endQuery(timeQuery);
    uint32_t timeTaken=0;
    timeQuery->getQueryResult(&timeTaken);
    os::Printer::log("irr::ext::AutoExposure CS Time Taken:", std::to_string(timeTaken).c_str(),ELL_ERROR);
#endif // PROFILE_TONEMAPPER

    return true;
}

#endif
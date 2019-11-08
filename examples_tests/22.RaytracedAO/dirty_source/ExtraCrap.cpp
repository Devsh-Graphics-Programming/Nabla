#include "ExtraCrap.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;


const char shaderHeaderDefines[] = R"======(
#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D depth;
layout(binding = 1) uniform sampler2D normals;

layout(binding = 0, rgba32f) uniform writeonly image2D out_img;

shared float interleavedData[3u*4u*64u];

vec3 decode(in float theta, float phi)
{
	float ang = phi*kPI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-theta*theta), theta);
}

uvec2 morton2xy_4bit(in uint id)
{
	return id;
}
uint xy2morton_4bit(in uvec2 coord)
{
//1111
	return id;
}

void main()
{
	uint subgroupInvocationIndex = gl_LocalInvocationIndex&63u;
	uint wideSubgroupID = gl_LocalInvocationIndex>>6u;

	uvec2 workGroupCoord = gl_WorkGroupID.uv*gl_WorkGroupSize.uv;
	vec2 sharedUV = (vec2(workGroupCoord+uvec2(8,8))+vec2(0.5))/textureSize();
	ivec2 offset = (ivec2(subgroupInvocationIndex&7u,subgroupInvocationIndex>>3u)<<1)-ivec2(8,8);
	vec4 data;
	switch (wideSubgroupID)
	{
		case 0u:
			data = textureGatherOffset(depth,sharedUV,offset);
			break;
		case 1u:
			data = textureGatherOffset(normals,sharedUV,offset,0);
			break;
		case 2u:
			data = textureGatherOffset(normals,sharedUV,offset,1);
			break;
		default:
			break;
	}
	if (wideSubgroupID!=3u)
	for (uint i=0; i<4u; i++)
		interleavedData[wideSubgroupID*256u+i*64u+subgroupInvocationIndex] = data[i];
	barrier();
	memoryBarrierShared();


	float depth = interleavedData[gl_LocalInvocationIndex];
	float phi = interleavedData[gl_LocalInvocationIndex+256u];
	float theta = interleavedData[gl_LocalInvocationIndex+512u];

	ivec2 outputLocation = ivec2(workGroupCoord)+gl_LocationInvocationID.xy;
	//outputLocation =
	vec3 normal = decode(theta,phi);

	imageStore(out_img,outputLocation,vec4(normal,1.0));
}

)======";


inline GLuint createComputeShader(const io::path& filename, const std::string& header)
{
    FILE* fp = fopen(filename.c_str(),"r");
    fseek(fp, 0, SEEK_END); // seek to end of file
    int32_t size = ftell(fp); // get current file pointer
    std::string modifiedSrc;
    modifiedSrc.resize(size);
    fseek(fp, 0, SEEK_SET); // seek back to beginning of file

    fread(const_cast<char*>(modifiedSrc.data()),size,1,fp);
    fclose(fp);

    modifiedSrc = header+modifiedSrc;


    GLuint program = video::COpenGLExtensionHandler::extGlCreateProgram();
	GLuint cs = video::COpenGLExtensionHandler::extGlCreateShader(GL_COMPUTE_SHADER);

	const char* tmp = modifiedSrc.c_str();
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
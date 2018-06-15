#include "../ext/AutoExposure/CToneMapper.h"

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLDriver.h"

using namespace irr;
using namespace ext;
using namespace AutoExposure;


#define MIN_HISTOGRAM_RAW16F_AS_UINT    13312 // Float16Compressor::compress(MIN_HISTOGRAM_VAL)
#define MAX_HISTOGRAM_RAW16F_AS_UINT    21503 // Float16Compressor::compress(MAX_HISTOGRAM_VAL)-1

//don't touch this, tis optimized
constexpr uint32_t _BIN_COUNT_ = 256;
constexpr uint32_t GLOBAL_REPLICATION = 4;

//! related constants
constexpr uint32_t HISTOGRAM_POT2_RAW16F_BIN_SIZE = 5; //would be cool to have a constexpr function to calculate this
// the ix of the bin for a value is calculated by ix = (float16BitsAsUint(value)-MIN_HISTOGRAM_RAW16F_AS_UINT)>>HISTOGRAM_POT2_RAW16F_BIN_SIZE

//checks
static_assert(_BIN_COUNT_==SUBCELL_SIZE*SUBCELL_SIZE, "Super Optimizations required BIN_COUNT==LOCAL_THREADS code broken otherwise");
static_assert((MAX_HISTOGRAM_RAW16F_AS_UINT-MIN_HISTOGRAM_RAW16F_AS_UINT+1)==(_BIN_COUNT_<<HISTOGRAM_POT2_RAW16F_BIN_SIZE), "Mismatched Histogram Parameters");


const char shaderHeaderDefines[] = R"======(
#version 430 core

//don't touch this
//histogram algo optimized for bin-count = 256
//unless you want to shift both by an integer constant
#define MIN_HISTOGRAM_RAW16F_AS_UINT    %d // Float16Compressor::compress(MIN_HISTOGRAM_VAL)
#define MAX_HISTOGRAM_RAW16F_AS_UINT    %d // Float16Compressor::compress(MAX_HISTOGRAM_VAL)-1
// the ix of the bin for a value is calculated by ix = (float16BitsAsUint(value)-MIN_HISTOGRAM_RAW16F_AS_UINT)>>HISTOGRAM_POT2_RAW16F_BIN_SIZE
#define HISTOGRAM_POT2_RAW16F_BIN_SIZE  %du

#define BIN_COUNT %d
#define BIN_COUNTu %du
#define SUBCELL_SIZE %du
#define GLOBAL_REPLICATION %du

#define TEXSCALE_UBO_OFFSET %du
#define PERCENTILE_UBO_OFFSET %du
#define OUTPUT_UBO_OFFSET %du


#define kLumaConvertCoeff vec3(0.299, 0.587, 0.114)

#define BURNOUT_THRESH_EXP 3.75
#define MIDDLE_GREY_EXP 0.25
#define P_EXP (-MIDDLE_GREY_EXP/0.5)



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

    m_histogramBuffer = m_driver->createGPUBuffer(GLOBAL_REPLICATION*_BIN_COUNT_*sizeof(uint32_t),NULL);
}

CToneMapper::~CToneMapper()
{
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_histogramProgram);
    video::COpenGLExtensionHandler::extGlDeleteProgram(m_autoExpParamProgram);

    m_histogramBuffer->drop();
}

//#define PROFILE_TONEMAPPER

bool CToneMapper::CalculateFrameExposureFactors(video::IGPUBuffer* outBuffer, video::IGPUBuffer* uniformBuffer, video::ITexture* inputTexture)
{
    bool highRes = false;
    if (!inputTexture)
        return false;

    video::COpenGLTexture* asGlTex = dynamic_cast<video::COpenGLTexture*>(inputTexture);
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
    found->setActiveTexture(0,inputTexture,params);


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
    found->setActiveSSBO(0,2,nullptr,nullptr,nullptr);
    found->setActiveUBO(0,1,nullptr,nullptr,nullptr);
    video::COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT);

#ifdef PROFILE_TONEMAPPER
    m_driver->endQuery(timeQuery);
    uint32_t timeTaken=0;
    timeQuery->getQueryResult(&timeTaken);
    os::Printer::log("irr::ext::AutoExposure CS Time Taken:", std::to_string(timeTaken).c_str(),ELL_ERROR);
#endif // PROFILE_TONEMAPPER

    return true;
}

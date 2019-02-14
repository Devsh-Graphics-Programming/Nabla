#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>

#include "../source/Irrlicht/COpenGLDriver.h"
#include "COpenGLStateManager.h"


using namespace irr;
using namespace core;


#define OPENGL_DEBUG

bool quit = false;

//!Same As Last Example
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver()
	{
	}

	bool OnEvent(const SEvent& event)
	{
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case irr::KEY_KEY_Q: // switch wire frame mode
                quit = true;
                return true;
            default:
                break;
            }
        }

		return false;
	}

private:
};


#ifdef OPENGL_DEBUG
void APIENTRY openGLCBFunc(GLenum source, GLenum type, GLuint id, GLenum severity,
                           GLsizei length, const GLchar* message, const void* userParam)
{
    core::stringc outStr;
    switch (severity)
    {
        //case GL_DEBUG_SEVERITY_HIGH:
        case GL_DEBUG_SEVERITY_HIGH_ARB:
            outStr = "[H.I.G.H]";
            break;
        //case GL_DEBUG_SEVERITY_MEDIUM:
        case GL_DEBUG_SEVERITY_MEDIUM_ARB:
            outStr = "[MEDIUM]";
            break;
        //case GL_DEBUG_SEVERITY_LOW:
        case GL_DEBUG_SEVERITY_LOW_ARB:
            outStr = "[  LOW  ]";
            break;
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            outStr = "[  LOW  ]";
            break;
        default:
            outStr = "[UNKNOWN]";
            break;
    }
    switch (source)
    {
        //case GL_DEBUG_SOURCE_API:
        case GL_DEBUG_SOURCE_API_ARB:
            switch (type)
            {
                //case GL_DEBUG_TYPE_ERROR:
                case GL_DEBUG_TYPE_ERROR_ARB:
                    outStr += "[OPENGL  API ERROR]\t\t";
                    break;
                //case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
                    outStr += "[OPENGL  DEPRECATED]\t\t";
                    break;
                //case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
                    outStr += "[OPENGL   UNDEFINED]\t\t";
                    break;
                //case GL_DEBUG_TYPE_PORTABILITY:
                case GL_DEBUG_TYPE_PORTABILITY_ARB:
                    outStr += "[OPENGL PORTABILITY]\t\t";
                    break;
                //case GL_DEBUG_TYPE_PERFORMANCE:
                case GL_DEBUG_TYPE_PERFORMANCE_ARB:
                    outStr += "[OPENGL PERFORMANCE]\t\t";
                    break;
                default:
                    outStr += "[OPENGL       OTHER]\t\t";
                    ///return;
                    break;
            }
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_SHADER_COMPILER:
        case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
            outStr += "[SHADER]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
            outStr += "[WINDOW SYS]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_THIRD_PARTY:
        case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
            outStr += "[3RDPARTY]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_APPLICATION:
        case GL_DEBUG_SOURCE_APPLICATION_ARB:
            outStr += "[APP]\t\t";
            outStr += message;
            break;
        //case GL_DEBUG_SOURCE_OTHER:
        case GL_DEBUG_SOURCE_OTHER_ARB:
            outStr += "[OTHER]\t\t";
            outStr += message;
            break;
        default:
            break;
    }
    outStr += "\n";
    printf("%s",outStr.c_str());
}
#endif // OPENGL_DEBUG


const char* screenquad_vert = R"===(
#version 330 core
layout(location = 0) in vec3 vPos;

void main()
{
    gl_Position = vec4(vPos,1.0);
}
)===";



std::string loadShaderFile(io::IFileSystem* fsys, const irr::io::path &filename, const std::string &headerAndExtensions)
{
    irr::io::IReadFile* file = fsys->createAndOpenFile(filename);

    size_t fileLen = file->getSize();

    std::string retval = headerAndExtensions;
    retval.resize(headerAndExtensions.size()+fileLen);

    file->read(&(*(retval.begin()+headerAndExtensions.size())),fileLen);

    file->drop();

    return retval;
}


#include "irr/irrpack.h"
struct ScreenQuadVertexStruct
{
    float Pos[3];
    uint8_t TexCoord[2];
} PACK_STRUCT;
#include "irr/irrunpack.h"






int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon


	IrrlichtDevice* device = createDeviceEx(params);
	if (device == 0)
		return 1; // could not create selected driver.

	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();
#ifdef OPENGL_DEBUG
    if (video::COpenGLExtensionHandler::FeatureAvailable[video::COpenGLExtensionHandler::IRR_KHR_debug])
    {
        glEnable(GL_DEBUG_OUTPUT);
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        video::COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        video::COpenGLExtensionHandler::pGlDebugMessageCallback(openGLCBFunc,NULL);
    }
    else
    {
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        video::COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        video::COpenGLExtensionHandler::pGlDebugMessageCallbackARB(openGLCBFunc,NULL);
    }
#endif // OPENGL_DEBUG


	//! Our atomic resolution (amount of atomic ops to do)
	uint32_t texSize[2];
	std::cout << "Give the Horizontal Pixel Count:";
	std::cin >> texSize[0];
	std::cout << "Give the Vertical Pixel Count:";
	std::cin >> texSize[1];

	if ((texSize[0]%16!=0)&&(texSize[1]%16!=0) && texSize[0]<16&&texSize[0]>4096 && texSize[1]<16&&texSize[1]>4096)
    {
        texSize[0] = 1920;
        texSize[1] = 1080;
    }

    //
    constexpr auto methodCount = 8;
    constexpr auto methodMax = methodCount-1;

    char methodChar = '0';
	std::cout << "Choose atomic op partitioning method: (0-" << methodMax << ")";
	std::cin >> methodChar;
	uint32_t method = methodChar-'0';
	if (method>methodMax)
        method = 0;

    std::string shaderHeader = R"===(#version 430 core
#define kHalfLog2CUCount 2u // will need changing at some point
#define kSqrtCUCount (1u<<kHalfLog2CUCount)
#define kSqrtCUCountMask (kSqrtCUCount-1u)

#define BINNING_METHOD )===";
    shaderHeader += std::to_string(method);
    shaderHeader += '\n';

    if (video::COpenGLExtensionHandler::IsIntelGPU)
        shaderHeader += "#define INTEL\n";

    size_t bufSize = 1;
    switch (method)
    {
        case 1:
        case 2:
        case 3:
            bufSize = 256;
            break;
        case 4:
            {
                int32_t smCount = 1;
                glGetIntegerv(GL_SM_COUNT_NV,&smCount);
                bufSize = 32*smCount;
            }
            break;
        case 5:
            bufSize = 64;
            break;
        case 6:
            {
                bufSize = 32;
            }
            break;
        default:
            break;
    }
    printf("BuffSize %d\n",bufSize);


    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.size = 1024*1024;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    video::IGPUBuffer* atomicSSBOBuf = driver->createGPUBufferOnDedMem(reqs,false);
/**
	//
	char yesOrNo = 'n';
	std::cout << "Benchmark compute shader? (y/n)";
	std::cin >> yesOrNo;
	if (yesOrNo=='y'||yesOrNo=='Y')
    {
        printf("Benchmarking Compute %d %d %d\n",method,texSize[0],texSize[1]);

        //! No time now
        printf("UNIMPLEMENTED!!!\n");
        exit(2);
    }
    else*/
    {
        printf("Benchmarking Pixel %d %d %d\n",method,texSize[0],texSize[1]);

        video::ITexture* depthBuffer = driver->createGPUTexture(video::ITexture::ETT_2D,texSize,1,asset::EF_D32_SFLOAT);


        video::IGPUMeshBuffer* screenQuadMeshBuffer = new video::IGPUMeshBuffer();
        {
            video::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
            screenQuadMeshBuffer->setMeshDataAndFormat(desc);
            desc->drop();

            ScreenQuadVertexStruct vertices[4];
            vertices[0].Pos[0] = -1.f;
            vertices[0].Pos[1] = -1.f;
            vertices[0].Pos[2] = 0.5f;
            vertices[0].TexCoord[0] = 0;
            vertices[0].TexCoord[1] = 0;
            vertices[1].Pos[0] = 1.f;
            vertices[1].Pos[1] = -1.f;
            vertices[1].Pos[2] = 0.5f;
            vertices[1].TexCoord[0] = 1;
            vertices[1].TexCoord[1] = 0;
            vertices[2].Pos[0] = -1.f;
            vertices[2].Pos[1] = 1.f;
            vertices[2].Pos[2] = 0.5f;
            vertices[2].TexCoord[0] = 0;
            vertices[2].TexCoord[1] = 1;
            vertices[3].Pos[0] = 1.f;
            vertices[3].Pos[1] = 1.f;
            vertices[3].Pos[2] = 0.5f;
            vertices[3].TexCoord[0] = 1;
            vertices[3].TexCoord[1] = 1;

            uint16_t indices_indexed16[] = {0,1,2,2,1,3};

            video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.size = sizeof(vertices)+sizeof(indices_indexed16);
            reqs.vulkanReqs.alignment = 4;
            reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
            reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
            reqs.mappingCapability = video::IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS;
            reqs.prefersDedicatedAllocation = true;
            reqs.requiresDedicatedAllocation = true;
            video::IGPUBuffer* buff = driver->createGPUBufferOnDedMem(reqs,true);
            buff->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,sizeof(vertices)),vertices);
            buff->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(sizeof(vertices),sizeof(indices_indexed16)),indices_indexed16);

            desc->setVertexAttrBuffer(buff,asset::EVAI_ATTR0,asset::EF_R32G32B32_SFLOAT,sizeof(ScreenQuadVertexStruct),0);
            desc->setVertexAttrBuffer(buff,asset::EVAI_ATTR1,asset::EF_R8G8_UNORM,sizeof(ScreenQuadVertexStruct),12); //this time we used unnormalized
            desc->setIndexBuffer(buff);
            screenQuadMeshBuffer->setIndexBufferOffset(sizeof(vertices));
            screenQuadMeshBuffer->setIndexType(asset::EIT_16BIT);
            screenQuadMeshBuffer->setIndexCount(6);
            buff->drop();
        }

        video::SGPUMaterial postProcMaterial;
        //! First need to make a material other than default to be able to draw with custom shader
        postProcMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
        postProcMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
        postProcMaterial.ZWriteEnable = false; //! Why even write depth?
        {
            postProcMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterial(screenquad_vert,
                                                                                nullptr,nullptr,nullptr, //! No Geometry or Tessellation Shaders
                                                                                loadShaderFile(device->getFileSystem(),"../atomics.frag",shaderHeader).c_str());
        }

		driver->beginScene( false,false );
		{
            video::IFrameBuffer* fbo = driver->addFrameBuffer();
            fbo->attach(video::EFAP_DEPTH_ATTACHMENT,depthBuffer);
            driver->setRenderTarget(fbo,true);
            fbo->drop();
        }
        bool first = true;
        while(device->run()&&(!quit))
        {
            video::IQueryObject* timeQuery = driver->createElapsedTimeQuery();
            driver->beginQuery(timeQuery);
            for (size_t drawC=0; drawC<1000; drawC++)
            {
                const video::COpenGLDriver::SAuxContext* foundConst = static_cast<video::COpenGLDriver*>(driver)->getThreadContext();
                video::COpenGLDriver::SAuxContext* found = const_cast<video::COpenGLDriver::SAuxContext*>(foundConst);
                //set UBO
                {
                    const video::COpenGLBuffer* buffers[1] = {static_cast<const video::COpenGLBuffer*>(atomicSSBOBuf)};
                    ptrdiff_t offsets[1] = {0};
                    ptrdiff_t sizes[1] = {atomicSSBOBuf->getSize()};
                    found->setActiveSSBO(0,1,buffers,offsets,sizes);
                }
                driver->setMaterial(postProcMaterial);
                driver->drawMeshBuffer(screenQuadMeshBuffer);
                if (first)
                {
                    uint32_t var = 1024;
                    GLint currProg;
                    glGetIntegerv(GL_CURRENT_PROGRAM,&currProg);
                    video::COpenGLExtensionHandler::extGlProgramUniform1uiv(currProg,0,1,&var);
                    first = false;
                }
            }
            glFlush();
            driver->endQuery(timeQuery);
            uint32_t timeTaken=0;
            timeQuery->getQueryResult(&timeTaken);
            printf("Atomic Operations Time Taken: %d\n",timeTaken);
        }
        screenQuadMeshBuffer->drop();
    }

    driver->endScene();
	device->drop();

	return 0;
}

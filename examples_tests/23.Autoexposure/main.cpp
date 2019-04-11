#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>

#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"

#include "../source/Irrlicht/COpenGLDriver.h"
#include "COpenGLStateManager.h"

#include "../ext/AutoExposure/CToneMapper.h"

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



//! NEW, Uniform Buffer Objects!
struct PerFrameUniformBlock
{
    vectorSIMDf dynamicResolutionScale; //with 2 float padding
    struct {
        float dynResScale[2];
        uint32_t percentileSearchVals[2];
    } autoExposureInput;
    struct {
        float autoExposureParameters[2];
        uint32_t padding[2];
    } autoExposureOutput;
};



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

	//load 16bit float screencap dump and its dimensions
    core::dimension2du dynamicResolutionSize;
    FILE *fp = fopen("../../media/preExposureScreenDump.dims", "r+");
    if (fp)
    {
        int ret = fscanf(fp, "%d %d \t %d %d",
                         &params.WindowSize.Width, &params.WindowSize.Height,
                         &dynamicResolutionSize.Width, &dynamicResolutionSize.Height);
        //
        if(ret != 4)
        {
            printf("Couldn't get screendump sizes!\n");
            return 2;
        }
        fclose(fp);
    }
    else
        return 3;

    void* tmpLoadingMem = malloc(dynamicResolutionSize.Width*dynamicResolutionSize.Height*16);
    fp = fopen("../../media/preExposureScreenDump.rgba16f", "r+");
    if (!fp)
    {
        free(tmpLoadingMem);
        return 4;
    }
    fread(tmpLoadingMem,dynamicResolutionSize.Width*dynamicResolutionSize.Height*8,1,fp);


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

	//upload screendump to a texture
	video::ITexture* hdrTex = driver->createGPUTexture(video::ITexture::ETT_2D,&params.WindowSize.Width,1,asset::EF_R16G16B16A16_SFLOAT);
	uint32_t zeroArray[3] = {0,0,0};
	uint32_t subSize[3] = {dynamicResolutionSize.Width,dynamicResolutionSize.Height,1};
	hdrTex->updateSubRegion(asset::EF_R16G16B16A16_SFLOAT,tmpLoadingMem,zeroArray,subSize);
	free(tmpLoadingMem);


    video::IGPUMeshBuffer* fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(driver);

    video::SGPUMaterial postProcMaterial;
    //! First need to make a material other than default to be able to draw with custom shader
    postProcMaterial.BackfaceCulling = false; //! Triangles will be visible from both sides
    postProcMaterial.ZBuffer = video::ECFN_ALWAYS; //! Ignore Depth Test
    postProcMaterial.ZWriteEnable = false; //! Why even write depth?
    postProcMaterial.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../fullscreentri.vert",
                                                                        "","","", //! No Geometry or Tessellation Shaders
                                                                        "../postproc.frag",
                                                                        3,video::EMT_SOLID); //! 3 vertices per primitive (this is tessellation shader relevant only)
    postProcMaterial.setTexture(0,hdrTex);



    irr::ext::AutoExposure::CToneMapper* toneMapper = irr::ext::AutoExposure::CToneMapper::instantiateTonemapper(driver,
                                                                                                                 "../../../ext/AutoExposure/lumaHistogramFirstPass.comp",
                                                                                                                 "../../../ext/AutoExposure/lumaHistogramSecondPass.comp",
                                                                                                                 offsetof(PerFrameUniformBlock,autoExposureInput.dynResScale),
                                                                                                                 offsetof(PerFrameUniformBlock,autoExposureInput.percentileSearchVals),
                                                                                                                 offsetof(PerFrameUniformBlock,autoExposureOutput));

    PerFrameUniformBlock block;
    block.dynamicResolutionScale = vectorSIMDf(dynamicResolutionSize.Width,dynamicResolutionSize.Height,dynamicResolutionSize.Width,dynamicResolutionSize.Height);
    block.dynamicResolutionScale /= vectorSIMDf(params.WindowSize.Width,params.WindowSize.Height,params.WindowSize.Width,params.WindowSize.Height);

    toneMapper->setHistogramSamplingRate(block.autoExposureInput.dynResScale,block.autoExposureInput.percentileSearchVals, //out
                                         dynamicResolutionSize,block.dynamicResolutionScale.pointer); //in

    video::IGPUBuffer* frameUniformBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(block),&block);

	uint64_t lastFPSTime = 0;

	while(device->run()&&(!quit))
	{
		driver->beginScene( false,false );


        toneMapper->CalculateFrameExposureFactors(frameUniformBuffer,frameUniformBuffer,hdrTex);


        const video::COpenGLDriver::SAuxContext* foundConst = static_cast<video::COpenGLDriver*>(driver)->getThreadContext();
        video::COpenGLDriver::SAuxContext* found = const_cast<video::COpenGLDriver::SAuxContext*>(foundConst);
        //set UBO
        {
            const video::COpenGLBuffer* buffers[1] = {static_cast<const video::COpenGLBuffer*>(frameUniformBuffer)};
            ptrdiff_t offsets[1] = {0};
            ptrdiff_t sizes[1] = {frameUniformBuffer->getSize()};
            found->setActiveUBO(0,1,buffers,offsets,sizes);
        }
        driver->setMaterial(postProcMaterial);
        driver->drawMeshBuffer(fullScreenTriangle);

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Autoexposure Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	toneMapper->drop();
	fullScreenTriangle->drop();


    //create a screenshot
	video::IImage* screenshot = driver->createImage(asset::EF_B8G8R8A8_UNORM,params.WindowSize);
        video::COpenGLExtensionHandler::extGlNamedFramebufferReadBuffer(0,GL_FRONT_LEFT);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch=screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i=0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete [] tmpBuffer;
    }
    asset::CImageData* img = new asset::CImageData(screenshot);
    asset::IAssetWriter::SAssetWriteParams wparams(img);
    device->getAssetManager().writeAsset("screenshot.png", wparams);
    img->drop();
    screenshot->drop();

	device->drop();

	return 0;
}

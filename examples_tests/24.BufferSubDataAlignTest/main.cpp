#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <iostream>
#include <cstdio>

#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

using namespace irr;
using namespace core;


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



int main()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing


	IrrlichtDevice* device = createDeviceEx(params);
	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();
    if (video::COpenGLExtensionHandler::FeatureAvailable[video::COpenGLExtensionHandler::IRR_KHR_debug])
    {
        //glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        video::COpenGLExtensionHandler::pGlDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        video::COpenGLExtensionHandler::pGlDebugMessageCallback(openGLCBFunc,NULL);
    }
    else
    {
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
        video::COpenGLExtensionHandler::pGlDebugMessageControlARB(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,NULL,true);

        video::COpenGLExtensionHandler::pGlDebugMessageCallbackARB(openGLCBFunc,NULL);
    }


	const size_t pageSize = 4096;

	const uint8_t uinitBitVal = 0xfbu;
	//const uint8_t guardBitsVal = 0xfcu;
	const uint8_t setBitVal = 0x7cu;

	uint8_t clearInitBuffer[pageSize];
	for (size_t i=0; i<pageSize; i++)
        clearInitBuffer[i] = uinitBitVal;
	uint8_t setBuffer[pageSize];
	for (size_t i=0; i<pageSize; i++)
        setBuffer[i] = setBitVal;

    bool globalFail = false;
    const size_t testOffsetRange = 256;
    for (size_t startOffset=0; startOffset<testOffsetRange; startOffset++)
    for (size_t endOffset=0; endOffset<testOffsetRange; endOffset++)
	{
        video::IGPUBuffer* testBuffer = driver->createGPUBuffer(pageSize,clearInitBuffer,true);

        //little extra test : bind to UBO+SSBO
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_UNIFORM_BUFFER,0,1,&reinterpret_cast<video::COpenGLBuffer*>(testBuffer)->getOpenGLName());
        video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,&reinterpret_cast<video::COpenGLBuffer*>(testBuffer)->getOpenGLName());
        //comment out to test subUpdate while bound to various UBO+SSBO targets
        //video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_UNIFORM_BUFFER,0,1,NULL);
        //video::COpenGLExtensionHandler::extGlBindBuffersBase(GL_SHADER_STORAGE_BUFFER,0,1,NULL);

        //upload
        testBuffer->updateSubRange(startOffset,pageSize-(startOffset+endOffset),setBuffer);
        //get back
        uint8_t resultBuffer[pageSize];
        video::COpenGLExtensionHandler::extGlGetNamedBufferSubData(reinterpret_cast<video::COpenGLBuffer*>(testBuffer)->getOpenGLName(),0,pageSize,resultBuffer);
        //test
        bool fail=false;
        for (size_t j=0; j<pageSize; j++)
        {
            if (j<startOffset)
            {
                if (resultBuffer[j]!=uinitBitVal)
                {
                    fail = true;
                    break;
                }
            }
            else if (j>=pageSize-endOffset)
            {
                if (resultBuffer[j]!=uinitBitVal)
                {
                    fail = true;
                    break;
                }
            }
            else if (resultBuffer[j]!=setBitVal)
            {
                fail = true;
                break;
            }
        }

        testBuffer->drop();

        if (fail)
        {
            globalFail = true;
            printf("Failed with startOffset: %d and endOffset: %d\n",startOffset,endOffset);
        }
	}

	if (globalFail)
        printf("There were failures in SubBuffer updates, if you didn't see any \"[OPENGL ERROR ...\" messages then its a driver bug!\n");



	device->drop();

	return 0;
}

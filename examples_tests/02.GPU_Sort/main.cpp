#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLBuffer.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

using namespace irr;
using namespace core;



#define BIG_PRIMORIAL 116396280

#include "irr/irrpack.h"
template<size_t extraDataCnt = 1> // do not use more than 16
class ParticleStruct
{
    public:
        //Key is always guaranteed to be >=0.f
        float Key;
        uint32_t extraData[extraDataCnt];

        inline bool operator<(const ParticleStruct& other) const
        {
            return Key<other.Key;
        }
} PACK_STRUCT;
#include "irr/irrunpack.h"

int main()
{

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
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

    video::IVideoDriver* driver = device->getVideoDriver();


    srand(69012345u);
#define EXTRA_DATA_AMOUNT 4 //valid amounts are [1,16]
    const size_t elementSize = sizeof(ParticleStruct<EXTRA_DATA_AMOUNT>);
    const size_t elementCount = BIG_PRIMORIAL/elementSize;

    ParticleStruct<EXTRA_DATA_AMOUNT>* randomData = reinterpret_cast<ParticleStruct<EXTRA_DATA_AMOUNT>*>(malloc(sizeof(uint32_t)*BIG_PRIMORIAL));
    assert(BIG_PRIMORIAL%elementSize==0);

    for (size_t i=0; i<elementCount; i++)
        randomData[i].Key = rand();

    uint32_t bufferSize = elementCount*elementSize;
    //! GPU buffer allocated using ARB_buffer_storage, you can rewrite this part into pure OpenGL of your choosing
    video::COpenGLBuffer* buffer = dynamic_cast<video::COpenGLBuffer*>(driver->createDeviceLocalGPUBufferOnDedMem(bufferSize));
    uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
    uint32_t alignment = 8u;
    auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
    upStreamBuff->multi_place(1u,(const void* const*)&randomData,&offset,&bufferSize,&alignment);
    if (upStreamBuff->needsManualFlushOrInvalidate())
    {
        driver->flushMappedMemoryRanges({{upStreamBuff->getBuffer()->getBoundMemory(),offset,bufferSize}});
    }
    driver->copyBuffer(upStreamBuff->getBuffer(),buffer,offset,0u,bufferSize);
    upStreamBuff->multi_free(1u,&offset,&bufferSize,driver->placeFence());
    // buffer will later be dropped

    //! sort on CPU to compare results
    std::sort(randomData,randomData+elementCount);

    //! Reset default OpenGL state
	video::COpenGLState defaultState;
    video::COpenGLStateDiff diff = defaultState.getStateDiff(video::COpenGLState::collectGLState());
    video::executeGLDiff(diff);

    /*
    You now have a valid OpenGL context and a window with default global state
    You can use any OpenGL function as long as you load it from the OpenGL
    Most functions like glUseProgram are already loaded and are available
    Try use OpenGL 4.5 with Direct State Access functions

    For compute shaders you'll need to look inside COpenGLExtensionHandler.cpp
    for how OpenGL function pointers are loaded for GL 1.2+ functions.
    */
	//!=======================================START SETUP===========================================

	// create some shaders, texture buffer objects etc. etc.
	// bind UBOs, SSBOs or whatever

	//!========================================END SETUP============================================
    /*

    Below is the part that I will time, it will help to put it in a loop
    and sort 1000 times to get reliable timings for your own testing.
    */
    video::IQueryObject* gpuElapsedTime = driver->createElapsedTimeQuery();
    driver->beginQuery(gpuElapsedTime);
	//!=======================================START SORT============================================

    // good luck
    // buffer->getOpenGLName() is the GLuint handle to an opengl buffer allocated through glBufferStorage

	//!========================================END SORT=============================================
	driver->endQuery(gpuElapsedTime);

	uint64_t timingResult;
	gpuElapsedTime->getQueryResult(&timingResult);
	printf("Elapsed GPU Time %d us\n",timingResult/1000ull);


	ParticleStruct<EXTRA_DATA_AMOUNT>* resultData = reinterpret_cast<ParticleStruct<EXTRA_DATA_AMOUNT>*>(malloc(sizeof(uint32_t)*BIG_PRIMORIAL));
	video::COpenGLExtensionHandler::extGlGetNamedBufferSubData(buffer->getOpenGLName(),0,elementCount*elementSize,resultData);
    buffer->drop();

	bool allGood = true;
    for (size_t i=0; i<elementCount; i++)
    {
        if (strcmp((const char*)randomData,(const char*)resultData))
        {
            allGood = false;
            break;
        }
    }
    printf(allGood ? "SUCCESS!\n":"FAIL!\n");

    free(resultData);
    free(randomData);

	device->drop();

	return 0;
}

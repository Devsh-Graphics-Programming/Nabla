#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "driverChoice.h"

#include "IGPUTransientBuffer.h"

using namespace irr;
using namespace core;


#define kNumHardwareInstancesX 10
#define kNumHardwareInstancesY 20
#define kNumHardwareInstancesZ 30

#define kHardwareInstancesTOTAL (kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ)




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
                exit(0);
                return true;
            default:
                break;
            }
        }

		return false;
	}

private:
};



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

    size_t allocsPerFrame = 10000;
    size_t allocSize = 128;



	scene::ISceneManager* smgr = device->getSceneManager();
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


    video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
    reqs.vulkanReqs.size = 0x1000000u;
    reqs.vulkanReqs.alignment = 4;
    reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
    reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
    reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE|video::IDriverMemoryAllocation::EMCF_COHERENT;
    reqs.prefersDedicatedAllocation = true;
    reqs.requiresDedicatedAllocation = true;
    video::IGPUTransientBuffer* buffer = video::IGPUTransientBuffer::createTransientBuffer(driver,driver->createGPUBufferOnDedMem(reqs,false),false);

	uint64_t lastFPSTime = 0;

	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

		uint64_t startTime = device->getTimer()->getRealTime();
        for (size_t i=0; i<allocsPerFrame; i++)
        {
            size_t offset;
            #define ALIGNMENT 32
            if (buffer->Alloc(offset,allocSize,ALIGNMENT,video::IGPUTransientBuffer::EWP_DONT_WAIT)==video::IGPUTransientBuffer::EARS_SUCCESS)
            {
                core::vector<video::IDriverMemoryAllocation::MappedMemoryRange> dummyFlushRanges;
                buffer->Commit(offset,offset+allocSize,dummyFlushRanges);
                assert(dummyFlushRanges.size()==0);
                buffer->fenceRangeUsedByGPU(offset,offset+allocSize);
                buffer->Free(offset,offset+allocSize);
            }
        }

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
		    std::wostringstream sstr;
		    sstr << L"Builtin Nodes Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS();

			device->setWindowCaption(sstr.str().c_str());
			lastFPSTime = time;
		}
	}
	buffer->drop();

	device->drop();

	return 0;
}

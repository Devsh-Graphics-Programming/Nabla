#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include <random>
#include "irr/video/alloc/StreamingTransientDataBuffer.h"


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
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	auto device = createDeviceEx(params);

	if (!device)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

    size_t allocSize = 128;

    constexpr size_t kMinAllocs = 10000u;
    constexpr size_t kMaxAllocs = 20000u;


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
    auto buffer = core::make_smart_refctd_ptr<video::StreamingTransientDataBufferST<> >(driver,reqs);

    std::mt19937 mt(0xdeadu);
    std::uniform_int_distribution<uint32_t> allocsPerFrame(kMinAllocs,kMaxAllocs);
    std::uniform_int_distribution<uint32_t> size(1, 1024*1024);
    std::uniform_int_distribution<uint32_t> alignment(1, 128);

	uint64_t lastFPSTime = 0;
	while(device->run())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

		auto allocsThisFrame = allocsPerFrame(mt);
		uint32_t outAddr[kMaxAllocs];
		uint32_t sizes[kMaxAllocs];
		uint32_t alignments[kMaxAllocs];
        for (size_t i=0; i<allocsThisFrame; i++)
        {
            outAddr[i]  = video::StreamingTransientDataBufferST<>::invalid_address;
            sizes[i]    = size(mt);
            alignments[i] = alignment(mt);
        }

        buffer->multi_alloc(allocsThisFrame,(uint32_t*)outAddr,(const uint32_t*)sizes,(const uint32_t*)alignments);
        buffer->multi_free(allocsThisFrame,(const uint32_t*)outAddr,(const uint32_t*)sizes,driver->placeFence());

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
		    std::wostringstream sstr;
		    sstr << L"Alloc Perf Test- Irrlicht Engine [" << driver->getName() << "] K-Allocs/second:" << driver->getFPS()*allocsThisFrame;

			device->setWindowCaption(sstr.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}

#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <random>
#include <cmath>

using namespace irr;
using namespace core;

#define ALLOCATOR_TEST
//#define ALLOCATOR_TEST_OLD
//#define ADDRESS_ALLOCATOR_TRAITS_TEST
//#define ALLOC_PREF_TEST

#ifdef ALLOCATOR_TEST

struct AllocationCreationParameters
{	
	size_t multiAllocCnt;              //! Specifies amount of adress to be allocated with certain choosen allocator
	size_t adressesToDeallocateCnt;    //! Specifies amount of adress to be deallocated with certain choosen allocator. Must be less than all allocated and must pick number less than `traits::max_multi_free`, but we don't have `max_multi_free`
};

constexpr size_t minAllocs = 10000u;
constexpr size_t maxAllocs = 20000u;
constexpr size_t maxSizeForSquaring = 1024u;
constexpr size_t maxAlignmentExp = 12u;                         // 4096
constexpr size_t minVirtualMemoryBufferSize = 2048;             // 2kB
constexpr size_t maxVirtualMemoryBufferSize = 2147483648;       // 2GB
constexpr size_t maxOffset = 512;
constexpr size_t maxAlignOffset = 64;
constexpr size_t maxBlockSize = 600;

class RandomNumberGenerator
{
public:
	inline uint32_t getRndAllocCnt()    { return  allocsPerFrameRange(mt);  }
	inline uint32_t getRndSize()        { return  sizePerFrameRange(mt); }
	inline uint32_t getRndMaxAlign()    { return  (1u << maxAlignmentExpPerFrameRange(mt)); } //128 is max
	inline uint32_t getRndBuffSize()    { return  buffSzRange(mt); }
	inline uint32_t getRndOffset()      { return  offsetPerFrameRange(mt); }
	inline uint32_t getRndAlignOffset() { return  alignOffsetPerFrameRange(mt); }
	inline uint32_t getRndBlockSize()   { return  allocsPerFrameRange(mt); }

	inline uint32_t getRandomNumber(uint32_t rangeBegin, uint32_t rangeEnd)   
	{
		std::uniform_int_distribution<uint32_t> dist(rangeBegin, rangeEnd);
		return dist(mt);
	}

private:
	std::mt19937 mt;
	const std::uniform_int_distribution<uint32_t> allocsPerFrameRange = std::uniform_int_distribution<uint32_t>(minAllocs, maxAllocs);
	const std::uniform_int_distribution<uint32_t> sizePerFrameRange = std::uniform_int_distribution<uint32_t>(1u, maxSizeForSquaring);
	const std::uniform_int_distribution<uint32_t> maxAlignmentExpPerFrameRange = std::uniform_int_distribution<uint32_t>(1, maxAlignmentExp);

	const std::uniform_int_distribution<uint32_t> buffSzRange = std::uniform_int_distribution<uint32_t>(minVirtualMemoryBufferSize, maxVirtualMemoryBufferSize);
	const std::uniform_int_distribution<uint32_t> offsetPerFrameRange = std::uniform_int_distribution<uint32_t>(0u, maxOffset);
	const std::uniform_int_distribution<uint32_t> alignOffsetPerFrameRange = std::uniform_int_distribution<uint32_t>(0u, maxAlignOffset);
	const std::uniform_int_distribution<uint32_t> blockSizePerFrameRange = std::uniform_int_distribution<uint32_t>(1, maxBlockSize);
};

template<typename AlctrType>
class AllocatorHandler
{
	using Traits = core::address_allocator_traits<AlctrType>;
public:
	
	AllocatorHandler(AllocationCreationParameters& allocationCreationParameters)
		: creationParameters(allocationCreationParameters) 
	{
	}
	
	void executeAllocatorTest()
	{
		for (size_t i = 0; i < creationParameters.multiAllocCnt; ++i)
			executeForFrame();
	}

private:
	struct AllocationData
	{
		uint32_t outAddr = AlctrType::invalid_address;
		uint32_t size = 0u;
		uint32_t align = 0u;
	};

	struct RandParams
	{
		uint32_t maxAlign;
		uint32_t addressSpaceSize;
		uint32_t alignOffset;
		uint32_t offset;
		uint32_t blockSz;
	};

private:
	void executeForFrame()
	{
		RandParams randAllocParams = getRandParams();
		void* reservedSpace = nullptr;
		AlctrType alctr;

		if constexpr (std::is_same<AlctrType, core::LinearAddressAllocator<uint32_t>>::value)
		{
			alctr = AlctrType(nullptr, randAllocParams.offset, randAllocParams.alignOffset, randAllocParams.maxAlign, randAllocParams.addressSpaceSize);
		}
		else
		{
			const auto reservedSize = AlctrType::reserved_size(randAllocParams.maxAlign, randAllocParams.addressSpaceSize, randAllocParams.blockSz);
			reservedSpace = _NBL_ALIGNED_MALLOC(reservedSize, _NBL_SIMD_ALIGNMENT);
			alctr = AlctrType(reservedSpace, randAllocParams.offset, randAllocParams.alignOffset, randAllocParams.maxAlign, randAllocParams.addressSpaceSize, randAllocParams.blockSz);
		}

		// randomly decide how many `multi_allocs` to do
		const uint32_t multiAllocCnt = rng.getRandomNumber(1u, 5u); //TODO: will change it later
		for (uint32_t i = 0u; i < multiAllocCnt; i++)
		{
			outAddresses.clear();
			sizes.clear();
			alignments.clear();

			// randomly decide how many allocs in a `multi_alloc` NOTE: must pick number less than `traits::max_multi_alloc`
			constexpr uint32_t upperBound = core::address_allocator_traits<AlctrType>::maxMultiOps;
			const uint32_t allocCntInMultiAlloc = rng.getRandomNumber(1u, upperBound);

			for (size_t i = 0u; i < allocCntInMultiAlloc; ++i)
			{
				// randomly decide sizes (but always less than `address_allocator_traits::max_size`)
				outAddresses.emplace_back(AlctrType::invalid_address);

				if constexpr (std::is_same<AlctrType, core::PoolAddressAllocator<uint32_t>>::value)
				{
					sizes.emplace_back(randAllocParams.blockSz);
					alignments.emplace_back(randAllocParams.blockSz);
				}
				else
				{
					sizes.emplace_back(rng.getRandomNumber(1u, Traits::max_size(alctr)));
					alignments.emplace_back(rng.getRndMaxAlign());
				}
			}

			Traits::multi_alloc_addr(alctr, allocCntInMultiAlloc, outAddresses.data(), sizes.data(), alignments.data());

			// record all successful alloc addresses to the `core::vector`
			for (uint32_t i = 0u; i < outAddresses.size(); i++)
			{
				if (outAddresses[i] != AlctrType::invalid_address)
					results.push_back({ outAddresses[i], sizes[i], alignments[i] });
			}

			// run random dealloc function
			// linear address allocator is always reseted here
			randFreeAllocatedAddresses(alctr);
		}

		// randomly choose between reset and freeing all `core::vector` elements
		if constexpr (!std::is_same<AlctrType, core::LinearAddressAllocator<uint32_t>>::value) //linear address allocator is always reseted in the `randFreeAllocatedAddresses()`
		{
			bool reset = static_cast<bool>(rng.getRandomNumber(0u, 1u));
			if (reset)
			{
				alctr.reset();
				results.clear();
			}
			else
			{
				// free everything with a series of multi_free
				while (results.size() != 0u)
					randFreeAllocatedAddresses(alctr);
			}
		}
		 
		if constexpr (!std::is_same<AlctrType, core::LinearAddressAllocator<uint32_t>>::value)
			_NBL_ALIGNED_FREE(reservedSpace);

	}

	// random dealloc function
	void randFreeAllocatedAddresses(AlctrType& alctr)
	{
		if(results.size() == 0u)
			return;

		outAddresses.clear();
		sizes.clear();
		alignments.clear();

		// randomly decide how many calls to `multi_free`
		const uint32_t multiFreeCnt = rng.getRandomNumber(1u, 10000u); //TODO

		//TODO:
		//shuffle results

		for (uint32_t i = 0u; i < multiFreeCnt; i++)
		{
			// randomly how many addresses we should deallocate (but obvs less than all allocated) NOTE: must pick number less than `traits::max_multi_free`
			const uint32_t addressesToFreeCnt = rng.getRandomNumber(0u, results.size()); //that should be restrained somehow, I think... so it it less likely to free all of allocated addresses

			for (uint32_t i = 0u; i < addressesToFreeCnt; i++)
			{
				outAddresses.push_back(results[i].outAddr);
				sizes.push_back(results[i].size);
			}

			Traits::multi_free_addr(alctr, addressesToFreeCnt, outAddresses.data(), sizes.data());
			results.erase(results.begin(), results.begin() + addressesToFreeCnt);
		}
	}
	
	RandParams getRandParams()
	{
		RandParams randParams;

		randParams.maxAlign = rng.getRndMaxAlign();
		randParams.addressSpaceSize = rng.getRndBuffSize();

		randParams.alignOffset = rng.getRandomNumber(0u, randParams.maxAlign - 1u);
		randParams.offset = rng.getRandomNumber(0u, randParams.addressSpaceSize - 1u);

		randParams.blockSz = rng.getRandomNumber(0u, (randParams.addressSpaceSize - randParams.offset) / 2u);
		assert(randParams.blockSz > 0u);

		return randParams;
	}

private:
	RandomNumberGenerator rng;
	AllocationCreationParameters creationParameters;
	core::vector<AllocationData> results;

	//these hold inputs for `multi_alloc_addr` and `multi_free_addr`
	core::vector<uint32_t> outAddresses;
	core::vector<uint32_t> sizes;
	core::vector<uint32_t> alignments;
};

template<>
void AllocatorHandler<core::LinearAddressAllocator<uint32_t>>::randFreeAllocatedAddresses(core::LinearAddressAllocator<uint32_t>& alctr)
{
	alctr.reset();
	results.clear();
}

template<>
void AllocatorHandler<core::StackAddressAllocator<uint32_t>>::randFreeAllocatedAddresses(core::StackAddressAllocator<uint32_t>& alctr)
{
	//TODO;
}

int main()
{
	AllocationCreationParameters creationParams;
	creationParams.multiAllocCnt = 1000;				// TODO
	creationParams.adressesToDeallocateCnt = 1000;		// TODO

	//TODO:
	/*{
		AllocatorHandler<core::StackAddressAllocator<uint32_t>> stackAlctrHandler(creationParams);
		stackAlctrHandler.executeAllocatorTest();
	}*/
	
	{
		AllocatorHandler<core::PoolAddressAllocator<uint32_t>> poolAlctrHandler(creationParams);
		poolAlctrHandler.executeAllocatorTest();
	}

	{
		AllocatorHandler<core::LinearAddressAllocator<uint32_t>> linearAlctrHandler(creationParams);
		linearAlctrHandler.executeAllocatorTest();
	}
	
	//crashes..
	{
		AllocatorHandler<core::GeneralpurposeAddressAllocator<uint32_t>> generalpurposeAlctrHandler(creationParams);
		generalpurposeAlctrHandler.executeAllocatorTest();
	}
}

#endif

#ifdef ADDRESS_ALLOCATOR_TRAITS_TEST

int main()
{
	printf("SINGLE THREADED======================================================\n");
	printf("Linear \n");
	irr::core::address_allocator_traits<core::LinearAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Stack \n");
	irr::core::address_allocator_traits<core::StackAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("Pool \n");
	irr::core::address_allocator_traits<core::PoolAddressAllocatorST<uint32_t> >::printDebugInfo();
	printf("General \n");
	irr::core::address_allocator_traits<core::GeneralpurposeAddressAllocatorST<uint32_t> >::printDebugInfo();

	printf("MULTI THREADED=======================================================\n");
	printf("Linear \n");
	irr::core::address_allocator_traits<core::LinearAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();
	printf("Pool \n");
	irr::core::address_allocator_traits<core::PoolAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();
	printf("General \n");
	irr::core::address_allocator_traits<core::GeneralpurposeAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();

	return 0;
}
#endif

#ifdef ALLOC_PREF_TEST

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
	reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE | video::IDriverMemoryAllocation::EMCF_COHERENT;
	reqs.prefersDedicatedAllocation = true;
	reqs.requiresDedicatedAllocation = true;
	auto buffer = core::make_smart_refctd_ptr<video::StreamingTransientDataBufferST<> >(driver, reqs);

	std::mt19937 mt(0xdeadu);
	std::uniform_int_distribution<uint32_t> allocsPerFrame(kMinAllocs, kMaxAllocs);
	std::uniform_int_distribution<uint32_t> size(1, 1024 * 1024);
	std::uniform_int_distribution<uint32_t> alignment(1, 128);

	uint64_t lastFPSTime = 0;
	while (device->run())
		//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

		auto allocsThisFrame = allocsPerFrame(mt);
		uint32_t outAddr[kMaxAllocs];
		uint32_t sizes[kMaxAllocs];
		uint32_t alignments[kMaxAllocs];
		for (size_t i = 0; i < allocsThisFrame; i++)
		{
			outAddr[i] = video::StreamingTransientDataBufferST<>::invalid_address;
			sizes[i] = size(mt);
			alignments[i] = alignment(mt);
		}

		buffer->multi_alloc(allocsThisFrame, (uint32_t*)outAddr, (const uint32_t*)sizes, (const uint32_t*)alignments);
		buffer->multi_free(allocsThisFrame, (const uint32_t*)outAddr, (const uint32_t*)sizes, driver->placeFence());

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream sstr;
			sstr << L"Alloc Perf Test- Irrlicht Engine [" << driver->getName() << "] K-Allocs/second:" << driver->getFPS() * allocsThisFrame;

			device->setWindowCaption(sstr.str().c_str());
			lastFPSTime = time;
		}
	}

	return 0;
}
#endif
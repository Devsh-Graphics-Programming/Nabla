#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <random>
#include <cmath>

using namespace irr;
using namespace core;

#define kNumHardwareInstancesX 10
#define kNumHardwareInstancesY 20
#define kNumHardwareInstancesZ 30

#define kHardwareInstancesTOTAL (kNumHardwareInstancesX*kNumHardwareInstancesY*kNumHardwareInstancesZ)

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

constexpr size_t minTestsCnt = 100u;
constexpr size_t maxTestsCnt = 200u;
constexpr size_t maxAlignmentExp = 12u;                         // 4096
constexpr size_t minVirtualMemoryBufferSize = 2048;             // 2kB
constexpr size_t maxVirtualMemoryBufferSize = 2147483648;       // 2GB

class RandomNumberGenerator
{
public:
	RandomNumberGenerator()
		: mt(rd())
	{
	}

	inline uint32_t getRndAllocCnt()    { return  allocsPerFrameRange(mt);  }
	inline uint32_t getRndMaxAlign()    { return  (1u << maxAlignmentExpPerFrameRange(mt)); } //4096 is max
	inline uint32_t getRndBuffSize()    { return  buffSzRange(mt); }

	inline uint32_t getRandomNumber(uint32_t rangeBegin, uint32_t rangeEnd)   
	{
		std::uniform_int_distribution<uint32_t> dist(rangeBegin, rangeEnd);
		return dist(mt);
	}

	inline std::mt19937& getMt()
	{
		return mt;
	}

private:
	std::random_device rd;
	std::mt19937 mt;
	const std::uniform_int_distribution<uint32_t> allocsPerFrameRange = std::uniform_int_distribution<uint32_t>(minTestsCnt, maxTestsCnt);
	const std::uniform_int_distribution<uint32_t> maxAlignmentExpPerFrameRange = std::uniform_int_distribution<uint32_t>(1, maxAlignmentExp);
	const std::uniform_int_distribution<uint32_t> buffSzRange = std::uniform_int_distribution<uint32_t>(minVirtualMemoryBufferSize, maxVirtualMemoryBufferSize);
};

RandomNumberGenerator rng;

template<typename AlctrType>
class AllocatorHandler
{
	using Traits = core::address_allocator_traits<AlctrType>;

public:
	void executeAllocatorTest()
	{
		uint32_t testsCnt = rng.getRndAllocCnt();

		for (size_t i = 0; i < testsCnt; i++)
		{
			AlctrType alctr;
			RandParams randAllocParams = getRandParams();
			void* reservedSpace = nullptr;

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

			testsCnt = rng.getRndAllocCnt();
			for (size_t i = 0; i < testsCnt; i++)
				executeForFrame(alctr, randAllocParams);

			if constexpr (!std::is_same<AlctrType, core::LinearAddressAllocator<uint32_t>>::value)
				_NBL_ALIGNED_FREE(reservedSpace);
		}
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
	void executeForFrame(AlctrType& alctr, RandParams& randAllocParams)
	{
		// randomly decide how many `multi_allocs` to do
		const uint32_t multiAllocCnt = rng.getRandomNumber(1u, 500u);
		for (uint32_t i = 0u; i < multiAllocCnt; i++)
		{
			uint32_t addressesToAllcate = allocDataSoA.randomizeAllocData(alctr, randAllocParams);
			// randomly decide how many allocs in a `multi_alloc` NOTE: must pick number less than `traits::max_multi_alloc`

			Traits::multi_alloc_addr(alctr, addressesToAllcate, allocDataSoA.outAddresses.data(), allocDataSoA.sizes.data(), allocDataSoA.alignments.data());

			// record all successful alloc addresses to the `core::vector`
			for (uint32_t j = 0u; j < allocDataSoA.size; j++)
			{
				if (allocDataSoA.outAddresses[j] != AlctrType::invalid_address)
					results.push_back({ allocDataSoA.outAddresses[j], allocDataSoA.sizes[j], allocDataSoA.alignments[j] });
			}

			// run random dealloc function
			randFreeAllocatedAddresses(alctr);
		}

		// randomly choose between reset and freeing all `core::vector` elements
		if constexpr (!std::is_same<AlctrType, core::LinearAddressAllocator<uint32_t>>::value)
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
		else
		{
			alctr.reset();
			results.clear();
		}
	}

	// random dealloc function
	void randFreeAllocatedAddresses(AlctrType& alctr)
	{
		if(results.size() == 0u)
			return;

		// randomly decide how many calls to `multi_free`
		const uint32_t multiFreeCnt = rng.getRandomNumber(1u, results.size());

		if (std::is_same<AlctrType, core::GeneralpurposeAddressAllocator<uint32_t>>::value)
		{
			std::shuffle(results.begin(), results.end(), rng.getMt());
		}

		for (uint32_t i = 0u; (i < multiFreeCnt) && results.size(); i++)
		{
			allocDataSoA.reset();

			// randomly how many addresses we should deallocate (but obvs less than all allocated) NOTE: must pick number less than `traits::max_multi_free`
			const uint32_t addressesToFreeUpperBound = min(Traits::maxMultiOps, results.size());
			const uint32_t addressesToFreeCnt = rng.getRandomNumber(0u, addressesToFreeUpperBound);

			auto it = results.end();
			for (uint32_t j = 0u; j < addressesToFreeCnt; j++)
			{
				it--;
				allocDataSoA.addElement(*it);
			}

			Traits::multi_free_addr(alctr, addressesToFreeCnt, allocDataSoA.outAddresses.data(), allocDataSoA.sizes.data());
			results.erase(results.end() - addressesToFreeCnt, results.end());
		}
	}
	
	RandParams getRandParams()
	{
		RandParams randParams;

		randParams.maxAlign = rng.getRndMaxAlign();
		randParams.addressSpaceSize = rng.getRndBuffSize();

		randParams.alignOffset = rng.getRandomNumber(0u, randParams.maxAlign - 1u);
		randParams.offset = rng.getRandomNumber(0u, randParams.addressSpaceSize - 1u);

		randParams.blockSz = rng.getRandomNumber(1u, (randParams.addressSpaceSize - randParams.offset) / 2u);
		assert(randParams.blockSz > 0u);

		return randParams;
	}

private:
	core::vector<AllocationData> results;

	//these hold inputs for `multi_alloc_addr` and `multi_free_addr`

	struct AllocationDataSoA
	{
		uint32_t randomizeAllocData(AlctrType& alctr, RandParams& randAllocParams)
		{
			constexpr uint32_t upperBound = Traits::maxMultiOps - 1u;
			uint32_t allocCntInMultiAlloc = rng.getRandomNumber(1u, upperBound);

			std::fill(outAddresses.begin(), outAddresses.begin() + allocCntInMultiAlloc, AlctrType::invalid_address);

			for (uint32_t j = 0u; j < allocCntInMultiAlloc; j++)
			{
				// randomly decide sizes (but always less than `address_allocator_traits::max_size`)

				if constexpr (std::is_same<AlctrType, core::PoolAddressAllocator<uint32_t>>::value)
				{
					sizes[j] = randAllocParams.blockSz;
					alignments[j] = randAllocParams.blockSz;
				}
				else
				{
					sizes[j] = rng.getRandomNumber(1u, std::max(Traits::max_size(alctr), 1u));
					alignments[j] = rng.getRandomNumber (1u, randAllocParams.maxAlign);
				}
			}

			size = allocCntInMultiAlloc;

			return allocCntInMultiAlloc;
		}

		void addElement(const AllocationData& allocData)
		{
			outAddresses[end] = allocData.outAddr;
			sizes[end] = allocData.size;
			end++;
			_NBL_DEBUG_BREAK_IF(end > Traits::maxMultiOps);
		}

		inline void reset() { end = 0u; }

		union
		{
			uint32_t end = 0u;
			uint32_t size;
		};

		core::vector<uint32_t> outAddresses = core::vector<uint32_t>(Traits::maxMultiOps, AlctrType::invalid_address);
		core::vector<uint32_t> sizes = core::vector<uint32_t>(Traits::maxMultiOps, 0u);
		core::vector<uint32_t> alignments = core::vector<uint32_t>(Traits::maxMultiOps, 0u);
	};

	AllocationDataSoA allocDataSoA;
};

template<>
void AllocatorHandler<core::LinearAddressAllocator<uint32_t>>::randFreeAllocatedAddresses(core::LinearAddressAllocator<uint32_t>& alctr)
{
	const bool performReset = rng.getRandomNumber(1, 10) == 1 ? true : false;

	if (performReset)
	{
		alctr.reset();
		results.clear();
	}
}

int main()
{

	// Allocator test
	{
		{
			AllocatorHandler<core::PoolAddressAllocator<uint32_t>> poolAlctrHandler;
			poolAlctrHandler.executeAllocatorTest();
		}

		{
			AllocatorHandler<core::LinearAddressAllocator<uint32_t>> linearAlctrHandler;
			linearAlctrHandler.executeAllocatorTest();
		}

		{
			AllocatorHandler<core::StackAddressAllocator<uint32_t>> stackAlctrHandler;
			stackAlctrHandler.executeAllocatorTest();
		}

		{
			AllocatorHandler<core::GeneralpurposeAddressAllocator<uint32_t>> generalpurposeAlctrHandler;
			generalpurposeAlctrHandler.executeAllocatorTest();
		}
	}
	

	// Address allocator traits test
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
	}

	// Alloc pref test
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
	}
}
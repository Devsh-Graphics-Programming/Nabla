#define _IRR_STATIC_LIB_
#include <nabla.h>
#include <random>
#include <cmath>
#include "../common/CommonAPI.h"
using namespace nbl;
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
		if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
		{
			switch (event.KeyInput.Key)
			{
			case nbl::KEY_KEY_Q: // switch wire frame mode
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

			uint32_t subTestsCnt = rng.getRndAllocCnt();
			for (size_t i = 0; i < subTestsCnt; i++)
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

		inline bool operator==(const AllocationData& other) const
		{
			return outAddr==other.outAddr;
		}

		struct Hash
		{
			inline size_t operator()(const AllocationData& _this) const
			{
				return std::hash<uint32_t>()(_this.outAddr);
			}
		};
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
			if constexpr (!std::is_same<AlctrType, core::LinearAddressAllocator<uint32_t>>::value)
			for (uint32_t j = 0u; j < allocDataSoA.size; j++)
			{
				if (allocDataSoA.outAddresses[j] != AlctrType::invalid_address)
					results.push_back({ allocDataSoA.outAddresses[j], allocDataSoA.sizes[j], allocDataSoA.alignments[j] });
			}
			checkStillIteratable(alctr);

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
			alctr.reset();
	}

	// random dealloc function
	void randFreeAllocatedAddresses(AlctrType& alctr)
	{
		if(results.size() == 0u)
			return;

		// randomly decide how many calls to `multi_free`
		const uint32_t multiFreeCnt = rng.getRandomNumber(1u, results.size());

		if constexpr (Traits::supportsArbitraryOrderFrees)
			std::shuffle(results.begin(), results.end(), rng.getMt());

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
			checkStillIteratable(alctr);
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
	inline void checkStillIteratable(const AlctrType& alctr)
	{
		if constexpr (std::is_same<AlctrType, core::IteratablePoolAddressAllocator<uint32_t>>::value)
		{
			core::unordered_set<AllocationData,AllocationData::Hash> allocationSet(results.begin(),results.end());
			for (auto addr : alctr)
			{
				AllocationData dummy; dummy.outAddr = addr;
				if (allocationSet.find(dummy)==allocationSet.end())
					exit(34);
			}
		}
	}

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

				if constexpr (std::is_same_v<AlctrType,core::PoolAddressAllocator<uint32_t>>||std::is_same_v<AlctrType,core::IteratablePoolAddressAllocator<uint32_t>>)
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
			AllocatorHandler<core::IteratablePoolAddressAllocator<uint32_t>> iterPoolAlctrHandler;
			iterPoolAlctrHandler.executeAllocatorTest();
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
		nbl::core::address_allocator_traits<core::LinearAddressAllocatorST<uint32_t> >::printDebugInfo();
		printf("Stack \n");
		nbl::core::address_allocator_traits<core::StackAddressAllocatorST<uint32_t> >::printDebugInfo();
		printf("Pool \n");
		nbl::core::address_allocator_traits<core::PoolAddressAllocatorST<uint32_t> >::printDebugInfo();
		printf("IteratablePool \n");
		nbl::core::address_allocator_traits<core::IteratablePoolAddressAllocatorST<uint32_t> >::printDebugInfo();
		printf("General \n");
		nbl::core::address_allocator_traits<core::GeneralpurposeAddressAllocatorST<uint32_t> >::printDebugInfo();

		printf("MULTI THREADED=======================================================\n");
		printf("Linear \n");
		nbl::core::address_allocator_traits<core::LinearAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();
		printf("Pool \n");
		nbl::core::address_allocator_traits<core::PoolAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();
		printf("Iteratable Pool \n");
		nbl::core::address_allocator_traits<core::IteratablePoolAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();
		printf("General \n");
		nbl::core::address_allocator_traits<core::GeneralpurposeAddressAllocatorMT<uint32_t, std::recursive_mutex> >::printDebugInfo();
	}

	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "Compute Shader");
	auto win = initOutp.window;
	auto gl = initOutp.apiConnection;
	auto surface = initOutp.surface;
	auto device = initOutp.logicalDevice;
	auto queue = initOutp.queue;
	auto sc = initOutp.swapchain;
	auto renderpass = initOutp.renderpass;
	auto fbo = initOutp.fbo;
	auto cmdpool = initOutp.commandPool;

	{
		video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;


		core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
		assert(cb);

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = WIN_W;
		vp.height = WIN_H;
		cb->setViewport(0u, 1u, &vp);

		cb->end();

		video::IGPUQueue::SSubmitInfo info;
		auto* cb_ = cb.get();
		info.commandBufferCount = 1u;
		info.commandBuffers = &cb_;
		info.pSignalSemaphores = nullptr;
		info.signalSemaphoreCount = 0u;
		info.pWaitSemaphores = nullptr;
		info.waitSemaphoreCount = 0u;
		info.pWaitDstStageMask = nullptr;
		queue->submit(1u, &info, nullptr);
	}

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
	device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = cmdbuf[i];
		auto& fb = fbo[i];

		cb->begin(0);

		size_t offset = 0u;
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		asset::VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = { WIN_W, WIN_H };
		clear.color.float32[0] = 0.f;
		clear.color.float32[1] = 0.f;
		clear.color.float32[2] = 1.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = renderpass;
		info.framebuffer = fb;
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		cb->endRenderPass();

		cb->end();
	}
	size_t allocSize = 128;

    constexpr size_t kMinAllocs = 10000u;
    constexpr size_t kMaxAllocs = 20000u;
	/*
		TODO: make all the commented below stuff from the previous API work here
	*/

//	video::IDriverMemoryBacked::SDriverMemoryRequirements reqs;
//	reqs.vulkanReqs.size = 0x1000000u;
//	reqs.vulkanReqs.alignment = 4;
//	reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
//	reqs.memoryHeapLocation = video::IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
//	reqs.mappingCapability = video::IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE | video::IDriverMemoryAllocation::EMCF_COHERENT;
//	reqs.prefersDedicatedAllocation = true;
//	reqs.requiresDedicatedAllocation = true;
//	auto buffer = core::make_smart_refctd_ptr<video::StreamingTransientDataBufferST<> >(driver, reqs);

	std::mt19937 mt(0xdeadu);
	std::uniform_int_distribution<uint32_t> allocsPerFrame(kMinAllocs, kMaxAllocs);
	std::uniform_int_distribution<uint32_t> size(1, 1024 * 1024);
	std::uniform_int_distribution<uint32_t> alignment(1, 128);
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		auto allocsThisFrame = allocsPerFrame(mt);
		//uint32_t outAddr[kMaxAllocs];
		uint32_t sizes[kMaxAllocs];
		uint32_t alignments[kMaxAllocs];
		for (size_t i = 0; i < allocsThisFrame; i++)
		{
			//outAddr[i] = video::StreamingTransientDataBufferST<>::invalid_address;
			sizes[i] = size(mt);
			alignments[i] = alignment(mt);
		}

		//buffer->multi_alloc(allocsThisFrame, (uint32_t*)outAddr, (const uint32_t*)sizes, (const uint32_t*)alignments);
		//buffer->multi_free(allocsThisFrame, (const uint32_t*)outAddr, (const uint32_t*)sizes, driver->placeFence());

		CommonAPI::Present<SC_IMG_COUNT>(device, sc, cmdbuf, queue);
	}

	device->waitIdle();
}


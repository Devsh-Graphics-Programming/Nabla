#define _IRR_STATIC_LIB_
#include <nabla.h>
#include <random>
#include <cmath>
#include "../common/CommonAPI.h"
using namespace nbl;
using namespace core;


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

	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};

	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};
	optionalInstanceFeatures.count = 1u;
	video::IAPIConnection::E_FEATURE optionalFeatures_Instance[] = { video::IAPIConnection::EF_SURFACE };
	optionalInstanceFeatures.features = optionalFeatures_Instance;

	CommonAPI::SFeatureRequest<video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};

	CommonAPI::SFeatureRequest< video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};
	optionalDeviceFeatures.count = 1u;
	video::ILogicalDevice::E_FEATURE optionalFeatures_Device[] = { video::ILogicalDevice::EF_SWAPCHAIN };
	optionalDeviceFeatures.features = optionalFeatures_Device;

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(
		video::EAT_VULKAN,
		"10.AllocatorTest",
		requiredInstanceFeatures,
		optionalInstanceFeatures,
		requiredDeviceFeatures,
		optionalDeviceFeatures,
		asset::IImage::EUF_COLOR_ATTACHMENT_BIT);

	auto win = std::move(initOutp.window);
	auto winCb = std::move(initOutp.windowCb);

	size_t allocSize = 128;

    constexpr size_t kMinAllocs = 10000u;
    constexpr size_t kMaxAllocs = 20000u;

	std::mt19937 mt(0xdeadu);
	std::uniform_int_distribution<uint32_t> allocsPerFrame(kMinAllocs, kMaxAllocs);
	std::uniform_int_distribution<uint32_t> size(1, 1024 * 1024);
	std::uniform_int_distribution<uint32_t> alignment(1, 128);
	while (winCb->isWindowOpen())
	{
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
	}
}


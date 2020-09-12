#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include <random>
#include <cmath>

using namespace irr;
using namespace core;

/*
	If traits say address allocator supports arbitrary order free,
	then ETAT_CHOOSE_RANDOMLY is valid. Otherwise ETAT_CHOOSE_MOST_RECENTLY_ALLOCATED
*/

enum E_TRAITS_ALLOCATION_TYPE
{
	ETAT_CHOOSE_RANDOMLY,
	ETAT_CHOOSE_MOST_RECENTLY_ALLOCATED,
	ETAT_COUNT
};

struct AllocationCreationParameters
{	
	size_t multiAllocCnt;              //! Specifies amount of adress to be allocated with certain choosen allocator
	size_t adressesToDeallocateCnt;    //! Specifies amount of adress to be deallocated with certain choosen allocator. Must be less than all allocated and must pick number less than `traits::max_multi_free`, but we don't have `max_multi_free`
};

constexpr size_t minAllocs = 10000u;
constexpr size_t maxAllocs = 20000u;
constexpr size_t maxSizeForSquaring = 1024u;
constexpr size_t maxAlignmentExp = 6u;
constexpr size_t minByteSizeForReservingMemory = 2048;			// 2kB
constexpr size_t maxByteSizeForReservingMemory = 2147483648;	// 2GB
constexpr size_t maxOffset = 512;
constexpr size_t maxAlignOffset = 64;
constexpr size_t maxBlockSize = 600;

//TODO: full static
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

	const std::uniform_int_distribution<uint32_t> buffSzRange = std::uniform_int_distribution<uint32_t>(minByteSizeForReservingMemory, maxByteSizeForReservingMemory);
	const std::uniform_int_distribution<uint32_t> offsetPerFrameRange = std::uniform_int_distribution<uint32_t>(0u, maxOffset);
	const std::uniform_int_distribution<uint32_t> alignOffsetPerFrameRange = std::uniform_int_distribution<uint32_t>(0u, maxAlignOffset);
	const std::uniform_int_distribution<uint32_t> blockSizePerFrameRange = std::uniform_int_distribution<uint32_t>(1, maxBlockSize);
};

class AllocatorHandler
{
public:
	using PoolAdressAllocator = core::PoolAddressAllocatorST<uint32_t>;
	using Traits = core::address_allocator_traits<PoolAdressAllocator>;
	
	AllocatorHandler(AllocationCreationParameters& allocationCreationParameters)
		: creationParameters(allocationCreationParameters) 
	{
	}
	
	struct EntryForFrameData
	{
		uint32_t outAddr[maxAllocs] = { ~0u };
		uint32_t sizes[maxAllocs] = { 0u };
		uint32_t alignments[maxAllocs] = { 0u };
	} perFrameData;
	
	void executeAllocatorTest()
	{
		os::Printer::log("Executing Pool Allocator Test!", ELL_INFORMATION);
	
		for (size_t i = 0; i < creationParameters.multiAllocCnt; ++i)
			executeForFrame();
	}

private:
	// function to test allocator
		// pick random alignment, rand buffer size (between 2kb and 2GB), random offset (less than buffer size), random alignment offset (less than alignment) and other parameters randomly

		// allocate reserved space (for allocator state)

		// create address allocator

		// randomly decide the number of iterations of allocation and reset
			// declare `core::vector` to hold allocated addresses and sizes

			// randomly decide how many `multi_allocs` to do
				// randomly decide how many allocs in a `multi_alloc` NOTE: must pick number less than `traits::max_multi_alloc`
					// randomly decide sizes (but always less than `address_allocator_traits::max_size`)
				// record all successful alloc addresses to the `core::vector`

				// run random dealloc function
			//

			// run random dealloc function

			// randomly choose between reset and freeing all `core::vector` elements
				// reset
			// ELSE
				// free everything with a series of multi_free
	void executeForFrame()
	{
		uint32_t randMaxAlign = rng.getRndMaxAlign();
		uint32_t randAddressSpaceSize = rng.getRndBuffSize();
		uint32_t randAlignOffset = rng.getRandomNumber(0u, randMaxAlign - 1u);
		uint32_t randOffset = rng.getRandomNumber(0u, randAddressSpaceSize - 1u); //?
		uint32_t randBlockSz = rng.getRandomNumber(0u, randAddressSpaceSize - 1u); //?

		uint32_t allocationAmountForSingeFrameTest = rng.getRndAllocCnt();
		
		const auto reservedSize = PoolAdressAllocator::reserved_size(randMaxAlign, randAddressSpaceSize, randBlockSz); // 3rd parameter onward is custom for each address alloc type
		void* reservedSpace = _IRR_ALIGNED_MALLOC(reservedSize, _IRR_SIMD_ALIGNMENT);
		auto poolAlctr = PoolAdressAllocator(reservedSpace, randOffset, randAlignOffset, randMaxAlign, randAddressSpaceSize, randBlockSz);

		// randomly decide how many `multi_allocs` to do
		const uint32_t multiAllocCnt = rng.getRandomNumber(1u, 5u); //range?
		for (uint32_t i = 0u; i < multiAllocCnt; i++)
		{
			// randomly decide how many allocs in a `multi_alloc` NOTE: must pick number less than `traits::max_multi_alloc`
			const uint32_t allocCntInMultiAlloc = rng.getRandomNumber(2u, 5u /*traits::max_multi_alloc*/);

			for (size_t i = 0u; i < allocCntInMultiAlloc; ++i)
			{
				// randomly decide sizes (but always less than `address_allocator_traits::max_size`)
				perFrameData.sizes[i] = rng.getRandomNumber(1u, Traits::max_size(poolAlctr));
				perFrameData.alignments[i] = rng.getRndMaxAlign();
			}

			Traits::multi_alloc_addr(poolAlctr, allocCntInMultiAlloc, perFrameData.outAddr, perFrameData.sizes, perFrameData.alignments);

			// record all successful alloc addresses to the `core::vector`
			if (perFrameData.outAddr[i] != PoolAdressAllocator::invalid_address)
				results.push_back({ perFrameData.outAddr[i], perFrameData.sizes[i], perFrameData.alignments[i] });

			/*for (auto begin = results.begin(); begin != results.end(); begin++)
			{
				std::cout << "addr: " << begin->outAddr << std::endl;
				std::cout << "size: " << begin->size << std::endl;
				std::cout << "align: " << begin->align << std::endl << std::endl;
				__debugbreak();
			}*/
		}

		

		 //TODO capturing states
		  //record all allocated addresses to the `core::vector`
		 
		_IRR_ALIGNED_FREE(reservedSpace);
	}

	// random dealloc function
		// randomly decide how many calls to `multi_alloc`
			// randomly how many addresses we should deallocate (but obvs less than all allocated) NOTE: must pick number less than `traits::max_multi_free`
				// if traits say address allocator supports arbitrary order free, then choose randomly, else choose most recently allocated
	
	struct AllocationResults
	{
		uint32_t outAddr;
		uint32_t size;
		uint32_t align;
	};

	RandomNumberGenerator rng;
	AllocationCreationParameters creationParameters;
	
	core::vector<AllocationResults> results;

};

int main()
{
	AllocationCreationParameters creationParams;
	creationParams.multiAllocCnt = 1000;				// TODO
	creationParams.adressesToDeallocateCnt = 1000;		// TODO

	AllocatorHandler handler(creationParams);
	handler.executeAllocatorTest();
}
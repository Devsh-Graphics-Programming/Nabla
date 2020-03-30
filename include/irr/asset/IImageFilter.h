// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_IMAGE_FILTER_H_INCLUDED__
#define __IRR_I_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <algorithm>

#include "irr/asset/ICPUImage.h"

namespace irr
{
namespace asset
{

// runtime polymorphic
class IImageFilter
{
	public:
		class IState
		{
			public:
				struct TexelRange
				{
					VkOffset3D	offset = { 0u,0u,0u };
					VkExtent3D	extent = { 0u,0u,0u };
				};
				struct ColorValue
				{
					ColorValue() {}
					~ColorValue() {}

					_IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_CHANNELS = 4u;
					_IRR_STATIC_INLINE_CONSTEXPR uint32_t LARGEST_COMPRESSED_BLOCK_SIZE = 16u;
					union
					{
						uint8_t				pointer[sizeof(double)*MAX_CHANNELS];
						uint8_t				asCompressedBlock[LARGEST_COMPRESSED_BLOCK_SIZE];
						double				asDouble[MAX_CHANNELS];
						core::vectorSIMDf	asFloat;
						core::vectorSIMDu32 asUint;
						core::vectorSIMDi32 asInt;
						uint16_t			asUShort[MAX_CHANNELS];
						int16_t				asShort[MAX_CHANNELS];
						uint8_t				asUByte[MAX_CHANNELS];
						int8_t				asByte[MAX_CHANNELS];
					};

					struct WriteMemoryInfo
					{
						WriteMemoryInfo(E_FORMAT colorFmt, void* outPtr) :
							outMemory(reinterpret_cast<uint8_t*>(outPtr)),
							blockByteSize(getTexelOrBlockBytesize(colorFmt))
						{
						}

						uint8_t* const	outMemory = nullptr;
						const uint32_t	blockByteSize = 0u;
					};
					inline void writeMemory(const WriteMemoryInfo& info, uint32_t offset)
					{
						memcpy(info.outMemory+offset,pointer,info.blockByteSize);
					}
					
					struct ReadMemoryInfo
					{
						ReadMemoryInfo(E_FORMAT colorFmt, const void* inPtr) :
							inMemory(reinterpret_cast<const uint8_t*>(inPtr)),
							blockByteSize(getTexelOrBlockBytesize(colorFmt))
						{
						}

						const uint8_t* const	inMemory = nullptr;
						const uint32_t			blockByteSize = 0u;
					};
					inline void readMemory(const ReadMemoryInfo& info, uint32_t offset)
					{
						memcpy(pointer,info.inMemory+offset,info.blockByteSize);
					}
				};
		};		

        //
		virtual bool pValidate(IState* state) const = 0;
		
		//
		virtual bool pExecute(IState* state) const = 0;
};


// static polymorphic
template<typename CRTP>
class CImageFilter : public IImageFilter
{
	public:
		static inline bool validate(IState* state)
		{
			return CRTP::validate(static_cast<typename CRTP::state_type*>(state));
		}
		
		inline bool pValidate(IState* state) const override
		{
			return validate(state);
		}

		static inline bool execute(IState* state)
		{
			return CRTP::execute(static_cast<typename CRTP::state_type*>(state));
		}

		inline bool pExecute(IState* state) const override
		{
			return execute(state);
		}
};

}
}

#endif
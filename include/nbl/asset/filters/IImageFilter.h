// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_I_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <algorithm>

#include "nbl/asset/ICPUImage.h"

namespace nbl
{
namespace asset
{

//! Base class for general filters with runtime polymorphism.
/*
	Filters can execute various actions basing on input image
	to get an output image.
	Available filters are as following:

	- Fill Filter
	- Copy Filter
	- Flatten Filter
	- Convert Filter
	- Swizzle && Convert Filter
	- Blit Filter
	- Generate Mip Maps Filter

	If you don't know what filter you'll be executing at runtime, 
	you can use the \ipolymorphic interface\i and operate on 
	IImageFilter pointers or references.

	There is complete freedom on filters inputs and outputs, this is why 
	each defines (or at least typedefs) it's own state.

	There are input only, output only (such as fill filter) or input-output filters.
*/

class IImageFilter
{
	public:

		//! Base class for filter's \bstate\b.
		/*
			To make use of the filter, it's \bstate\b must be provided.
			State contains information about data needed to execute
			some processes needed to get final output image or it's bundle.

			Sometimes filter may require you to specify input image as
			a reference for final output image, but there are also such
			filters which only take output image. Multiple-input and
			multiple-output filters will be provided soon as well.
			Generally you are able to perform various image converting
			processes with  different layers and faces, but keep in mind
			that usually for certain mipmaps you will have to change appropriate
			state fields to make it work, because filters work for
			one mip-map level at a time. There is an exception when
			filters work with many mip maps at a time - when using
			\bGenerateMipMaps\b filter.

			Because of various inputs, outputs or even lack of one of those,
			it's doesn't declare any members, just type definitions as an interface.
			A particular filter delcares it's own \bstate_type\b typedef or alias and
			different filters require different states.
		*/

		class IState
		{
			public:
				virtual ~IState() {}

				/*
					Class for holding information about handled texel 
					range in a buffer attached to an image.
				*/

				struct TexelRange
				{
					VkOffset3D	offset = { 0u,0u,0u };
					VkExtent3D	extent = { 0u,0u,0u };
				};

				/*
					Class for reinterpreting a single color value,
					it may be a texel or single compressed block.
				*/

				struct ColorValue
				{
					ColorValue() {}
					~ColorValue() {}

					_NBL_STATIC_INLINE_CONSTEXPR uint32_t MAX_CHANNELS = 4u;
					_NBL_STATIC_INLINE_CONSTEXPR uint32_t LARGEST_COMPRESSED_BLOCK_SIZE = 16u;
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

					inline ColorValue& operator=(const ColorValue& other)
					{
						memcpy(pointer,other.pointer,sizeof(double)*MAX_CHANNELS);
						return *this;
					}

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

/*
	Filter class for static polymorphism
*/

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
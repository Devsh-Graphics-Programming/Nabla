// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__
#define __NBL_ASSET_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__

#include "nbl/core/core.h"

#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/filters/dithering/CDither.h"
#include "nbl/asset/filters/NormalizationStates.h"
#include "nbl/asset/filters/Swizzles.h"

namespace nbl::asset::impl
{

/*
	Common base class for Swizzleable or Ditherable ones
	with custom compile time swizzle, dither, normalization and clamp.
	The order of the parameters is the order in which they get applied.

	Normalize paramteter is required to decide whether the
	values being encoded should be previously normalized.

	Clamping paramter is required to decide whether values
	being encoded should be previously clamped to valid format 
	max and min values.
*/
template<typename Swizzle, typename Dither, typename Normalization, bool Clamp>
class CSwizzleableAndDitherableFilterBase
{
	public:
		class CState : public Swizzle
		{
			public:
				CState() {}
				virtual ~CState() {}

				Dither dither;
				using DitherState = typename Dither::state_type;
				DitherState* ditherState = nullptr;							//! Allocation, creation of the dither state and making the pointer valid is necessary!

				conditional_normalization_state<Normalization> normalization;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state->normalization.validate())
				return false;

			if (!state->ditherState)
				return false;

			return true;
		}

		/*
			Performs decode doing swizzle at first on a given pointer
			and putting the result to the encodeBuffer.

			The function supports compile-time decode.
		*/
		template<E_FORMAT inFormat, typename Tdec, typename Tenc>
		static void onDecode(state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
		{
			static_assert(sizeof(Tdec)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			asset::decodePixels<inFormat>(srcPix, decodeBuffer, blockX, blockY);
			// TODO: shouldn't swizzles be performed in-place on decode buffers and their values?
			static_cast<Swizzle&>(*state).operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
			state->normalization.operator()<Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Runtime version of onDecode

			@see onDecode
		*/
		template<typename Tdec, typename Tenc>
		static void onDecode(E_FORMAT inFormat, state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
		{
			static_assert(sizeof(Tdec)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			asset::decodePixelsRuntime(inFormat, srcPix, decodeBuffer, blockX, blockY);
			// TODO: shouldn't swizzles be performed in-place on decode buffers and their values?
			static_cast<Swizzle&>(*state).operator()<Tdec,Tenc>(decodeBuffer,encodeBuffer);
			state->normalization.operator()<Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Performs encode doing dithering at first on a given encode buffer in pointer.
			The encode buffer is a buffer holding decoded (and swizzled optionally) values.

			The function may perform clamping.
			@see CSwizzleableAndDitherableFilterBase

			The function may perform normalizing if normalizing is
			available and query normalize bool is true.
			@see CGlobalNormalizationState

			The function supports compile-time encode.
		*/
		template<E_FORMAT outFormat, typename Tenc>
		static void onEncode(state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels, bool queryNormalizing = false)
		{
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			for (uint8_t i = 0; i < channels; ++i)
			{
				const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
				auto* encodeValue = encodeBuffer + i;
				const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
				*encodeValue += static_cast<Tenc>(ditheredValue) * scale;
			}

			if (queryNormalizing)
				state->normalization.operator()<outFormat,Tenc>(encodeBuffer,position,blockX,blockY,channels);

			if constexpr (Clamp)
			{
				for (uint8_t i = 0; i < channels; ++i)
				{
					auto&& [min, max, encodeValue] = std::make_tuple<Tenc&&, Tenc&&, Tenc*>(asset::getFormatMinValue<Tenc>(outFormat, i), asset::getFormatMaxValue<Tenc>(outFormat, i), encodeBuffer + i);
					*encodeValue = core::clamp(*encodeValue, min, max);
				}
			}

			asset::encodePixels<outFormat>(dstPix, encodeBuffer);
		}

		/*
			Runtime version of onEncode

			@see onEncode
		*/
		template<typename Tenc>
		static void onEncode(E_FORMAT outFormat, state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels, bool queryNormalizing = false)
		{
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			for (uint8_t i = 0; i < channels; ++i)
			{
				const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
				auto* encodeValue = encodeBuffer + i;
				const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
				*encodeValue += static_cast<Tenc>(ditheredValue)* scale;
			}

			if (queryNormalizing)
				state->normalization.operator()<Tenc>(outFormat,encodeBuffer,position,blockX,blockY,channels);

			if constexpr (Clamp)
			{
				for (uint8_t i = 0; i < channels; ++i)
				{
					auto&& [min, max, encodeValue] = std::make_tuple<Tenc&&, Tenc&&, Tenc*>(asset::getFormatMinValue<Tenc>(outFormat, i), asset::getFormatMaxValue<Tenc>(outFormat, i), encodeBuffer + i);
					*encodeValue = core::clamp(*encodeValue, min, max);
				}
			}

			asset::encodePixelsRuntime(outFormat, dstPix, encodeBuffer);
		}
};

/*
	Non-dither version of CSwizzleableAndDitherableFilterBase with custom compile time swizzle.

	Normalize paramteter is required to decide whether the
	values being encoded should be previously normalized.

	Clamping paramter is required to decide whether values
	being encoded will be previously clamped to valid format
	max values.
*/
template<typename Swizzle, typename Normalization, bool Clamp>
class CSwizzleableAndDitherableFilterBase<Swizzle,IdentityDither,Normalization,Clamp>
{
	public:
		virtual ~CSwizzleableAndDitherableFilterBase() {}

		class CState : public Swizzle
		{
			public:
				CState() {}
				virtual ~CState() {}

				conditional_normalization_state<Normalization> normalization;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state->normalization.validate())
				return false;
			return true;
		}

		/*
			Performs decode doing swizzle at first on a given pointer
			and putting the result to the encodeBuffer.

			The function supports compile-time decode.
		*/
		template<E_FORMAT inFormat, typename Tdec, typename Tenc>
		static void onDecode(state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
		{
			static_assert(sizeof(Tdec)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			asset::decodePixels<inFormat>(srcPix, decodeBuffer, blockX, blockY);
			// TODO: shouldn't swizzles be performed in-place on decode buffers and their values?
			static_cast<Swizzle&>(*state).operator()<Tdec,Tenc>(decodeBuffer,encodeBuffer);
			state->normalization.operator()<Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Runtime version of onDecode

			@see onDecode
		*/
		template<typename Tdec, typename Tenc>
		static void onDecode(E_FORMAT inFormat, state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
		{
			static_assert(sizeof(Tdec)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			asset::decodePixelsRuntime(inFormat, srcPix, decodeBuffer, blockX, blockY);
			// TODO: shouldn't swizzles be performed in-place on decode buffers and their values?
			static_cast<Swizzle&>(*state).operator()<Tdec,Tenc>(decodeBuffer,encodeBuffer);
			state->normalization.operator()<Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Performs encode.
			The encode buffer is a buffer holding decoded (and swizzled optionally) values.

			The function may perform clamping.
			@see CSwizzleableAndDitherableFilterBase

			The function may perform normalizing if normalizing is
			available and query normalize bool is true.
			@see detail::CNormalizeState

			The function supports compile-time encode.
		*/
		template<E_FORMAT outFormat, typename Tenc>
		static void onEncode(state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels, bool queryNormalizing = false)
		{
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			
			if (queryNormalizing)
				state->normalization.operator()<outFormat,Tenc>(encodeBuffer,position,blockX,blockY,channels);

			if constexpr (Clamp)
			{
				for (uint8_t i = 0; i < channels; ++i)
				{
					auto&& [min, max, encodeValue] = std::make_tuple<Tenc&&, Tenc&&, Tenc*>(asset::getFormatMinValue<Tenc>(outFormat, i), asset::getFormatMaxValue<Tenc>(outFormat, i), encodeBuffer + i);
					*encodeValue = core::clamp(*encodeValue, min, max);
				}
			}

			asset::encodePixels<outFormat>(dstPix, encodeBuffer);
		}

		/*
			Runtime version of onEncode

			@see onEncode
		*/
		template<typename Tenc>
		static void onEncode(E_FORMAT outFormat, state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels, bool queryNormalizing = false)
		{
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");

			if (queryNormalizing)
				state->normalization.operator()<Tenc>(outFormat,encodeBuffer,position,blockX,blockY,channels);

			if constexpr (Clamp)
			{
				for (uint8_t i = 0; i < channels; ++i)
				{
					auto&& [min, max, encodeValue] = std::make_tuple<Tenc&&, Tenc&&, Tenc*>(asset::getFormatMinValue<Tenc>(outFormat, i), asset::getFormatMaxValue<Tenc>(outFormat, i), encodeBuffer + i);
					*encodeValue = core::clamp(*encodeValue, min, max);
				}
			}

			asset::encodePixelsRuntime(outFormat, dstPix, encodeBuffer);
		}
};

/*
	Swizzle-runtime version of CSwizzleableAndDitherableFilterBase with custom dither.

	Normalize paramteter is required to decide whether the
	values being encoded should be previously normalized.

	Clamping paramter is required to decide whether values
	being encoded will be previously clamped to valid format
	max values.
*/
template<typename Dither, typename Normalization, bool Clamp>
class CSwizzleableAndDitherableFilterBase<PolymorphicSwizzle,Dither,Normalization,Clamp>
{
	public:
		virtual ~CSwizzleableAndDitherableFilterBase() {}

		class CState
		{
			public:
				CState() {}
				virtual ~CState() {}

				PolymorphicSwizzle* swizzle;

				Dither dither;
				using DitherState = typename Dither::state_type;
				DitherState* ditherState = nullptr;					//! Allocation, creation of the dither state and making the pointer valid is necessary!

				conditional_normalization_state<Normalization> normalization;
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state->swizzle)
				return false;

			if (!state->ditherState)
				return false;

			if (!normalization.validate())
				return false;

			return true;
		}
					
		/*
			Performs decode doing swizzle at first on a given pointer
			and putting the result to the encodeBuffer.

			The function supports compile-time decode.
		*/
		template<E_FORMAT inFormat, typename Tdec, typename Tenc>
		static void onDecode(state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
		{
			static_assert(sizeof(Tdec)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			asset::decodePixels<inFormat>(srcPix, decodeBuffer, blockX, blockY);
			// TODO: shouldn't swizzles be performed in-place on decode buffers and their values?
			static_cast<Swizzle&>(*state).operator()<Tdec,Tenc>(decodeBuffer,encodeBuffer);
			state->normalization.operator()<Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Runtime version of onDecode

			@see onDecode
		*/

		template<typename Tdec, typename Tenc>
		static void onDecode(E_FORMAT inFormat, state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
		{
			static_assert(sizeof(Tdec)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			asset::decodePixelsRuntime(inFormat, srcPix, decodeBuffer, blockX, blockY);
			// TODO: shouldn't swizzles be performed in-place on decode buffers and their values?
			static_cast<Swizzle&>(*state).operator()<Tdec,Tenc>(decodeBuffer,encodeBuffer);
			state->normalization.operator()<Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Performs encode doing dithering at first on a given encode buffer in pointer.
			The encode buffer is a buffer holding decoded (and swizzled optionally) values.

			The function may perform clamping.
			@see CSwizzleableAndDitherableFilterBase

			The function may perform normalizing if normalizing is
			available and query normalize bool is true.
			@see detail::CNormalizeState

			The function supports compile-time encode.
		*/
		template<E_FORMAT outFormat, typename Tenc>
		static void onEncode(state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels, bool queryNormalizing = false)
		{
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			for (uint8_t i = 0; i < channels; ++i)
			{
				const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
				auto* encodeValue = encodeBuffer + i;
				const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
				*encodeValue += static_cast<Tenc>(ditheredValue) * scale;
			}
			
			if (queryNormalizing)
				state->normalization.operator()<outFormat,Tenc>(encodeBuffer,position,blockX,blockY,channels);

			if constexpr (Clamp)
			{
				for (uint8_t i = 0; i < channels; ++i)
				{
					auto&& [min, max, encodeValue] = std::make_tuple<Tenc&&, Tenc&&, Tenc*>(asset::getFormatMinValue<Tenc>(outFormat, i), asset::getFormatMaxValue<Tenc>(outFormat, i), encodeBuffer + i);
					*encodeValue = core::clamp(*encodeValue, min, max);
				}
			}

			asset::encodePixels<outFormat>(dstPix, encodeBuffer);
		}

		/*
			Runtime version of onEncode

			@see onEncode
		*/
		template<typename Tenc>
		static void onEncode(E_FORMAT outFormat, state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels, bool queryNormalizing = false)
		{
			static_assert(sizeof(Tenc)==8u, "Encode/Decode types must be double, int64_t or uint64_t!");
			for (uint8_t i = 0; i < channels; ++i)
			{
				const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
				auto* encodeValue = encodeBuffer + i;
				const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
				*encodeValue += static_cast<Tenc>(ditheredValue)* scale;
			}

			if (queryNormalizing)
				state->normalization.operator()<Tenc>(outFormat,encodeBuffer,position,blockX,blockY,channels);

			if constexpr (Clamp)
			{
				for (uint8_t i = 0; i < channels; ++i)
				{
					auto&& [min, max, encodeValue] = std::make_tuple<Tenc&&, Tenc&&, Tenc*>(asset::getFormatMinValue<Tenc>(outFormat, i), asset::getFormatMaxValue<Tenc>(outFormat, i), encodeBuffer + i);
					*encodeValue = core::clamp(*encodeValue, min, max);
				}
			}

			asset::encodePixelsRuntime(outFormat, dstPix, encodeBuffer);
		}
};


} // end namespace nbl::asset::impl

#endif
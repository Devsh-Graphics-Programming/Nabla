// Copyright (C) 2020 AnastaZIuk
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__
#define __IRR_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__

#include "irr/core/core.h"
#include "irr/asset/format/convertColor.h"
#include "irr/asset/ICPUImageView.h"
#include "irr/asset/filters/dithering/CDither.h"
#include <type_traits>

namespace irr
{
	namespace asset
	{
		namespace impl
		{
			/*
				Common base class Swizzleable or Ditherable ones
				with custom compile time swizzle and custom dither.

				Clamping paramter is required to decide whether values
				being encoded will be previously clamped to valid format 
				max values.
			*/

			template<bool Clamp, typename Swizzle, typename Dither>
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
					};
					using state_type = CState;

					static inline bool validate(state_type* state)
					{
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
						asset::decodePixels<inFormat>(srcPix, decodeBuffer, blockX, blockY);
						static_cast<Swizzle&>(*state).operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
					}

					/*
						Performs encode doing dithering at first on a given encode buffer in pointer.
						The encode buffer is a buffer holding decoded (and swizzled optionally) values.

						The function may perform clamping.
						@see CSwizzleableAndDitherableFilterBase

						The function supports compile-time encode.
					*/

					template<E_FORMAT outFormat, typename Tenc>
					static void onEncode(state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)
					{
						for (uint8_t i = 0; i < channels; ++i)
						{
							const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
							auto* encodeValue = encodeBuffer + i;
							const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
							*encodeValue += static_cast<Tenc>(ditheredValue)* scale;
						}

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
			};

			/*
				Non-dither version of CSwizzleableAndDitherableFilterBase with custom compile time swizzle.

				Clamping paramter is required to decide whether values
				being encoded will be previously clamped to valid format
				max values.
			*/

			template<bool Clamp, typename Swizzle>
			class CSwizzleableAndDitherableFilterBase<Clamp, Swizzle, IdentityDither>
			{
				public:
					virtual ~CSwizzleableAndDitherableFilterBase() {}

					class CState : public Swizzle
					{
						public:
							CState() {}
							virtual ~CState() {}
					};
					using state_type = CState;

					static inline bool validate(state_type* state)
					{
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
						asset::decodePixels<inFormat>(srcPix, decodeBuffer, blockX, blockY);
						static_cast<Swizzle&>(*state).operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
					}

					/*
						Performs encode.
						The encode buffer is a buffer holding decoded (and swizzled optionally) values.

						The function may perform clamping.
						@see CSwizzleableAndDitherableFilterBase

						The function supports compile-time encode.
					*/

					template<E_FORMAT outFormat, typename Tenc>
					static void onEncode(state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)
					{
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
			};

			/*
				Swizzle-runtime version of CSwizzleableAndDitherableFilterBase with custom dither.

				Clamping paramter is required to decide whether values
				being encoded will be previously clamped to valid format
				max values.
			*/
			
			template<bool Clamp, typename Dither>
			class CSwizzleableAndDitherableFilterBase<Clamp, PolymorphicSwizzle, Dither>
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
					};
					using state_type = CState;

					static inline bool validate(state_type* state)
					{
						if (!state->swizzle)
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
						asset::decodePixels<inFormat>(srcPix, decodeBuffer, blockX, blockY);
						state->swizzle->operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
					}

					/*
						Performs encode doing dithering at first on a given encode buffer in pointer.
						The encode buffer is a buffer holding decoded (and swizzled optionally) values.

						The function may perform clamping.
						@see CSwizzleableAndDitherableFilterBase

						The function supports compile-time encode.
					*/

					template<E_FORMAT outFormat, typename Tenc>
					static void onEncode(state_type* state, void* dstPix, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)
					{
						for (uint8_t i = 0; i < channels; ++i)
						{
							const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
							auto* encodeValue = encodeBuffer + i;
							const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
							*encodeValue += static_cast<Tenc>(ditheredValue) * scale;
						}

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


			};
		}

		/*
			Default Swizzle for compile time cases
		*/

		struct DefaultSwizzle
		{
			ICPUImageView::SComponentMapping swizzle;

			/*
				Performs swizzle on out compoments following
				swizzle member with all four compoments. You 
				can specify channels for custom pointers.
			*/

			template<typename InT, typename OutT>
			void operator()(const InT* in, OutT* out, uint8_t channels = SwizzleBase::MaxChannels) const;
		};

		template<>
		inline void DefaultSwizzle::operator() <void, void> (const void* in, void* out, uint8_t channels) const
		{
			operator()(reinterpret_cast<const uint64_t*>(in), reinterpret_cast<uint64_t*>(out), channels);
		}

		template<typename InT, typename OutT>
		inline void DefaultSwizzle::operator()(const InT* in, OutT* out, uint8_t channels) const
		{
			auto getComponent = [&in](ICPUImageView::SComponentMapping::E_SWIZZLE s, auto id) -> InT
			{
				if (s < ICPUImageView::SComponentMapping::ES_IDENTITY)
					return in[id];
				else if (s < ICPUImageView::SComponentMapping::ES_ZERO)
					return InT(0);
				else if (s == ICPUImageView::SComponentMapping::ES_ONE)
					return InT(1);
				else
					return in[s - ICPUImageView::SComponentMapping::ES_R];
			};
			for (auto i = 0; i < channels; i++)
				out[i] = OutT(getComponent((&swizzle.r)[i], i));
		}
	} // end namespace asset
} // end namespace irr

#endif // __IRR_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__
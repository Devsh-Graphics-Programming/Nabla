// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__
#define __NBL_ASSET_C_SWIZZLEABLE_AND_DITHERABLE_FILTER_BASE_H_INCLUDED__

#include "nbl/core/declarations.h"
#include "nbl/asset/format/convertColor.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/filters/dithering/CDither.h"
#include <type_traits>

namespace nbl
{
	namespace asset
	{
		namespace impl
		{
			namespace detail
			{
				union NormalizeValues
				{
					NormalizeValues() {}
					~NormalizeValues() {}

					NormalizeValues(NormalizeValues& copy)
					{
						std::memmove(this, &copy, sizeof(NormalizeValues));
					}

					NormalizeValues(const NormalizeValues& copy)
					{
						std::memmove(this, &copy, sizeof(NormalizeValues));
					}

					NormalizeValues& operator=(NormalizeValues& copy)
					{
						std::memmove(this, &copy, sizeof(NormalizeValues));
						return *this;
					}

					NormalizeValues& operator=(const NormalizeValues& copy)
					{
						std::memmove(this, &copy, sizeof(NormalizeValues));
						return *this;
					}

					core::vectorSIMDu32 f;
					core::vectorSIMDu32 u;
					core::vectorSIMDi32 i;
				};

				//! Normalizing state
				/*
					The class provides mininum and maximum values per texel
					that may be used for normalizing texels.
				*/

				template<bool Normalize>
				class CNormalizeState
				{
					public:
						CNormalizeState() {}
						virtual ~CNormalizeState() {}

						NormalizeValues oldMaxValue, oldMinValue;

					protected:

						/*
							The function normalizes curret texel passed to the function
							using oldMaxValue and oldMinValue provided by the state.

							It's a user or a programmer responsibility to update those
							variables to contain proper values for normalizing the texel.

							The function supports compile-time normalize.
						*/

						template<E_FORMAT format, typename Tenc>
						void normalize(Tenc* encodeBuffer, const uint8_t& channels)
						{
							static_assert(!isSignedFormat<format>(), "The format musn't be a pure-integer!");

							if constexpr (isSignedFormat<format>())
								for (uint8_t channel = 0; channel < channels; ++channel)
									encodeBuffer[channel] = (2.0 * encodeBuffer[channel] - oldMaxValue.f[channel] - oldMinValue.f[channel]) / (oldMaxValue.f[channel] - oldMinValue.f[channel]);
							else
								for (uint8_t channel = 0; channel < channels; ++channel)
									encodeBuffer[channel] = (encodeBuffer[channel] - oldMinValue.f[channel]) / (oldMaxValue.f[channel] - oldMinValue.f[channel]);
						}

						/*
							Runtime version of normalize

							@see normalize
						*/

						template<typename Tenc>
						void normalize(const E_FORMAT& format, Tenc* encodeBuffer, const uint8_t& channels)
						{
							#ifdef _NBL_DEBUG
							bool status = !isScaledFormat(format);
							assert(status);
							#endif // _NBL_DEBUG

							if (isSignedFormat(format))
								for (uint8_t channel = 0; channel < channels; ++channel)
									encodeBuffer[channel] = (2.0 * encodeBuffer[channel] - oldMaxValue.f[channel] - oldMinValue.f[channel]) / (oldMaxValue.f[channel] - oldMinValue.f[channel]);
							else
								for (uint8_t channel = 0; channel < channels; ++channel)
									encodeBuffer[channel] = (encodeBuffer[channel] - oldMinValue.f[channel]) / (oldMaxValue.f[channel] - oldMinValue.f[channel]);
						}
				};

				template<>
				class CNormalizeState<false>
				{
					public:
						CNormalizeState() {}
						virtual ~CNormalizeState() {}

						NormalizeValues oldMaxValue, oldMinValue;
				};
			}

			/*
				Common base class for Swizzleable or Ditherable ones
				with custom compile time swizzle and custom dither.

				Normalize paramteter is required to decide whether the
				values being encoded should be previously normalized.

				Clamping paramter is required to decide whether values
				being encoded should be previously clamped to valid format 
				max and min values.
			*/

			template<bool Normalize, bool Clamp, typename Swizzle, typename Dither>
			class CSwizzleableAndDitherableFilterBase
			{
				public:
					class CState : public Swizzle, public detail::CNormalizeState<Normalize>
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
						static_cast<Swizzle&>(*state).template operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
					}

					/*
						Runtime version of onDecode

						@see onDecode
					*/

					template<typename Tdec, typename Tenc>
					static void onDecode(E_FORMAT inFormat, state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
					{
						asset::decodePixelsRuntime(inFormat, srcPix, decodeBuffer, blockX, blockY);
						static_cast<Swizzle&>(*state).template operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
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
						for (uint8_t i = 0; i < channels; ++i)
						{
							const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
							auto* encodeValue = encodeBuffer + i;
							const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
							*encodeValue += static_cast<Tenc>(ditheredValue) * scale;
						}

						if constexpr (Normalize)
							if (queryNormalizing)
								static_cast<detail::CNormalizeState<Normalize>&>(*state).normalize<outFormat>(encodeBuffer, channels);

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
						for (uint8_t i = 0; i < channels; ++i)
						{
							const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
							auto* encodeValue = encodeBuffer + i;
							const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
							*encodeValue += static_cast<Tenc>(ditheredValue)* scale;
						}

						if constexpr (Normalize)
							if (queryNormalizing)
								static_cast<detail::CNormalizeState<Normalize>&>(*state).normalize(outFormat, encodeBuffer, channels);

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

			template<bool Normalize, bool Clamp, typename Swizzle>
			class CSwizzleableAndDitherableFilterBase<Normalize, Clamp, Swizzle, IdentityDither>
			{
				public:
					virtual ~CSwizzleableAndDitherableFilterBase() {}

					class CState : public Swizzle, public detail::CNormalizeState<Normalize>
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
						static_cast<Swizzle&>(*state).template operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
					}

					/*
						Runtime version of onDecode

						@see onDecode
					*/

					template<typename Tdec, typename Tenc>
					static void onDecode(E_FORMAT inFormat, state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
					{
						asset::decodePixelsRuntime(inFormat, srcPix, decodeBuffer, blockX, blockY);
						static_cast<Swizzle&>(*state).template operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
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
						if constexpr (Normalize)
							if (queryNormalizing)
								static_cast<detail::CNormalizeState<Normalize>&>(*state).normalize<outFormat>(encodeBuffer, channels);

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
						if constexpr (Normalize)
							if (queryNormalizing)
								static_cast<detail::CNormalizeState<Normalize>&>(*state).normalize(outFormat, encodeBuffer, channels);

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
			
			template<bool Normalize, bool Clamp, typename Dither>
			class CSwizzleableAndDitherableFilterBase<Normalize, Clamp, PolymorphicSwizzle, Dither>
			{
				public:
					virtual ~CSwizzleableAndDitherableFilterBase() {}

					class CState : public detail::CNormalizeState<Normalize>
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
						state->swizzle->template operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
					}

					/*
						Runtime version of onDecode

						@see onDecode
					*/

					template<typename Tdec, typename Tenc>
					static void onDecode(E_FORMAT inFormat, state_type* state, const void* srcPix[4], Tdec* decodeBuffer, Tenc* encodeBuffer, uint32_t blockX, uint32_t blockY)
					{
						asset::decodePixelsRuntime(inFormat, srcPix, decodeBuffer, blockX, blockY);
						state->swizzle->template operator() < Tdec, Tenc > (decodeBuffer, encodeBuffer);
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
						for (uint8_t i = 0; i < channels; ++i)
						{
							const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
							auto* encodeValue = encodeBuffer + i;
							const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
							*encodeValue += static_cast<Tenc>(ditheredValue) * scale;
						}

						if constexpr (Normalize)
							if (queryNormalizing)
								static_cast<detail::CNormalizeState<Normalize>&>(*state).normalize<outFormat>(encodeBuffer, channels);

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
						for (uint8_t i = 0; i < channels; ++i)
						{
							const float ditheredValue = state->dither.pGet(state->ditherState, position + core::vectorSIMDu32(blockX, blockY), i);
							auto* encodeValue = encodeBuffer + i;
							const Tenc scale = asset::getFormatPrecision<Tenc>(outFormat, i, *encodeValue);
							*encodeValue += static_cast<Tenc>(ditheredValue)* scale;
						}

						if constexpr (Normalize)
							if (queryNormalizing)
								static_cast<detail::CNormalizeState<Normalize>&>(*state).normalize(outFormat, encodeBuffer, channels);

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
} // end namespace nbl

#endif
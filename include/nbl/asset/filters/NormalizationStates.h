// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_NORMALIZATION_STATES_H_INCLUDED_
#define _NBL_ASSET_NORMALIZATION_STATES_H_INCLUDED_

#include "nbl/core/core.h"

#include <type_traits>

namespace nbl::asset
{

//! Global per channel Normalizing state
// TODO: support using the min/max from Tdec values
class CGlobalNormalizationState
{
	public:
		inline bool validate() const {return true;}

		/*
			Need to be called prior to `void operator()(const Tdec* decodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)`
		*/
		template<typename Tdec>
		inline void initialize()
		{
			static_assert(std::is_floating_point_v<Tdec>, "Integer decode not supported yet!");
			for (auto i=0u; i<4u; i++)
			{
				oldMaxValue.f[i] = -FLT_MAX;
				oldMinValue.f[i] = FLT_MAX;
			}
		}

		/*
			The function examines a decoded pixel and changes oldMaxValue and oldMinValue as appropriate.
		*/
		template<typename Tdec>
		void operator()(const Tdec* decodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)
		{
			static_assert(std::is_floating_point_v<Tdec>, "Integer decode not supported yet!");
			for (uint8_t channel=0u; channel<channels; ++channel)
			{
				const auto val = decodeBuffer[channel];
				if constexpr (std::is_floating_point_v<Tdec>)
				{
					if (val<oldMinValue.f[i]) oldMinValue.f[i] = val;
					if (val>oldMaxValue.f[i]) oldMaxValue.f[i] = val;
				}
			}
		}

		/*
			The function normalizes current texel passed to the function
			using oldMaxValue and oldMinValue provided by the state.

			The function supports compile-time normalize.
		*/
		template<E_FORMAT format, typename Tenc>
		void operator()(Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(std::is_floating_point_v<Tenc>, "Encode/Decode types must be double or float!");
			static_assert(isFloatingPointFormat<format>()||isNormalizedFormat<format>(), "The format musn't be a pure-integer!");

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
		void operator()(E_FORMAT format, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(std::is_floating_point_v<Tenc>, "Encode/Decode types must be double or float!");
			#ifdef _NBL_DEBUG
			bool status = isFloatingPointFormat(format)||isNormalizedFormat(format);
			assert(status);
			#endif // _NBL_DEBUG

			if (isSignedFormat(format))
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = (2.0 * encodeBuffer[channel] - oldMaxValue.f[channel] - oldMinValue.f[channel]) / (oldMaxValue.f[channel] - oldMinValue.f[channel]);
			else
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = (encodeBuffer[channel] - oldMinValue.f[channel]) / (oldMaxValue.f[channel] - oldMinValue.f[channel]);
		}

	protected:
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

			core::vectorSIMDf f;
			core::vectorSIMDu32 u;
			core::vectorSIMDi32 i;
		};
		NormalizeValues oldMaxValue,oldMinValue;
};

//! Derivative Map Normalizing state
// TODO: support using the min/max from Tdec values
class CDerivativeMapNormalizationState
{
	public:
		inline bool validate() const {return true;}

		/*
			Need to be called prior to `void operator()(const Tdec* decodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)`
		*/
		template<typename Tdec>
		inline void initialize()
		{
			static_assert(std::is_floating_point_v<Tdec>, "Integer decode not supported yet!");
			maxAbsPerChannel.set(0.f,0.f,0.f,0.f);
		}

		//
		template<typename Tdec>
		void operator()(const Tdec* decodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)
		{
			static_assert(std::is_floating_point_v<Tdec>, "Integer decode not supported yet!");
			for (uint8_t channel=0u; channel<channels; ++channel)
			{
				const auto val = core::abs(decodeBuffer[channel]);
				if (val>maxAbsPerChannel[i]) maxAbsPerChannel[i] = val;
			}
		}

		//
		template<E_FORMAT format, typename Tenc>
		void operator()(Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(std::is_floating_point_v<Tenc>, "Encode/Decode types must be double or float!");
			static_assert(isFloatingPointFormat<format>()||isNormalizedFormat<format>(), "The format musn't be a pure-integer!");

			if constexpr (isSignedFormat<format>())
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = encodeBuffer[channel]/maxAbsPerChannel[channel];
			else
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = encodeBuffer[channel]*0.5f/maxAbsPerChannel[channel]+0.5f;
		}

		/*
			Runtime version of normalize

			@see normalize
		*/
		template<typename Tenc>
		void operator()(E_FORMAT format, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(std::is_floating_point_v<Tenc>, "Encode/Decode types must be double or float!");
			#ifdef _NBL_DEBUG
			bool status = isFloatingPointFormat(format)||isNormalizedFormat(format);
			assert(status);
			#endif // _NBL_DEBUG

			if (isSignedFormat(format))
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = encodeBuffer[channel]/maxAbsPerChannel[channel];
			else
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = encodeBuffer[channel]*0.5f/maxAbsPerChannel[channel]+0.5f;
		}

	protected:
		core::vectorSIMDf maxAbsPerChannel;
};

//! Wrapper that makes it easy to put inside states
template<class NormalizationState>
class conditional_normalization_state;

template<>
class conditional_normalization_state<void>
{
	public:
		inline bool validate() const {return true;}

		template<typename Tdec>
		inline void initialize()
		{
		}

		template<typename Tdec>
		void operator()(const Tdec* decodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels)
		{
		}

		template<E_FORMAT format, typename Tenc>
		void operator()(Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
		}

		template<typename Tenc>
		void operator()(E_FORMAT format, Tenc* encodeBuffer, core::vectorSIMDu32 position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
		}
};

template<class NormalizationState>
class conditional_normalization_state : public NormalizationState
{
};

} // end namespace nbl::asset

#endif
// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_NORMALIZATION_STATES_H_INCLUDED_
#define _NBL_ASSET_NORMALIZATION_STATES_H_INCLUDED_

#include "nbl/core/declarations.h"

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
			Needs to be called prior to `void prepass`
		*/
		template<typename Tdec>
		inline void initialize()
		{
			static_assert(std::is_floating_point_v<Tdec>, "Integer decode not supported yet!");
			for (auto i=0u; i<4u; i++)
			{
				oldMinValue[i] = FLT_MAX;
				oldMaxValue[i] = -FLT_MAX;
			}
		}

		/*
			The function examines a pixel and changes oldMaxValue and oldMinValue as appropriate.
		*/
		template<typename Tenc>
		void prepass(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels)
		{
			static_assert(std::is_floating_point_v<Tenc>, "Integer decode not supported yet!");
			for (uint8_t channel=0u; channel<channels; ++channel)
			{
				const auto val = encodeBuffer[channel];
				if constexpr (std::is_floating_point_v<Tenc>)
				{
					core::atomic_fetch_min(oldMinValue+channel,val);
					core::atomic_fetch_max(oldMaxValue+channel,val);
				}
			}
		}

		/*
			Needs to be called prior to `void operator()`
		*/
		template<typename Tdec>
		inline void finalize() {}

		/*
			The function normalizes current texel passed to the function
			using oldMaxValue and oldMinValue provided by the state.

			The function supports compile-time normalize.
		*/
		template<E_FORMAT format, typename Tenc>
		void operator()(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(isFloatingPointFormat<format>()||isNormalizedFormat<format>(), "The format musn't be a pure-integer!");
			impl<isSignedFormat<format>(),Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		/*
			Runtime version of normalize

			@see normalize
		*/
		template<typename Tenc>
		void operator()(E_FORMAT format, Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			#ifdef _NBL_DEBUG
			bool status = isFloatingPointFormat(format)||isNormalizedFormat(format);
			assert(status);
			#endif // _NBL_DEBUG
			if (isSignedFormat(format))
				impl<true,Tenc>(encodeBuffer,position,blockX,blockY,channels);
			else
				impl<false,Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		core::atomic<float> oldMinValue[4];
		core::atomic<float> oldMaxValue[4];
	protected:
		template<bool isSignedFormat, typename Tenc>
		void impl(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(std::is_floating_point_v<Tenc>, "Encode types must be double or float!");

			constexpr float Threshold = 1e-6f;
			if constexpr (isSignedFormat)
			{
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = (oldMaxValue[channel] - oldMinValue[channel]) < Threshold ? 1.f : (2.0 * encodeBuffer[channel] - oldMaxValue[channel] - oldMinValue[channel]) / (oldMaxValue[channel] - oldMinValue[channel]);
			}
			else
			{
				for (uint8_t channel = 0; channel < channels; ++channel)
					encodeBuffer[channel] = (oldMaxValue[channel] - oldMinValue[channel]) < Threshold ? 1.f : (encodeBuffer[channel] - oldMinValue[channel]) / (oldMaxValue[channel] - oldMinValue[channel]);
			}
		}
};


//! Derivative Map Normalizing state
namespace impl
{

class CDerivativeMapNormalizationStateBase
{
	protected:
		// we only care about R and G
		static inline constexpr uint32_t kChannels = 2;

		template<bool isSignedFormat, typename Tenc>
		void impl(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(std::is_floating_point_v<Tenc>, "Encode types must be double or float!");

			if constexpr (isSignedFormat)
				for (uint8_t channel = 0; channel < kChannels; ++channel)
					encodeBuffer[channel] = encodeBuffer[channel]/maxAbsPerChannel[channel];
			else
				for (uint8_t channel = 0; channel < kChannels; ++channel)
					encodeBuffer[channel] = encodeBuffer[channel]*0.5f/maxAbsPerChannel[channel]+0.5f;
		}

	public:
		inline bool validate() const {return true;}

		//
		template<typename Tenc>
		inline void initialize()
		{
			static_assert(std::is_floating_point_v<Tenc>, "Integer encode not supported yet!");
			std::fill_n(maxAbsPerChannel,kChannels,0.f);
		}

		//
		template<typename Tenc>
		void prepass(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels)
		{
			static_assert(std::is_floating_point_v<Tenc>, "Integer encode not supported yet!");
			for (uint8_t channel=0u; channel<kChannels; ++channel)
				core::atomic_fetch_max(maxAbsPerChannel+channel,core::abs(encodeBuffer[channel]));
		}

		//
		template<E_FORMAT format, typename Tenc>
		void operator()(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			static_assert(isFloatingPointFormat<format>()||isNormalizedFormat<format>(), "The format musn't be a pure-integer!");
			impl<isSignedFormat<format>(),Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		//
		template<typename Tenc>
		void operator()(E_FORMAT format, Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
			assert(channels>=kChannels);
			#ifdef _NBL_DEBUG
			bool status = isFloatingPointFormat(format)||isNormalizedFormat(format);
			assert(status);
			#endif // _NBL_DEBUG

			if (isSignedFormat(format))
				impl<true,Tenc>(encodeBuffer,position,blockX,blockY,channels);
			else
				impl<false,Tenc>(encodeBuffer,position,blockX,blockY,channels);
		}

		core::atomic<float> maxAbsPerChannel[kChannels];
};

}

template<bool isotropic>
class CDerivativeMapNormalizationState : public impl::CDerivativeMapNormalizationStateBase
{
	public:
		template<typename Tenc>
		inline void finalize()
		{
			static_assert(std::is_floating_point_v<Tenc>, "Integer encode types not supported yet!");
			if constexpr (isotropic)
			{
				const float isotropicMax = core::max<float>(maxAbsPerChannel[0],maxAbsPerChannel[1]);
				for (auto i=0u; i<kChannels; i++)
					maxAbsPerChannel[i] = isotropicMax;
			}
		}
};

//! Wrapper that makes it easy to put inside states
template<class NormalizationState>
class conditional_normalization_state;

template<>
class conditional_normalization_state<void>
{
	public:
		inline bool validate() const {return true;}

		template<typename Tenc>
		inline void initialize()
		{
		}

		template<typename Tenc>
		void prepass(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels)
		{
		}

		template<typename Tenc>
		inline void finalize()
		{
		}

		template<E_FORMAT format, typename Tenc>
		void operator()(Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
		}

		template<typename Tenc>
		void operator()(E_FORMAT format, Tenc* encodeBuffer, const core::vectorSIMDu32& position, uint32_t blockX, uint32_t blockY, uint8_t channels) const
		{
		}
};

template<class NormalizationState>
class conditional_normalization_state : public NormalizationState
{
};

} // end namespace nbl::asset

#endif
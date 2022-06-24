// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BLIT_UTILITIES_H_INCLUDED__
#define __NBL_ASSET_C_BLIT_UTILITIES_H_INCLUDED__

#include "nbl/asset/filters/kernels/kernels.h"

namespace nbl::asset
{
class IBlitUtilities
{
public:
	enum E_ALPHA_SEMANTIC : uint32_t
	{
		EAS_NONE_OR_PREMULTIPLIED = 0u, // just filter the channels independently (also works for a texture for blending equation `dstCol*(1-srcAlpha)+srcCol`)
		EAS_REFERENCE_OR_COVERAGE, // try to preserve coverage (percentage of pixels above a threshold value) across mipmap levels
		EAS_SEPARATE_BLEND, // compute a new alpha value for a texture to be used with the blending equation `mix(dstCol,srcCol,srcAlpha)`
		EAS_COUNT
	};

	static inline core::vectorSIMDu32 getPhaseCount(const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType)
	{
		core::vectorSIMDu32 result(0u);
		for (uint32_t i = 0u; i <= inImageType; ++i)
			result[i] = outExtent[i] / std::gcd(inExtent[i], outExtent[i]);
		return result;
	}

	// we'll need to rescale the kernel support to be relative to the output image but in the input image coordinate system
	// (if support is 3 pixels, it needs to be 3 output texels, but measured in input texels)
	template<class Kernel>
	static inline auto constructScaledKernel(const Kernel& kernel, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent)
	{
		const core::vectorSIMDf fInExtent(inExtent);
		const core::vectorSIMDf fOutExtent(outExtent);
		const auto fScale = fInExtent.preciseDivision(fOutExtent);
		return CScaledImageFilterKernel<Kernel>(fScale, kernel);
	}
};

template <class KernelX = CBoxImageFilterKernel, class KernelY = KernelX, class KernelZ = KernelX>
class CBlitUtilities : public IBlitUtilities
{
	static_assert(std::is_same<typename KernelX::value_type, typename KernelY::value_type>::value&& std::is_same<typename KernelZ::value_type, typename KernelY::value_type>::value, "Kernel value_type need to be identical");

public:
	_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = std::max<decltype(KernelX::MaxChannels)>(std::max<decltype(KernelX::MaxChannels)>(KernelX::MaxChannels, KernelY::MaxChannels), KernelZ::MaxChannels);

	template <typename lut_value_type = KernelX::value_type>
	static inline size_t getScaledKernelPhasedLUTSize(const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ)
	{
		const auto scaledKernelX = constructScaledKernel(kernelX, inExtent, outExtent);
		const auto scaledKernelY = constructScaledKernel(kernelY, inExtent, outExtent);
		const auto scaledKernelZ = constructScaledKernel(kernelZ, inExtent, outExtent);

		const auto phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

		return ((phaseCount[0] * scaledKernelX.getWindowSize().x) + (phaseCount[1] * scaledKernelY.getWindowSize().y) + (phaseCount[2] * scaledKernelZ.getWindowSize().z)) * sizeof(lut_value_type) * MaxChannels;
	}

	template <typename lut_value_type = KernelX::value_type>
	static bool computeScaledKernelPhasedLUT(void* outKernelWeights, const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType,
		const KernelX& kernelX, const KernelY& kernelY, const KernelZ& kernelZ)
	{
		const core::vectorSIMDu32 phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

		for (auto i = 0; i <= inImageType; ++i)
		{
			if (phaseCount[i] == 0)
				return false;
		}

		const auto scaledKernelX = constructScaledKernel(kernelX, inExtent, outExtent);
		const auto scaledKernelY = constructScaledKernel(kernelY, inExtent, outExtent);
		const auto scaledKernelZ = constructScaledKernel(kernelZ, inExtent, outExtent);

		const auto windowDims = getRealWindowSize(inImageType, scaledKernelX, scaledKernelY, scaledKernelZ);
		const auto axisOffsets = getScaledKernelPhasedLUTAxisOffsets<lut_value_type>(phaseCount, windowDims);

		const core::vectorSIMDf fInExtent(inExtent);
		const core::vectorSIMDf fOutExtent(outExtent);
		const auto fScale = fInExtent.preciseDivision(fOutExtent);

		// a dummy load functor
		// does nothing but fills up the `windowSample` with 1s (identity) so we can preserve the value of kernel
		// weights when eventually `windowSample` gets multiplied by them later in
		// `CFloatingPointSeparableImageFilterKernelBase<CRTP>::sample_functor_t<PreFilter,PostFilter>::operator()`
		// this exists only because `evaluateImpl` expects a pre filtering step.
		auto dummyLoad = [](double* windowSample, const core::vectorSIMDf&, const core::vectorSIMDi32&, const IImageFilterKernel::UserData*) -> void
		{
			for (auto h = 0; h < MaxChannels; h++)
				windowSample[h] = 1.0;
		};

		double kernelWeight[MaxChannels];
		// actually used to put values in the LUT
		auto dummyEvaluate = [&kernelWeight](const double* windowSample, const core::vectorSIMDf&, const core::vectorSIMDi32&, const IImageFilterKernel::UserData*) -> void
		{
			for (auto h = 0; h < MaxChannels; h++)
				kernelWeight[h] = windowSample[h];
		};

		auto computeForAxis = [&](const asset::IImage::E_TYPE axis, const auto& scaledKernel)
		{
			if (axis > inImageType)
				return;

			const auto windowSize = scaledKernel.getWindowSize()[axis];

			IImageFilterKernel::ScaleFactorUserData scale(1.f / fScale[axis]);
			const IImageFilterKernel::ScaleFactorUserData* otherScale = nullptr;
			switch (axis)
			{
			case IImage::ET_1D:
				otherScale = IImageFilterKernel::ScaleFactorUserData::cast(kernelX.getUserData());
				break;
			case IImage::ET_2D:
				otherScale = IImageFilterKernel::ScaleFactorUserData::cast(kernelY.getUserData());
				break;
			case IImage::ET_3D:
				otherScale = IImageFilterKernel::ScaleFactorUserData::cast(kernelZ.getUserData());
				break;
			}
			if (otherScale)
			{
				for (auto k = 0; k < MaxChannels; k++)
					scale.factor[k] *= otherScale->factor[k];
			}

			lut_value_type* outKernelWeightsPixel = reinterpret_cast<lut_value_type*>(reinterpret_cast<uint8_t*>(outKernelWeights) + axisOffsets[axis]);
			for (uint32_t i = 0u; i < phaseCount[axis]; ++i)
			{
				core::vectorSIMDf tmp(0.f);
				tmp[axis] = float(i) + 0.5f;

				const int32_t windowCoord = scaledKernel.getWindowMinCoord(tmp * fScale, tmp)[axis];

				float relativePos = tmp[axis] - float(windowCoord); // relative position of the last pixel in window from current (ith) output pixel having a unique phase sequence of kernel evaluation points

				for (int32_t j = 0; j < windowSize; ++j)
				{
					core::vectorSIMDf tmp(relativePos, 0.f, 0.f);
					scaledKernel.evaluateImpl(dummyLoad, dummyEvaluate, kernelWeight, tmp, core::vectorSIMDi32(), &scale);
					for (uint32_t ch = 0; ch < MaxChannels; ++ch)
					{
						if constexpr (std::is_same_v<lut_value_type, uint16_t>)
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = core::Float16Compressor::compress(float(kernelWeight[ch]));
						else
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = lut_value_type(kernelWeight[ch]);

					}
					relativePos -= 1.f;
				}
			}
		};

		computeForAxis(asset::IImage::ET_1D, scaledKernelX);
		computeForAxis(asset::IImage::ET_2D, scaledKernelY);
		computeForAxis(asset::IImage::ET_3D, scaledKernelZ);

		return true;
	}

	static inline core::vectorSIMDi32 getRealWindowSize(const IImage::E_TYPE inImageType,
		const CScaledImageFilterKernel<KernelX>& kernelX,
		const CScaledImageFilterKernel<KernelY>& kernelY,
		const CScaledImageFilterKernel<KernelZ>& kernelZ)
	{
		core::vectorSIMDi32 last(kernelX.getWindowSize().x, 0, 0, 0);
		if (inImageType >= IImage::ET_2D)
			last.y = kernelY.getWindowSize().y;
		if (inImageType >= IImage::ET_3D)
			last.z = kernelZ.getWindowSize().z;
		return last;
	}

	template <typename lut_value_type = KernelX::value_type>
	static inline core::vectorSIMDu32 getScaledKernelPhasedLUTAxisOffsets(const core::vectorSIMDu32& phaseCount, const core::vectorSIMDi32& real_window_size)
	{
		core::vectorSIMDu32 result;
		result.x = 0u;
		result.y = (phaseCount[0] * real_window_size.x);
		result.z = ((phaseCount[0] * real_window_size.x) + (phaseCount[1] * real_window_size.y));
		return result * sizeof(lut_value_type) * MaxChannels;
	}
};
}

#endif
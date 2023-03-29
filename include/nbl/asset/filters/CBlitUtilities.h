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
	static constexpr uint32_t MinAlphaBinCount = 256u;
	static constexpr uint32_t MaxAlphaBinCount = 4096u;
	static constexpr uint32_t DefaultAlphaBinCount = MinAlphaBinCount;

	enum E_ALPHA_SEMANTIC : uint32_t
	{
		EAS_NONE_OR_PREMULTIPLIED = 0u, // just filter the channels independently (also works for a texture for blending equation `dstCol*(1-srcAlpha)+srcCol`)
		EAS_REFERENCE_OR_COVERAGE, // try to preserve coverage (percentage of pixels above a threshold value) across mipmap levels
		EAS_SEPARATE_BLEND, // compute a new alpha value for a texture to be used with the blending equation `mix(dstCol,srcCol,srcAlpha)`
		EAS_COUNT
	};

	static inline core::vectorSIMDu32 getPhaseCount(const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const IImage::E_TYPE inImageType)
	{
		core::vectorSIMDu32 result(0u);
		for (uint32_t i = 0u; i <= inImageType; ++i)
			result[i] = outExtent[i] / std::gcd(inExtent[i], outExtent[i]);
		return result;
	}
};

template<
	typename ReconstructionFunctionX	= SBoxFunction,
	typename ResamplingFunctionX		= ReconstructionFunctionX,
	typename ReconstructionFunctionY	= ReconstructionFunctionX,
	typename ResamplingFunctionY		= ResamplingFunctionX,
	typename ReconstructionFunctionZ	= ReconstructionFunctionX,
	typename ResamplingFunctionZ		= ResamplingFunctionX>
class CBlitUtilities : public IBlitUtilities
{
	// TODO(achal): Get MaxChannels somehow.
	static inline constexpr uint32_t MaxChannels = 4;

public:
	template <typename LutDataType>
	static inline size_t getScaledKernelPhasedLUTSize(
		const core::vectorSIMDu32&		inExtent,
		const core::vectorSIMDu32&		outExtent,
		const asset::IImage::E_TYPE		inImageType)
	{
		const auto windowSize = getWindowSize(inExtent, outExtent, inImageType);
		const auto phaseCount = getPhaseCount(inExtent, outExtent, inImageType);
		return ((phaseCount.x * windowSize.x) + (phaseCount.y * windowSize.y) + (phaseCount.z * windowSize.z)) * sizeof(LutDataType) * MaxChannels;
	}

	template <typename LutDataType>
	static bool computeScaledKernelPhasedLUT(
		void*							outKernelWeights,
		const core::vectorSIMDu32&		inExtent,
		const core::vectorSIMDu32&		outExtent,
		const asset::IImage::E_TYPE		inImageType)
	{
		const core::vectorSIMDu32 phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

		for (auto i = 0; i <= inImageType; ++i)
		{
			if (phaseCount[i] == 0)
				return false;
		}

		const auto windowSize = getWindowSize(inExtent, outExtent, inImageType);
		const auto axisOffsets = getScaledKernelPhasedLUTAxisOffsets<LutDataType>(phaseCount, windowSize);

		const core::vectorSIMDf inExtent_f32(inExtent);
		const core::vectorSIMDf outExtent_f32(outExtent);
		const auto scale = inExtent_f32.preciseDivision(outExtent_f32);

		auto computeForAxis = [&](const asset::IImage::E_TYPE axis, const auto& convWeightFunction, const int32_t windowSize, const float invStretchFactor)
		{
			LutDataType* outKernelWeightsPixel = reinterpret_cast<LutDataType*>(reinterpret_cast<uint8_t*>(outKernelWeights) + axisOffsets[axis]);

			// One phase corresponds to one window (not to say that every window has a unique phase, many will share the same phase) and one window gets
			// reduced to one output pixel, so this for loop will run exactly the number of times as there are output pixels with unique phases.
			for (uint32_t i = 0u; i < phaseCount[axis]; ++i)
			{
				core::vectorSIMDf outPixelCenter(0.f);
				outPixelCenter[axis] = float(i) + 0.5f; // output pixel center in output space
				outPixelCenter *= scale; // output pixel center in input space

				const auto [minSupport, maxSupport] = getConvolutionWeightFunctionSupports(axis, 1.f/invStretchFactor);

				// const int32_t windowCoord = kernel.getWindowMinCoord(outPixelCenter, outPixelCenter)[axis];
				int32_t windowCoord;
				{
					outPixelCenter[axis] -= 0.5f;
					windowCoord = static_cast<int32_t>(core::ceil(outPixelCenter[axis] + minSupport));
				}

				float relativePos = outPixelCenter[axis] - float(windowCoord); // relative position of the last pixel in window from current (ith) output pixel having a unique phase sequence of kernel evaluation points

				for (int32_t j = 0; j < windowSize; ++j)
				{
					const float domainScaledRelativePos = relativePos * invStretchFactor;

					for (uint32_t ch = 0; ch < MaxChannels; ++ch)
					{
						const double weight = convWeightFunction<0>(domainScaledRelativePos, ch);
						if constexpr (std::is_same_v<LutDataType, uint16_t>)
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = core::Float16Compressor::compress(float(weight));
						else
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = LutDataType(weight);
					}
					
					relativePos -= 1.f;
				}
			}
		};

		const auto rcp_c2 = core::vectorSIMDf(inExtent).preciseDivision(core::vectorSIMDf(outExtent));

		if (inImageType >= asset::IImage::ET_1D)
		{
			CConvolutionWeightFunction<ReconstructionFunctionX, ResamplingFunctionX> convX({}, {}, rcp_c2.x);
			computeForAxis(asset::IImage::ET_1D, convX, windowSize.x, 1.f/rcp_c2.x);
		}

		if (inImageType >= asset::IImage::ET_2D)
		{
			CConvolutionWeightFunction<ReconstructionFunctionY, ResamplingFunctionY> convY({}, {}, rcp_c2.y);
			computeForAxis(asset::IImage::ET_2D, convY, windowSize.y, 1.f/rcp_c2.y);
		}

		if (inImageType == asset::IImage::ET_3D)
		{
			CConvolutionWeightFunction<ReconstructionFunctionZ, ResamplingFunctionZ> convZ({}, {}, rcp_c2.z);
			computeForAxis(asset::IImage::ET_3D, convZ, windowSize.z, 1.f/rcp_c2.z);
		}

		return true;
	}

	static inline core::vectorSIMDi32 getWindowSize(
		const core::vectorSIMDu32&	inExtent,
		const core::vectorSIMDu32&	outExtent,
		const asset::IImage::E_TYPE	inImageType)
	{
		// Stretch and scale the resampling kernels.
		// we'll need to stretch the kernel support to be relative to the output image but in the input image coordinate system
		// (if support is 3 pixels, it needs to be 3 output texels, but measured in input texels)

		core::vectorSIMDi32 windowSize(0, 0, 0, 0);

		const auto rcp_c2 = core::vectorSIMDf(inExtent).preciseDivision(core::vectorSIMDf(outExtent));

		if (inImageType >= asset::IImage::E_TYPE::ET_1D)
		{
			const auto [minSupport, maxSupport] = getConvolutionWeightFunctionSupports(asset::IImage::E_TYPE::ET_1D, rcp_c2.x);
			windowSize.x = static_cast<int32_t>(core::ceil(maxSupport - minSupport));
		}

		if (inImageType >= asset::IImage::E_TYPE::ET_2D)
		{
			const auto [minSupport, maxSupport] = getConvolutionWeightFunctionSupports(asset::IImage::E_TYPE::ET_2D, rcp_c2.y);
			windowSize.y = static_cast<int32_t>(core::ceil(maxSupport - minSupport));
		}

		if (inImageType == asset::IImage::E_TYPE::ET_3D)
		{
			const auto [minSupport, maxSupport] = getConvolutionWeightFunctionSupports(asset::IImage::E_TYPE::ET_3D, rcp_c2.z);
			windowSize.z = static_cast<int32_t>(core::ceil(maxSupport - minSupport));
		}

		return windowSize;
	}

	// TODO(achal): This can be just a simple lambda in computeScaledKernelPhasedLUT.
	template <typename LutDataType>
	static inline core::vectorSIMDu32 getScaledKernelPhasedLUTAxisOffsets(const core::vectorSIMDu32& phaseCount, const core::vectorSIMDi32& windowSize)
	{
		core::vectorSIMDu32 result;
		result.x = 0u;
		result.y = (phaseCount[0] * windowSize.x);
		result.z = ((phaseCount[0] * windowSize.x) + (phaseCount[1] * windowSize.y));
		return result * sizeof(LutDataType) * MaxChannels;
	}

	inline std::pair<float, float> getConvolutionWeightFunctionSupports(const asset::IImage::E_TYPE axis, const float resamplingStretchFactor) const
	{
		switch (axis)
		{
		case asset::IImage::E_TYPE::ET_1D:
		{
			const auto minSupport = ReconstructionFunctionX::min_support + ResamplingFunctionX::min_support * resamplingStretchFactor;
			const auto maxSupport = ReconstructionFunctionX::max_support + ResamplingFunctionX::max_support * resamplingStretchFactor;
			return { minSupport, maxSupport };
		}

		case asset::IImage::E_TYPE::ET_2D:
		{
			const auto minSupport = ReconstructionFunctionY::min_support + ResamplingFunctionY::min_support * resamplingStretchFactor;
			const auto maxSupport = ReconstructionFunctionY::max_support + ResamplingFunctionY::max_support * resamplingStretchFactor;
			return { minSupport, maxSupport };
		}

		case asset::IImage::E_TYPE::ET_3D:
		{
			const auto minSupport = ReconstructionFunctionZ::min_support + ResamplingFunctionZ::min_support * resamplingStretchFactor;
			const auto maxSupport = ReconstructionFunctionZ::max_support + ResamplingFunctionZ::max_support * resamplingStretchFactor;
			return { minSupport, maxSupport };
		}

		default:
			assert(!"Invalid code path");
			return { 0.f, 0.f };
		}
	}
};

}

#endif
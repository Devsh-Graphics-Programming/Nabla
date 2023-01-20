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

	static inline core::vectorSIMDu32 getPhaseCount(const core::vectorSIMDu32& inExtent, const core::vectorSIMDu32& outExtent, const asset::IImage::E_TYPE inImageType)
	{
		core::vectorSIMDu32 result(0u);
		for (uint32_t i = 0u; i <= inImageType; ++i)
			result[i] = outExtent[i] / std::gcd(inExtent[i], outExtent[i]);
		return result;
	}
};

template <typename BlitUtilities>
concept Blittable = requires(BlitUtilities utils)
{
	// All Kernel value_type need to be identical.
	requires std::is_same_v<typename BlitUtilities::reconstruction_x_t::value_type, typename BlitUtilities::reconstruction_y_t::value_type> &&
		std::is_same_v<typename BlitUtilities::reconstruction_z_t::value_type, typename BlitUtilities::reconstruction_y_t::value_type>;

	requires std::is_same_v<typename BlitUtilities::resampling_x_t::value_type, typename BlitUtilities::resampling_y_t::value_type> &&
		std::is_same_v<typename BlitUtilities::resampling_z_t::value_type, typename BlitUtilities::resampling_y_t::value_type>;

	requires std::is_same_v<typename BlitUtilities::reconstruction_x_t::value_type, typename BlitUtilities::resampling_x_t::value_type>;

	// Alpha Handling requires high precision and multipass filtering!
	// We'll probably never remove this requirement.
	requires BlitUtilities::reconstruction_x_t::is_separable && BlitUtilities::reconstruction_y_t::is_separable && BlitUtilities::reconstruction_z_t::is_separable;
	requires BlitUtilities::resampling_x_t::is_separable && BlitUtilities::resampling_y_t::is_separable && BlitUtilities::resampling_z_t::is_separable;
};

template<
	typename ReconstructionKernelX	= CBoxImageFilterKernel,
	typename ResamplingKernelX		= ReconstructionKernelX,
	typename ReconstructionKernelY	= ReconstructionKernelX,
	typename ResamplingKernelY		= ResamplingKernelX,
	typename ReconstructionKernelZ	= ReconstructionKernelX,
	typename ResamplingKernelZ		= ResamplingKernelX>
class CBlitUtilities : public IBlitUtilities
{
	using convolution_kernels_t = std::tuple<CConvolutionImageFilterKernel<ReconstructionKernelX, ResamplingKernelX>, CConvolutionImageFilterKernel<ReconstructionKernelY, ResamplingKernelY>, CConvolutionImageFilterKernel<ReconstructionKernelZ, ResamplingKernelZ>>;

public:
	using reconstruction_x_t	= ReconstructionKernelX;
	using resampling_x_t		= ResamplingKernelX;
	using reconstruction_y_t	= ReconstructionKernelY;
	using resampling_y_t		= ResamplingKernelY;
	using reconstruction_z_t	= ReconstructionKernelZ;
	using resampling_z_t		= ResamplingKernelZ;

	using value_t = reconstruction_x_t::value_type;

	static inline constexpr auto MaxChannels = std::max<decltype(ReconstructionKernelX::MaxChannels)>(std::max<decltype(ReconstructionKernelX::MaxChannels)>(ReconstructionKernelX::MaxChannels, ReconstructionKernelY::MaxChannels), ReconstructionKernelZ::MaxChannels);

	static inline convolution_kernels_t getConvolutionKernels(
		const core::vectorSIMDu32&		inExtent,
		const core::vectorSIMDu32&		outExtent,
		const ReconstructionKernelX&	reconstructionX,
		const ResamplingKernelX&		resamplingX,
		const ReconstructionKernelY&	reconstructionY,
		const ResamplingKernelY&		resamplingY,
		const ReconstructionKernelZ&	reconstructionZ,
		const ResamplingKernelZ&		resamplingZ)
	{
		const auto stretchFactor = core::vectorSIMDf(inExtent).preciseDivision(core::vectorSIMDf(outExtent));
		
		// Stretch and scale the resampling kernels.
		// we'll need to stretch the kernel support to be relative to the output image but in the input image coordinate system
		// (if support is 3 pixels, it needs to be 3 output texels, but measured in input texels)
		auto resamplingX_stretched = ResamplingKernelX(resamplingX);
		resamplingX_stretched.stretchAndScale(stretchFactor);

		auto resamplingY_stretched = ResamplingKernelY(resamplingY);
		resamplingY_stretched.stretchAndScale(stretchFactor);

		auto resamplingZ_stretched = ResamplingKernelZ(resamplingZ);
		resamplingZ_stretched.stretchAndScale(stretchFactor);

		convolution_kernels_t kernels = std::make_tuple(
			asset::CConvolutionImageFilterKernel(ReconstructionKernelX(reconstructionX), std::move(resamplingX_stretched)),
			asset::CConvolutionImageFilterKernel(ReconstructionKernelY(reconstructionY), std::move(resamplingY_stretched)),
			asset::CConvolutionImageFilterKernel(ReconstructionKernelZ(reconstructionZ), std::move(resamplingZ_stretched)));

		return kernels;
	}

	template <typename LutDataType>
	static inline size_t getScaledKernelPhasedLUTSize(
		const core::vectorSIMDu32&		inExtent,
		const core::vectorSIMDu32&		outExtent,
		const asset::IImage::E_TYPE		inImageType,
		const ReconstructionKernelX&	reconstructionX,
		const ResamplingKernelX&		resamplingX,
		const ReconstructionKernelY&	reconstructionY,
		const ResamplingKernelY&		resamplingY,
		const ReconstructionKernelZ&	reconstructionZ,
		const ResamplingKernelZ&		resamplingZ)
	{
		auto [kernelX, kernelY, kernelZ] = getConvolutionKernels(inExtent, outExtent,
			reconstructionX, resamplingX,
			reconstructionY, resamplingY,
			reconstructionZ, resamplingZ);

		return getScaledKernelPhasedLUTSize<LutDataType>(inExtent, outExtent, inImageType, kernelX, kernelY, kernelZ);
	}

	template <typename LutDataType>
	static inline size_t getScaledKernelPhasedLUTSize(
		const core::vectorSIMDu32&								inExtent,
		const core::vectorSIMDu32&								outExtent,
		const asset::IImage::E_TYPE								inImageType,
		const std::tuple_element_t<0, convolution_kernels_t>&	kernelX,
		const std::tuple_element_t<1, convolution_kernels_t>&	kernelY,
		const std::tuple_element_t<2, convolution_kernels_t>&	kernelZ)
	{
		const auto phaseCount = getPhaseCount(inExtent, outExtent, inImageType);
		return ((phaseCount[0] * kernelX.getWindowSize().x) + (phaseCount[1] * kernelY.getWindowSize().y) + (phaseCount[2] * kernelZ.getWindowSize().z)) * sizeof(LutDataType) * MaxChannels;
	}

	template <typename LutDataType>
	static bool computeScaledKernelPhasedLUT(
		void*							outKernelWeights,
		const core::vectorSIMDu32&		inExtent,
		const core::vectorSIMDu32&		outExtent,
		const asset::IImage::E_TYPE		inImageType,
		const ReconstructionKernelX&	reconstructionX,
		const ResamplingKernelX&		resamplingX,
		const ReconstructionKernelY&	reconstructionY,
		const ResamplingKernelY&		resamplingY,
		const ReconstructionKernelZ&	reconstructionZ,
		const ResamplingKernelZ&		resamplingZ)
	{
		const core::vectorSIMDu32 phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

		for (auto i = 0; i <= inImageType; ++i)
		{
			if (phaseCount[i] == 0)
				return false;
		}

		auto [kernelX, kernelY, kernelZ] = getConvolutionKernels(inExtent, outExtent,
			reconstructionX, resamplingX,
			reconstructionY, resamplingY,
			reconstructionZ, resamplingZ);

		const auto windowSize = getRealWindowSize(inImageType, kernelX, kernelY, kernelZ);
		const auto axisOffsets = getScaledKernelPhasedLUTAxisOffsets<LutDataType>(phaseCount, windowSize);

		const core::vectorSIMDf inExtent_f32(inExtent);
		const core::vectorSIMDf outExtent_f32(outExtent);
		const auto scale = inExtent_f32.preciseDivision(outExtent_f32);

		// a dummy load functor
		// does nothing but fills up the `windowSample` with 1s (identity) so we can preserve the value of kernel
		// weights when eventually `windowSample` gets multiplied by them later in
		// `CFloatingPointSeparableImageFilterKernelBase<CRTP>::sample_functor_t<PreFilter,PostFilter>::operator()`
		// this exists only because `evaluateImpl` expects a pre filtering step.
		auto dummyLoad = [](double* windowSample, const core::vectorSIMDf&, const core::vectorSIMDi32&, const core::vectorSIMDf&) -> void
		{
			for (auto h = 0; h < MaxChannels; h++)
				windowSample[h] = 1.0;
		};

		double kernelWeight[MaxChannels];
		// actually used to put values in the LUT
		auto dummyEvaluate = [&kernelWeight](const double* windowSample, const core::vectorSIMDf&, const core::vectorSIMDi32&, const core::vectorSIMDf&) -> void
		{
			for (auto h = 0; h < MaxChannels; h++)
				kernelWeight[h] = windowSample[h];
		};

		auto computeForAxis = [&](const asset::IImage::E_TYPE axis, const auto& kernel)
		{
			if (axis > inImageType)
				return;

			// TODO(achal): Why do we compute this again?! Don't we already have it.
			const auto windowSize = kernel.getWindowSize()[axis];

			LutDataType* outKernelWeightsPixel = reinterpret_cast<LutDataType*>(reinterpret_cast<uint8_t*>(outKernelWeights) + axisOffsets[axis]);

			// One phase corresponds to one window (not to say that every window has a unique phase, many will share the same phase) and one window gets
			// reduced to one output pixel, so this for loop will run exactly the number of times as there are output pixels with unique phases.
			for (uint32_t i = 0u; i < phaseCount[axis]; ++i)
			{
				core::vectorSIMDf outPixelCenter(0.f);
				outPixelCenter[axis] = float(i) + 0.5f; // output pixel center in output space
				outPixelCenter *= scale; // output pixel center in input space

				const int32_t windowCoord = kernel.getWindowMinCoord(outPixelCenter, outPixelCenter)[axis];

				float relativePos = outPixelCenter[axis] - float(windowCoord); // relative position of the last pixel in window from current (ith) output pixel having a unique phase sequence of kernel evaluation points

				for (int32_t j = 0; j < windowSize; ++j)
				{
					core::vectorSIMDf tmp(relativePos, 0.f, 0.f);
					kernel.evaluateImpl(dummyLoad, dummyEvaluate, kernelWeight, tmp, core::vectorSIMDi32());
					for (uint32_t ch = 0; ch < MaxChannels; ++ch)
					{
						if constexpr (std::is_same_v<LutDataType, uint16_t>)
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = core::Float16Compressor::compress(float(kernelWeight[ch]));
						else
							outKernelWeightsPixel[(i * windowSize + j) * MaxChannels + ch] = LutDataType(kernelWeight[ch]);

					}
					relativePos -= 1.f;
				}
			}
		};

		computeForAxis(asset::IImage::ET_1D, kernelX);
		computeForAxis(asset::IImage::ET_2D, kernelY);
		computeForAxis(asset::IImage::ET_3D, kernelZ);

		return true;
	}

	static inline core::vectorSIMDi32 getRealWindowSize(const IImage::E_TYPE inImageType,
		const std::tuple_element_t<0, convolution_kernels_t>& kernelX,
		const std::tuple_element_t<1, convolution_kernels_t>& kernelY,
		const std::tuple_element_t<2, convolution_kernels_t>& kernelZ)
	{
		core::vectorSIMDi32 last(kernelX.getWindowSize().x, 0, 0, 0);
		if (inImageType >= IImage::ET_2D)
			last.y = kernelY.getWindowSize().y;
		if (inImageType >= IImage::ET_3D)
			last.z = kernelZ.getWindowSize().z;
		return last;
	}

	template <typename LutDataType>
	static inline core::vectorSIMDu32 getScaledKernelPhasedLUTAxisOffsets(const core::vectorSIMDu32& phaseCount, const core::vectorSIMDi32& windowSize)
	{
		core::vectorSIMDu32 result;
		result.x = 0u;
		result.y = (phaseCount[0] * windowSize.x);
		result.z = ((phaseCount[0] * windowSize.x) + (phaseCount[1] * windowSize.y));
		return result * sizeof(LutDataType) * MaxChannels;
	}
};

}

#endif
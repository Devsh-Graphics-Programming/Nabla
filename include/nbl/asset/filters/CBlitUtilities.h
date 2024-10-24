// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_BLIT_UTILITIES_H_INCLUDED_
#define _NBL_ASSET_C_BLIT_UTILITIES_H_INCLUDED_


#include "nbl/asset/filters/kernels/WeightFunctions.h"
#include "nbl/asset/filters/kernels/CConvolutionWeightFunction.h"
#include "nbl/asset/filters/kernels/CChannelIndependentWeightFunction.h"


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

		static inline hlsl::uint32_t3 getPhaseCount(const hlsl::uint32_t3& inExtent, const hlsl::uint32_t3& outExtent, const IImage::E_TYPE inImageType)
		{
			hlsl::uint32_t3 result(0u);
			for (uint32_t i = 0u; i <= inImageType; ++i)
				result[i] = outExtent[i] / std::gcd(inExtent[i], outExtent[i]);
			return result;
		}
};

namespace impl
{

// Make all instantiations of `is_instantiation_of_CConvolutionWeightFunction1D/ChannelIndependentWeightFunctionOfConvolutions` have
// a false value, but specialize only the ones we want, to be true.

template <typename T>
struct is_instantiation_of_CConvolutionWeightFunction1D : std::false_type {};

template <SimpleWeightFunction1D WeightFunction1DA, SimpleWeightFunction1D WeightFunction1DB>
struct is_instantiation_of_CConvolutionWeightFunction1D<CConvolutionWeightFunction1D<WeightFunction1DA, WeightFunction1DB>> : std::true_type {};

template <typename T>
struct ChannelIndependentWeightFunctionOfConvolutions : std::false_type {};

template <WeightFunction1D FirstWeightFunction1D, WeightFunction1D... OtherWeightFunctions>
struct ChannelIndependentWeightFunctionOfConvolutions<CChannelIndependentWeightFunction1D<FirstWeightFunction1D, OtherWeightFunctions...>>
	: std::bool_constant<(is_instantiation_of_CConvolutionWeightFunction1D<FirstWeightFunction1D>::value) && (is_instantiation_of_CConvolutionWeightFunction1D<OtherWeightFunctions>::value && ...)>
{};

} // namespace impl

template <typename T>
concept ChannelIndependentWeightFunctionOfConvolutions = requires(T t, const float unnormCenterSampledCoord, float& cornerSampledCoord, const float x, const uint8_t channel)
{
	requires impl::ChannelIndependentWeightFunctionOfConvolutions<T>::value;

	{ t.getWindowSize() } -> std::same_as<int32_t>;

	{ t.getWindowMinCoord(unnormCenterSampledCoord, cornerSampledCoord) } -> std::same_as<int32_t>;

	{ t.weight(x, channel) } -> std::same_as<typename T::value_t>;
};

template<
	ChannelIndependentWeightFunctionOfConvolutions KernelX = CDefaultChannelIndependentWeightFunction1D<
		CConvolutionWeightFunction1D<
			CWeightFunction1D<SBoxFunction>,
			CWeightFunction1D<SBoxFunction>
		>
	>,
	ChannelIndependentWeightFunctionOfConvolutions KernelY = KernelX,
	ChannelIndependentWeightFunctionOfConvolutions KernelZ = KernelX,
	typename LutDataType = float
>
class CBlitUtilities : public IBlitUtilities
{
	public:
		using value_type = KernelX::value_t;
		static_assert(std::is_same_v<value_type,KernelY::value_t> && std::is_same_v<value_type,KernelZ::value_t>);

		static inline constexpr uint32_t ChannelCount = KernelX::ChannelCount;
		static_assert(ChannelCount>0 && ChannelCount<=4 && KernelY::ChannelCount==ChannelCount && KernelZ::ChannelCount==ChannelCount);

		struct ConvolutionKernels
		{
			inline ConvolutionKernels(const KernelX& _x={}, const KernelY& _y={}, const KernelZ& _z={}) : x(_x), y(_y), z(_z) {}
			inline ConvolutionKernels(const ConvolutionKernels& other)
			{
				operator=(other);
			}
			inline ConvolutionKernels(ConvolutionKernels&& other)
			{
				operator=(std::move(other));
			}
			inline ConvolutionKernels& operator=(const ConvolutionKernels& other)
			{
				x = other.x;
				y = other.y;
				z = other.z;
				return *this;
			}
			inline ConvolutionKernels& operator=(ConvolutionKernels&& other)
			{
				x = std::move(other.x);
				y = std::move(other.y);
				z = std::move(other.z);
				return *this;
			}

			KernelX x;
			KernelY y;
			KernelZ z;
		};

		using lut_value_type = LutDataType;
		static_assert(std::is_same_v<lut_value_type,hlsl::float16_t> || std::is_same_v<lut_value_type,hlsl::float32_t>, "Invalid LUT data type.");


		static inline size_t getScaledKernelPhasedLUTSize(
			const hlsl::uint32_t3&		inExtent,
			const hlsl::uint32_t3&		outExtent,
			const IImage::E_TYPE		inImageType,
			const ConvolutionKernels&	kernels
		)
		{
			const auto windowSize = getWindowSize(inImageType, kernels);
			return getScaledKernelPhasedLUTSize(inExtent, outExtent, inImageType, windowSize);
		}

		static inline size_t getScaledKernelPhasedLUTSize(
			const hlsl::uint32_t3&		inExtent,
			const hlsl::uint32_t3&		outExtent,
			const IImage::E_TYPE		inImageType,
			const hlsl::int32_t3&		windowSize
		)
		{
			const auto phaseCount = getPhaseCount(inExtent, outExtent, inImageType);
			return ((phaseCount.x * windowSize.x) + (phaseCount.y * windowSize.y) + (phaseCount.z * windowSize.z)) * sizeof(double) * ChannelCount;
		}

		static bool computeScaledKernelPhasedLUT(
			void*						outKernelWeights,
			const hlsl::uint32_t3&		inExtent,
			const hlsl::uint32_t3&		outExtent,
			const IImage::E_TYPE		inImageType,
			const ConvolutionKernels&	kernels,
			const double				normalizeWeightsTo=1.0
		)
		{
			const bool shouldNormalize = !core::isnan(normalizeWeightsTo);

			const hlsl::uint32_t3 phaseCount = getPhaseCount(inExtent, outExtent, inImageType);

			for (auto i = 0; i <= inImageType; ++i)
			{
				if (phaseCount[i] == 0)
					return false;
			}

			const auto windowSize = getWindowSize(inImageType, kernels);
			const auto axisOffsets = getScaledKernelPhasedLUTAxisOffsets(phaseCount, windowSize);
			const auto axisOffsets_f64 = getScaledKernelPhasedLUTAxisOffsets(phaseCount, windowSize);

			const hlsl::float64_t3 inExtent_f32(inExtent);
			const hlsl::float64_t3 outExtent_f32(outExtent);
			const auto scale = inExtent_f32/outExtent_f32;

			auto computeForAxis = [&](const IImage::E_TYPE axis, const auto& kernel, const int32_t _windowSize)
			{
				if (axis > inImageType)
					return;

				LutDataType* outKernelWeightsPixel = reinterpret_cast<LutDataType*>(reinterpret_cast<uint8_t*>(outKernelWeights) + axisOffsets[axis]);
				double* outKernelWeightsPixel_f64 = reinterpret_cast<double*>(reinterpret_cast<uint8_t*>(outKernelWeights) + axisOffsets_f64[axis]);

				// One phase corresponds to one window (not to say that every window has a unique phase, many will share the same phase) and one window gets
				// reduced to one output pixel, so this for loop will run exactly the number of times as there are output pixels with unique phases.
				for (uint32_t i = 0u; i < phaseCount[axis]; ++i)
				{
					hlsl::float32_t outPixelCenter = (hlsl::float64_t(i) + 0.5)*scale[axis]; // output pixel center in input space

					const int32_t windowCoord = kernel.getWindowMinCoord(outPixelCenter,outPixelCenter);

					float relativePos = outPixelCenter - float(windowCoord); // relative position of the last pixel in window from current (ith) output pixel having a unique phase sequence of kernel evaluation points

					double accum[ChannelCount] = { };
					for (int32_t j = 0; j < _windowSize; ++j)
					{
						for (uint32_t ch = 0; ch < ChannelCount; ++ch)
						{
							const size_t okwpIndex = (i * _windowSize + j) * ChannelCount + ch;

							const double weight = static_cast<double>(kernel.weight(relativePos, ch));
							if (!shouldNormalize)
								outKernelWeightsPixel[okwpIndex] = static_cast<LutDataType>(weight);
							else
							{
								accum[ch] += weight;
								outKernelWeightsPixel_f64[okwpIndex] = weight;
							}
						}
					
						relativePos -= 1.f;
					}

					if (shouldNormalize)
					{
						constexpr double Threshold = 1e-6;

						double normalizationFactor[ChannelCount] = { };
						for (uint32_t ch = 0; ch < ChannelCount; ++ch)
						{
							if (core::abs(accum[ch]) >= Threshold)
								normalizationFactor[ch] = normalizeWeightsTo / accum[ch];
							else
								normalizationFactor[ch] = normalizeWeightsTo / double(_windowSize);
						}

						for (int32_t j = 0; j < _windowSize; ++j)
						{
							for (uint32_t ch = 0; ch < ChannelCount; ++ch)
							{
								const uint64_t idx = (i * _windowSize + j) * ChannelCount + ch;

								double normalized;
								if (core::abs(accum[ch]) >= Threshold)
									normalized = outKernelWeightsPixel_f64[idx] * normalizationFactor[ch];
								else
									normalized = normalizationFactor[ch];

								outKernelWeightsPixel[idx] = static_cast<LutDataType>(normalized);
							}
						}
					}
				}
			};

			computeForAxis(IImage::ET_1D, kernels.x, windowSize.x);
			computeForAxis(IImage::ET_2D, kernels.y, windowSize.y);
			computeForAxis(IImage::ET_3D, kernels.z, windowSize.z);

			return true;
		}

		// the initial kernels can have their scales/states individually ajusted etc. before calling this for a particular blit run
		static inline void rescaleKernels(ConvolutionKernels& kernels, const hlsl::uint32_t3& inExtent, const hlsl::uint32_t3& outExtent)
		{
			// could be the `ChannelIndependentWeightFunctionOfConvolutions`
			auto rescaleChannel = []<typename Recon, typename Resam>(const CConvolutionWeightFunction1D<Recon,Resam>& kernel, const double rcp_c2)->CConvolutionWeightFunction1D<Recon,Resam>
			{
				Resam newResam = kernel.getFuncB();
				newResam.stretchAndScale(rcp_c2);
				return CConvolutionWeightFunction1D<Recon,Resam>(Recon(kernel.getFuncA()),std::move(newResam));
			};
			// could be the `ChannelIndependentWeightFunctionOfConvolutions`
			auto rescaleAxis = [&]<typename Kernel>(Kernel& kernel, const double rcp_c2)->Kernel
			{
				if constexpr (ChannelCount==1)
					return Kernel(rescaleChannel(kernel.getFunction<0>(),rcp_c2));
				if constexpr (ChannelCount==2)
					return Kernel(
						rescaleChannel(kernel.getFunction<0>(),rcp_c2),
						rescaleChannel(kernel.getFunction<1>(),rcp_c2)
					);
				if constexpr (ChannelCount==3)
					return Kernel(
						rescaleChannel(kernel.getFunction<0>(),rcp_c2),
						rescaleChannel(kernel.getFunction<1>(),rcp_c2),
						rescaleChannel(kernel.getFunction<2>(),rcp_c2)
					);
				if constexpr (ChannelCount==4)
					return Kernel(
						rescaleChannel(kernel.getFunction<0>(),rcp_c2),
						rescaleChannel(kernel.getFunction<1>(),rcp_c2),
						rescaleChannel(kernel.getFunction<2>(),rcp_c2),
						rescaleChannel(kernel.getFunction<3>(),rcp_c2)
					);
			};

			// Stretch and scale the resampling kernels.
			// we'll need to stretch the kernel support to be relative to the output image but in the input image coordinate system
			// (if support is 3 pixels, it needs to be 3 output texels, but measured in input texels)
			const auto rcp_c2 = hlsl::float64_t3(inExtent)/hlsl::float64_t3(outExtent);
			kernels = ConvolutionKernels{
				/*.x = */rescaleAxis(kernels.x,rcp_c2.x),
				/*.y = */rescaleAxis(kernels.y,rcp_c2.y),
				/*.z = */rescaleAxis(kernels.z,rcp_c2.z)
			};
		}

		static inline hlsl::int32_t3 getWindowSize(const IImage::E_TYPE inImageType, const ConvolutionKernels& kernels)
		{
			hlsl::int32_t3 windowSize(kernels.x.getWindowSize(), 0, 0);
			if (inImageType >= IImage::ET_2D)
				windowSize.y = kernels.y.getWindowSize();
			if (inImageType == IImage::ET_3D)
				windowSize.z = kernels.z.getWindowSize();

			return windowSize;
		}

		static inline hlsl::uint32_t3 getScaledKernelPhasedLUTAxisOffsets(const hlsl::uint32_t3& phaseCount, const hlsl::int32_t3& windowSize)
		{
			hlsl::uint32_t3 result;
			result.x = 0u;
			result.y = (phaseCount[0] * windowSize.x);
			result.z = ((phaseCount[0] * windowSize.x) + (phaseCount[1] * windowSize.y));
			return result * uint32_t(sizeof(LutDataType)) * ChannelCount;
		}
};

}

#endif
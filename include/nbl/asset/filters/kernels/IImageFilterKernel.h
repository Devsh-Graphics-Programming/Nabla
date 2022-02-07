// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_I_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUImage.h"

namespace nbl
{
namespace asset
{
// TODO: do every member var in 4 copies, support, window sizes, etc. ?

// runtime polymorphic interface for a kernel
class IImageFilterKernel
{
public:
    // All kernels are by default, defined on max 4 channels
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;

    // some user data structs commonly used
    struct UserData
    {
        enum E_USER_DATA_TYPE
        {
            EUDT_SCALE_FACTOR,
            EUDT_COUNT_OR_UNKNOWN
        };
        E_USER_DATA_TYPE type = EUDT_COUNT_OR_UNKNOWN;
        // idea: a pNext chain?
    };
    struct ScaleFactorUserData : UserData
    {
        _NBL_STATIC_INLINE_CONSTEXPR E_USER_DATA_TYPE Type = EUDT_SCALE_FACTOR;
        ScaleFactorUserData(float _factor = 1.f)
            : UserData{Type}, factor{_factor, _factor, _factor, _factor} {}

        inline bool valid() const { return type == Type; }

        static inline ScaleFactorUserData* cast(UserData* data)
        {
            if(!data)
                return nullptr;

            auto retval = static_cast<ScaleFactorUserData*>(data);
            return retval->valid() ? retval : nullptr;
        }
        static inline const ScaleFactorUserData* cast(const UserData* data)
        {
            if(!data)
                return nullptr;

            auto retval = static_cast<const ScaleFactorUserData*>(data);
            return retval->valid() ? retval : nullptr;
        }

        float factor[MaxChannels];
    };

    // `evaluate` is actually implemented in a funny way, because a Kernel won't always be used as a convolution kernel (i.e. for implementing a Median filter)
    // so the evaluation of a Kernel on an window grid relies of a PreFilter and PostFilter functor which are applied before the window of samples is treated with a kernel and after respectively
    // such functors will handle loading/writing the samples from/to a texture (or better, temporary memory)
    // this is the default functor which does nothing
    struct default_sample_functor_t
    {
        // `windowSample` holds the storage of the channel values of the current pixel processed by `CImageFilterKernel::evaluateImpl`
        // `relativePosAndFactor.xyz` holds the coordinate of the pixel relative to the window's center (can be fractional so its a float),
        // `globalTexelCoord` is the unnormalized corner sampled coordinate of the pixel relative to the image origin,
        // `userData` holds a pointer to a case-dependent type (usually its a scale factor for our kernel)
        inline void operator()(void* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const UserData* userData)
        {
        }
    };
    using sample_functor_operator_t = void(void*, core::vectorSIMDf&, const core::vectorSIMDi32&, const UserData* userData);

    // Whether we can break up the convolution in multiple dimensions as a separate convlution per dimension all followed after each other,this is very important for performance
    // as it turns a convolution from a O(window_size.x*image_extent.x*window_size.y*image_extent.y...) to O(window_size.x*image_extent.x+window_size.y*image_extent.y+..)
    virtual bool pIsSeparable() const = 0;
    virtual bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const = 0;

    // Returns a pointer to implementation defined metadata struct/class which is constant for the Kernel but not static (e.g. Kernel's scale)
    virtual const UserData* pGetUserData() const = 0;

    // function to evaluate the kernel at a pixel position
    // `globalPos` is the unnormalized (measured in pixels) center sampled (origin is at the center of the first pixel) position
    // The`preFilter` and `postFilter` are supposed to be executed for-each-pixel-in-the-window immediately before and after, the business logic of the kernel.
    virtual void pEvaluate(const core::vectorSIMDf& globalPos, std::function<sample_functor_operator_t>& preFilter, std::function<sample_functor_operator_t>& postFilter) const = 0;

    // convenience function in-case we don't want to run any pre or post filters
    inline void pEvaluate(const core::vectorSIMDf& globalPos) const
    {
        std::function void_functor = default_sample_functor_t();
        pEvaluate(globalPos, void_functor, void_functor);
    }

    // given an unnormalized (measured in pixels), center sampled coordinate (origin is at the center of the first pixel),
    // return corner sampled coordinate (origin at the very edge of the first pixel) as well as the
    // corner sampled coordinate of the first pixel that lays inside the kernel's support when centered on the given pixel
    inline core::vectorSIMDi32 getWindowMinCoord(const core::vectorSIMDf& unnormCeterSampledCoord, core::vectorSIMDf& cornerSampledCoord) const
    {
        cornerSampledCoord = unnormCeterSampledCoord - core::vectorSIMDf(0.5f, 0.5f, 0.5f, 0.f);
        return core::vectorSIMDi32(core::ceil<core::vectorSIMDf>(cornerSampledCoord - negative_support));
    }
    // overload that does not return the cornern sampled coordinate of the given center sampled coordinate
    inline core::vectorSIMDi32 getWindowMinCoord(const core::vectorSIMDf& unnormCeterSampledCoord) const
    {
        core::vectorSIMDf dummy;
        return getWindowMinCoord(unnormCeterSampledCoord, dummy);
    }

    // get the kernel support (measured in pixels)
    inline const auto& getWindowSize() const
    {
        return window_size;
    }

    const core::vectorSIMDf negative_support;
    const core::vectorSIMDf positive_support;
    const core::vectorSIMDi32 window_size;
    const core::vectorSIMDi32 window_strides;

protected:
    // derived classes need to let us know where the function starts and stops having non-zero values, this is measured in pixels
    IImageFilterKernel(const float* _negative_support, const float* _positive_support)
        : negative_support(_negative_support[0], _negative_support[1], _negative_support[2]),
          positive_support(_positive_support[0], _positive_support[1], _positive_support[2]),
          window_size(core::ceil<core::vectorSIMDf>(negative_support + positive_support)),
          window_strides(1, window_size[0], window_size[0] * window_size[1])
    {}
    IImageFilterKernel(const std::initializer_list<float>& _negative_support, const std::initializer_list<float>& _positive_support)
        : IImageFilterKernel(_negative_support.begin(), _positive_support.begin())
    {}
};

// statically polymorphic version of the interface for a kernel
template<class CRTP, typename value_type>
class CImageFilterKernel : public IImageFilterKernel
{
public:
    using IImageFilterKernel::IImageFilterKernel;

    struct default_sample_functor_t : IImageFilterKernel::default_sample_functor_t
    {
        // every functor is expected to have this function call operator with this particular signature (the const-ness of the arguments can vary).
        // `windowSample` is adjusted to have the correct pointer type compared the the parent class' struct
        inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
        {
        }
    };

    // These are same as the functions declated here without the `p` prefix but allow us to use polymorphism at a higher level
    inline bool pIsSeparable() const override
    {
        return CRTP::is_separable;
    }
    inline bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const override
    {
        return CRTP::validate(inImage, outImage);
    }
    const UserData* pGetUserData() const override
    {
        return static_cast<const CRTP*>(this)->getUserData();
    }
    void pEvaluate(const core::vectorSIMDf& globalPos, std::function<sample_functor_operator_t>& preFilter, std::function<sample_functor_operator_t>& postFilter) const override
    {
        static_cast<const CRTP*>(this)->evaluate(globalPos, preFilter, postFilter);
    }

    // `globalPos` is an unnormalized and center sampled pixel coordinate
    // each derived class needs to declare a method `template<class PreFilter, class PostFilter> auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const`
    // the created functor must execute the given `preFilter` before, and the given `postFilter` after, the business logic that the class wants to perform on the pixel window.
    template<class PreFilter = const default_sample_functor_t, class PostFilter = const default_sample_functor_t>
    inline void evaluate(const core::vectorSIMDf& globalPos, PreFilter& preFilter, PostFilter& postFilter) const
    {
        // offsetGlobalPos now is a unnormalized but corner sampled coord
        core::vectorSIMDf offsetGlobalPos;
        // get the first and one-past-the-last integer coordinates of the window
        const auto startCoord = getWindowMinCoord(globalPos, offsetGlobalPos);
        const auto endCoord = startCoord + window_size;

        // temporary storage for a single pixel
        value_type windowSample[CRTP::MaxChannels];

        core::vectorSIMDi32 windowCoord(0, 0, 0);
        // loop over the window's global coordinates
        for(auto& z = (windowCoord.z = startCoord.z); z != endCoord.z; z++)
            for(auto& y = (windowCoord.y = startCoord.y); y != endCoord.y; y++)
                for(auto& x = (windowCoord.x = startCoord.x); x != endCoord.x; x++)
                {
                    // get position relative to kernel origin, note that it is in reverse (tau-x), in accordance with Mathematical Convolution notation
                    auto relativePos = offsetGlobalPos - core::vectorSIMDf(windowCoord);
                    evaluateImpl<PreFilter, PostFilter>(preFilter, postFilter, windowSample, relativePos, windowCoord, static_cast<const CRTP*>(this)->getUserData());
                }
    }

    // This function is called once for each pixel in the kernel window, for explanation of `preFilter` and `postFilter` @see evaluate.
    // The `windowSample` holds the temporary storage (channels) for the current pixel, but at the time its passed to this function the contents are garbage.
    // Its the `preFilter` and `postFilter` that deals with actually loading and saving the pixel's value.
    // The `relativePosAndFactor.xyz` holds the NEGATIVE coordinate of the pixel relative to the window's center (can be fractional so its a float)
    // `userData` is a mechanism to pass non-static variables from the derived class to this kernel inner loop
    template<class PreFilter, class PostFilter>
    void evaluateImpl(PreFilter& preFilter, PostFilter& postFilter, value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const UserData* userData) const;
};

//use this whenever you have diamond inheritance and ambiguous resolves
#define NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(BASENAME)                                                                                                                                                                            \
    template<typename... Args>                                                                                                                                                                                                                    \
    inline core::vectorSIMDi32 getWindowMinCoord(Args&&... args) const                                                                                                                                                                            \
    {                                                                                                                                                                                                                                             \
        return BASENAME::getWindowMinCoord(std::forward<Args>(args)...);                                                                                                                                                                          \
    }                                                                                                                                                                                                                                             \
    inline const auto& getWindowSize() const                                                                                                                                                                                                      \
    {                                                                                                                                                                                                                                             \
        return BASENAME::getWindowSize();                                                                                                                                                                                                         \
    }                                                                                                                                                                                                                                             \
    template<class PreFilter = const typename BASENAME::default_sample_functor_t, class PostFilter = const typename BASENAME::default_sample_functor_t>                                                                                           \
    inline void evaluate(const core::vectorSIMDf& globalPos, PreFilter& preFilter, PostFilter& postFilter) const                                                                                                                                  \
    {                                                                                                                                                                                                                                             \
        BASENAME::evaluate(globalPos, preFilter, postFilter);                                                                                                                                                                                     \
    }                                                                                                                                                                                                                                             \
    template<class PreFilter, class PostFilter>                                                                                                                                                                                                   \
    inline void evaluateImpl(PreFilter& preFilter, PostFilter& postFilter, value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const nbl::asset::IImageFilterKernel::UserData* userData) const \
    {                                                                                                                                                                                                                                             \
        BASENAME::evaluateImpl(preFilter, postFilter, windowSample, relativePos, globalTexelCoord, userData);                                                                                                                                     \
    }

}  // end namespace asset
}  // end namespace nbl

#endif
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FFT_INCLUDED_
#define _NBL_EXT_FFT_INCLUDED_

#include "nabla.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUShader.h"

namespace nbl
{
namespace ext
{
namespace FFT
{

class FFT : public core::TotalInterface
{
    public:
        struct DispatchInfo_t
        {
            uint32_t workGroupDims[3];
            uint32_t workGroupCount[3];
        };

        // returns dispatch size (and wg size in x)
        static inline DispatchInfo_t buildParameters(asset::VkExtent3D const & imageSize, uint32_t workGroupXdim = DEFAULT_WORK_GROUP_X_DIM)
        {
            return {};
        }

        //
        static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

        //
        static core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver);

        //
        static inline size_t getOutputBufferSize(asset::VkExtent3D const & imageSize, asset::E_FORMAT format)
        {
            return (imageSize.width * imageSize.height * imageSize.depth) * asset::getTexelOrBlockBytesize(format);
        }

        static core::smart_refctd_ptr<video::IGPUShader> createShader(asset::E_FORMAT format);

        // we expect user binds correct pipeline, descriptor sets and pushes the push constants by themselves
        static inline void dispatchHelper(video::IVideoDriver* driver, const DispatchInfo_t& dispatchInfo, bool issueDefaultBarrier=true)
        {
            driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);

            if (issueDefaultBarrier)
                defaultBarrier();
        }

    private:
        FFT() = delete;
        //~FFT() = delete;

        _NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_X_DIM = 256u;

        static void defaultBarrier();
};


}
}
}

#endif

// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_ANIMATION_BLEND_MANAGER_H_INCLUDED__
#define __NBL_SCENE_I_ANIMATION_BLEND_MANAGER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/video/video.h"

#include "nbl/scene/ITransformTreeManager.h"

namespace nbl
{
namespace scene
{
class IAnimationBlendManager : public virtual core::IReferenceCounted
{
public:
    using blend_t = uint32_t;

    // creation (TODO: should we support adding and removing animation libraries at runtime?)
    static inline core::smart_refctd_ptr<IAnimationBlendManager> create(video::IVideoDriver* _driver, core::smart_refctd_ptr<video::IGPUAnimationLibrary>&& _animationLibrary)
    {
        if(true)  // TODO: some checks and validation before creating?
            return nullptr;

        auto* abm = new IAnimationBlendManager(_driver, std::move(_animationLibrary));
        return core::smart_refctd_ptr<IAnimationBlendManager>(abm, core::dont_grab);
    }

    // TODO: we should probably group blends of different update frequencies separately
    void registerBlends(blend_t* blendsOut)
    {
        // TODO: register animation blends as <video::IGPUAnimationLibrary::animation_t,<updateFrequency,finishAndPause/finish/loop/<pingpongCurrFWD,pingpongCurrBWD>>,target node_t> (use a pool allocator)
    }

    // add to a contiguous list in GPU memory
    void startBlends(const blend_t* begin, const blend_t* end)
    {
        // easy enough, just push the `blend_t`s to a GPU `sparse_vector`
    }
    // remove from a contiguous list in GPU memory
    void pauseBlends(const blend_t* begin, const blend_t* end)
    {
        // easy enough, just erase the `blend_t`s from a GPU `sparse_vector`
    }

    //
    void seekBlends(const blend_t* begin, const blend_t* end, const ITransformTreeManager::timestamp_t* timestamps)
    {
    }

    // need to pause the blends first
    void deregisterBlends(const blend_t* begin, const blend_t* end)
    {
    }

    void computeBlends(ITransformTreeManager::timestamp_t newTimestamp,
        const asset::SBufferBinding<video::IGPUBuffer>& relativeTformUpdateIndirectParameters,
        const asset::SBufferBinding<video::IGPUBuffer>& nodeIDBuffer,  // first uint in the nodeIDBuffer is used to denote how many requests we have
        const asset::SBufferBinding<video::IGPUBuffer>& modificationRequestBuffer,
        const asset::SBufferBinding<video::IGPUBuffer>& modificationRequestTimestampBuffer)
    {
        m_driver->bindComputePipeline(m_computeBlendsPipeline.get());
        // TODO: bind descriptor sets
        m_driver->dispatchIndirect(m_dispatchIndirectCommandBuffer.get(), 0u);
        // TODO: pipeline barrier for SSBO and TBO and COMMAND_BIT too
    }

protected:
    IAnimationBlendManager(video::IVideoDriver* _driver, core::smart_refctd_ptr<video::IGPUAnimationLibrary>&& _animationLibrary)
        : m_driver(_driver), m_animationLibrary(std::move(_animationLibrary))
    {
    }
    ~IAnimationBlendManager()
    {
        // everything drops itself automatically
    }

    video::IVideoDriver* m_driver;
    core::smart_refctd_ptr<video::IGPUAnimationLibrary> m_animationLibrary;
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_computeBlendsPipeline;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> m_animationDS;  // animation library (keyframes + animations) + registered blends + active blends
    core::smart_refctd_ptr<video::IGPUBuffer> m_dispatchIndirectCommandBuffer;
};

}  // end namespace scene
}  // end namespace nbl

#endif

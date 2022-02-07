// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__
#define __NBL_SCENE_I_LEVEL_OF_DETAIL_LIBRARY_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/video/video.h"

namespace nbl
{
namespace scene
{
class ILevelOfDetailLibrary : public virtual core::IReferenceCounted
{
public:
    // TODO: Drawcall struct?
    using draw_call_t = uint32_t;
    // LoD will store a contiguous list of draw_call_t inside itself (first uint is the count)
    using lod_t = uint32_t;
    struct CullParameters
    {
        float distanceSq;

        inline bool operator<(const CullParameters& other) const
        {
            return distanceSq < other.distanceSq;
        }
    };
    // LoDTable will store a contiguous list of lod_t inside itself (first uint is the count)
    using lod_table_t = uint32_t;

    static inline core::smart_refctd_ptr<ILevelOfDetailLibrary> create(video::IVideoDriver* _driver, const uint32_t tableCapacity, const uint32_t lodLevelsCapacity, const uint32_t drawCallCapacity)
    {
        if(true)  // TODO: some checks and validation before creating?
            return nullptr;

        auto* lodl = new ILevelOfDetailLibrary(_driver /*,std::move(),std::move(),std::move()*/);
        return core::smart_refctd_ptr<ILevelOfDetailLibrary>(lodl, core::dont_grab);
    }

    // TODO: register/deregister drawcalls/lods/tables
    template<typename MeshIterator, typename CullParamsIterator>
    struct RegisterLoDTable
    {
        MeshIterator beginMeshes;
        MeshIterator endMeshes;
        CullParamsIterator beginCullParams;
    };
    template<typename MeshBufferIterator>
    struct RegisterLoD
    {
        MeshBufferIterator beginMeshBuffers;
        MeshBufferIterator endMeshBuffers;
    };

    template<typename MeshBufferIterator>
    draw_call_t registerDrawcalls(MeshBufferIterator begin, MeshBufferIterator end)
    {
        assert(false);  // TODO
    }
    template<typename MeshBufferIterator>
    draw_call_t deregisterDrawcalls(MeshBufferIterator begin, MeshBufferIterator end)
    {
        assert(false);  // TODO
    }

protected:
    ILevelOfDetailLibrary(video::IVideoDriver* _driver)
        : m_driver(_driver)
    {
    }
    ~ILevelOfDetailLibrary()
    {
        // everything drops itself automatically
    }

    video::IVideoDriver* m_driver;
};

}  // end namespace scene
}  // end namespace nbl

#endif

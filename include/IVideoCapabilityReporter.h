// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__
#define __NBL_I_VIDEO_CAPABILITY_REPORTER_H_INCLUDED__

#include <string>

#include "nbl/video/IGPUImageView.h"
#include "EDriverTypes.h"

namespace nbl
{
namespace video
{
//! .
class NBL_FORCE_EBO IVideoCapabilityReporter
{
public:
    //! Get type of video driver
    /** \return Type of driver. */
    virtual E_DRIVER_TYPE getDriverType() const = 0;

    //! enumeration for querying features of the video driver.
    enum E_DRIVER_FEATURE
    {
        //! Supports Alpha To Coverage (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
        EDF_ALPHA_TO_COVERAGE = 0,

        //! Supports geometry shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
        EDF_GEOMETRY_SHADER,

        //! Supports tessellation shaders (always in OpenGL 4.3+, Vulkan Mobile GPUs don't)
        EDF_TESSELLATION_SHADER,

        //! Whether we can download sub-areas of an IGPUTexture TODO: OpenGL 4.4+ ? Do we even ened this?
        EDF_GET_TEXTURE_SUB_IMAGE,

        //! Whether one cycle of read->write to the same pixel on an active FBO is supported (always in Vulkan) TODO: OpenGL 4.4+ ?
        EDF_TEXTURE_BARRIER,

        //! If we can attach a stencil only texture to an FBO, if not must use Depth+Stencil
        EDF_STENCIL_ONLY_TEXTURE,

        //! Whether we can get gl_DrawIDARB in GLSL (if not see https://www.g-truc.net/post-0518.html for ways to circumvent)
        EDF_SHADER_DRAW_PARAMS,

        //! Whether we can indirectly tell how many indirect draws to issue (rather than issuing 0 triangle draw calls)
        EDF_MULTI_DRAW_INDIRECT_COUNT,

        //! Whether we can know if the whole warp has a condition true, false, mixed, etc. NV_gpu_shader5 or ARB_shader_group_vote  TODO: Check the KHR_shader_subgroup_* extension
        EDF_SHADER_GROUP_VOTE,

        //! Whether we can know the warp/wavefront size and use ballot operations etc. NV_shader_thread_group or ARB_shader_ballot  TODO: Check the KHR_shader_subgroup_* extension
        EDF_SHADER_GROUP_BALLOT,

        //! Whether we can use Kepler-style shuffle instructions in a shader NV_shader_thread_shuffle  TODO: Check the KHR_shader_subgroup_* extension
        EDF_SHADER_GROUP_SHUFFLE,

        //! Whether we can force overlapping pixels to not rasterize in parallel, INTEL_fragment_shader_ordering, NV_fragment_shader_interlock or ARB_fragment_shader_interlock
        EDF_FRAGMENT_SHADER_INTERLOCK,

        //! Whether textures can be used by their hardware handles bindlessly (without specifying them in descriptor sets) TODO: What to do about this?
        EDF_BINDLESS_TEXTURE,

        //! Whether we can index samplers dynamically in a shader TODO: only in Vulkan or NV_gpu_shader5
        EDF_DYNAMIC_SAMPLER_INDEXING,

        //! A way to pass information between fragment shader invocations covering the same pixel
        EDF_INPUT_ATTACHMENTS,

        //other feature ideas are; bindless buffers, sparse texture, sparse texture 2

        //! Only used for counting the elements of this enum
        EDF_COUNT
    };

    virtual const core::smart_refctd_dynamic_array<std::string> getSupportedGLSLExtensions() const { return nullptr; };

    //! Queries the features of the driver.
    /** Returns true if a feature is available
		\param feature Feature to query.
		\return True if the feature is available, false if not. */
    virtual bool queryFeature(const E_DRIVER_FEATURE& feature) const { return false; }

    //! Gets name of this video driver.
    /** \return Returns the name of the video driver, e.g. in case
		of the Direct3D8 driver, it would return "Direct3D 8.1". */
    virtual const wchar_t* getName() const = 0;

    //! Get the current color format of the color buffer
    /** \return Color format of the color buffer. */
    virtual asset::E_FORMAT getColorFormat() const = 0;

    //! Get the graphics card vendor name.
    virtual std::string getVendorInfo() = 0;

    virtual uint32_t getMaxComputeWorkGroupSize(uint32_t _dimension) const { return 0u; }

    //! Get the maximum texture size supported.
    virtual const uint32_t* getMaxTextureSize(IGPUImageView::E_TYPE type) const = 0;

    //!
    virtual uint32_t getRequiredUBOAlignment() const { return 0u; }

    //!
    virtual uint32_t getRequiredSSBOAlignment() const { return 0u; }

    //!
    virtual uint32_t getRequiredTBOAlignment() const { return 0u; }

    //!
    virtual uint32_t getMinimumMemoryMapAlignment() const { return _NBL_MIN_MAP_BUFFER_ALIGNMENT; }

    virtual uint16_t retrieveDisplayRefreshRate() const { return 0u; }

    virtual uint64_t getMaxUBOSize() const { return 0ull; }
    virtual uint64_t getMaxSSBOSize() const { return 0ull; }
    virtual uint64_t getMaxTBOSizeInTexels() const { return 0ull; }
    virtual uint64_t getMaxBufferSize() const { return 0ull; }

    virtual uint32_t getMaxUBOBindings() const { return 0u; }
    virtual uint32_t getMaxSSBOBindings() const { return 0u; }
    virtual uint32_t getMaxTextureBindings() const { return 0u; }
    virtual uint32_t getMaxTextureBindingsCompute() const { return 0u; }
    virtual uint32_t getMaxImageBindings() const { return 0u; }

    virtual bool isAllowedBufferViewFormat(asset::E_FORMAT _fmt) const { return false; }
    virtual bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const { return false; }
    virtual bool isColorRenderableFormat(asset::E_FORMAT _fmt) const { return false; }
    virtual bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const { return false; }
    virtual bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const { return false; }
    virtual bool isHardwareBlendableFormat(asset::E_FORMAT _fmt) const { return false; }
};

}  // end namespace video
}  // end namespace nbl

#endif

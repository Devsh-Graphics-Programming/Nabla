// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_SCENE_I_RENDERPASS_MANAGER_H_INCLUDED__
#define __NBL_SCENE_I_RENDERPASS_MANAGER_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/video/video.h"

#include "nbl/scene/ITransformTreeManager.h"
//#include "nbl/scene/IRenderpass.h"

namespace nbl
{
namespace scene
{
// assign one <node_t,SensorProps> to be the camera of the renderpass
// register <draw_call_t,draw_indirect_t> when building a renderpass and keep it sorted
// when we want to translate draw_call_t to draw_indirect_t we just do a binary search

// register many <renderpass_t,node_t,lod_table_t> to be rendered
// TODO: Make LoDTable entries contain pointers to bindpose matrices and joint AABBs (culling of skinned models can only occur after LoD choice)
// SPECIAL: when registering skeletons, allocate the registrations contiguously to get a free translation table for skinning
// but reserve a per_view_data_t=<MVP,chosen_lod_t> for the output
// keep in `sparse_vector` to make contiguous
// convert <renderpass_t,node_t,per_view_data_t> into properly set up MDI calls, and compacted subranges of <node_t,per_view_data_t> ordered by <renderpass_t,draw_indirect_t> for use as per-Instance vertex attributes
class IRenderpassManager : public virtual core::IReferenceCounted
{
};

}  // end namespace scene
}  // end namespace nbl

#endif

// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_

#include <iosfwd>

#include "CCameraKeyframeTrack.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

/// @brief Serialize one camera keyframe track into an existing stream.
bool writeKeyframeTrack(std::ostream& out, const core::CCameraKeyframeTrack& track, int indent = 2);
/// @brief Deserialize one camera keyframe track from an existing stream.
bool readKeyframeTrack(std::istream& in, core::CCameraKeyframeTrack& track);

/// @brief Save one camera keyframe track to a file.
bool saveKeyframeTrackToFile(ISystem& system, const path& path, const core::CCameraKeyframeTrack& track, int indent = 2);
/// @brief Load one camera keyframe track from a file.
bool loadKeyframeTrackFromFile(ISystem& system, const path& path, core::CCameraKeyframeTrack& track);

} // namespace nbl::system

#endif // _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_

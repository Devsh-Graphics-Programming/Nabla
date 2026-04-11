// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_

#include <string>
#include <string_view>

#include "CCameraKeyframeTrack.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

struct CCameraKeyframeTrackPersistenceUtilities final
{
    /// @brief Serialize one camera keyframe track to JSON text.
    static std::string serializeKeyframeTrack(const core::CCameraKeyframeTrack& track, int indent = 2);
    /// @brief Deserialize one camera keyframe track from JSON text.
    static bool deserializeKeyframeTrack(std::string_view text, core::CCameraKeyframeTrack& track, std::string* error = nullptr);

    /// @brief Save one camera keyframe track to a file.
    static bool saveKeyframeTrackToFile(ISystem& system, const path& path, const core::CCameraKeyframeTrack& track, int indent = 2);
    /// @brief Load one camera keyframe track from a file.
    static bool loadKeyframeTrackFromFile(ISystem& system, const path& path, core::CCameraKeyframeTrack& track, std::string* error = nullptr);
};

} // namespace nbl::system

#endif // _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_

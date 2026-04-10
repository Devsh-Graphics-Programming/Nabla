// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_KEYFRAME_TRACK_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_HPP_

#include <algorithm>
#include <cmath>
#include <vector>

#include "CCameraPreset.hpp"

namespace nbl::core
{

/// @brief Reusable keyframe container plus selection state for playback tooling.
struct CCameraKeyframeTrack
{
    std::vector<CCameraKeyframe> keyframes;
    int selectedKeyframeIx = -1;
};

struct CCameraKeyframeTrackUtilities final
{
public:
    /// @brief Compare two keyframes by authored time and shared preset state.
    static bool compareKeyframes(const CCameraKeyframe& lhs, const CCameraKeyframe& rhs,
        double timeEps, double posEps, double rotEpsDeg, double scalarEps);

    /// @brief Compare two authored keyframe tracks with optional selection-state checking.
    static bool compareKeyframeTracks(const CCameraKeyframeTrack& lhs, const CCameraKeyframeTrack& rhs,
        double timeEps, double posEps, double rotEpsDeg, double scalarEps, bool compareSelection = true);

    /// @brief Compare only the serialized/authored content of two tracks and ignore transient UI selection state.
    static bool compareKeyframeTrackContent(const CCameraKeyframeTrack& lhs, const CCameraKeyframeTrack& rhs,
        double timeEps, double posEps, double rotEpsDeg, double scalarEps);

    static bool tryBuildKeyframeTrackPresetAtTime(const CCameraKeyframeTrack& track, float time, CCameraPreset& preset);

    static void sortKeyframeTrackByTime(CCameraKeyframeTrack& track);

    static void clampTrackTimeToKeyframes(const CCameraKeyframeTrack& track, float& time);

    static int selectKeyframeTrackNearestTime(CCameraKeyframeTrack& track, float time);

    static void normalizeSelectedKeyframeTrack(CCameraKeyframeTrack& track);

    static CCameraKeyframe* getSelectedKeyframe(CCameraKeyframeTrack& track);

    static const CCameraKeyframe* getSelectedKeyframe(const CCameraKeyframeTrack& track);

    static bool replaceSelectedKeyframePreset(CCameraKeyframeTrack& track, CCameraPreset preset);
};

} // namespace nbl::core

#endif // _C_CAMERA_KEYFRAME_TRACK_HPP_

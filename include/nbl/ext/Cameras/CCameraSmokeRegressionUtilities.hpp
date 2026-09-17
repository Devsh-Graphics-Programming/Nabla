// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_

#include <span>

#include "CCameraKeyframeTrack.hpp"
#include "CCameraMathUtilities.hpp"
#include "CCameraPresetFlow.hpp"
#include "ICamera.hpp"

namespace nbl::ext::cameras
{

using SCameraManipulationDelta = SCameraPoseDelta<hlsl::float64_t>;

struct SCameraSmokeComparisonThresholds final
{
    static constexpr double TinyScalarEpsilon = SCameraToolingThresholds::TinyScalarEpsilon;
    static constexpr double DefaultPositionTolerance = SCameraToolingThresholds::DefaultPositionTolerance;
    static constexpr double DefaultAngularToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static constexpr double DefaultScalarTolerance = SCameraToolingThresholds::ScalarTolerance;
    static constexpr double StrictPositionTolerance = SCameraToolingThresholds::ScalarTolerance;
    static constexpr double StrictAngularToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static constexpr double StrictScalarTolerance = SCameraToolingThresholds::ScalarTolerance;
    static constexpr double TrackTimeTolerance = SCameraToolingThresholds::ScalarTolerance;
};

struct CCameraSmokeRegressionUtilities final
{
public:
    /// @brief Measure one camera pose delta against an authored reference pose.
    static inline bool tryComputeCameraManipulationDelta(
        ICamera* camera,
        const hlsl::float64_t3& beforePosition,
        const hlsl::math::quaternion<hlsl::float64_t>& beforeOrientation,
        SCameraManipulationDelta& outDelta)
    {
        outDelta = {};
        if (!camera)
            return false;

        const auto& gimbal = camera->getGimbal();
        const auto afterPosition = gimbal.getPosition();
        const auto afterOrientation = hlsl::normalize(gimbal.getOrientation());
        return CCameraMathUtilities::tryComputePoseDelta(afterPosition, afterOrientation, beforePosition, beforeOrientation, outDelta);
    }

    /// @brief Manipulate a camera and report how far its pose moved in position and Euler-angle terms.
    static inline bool tryManipulateCameraAndMeasureDelta(
        ICamera* camera,
        std::span<const CVirtualGimbalEvent> events,
        SCameraManipulationDelta& outDelta,
        const double tinyEpsilon = SCameraSmokeComparisonThresholds::TinyScalarEpsilon)
    {
        outDelta = {};
        if (!camera || events.empty())
            return false;

        const auto& beforeGimbal = camera->getGimbal();
        const auto beforePosition = beforeGimbal.getPosition();
        const auto beforeOrientation = hlsl::normalize(beforeGimbal.getOrientation());
        if (!CCameraMathUtilities::isFiniteVec3(beforePosition) || !CCameraMathUtilities::isFiniteQuaternion(beforeOrientation))
            return false;

        if (!camera->manipulate(events))
            return false;

        if (!tryComputeCameraManipulationDelta(camera, beforePosition, beforeOrientation, outDelta))
            return false;

        return outDelta.position > tinyEpsilon || outDelta.rotationDeg > tinyEpsilon;
    }

    static inline bool comparePresetToCameraStateWithDefaultThresholds(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CCameraPreset& preset)
    {
        return CCameraPresetFlowUtilities::comparePresetToCameraState(
            solver,
            camera,
            preset,
            SCameraSmokeComparisonThresholds::DefaultPositionTolerance,
            SCameraSmokeComparisonThresholds::DefaultAngularToleranceDeg,
            SCameraSmokeComparisonThresholds::DefaultScalarTolerance);
    }

    static inline bool comparePresetToCameraStateWithStrictThresholds(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CCameraPreset& preset)
    {
        return CCameraPresetFlowUtilities::comparePresetToCameraState(
            solver,
            camera,
            preset,
            SCameraSmokeComparisonThresholds::StrictPositionTolerance,
            SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
            SCameraSmokeComparisonThresholds::StrictScalarTolerance);
    }

    static inline bool compareKeyframeTrackContentWithStrictThresholds(
        const CCameraKeyframeTrack& lhs,
        const CCameraKeyframeTrack& rhs)
    {
        return CCameraKeyframeTrackUtilities::compareKeyframeTrackContent(
            lhs,
            rhs,
            SCameraSmokeComparisonThresholds::TrackTimeTolerance,
            SCameraSmokeComparisonThresholds::StrictPositionTolerance,
            SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
            SCameraSmokeComparisonThresholds::StrictScalarTolerance);
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_

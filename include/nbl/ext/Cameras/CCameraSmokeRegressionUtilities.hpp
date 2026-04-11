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

namespace nbl::system
{

using SCameraManipulationDelta = hlsl::SCameraPoseDelta<hlsl::float64_t>;

struct SCameraSmokeComparisonThresholds final
{
    static constexpr double TinyScalarEpsilon = core::SCameraToolingThresholds::TinyScalarEpsilon;
    static constexpr double DefaultPositionTolerance = core::SCameraToolingThresholds::DefaultPositionTolerance;
    static constexpr double DefaultAngularToleranceDeg = core::SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static constexpr double DefaultScalarTolerance = core::SCameraToolingThresholds::ScalarTolerance;
    static constexpr double StrictPositionTolerance = core::SCameraToolingThresholds::ScalarTolerance;
    static constexpr double StrictAngularToleranceDeg = core::SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static constexpr double StrictScalarTolerance = core::SCameraToolingThresholds::ScalarTolerance;
    static constexpr double TrackTimeTolerance = core::SCameraToolingThresholds::ScalarTolerance;
};

struct CCameraSmokeRegressionUtilities final
{
public:
    /// @brief Measure one camera pose delta against an authored reference pose.
    static inline bool tryComputeCameraManipulationDelta(
        core::ICamera* camera,
        const hlsl::float64_t3& beforePosition,
        const hlsl::camera_quaternion_t<hlsl::float64_t>& beforeOrientation,
        SCameraManipulationDelta& outDelta)
    {
        outDelta = {};
        if (!camera)
            return false;

        const auto& gimbal = camera->getGimbal();
        const auto afterPosition = gimbal.getPosition();
        const auto afterOrientation = hlsl::CCameraMathUtilities::normalizeQuaternion(gimbal.getOrientation());
        return hlsl::CCameraMathUtilities::tryComputePoseDelta(afterPosition, afterOrientation, beforePosition, beforeOrientation, outDelta);
    }

    /// @brief Manipulate a camera and report how far its pose moved in position and Euler-angle terms.
    static inline bool tryManipulateCameraAndMeasureDelta(
        core::ICamera* camera,
        std::span<const core::CVirtualGimbalEvent> events,
        SCameraManipulationDelta& outDelta,
        const double tinyEpsilon = SCameraSmokeComparisonThresholds::TinyScalarEpsilon)
    {
        outDelta = {};
        if (!camera || events.empty())
            return false;

        const auto& beforeGimbal = camera->getGimbal();
        const auto beforePosition = beforeGimbal.getPosition();
        const auto beforeOrientation = hlsl::CCameraMathUtilities::normalizeQuaternion(beforeGimbal.getOrientation());
        if (!hlsl::CCameraMathUtilities::isFiniteVec3(beforePosition) || !hlsl::CCameraMathUtilities::isFiniteQuaternion(beforeOrientation))
            return false;

        if (!camera->manipulate(events))
            return false;

        if (!tryComputeCameraManipulationDelta(camera, beforePosition, beforeOrientation, outDelta))
            return false;

        return outDelta.position > tinyEpsilon || outDelta.rotationDeg > tinyEpsilon;
    }

    static inline bool comparePresetToCameraStateWithDefaultThresholds(
        const core::CCameraGoalSolver& solver,
        core::ICamera* camera,
        const core::CCameraPreset& preset)
    {
        return core::CCameraPresetFlowUtilities::comparePresetToCameraState(
            solver,
            camera,
            preset,
            SCameraSmokeComparisonThresholds::DefaultPositionTolerance,
            SCameraSmokeComparisonThresholds::DefaultAngularToleranceDeg,
            SCameraSmokeComparisonThresholds::DefaultScalarTolerance);
    }

    static inline bool comparePresetToCameraStateWithStrictThresholds(
        const core::CCameraGoalSolver& solver,
        core::ICamera* camera,
        const core::CCameraPreset& preset)
    {
        return core::CCameraPresetFlowUtilities::comparePresetToCameraState(
            solver,
            camera,
            preset,
            SCameraSmokeComparisonThresholds::StrictPositionTolerance,
            SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
            SCameraSmokeComparisonThresholds::StrictScalarTolerance);
    }

    static inline bool compareKeyframeTrackContentWithStrictThresholds(
        const core::CCameraKeyframeTrack& lhs,
        const core::CCameraKeyframeTrack& rhs)
    {
        return core::CCameraKeyframeTrackUtilities::compareKeyframeTrackContent(
            lhs,
            rhs,
            SCameraSmokeComparisonThresholds::TrackTimeTolerance,
            SCameraSmokeComparisonThresholds::StrictPositionTolerance,
            SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
            SCameraSmokeComparisonThresholds::StrictScalarTolerance);
    }
};

} // namespace nbl::system

#endif // _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_

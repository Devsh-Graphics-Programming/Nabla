// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _S_CAMERA_RIG_POSE_HPP_
#define _S_CAMERA_RIG_POSE_HPP_

#include "CCameraMathUtilities.hpp"

namespace nbl::core
{

/// @brief Canonical camera pose consisting of world-space position and orientation.
///
/// This type stores only pose data. Higher-level types add target-relative,
/// dynamic-perspective, or path-specific state around it.
struct SCameraRigPose
{
    /// @brief Camera origin in world space.
    hlsl::float64_t3 position = hlsl::float64_t3(0.0);
    /// @brief Camera orientation in world space expressed as a unit quaternion.
    hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>();
};

} // namespace nbl::core

#endif // _S_CAMERA_RIG_POSE_HPP_

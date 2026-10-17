// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_C_CAMERA_GIMBAL_HPP_
#define _NBL_C_CAMERA_GIMBAL_HPP_

#include <cassert>

#include "IGimbal.hpp"

namespace nbl::ext::cameras
{
    /// @brief Camera pose that also serves the world-to-view matrix derived from it.
    ///
    /// Every runtime camera owns one. The view matrix is rebuilt on the first read that follows a manipulation
    /// and cached against `getManipulationCounter()` until the next one, so writing the pose several times in a
    /// row costs one rebuild. The cache is mutated while reading, so a single instance is not safe to read from
    /// several threads at once.
    class CCameraGimbal : public IGimbal
    {
    public:
        using base_t = IGimbal;
        using base_t::base_t;

        /// @brief World-to-view matrix: rows are the camera's local X, Y, Z in world space, each with `-dot(axis, position)`.
        inline const hlsl::float64_t3x4& getViewMatrix() const
        {
            const auto counter = getManipulationCounter();
            if (m_cachedCounter != counter)
            {
                rebuildView();
                m_cachedCounter = counter;
            }
            return m_cachedView;
        }

    private:
        inline void rebuildView() const
        {
            // The view matrix is the inverse of the pose: the transpose of the orientation's rotation, with the
            // translation that moves the position to the origin. No handedness enters here. The quaternion is only a
            // rotation: it says how the world axes are turned to become the camera's local axes, and the view matrix
            // undoes that turn. Which local axis is the look direction is not stored in either; it is decided by the
            // projection paired with this matrix (`lhPerspective` looks along +Z of view space, `rhPerspective`
            // along -Z), and `orientationFromBasis` built the quaternion so that the camera's forward sits on the
            // axis that projection looks along.
            // columns are the camera's local axes in world space, so the rows of the transpose are the view rotation
            const auto rotation = hlsl::_static_cast<hlsl::float64_t3x3>(getOrientation());
            assert((hlsl::math::linalg::RuntimeTraits<hlsl::float64_t3x3>::create(rotation).orthonormal));
            const auto rotationT = hlsl::transpose(rotation);

            const auto& position = getPosition();
            for (uint32_t row = 0u; row < 3u; ++row)
                m_cachedView[row] = hlsl::float64_t4(rotationT[row], -hlsl::dot(rotationT[row], position));
        }

        mutable hlsl::float64_t3x4 m_cachedView;
        /// @brief Counter value the cache was built from; the initial value forces the first read to build it.
        mutable uint64_t m_cachedCounter = ~0ull;
    };
} // namespace nbl::ext::cameras

#endif // _NBL_C_CAMERA_GIMBAL_HPP_

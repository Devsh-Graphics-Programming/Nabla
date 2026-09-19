// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_C_CAMERA_GIMBAL_HPP_
#define _NBL_C_CAMERA_GIMBAL_HPP_

#include <cassert>

#include "IGimbal.hpp"

namespace nbl::ext::cameras
{
    /// @brief Camera pose that also serves the left-handed world-to-view matrix derived from it.
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

        /// @brief Left-handed world-to-view matrix: rows are `(axis, -dot(axis, position))` for right, up, forward.
        inline const hlsl::float64_t3x4& getViewMatrixLH() const
        {
            const auto counter = getManipulationCounter();
            if (m_cachedCounter != counter)
            {
                rebuildView();
                m_cachedCounter = counter;
            }
            return m_cachedViewLH;
        }

        /// @brief Right-handed world-to-view matrix: the left-handed one with the forward row negated.
        inline hlsl::float64_t3x4 getViewMatrixRH() const
        {
            auto rhViewMatrix = getViewMatrixLH();
            rhViewMatrix[2u] *= -1.0;
            return rhViewMatrix;
        }

    private:
        inline void rebuildView() const
        {
            const auto basis = getBasis();
            assert((hlsl::math::linalg::RuntimeTraits<hlsl::float64_t3x3>::create(basis.getRotationMatrix()).orthonormal));

            const auto& position = getPosition();
            m_cachedViewLH[0u] = hlsl::float64_t4(basis.right, -hlsl::dot(basis.right, position));
            m_cachedViewLH[1u] = hlsl::float64_t4(basis.up, -hlsl::dot(basis.up, position));
            m_cachedViewLH[2u] = hlsl::float64_t4(basis.forward, -hlsl::dot(basis.forward, position));
        }

        mutable hlsl::float64_t3x4 m_cachedViewLH;
        /// @brief Counter value the cache was built from; the initial value forces the first read to build it.
        mutable uint64_t m_cachedCounter = ~0ull;
    };
} // namespace nbl::ext::cameras

#endif // _NBL_C_CAMERA_GIMBAL_HPP_

// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_C_CAMERA_WITH_PROJECTIONS_HPP_
#define _NBL_C_CAMERA_WITH_PROJECTIONS_HPP_

#include <vector>

#include "ICameraWithProjections.hpp"

namespace nbl::ext::cameras
{
	/// @brief Concrete implementation of `ICameraWithProjections` owning its projection entries in a vector.
	class CCameraWithProjections : public ICameraWithProjections
	{
	public:
		virtual ~CCameraWithProjections() = default;

		/// @brief Create the wrapper only when a valid camera instance is available.
		inline static core::smart_refctd_ptr<CCameraWithProjections> create(core::smart_refctd_ptr<ICamera>&& camera)
		{
			if (!camera)
				return nullptr;

			return core::smart_refctd_ptr<CCameraWithProjections>(new CCameraWithProjections(core::smart_refctd_ptr(camera)), core::dont_grab);
		}

		/// @brief Return the number of stored projection entries.
		virtual uint32_t getProjectionCount() const override
		{
			return static_cast<uint32_t>(m_projections.size());
		}

		/// @brief Return one stored projection entry.
		virtual const CPlanarProjection& getProjection(uint32_t index) const override
		{
			assert(index < m_projections.size());
			return m_projections[index];
		}

		/// @brief Expose mutable access to the owned projection entries.
		inline std::vector<CPlanarProjection>& getProjections()
		{
			return m_projections;
		}

	protected:
		CCameraWithProjections(core::smart_refctd_ptr<ICamera>&& camera)
			: ICameraWithProjections(core::smart_refctd_ptr(camera)) {}

		std::vector<CPlanarProjection> m_projections;
	};

} // namespace nbl::ext::cameras

#endif // _NBL_C_CAMERA_WITH_PROJECTIONS_HPP_

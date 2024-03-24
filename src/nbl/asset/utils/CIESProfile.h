// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
#define __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__

#include "nbl/core/Types.h"
#include <sstream>

namespace nbl 
{
    namespace asset 
    {
        class CIESProfile 
        {
            public:
                using IES_STORAGE_FORMAT = double;

                _NBL_STATIC_INLINE_CONSTEXPR size_t CDC_DEFAULT_TEXTURE_WIDTH = 1024;
                _NBL_STATIC_INLINE_CONSTEXPR size_t CDC_DEFAULT_TEXTURE_HEIGHT = 1024;

                _NBL_STATIC_INLINE_CONSTEXPR IES_STORAGE_FORMAT MAX_VANGLE = 180.0;
                _NBL_STATIC_INLINE_CONSTEXPR IES_STORAGE_FORMAT MAX_HANGLE = 360.0;

                enum PhotometricType : uint32_t 
                {
                    TYPE_NONE,
                    TYPE_C,
                    TYPE_B,
                    TYPE_A,
                };

                CIESProfile() = default;
                ~CIESProfile() = default;

                const core::vector<IES_STORAGE_FORMAT>& getHoriAngles() const { return hAngles; }
                const core::vector<IES_STORAGE_FORMAT>& getVertAngles() const { return vAngles; }
                const core::vector<IES_STORAGE_FORMAT>& getData() const { return data; }

                IES_STORAGE_FORMAT getValue(size_t i, size_t j) const { return data[vAngles.size() * i + j]; }
                IES_STORAGE_FORMAT getMaxValue() const { return maxValue; }

                const IES_STORAGE_FORMAT& getIntegral() const;
                const IES_STORAGE_FORMAT& getIntegralFromGrid() const { return integral; }

                //! Candlepower distribution curve plot as ICPUImageView
                /*
                    Creates 2D texture of CDC with width & height extent, zAngleDegreeRotation may be
                    used to rotate normalized direction vector obtained from octahdronUVToDir utility with
                */

                core::smart_refctd_ptr<asset::ICPUImageView> createCDCTexture(const size_t& width = CDC_DEFAULT_TEXTURE_WIDTH, const size_t& height = CDC_DEFAULT_TEXTURE_HEIGHT) const;

            private:
                CIESProfile(PhotometricType type, size_t hSize, size_t vSize)
                    : type(type), hAngles(hSize), vAngles(vSize), data(hSize* vSize) {}

                // TODO for @Hazard, I would move it into separate file, we may use this abstraction somewhere too
                static inline std::pair<float, float> sphericalDirToAngles(const core::vectorSIMDf& dir);
                static inline core::vectorSIMDf octahdronUVToDir(const float& u, const float& v);
                
                void addHoriAngle(IES_STORAGE_FORMAT hAngle)
                {
                    hAngles.push_back(hAngle);
                    data.resize(getHoriAngles().size() * vAngles.size());
                }

                void setValue(size_t i, size_t j, IES_STORAGE_FORMAT val) { data[vAngles.size() * i + j] = val; }

                const IES_STORAGE_FORMAT& sample(IES_STORAGE_FORMAT vAngle, IES_STORAGE_FORMAT hAngle) const;

                PhotometricType type;
                core::vector<IES_STORAGE_FORMAT> hAngles;
                core::vector<IES_STORAGE_FORMAT> vAngles;
                core::vector<IES_STORAGE_FORMAT> data;
                IES_STORAGE_FORMAT maxValue = {};
                mutable IES_STORAGE_FORMAT integral = {};

                friend class CIESProfileParser;
        };
    }
}

#endif // __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
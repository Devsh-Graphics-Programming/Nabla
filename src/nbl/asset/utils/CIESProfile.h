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
                _NBL_STATIC_INLINE_CONSTEXPR size_t CDC_DEFAULT_TEXTURE_WIDTH = 1024;
                _NBL_STATIC_INLINE_CONSTEXPR size_t CDC_DEFAULT_TEXTURE_HEIGHT = 1024;

                _NBL_STATIC_INLINE_CONSTEXPR double MAX_VANGLE = 180.0;
                _NBL_STATIC_INLINE_CONSTEXPR double MAX_HANGLE = 360.0;

                enum PhotometricType : uint32_t {
                    TYPE_NONE,
                    TYPE_C,
                    TYPE_B,
                    TYPE_A,
                };

                CIESProfile() = default;
                CIESProfile(PhotometricType type, size_t hSize, size_t vSize)
                    : type(type), hAngles(hSize), vAngles(vSize), data(hSize* vSize) {}

                ~CIESProfile() = default;

                core::vector<double>& getHoriAngles() { return hAngles; }
                const core::vector<double>& getHoriAngles() const { return hAngles; }
                core::vector<double>& getVertAngles() { return vAngles; }
                const core::vector<double>& getVertAngles() const { return vAngles; }
                const size_t& getHoriSize() const { return hAngles.size(); }
                const size_t& getVertSize() const { return vAngles.size(); }
                double getValue(size_t i, size_t j) const { return data[getVertSize() * i + j]; }
                double getMaxValue() const { return *std::max_element(std::begin(data), std::end(data)); }

                void addHoriAngle(double hAngle) 
                {
                    hAngles.push_back(hAngle);
                    data.resize(getHoriSize() * getVertSize());
                }

                void setValue(size_t i, size_t j, double val) { data[getVertSize() * i + j] = val; }
                
                const double& sample(double vAngle, double hAngle) const;
                const double& getIntegral() const;

                //! Candlepower distribution curve plot as ICPUImageView
                /*
                    Creates 2D texture of CDC with width & height extent, zAngleDegreeRotation may be
                    used to rotate normalized direction vector obtained from octahdronUVToDir utility with
                */
                core::smart_refctd_ptr<asset::ICPUImageView> createCDCTexture(const float& zAngleDegreeRotation = 0.f, const size_t& width = CDC_DEFAULT_TEXTURE_WIDTH, const size_t& height = CDC_DEFAULT_TEXTURE_HEIGHT) const;

            private:
                // TODO for @Hazard, I would move it into separate file, we may use this abstraction somewhere too
                static inline std::pair<float, float> sphericalDirToAngles(const core::vectorSIMDf& dir);
                static inline core::vectorSIMDf octahdronUVToDir(const float& u, const float& v, const float& zAngleDegrees);

                PhotometricType type;
                core::vector<double> hAngles;
                core::vector<double> vAngles;
                core::vector<double> data;
        };
    }
}

#endif // __NBL_ASSET_C_IES_PROFILE_H_INCLUDED__
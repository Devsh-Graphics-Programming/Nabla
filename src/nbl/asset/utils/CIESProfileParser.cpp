#include "CIESProfileParser.h"

using namespace nbl;
using namespace asset;

int CIESProfileParser::getInt(const char* errorMsg)
{
    return readStream<int>(errorMsg);
}

double CIESProfileParser::getDouble(const char* errorMsg)
{
    return readStream<double>(errorMsg);
}

bool CIESProfileParser::parse(CIESProfile& result) 
{
    auto removeTrailingWhiteChars = [](std::string& str) -> void
    {
        if (std::isspace(str.back()))
        {
            auto it = str.rbegin();
            while (it != str.rend() && std::isspace(static_cast<unsigned char>(*it)))
                ++it;

            str.erase(it.base(), str.end());
        }
    };

    // skip metadata
    std::string line;

    std::getline(ss, line);
    removeTrailingWhiteChars(line);

   
    CIESProfile::properties_t::Version iesVersion;
    if (line.find(SIG_LM63_1995.data()) != std::string::npos)
        iesVersion = CIESProfile::properties_t::V_1995;
    else if (line.find(SIG_LM63_2002.data()) != std::string::npos)
        iesVersion = CIESProfile::properties_t::V_2002;
    else if (line.find(SIG_IESNA91.data()) != std::string::npos)
        iesVersion = CIESProfile::properties_t::V_1995;
    else if (line.find(SIG_ERCO_LG.data()) != std::string::npos)
        iesVersion = CIESProfile::properties_t::V_1995;
    else
    {
        errorMsg = "Unknown IESNA:LM-63 version, the IES input being parsed is invalid!";
        return false;
    }

     while (std::getline(ss, line)) 
    {
        removeTrailingWhiteChars(line);
        if (line == "TILT=INCLUDE" || line == "TILT=NONE")
            break;
    }

    if (line == "TILT=INCLUDE") {
        double lampToLuminaire = getDouble("lampToLuminaire truncated");
        int numTilt = getDouble("numTilt truncated");
        for (int i = 0; i < numTilt; i++)
            getDouble("tilt angle truncated");
        for (int i = 0; i < numTilt; i++)
            getDouble("tilt multiplying factor truncated");
    }
    else if (line != "TILT=NONE") {
        errorMsg = "TILT not specified";
        return false;
    }

    int numLamps = getInt("numLamps truncated");
    double lumensPerLamp = getDouble("lumensPerLamp truncated");
    double candelaMultiplier = getDouble("candelaMultiplier truncated");
    int vSize = getInt("vSize truncated");
    int hSize = getInt("hSize truncated");

    int type_ = getInt("type truncated");
    if (error)
        return false;
    if (type_ <= 0 || type_ > 3) {
        errorMsg = "unrecognized type";
        return false;
    }
    auto type = static_cast<CIESProfile::properties_t::PhotometricType>(type_);
    if (type != CIESProfile::properties_t::TYPE_C) {
        errorMsg = "Only type C is supported for now";
        return false;
    }

    int unitsType = getInt("unitsType truncated");
    double width = getDouble("width truncated"),
    length = getDouble("length truncated"),
    height = getDouble("height truncated"),
    ballastFactor = getDouble("ballastFactor truncated"),
    reserved = getDouble("reserved truncated"),
    inputWatts = getDouble("inputWatts truncated");

    if (error)
        return false;

    {
        CIESProfile::properties_t init;
		init.setType(type);
		init.setVersion(iesVersion);
		init.maxCandelaValue = 0.f;
		init.totalEmissionIntegral = 0.f;
		init.avgEmmision = 0.f;
        result = CIESProfile(init, hlsl::uint32_t2(hSize, vSize));
    }

    if (vSize < 2)
        return false;

    using angle_t = CIESProfile::accessor_t::angle_t;
    using candela_t = CIESProfile::accessor_t::candela_t;

    auto& vAngles = result.accessor.vAngles;
    for (int i = 0; i < vSize; i++) {
        vAngles[i] = static_cast<angle_t>(getDouble("vertical angle truncated"));
    }
    if (!std::is_sorted(vAngles.begin(), vAngles.end())) {
        errorMsg = "Vertical angles should be sorted";
        return false;
    }
    if (vAngles[0] != 0.f && vAngles[0] != 90.f) {
        errorMsg = "First vertical angle must be 0 or 90 in type C";
        return false;
    }
    if (vAngles[vSize - 1] != 90.f && vAngles[vSize - 1] != 180.f) {
        errorMsg = "Last vertical angle must be 90 or 180 in type C";
        return false;
    }

    auto& hAngles = result.accessor.hAngles;
    for (int i = 0; i < hSize; i++) {
        hAngles[i] = static_cast<angle_t>(getDouble("horizontal angle truncated"));
        if (i != 0 && hAngles[i - 1] > hAngles[i])
            return false; // Angles should be sorted
    }

    float fluxMultiplier = 1.0f;
    {
        const auto firstHAngle = hAngles.front();
        const auto lastHAngle = hAngles.back();

        if (lastHAngle == 0.f)
            result.accessor.properties.setSymmetry(CIESProfile::properties_t::ISOTROPIC);
        else if (lastHAngle == 90.f)
        {
            result.accessor.properties.setSymmetry(CIESProfile::properties_t::QUAD_SYMETRIC);
            fluxMultiplier = 4.f;
        }
        else if (lastHAngle == 180.f)
        {
            result.accessor.properties.setSymmetry(CIESProfile::properties_t::HALF_SYMETRIC);
            fluxMultiplier = 2.0;
        }
        else if (lastHAngle == 360.f)
            result.accessor.properties.setSymmetry(CIESProfile::properties_t::NO_LATERAL_SYMMET);
        else
        {
            if (firstHAngle == 90.f && lastHAngle == 270.f && iesVersion == CIESProfile::properties_t::V_1995)
            {
                result.accessor.properties.setSymmetry(CIESProfile::properties_t::OTHER_HALF_SYMMETRIC);
                fluxMultiplier = 2.f;

                for (auto& angle : hAngles)
                    angle -= firstHAngle; // patch the profile to HALF_SYMETRIC by shifting [90,270] range to [0, 180]
            }
            else
                return false;
        }
    }
	const auto symmetry = result.accessor.properties.getSymmetry();

    {
        const double factor = ballastFactor * candelaMultiplier;
        for (int i = 0; i < hSize; i++)
            for (int j = 0; j < vSize; j++)
                result.accessor.setValue(hlsl::uint32_t2(i, j), static_cast<candela_t>(factor * getDouble("intensity value truncated")));
    }

    float totalEmissionIntegral = 0.0, nonZeroEmissionDomainSize = 0.0;
    constexpr auto FULL_SOLID_ANGLE = 4.0f * core::PI<float>();

    // TODO: this code could have two separate inner for loops for `result.symmetry != CIESProfile::ISOTROPIC` cases 
    const auto H_ANGLES_I_RANGE = symmetry != CIESProfile::properties_t::ISOTROPIC ? result.accessor.hAngles.size() - 1 : 1;
    const auto V_ANGLES_I_RANGE = result.accessor.vAngles.size() - 1;

    float smallestRangeSolidAngle = FULL_SOLID_ANGLE;
    for (size_t j = 0; j < V_ANGLES_I_RANGE; j++)
    {
        const float thetaRad = core::radians<float>(result.accessor.vAngles[j]);
        const float cosLo = std::cos(thetaRad);
        const float cosHi = std::cos(core::radians<float>(result.accessor.vAngles[j+1]));
        const float dsinTheta = cosLo - cosHi;

        float stripIntegral = 0.f;
        float nonZeroStripDomain = 0.f;
        for (size_t i = 0; i < H_ANGLES_I_RANGE; i++)
        {
            const float dPhiRad = symmetry != CIESProfile::properties_t::ISOTROPIC ? core::radians<float>(hAngles[i + 1] - hAngles[i]) : (core::PI<float>() * 2.0f);
            // TODO: in reality one should transform the 4 vertices (or 3) into octahedral map, work out the dUV/dPhi and dUV/dTheta vectors as-if for Anisotropic Filtering
            // then choose the minor axis length, and use that as a pixel size (since looking for smallest thing, dont have to worry about handling discont)
            const float solidAngle = dsinTheta * dPhiRad;
            if (solidAngle<smallestRangeSolidAngle)
                smallestRangeSolidAngle = solidAngle;

            const auto candelaValue = result.accessor.value(hlsl::uint32_t2(i, j));

            // interpolate candela value spanned onto a solid angle
            const auto candelaAverage = symmetry != CIESProfile::properties_t::ISOTROPIC ?
                  0.25f * (candelaValue + result.accessor.value(hlsl::uint32_t2(i + 1, j)) + result.accessor.value(hlsl::uint32_t2(i, j + 1)) + result.accessor.value(hlsl::uint32_t2(i + 1, j + 1)))
                : 0.5f * (candelaValue + result.accessor.value(hlsl::uint32_t2(i, j + 1)));

            if (result.accessor.properties.maxCandelaValue < candelaValue)
                result.accessor.properties.maxCandelaValue = candelaValue;

            stripIntegral += candelaAverage*dPhiRad;
            if (candelaAverage>0.f)
                nonZeroStripDomain += dPhiRad;
        }
        totalEmissionIntegral += stripIntegral*dsinTheta;
        nonZeroEmissionDomainSize += nonZeroStripDomain*dsinTheta;
    }

    // assuming octahedral map
    {
        const uint32_t maxDimMeasureSize = core::sqrt(FULL_SOLID_ANGLE/smallestRangeSolidAngle);
        result.accessor.properties.optimalIESResolution = decltype(result.accessor.properties.optimalIESResolution){ maxDimMeasureSize, maxDimMeasureSize };
        auto& res = result.accessor.properties.optimalIESResolution *= 2u; // safe bias for our bilinear interpolation to work nicely and increase resolution of a profile
		res.x = core::max(res.x, CIESProfile::texture_t::MinTextureWidth);
		res.y = core::max(res.y, CIESProfile::texture_t::MinTextureHeight);
    }

    assert(nonZeroEmissionDomainSize >= 0.f);
    //assert(nonZeroEmissionDomainSize*fluxMultiplier =approx= 2.f*(cosBack-cosFront)*PI);
    if (nonZeroEmissionDomainSize <= std::numeric_limits<float>::min()) // protect us from division by small numbers (just in case, we should never hit it)
        return false;

    result.accessor.properties.avgEmmision = totalEmissionIntegral / static_cast<decltype(totalEmissionIntegral)>(nonZeroEmissionDomainSize);
    result.accessor.properties.totalEmissionIntegral = totalEmissionIntegral * fluxMultiplier; // we use fluxMultiplier to calculate final total emission for case where we have some symmetry between planes (fluxMultiplier is 1.0f if ISOTROPIC or NO_LATERAL_SYMMET because they already have correct total emission integral calculated), also note it doesn't affect average emission at all
	{
        const float cosLo = std::cos(core::radians(result.accessor.vAngles.front()));
        const float cosHi = std::cos(core::radians<float>(result.accessor.vAngles.back()));
        const float dsinTheta = cosLo - cosHi;
        result.accessor.properties.fullDomainAvgEmission = result.accessor.properties.totalEmissionIntegral*(0.5f/core::PI<float>())/dsinTheta;
	}

    return !error;
}

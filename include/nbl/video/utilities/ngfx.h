#ifndef _NBL_VIDEO_UTILITIES_NGFX_H_INCLUDED_
#define _NBL_VIDEO_UTILITIES_NGFX_H_INCLUDED_

#include "C:\Program Files\NVIDIA Corporation\Nsight Graphics 2024.1.0\SDKs\NsightGraphicsSDK\0.8.0\include\NGFX_Injection.h"

namespace nbl::video
{
    struct SNGFXIntegration
    {
        bool useNGFX;
        NGFX_Injection_InstallationInfo versionInfo;
    };

    bool injectNGFXToProcess(SNGFXIntegration& api)
    {
        uint32_t numInstallations = 0;
        auto result = NGFX_Injection_EnumerateInstallations(&numInstallations, nullptr);
        if (numInstallations == 0 || NGFX_INJECTION_RESULT_OK != result)
        {
            api.useNGFX = false;
            return false;
        }

        std::vector<NGFX_Injection_InstallationInfo> installations(numInstallations);
        result = NGFX_Injection_EnumerateInstallations(&numInstallations, installations.data());
        if (numInstallations == 0 || NGFX_INJECTION_RESULT_OK != result)
        {
            api.useNGFX = false;
            return false;
        }

        // get latest installation
        api.versionInfo = installations.back();

        uint32_t numActivities = 0;
        result = NGFX_Injection_EnumerateActivities(&api.versionInfo, &numActivities, nullptr);
        if (numActivities == 0 || NGFX_INJECTION_RESULT_OK != result)
        {
            api.useNGFX = false;
            return false;
        }

        std::vector<NGFX_Injection_Activity> activities(numActivities);
        result = NGFX_Injection_EnumerateActivities(&api.versionInfo, &numActivities, activities.data());
        if (NGFX_INJECTION_RESULT_OK != result)
        {
            api.useNGFX = false;
            return false;
        }

        const NGFX_Injection_Activity* pActivityToInject = nullptr;
        for (const NGFX_Injection_Activity& activity : activities)
        {
            if (activity.type == NGFX_INJECTION_ACTIVITY_FRAME_DEBUGGER)    // only want frame debugger
            {
                pActivityToInject = &activity;
                break;
            }
        }

        if (!pActivityToInject) {
            api.useNGFX = false;
            return false;
        }

        result = NGFX_Injection_InjectToProcess(&api.versionInfo, pActivityToInject);
        if (NGFX_INJECTION_RESULT_OK != result)
        {
            api.useNGFX = false;
            return false;
        }

        return true;
    }

    using ngfx_api_t = SNGFXIntegration;
}

#endif //_NBL_VIDEO_UTILITIES_NGFX_H_INCLUDED_
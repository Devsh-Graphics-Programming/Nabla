#ifndef __NBL_VIDEO_I_DEBUG_CALLBACK_H_INCLUDED__
#define __NBL_VIDEO_I_DEBUG_CALLBACK_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <cstdint>

#include "nbl/system/ILogger.h"

namespace nbl::video
{

class IDebugCallback
{
    public:
        system::ILogger* getLogger() const { return m_logger.get(); }
        void* getExtraUserData() const { return m_extraUserData; }

    protected:
        IDebugCallback(core::smart_refctd_ptr<system::ILogger>&& _logger, void* _extraUserData=nullptr) : m_logger(std::move(_logger)), m_extraUserData(_extraUserData) {}

        core::smart_refctd_ptr<system::ILogger> m_logger;
        void* m_extraUserData;
};

}

#endif

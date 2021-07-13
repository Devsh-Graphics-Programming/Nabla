#include "CVulkanConnection.h"

namespace nbl
{
namespace video
{

core::smart_refctd_ptr<IAPIConnection> createVulkanConnection(uint32_t appVer, const char* appName, const SDebugCallback& dbgCb)
{
    return core::make_smart_refctd_ptr<CVulkanConnection>(appVer, appName, dbgCb);
}

}
}
#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{
	
ISemaphore::WAIT_RESULT ISemaphore::future_base_t::wait() const
{
	if (!m_semaphore)
		return ISemaphore::WAIT_RESULT::SUCCESS;
	auto* device = const_cast<ILogicalDevice*>(m_semaphore->getOriginDevice());
	const SWaitInfo waitInfos[] = {{.semaphore=m_semaphore.get(),.value=m_waitValue}};
	return device->blockForSemaphores(waitInfos);
}

}

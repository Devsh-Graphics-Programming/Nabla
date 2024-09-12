#if defined(NBL_LOG) || defined(NBL_LOG_ERROR)
	#error redefinition of NBL_LOG/NBL_LOG_ERROR. did you forgot to undefine logging macros somewhere? #include "nbl/undefine_logging_macros.h"
#elif !defined(_GIT_INFO_H_INCLUDED_)
	#error logging macros require git meta info, include "git_info.h"
#else
	#define NBL_LOG(SEVERITY, FORMAT, ...) NBL_LOG_FUNCTION(FORMAT" [%s][%s - %s:%d]", SEVERITY, __VA_ARGS__, nbl::gtml::nabla_git_info.commitShortHash, __FUNCTION__, __FILE__, __LINE__);
	#define NBL_LOG_ERROR(FORMAT, ...) NBL_LOG(nbl::system::ILogger::ELL_ERROR, FORMAT, __VA_ARGS__)
#endif
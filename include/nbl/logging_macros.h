#ifdef _GIT_INFO_H_INCLUDED_
// when "git_info.h" is included
#	define NBL_LOG(SEVERITY, FORMAT, ...) NBL_LOG_FUNCTION(FORMAT" [%s][%s - %s:%d]", SEVERITY, __VA_ARGS__, nbl::gtml::nabla_git_info.commitShortHash, __FUNCTION__, __FILE__, __LINE__);
#	define NBL_LOG_ERROR(FORMAT, ...) NBL_LOG(nbl::system::ILogger::ELL_ERROR, FORMAT, __VA_ARGS__)
#else
#	define NBL_LOG(SEVERITY, FORMAT, ...) NBL_LOG_FUNCTION(FORMAT" [%s - %s:%d]", SEVERITY, __VA_ARGS__, __FUNCTION__, __FILE__, __LINE__);
#	define NBL_LOG_ERROR(FORMAT, ...) NBL_LOG(nbl::system::ILogger::ELL_ERROR, FORMAT, __VA_ARGS__)
#endif
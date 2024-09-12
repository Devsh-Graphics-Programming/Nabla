#ifdef __GIT_INFO_H_INCLUDED__
// when "git_info.h" is included
#	define LOG(SEVERITY, FORMAT, ...) LOG_FUNCTION(FORMAT" [%s][%s - %s:%d]", SEVERITY, __VA_ARGS__, nbl::gtml::nabla_git_info.commitShortHash, __FUNCTION__, __FILE__, __LINE__);
#	define LOG_ERROR(FORMAT, ...) LOG(nbl::system::ILogger::ELL_ERROR, FORMAT, __VA_ARGS__)
#else
#	define LOG(SEVERITY, FORMAT, ...) LOG_FUNCTION(FORMAT" [%s - %s:%d]", SEVERITY, __VA_ARGS__, __FUNCTION__, __FILE__, __LINE__);
#	define LOG_ERROR(FORMAT, ...) LOG(nbl::system::ILogger::ELL_ERROR, FORMAT, __VA_ARGS__)
#endif
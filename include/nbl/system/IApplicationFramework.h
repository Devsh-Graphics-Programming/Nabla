#ifndef	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_
#define	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_

namespace nbl::system
{
	class IApplicationFramework : public core::IReferenceCounted
	{
	public:
        virtual void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) = 0;
        IApplicationFramework(const system::path& _cwd) : CWDOnStartup(_cwd)
		{

		}
        void onAppInitialized()
        {
            return onAppInitialized_impl();
        }
        void onAppTerminated()
        {
            return onAppTerminated_impl();
        }
        virtual void workLoopBody() = 0;
        virtual bool keepRunning() = 0;
    protected:
        virtual void onAppInitialized_impl() {}
        virtual void onAppTerminated_impl() {}
    protected:
        system::path CWDOnStartup;
    };
}

#endif
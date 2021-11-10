// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPENCL_HANDLER_H__
#define __NBL_C_OPENCL_HANDLER_H__

#include "nbl/core/core.h"
#include "nbl/system/system.h"

#include <string>

#ifdef _NBL_COMPILE_WITH_OPENCL_

#include "CL/opencl.h"
#ifdef _NBL_COMPILE_WITH_OPENGL_
    #include "COpenGLExtensionHandler.h"
#endif // _NBL_COMPILE_WITH_OPENGL_

#include "os.h"

namespace nbl
{
namespace ocl
{


static const char* const OpenCLFeatureStrings[] = {
    "cl_khr_gl_sharing",
	"cl_khr_gl_event"
};


class COpenCLHandler
{
        COpenCLHandler() = delete;
    public:
        class SOpenCLPlatformInfo : public core::AllocationOverrideDefault
        {
            public:
                enum EOpenCLFeatures
                {
                    NBL_KHR_GL_SHARING=0,
					NBL_KHR_GL_EVENT,
                    NBL_OpenCL_Feature_Count
                };
                class SOpenCLDeviceInfo : public core::AllocationOverrideDefault
                {
                    public:
                        std::string Name;
                        std::string ReportedExtensions;

                        uint32_t    MaxComputeUnits;
                        size_t      MaxWorkGroupSize;
                        size_t      ProbableUnifiedShaders;
                };

                cl_platform_id id;
                std::string Vendor;
                std::string Name;
                uint32_t Version;
                std::string ReportedExtensions;
                bool FeatureAvailable[NBL_OpenCL_Feature_Count] = { false };

                core::vector<cl_device_id> devices;
                core::vector<SOpenCLDeviceInfo> deviceInformation;
        };

        static inline bool CL_ERROR(cl_int retval)
        {
            return retval!=CL_SUCCESS;
        }

        static bool enumeratePlatformsAndDevices()
        {
            if (alreadyEnumeratedPlatforms)
                return platformInformation.size()!=0ull;
            ocl = OpenCL("OpenCL");
            ocl_ext = OpenCLExtensions(&ocl.pclGetExtensionFunctionAddress);
            alreadyEnumeratedPlatforms = true;

            cl_uint actualPlatformCount = 0u;
            if (CL_ERROR(ocl.pclGetPlatformIDs(~0u,nullptr,&actualPlatformCount)) || actualPlatformCount==0u)
                return false;

            core::vector<cl_platform_id> platforms(actualPlatformCount);
            if (CL_ERROR(ocl.pclGetPlatformIDs(platforms.size(),platforms.data(),nullptr)))
                return false;

            //printf("Found %d OpenCL Platforms\n",actualPlatformCount);

            
            char tmpBuf[128*1024];
            for (auto platform : platforms)
            {
                auto getPlatformInfoString = [platform,&tmpBuf](auto infoEnum, std::string& outStr) -> bool
                {
                    size_t actualSize = 0;
                    if (CL_ERROR(ocl.pclGetPlatformInfo(platform,infoEnum,sizeof(tmpBuf),tmpBuf,&actualSize)))
                        return true;

                    if (actualSize)
                        outStr.assign(tmpBuf,actualSize-1u);
                    else
                        outStr.clear();
                    return false;
                };

                // check profile
                {
                    std::string profile;
                    if (getPlatformInfoString(CL_PLATFORM_PROFILE,profile))
                        continue;

                    if (profile!="FULL_PROFILE")
                        continue;
                }

                // fill out info
                SOpenCLPlatformInfo info;
                info.id = platform;

                if (getPlatformInfoString(CL_PLATFORM_VENDOR,info.Vendor))
                    continue;
                //printf("VENDOR = %s\n",tmpBuf);
                if (getPlatformInfoString(CL_PLATFORM_EXTENSIONS,info.ReportedExtensions))
                    continue;
                //printf("CL_PLATFORM_EXTENSIONS = %s\n",tmpBuf);
                if (getPlatformInfoString(CL_PLATFORM_NAME,info.Name))
                    continue;
                //printf("NAME = %s\n",tmpBuf);

                // TODO: Redo this version extraction at some point
                {
                    size_t actualSize = 0;
                    if (CL_ERROR(ocl.pclGetPlatformInfo(platform,CL_PLATFORM_VERSION,sizeof(tmpBuf),tmpBuf,&actualSize)))
                        continue;
                    //printf("VERSION = %s\n",tmpBuf);
                    size_t j=7;
                    for (; j<actualSize; j++)
                    {
                        if (tmpBuf[j]=='.')
                        {
                            tmpBuf[j] = 0;
                            break;
                        }
                    }
                    size_t minorStart = j+1;
                    info.Version = atoi(tmpBuf+7)*100;
                    for (; j<actualSize; j++)
                    {
                        if (tmpBuf[j]==' ')
                        {
                            tmpBuf[j] = 0;
                            break;
                        }
                    }
                    info.Version += atoi(tmpBuf+minorStart);
                    //printf("Parsed Version: %d\n",info.Version);
                }

                {
                    std::istringstream iss(info.ReportedExtensions);
                    for (std::string extension; iss>>extension; )
                    {
                        // TODO: turn into find_if
                        for (size_t m=0; m<SOpenCLPlatformInfo::NBL_OpenCL_Feature_Count; m++)
                        {
                            if (extension==OpenCLFeatureStrings[m])
                            {
                                info.FeatureAvailable[m] = true;
                                break;
                            }
                        }
                    }
                }
                
                // get devices
                {
                    // get count
                    {
                        cl_uint deviceCount;
                        if (CL_ERROR(ocl.pclGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,~0u,nullptr,&deviceCount)) || deviceCount==0u)
                            continue;
                        info.devices.resize(deviceCount);
                    }
                    if (CL_ERROR(ocl.pclGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,info.devices.size(),info.devices.data(),nullptr)))
                        continue;

                    info.deviceInformation.resize(info.devices.size());
                    auto deviceInfoIt = info.deviceInformation.begin();
                    for (auto device : info.devices)
                    {
                        size_t tmpSize;
                        ocl.pclGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(tmpBuf),tmpBuf,&tmpSize);
                        deviceInfoIt->Name = std::string(tmpBuf,tmpSize);
                        //printf("Device Name: %s\n",tmpBuf);

                        tmpSize = 0;
                        ocl.pclGetDeviceInfo(device,CL_DEVICE_EXTENSIONS,sizeof(tmpBuf),tmpBuf,&tmpSize);
                        deviceInfoIt->ReportedExtensions = std::string(tmpBuf,tmpSize);
                        //printf("Device Extensions: %s\n",tmpBuf);

                        ocl.pclGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,4,&deviceInfoIt->MaxComputeUnits,&tmpSize);
                        ocl.pclGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,8,&deviceInfoIt->MaxWorkGroupSize,&tmpSize);
                        deviceInfoIt->ProbableUnifiedShaders = deviceInfoIt->MaxComputeUnits*deviceInfoIt->MaxWorkGroupSize;
                        //printf("Device %d has probably %d shader cores!\n",j,deviceInfoIt->ProbableUnifiedShaders);

                        deviceInfoIt++;
                    }
                }

                platformInformation.push_back(std::move(info));
            }

            return platformInformation.size()!=0ull;
        }

        static auto getPlatformCount() {return platformInformation.size();}

        static const SOpenCLPlatformInfo& getPlatformInfo(const size_t& ix) {return platformInformation[ix];}

#ifdef _NBL_COMPILE_WITH_OPENGL_
        static bool getCLDeviceFromGLContext(cl_device_id& outDevice, cl_context_properties properties[7],
#if defined(_NBL_WINDOWS_API_)
                                             const HGLRC& context, const HDC& hDC)
#else
                                             const GLXContext& context, const Display* display)
#endif
        {
            if (!alreadyEnumeratedPlatforms)
                return false;

            if (!ocl_ext.pclGetGLContextInfoKHR)
                return false;

            properties[0] = CL_GL_CONTEXT_KHR;
			properties[1] = (cl_context_properties)context;
#if defined(_NBL_WINDOWS_API_)
			properties[2] = CL_WGL_HDC_KHR;
			properties[3] = (cl_context_properties)hDC;
#else
			properties[2] = CL_GLX_DISPLAY_KHR;
			properties[3] = (cl_context_properties)display;
#endif
			properties[4] = properties[5] = properties[6] = 0;

            auto getCurrentDeviceForGL = [&properties,&outDevice]()
            {
                size_t writeOutSize = 0;
                auto retval = ocl_ext.pclGetGLContextInfoKHR(properties,CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,sizeof(cl_device_id),&outDevice,&writeOutSize);
                if (retval!=CL_SUCCESS)
                    return retval;
                return writeOutSize!=sizeof(cl_device_id) ? CL_INVALID_BUFFER_SIZE:CL_SUCCESS;
            };
            auto retval = getCurrentDeviceForGL();
            if (retval==CL_INVALID_PLATFORM)
            {
                properties[4] = CL_CONTEXT_PLATFORM;
                for (auto& platform : platformInformation)
                {
                    properties[5] = (cl_context_properties)platform.id;
                    retval = getCurrentDeviceForGL();
                    if (retval==CL_SUCCESS)
                        break;
                }
            }

            return retval==CL_SUCCESS;
        }
#endif // defined
        
        // function pointers
		using LibLoader = system::DefaultFuncPtrLoader;

		NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(OpenCL, LibLoader
            ,clGetExtensionFunctionAddress
            ,clGetPlatformIDs
            ,clGetPlatformInfo
            ,clGetDeviceIDs
            ,clGetDeviceInfo
            ,clFlush
            ,clFinish
            ,clEnqueueWaitForEvents
            ,clEnqueueMarker
            ,clWaitForEvents
            ,clReleaseMemObject
            ,clEnqueueAcquireGLObjects
            ,clEnqueueReleaseGLObjects
		);
		static OpenCL ocl;


    private:
        static bool alreadyEnumeratedPlatforms;
        static core::vector<SOpenCLPlatformInfo> platformInformation;
        
        class OpenCLExtensionLoader final : system::FuncPtrLoader
        {
                using FUNC_PTR_TYPE = decltype(clGetExtensionFunctionAddress)*;
                FUNC_PTR_TYPE pClGetExtensionFunctionAddress;

	        public:
                OpenCLExtensionLoader() : pClGetExtensionFunctionAddress(nullptr) {}
                OpenCLExtensionLoader(FUNC_PTR_TYPE _pClGetExtensionFunctionAddress) : pClGetExtensionFunctionAddress(_pClGetExtensionFunctionAddress) {}
                OpenCLExtensionLoader(OpenCLExtensionLoader&& other) : OpenCLExtensionLoader()
                {
                    operator=(std::move(other));
                }


		        inline OpenCLExtensionLoader& operator=(OpenCLExtensionLoader&& other)
		        {
			        std::swap(pClGetExtensionFunctionAddress, other.pClGetExtensionFunctionAddress);
			        return *this;
		        }

		        inline bool isLibraryLoaded() override final
		        {
			        return pClGetExtensionFunctionAddress!=nullptr;
		        }

		        inline void* loadFuncPtr(const char* funcname) override final
		        {
			        return pClGetExtensionFunctionAddress(funcname);
		        }
        };
        NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(OpenCLExtensions, OpenCLExtensionLoader
            ,clGetGLContextInfoKHR
        );
        static OpenCLExtensions ocl_ext;
};

}
}

#endif // _NBL_COMPILE_WITH_OPENCL_

#endif

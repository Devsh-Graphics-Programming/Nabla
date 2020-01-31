#ifndef __C_OPENCL_HANDLER_H__
#define __C_OPENCL_HANDLER_H__

#include "IrrCompileConfig.h"
#include <string>

#ifdef _IRR_COMPILE_WITH_OPENCL_

#include "CL/opencl.h"
#ifdef _IRR_COMPILE_WITH_OPENGL_
    #include "COpenGLExtensionHandler.h"
#endif // _IRR_COMPILE_WITH_OPENGL_

#include "os.h"

namespace irr
{
namespace ocl
{


static const char* const OpenCLFeatureStrings[] = {
    "cl_khr_gl_sharing",
	"cl_khr_gl_event"
};


#define IRR_MAX_OCL_PLATFORMS 5
#define IRR_MAX_OCL_DEVICES 8

class COpenCLHandler
{
        COpenCLHandler() = delete;
    public:
        class SOpenCLPlatformInfo : public core::AllocationOverrideDefault
        {
            public:
                enum EOpenCLFeatures
                {
                    IRR_KHR_GL_SHARING=0,
					IRR_KHR_GL_EVENT,
                    IRR_OpenCL_Feature_Count
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

                std::string Vendor;
                std::string Name;
                uint32_t Version;
                std::string ReportedExtensions;
                bool FeatureAvailable[IRR_OpenCL_Feature_Count];

                cl_device_id devices[IRR_MAX_OCL_DEVICES];
                SOpenCLDeviceInfo deviceInformation[IRR_MAX_OCL_DEVICES];
                cl_uint deviceCount;
        };

        static bool enumeratePlatformsAndDevices()
        {
            if (alreadyEnumeratedPlatforms)
                return actualPlatformCount>0;
            alreadyEnumeratedPlatforms = true;

#if defined(_IRR_WINDOWS_API_)
            pClGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
#endif // defined

            cl_int retval = -0x80000000;
            //try
            //{
                retval = clGetPlatformIDs(IRR_MAX_OCL_PLATFORMS,platforms,&actualPlatformCount);
            //}
            //catch ()
            //{
                //
            //}

            if (retval!=CL_SUCCESS)
                return false;

            //printf("Found %d OpenCL Platforms\n",actualPlatformCount);

            char tmpBuf[128*1024];
            cl_uint outCounter = 0;
            for (cl_uint i=0; i<actualPlatformCount; i++)
            {
                size_t actualSize = 0;
                clGetPlatformInfo(platforms[i],CL_PLATFORM_PROFILE,128*1024,tmpBuf,&actualSize);
                if (strcmp(tmpBuf,"FULL_PROFILE")!=0)
                    continue;


                SOpenCLPlatformInfo info;
                //printf("Platform %d\n",outCounter);

                actualSize = 0;
                clGetPlatformInfo(platforms[i],CL_PLATFORM_VENDOR,128*1024,tmpBuf,&actualSize);
                info.Vendor = std::string(tmpBuf,actualSize);
                //printf("VENDOR = %s\n",tmpBuf);

                actualSize = 0;
                clGetPlatformInfo(platforms[i],CL_PLATFORM_EXTENSIONS,128*1024,tmpBuf,&actualSize);
                info.ReportedExtensions = std::string(tmpBuf,actualSize);
                //printf("CL_PLATFORM_EXTENSIONS = %s\n",tmpBuf);

                actualSize = 0;
                clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,128*1024,tmpBuf,&actualSize);
                info.Name = std::string(tmpBuf,actualSize);
                //printf("NAME = %s\n",tmpBuf);

                actualSize = 0;
                clGetPlatformInfo(platforms[i],CL_PLATFORM_VERSION,128*1024,tmpBuf,&actualSize);
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

                for (size_t m=0; m<SOpenCLPlatformInfo::IRR_OpenCL_Feature_Count; m++)
                    info.FeatureAvailable[m] = false;
                actualSize = 0;
                clGetPlatformInfo(platforms[i],CL_PLATFORM_EXTENSIONS,128*1024,tmpBuf,&actualSize);
                //printf("\t\t%s\n====================================================================\n",tmpBuf);
                j=0;
                for (size_t k=0; k<actualSize; k++)
                {
                    if (tmpBuf[k]==' '&&k>j)
                    {
                        std::string extension(tmpBuf+j,k-j);
                        j = k+1;
                        for (size_t m=0; m<SOpenCLPlatformInfo::IRR_OpenCL_Feature_Count; m++)
                        {
                            if (extension==OpenCLFeatureStrings[m])
                            {
                                info.FeatureAvailable[m] = true;
                                break;
                            }
                        }
                    }
                }
                if (j<actualSize)
                {
                    std::string extension(tmpBuf+j,actualSize-j);
                    for (size_t m=0; m<SOpenCLPlatformInfo::IRR_OpenCL_Feature_Count; m++)
                    {
                        if (extension==OpenCLFeatureStrings[m])
                        {
                            info.FeatureAvailable[m] = true;
                            break;
                        }
                    }
                }

                clGetDeviceIDs(platforms[i],CL_DEVICE_TYPE_GPU,IRR_MAX_OCL_DEVICES,info.devices,&info.deviceCount);
                if (info.deviceCount<=0)
                    continue;

                for (j=0; j<info.deviceCount; j++)
                {
                    size_t tmpSize;
                    clGetDeviceInfo(info.devices[j],CL_DEVICE_NAME,128*1024,tmpBuf,&tmpSize);
                    info.deviceInformation[j].Name = std::string(tmpBuf,tmpSize);
                    //printf("Device Name: %s\n",tmpBuf);

                    tmpSize = 0;
                    clGetDeviceInfo(info.devices[j],CL_DEVICE_EXTENSIONS,128*1024,tmpBuf,&tmpSize);
                    info.deviceInformation[j].ReportedExtensions = std::string(tmpBuf,tmpSize);
                    //printf("Device Extensions: %s\n",tmpBuf);

                    clGetDeviceInfo(info.devices[j],CL_DEVICE_MAX_COMPUTE_UNITS,4,&info.deviceInformation[j].MaxComputeUnits,&tmpSize);
                    clGetDeviceInfo(info.devices[j],CL_DEVICE_MAX_WORK_GROUP_SIZE,8,&info.deviceInformation[j].MaxWorkGroupSize,&tmpSize);
                    info.deviceInformation[j].ProbableUnifiedShaders = info.deviceInformation[j].MaxComputeUnits*info.deviceInformation[j].MaxWorkGroupSize;
                    //printf("Device %d has probably %d shader cores!\n",j,info.deviceInformation[j].ProbableUnifiedShaders);
                }


                platformInformation[outCounter] = info;
                if (outCounter!=i)
                    platforms[outCounter] = platforms[i];
                outCounter++;
            }
            actualPlatformCount = outCounter;

            return actualPlatformCount>0;
        }

        static const cl_uint& getPlatformCount() {return actualPlatformCount;}

        static const SOpenCLPlatformInfo& getPlatformInfo(const size_t& ix) {return platformInformation[ix];}

#ifdef _IRR_COMPILE_WITH_OPENGL_
        static bool getCLDeviceFromGLContext(cl_device_id& outDevice, cl_context_properties properties[7],
#if defined(_IRR_WINDOWS_API_)
                                             const HGLRC& context, const HDC& hDC)
#else
                                             const GLXContext& context, const Display* display)
#endif
        {
#if defined(_IRR_WINDOWS_API_)
            if (!pClGetGLContextInfoKHR)
                return false;
#endif // defined

            if (!alreadyEnumeratedPlatforms)
                return false;

            properties[0] = CL_GL_CONTEXT_KHR;
			properties[1] = (cl_context_properties)context;
#if defined(_IRR_WINDOWS_API_)
			properties[2] = CL_WGL_HDC_KHR;
			properties[3] = (cl_context_properties)hDC;
#else
			properties[2] = CL_GLX_DISPLAY_KHR;
			properties[3] = (cl_context_properties)display;
#endif
			properties[4] = properties[5] = properties[6] = 0;

            size_t dummy=0;
#if defined(_IRR_WINDOWS_API_)
            cl_int retval = pClGetGLContextInfoKHR(properties,CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,sizeof(cl_device_id),&outDevice,&dummy);
#else
            cl_int retval = clGetGLContextInfoKHR(properties,CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,sizeof(cl_device_id),&outDevice,&dummy);
#endif
            if (retval==CL_INVALID_PLATFORM)
            {
                for (cl_uint i=0; i<actualPlatformCount&&dummy==0; i++)
                {
                    properties[4] = CL_CONTEXT_PLATFORM;
                    properties[5] = (cl_context_properties) platforms[i];
#if defined(_IRR_WINDOWS_API_)
                    retval = pClGetGLContextInfoKHR(properties,CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,sizeof(cl_device_id),&outDevice,&dummy);
#else
                    retval = clGetGLContextInfoKHR(properties,CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,sizeof(cl_device_id),&outDevice,&dummy);
#endif
                }
            }

            return retval==CL_SUCCESS&&dummy>0;
        }
#endif // defined

    private:
        static bool alreadyEnumeratedPlatforms;
#if defined(_IRR_WINDOWS_API_)
        static clGetGLContextInfoKHR_fn pClGetGLContextInfoKHR;
#endif // defined
        static cl_platform_id platforms[IRR_MAX_OCL_PLATFORMS];
        static cl_uint actualPlatformCount;
        static SOpenCLPlatformInfo platformInformation[IRR_MAX_OCL_PLATFORMS];
};

}
}

#endif // _IRR_COMPILE_WITH_OPENCL_

#endif

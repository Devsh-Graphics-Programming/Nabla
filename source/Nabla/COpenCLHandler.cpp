// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "COpenCLHandler.h"

#ifdef _NBL_COMPILE_WITH_OPENCL_
namespace nbl
{
namespace ocl
{
bool COpenCLHandler::alreadyEnumeratedPlatforms = false;

core::vector<COpenCLHandler::SOpenCLPlatformInfo> COpenCLHandler::platformInformation;

COpenCLHandler::OpenCL COpenCLHandler::ocl;
COpenCLHandler::OpenCLExtensions COpenCLHandler::ocl_ext;

}
}

#endif  // _NBL_COMPILE_WITH_OPENCL_

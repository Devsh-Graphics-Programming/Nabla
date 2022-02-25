// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_OS_OPERATOR_H_INCLUDED__
#define __NBL_C_OS_OPERATOR_H_INCLUDED__

#include "IOSOperator.h"

namespace nbl
{

class CIrrDeviceLinux;

//! The Operating system operator provides operation system specific methods and informations.
class COSOperator : public IOSOperator
{
public:

	// constructor
#if defined(_NBL_COMPILE_WITH_X11_DEVICE_)
	COSOperator(const core::stringc& osversion, CIrrDeviceLinux* device);
#endif
 	COSOperator(const core::stringc& osversion);

	//! returns the current operation system version as string.
	virtual const core::stringc& getOperatingSystemVersion() const;

	//! gets the processor speed in megahertz
	//! \param Mhz:
	//! \return Returns true if successful, false if not
	virtual bool getProcessorSpeedMHz(uint32_t* MHz) const;

	//! gets the total and available system RAM in kB
	//! \param Total: will contain the total system memory
	//! \param Avail: will contain the available memory
	//! \return Returns true if successful, false if not
	virtual bool getSystemMemory(uint32_t* Total, uint32_t* Avail) const;

private:

	core::stringc OperatingSystem;

#if defined(_NBL_COMPILE_WITH_X11_DEVICE_)
    CIrrDeviceLinux * IrrDeviceLinux;
#endif

};

} // end namespace

#endif


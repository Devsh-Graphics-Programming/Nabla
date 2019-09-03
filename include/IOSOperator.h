// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_OS_OPERATOR_H_INCLUDED__
#define __I_OS_OPERATOR_H_INCLUDED__

#include "irr/core/core.h"

namespace irr
{

//! The Operating system operator provides operation system specific methods and informations.
class IOSOperator : public virtual core::IReferenceCounted
{
public:
	//! Get the current operation system version as string.
	virtual const core::stringc& getOperatingSystemVersion() const = 0;

	//! Copies text to the clipboard
	virtual void copyToClipboard(const char* text) const = 0;

	//! Get text from the clipboard
	/** \return Returns 0 if no string is in there. */
	virtual const char* getTextFromClipboard() const = 0;

	//! Get the processor speed in megahertz
	/** \param MHz The integer variable to store the speed in.
	\return True if successful, false if not */
	virtual bool getProcessorSpeedMHz(uint32_t* MHz) const = 0;

	//! Get the total and available system RAM
	/** \param Total: will contain the total system memory
	\param Avail: will contain the available memory
	\return True if successful, false if not */
	virtual bool getSystemMemory(uint32_t* Total, uint32_t* Avail) const = 0;

};

} // end namespace

#endif

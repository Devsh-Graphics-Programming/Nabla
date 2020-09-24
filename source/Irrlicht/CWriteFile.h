// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __C_WRITE_FILE_H_INCLUDED__
#define __C_WRITE_FILE_H_INCLUDED__

#include <stdio.h>
#include "IWriteFile.h"


namespace irr
{

namespace io
{

	/*!
		Class for writing a real file to disk.
	*/
	class CWriteFile : public IWriteFile
	{
        protected:
            virtual ~CWriteFile();

        public:
            CWriteFile(const io::path& fileName, bool append);

            //! Reads an amount of bytes from the file.
            virtual int32_t write(const void* buffer, uint32_t sizeToWrite);

            //! Changes position in file, returns true if successful.
            virtual bool seek(const size_t& finalPos, bool relativeMovement = false);

            //! Returns the current position in the file.
            virtual size_t getPos() const;

            //! Returns name of file.
            virtual const io::path& getFileName() const;

            //! returns if file is open
            bool isOpen() const;

        private:

            //! opens the file
            void openFile(bool append);

            io::path Filename;
            FILE* File;
            size_t FileSize;
	};

} // end namespace io
} // end namespace irr

#endif


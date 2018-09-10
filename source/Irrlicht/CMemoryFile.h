// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_MEMORY_READ_FILE_H_INCLUDED__
#define __C_MEMORY_READ_FILE_H_INCLUDED__

#include "IReadFile.h"
#include "IWriteFile.h"

namespace irr
{

namespace io
{

	/*!
		Class for reading and writing from memory.
	*/
	template <class T>
	class CMemoryFile
	{
	    protected:
            T           Buffer;
            size_t      Len;
            size_t      Pos;

            //! changes position in file, returns true if successful
            inline bool seek(const size_t& finalPos, bool relativeMovement = false)
            {
                if (relativeMovement)
                {
                    if (Pos + finalPos > Len)
                        return false;

                    Pos += finalPos;
                }
                else
                {
                    if (finalPos > Len)
                        return false;

                    Pos = finalPos;
                }

                return true;
            }

            //! returns size of file
            inline size_t getSize() const {return Len;}

            //! returns where in the file we are.
            inline size_t getPos() const {return Pos;}

            //! returns name of file
            inline const io::path& getFileName() const {return Filename;}

            //! Constructor
            CMemoryFile(T memory, const size_t& len, const io::path& fileName, bool deleteMemoryWhenDropped);

            //! Destructor
            virtual ~CMemoryFile();


        private:
            io::path    Filename;
            bool        deleteMemoryWhenDropped;
	};

	class CMemoryReadFile : public IReadFile, public CMemoryFile<const void*>
	{
        public:
            //! Constructor
            CMemoryReadFile(const void* memory, const size_t& len, const io::path& fileName, bool deleteMemoryWhenDropped);

            //! changes position in file, returns true if successful
            virtual bool seek(const size_t& finalPos, bool relativeMovement = false) {return CMemoryFile::seek(finalPos,relativeMovement);}

            //! returns size of file
            virtual size_t getSize() const {return CMemoryFile::getSize();}

            //! returns where in the file we are.
            virtual size_t getPos() const {return CMemoryFile::getPos();}

            //! returns name of file
            virtual const io::path& getFileName() const {return CMemoryFile::getFileName();}

            //! returns how much was read
            virtual int32_t read(void* buffer, uint32_t sizeToRead);
	};

	class CMemoryWriteFile : public IWriteFile, public CMemoryFile<void*>
	{
        public:
            //! Constructor
            CMemoryWriteFile(void* memory, const size_t& len, const io::path& fileName, bool deleteMemoryWhenDropped);

            //! changes position in file, returns true if successful
            virtual bool seek(const size_t& finalPos, bool relativeMovement = false) {return CMemoryFile::seek(finalPos,relativeMovement);}

            //! returns size of file
            virtual size_t getSize() const {return CMemoryFile::getSize();}

            //! returns where in the file we are.
            virtual size_t getPos() const {return CMemoryFile::getPos();}

            //! returns name of file
            virtual const io::path& getFileName() const {return CMemoryFile::getFileName();}

            //! returns how much was written
            virtual int32_t write(const void* buffer, uint32_t sizeToWrite);
	};

} // end namespace io
} // end namespace irr

#endif


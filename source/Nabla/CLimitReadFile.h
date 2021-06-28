// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_C_LIMIT_READ_FILE_H_INCLUDED__
#define __NBL_C_LIMIT_READ_FILE_H_INCLUDED__
#if 0
#include "nbl/system/IFile.h"

namespace nbl
{
	class CUnicodeConverter;

namespace io
{

	/*! this is a read file, which is limited to some boundaries,
		so that it may only start from a certain file position
		and may only read until a certain file position.
		This can be useful, for example for reading uncompressed files
		in an archive (zip, tar).
	!*/
	class CLimitReadFile : public IReadFile
	{
        protected:
            virtual ~CLimitReadFile();

        public:
            CLimitReadFile(IReadFile* alreadyOpenedFile, const size_t& pos, const size_t& areaSize, const io::path& name);

            //! returns how much was read
            virtual int32_t read(void* buffer, uint32_t sizeToRead);

            //! changes position in file, returns true if successful
            //! if relativeMovement==true, the pos is changed relative to current pos,
            //! otherwise from begin of file
            virtual bool seek(const size_t& finalPos, bool relativeMovement = false);

            //! returns size of file
            virtual size_t getSize() const;

            //! returns where in the file we are.
            virtual size_t getPos() const;

            //! returns name of file
            virtual const io::path& getFileName() const;

        private:

            io::path Filename;
            size_t AreaStart;
            size_t AreaEnd;
            size_t Pos;
            IReadFile* File;
	};

} // end namespace io
} // end namespace nbl

#endif

#endif
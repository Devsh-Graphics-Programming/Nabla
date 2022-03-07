// Copyright (C) 2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/system/IFileBase.h"

using namespace nbl::system;

std::string deletePathFromPath(const std::string& filename, int32_t pathcount)
{
	auto str = filename;

	int32_t i = static_cast<int32_t>( str.size() ) - 1;

	while (i >= 0)
	{
		if (str[i] == '/' || str[i] == '\\')
		{
			if (--pathcount <= 0)
				break;
		}
		--i;
	}

	if (i > 0)
	{
		str[i + 1] = 0;
	}
	else
	{
		str.clear();
	}

	return str.c_str();
}

path IFileBase::flattenFilename(const path& p)
{
	if (p.empty()) return p;

	auto str = p.string();
	std::replace(str.begin(), str.end(), '\\', '/');

	std::string dir;
	std::string subdir;

	size_t lastpos = 0u;
	size_t pos = 0u;
	bool lastWasRealDir = false;

	auto process = [&]() -> void
	{
		subdir = str.substr(lastpos, pos - lastpos + 1u);

		if (subdir == "../")
		{
			if (lastWasRealDir)
			{
				dir = deletePathFromPath(dir, 2);
				lastWasRealDir = (dir.size() != 0u);
			}
			else
			{
				dir.append(subdir);
				lastWasRealDir = false;
			}
		}
		else if (subdir == "/")
		{
			dir = "/"; // root
		}
		else if (subdir != "./")
		{
			dir.append(subdir);
			lastWasRealDir = true;
		}

		lastpos = pos + 1u;
	};

	while ((pos = str.find('/', lastpos)) != std::string::npos)
	{
		process();
	}

	if (str.back() != '/')
	{
		pos = str.size();
		process();
	}

	return path(dir);
}

#include "nbl/core/definitions.h"
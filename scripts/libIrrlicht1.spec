# Copyright (c) 2007-2011 oc2pus
# This file and all modifications and additions to the pristine
# package are under the same license as the package itself.
#
# Please submit bugfixes or comments to toni@links2linux.de

# Packmangroup: Libraries
# Packmanpackagename: Irrlicht
# Packman: Toni Graffy

# norootforbuild

Name:			libIrrlicht1
Version:		1.8.0
Release:		0.pm.1
Summary:		The Irrlicht Engine SDK
License:		see readme.txt
Group:			System/Libraries
URL:			http://irrlicht.sourceforge.net/
Source:			irrlicht-%{version}.tar.bz2
BuildRoot:		%{_tmppath}/%{name}-%{version}-build
BuildRequires:	freeglut-devel
BuildRequires:	ImageMagick
BuildRequires:	gcc-c++
%if %suse_version >= 1020
BuildRequires:	Mesa-devel
%else
BuildRequires:	xorg-x11-devel
%endif
BuildRequires:	update-desktop-files

%description
The Irrlicht Engine is an open source high performance realtime 3d engine
written and usable in C++. It is completely cross-platform, using D3D, OpenGL
and its own software renderer, and has all of the state-of-the-art features
which can be found in commercial 3d engines.

We've got a huge active community, and there are lots of projects in
development that use the engine. You can find enhancements for Irrlicht all
over the web, like alternative terrain renderers, portal renderers, exporters,
world layers, tutorials, editors, language bindings for .NET, Java, Perl, Ruby,
Basic, Python, Lua, and so on. And best of all: It's completely free.

%package -n libIrrlicht-devel
Summary:	Development package for the Irrlicht library
Group:		Development/Languages/C and C++
Requires:	libIrrlicht1 = %{version}
# Packmandepends: libIrrlicht1

%description -n libIrrlicht-devel
The Irrlicht Engine is an open source high performance realtime 3d engine
written and usable in C++. It is completely cross-platform, using D3D, OpenGL
and its own software renderer, and has all of the state-of-the-art features
which can be found in commercial 3d engines.

We've got a huge active community, and there are lots of projects in
development that use the engine. You can find enhancements for Irrlicht all
over the web, like alternative terrain renderers, portal renderers, exporters,
world layers, tutorials, editors, language bindings for .NET, Java, Perl, Ruby,
Basic, Python, Lua, and so on. And best of all: It's completely free.

%package -n Irrlicht-examples
Summary:	Demos and examples for the Irrlicht-SDK
Group:		Development/Tools/GUI Builders
Requires:	libIrrlicht1 = %{version}
# Packmandepends: Irrlicht-media libIrrlicht1

%description -n Irrlicht-examples
The Irrlicht Engine is an open source high performance realtime 3d engine
written and usable in C++. It is completely cross-platform, using D3D, OpenGL
and its own software renderer, and has all of the state-of-the-art features
which can be found in commercial 3d engines.

We've got a huge active community, and there are lots of projects in
development that use the engine. You can find enhancements for Irrlicht all
over the web, like alternative terrain renderers, portal renderers, exporters,
world layers, tutorials, editors, language bindings for .NET, Java, Perl, Ruby,
Basic, Python, Lua, and so on. And best of all: It's completely free.

This package contains some demos and examples for the Irrlicht-SDK.

%package -n Irrlicht-doc
Summary:	User documentation for the Irrlicht SDK.
Group:		Documentation/Other

%description -n Irrlicht-doc
User documentation for the Irrlicht SDK.

You need a chm-viewer to read the docs (e.g. kchmviewer).

%package -n Irrlicht-tools
Summary:	Some tools for the Irrlicht-SDK
Group:		Development/Tools/GUI Builders
Requires:	libIrrlicht1 = %{version}
Requires:	Irrlicht-media
# Packmandepends: Irrlicht-media libIrrlicht1

%description -n Irrlicht-tools
Some tools for the Irrlicht-SDK.

GUIEditor, FontTool and Meshviewer.

%package -n Irrlicht-media
Summary:	Some media files for Irrlicht SDK
Group:		Development/Languages/C and C++

%description -n Irrlicht-media
Some media files for Irrlicht tools and demos.

%debug_package

%prep
%setup -q -n irrlicht-%{version}

# only for windows-platform
%__rm -r examples/14.Win32Window

# avoid irrKlang
%__sed -i -e 's|^#define USE_IRRKLANG|//#define USE_IRRKLANG|g' \
	examples/Demo/CDemo.h

# adjust media path in examples
find ./examples -name *.cpp | xargs %__sed -i -e 's|../../media/|%{_datadir}/irrlicht/|g'
# adjust media path in tools
%__sed -i -e 's|../../media/|%{_datadir}/irrlicht/|g' \
	tools/GUIEditor/main.cpp

# pack example-sources
tar cj \
	--exclude *.cbp \
	--exclude *.dev \
	--exclude *.dsp \
	--exclude *.dsw \
	--exclude *.html \
	--exclude *.sln \
	--exclude *.vcproj \
	-f irrlicht-examples-src.tar.bz2 examples/*

%build
# create shared-lib first
pushd source/Irrlicht
%__make sharedlib %{?_smp_mflags}
popd

# create necessary links to avoid linker-error for tools/examples
pushd lib/Linux
ln -s libIrrlicht.so.%{version} libIrrlicht.so.1
ln -s libIrrlicht.so.%{version} libIrrlicht.so
popd

# tools
pushd tools
cd GUIEditor
%__make %{?_smp_mflags}
cd ..
cd IrrFontTool/newFontTool
%__make %{?_smp_mflags}
cd ../..
popd

# examples
pushd examples
sh buildAllExamples.sh
popd

# build static lib
pushd source/Irrlicht
%__make %{?_smp_mflags}
popd

%install
%__install -dm 755 %{buildroot}%{_libdir}
%__install -m 644 lib/Linux/libIrrlicht.a \
	%{buildroot}%{_libdir}
%__install -m 644 lib/Linux/libIrrlicht.so.%{version} \
	%{buildroot}%{_libdir}

pushd %{buildroot}%{_libdir}
ln -s libIrrlicht.so.%{version} libIrrlicht.so.1
ln -s libIrrlicht.so.%{version} libIrrlicht.so
popd

# includes
%__install -dm 755 %{buildroot}%{_includedir}/irrlicht
%__install -m 644 include/*.h \
	%{buildroot}%{_includedir}/irrlicht

# tools
%__install -dm 755 %{buildroot}%{_bindir}
%__install -m 755 tools/GUIEditor/GUIEditor \
	%{buildroot}%{_bindir}/irrlicht-GUIEditor
%__install -m 755 bin/Linux/FontTool \
	%{buildroot}%{_bindir}/irrlicht-FontTool

# examples
%__install -dm 755 %{buildroot}%{_bindir}
ex_list=`ls -1 bin/Linux/??.*`
for i in $ex_list; do
	FE=`echo $i | awk 'BEGIN { FS="." }{ print $2 }'`
	%__install -m 755 $i \
		%{buildroot}%{_bindir}/irrlicht-$FE
done

# examples-docs
pushd examples
%__install -dm 755 %{buildroot}%{_docdir}/Irrlicht-examples
ex_dir=`find . -name tutorial.html`
for i in $ex_dir; do
	dir_name=`dirname $i`
	%__install -dm 755 %{buildroot}%{_docdir}/Irrlicht-examples/$dir_name
	%__install -m 644 $i \
		%{buildroot}%{_docdir}/Irrlicht-examples/$dir_name
done
%__rm -r %{buildroot}%{_docdir}/Irrlicht-examples/09.Meshviewer
popd

# examples sources
%__install -m 644 irrlicht-examples-src.tar.bz2 \
	%{buildroot}%{_docdir}/Irrlicht-examples

# media
%__install -dm 755 %{buildroot}%{_datadir}/irrlicht
%__install -m 755 media/* \
	%{buildroot}%{_datadir}/irrlicht

# icons
%__install -dm 755 %{buildroot}%{_datadir}/pixmaps
convert examples/09.Meshviewer/icon.ico -resize 48x48! \
	irrlicht-Meshviewer.png
convert bin/Win32-gcc/irrlicht.ico -resize 48x48! \
	irrlicht.png
%__install -m 644 irrlicht*.png \
	%{buildroot}%{_datadir}/pixmaps

# menu-entries
%__cat > irrlicht-GUIEditor.desktop << EOF
[Desktop Entry]
Comment=Irrlicht GUI Editor
Name=Irrlicht GUI Editor
GenericName=
Type=Application
Exec=irrlicht-GUIEditor
Icon=irrlicht.png
DocPath=
Terminal=0
EOF
%suse_update_desktop_file -i irrlicht-GUIEditor Development GUIDesigner

%__cat > irrlicht-FontTool.desktop << EOF
[Desktop Entry]
Comment=Irrlicht-FontTool
Name=Irrlicht-FontTool
GenericName=
Type=Application
Exec=irrlicht-IrrFontTool
Icon=irrlicht.png
DocPath=
Terminal=0
EOF
%suse_update_desktop_file -i irrlicht-FontTool Development GUIDesigner

%__cat > irrlicht-Meshviewer.desktop << EOF
[Desktop Entry]
Comment=Irrlicht-Meshviewer
Name=Irrlicht-Meshviewer
GenericName=
Type=Application
Exec=irrlicht-Meshviewer
Icon=irrlicht-Meshviewer.png
DocPath=
Terminal=1
EOF
%suse_update_desktop_file -i irrlicht-Meshviewer Graphics Viewer

%clean
[ -d %{buildroot} -a "%{buildroot}" != "" ] && %__rm -rf %{buildroot}

%files
%defattr(-, root, root)
%doc *.txt
%{_libdir}/lib*.so.*

%files -n libIrrlicht-devel
%defattr(-, root, root)
%{_libdir}/lib*.so
%{_libdir}/lib*.a
%dir %{_includedir}/irrlicht
%{_includedir}/irrlicht/*.h

%files -n Irrlicht-examples
%defattr(-, root, root)
%dir %{_docdir}/Irrlicht-examples
%{_docdir}/Irrlicht-examples/*
%exclude %{_bindir}/irrlicht-GUIEditor
%exclude %{_bindir}/irrlicht-FontTool
%exclude %{_bindir}/irrlicht-Meshviewer
%{_bindir}/irrlicht-*

%files -n Irrlicht-tools
%defattr(-, root, root)
%doc examples/09.Meshviewer/tutorial.html
%{_bindir}/irrlicht-GUIEditor
%{_bindir}/irrlicht-FontTool
%{_bindir}/irrlicht-Meshviewer
%{_datadir}/applications/irrlicht-GUIEditor.desktop
%{_datadir}/applications/irrlicht-FontTool.desktop
%{_datadir}/applications/irrlicht-Meshviewer.desktop
%{_datadir}/pixmaps/irrlicht*.png

%files -n Irrlicht-doc
%defattr(-, root, root)
%doc doc/irrlicht.chm
%doc doc/*.txt

%files -n Irrlicht-media
%defattr(-, root, root)
%dir %{_datadir}/irrlicht
%{_datadir}/irrlicht/*

%changelog
* Wed Jun 20 2007 Toni Graffy <toni@links2linux.de> - 1.3.1-0.pm.1
- update to 1.3.1
* Sat Jun 16 2007 Toni Graffy <toni@links2linux.de> - 1.3-0.pm.1
- initial build 1.3

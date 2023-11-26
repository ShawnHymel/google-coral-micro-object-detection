> **Important!**
> Coral Micro applications only build on Linux. This project has been tested on Ubuntu 20.04.

To build this project, you need to install the Coral Micro SDK, link to it, and then  build. 

If you have not done so already, download the [Coral Micro](https://github.com/google-coral/coralmicro) with the required submodules somewhere on your computer and install the required tools:

```
git clone --recurse-submodules -j8 https://github.com/google-coral/coralmicro
cd coralmicro && bash setup.sh
```

Next, create a symbolic link to wherever you downloaded the *coralmicro* repo:

```
cd <this_directory>
ln -s <path_to>/coralmicro/ coralmicro
```

Run CMake, build, and upload:

```
cmake -B build/ -S .
make -C build/ -j4
python3 coralmicro/scripts/flashtool.py --build_dir build/ --elf_path build/coralmicro-app
```

To avoid re-uploading the model and webpages, use the `--nodata` flag:

```
python3 coralmicro/scripts/flashtool.py --build_dir build/ --elf_path build/coralmicro-app --nodata
```

Use a serial terminal, such as [picocom](), to view debugging information:

picocom /dev/ttyACM0 -b 115200

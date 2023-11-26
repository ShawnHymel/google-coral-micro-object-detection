> **Important!**
> Coral Micro applications only build on Linux. This project has been tested on Ubuntu 20.04.

To build this project, you need to install the Coral Micro SDK, link to it, and then  build. The following steps show you how to do that.

If you have not done so already, download the [Coral Micro](https://github.com/google-coral/coralmicro) with the required submodules somewhere on your computer (e.g. `$HOME`) and install the required tools:

```
cd $HOME
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

To avoid re-uploading the model and webpages, use the `--nodata` flag. To change the IP address (from the default 10.10.10.1), use the `--usb_ip_address` flag. For example:

```
python3 coralmicro/scripts/flashtool.py --build_dir build/ --elf_path build/coralmicro-app --nodata --usb_ip_address 192.168.2.1
```

Use a serial terminal, such as [picocom](https://github.com/npat-efault/picocom), to view debugging information:

```
picocom /dev/ttyACM0 -b 115200
```

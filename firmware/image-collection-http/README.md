# Image Collection via HTTP Server

Firmware for the Google Coral Micro to stream images from the camera via Ethernet over USB. Images are hosted from an HTTP server. Flash the firmware, and with the board plugged into your computer, navigate to [http://10.10.10.1](http://10.10.10.1) on a browser. You should see images being streamed.

Adjust the zoom level to make the image appear larger or smaller. Change the label if you wish to do image classification instead of object detection (i.e. it changes the prefix on the saved JPG). Click *Save Image* to save the image (original resolution, not affected by *zoom* multiplier) to your downloads folder.

## Build and Flash Locally

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

## Hosting the Web Page From Your Computer

By default, the web page (*index.html*) is uploaded to the Coral Micro's internal storage and served whenever a browser requests it from 10.10.10.1. As re-flashing takes a while, you may want to host the web page from your computer instead, as it allows you to view changes very quickly (simply refresh the page!). 

To start, comment out the `const imgUrl = "/camera_stream";` line in *index.html* and uncomment `const imgUrl = "http://10.10.10.1/camera_stream";`, and save the file. We want to use absolute addressing to get the camera elements from the Coral Micro (10.10.10.1) instead of the local server (localhost).

Next, install a simple HTTP server:

```
npm install --global http-server
```

If you run into dependency issues, you may need to install a newer version of NodeJS and npm. See [here for more information](https://stackoverflow.com/questions/55464934/npm-depends-node-gyp-3-6-2-but-it-is-not-going-to-be-installed).

Additionally, you will need to allow for CORS, as the page (hosted on *localhost*) requests elements from a difference location (e.g. *10.10.10.1*). First, install an extension in your browser to allow for CORS (e.g. [Chrome CORS extension](https://chromewebstore.google.com/detail/allow-cors-access-control/lhobafahddgcelffkeicbaginigeejlf)). Next, browser to this directory and run your HTTP server with CORS enabled (it will, by default, look for *index.html* in this directory).

```
http-server -c-1 --cors
```

Finally, browse to [http://localhost:8080](http://localhost:8080) to see the camera stream.
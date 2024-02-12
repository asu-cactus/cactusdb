## Add JSON library to Build Your Project

### Add JSON lib in C++
1.Install the jsoncpp library using the package manager
```bash
sudo apt-get install libjsoncpp-dev
```
2.Add JSON lib in cmakefile

In velox/optimizer/CMakeLists.txt, add
```bash
find_package(jsoncpp REQUIRED)
```

In target_link_libraries, add 
```bash
jsoncpp_lib
```

3.In cpp file, add
```bash
#include <json/json.h>
```




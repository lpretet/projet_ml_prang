This repository contains our code. It is organised in three sub-directories :

* lib : the actual code module.  
* models : serialized versions of trained networks (`.h5`) and their respective training history (`.pickle`). 
This allows us to directly resume from a usable state without having to re-train the networks each time.  
* test : an automatic test suite, which will be released in a future version.
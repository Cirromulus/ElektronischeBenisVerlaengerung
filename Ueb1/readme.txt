On unix you can compile and start the program as following

cmake .
make
cutlery images/*.jpg

-------

On Mac OS X it is possible to use the above but also to use Xcode

make -G Xcode .
open cutlery/cutlery.xcodeproj

now you need to set

Product -> Edit Scheme... -> cutlery -> Run -> Arguments 
to ../images/*.jpg

------

(If anyone can write a guide how to start the program on windows, please email me)
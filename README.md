This is the repository for my physical Qcar code. This is being run on a Quanser QCar 1.


Some important things to know:


IPs may change frequently. Refer to Quanser documentation on how to figure out the IPs of traffic lights. For the computer IP, just run ipconfig and it will be the IPv4 address.


For deploying the code, use winscp to connect to the qcar, and then a sudo python command for running it (additionally refer to Quanser documentation)


Because the position of the traffic lights for v2x have been hard coded, they should resemble the positions in the attached image

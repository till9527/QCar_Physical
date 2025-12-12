The first thing a user will need to do is figure out what the IP address of their computer is. While connected to Quanser_UVS, open a command prompt and type “ipconfig” and take note of what the IPv4 address in the bottom section is


Ensure that this matches the COMPUTER_IP in qcar_native_sender.py, otherwise the qcar will have no idea where to connect.

The next thing to do is make sure to put that updated qcar_native_sender.py on the QCar itself. I recommend putting it under Documents/Quanser/src/SDCS/skill_activities/04-vehicle_control using WinSCP

Then on your computer, you’ll want to run “computer_receiver_opengl.py”. This will essentially treat your computer as a “server” that can accept multiple connections from qcars. It also handles all the YOLO processing. You’ll want to run the respective 
“computer_receiver_opengl.py” file that corresponds to what simulation you’re trying to run

Next, you want to figure out what the IPs of the traffic lights are. Ensure that the power bank is connected to the traffic lights so that they will be turned on. Then you can either follow the morse code logic described by Quanser in their traffic light manual, or open a 

web browser and type http://192.168.2.xx:5000/immediate/green where xx is between 10-25, until you see a page that says “Green on”.

Once you’ve figured out what the traffic light IPs are, make sure that in Multi_V2X.py you set TRAFFIC_LIGHT_IPS to those. Then run http://192.168.2.xx:5000/timed/5/2/10 for both of them. This will set their timed sequences

Additionally, make sure that the traffic lights are positioned like they are on https://github.com/till9527/QCar_Physical/blob/master/traffic_lights.jpg since I had to hardcode the geofences

Lastly, open up PUTTY to connect to the QCar and cd into Documents/Quanser/src/SDCS/skill_activities/04-vehicle_control or wherever you put qcar_native_sender.py/qcar_native_sender_v2x.py/Multi_V2X.py and then run the following command:  sudo PYTHONPATH=$PYTHONPATH python3 {qcar_native_sender.py/qcar_native_sender_v2x.py/Multi_V2X.py}

Note: You may need to rerun XLaunch if the X11 server isn’t working. 

Note: sometimes you may get an error saying that the video format is not supported. The current workaround for this is to just restart the QCar and PUTTY. It’s a frustrating error that seems to happen after the QCar has been running long enough

Running the QCar physically (some of these instructions are specific to Kettering University's setup):

Connecting to physical environment (using desktop in lab):

MAKE SURE YOU ARE ON QUANSER WIFI. TURN THE QCAR ON AND PLACE IT WHERE YOUR FIRST NODE IS. MAKE SURE TO CHECK qcar_native_sender.py/qcar_native_sender_v2x.py/Multi_V2X.py FOR THE NODE SEQUENCE. REFER TO https://github.com/till9527/QCar_Virtual/blob/main/SDCS_RoadMap_RightHandTraffic.png FOR FIGURING OUT WHERE THE NODES ARE

1. After updating the code for qcar_native_sender.py/qcar_native_sender_v2x.py/Multi_V2X.py, run WinSCP. Username and password are both nvidia, and the host name will be the IP address on the QCar

2. Drag and drop the new code for qcar_native_sender.py/qcar_native_sender_v2x.py/Multi_V2X.py to Documents/Quanser/src/SDCS/skill_activities/04-vehicle_control. It will prompt asking to overwrite, which you say yes

3. Afterwards, make sure to close out of WinSCP. Run XLabs and follow the default prompts. 

4. Then open putty. Under SSH -> X11, click the checkbox on enable X11 forwarding, and then set the X display location to localhost:0.0, and click "open". Then under session, enter the same IP address you did in WinSCP for host name, then click "open"

5. Login using nvidia as both username and password. Then cd into Documents/Quanser/src/SDCS/skill_activities/04-vehicle_control. Then run this command:
sudo PYTHONPATH=$PYTHONPATH python3 {qcar_native_sender.py/qcar_native_sender_v2x.py/Multi_V2X.py}, which will prompt you to calibrate. Say yes for the first time

6. To stop the simulation, press ctrl + c. To rerun it, use the command from step 4, and if you haven't shut the QCar off or disconnected from putty, you can just place the QCar where it's intended starting point is and skip the calibration

Whatever you have the node sequence set to, you will have to place the QCar in the respective starting location

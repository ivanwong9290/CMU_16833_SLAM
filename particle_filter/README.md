Refer to robotlog1.gif for a demonstration of the project <br/>
<br/>
At a macro, the particle filter runs as follows, which can be broken down into 4 steps: <br/>
![image](https://user-images.githubusercontent.com/71652695/129286679-fb046b8f-9ac4-404e-a07c-c63acec66393.png) <br/>
<br/>
## Part 1: Initialization <br/>
&nbsp; - 500 particles <br/>
&nbsp; - 30 equally separated rays centered about particle orientation (These rays are used to scan the environment to detect interference) <br/>
&nbsp; - Occupancy threshold = 0.35 (Any value greater than that is considered an obstacle for the robot) <br/>
## Part 2: Motion Model <br/>
The following is the odometry model used for each particle, the α values represent motion noises <br/>
![image](https://user-images.githubusercontent.com/71652695/129286254-0979caac-542c-4edd-a430-ffd36e2f30f1.png) <br/>
## Part 3: Ray Casting Sensor Model <br/>
The following is the sensor model used for each particle, the z values can be conveyed as weight for different ray casting noises  <br/>
![image](https://user-images.githubusercontent.com/71652695/129286435-64de5176-91d9-44f9-ab3f-896b02560426.png) <br/>
## Part 4: Resampling <br/>
Finally, the following algorithm resamples particles based on the weights assigned in the previous step <br/>
![image](https://user-images.githubusercontent.com/71652695/129286802-a12a393d-2ade-420b-b8de-8a341ac02660.png) <br/>

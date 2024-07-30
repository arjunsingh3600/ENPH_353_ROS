
Repository for the ENPH 353 project competition. Given a parking lot environment with car agent, the competition task was to navigate the parking alot as fast as possibele while accurately scanning all the parking plates it encounters. 

The plate detector ROS package contains the node used to read, process and parse the plates as well as the text indicating the parking location. To parse images, a CNN was trained based on simulated images as well as real time training data. 

The package also contains an implementation of a NN driver based on the following [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Note this impementation was not used in the final competition. A simpler PID control version with obstactle detection was opted for instead. While the NNdriver was an interesting proof of concept, it did not out perform the classic PID control which was easier to integrate and more interpreatable.

# Picar-autopilot
---

## Video Links
[Short Video Page](https://youtube.com/shorts/faxs8olFlaE?feature=share)


## Overview
The Picar project is a project to develop artificial intelligence models for self-driving cars. This project analyzes a given dataset, uses it to predict whether a vehicle is driving and its direction, and verifies the model's performance through real-world driving tests and heatmaps. This document includes an overall description of the project, how it is executed, and the test results.

Autopilot and PiCar are provided by the UoN-MLiS team. Template code for the [autopilot](https://github.com/adammoss/autopilot) self-driving software on the PiCar.


## Project Plan

The central concept behind the auto driving technology is to use trained object detection models to make principled decisions about what manoeuvres a car should make. In this project task is to collect a data set of images and use them to train a machine learning algorithm to autonomously navigate around a realistic test circuit, and make the appropriate manoeuvres where necessary.

The first part of the project requires maximum performance from Kaggle challenge. The challenge consists of 13.8k images, along with the target car response (speed and steering angle). It allows you to automate the process of model submission, and obtain an indication of performance (using a small set of test data) before evaluate them on the final, unseen data. 

The second part of the project calls for actual driving on three tracks with various variables. We evaluate driving performance in a real environment by applying the learned model to a physical vehicle. Driving tests are conducted by collecting the vehicle's sensor data in real time, and controlling the vehicle to follow the driving path predicted by the model.


<p align="center">
<img src="/assets/track.jpg" width="70%" height="60%" />
</p>

The process requires a variety of hardware equipment, such as raspberry pie and cameras, and the accuracy and stability of the model can be checked in real-world driving environments.



## Methods
##### Network Architecture
<p align="center">
<img src="/assets/network.png" width="90%" height="60%" />
</p>

##### Heat map output

###### How to generate a heat map
We use a heatmap.ipynb script to visualize the results of the learned model, which generates a heatmap relative to the actual location based on the location of the obstacle when driving.

<p align="center">
  <img src="/assets/heatmap1.png" alt="Image 1" width="45%" />
  <img src="/assets/heatmap2.png" alt="Image 2" width="45%" />
</p>
<p align="center">
  <em>The left-hand side figure illustrates before applying the Attention model, and the right-hand side figure represents after applying the Attention model.</em>
</p>


---
Installation and Testing code can find from the template code for the [autopilot](https://github.com/adammoss/autopilot) self-driving software on the PiCar.

---

*For more details, please refer to the MLiS report. The report is available only to a limited number of people who requested to check it as a reference*

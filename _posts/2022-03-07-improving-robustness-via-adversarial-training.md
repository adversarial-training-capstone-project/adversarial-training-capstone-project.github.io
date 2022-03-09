---
layout: post
title: Improving on Single Source Robustness in Self-Driving Cars
subtitle: Applications of Adversarial Training 
cover-img: /assets/img/cars_on_a_road.webp
thumbnail-img: /assets/img/tesla.jpeg
# share-img: /assets/img/path.jpg
# tags: [books, test]
---

# Introduction

Autonomous vehicles have been populating streets throughout the world in recent years and growing in popularity mainly through companies such as Tesla, Cruise, Aurora, and many more. These vehicles operate through utilizing multiple sensors that are able to collect real-time data about the surrounding environment and make a proper judgement on what the vehicle should do. For instance, these vehicles use LIDAR (Light Detection and Ranging) sensors and RGB cameras, both of which provide different information about distance and detection. In different conditions, some sensors work better than others - for example, LIDAR sensors are more effective at nighttime than compared to RGB cameras. Thus, by utilizing multiple inputs within the deep learning system to make driving decisions, otherwise known as deep fusion modeling, the vehicles are relying on these models for tasks such as object detection. It is important that these models are significantly accurate in detecting objects such as other vehicles and pedestrians, however, it is essential that these models are robust against corruption of the input sources. If the autonomous vehicle is in an environment where one sensor is weighted more heavily than the others and it faces some form of corruption, we need to ensure that the model is robust enough to make the proper judgement. If this issue is not addressed, this could lead to severe consequences on the road and thus making these vehicles unsafe to be on the roads. We explore different approaches in improving the robustness of the deep fusion models in autonomous driving and experiment with a novel approach through adversarial training.

# Previous Works

There has been research on improving the single source robustness in deep fusion models, specifically done by Taewan Kim and Joydeep Ghosh. In their research paper, the authors explored their novel training algorithms, in which they added perturbations (noise) to the data from the input sources and trained the model under these conditions. They finetuned the model on clean data and then introduced noisy data either through downsampling or generating random Gaussian noise. 

[Add image of noise from Gaussian]

The experimental setup consisted of developing and testing the model on clean data, data where one input source is corrupted (single source noise or SSN), and data where all input sources are corrupted (all source noise or ASN). For their metrics, they used the minumum average precision (minAP) score, which is the lowest AP score across all the input sources and the maximum difference between the AP scores (maxDiffAP), which finds the maximum difference among the scores as a measure of balanced robustness.

For their results, the authors observed that the model trained with SSN performed the best on Gaussian noisy data and was still comparable in its performance on clean data in comparison to the model trained on clean data and models trained with ASN. This showcased a new method in training these models so that it is robust against single source corruption yet can perform as well as a model trained normally on clean data. 

[Add image of results]

# Background

The experiment that the authors conducted consist of two components that we will share for our experiment as well: the KITTI dataset and the AVOD model. 

The KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute) dataset is a popular benchmark dataset for autonomous driving research. This contains six hours of traffic scenarios, which were recorded using various modalities such as color stereo cameras and a Velodyne 3D laser scanner. The scenarios recorded range between different locations such as rural streets, freeways, and city roads. For our purposes of the experiment, we utilize the benchmarks for object detection tasks, which provides accurate bounding boxes in both 3D and BEV (Bird’s Eye View) for object types such as cars, cyclists, and pedestrians. 

[Add image of data]

We use the AVOD (Aggregate View Object Detection) model, which is a neural network that uses LIDAR point clouds and RGB images to deliver real-time object detection in the form of bounding boxes and labels for objects in an image. It is structured by two subnetworks, a region proposal network (RPN) and a second stage detector network, the former generating 3D object proposals for multiple object classes and the latter creating accurate oriented 3D bounding boxes and category classifications for predictions. The AVOD model has state of the art results on the KITTI object detection benchmark, making it a great candidate for our baseline model. Using the same setup as Taewan Kim and Joydeep Ghosh, we will train the model solely on the car class for the object detection tasks and use the feature pyramid network for feature extraction. Below highlights the structure of the AVOD model, in which the blue components represent the feature extractors, pink components represent the region proposal network (RPN), green components represent the second stage detector network, and the yellow components representing the adversarial examples generation process.

[Add image of AVOD model]

# Our Proposal

# Methodology

# Experiment

# Conclusion

Under what circumstances should we step off a path? When is it essential that we finish what we start? If I bought a bag of peanuts and had an allergic reaction, no one would fault me if I threw it out. If I ended a relationship with a woman who hit me, no one would say that I had a commitment problem. But if I walk away from a seemingly secure route because my soul has other ideas, I am a flake?

The truth is that no one else can definitively know the path we are here to walk. It’s tempting to listen—many of us long for the omnipotent other—but unless they are genuine psychic intuitives, they can’t know. All others can know is their own truth, and if they’ve actually done the work to excavate it, they will have the good sense to know that they cannot genuinely know anyone else’s. Only soul knows the path it is here to walk. Since you are the only one living in your temple, only you can know its scriptures and interpretive structure.

At the heart of the struggle are two very different ideas of success—survival-driven and soul-driven. For survivalists, success is security, pragmatism, power over others. Success is the absence of material suffering, the nourishing of the soul be damned. It is an odd and ironic thing that most of the material power in our world often resides in the hands of younger souls. Still working in the egoic and material realms, they love the sensations of power and focus most of their energy on accumulation. Older souls tend not to be as materially driven. They have already played the worldly game in previous lives and they search for more subtle shades of meaning in this one—authentication rather than accumulation. They are often ignored by the culture at large, although they really are the truest warriors.

A soulful notion of success rests on the actualization of our innate image. Success is simply the completion of a soul step, however unsightly it may be. We have finished what we started when the lesson is learned. What a fear-based culture calls a wonderful opportunity may be fruitless and misguided for the soul. Staying in a passionless relationship may satisfy our need for comfort, but it may stifle the soul. Becoming a famous lawyer is only worthwhile if the soul demands it. It is an essential failure if you are called to be a monastic this time around. If you need to explore and abandon ten careers in order to stretch your soul toward its innate image, then so be it. Flake it till you make it.
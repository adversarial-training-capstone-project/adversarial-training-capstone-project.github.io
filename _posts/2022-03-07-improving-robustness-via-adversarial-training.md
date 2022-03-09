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

Autonomous vehicles have been populating streets throughout the world in recent years and growing in popularity mainly through companies such as Tesla, Cruise, Aurora, and many more. These vehicles operate through utilizing multiple sensors that are able to collect real-time data about the surrounding environment and make a proper judgement on what the vehicle should do. For instance, these vehicles use LIDAR (Light Detection and Ranging) sensors and RGB cameras, both of which provide different information about distance and detection. In different conditions, some sensors work better than others - for example, LIDAR sensors are more effective at nighttime than compared to RGB cameras. Thus, by utilizing multiple inputs within the deep learning system to make driving decisions, otherwise known as deep fusion modeling, the vehicles are relying on these models for tasks such as object detection. 

It is important that these models are significantly accurate in detecting objects such as other vehicles and pedestrians, however, **it is essential that these models are robust against corruption of the input sources**. If the autonomous vehicle is in an environment where one sensor is weighted more heavily than the others and it faces some form of corruption, we need to ensure that the model is robust enough to make the proper judgement. If this issue is not addressed, this could lead to severe consequences on the road and thus making these vehicles unsafe to be on the roads.

![Poster SSN](/assets/img/taewan_poster_ssn.png){: .mx-auto.d-block :}

# Previous Works

There has been research on improving the single source robustness in deep fusion models, specifically done by Taewan Kim and Joydeep Ghosh. In their research paper, the authors explored their novel training algorithms, in which they added perturbations (noise) to the data from the input sources and trained the model under these conditions. They finetuned the model on clean data and then introduced noisy data either through downsampling or generating random Gaussian noise. 

![Noise](/assets/img/noise.png){: .mx-auto.d-block :}

The experimental setup consisted of developing and testing the model on clean data, data where one input source is corrupted (single source noise or SSN), and data where all input sources are corrupted (all source noise or ASN). For their metrics, they used the minumum average precision (minAP) score, which is the lowest AP score across all the input sources and the maximum difference between the AP scores (maxDiffAP), which finds the maximum difference among the scores as a measure of balanced robustness. **The authors observed that the model trained with SSN performed the best on Gaussian noisy data and was still comparable in its performance on clean data in comparison to the model trained on clean data and models trained with ASN.** This showcased a new method in training these models so that it is robust against single source corruption yet can perform as well as a model trained normally on clean data. 

# Background

The experiment that the authors conducted consist of two components that we will share for our experiment as well: the KITTI dataset and the AVOD model. 

The KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute) dataset is a popular benchmark dataset for autonomous driving research. This contains six hours of traffic scenarios, which were recorded using various modalities such as color stereo cameras and a Velodyne 3D laser scanner. The scenarios recorded range between different locations such as rural streets, freeways, and city roads. For our purposes of the experiment, we utilize the benchmarks for object detection tasks, which provides accurate bounding boxes in both 3D and BEV (Bird’s Eye View) for object types such as cars, cyclists, and pedestrians. 

![Example Data](/assets/img/example_data.png){: .mx-auto.d-block :}

We use the AVOD (Aggregate View Object Detection) model, which is a neural network that uses LIDAR point clouds and RGB images to deliver real-time object detection in the form of bounding boxes and labels for objects in an image. It is structured by two subnetworks, a region proposal network (RPN) and a second stage detector network, the former generating 3D object proposals for multiple object classes and the latter creating accurate oriented 3D bounding boxes and category classifications for predictions. The AVOD model has state of the art results on the KITTI object detection benchmark, making it a great candidate for our baseline model. Using the same setup as Taewan Kim and Joydeep Ghosh, we will train the model solely on the car class for the object detection tasks and use the feature pyramid network for feature extraction. Below highlights the structure of the AVOD model, in which the blue components represent the feature extractors, pink components represent the region proposal network (RPN), green components represent the second stage detector network, and the yellow components representing the adversarial examples generation process.

![AVOD Architecture](/assets/img/avod_arch.png){: .mx-auto.d-block :}

# Our Proposal

While the previous works on this subject have proven to have substantial results, we explored another approach in improving the robustness of these models through adversarial training. But what is adversarial training?

## Methodology

For us to understand this approach, we need to define what an adversarial example is. An adversarial example is an intentionally mislabeled image by altering the pixel values so that the changes made to the image are indistinguishable to the human eye, but recognizable by a model. Although it is obvious to us that this is an image of a pig, the model interprets this as an airliner given the small perturbations added to the pixel values of this image. 

![Pig](/assets/img/pig.png){: .mx-auto.d-block :}

This poses a particular threat to safety-critical applications of ML, notably self-driving cars, as the noise can be intentionally optimized on the inputted data to control the decisions made by the model. For instance, an input source can face intentional corruption to where certain objects on the road are no longer detected such as pedestrians. 

![Pedestrian Bounding Box](/assets/img/ped_box.png){: .mx-auto.d-block :}

There are different approaches for adversarial training, but for our experiment, we will use Fast Gradient Sign Method (FGSM) as our strategy of finding the best perturbations. Additionally, given the structure of the AVOD model, we will be adding the perturbations to the feature maps, which are the transformations of the input data so that it can be passed through the deep fusion layers. 

![AVOD Arch Addition](/assets/img/avod_arch_add.png){: .mx-auto.d-block :}

Usually, we focus on minimizing the loss to optimize the parameters of the model given the input and the correct labels. However, for creating adversarial examples, our goal is to maximize the loss to optimize the noise added to the input enough so that it is mislabeled. Through FGSM, we select an epsilon value that represents the maximum magnitude of delta and update the values of the input through adding the epsilon in the direction of the gradient descent for the parameters. Thus, for all the values in the input, we are either adding or subtracting a perturbation that we defined as epsilon but small enough to where the changes to the image are unrecognizable. 

![FGSM Summary](/assets/img/fgsm_summary.png){: .mx-auto.d-block :}

Overall, we are solving for the following optimization problem where we optimize the input to find the maximum loss given a perturbation that is constrained to our epsilon value, but then optimizing the model parameters to minimize the overall loss in order to build its robustness.

![Big Picture](/assets/img/overall_proc.png){: .mx-auto.d-block :}

# Experiment

For our experiment, we set it up so that we are evaluating three models: a model trained normally on clean data, a model fine-tuned with adversarial data, and a model fine-tuned with single source noise (SSN) data. We then evaluate their performances on clean data, adversarial data, and SSN data and compare their minAP scores and regular AP scores as done before in the previous works.   

![Experimental Setup](/assets/img/experimental_setup.png){: .mx-auto.d-block :}

We were able to observe the following results from our experiment:

[Add image of the results]

The models that were trained using SSN and clean data performed poorly on adversarial data, resulting in extremely low minAP scores. This indicates that the attacks on the input sources were succesful. However, the model trained using adversarial examples had a signficiantly better performance on corrupted adversarial data and had comparable results on clean and SSN data to the other models. Thus, we can observe that training with adverserial examples proved to be successful in improving the robustness of the deep fusion model and did not decrease significantly in performance for the other measures of data. 

# Conclusion

We explored the importance of developing robustness in deep fusion modeling as seen in the area of autonomous vehicles. While there has been much research done in making these models as accurate as they can be, it is imperative that we focus on ensuring that the model can still make proper and reasonable inferences when faced with unforeseen circumstances. As these vehicles are driving around in the streets, it is a huge responsibility to protect our lives and avoid any severe consequences. Through adversarial training, we were able to demonstrate that this is a viable approach in improving the robustness against single source corruption in addition to previous works. Our approach is both robust against randomly generate noise as well as adversarial examples, which is important to consider if there is any intentional corruption from a third party. We hope our work inspires further exploration of using adversarial training in developing robustness.

# Weather-Detection-using-CLiP


CLIP (Contrastive Language-Image Pre-Training) is a foundational model trained on a large amount of (image, text) pairs that have been directly scraped from the internet. The shear size of the model and the dataset that it has been trained on gives it some amazing properties that allow it to perform tasks like "zero-shot" prediction on a variety of tasks. It has similar zero-shot capabilities like GPT-3 where it can find the most relevant text snippet given an image through a training-procedure called "contrastive-learning". It has been found that CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples in the dataset, overcoming several major challenges in computer vision.

### INSTALLATION

Install the following packages to use the CliP model
```
pip install ftfy regex tqdm
```
After that execute the following command as it will copy the CLiP repository into a CLiP API
```
pip install git+https://github.com/openai/CLIP.git
```
Alternatively, we can just execute the following statement to install everything at once.
```
pip3 install -r requirements.txt
```
Once the installation is done, We must put additional images in folder **images**. Around 5 images are already provided in the folder but you may add additional dashcam images to test the model with.

After all this is done, there is just one more step left. Go to the code **linear_dataset_classifier.py** and **list_of_dataset_classifiers.py** and change the number 67 in line number 19 with however many images are there in folder **images**.

Congartulations! you have finished modifying the code, now simply run the python codes as below

```
python linear_dataset_classifier.py
```
and
```
python list_of_dataset_classifiers.py
```

# Technologies for processing medical images 

The field of Computer Vision (CV) represents the technologies used for processing medical images. Computer Vision is "a broad term for the processing of image data" (Fisher 2004, p. 47). It distinguishes from terms like mage processing or pattern recognition with a focus on the human perception of what the understanding of an image means  (Fisher 2004, p. 47). 
The described application requires *segmentation* and *recognition* algorithms from this field of computer science. This chapter introduces the reader in the basic concepts of these two algorithm areas and presents the current state of research. The focus lies only on algorithms the can be used for medical images.

<!-- The foundation for this chapter is build on an extensive literature review, using the databases of arvix and google scholar. The literature review does focus on CV algorithms and known medical applications. -->

## Machine Learning for medical image analysis 

With Machine Learning (ML) computer vision did become even more powerful (Freeman 2008), to that a variety of problems can be solved using ML algorithms (Szeliski 2010, p. 5). This section provides a short introduction into segmentation and recognition algorithms and contains the  basic terminology of the relevant CV algorithms. Thereby the relevance of object recognition in contrast to simple classification will be highlighted. <!-- The presented knowledge is a compilation from standard computer vision literature, for more detailed explainations please consult the books of .. -->

### Terminology of Computer Vision tasks

*Segmentation* denotes the process of dividing a data set into parts based on defined rules (Fisher 2004, p. 229). In the context of computer vision the image segmentation implies the grouping of the image pixels into reasonable structures like edges or regions (Fisher 2004, p. 117). <!--, the regions are also called segments (Concise CV, p. 167). Concise CV sees only regions as segmentation-->

*Object recognition* is a closely related task with a definition focus on identifying one or more known objects in the image (Fisher 2004, p. 172). If this process includes also the objects scene position, it is an object detection process. An generic *object detection system* is based on three steps. Initially the system performs a *localization* of *object candidates*. These candidates are commonly bounding boxes (<!-- source paper with link for prove that common -->). Bounding boxes are a special case of a region of interest (RoI) being the smallest rectangular frame around an object  (Fisher 2004, p. 31). In the second stage the boxes are mapped by an classification algorithm into sets of detected objects or rejected candidates. Which sets are finally used for evaluating the algorithms performance. (Concise CV, p. 375)
Using classification algorithms directly without a prior splitting in sub regions makes the detection of multiple objects within one image impossible. Using regions of interest has also two other advantages for the regarded use case. RoI do reduce the amount of computation needed and therefore improve the speed on larger images and these regions minimize the distractions for the classification algorithm (Fisher 2004, p. 215). 

As further explained in the previous chapter the image source for this use case have been high resolution scans and photography's with multiple relevant objects which are technical reasons to use object detection. From a physician point of view object detection is helpful to understand to focus reduce object shaking in a time series. Even tough segmentation can be applied afterwards, it might distract the analyst in this specific scenario. Therefore this paper uses an object detection system and will not focus on segmentation algorithms. <!--source + nicht gut durchgebründet, da use case unklar-->

In general object recognition divides into two broader categories: instance recognition and class recognition. 
Instance recognition or instance-level recognition  is are technics to re-recognizing objects by abstracting different scaling, viewpoints, lightning and other aspects of the picture (Szeliski 2011, p.685). Commonly these algorithms classify using viewpoint-invariant 2D features (Vedaldi 2018, Szeliski 2011 p.685f).  This works pretty good to recognize popular buildings or known faces. <!--but for faces = limitation source Google Googles key point extraction-->
Class recognition, also known as category-level or generic object recognition aims to solve more abstract problem of matching any instance of a generic class. This task is harder to solve. For Example having a picture of a pug[^1] and a terrier even both are kinds of dogs simple feature extraction form the image might not be optimal.
The goal of our client is a stable and general algorithm for it's datasets. 
Recent scientific research does also focus on generic object recognition, since latest publications show that deep learning generic object detection does outperform, classical machine learning if if uses as instance recognition. Therefore the paper will focus mostly on the category object detection algorithms. 

Besides the scientific claims Computer Vision tasks have a big engineering part as well. By this meaning multiple algorithms and approaches should be considered. Statistical approaches have been proven to be a useful way to solve Computer Vision problems. Especially on small datasets there are better to apply<!--source ML vs deep learning-->. Therefore statistical approaches and classic machine learning algorithms will presented ahead, before diving into complex class recognition algorithms, which will require deep learning. <!-- ML = only instance recognition--> <!-- compare ML and DL also to understand heading--> <!-- or more likely supervised vs. unsupervised -->

### Object detection with manual feature engineering

Machine Learning in the context of medical image analysis has been quite common in past years, until it was gradually replaced by deep learning <!--source-->. Many common medical applications rely on machine learning algorithms.<!-- name some example + they have to be common + paper of methods --> This rather helpful since most medical applications provide only a small set of images.<!--medical image database, for validation --> In these use cases applying deep learning, if even possible, can only be achieved various strategies like using pretrained weights for non medical images or utilizing data augmentation (Shen 2017, p. 223). To baseline the score of the deep learning models well proven classical machine learning models should be used.

The field of Machine Learning contains many object detectors with handcrafted features. This section will review solely the supervised object detectors, that have been proven in a medial application. <!-- handcrafted vs supervised-->
These algorithms divide in two kinds of models. The first are follow the idea of visual words (Source Book) behaving like bag of words featured systems. The latter use the an approach called sliding windows (also called windowing), where the image is divided small sub images (patches) for which the detection of classes will be performed (def book). <!-- define patches--> The part-based models, which do recognize objects by their geometric relations are not discussed in this thesis since they have not been applied for medical images in the recent years. (algorithms and applications)

category recognition is the bag of words (also known as
bag of features or bag of keypoints) approach <!-- image from book ? -->

The biggest difference from
instance recognition the absence of a geometric verification stage (Section

bag of words = Models of visual words

different Object detection classifiers

sliding window approach

part based models for object recognition

Dictionary learning and sparse representation  

https://www.nature.com/articles/s41598-017-15720-y

support vector machines

hog

har wavelets

adaboost

radom forest

+++ concrete papers of medical to show that with results

https://arxiv.org/pdf/1706.01513.pdf

Spatial Distance

Weighted Fuzzy C-Means (SDWFCM) of Guo et al. [2], Dictionary-based Model

(DICT) of Dahl and Larser [3] and Convolutional Neural Network (CNN) https://arxiv.org/pdf/1610.09493.pdf

<!-- WTF do with recognition with segmentation even since segmentation is a common technique you can create bounding boxes from segmentation -->

<!-- Instance aware segmentation is relevant -->

### Neural Nets for object detection

> Anschließend werden Bildanalyseverfahren mit neuronalen Netzwerken vorgestellt. Dabei werden grundlegende Techniken wie Convolutional Neural Networks vorgestellt, aber auch spezielle Ergebnisse von Bildanalysewettbewerben wie dem ILSVRC.

Review paper last algorithms

Percentatage DEEP LEARNING FOR HEALTH INFORMATICS

Directed Belief Networks: https://arxiv.org/pdf/1603.06624.pdf

### Performance evaluation of Computer Vision algorithms



While trying to decide for a object detection algorithm, the question - How to measure its perfomance? - comes in mind. Computer vision algorithm can be measured using three major categories: 

> Successful solution of task. Any practitioner gives this a top priority.
> But also the designer of an algorithm should define precisely for
> which task it is suitable and what the limits are.
> Accuracy. This includes an analysis of the statistical and systematic
> errors under carefully defined conditions (such as given signal-tonoise
> ratio (SNR), etc.).
> Speed. Again this is an important criterion for the applicability of an
> algorithm.

- Book 3 things
- Accurary recall precision, ... -> sehr kurz fassen

## Commonly used technologies in medical software

> **Bestehende Verfahren zur Analyse medizinischer Bilder**
>
> - cell Profiler algorithms (instance recognition)
>
> - 3D Slicer segmentation algorithms
>
> Auch aktuelle medizinische Verfahren sollen in dieser Arbeit untersucht werden und Umständen kann aktuelle Forschung aus diesem Bereich helfen die Probleme zu lösen.

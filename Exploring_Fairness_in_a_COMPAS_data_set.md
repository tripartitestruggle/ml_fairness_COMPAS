Exploring_Fairness_in_a_COMPAS_data_set

# Exploring Fairness in a COMPAS dataset    

## Introduction

The recent events of the Black Lives Matter movement bring into the spotlgiht the fraught issues of racial discrimination in policing. Amazon and Microsoft both pledged a year long moratorium on the use of facial-recognition technology in policing in order to mitigate issues of fairness. Tech companies that use AI and ML to build policing tools to claim improve fairer and more efficient use of police resources. with products that predict recidivism or probability of occurence of crime using tools such as the COMPAS score. It is very important to make sure AI systems are as fair as possible when used in delicate contexts that determine the future life of an individual.
In order to check for fairness in this analysis, we look at a data set of COMPAS recidivism scores and examine it as per various fairness criteria. We select 2 sensitive attributes for our analysis - ethnic background and gender.

We begin by simply examining the ROC curves for both these sensitive attributes. The ROC Curve is an evaluation metrics that is used to check and visualise the classification by showing the trade-off between sensitivity (TPR) and specificity (1-FPR). 


![png](../_resources/c20d34f3f29b407c8354722b4af24797.png)



![png](../_resources/fcf567cfb1c74eb2abe81598696665d9.png)


### Interpretation of ROC Curves by Attribute

The ROC curves for Caucasians and African American (and Hispanic) groups follow one another closely and show relatively poor performance. These curves lie closer to the 45-degree diagonal, the classification tend to be more random and the test is less accurate (AUC < 0.7 for all). The ROC curve of the Asian and Native American groups shows a more accurate classification. However,by checking the distribution of samples, we observed that the Native American group is undersampled and therefore that might be the cause of the shape of the curve. This does not mean that the predictions are accurate, because the lower number of samples, the least significant inferences can be made. 

In this plot, we compared the different rates  for the african american and caucasian group, as they are the ones with the highest number of samples in the dataset and therefore some significant inferences can be made. Predicting african-americans as more recidivistic, comes with a high cost in civil liberties rights, because a false positive prediction directly impacts the life of the individual, by determining an unfair future based on an inaccurate prediction, which is more severe than negatively predicting a low likelihood of recidivism. If the individual were to recommit a crime, they could always be re-arrested in the future, but someone that does not end up committing a crime and has to stay in jail because the algorithm predicted that they would have, cannot get those years of freedom back.

Already we've seen that the classifier performance is not great for ethnic groups. We now examine how well the COMPAS score performs on metrics of fairness - independence, separation and sufficiency.

## TASK 1
> ### 1.1 Independence - $R ⊥ A$
* Independence is satisfied when the acceptance rate ($R = 1$) of all groups are equal
* Plot acceptance rates for all possible thresholds to see if it is possible to get a classifier based on COMPAS Scores that satisfies independence


![png](../_resources/f91bb6694d25403c87ba7e36f690c7c1.png)


### Observations:

We can see from the graph that for different groups based on the sensitive attribute, the 'acceptance rate' is different for all groups over all thresholds (except 0 and 1). Independence, defined as $P( R=1 | A=a ) = P( R=1 | A=b ) \hspace{0.5cm} \forall a,b \in A$, i.e. the same acceptance rate (ideally) for every ethnic group, is therefore not satisified. 

Clearly the COMPAS score is not independent of the sensitive attribute of ethnic background and correlates with the outcome. The shape of the curve indicates the distribution of scores from 0 to 1. Looking at the different shapes of the curves it is clear that the acceptance rates are very different for the various ethnical groups. The african-american group, for instance, has the highest acceptance rates across all thresholds and their scores seem to be uniformly distributed in (0,1). This means that this group is more likely to be classified as recidivistic. In comparison, the Caucasian group have a distribution that is skewed towards lower COMPAS scores. A fair classifier, should enable everyone to have the same chances, so in this case to make it fairer different thresholds have to be established for the acceptance rates.

Independence holds only at thresholds 0 and 1 as these are the only thresholds that give the same acceptance rates for all ethnicities. However these are not viable classifiers.




![png](../_resources/089d385041194a5999ed4e910217ca61.png)


> ## 1.2 Separation - $R ⊥ A|Y$
* Separation is satisfied when the false negative rates ( FNR -> $P(R=0|Y=1)$ ) and false positive rates ( FPR -> $P(R=1|Y=0)$ ) of all groups are equal


![png](../_resources/4f26382656bb4e3bb4738d9b2f8ec9c2.png)


### Observations
In this plot we can clearly observe that the rates (FPR, FNR) are not equal for both false positives and false negatives. A false positive prediction is more costly, in these circumstances, because the freedom of an individual might directly be harmed by an inaccurate prediction. In this plot, we can observe that the African American group has the highest rate in false positive predictions and the lowest rate in false negative predictions, which again points to the unfairness of the model because the dataset is clearly disproportionately **falsely** targeting this group as more likely to re-offend in the future compared to other groups. When considering that the samples analysed by the model were unequally distributed and consisted of a high-number of African American samples. Ideally a separation-satisfying fair model should have equal false negatives and false positives rates for each group, but this imbalance shows the structural bias present in the american society being mirrored in these algorithms.


![png](../_resources/640dd57cff6c4d77822a16cfcba5d411.png)


> ## 1.3 Sufficiency - $Y⊥ A|R$
* Sufficiency is satisfied when the positive predictive values ( PPV or precision -> $P(Y=1|R=1)$ ) or negative predictive values (NPV -> $P(Y=0|R=0)$ for all groups are equal


![png](../_resources/c86fcadf703a41ed8c3eb1580777a500.png)



![png](../_resources/409e0e6c9b0445eb9fa2033a00c80d06.png)


### Observations
In general we can observe that the classifier does not provide a high dynamic range, which implies that the classifier is not good at all and it is very likely to undertake random predictions. By setting the threshold to high court, high ppv and npv can be reached. However, since the ppv count has a higher dynamic range and it is ethically more important to correctly classify positiv cases, it is better to choose a value where ppv is higher. Unfortunately, the African American group has a ppv range between approx 0,4 and 0,6. From that
we can conclude that whatever threshold is chosen, there will be around 50% falsely classified out of all positiv classified. Furthermore, this plot shows that the sample is dispoportionally classifying the African Americans as more recidivistic, which means that the samples that are fed to the model are inherently biased and could lead the classifier to determine a high correlation between the African American ethnical background and the likelihood to re-commit a crime in the future. Addtionally, Caucasians also have a small range and an even lower ppv count so there will be even more falsly positively classified people relative to the total amount of positively classified. This also gets obvious by looking at the performance of the predicted value. 
We also observed that the curve belonging to the Asian group goes slightly down between 0.7 and 0.8, which means that at 0.8 more fpv were predicted.
In general, the model is clearly not satisfying sufficiency because the sensitive characteristics are still determining different npv and ppv.

---
## TASK 2

The previous section shows that the COMPAS score does not satisfy any of the fairness criteria we examined it for. We attribute this unfairness to the bias in the nature of how and how much training data, per group, is collected. One can expect the nature of the sampled African Americans to be riddled with the structural racial bias against the group in reporting crime and harsher policing of African Americans. The number of African Americans sampled in this data set is disproportionately higher compared to the size of the group in the population - ideally, more training data would lead to better performance, but in this case it amplifies the error of falsely predicting recidivism among African Americans due to bias. Ideally, we should account for this societal structural bias that show up in the training data, and ensure that ethnicity plays no role in scoring the recidivism probability of an individual.  

In this task, we try to see if we can achieve any of the fairness criteria by using different thresholds for each group. 

> ## 2.1 Aligning for Independence - $R ⊥ A$


    C:\Users\Maheep\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
    




    (-0.1, 1)




![png](../_resources/b39f965c8f2f4eadbc3a7b48d4b506b8.png)



![png](../_resources/879c2424c9b94d9489b09926ce429316.png)


## Observations Task 2

Here we use the python implementation developed by Barocs et. al. for Chapter-2 of Fairness and machine learning, Limitations and Opportunities.  

The above plots show different thresholds for each group used to satisfy independence, the different thresholds are observable in the first plot.  

Again, ideally the acceptance rates should be equal among the groups, in order for independence to hold. However the implementation of independence used here introduces a relaxation and calculates thresholds for acceptance rates with the minimum distance. Best case, the thresholds should be equal. From this plot we can read that the thresholds need to be different for each group, which indicates that the model is not fair.

    Training:
    
    \begin{tabular}{lrrrrrr}
    \toprule
    {} &  African-American &     Asian &  Caucasian &  Hispanic &  Native American &    Others \\
    \midrule
    Independence     &          0.479857 &  0.514286 &   0.447466 &  0.476499 &         0.500000 &  0.589610 \\
    Seperation: tpr  &          0.624727 &  0.833333 &   0.627841 &  0.625000 &         1.000000 &  0.795918 \\
    Seperation: fpr  &          0.379049 &  0.448276 &   0.373767 &  0.420935 &         0.250000 &  0.519164 \\
    Sufficiency: ppv &          0.534204 &  0.277778 &   0.406998 &  0.357143 &         0.666667 &  0.343612 \\
    Sufficiency: npv &          0.703959 &  0.941176 &   0.804623 &  0.804954 &         1.000000 &  0.873418 \\
    \bottomrule
    \end{tabular}
    
    

    Testing:
    
    \begin{tabular}{lrrrrrr}
    \toprule
    {} &  African-American &     Asian &  Caucasian &  Hispanic &  Native American &    Others \\
    \midrule
    Independence     &          0.477562 &  0.562500 &   0.452096 &  0.514286 &         0.454545 &  0.549383 \\
    Seperation: tpr  &          0.606419 &  1.000000 &   0.648829 &  0.711864 &         1.000000 &  0.794872 \\
    Seperation: fpr  &          0.392897 &  0.363636 &   0.368421 &  0.451613 &         0.142857 &  0.471545 \\
    Sufficiency: ppv &          0.503506 &  0.555556 &   0.428256 &  0.333333 &         0.800000 &  0.348315 \\
    Sufficiency: npv &          0.701282 &  1.000000 &   0.808743 &  0.857143 &         1.000000 &  0.890411 \\
    \bottomrule
    \end{tabular}
    
    

## Observation
Theoretically in formal mathematical terms, independence and sufficiency can not both hold together given that the sensitive attribute ethnical background and the outcome (recidivism) are not independent, which is the case in our analysis.
These Thresholds look like they are not completely fair but if you consider a soft margin for every metric all of the criteria can nearly hold. 


    Thresholds:  [0.2, 0.5, 0.4, 0.5, 0.2, 0.4]
    

    Training:
    \begin{tabular}{lrrrrrr}
    \toprule
    {} &  African-American &     Asian &  Caucasian &  Hispanic &  Native American &    Others \\
    \midrule
    Independence     &          0.878544 &  0.285714 &   0.447466 &  0.280389 &         0.888889 &  0.324675 \\
    Seperation: tpr  &          0.950545 &  0.666667 &   0.627841 &  0.428571 &         1.000000 &  0.520408 \\
    Seperation: fpr  &          0.828441 &  0.206897 &   0.373767 &  0.224944 &         0.833333 &  0.257840 \\
    Sufficiency: ppv &          0.443954 &  0.400000 &   0.406998 &  0.416185 &         0.375000 &  0.408000 \\
    Sufficiency: npv &          0.832924 &  0.920000 &   0.804623 &  0.783784 &         1.000000 &  0.819231 \\
    \bottomrule
    \end{tabular}
    
    

    Testing:
    \begin{tabular}{lrrrrrr}
    \toprule
    {} &  African-American &     Asian &  Caucasian &  Hispanic &  Native American &    Others \\
    \midrule
    Independence     &          0.890824 &  0.187500 &   0.452096 &  0.277551 &         0.818182 &  0.259259 \\
    Seperation: tpr  &          0.954392 &  0.600000 &   0.648829 &  0.525424 &         1.000000 &  0.435897 \\
    Seperation: fpr  &          0.849057 &  0.000000 &   0.368421 &  0.198925 &         0.714286 &  0.203252 \\
    Sufficiency: ppv &          0.424812 &  1.000000 &   0.428256 &  0.455882 &         0.444444 &  0.404762 \\
    Sufficiency: npv &          0.834356 &  0.846154 &   0.808743 &  0.841808 &         1.000000 &  0.816667 \\
    \bottomrule
    \end{tabular}
    
    

# Observation of sufficiency table

From this tables we can understand that sufficiency does not hold for a hard margin. Trying to achieve equal ppv and npv for all groups is not possible, which is why we used a function that considers values in the same range, which means that the function tries to divide all the ppv with each other. By plotting these values in the test and train set we can see that the distance of thresholds differs - some are close while others are rather far. However, sufficiency can hold, if the relation between all ppvs and npvs is smaller than 0,56 which means that some npvs or ppvs are up to 50% different to each other. This points to an unfair model, as the nvps and ppvs should ideally be alike. 

    Training:
    
    \begin{tabular}{lrrrrrr}
    \toprule
    {} &  African-American &     Asian &  Caucasian &  Hispanic &  Native American &    Others \\
    \midrule
    Independence     &          0.668756 &  0.514286 &   0.563659 &  0.640194 &         0.888889 &  0.589610 \\
    Seperation: tpr  &          0.803636 &  0.833333 &   0.724432 &  0.797619 &         1.000000 &  0.795918 \\
    Seperation: fpr  &          0.574899 &  0.448276 &   0.497969 &  0.581292 &         0.833333 &  0.519164 \\
    Sufficiency: ppv &          0.493083 &  0.277778 &   0.372807 &  0.339241 &         0.375000 &  0.343612 \\
    Sufficiency: npv &          0.756757 &  0.941176 &   0.816808 &  0.846847 &         1.000000 &  0.873418 \\
    \bottomrule
    \end{tabular}
    
    

    Testing:
    
    \begin{tabular}{lrrrrrr}
    \toprule
    {} &  African-American &     Asian &  Caucasian &  Hispanic &  Native American &    Others \\
    \midrule
    Independence     &          0.688547 &  0.562500 &   0.573852 &  0.714286 &         0.818182 &  0.549383 \\
    Seperation: tpr  &          0.810811 &  1.000000 &   0.742475 &  0.813559 &         1.000000 &  0.794872 \\
    Seperation: fpr  &          0.608213 &  0.363636 &   0.502134 &  0.682796 &         0.714286 &  0.471545 \\
    Sufficiency: ppv &          0.466926 &  0.555556 &   0.386087 &  0.274286 &         0.444444 &  0.348315 \\
    Sufficiency: npv &          0.759140 &  1.000000 &   0.819672 &  0.842857 &         1.000000 &  0.890411 \\
    \bottomrule
    \end{tabular}
    
    

## Observations for separation
Could hold for the trivial cases (thresholds 0.1 and 1.1) but these are not valid classifiers.

This table shows that if you introduce a soft margin, then separation could hold within this margin.

---
## Task 3

The first plot shows that the dataset is better at predicting non-recidivism (0) than recifivism (1) because it is largely trained on non reconvicts The second plot shows a disproportional amount of Afrian Americans and Caucasians, the predictions tend to be biased as the algoritms tends to predict better on these ethnical groups, therefore, the model is more likley to classify african americans as recidivistic as not enough data is collected on other ethnic groups.

    Accuracy of the default parameters:  0.5
    

    Accuracy of the default parameters: 0.57134
    

    Accuracy of the md and mss optimized parameters: 0.64879
    

    Accuracy of the final optimized parameters: 0.62322
    

    Accuracy of the Classifier Model: 0.68594
    


![png](../_resources/fe683beed4544d75a49f424933119b52.png)



![png](../_resources/d4fe523af01145218ba8259257b07ed1.png)


## Observations

Because of the difficulties with using the COMPAS score as a classifier, we decided use a decision tree as a binary classifier, because we had experience with it and it seemed to work well with the COMPAS dataset in allowing us to make some useful inferences.

For instance, we are able to see exactly which of the features are used to classify a defendant as recidivistic and therefore, we can make sure to observe that the sensitive attributes such as ethnical background or gender are not determining the prediction of the likelihood to re-commit a crime in the future. We modified the tree many times in order to make our model as fair as possible, as we wanted to exclude the sensitive attributes of gender and ethnic background for being a discriminative attribute in the decision-making context of the algorithm. Moreover, we checked the tree for fairness with gender and ethnic background as the sensitive attributes, and the tree seems to be fair for ethnic background as the likelihood of recidivism of the defendants doesn't depend on this attribute. However, we still did not manage to completely erase the bias for gender.

We modified the code many times, by defining different random states that would give us results that tried to exclude gender and ethnic background in the predictions. However, the classifier seems to be slightly more discriminatory towards women, which still makes it an unfair model. Hence, considering such inaccurate classification performance, we are very aware of the danger of using this model in such delicate contexts that impact the future life of an individual. It is unacceptable to consider that these algorithms might ruin someone’s future based on an inaccurate prediction. In conclusion, it is still not fair because it slightly discriminates one group based on gender, which is a sensitive attribute that people do not choose and it is unacceptable to discriminate on this basis. Before it gets deployed in such contexts, fairness must be improved and assured, which is mainly depending on the data samples that are fed to the algorithm.

This ROC curve shows that all ethnical groups have the same AUC (Area Under the Curve), which means that the ethnic background is not a decisive factor in the predictions about the individual's likelihood to re-commit a crime.

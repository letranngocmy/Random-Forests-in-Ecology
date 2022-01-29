---
title: Applications of Random Forest in a Study of Animals’ Distribution in Different Climates
author: "My Le"
date: "9/26/2021"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, message=FALSE, warning=FALSE, fig.align='center')
library(skimr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(NLP)
library(caret)
library(missForest)
library(randomForest)
library(ROCR)
library(kableExtra)
library(cowplot)
library(ggridges)
```

## Abstract

The objective of this study is to develop an approach to study the association between animals' distribution in different climates and their behaviors or functional traits based on a data set of 214 mammals in Madagascar.$^1$ After imputing missing data, six random forests are grown to predict the presence or absence of mammals in dry, humid, montane, sub-arid, and sub-humid regions (by classification) and the number of climates in which the species are present (by regression). For each random forest, the best minimum size of terminal nodes ($nodesize$), the best number of variables randomly sampled at each split in the trees ($mtry$), and the best taxonomic rank to include as a predictor are determined by repeated cross-validation.

The variable importance reveals that the species' taxon, cranial capacity, home range, population density, and social group size mostly impact these species' ability to be present in different climates. Based on the receiver operating characteristic (ROC) curves, the classification models that access the mammals' distribution in montane and dry areas have the best diagnostic ability, followed by that of humid, sub-arid, and sub-humid areas. When predicting the number of climates where the species can be present, we should use just one random forest that best answers our topics of concern rather than combining different random forests in order to avoid unwanted errors caused by the misuse of explanatory variables.

\newpage

## 1. Background and Significance

Climate change and deforestation have been causing ecosystem degradation, which pushes numerous animal species into the risk of extinction.$^2$ Thus, studying the distribution of animals and factors that determine their ability to be present in different types of environments can produce valuable resources for biological conservationists to make appropriate decisions on saving endangered species and maintaining biodiversity in different regions. There are a lot of things that affect animals' presence or absence in particular types of climate, and one of the most important factors is their ability to adapt to such environments.$^3$ This ability is determined by the animals' behaviors and functional traits that have been developed through many generations in response to natural selection.$^4$

This study focuses on investigating the association between animals' behaviors, functional traits, and their distributions in different climates. Firstly, we use repeated cross-validation to generate a number of random forests to identify the optimal number of variables randomly sampled at each split in the random forests' trees ($mtry$), the minimum size of terminal nodes in the trees ($nodesize$), and which taxonomic rank to include in our random forests. With that data, we fit six random forests, five of which perform classification with the presence or absence of animals in dry, humid, montane, sub-arid, and sub-humid regions being the responses while the other uses a numeric response that informs the number of climates in which the species are present. Then, we analyze the variable importance of each random forest to see which factors are mostly associated with the distributions of animals in the five environments. Using the receiver operating characteristic (ROC) curves, we can access the diagnostic ability of our random forest algorithm in predicting the animals' presence or absence in each of the five climates. We can also evaluate the performance of our random forests through two methods of estimating the number of climates where the species can be present, one of which employs the regression forest created previously while the other combines the remaining five classification forests.

## 2. Data Pre-processing

The data employed in this study is taken from the Malagasy Animal trait Data Archive (MADA),$^1$ which gives information on the geographical and functional diversity of 214 mammal species in Madagascar. There are 61 variables in the data set, but we consider only 28 of these, which give information on the mammals' taxon (a taxonomic group of any rank), their living conditions and functionalities, and whether or not they are present in each of dry, humid, montane, sub-arid, and sub-humid environments. These 28 variables are described as follows:

  - `Order` = Taxonomic order
  - `Family` = Taxonomic family
  - `Genus` = Taxonomic genus
  - `Species` = Scientific name
  - `AdultBodyMass` = Body mass in grams
  - `CranialCapacity` = Cranial capacity of each species in cubic centimeters
  - `Diet_Invertebrates` = Presence of invertebrates in diet (yes/no)
  - `Diet_Vertebrates` = Presence of vertebrates in diet (yes/no)
  - `Diet_Fruits` = Presence of fruits in diet (yes/no)
  - `Diet_Flowers` = Presence of flower/nectar/pollen/gum in diet (yes/no)
  - `Diet_Seeds` = Presence of seeds in diet (yes/no)
  - `Diet_Plants` = Presence of other plant materials in diet (yes/no)
  - `Diet_Other` = Presence of other food items, such as scavenge, garbage, carrion, offal, and carcasses, in diet (yes/no)
  - `HabitatBreadth` = Number of habitat layers used by each species
  - `ActivityCycle` = Activity cycle of each species measured for non-captive populations: 
    + (1) nocturnal only;
    + (2) nocturnal/crepuscular, cathemeral, crepuscular, or diurnal/crepuscular; and
    + (3) diurnal only.
  - `ForagingStratum` = Stratum where the species typically forage: 
    + (1) ground level (in/near water, understory, tidal, and forest floor); 
    + (2) scansorial; 
    + (3) arboreal (mid-high canopy); and
    + (4) aerial.
  - `GestationLength` = Length of time of non-inactive fetal growth in days
  - `LitterSize` = Average number of offspring per female per litter, or midpoints of ranges if averages are not available
  - `InterbirthInterval` = Length of time between successive births of the same female(s) after a successful or unspecified litter in days
  - `HomeRange` = Size of the area within which everyday activities of individuals or groups are typically restricted in square kilometers
  - `PopulationDensity` = Number of individuals per square kilometer
  - `SocialGrpSize` = Number of individuals in a group that spends the majority of their time in a 24-hour cycle together
  - `Longevity` = Maximum adult longevity in months
  - `Dry` = Presence of species in dry regions (1 = yes, 0 = no)
  - `Humid` = Presence of species in humid regions (1 = yes, 0 = no)
  - `Montane` = Presence of species in montane regions (1 = yes, 0 = no)
  - `Subarid` = Presence of species in subarid regions (1 = yes, 0 = no)
  - `Subhumid` = Presence of species in subhumid regions (1 = yes, 0 = no)

```{r}
mammals <- read.csv("MamTraitData.csv",header=TRUE) %>%
  select(-Authority, -References, -CommonName, -DietBreadth, -TrophicLevel) %>%
  rename(Diet_Invertebrates = Diet..invertebrates,
         Diet_Vertebrates = Diet..vertebrates,
         Diet_Fruits = Diet..fruits,
         Diet_Flowers = Diet..flower.nectar.pollen.gums,
         Diet_Seeds = Diet..seeds,
         Diet_Plants = Diet..other.plant.materials,
         Diet_Other = Diet..scavenge..garbage..carrion..carcasses)
mammals[mammals == -999] <- NA
mammals[sapply(mammals, is.character)] <- lapply(mammals[sapply(mammals, is.character)], as.factor)
mammals_data <- mammals[,1:28] %>%
  transform(ActivityCycle = as.factor(ActivityCycle),
            ForagingStratum = as.factor(ForagingStratum),
            Dry = as.factor(Dry),
            Humid = as.factor(Humid),
            Montane = as.factor(Montane),
            Subarid = as.factor(Subarid),
            Subhumid = as.factor(Subhumid))
temp <- mammals_data %>%
  mutate(nClimates = strtoi(Dry)+strtoi(Humid)+strtoi(Montane)+strtoi(Subarid)+strtoi(Subhumid))
vars <- data_frame(1:length(temp),
                   names(temp),
                   sapply(temp, class),
                   sapply(temp, function(x) sum(is.na(x))), 
                   sapply(temp, function(x) n_unique(x)))
```

In organisms, a group of species forms a genus, a group of genera forms a family, a group of families forms an order, and the taxons of the same taxonomic rank do not overlap in their elements. If we have more than one taxonomic rank in a model as the predictors, the higher ranks do nothing but partly replicate the separation of animals done by the lower ranks. Thus, we should only use one taxonomic rank when creating a predictive model, and how we decide which rank to employ depends on the size of our data set and how many distinct taxons there are in that rank. Our 214 mammal species are within 6 orders, 16 families, and 61 genera (see **Table 5** in **Appendix** for this information and the complete summary of all selected variables). Since 61 and 214 are relatively high numbers of levels for a data set of length 214, we are not going to use `Genus` and `Species` to avoid overfitting. In contrast, we can include either `Order` or `Family` in the model as 6 and 16 are good numbers. The process of choosing whether to include `Order` or `Family` in our model is further discussed in **Section 3**.

```{r, fig.cap='<b>Figure 1. Frequency Plot for the Number of Climates Where the Species Are Present</b>', fig.height=3}
ggplot(data=temp, aes(x=nClimates)) + 
  geom_bar() +
  labs(y = "Number of Species", x = "Number of Climates Where the Species Are Present") +
  theme_bw()
```

From `Dry`, `Humid`, `Montane`, `Subarid`, and `Subhumid`, we create a new variable that shows the number of climates in which the species are present and name it `nClimates`. In the frequency plot of `nClimates` in **Figure 1**, we see that the numbers of species detected in exactly 1, 2, 3, and 5 types of environments do not differ significantly while that of 4 climates is noticeably lower.

```{r}
mammals_data <- mammals_data %>%
  select(-Genus, -Species)
temp <- mammals_data %>% 
  pivot_longer(c(Dry, Humid, Montane, Subarid, Subhumid), names_to = "Climate", values_to = "Present") %>%
  filter(Present != "0") 
summary <- temp %>%
  group_by(Climate) %>%
  summarize(nSpecies = n(),
            nFamilies = n_unique(Family),
            nOrders = n_unique(Order)) %>%
  arrange(desc(nSpecies))
names(summary)[2] <- "Number of Species"
names(summary)[3] <- "Number of Families"
names(summary)[4] <- "Number of Orders"
kable(summary, 
      caption = '<b>Table 1. Summary of the Five Climates<b>', 
      format = 'html') %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)
```

**Table 1** shows the number of species, families, and orders present in each of the five climates. We notice from the table that all six orders appear in every climate. Considering only the species given in the chosen data set, we find that sub-humid regions have the highest number of species, followed by that of humid regions. Although the number of species is higher for sub-arid areas than for montane areas, the number of families present in the sub-arid environment is lower than that of other places, including the montane zones. Note that **Table 1** only gives the number of species, families, and orders, which is not informative of how the mammal species in each taxonomic order and family are distributed in the five climates. Thus, we create **Figure 2** to present this information.

Let $N_{dry}$, $N_{humid}$, $N_{montane}$, $N_{subarid}$, $N_{subhumid}$ be the number of species present in dry, humid, montane, sub-arid, and sub-humid climates, respectively, and $N$ be the sum of these. Note that $N$ is greater than the number of mammal species in our data set because the number of climates where a species is present–$n$–is between 1 and 5, and so that species is counted $n$ times towards $N$. In **Figure 2**, within each order or family, the red region represents the portion of $N_{dry}$ in $N$, the brown region is for $N_{humid}$, the green region is for $N_{montane}$, the blue region is for $N_{subarid}$, and the violet region is for $N_{subhumid}$. Since the number of species varies between different orders and families (see **Table 6** in **Appendix** for the exact number of species in each taxon), normalizing $N$ to the 0-1 range helps us compare the distributions of mammals between different taxons more easily. We also need to keep in mind that the distribution of species of one order is not the average of the distributions of that order's families since different families have different numbers of species that lead to different weights for different families towards the order' distribution. Additionally, we attach the family's name to its order in the form "Order/Family" so that we can identify which order each of the families belongs to.

```{r, fig.cap='<b>Figure 2. Distribution of Mammals of Different Taxons in Different Climates</b>', fig.height=8}
p1 <- ggplot(temp, aes(x=Order, fill = Climate))+
  geom_bar(position = "fill") +
  labs(y = "") +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "none")
temp <- temp %>%
  mutate(Ord.Fam = paste0(Order,"/",Family))
p2 <- ggplot(temp, aes(x=Ord.Fam, fill = Climate))+
  geom_bar(position = "fill") +
  labs(y = "Distribution of Mammals", x = "Family") +
  coord_flip() +
  theme_bw() +
  theme(legend.position = "bottom")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(2/7, 5/7))
```

It can be seen from **Figure 2** that species in some taxonomic orders are evenly distributed in all environments, such as Carnivora and Eulipotyphla. The same thing happens with some taxonomic families, including Carnivora/Eupleridae and Chiroptera/Myzopodidae, no matter whether they belong to an order with even distribution or not. In some taxons, such as Primates, Rodentia, Chiroptera/Vespertilionidae, and Rodentia/Nesomyidae, more species appear in some climates than in others. Whereas all orders are present in all climates, none of the species in Chiroptera/Nycteridae has been detected in sub-arid regions. The sub-humid areas are the most biodiverse since the violet color takes the largest area in **Figure 2**, meaning that these areas consist of the largest number of species in most orders and families. This is consistent with the information provided in **Table 1**. In contrast, montane and sub-arid regions are the least biodiverse since the blue and green colors span the smallest area in **Figure 2**.

## 3. Proposed Methods

In this study, we focus on developing a method that primarily uses the random forest algorithm to explore the association between mammals' characteristics and their presence or absence in different climates. In R, we use the package `randomForest`$^5$ to grow a random forest. The process of growing a random forest and obtaining the predictions for each of the observations in the training set is summarized as below:

  1. For $k=1,2,...,ntree$, perform the following steps to obtain a random forest of $ntree$ trees.
    a. Generate a bootstrap sample from the original data.
    b. Grow an unpruned classification or regression tree using the sample that has just been created. Initially, all the observations in the bootstrap sample are contained in one node. If the node has size $nodesize$ or larger, it will be split into two subnodes. The subnodes and their descendants will also be split until all the nodes in the tree achieve a size smaller than $nodesize$. Note that the random forest algorithm is constructed in the way that the best split is chosen every time a node is split. To search for the best split, the algorithm randomly samples $min\{mtry,p\}$ predictors where $p$ is the total number of predictors and calculates the homogeneity of the resulting subnodes for every split made based on every predictor sampled. The best split is the one that results in the optimal homogeneity, and the predictor giving the optimal homogeneity is then selected as the criteria for splitting the node. There are a lot of ways to measure the homogeneity of the response values, and one of the most common approaches is using the Gini index, which is defined by Breiman (1984) as
$$
Gini(P) = \sum_{i=1}^{n}p_i(1-p_i)=1-\sum_{i=1}^{n}(p_i)^2
$$
    where $n$ is the length of the training data, $P=(p_1,p_2,...,p_n)$ where $p_i$ is the probability of the $i^{th}$ observation being classified to a particular class. From the above formula, we can see that the Gini index varies from 0 to 1. The Gini index of 0 informs that all the observations are allied to one single class while the value of 1 means the observations are distributed evenly in all classes. The higher the Gini index, the better the homogeneity, the better the split.
    
  2. Aggregate the predictions of all trees and predict the response values for all the observations in the original data set.
  
  3. For classification trees, the random forest's prediction of one observation is the value with majority votes by $ntree$ trees pertaining to that observation. For regression models, the predicted value of one observation is the average of $ntree$ predictions of that observation computed by all $ntree$ trees.

The performance of classification is indicated by the resulting accuracy, which measures the closeness of the model's predictions to the actual data. For regression, the model's performance is determined by the resulting root mean square error (RMSE), which is the standard deviation of the residuals. Let $I(.)$ denote a generic indicator taking value one if the condition within the parentheses is true and zero otherwise. Then accuracy and RMSE are given by:

$$
Accuracy = \frac{\sum_{i=1}^{n}I(y_i=\hat{y_i})}{n} \\
RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}{n}}
$$

where $n$ is the length of the test data, $y_i$ is the actual response value of the $i^{th}$ observation, and $\hat{y}_i$ is the $i^{th}$ observation's prediction resulting from $ntree$ trees in the random forest. The higher the accuracy and the lower the RMSE, the better the performance. For more information regarding accuracy and RMSE, see Breiman (2001).

One of the most advantageous features of random forest is its low correlation between different trees, which produce ensemble predictions that tend to be more accurate than any of the individual predictions.$^8$ Although some trees may be wrong, many other trees will be right, forcing the aggregation of the whole random forest to move in the correct direction. The following part of this section describes in detail how the random forest algorithm is involved in this study and how we can extract from the results the information of mammals' attributes and their distribution in different climates.

### Step 1: Impute missing data with `missForest`

There is a considerable number of missing values in the data set (see **Table 5** in **Appendix** for more details), which will lead to errors when the random forest algorithm is implemented. This issue can be handled by various techniques for data imputation. In this study, we use the function `missForest` in the R package `missForest`$^{9,10}$ to impute missing data.

As Stekhoven (2011) states, `missForest` is a nonparametric random forest imputation algorithm that can be applied to any data type with pairwise-independent observations. For each variable in the given data set, `missForest` grows a random forest on the observed part to predict the missing part. This step is repeated until the algorithm passes the user-specified maximum of iterations or meets a stopping criterion, which is determined by the imputation error estimates. The error estimate for the categorical part of the imputed data set is the proportion of falsely classified entries (PFC), while that of the continuous part is the normalized root mean squared error (NRMSE, see Oba et al. [2003]). Based on these estimations of imputation errors, `missForest` computes the difference(s) between the previous and the new imputation results. When the difference (for one type of variable) or the differences (for mixed-type of variables) increase, the algorithm stops. One advantage of using `missForest` is that it works well with mixed-type of variables, nonlinear relations, complex interactions, and high dimensionality. Note that to use `missForest` in R, we need to also include the R package `randomForest` by Liaw & Wiener (2002), which will then be used again in later steps.

In this study, we let `missForest` grow 1000 trees in each random forest as this is a large enough number to ensure the low correlation between individual trees. We set $mtry$ to the default value (the square root of the number of selected variables minus `nClimates` since this variable re-informs the data provided by five other variables that are also included in our data set). Since no value has been assigned to $nodesize$, the default will be applied to this parameter with 1 for continuous and 5 for categorical variables. Additionally, we assign 10 to be the maximum number of iterations to be performed in case the stopping criterion is not met beforehand. For the sake of analysis, we set seed 100 to create a fixed random-number-generator state so that our results will not be altered due to the generation of random numbers.

### Step 2: Identify the best settings for random forests with cross-validation

To study the connection between mammals' distributions in different environments and their behaviors or functional traits, we fit six random forests with `Dry`, `Humid`, `Montane`, `Subarid`, `Subhumid`, and `nClimates` as the responses and some of the remaining variables as the predictors. Note that the first five are binary variables that require classification forest to be created while the last one is a numerical variable, which is employed in regression models. 

First of all, we need to search for the optimal $mtry$ and $nodesize$ values to grow the random forest as well as which of the taxonomic ranks to include in our models, and one way to do that is performing cross-validation using the R package `caret`.$^{13}$ In cross-validation, the data is randomly partitioned into two unequal-sized data sets—the bigger one is used to train a model while the smaller one is to test the model's effectiveness. This process is repeated to test every value of the parameters that the user wants to test on, and the best model will be returned after all the values are tested. Since the existing resources in `caret` do not support multiple tuning of multiple parameters, we modify the code provided by Brownlee (2016) to create a new algorithm allowing cross-validation on different $mtry$ and $nodesize$ for both classification and regression problems.

```{r}
## Classification
customRF_classification <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF_classification$parameters <- data.frame(parameter = c("mtry", "nodesize"), class = rep("numeric", 2), label = c("mtry", "nodesize"))
customRF_classification$grid <- function(x, y, len = NULL, search = "grid") {}
customRF_classification$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, nodesize=param$nodesize, ntree=1000, ...)
}
customRF_classification$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata)
customRF_classification$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata, type = "prob")
customRF_classification$sort <- function(x) x[order(x[,1]),]
customRF_classification$levels <- function(x) x$classes
## Regression
customRF_regression <- list(type = "Regression", library = "randomForest", loop = NULL)
customRF_regression$parameters <- data.frame(parameter = c("mtry", "nodesize"), class = rep("numeric", 2), label = c("mtry", "nodesize"))
customRF_regression$grid <- function(x, y, len = NULL, search = "grid") {}
customRF_regression$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, nodesize=param$nodesize, ntree=1000, ...)
}
customRF_regression$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata)
customRF_regression$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
   predict(modelFit, newdata, type = "prob")
customRF_regression$sort <- function(x) x[order(x[,1]),]
customRF_regression$levels <- function(x) x$classes
```

Using the algorithm that we just created, we perform 5 repeats of 5-fold cross-validation, testing on the random forests using either `Order` or `Family`, $nodesize \in \{5,10,15,20,25\}$, and $mtry \in \{1,2,...,n\}$ where $n$ is the number of possible predictors to be included. Since there are 29 variables in the data set and 6 of them are the responses, we have a total of 23 predictors (see **Table 5** in **Appendix**). Since we use either `Order` or `Family` in every random forest, $n=22$. Since we grow 1000 trees in each random forest, the algorithm will create $5 \times 2 \times 5 \times 22 \times 1000 = 1,100,000$ trees with the same response, and there are six responses in total. After creating all the required random forests, our algorithm evaluates their performance and returns the classification forest with the highest accuracy and the regression forest with the lowest RMSE. We can then use the taxonomic rank included in these models and their $nodesize$ and $mtry$ values to grow the six random forests from the data of all 214 mammals for later evaluation. For detailed results of cross-validation, see **Figures 10-15** in **Appendix**.

### Step 3: Analyze the variable importance

Variable importance is one of the most popular features of random forest to quantify the importance of explanatory variables in predicting the specified response. The importance of a variable in a model depends on a number of factors including how much it is employed to make accurate predictions and how much it interacts with other variables. How the variable importance for a predictor variable $X_j$ is described by Sage (2018) as follows:

  1. Grow a random forest of $n$ trees on the set of training data.
  2. For $k=1,2,...,n$, perform the following steps.

      a. Let $\theta_k$ be the subset of indices corresponding to the out-of-bag (OOB) cases for tree $k$. Note that OOB cases are defined as the observations not included in the bootstrap sample based on which tree $k$ is grown. Let $|\theta_k|$ be the size of $\theta_k$. Recall that $I(.)$ is an indicator with value one if the condition within the parentheses is true or zero otherwise.
      b. For each $i \in \theta_t$, predict the response for case i using tree $k$. Call this prediction $\hat{y}_{ik}$.
      c. If tree $k$ is a regression tree, calculate the mean squared error. Otherwise, calculate the tree's misclassification rate. The formulas of these two statistics are given below:
$$
MSE_k=\frac{1}{|\theta_k|}\sum_{i \in \theta_k}(y_i-\hat{y}_{ik})^2\\
MCR_k=\frac{1}{|\theta_k|}\sum_{i \in \theta_k}I(y_i≠\hat{y}_{ik})
$$
      where $y_i$ is the actual response value of the $i^{th}$ observation and $\hat{y}_{ik}$ is the $i^{th}$ observation's prediction resulting from tree $k$.
      d. Randomly permute the values of predictors $X_j$ for all OOB cases and predict OOB cases again. Call these predictions $\hat{y}_{ik}^{(p)}$.
      e. Calculate $MSE_k^{(p)}$ for regression and $MCR_k^{(p)}$ for classification as follows:
$$
MSE_k^{(p)}=\frac{1}{|\theta_k|}\sum_{i \in \theta_k}(y_i-\hat{y}_{ik}^{(p)})^2\\
MCR_k^{(p)}=\frac{1}{|\theta_k|}\sum_{i \in \theta_k}I(y_i≠\hat{y}_{ik}^{(p)})\\
$$
      f. Calculate the difference in predictive performance $D_k$. 
          * For regression:
$$
D_k = MSE_k^{(p)}-MSE_k
$$
          * For classification:
$$
D_k = MCR_k^{(p)}-MCR_k
$$

  3. Obtain an overall variable importance score for $X_j$, which is equal to $\frac{1}{n}\sum_{k=1}^{n}D_k$. In regression, this value is the mean increase in MSE as we eliminate variable $X_j$ from our random forest. In classification, it is the mean decrease in accuracy of variable $X_j$ as this variable is removed. The higher a variable's mean increase in MSE or mean decrease in accuracy, the more important that variable is in predicting the response.
  
We can also access the importance of variable $X_j$ alternatively by evaluating the change in the node impurity, which indicates how well the data is split if $X_j$ is included in the random forest. In regression, the variable importance is determined by the mean decrease in Gini score as $X_j$ is eliminated from our random forest. In classification, it is measured by the increase in node purity as $X_j$ is added to the model. The higher the mean decrease in Gini and the increase in node purity, the more important the corresponding predictor is in the model. See Louppe et al. (n.d.) for more information regarding the node impurities and how the changes in impurity are computed.

From **Step 2**, we obtain six random forests created with the same set of predictions and different responses. We will analyze the variable importance for each of them to see which factors are dominant in the relationship with the mammals' ability to be present in different climates.

### Step 4: Evaluate the classification performance with ROC curves

ROC curve is one of the features to evaluate the diagnostic ability of the random forest algorithm in binary classification problems. It is created by plotting the true positive rate (sensitivity) against the false positive rate (specificity) as the decision threshold is varied. Different ROC curves result from different classification models, but to make comparisons between the models, it is required that these models all have the same training data set and the same test data set. If one ROC curve has an area under the curve larger than that of other ROC curves, we can conclude that its corresponding random forest appears to perform better than other random forests. Note that we cannot evaluate the performance of regression models with ROC curves since it cannot be created when a numerical response or a categorical variable with more than three levels is employed. See Song (2012) for further discussion about ROC curves. In the context of our study, ROC curves of the five classification forests demonstrate in which climates the random forest algorithm best informs the mammals' presence or absence, provided we know their living behaviors and functional traits.

### Step 5: Predict `nClimates` in two different ways and evaluate their performance

Since `nClimates` is not a binary variable, we cannot create a ROC curve to access the performance of our random forest algorithm in capturing the relationship between mammals' attributes and the number of climates in which they are present. Instead, we can evaluate how good a predictive regression model is based on its RMSE (see how RMSE is computed at the beginning of **Section 3**).

With the six random forests created above, we have two ways to obtain the predictions for `nClimates`. The first method uses the regression forest with `nClimates` as the response to directly generate predictions for this variable, and we name this **Method 1**. Alternatively, the second method predicts `nClimates` indirectly by taking the element-wise sums of the predictions obtained from the five classification forests, just like how we compute the true value of `nClimates` using the true values of `Dry`, `Humid` `Montane`, `Subarid`, and `Subhumid`. We call this one **Method 2**.

## 4. Results and Discussion

### Step 1: Impute missing data with `missForest`

Imputing missing data with `missForest`, we obtain the information of the performance of this algorithm on our mammals' data set as follows:

```{r, include=FALSE}
set.seed(100)
imputed_mammals <- missForest(mammals_data, maxiter=10, ntree=1000, replace=TRUE, verbose = TRUE)
mammals_data <- imputed_mammals$ximp
```

```{r}
Iteration <- 1:4
NRMSE <- c(0.3589244,0.3197995,0.3193118,0.3136715)
PFC <- c(0.03728657,0.03110745,0.03207941,0.03268122)
Diff_NRMSE <- c(0.008662978,0.000322855,0.0001079977,0.0001703932)
Diff_PFC <- c(0.01606308,0.0002920561,0,0)
Runtime <- c(6.921,6.863,6.705,6.806)
temp <- data_frame(Iteration, NRMSE, PFC, Diff_NRMSE, Diff_PFC, Runtime)
names(temp)[4] <- "Difference in NRMSE"
names(temp)[5] <- "Difference in PFC"
names(temp)[6] <- "Runtime in seconds"
kable(temp, 
      caption = '<b>Table 2. Summary of Iterations in Missing-Data Imputation<b>', 
      format = 'html') %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)
```

From **Table 2**, we see that `missForest` needs four iterations to meet the stopping criterion. From iteration 1 to iteration 4, NRMSE decreases while PFC decreases to 0.0311074 in iteration 2 and then increases again. From iteration 1 to iteration 3, the difference in RMSE, the difference in PFC, and the iteration runtime decrease. From iteration 3 to iteration 4, the difference in RMSE and the runtime increases, while the difference in PFC stays the same at zero. Since the total of differences is found to increase after iteration 3 takes place, the stopping criterion is triggered at this point. Thus, iteration 3 is likely to be the most accurate imputation, and its setting is selected to finalize the missing-data imputation.

### Step 2: Identify the best settings for random forests with cross-validation

Having done the imputation of missing data, we proceed to the next step to use cross-validation to find the best settings for our random forests. **Table 3** provides the summary of the most well-performing random forest models found in cross-validation. The models are sorted in the descending order of the models' performance (accuracy for classifications and RMSE for regressions).

```{r}
set.seed(100)
control <- trainControl(method="repeatedcv", number=5, repeats=5)
## Response = Dry
data_Dry <- mammals_data %>%
  select(-Humid, -Montane, -Subarid, -Subhumid)
# rf_useOrder_Dry <- train(data=data_Dry %>% select(-Family), Dry~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Dry)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# rf_useFamily_Dry <- train(data=data_Dry %>% select(-Order), Dry~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Dry)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# saveRDS(rf_useOrder_Dry, "rf_useOrder_Dry.rds")
# saveRDS(rf_useFamily_Dry, "rf_useFamily_Dry.rds")
rf_useOrder_Dry <- readRDS("rf_useOrder_Dry.rds")
rf_useFamily_Dry <- readRDS("rf_useFamily_Dry.rds")
## Response = Humid
data_Humid <- mammals_data %>%
  select(-Dry, -Montane, -Subarid, -Subhumid)
# rf_useOrder_Humid <- train(data=data_Humid %>% select(-Family), Humid~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Humid)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# rf_useFamily_Humid <- train(data=data_Humid %>% select(-Order), Humid~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Humid)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# saveRDS(rf_useOrder_Humid, "rf_useOrder_Humid.rds")
# saveRDS(rf_useFamily_Humid, "rf_useFamily_Humid.rds")
rf_useOrder_Humid <- readRDS("rf_useOrder_Humid.rds")
rf_useFamily_Humid <- readRDS("rf_useFamily_Humid.rds")
## Response = Montane
data_Montane <- mammals_data %>%
  select(-Dry, -Humid, -Subarid, -Subhumid)
# rf_useOrder_Montane <- train(data=data_Montane %>% select(-Family), Montane~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Montane)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# rf_useFamily_Montane <- train(data=data_Montane %>% select(-Order), Montane~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Montane)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# saveRDS(rf_useOrder_Montane, "rf_useOrder_Montane.rds")
# saveRDS(rf_useFamily_Montane, "rf_useFamily_Montane.rds")
rf_useOrder_Montane <- readRDS("rf_useOrder_Montane.rds")
rf_useFamily_Montane <- readRDS("rf_useFamily_Montane.rds")
## Response = Subarid
data_Subarid <- mammals_data %>%
  select(-Dry, -Humid, -Montane, -Subhumid)
# rf_useOrder_Subarid <- train(data=data_Subarid %>% select(-Family), Subarid~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Subarid)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# rf_useFamily_Subarid <- train(data=data_Subarid %>% select(-Order), Subarid~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Subarid)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# saveRDS(rf_useOrder_Subarid, "rf_useOrder_Subarid.rds")
# saveRDS(rf_useFamily_Subarid, "rf_useFamily_Subarid.rds")
rf_useOrder_Subarid <- readRDS("rf_useOrder_Subarid.rds")
rf_useFamily_Subarid <- readRDS("rf_useFamily_Subarid.rds")
## Response = Subhumid
data_Subhumid <- mammals_data %>%
  select(-Dry, -Humid, -Montane, -Subarid)
# rf_useOrder_Subhumid <- train(data=data_Subhumid %>% select(-Family), Subhumid~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Subhumid)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# rf_useFamily_Subhumid <- train(data=data_Subhumid %>% select(-Order), Subhumid~., method=customRF_classification, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_Subhumid)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# saveRDS(rf_useOrder_Subhumid, "rf_useOrder_Subhumid.rds")
# saveRDS(rf_useFamily_Subhumid, "rf_useFamily_Subhumid.rds")
rf_useOrder_Subhumid <- readRDS("rf_useOrder_Subhumid.rds")
rf_useFamily_Subhumid <- readRDS("rf_useFamily_Subhumid.rds")
## Response = nClimates
data_nClimates <- mammals_data %>%
  mutate(nClimates = strtoi(Dry)+strtoi(Humid)+strtoi(Montane)+strtoi(Subarid)+strtoi(Subhumid)) %>%
  select(-Dry, -Humid, -Montane, -Subarid, -Subhumid)
# rf_useOrder_nClimates <- train(data=data_nClimates %>% select(-Family), nClimates~., method=customRF_regression, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_nClimates)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# rf_useFamily_nClimates <- train(data=data_nClimates %>% select(-Order), nClimates~., method=customRF_regression, tuneGrid=expand.grid(.mtry=c(1:(ncol(data_nClimates)-2)), .nodesize=c(5, 10, 15, 20, 25)), trControl=control)
# saveRDS(rf_useOrder_nClimates, "rf_useOrder_nClimates.rds")
# saveRDS(rf_useFamily_nClimates, "rf_useFamily_nClimates.rds")
rf_useOrder_nClimates <- readRDS("rf_useOrder_nClimates.rds")
rf_useFamily_nClimates <- readRDS("rf_useFamily_nClimates.rds")
```

```{r}
Rank <- c("Order", "Family")
Model <- c("Classification", "Classification")
RMSE <- c("--","--")
## Response = Dry
Response <- c("Dry", "Dry")
best_mtry <- c(rf_useOrder_Dry$bestTune[1,1],rf_useFamily_Dry$bestTune[1,1])
best_nodesize <- c(rf_useOrder_Dry$bestTune[1,2],rf_useFamily_Dry$bestTune[1,2])
temp1 <- rf_useOrder_Dry$results %>%
  filter(mtry == best_mtry[1], nodesize == best_nodesize[1])
temp2 <- rf_useFamily_Dry$results %>%
  filter(mtry == best_mtry[2], nodesize == best_nodesize[2])
Accuracy <- c(as.String(round(temp1[1,3],4)), as.String(round(temp2[1,3],4)))
bestModels_Dry <- data_frame(Model,Response,Rank,best_mtry,best_nodesize,Accuracy,RMSE)
## Response = Humid
Response <- c("Humid", "Humid")
best_mtry <- c(rf_useOrder_Humid$bestTune[1,1],rf_useFamily_Humid$bestTune[1,1])
best_nodesize <- c(rf_useOrder_Humid$bestTune[1,2],rf_useFamily_Humid$bestTune[1,2])
temp1 <- rf_useOrder_Humid$results %>%
  filter(mtry == best_mtry[1], nodesize == best_nodesize[1])
temp2 <- rf_useFamily_Humid$results %>%
  filter(mtry == best_mtry[2], nodesize == best_nodesize[2])
Accuracy <- c(as.String(round(temp1[1,3],4)), as.String(round(temp2[1,3],4)))
bestModels_Humid <- data_frame(Model,Response,Rank,best_mtry,best_nodesize,Accuracy,RMSE)
## Response = Montane
Response <- c("Montane", "Montane")
best_mtry <- c(rf_useOrder_Montane$bestTune[1,1],rf_useFamily_Montane$bestTune[1,1])
best_nodesize <- c(rf_useOrder_Montane$bestTune[1,2],rf_useFamily_Montane$bestTune[1,2])
temp1 <- rf_useOrder_Montane$results %>%
  filter(mtry == best_mtry[1], nodesize == best_nodesize[1])
temp2 <- rf_useFamily_Montane$results %>%
  filter(mtry == best_mtry[2], nodesize == best_nodesize[2])
Accuracy <- c(as.String(round(temp1[1,3],4)), as.String(round(temp2[1,3],4)))
bestModels_Montane <- data_frame(Model,Response,Rank,best_mtry,best_nodesize,Accuracy,RMSE)
## Response = Subarid
Response <- c("Subarid", "Subarid")
best_mtry <- c(rf_useOrder_Subarid$bestTune[1,1],rf_useFamily_Subarid$bestTune[1,1])
best_nodesize <- c(rf_useOrder_Subarid$bestTune[1,2],rf_useFamily_Subarid$bestTune[1,2])
temp1 <- rf_useOrder_Subarid$results %>%
  filter(mtry == best_mtry[1], nodesize == best_nodesize[1])
temp2 <- rf_useFamily_Subarid$results %>%
  filter(mtry == best_mtry[2], nodesize == best_nodesize[2])
Accuracy <- c(as.String(round(temp1[1,3],4)), as.String(round(temp2[1,3],4)))
bestModels_Subarid <- data_frame(Model,Response,Rank,best_mtry,best_nodesize,Accuracy,RMSE)
## Response = Subhumid
Response <- c("Subhumid", "Subhumid")
best_mtry <- c(rf_useOrder_Subhumid$bestTune[1,1],rf_useFamily_Subhumid$bestTune[1,1])
best_nodesize <- c(rf_useOrder_Subhumid$bestTune[1,2],rf_useFamily_Subhumid$bestTune[1,2])
temp1 <- rf_useOrder_Subhumid$results %>%
  filter(mtry == best_mtry[1], nodesize == best_nodesize[1])
temp2 <- rf_useFamily_Subhumid$results %>%
  filter(mtry == best_mtry[2], nodesize == best_nodesize[2])
Accuracy <- c(as.String(round(temp1[1,3],4)), as.String(round(temp2[1,3],4)))
bestModels_Subhumid <- data_frame(Model,Response,Rank,best_mtry,best_nodesize,Accuracy,RMSE)
## Response = nClimates
Response <- c("nClimates", "nClimates")
Model <- c("Regression", "Regression")
Accuracy <- c("--","--")
best_mtry <- c(rf_useOrder_nClimates$bestTune[1,1],rf_useFamily_nClimates$bestTune[1,1])
best_nodesize <- c(rf_useOrder_nClimates$bestTune[1,2],rf_useFamily_nClimates$bestTune[1,2])
temp1 <- rf_useOrder_nClimates$results %>%
  filter(mtry == best_mtry[1], nodesize == best_nodesize[1])
temp2 <- rf_useFamily_nClimates$results %>%
  filter(mtry == best_mtry[2], nodesize == best_nodesize[2])
RMSE <- c(as.String(round(temp1[1,3],4)), as.String(round(temp2[1,3],4)))
bestModels_nClimates <- data_frame(Model,Response,Rank,best_mtry,best_nodesize,Accuracy,RMSE)
bestModels <- rbind(bestModels_Dry,bestModels_Humid,bestModels_Montane,bestModels_Subarid,bestModels_Subhumid,bestModels_nClimates) %>%
  arrange(desc(Accuracy), RMSE)
names(bestModels)[1] <- "Model Type"
names(bestModels)[3] <- "Taxonomic Rank Included"
names(bestModels)[4] <- "Best $mtry$"
names(bestModels)[5] <- "Best $nodesize$"
kable(bestModels, 
      caption = '<b>Table 3. Summary of the Most Well-Performing Random Forests in Cross-Validation<b>', 
      format = 'html') %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)
```

As **Table 3** illustrates, all these 12 random forests are good at predicting the responses since they have high accuracy (above 0.7) or low RMSE (below 1.2) for such real data. Thus, the random forest algorithm does a good job of investigating the association between the mammals' attributes and their distribution in different climates. There is no major difference in the random forests' performance if they include different taxonomic ranks to predict the same response. Therefore, we can either employ just one taxonomic rank to predict all the responses or use the settings that **Table 3** suggests to create six random forests for six response variables. For example, if we want to grow a random forest with `Humid` as the response and `Family` as a predictor, we set $mtry = 10$ and $nodesize = 5$. Since `Family` helps partition the mammals into smaller groups than `Order` does without causing overfitting, we use this variable as one of the predictors in `RF_Dry`, `RF_Humid`, `RF_Montane`, `RF_Subarid`, `RF_Subhumid`, `RF_nClimates`, which allows us to understand the association between animals' characteristics and their distribution in different climates more thoroughly.

```{r}
set.seed(100)
RF_Dry <-randomForest(data=data_Dry%>%select(-Order), Dry~., mtry=18, nodesize=10, ntree=1000, importance=TRUE, replace=TRUE)
RF_Humid <-randomForest(data=data_Humid%>%select(-Order), Humid~., mtry=10, nodesize=5, ntree=1000, importance=TRUE, replace=TRUE) 
RF_Montane <-randomForest(data=data_Montane%>%select(-Order), Montane~., mtry=3, nodesize=5, ntree=1000, importance=TRUE, replace=TRUE)
RF_Subarid <-randomForest(data=data_Subarid%>%select(-Order), Subarid~., mtry=12, nodesize=25, ntree=1000, importance=TRUE, replace=TRUE)
RF_Subhumid <-randomForest(data=data_Subhumid%>%select(-Order), Subhumid~., mtry=1, nodesize=5, ntree=1000, importance=TRUE, replace=TRUE)
RF_nClimates <-randomForest(data=data_nClimates%>%select(-Order), nClimates~., mtry=6, nodesize=5, ntree=1000, importance=TRUE, replace=TRUE)
```

### Step 3: Analyze the variable importance

```{r, fig.cap='<b>Figure 3. Variable Importance in `RF_Dry`</b>'}
varImpPlot(RF_Dry)
```

```{r, fig.cap='<b>Figure 4. Variable Importance in `RF_Humid`</b>'}
varImpPlot(RF_Humid)
```

```{r, fig.cap='<b>Figure 5. Variable Importance in `RF_Montane`</b>'}
varImpPlot(RF_Montane)
```

```{r, fig.cap='<b>Figure 6. Variable Importance in `RF_Subarid`</b>'}
varImpPlot(RF_Subarid)
```

```{r, fig.cap='<b>Figure 7. Variable Importance in `RF_Subhumid`</b>'}
varImpPlot(RF_Subhumid)
```

```{r, fig.cap='<b>Figure 8. Variable Importance in `RF_nClimates`</b>'}
varImpPlot(RF_nClimates)
```

**Figures 3-8** display the variable importance of our six random forests with the predictors being ordered in the decreasing direction of importance. If we omit `Family`, `LitterSize`, or `PopulationDensity` from `RF_Dry`, `RF_Humid`, and `RF_Montane`, the corresponding accuracy and Gini score are estimated to decrease by at least 15 and 5, respectively. The same thing happens if we remove `CranialCapacity` and `SocialGrpSize` from `RF_Humid` and `RF_Montane`, `HomeRange` from `RF_Dry`, `RF_Humid`, and `RF_Subarid`, and `GestationLength` from `RF_Dry` and `RF_Montane`. In `RF_nClimates`, eliminating any of `Family`, `CranialCapacity`, `SocialGrpSize`, `AdultBodyMass`, `PopulationDensity`, `ForagingStratum`, `GestationLength`, and `InterbirthInterval` can lead to an increase of at least 15% in the mean squared error and at least 27 in the node impurity. As can be seen from **Figure 7**, removing any of the variables in `RF_Subhumid` does not make any major changes in the model's performance, which means these variables contribute nearly equally to this model. Overall, `HomeRange`, `Family`, `PopulationDensity`, `CranialCapacity`, and `SocialGrpSize` are among the five most important predictors in at least three of the six random forests. In the context of our study, these are dominant factors in determining the presence or absence of the mammal species in different climates. On the other hand, `HabitatBreadth`, `ActivityCycle`, and factors concerning the animals' diet are the least important in all the random forests, which reveals the weak connections between these variables and the mammals' distribution in the five climates.

### Step 4: Evaluate the classification performance with ROC curves

```{r, fig.cap='<b>Figure 9. ROC Curves of the Five Classification Models</b>'}
## Response = Dry
pred1_Dry = predict(RF_Dry,type = "prob")
perf = prediction(pred1_Dry[,2], data_Dry$Dry)
auc_Dry <- round(performance(perf, "auc")@y.values[[1]],4)
pred2 = performance(perf, "tpr","fpr")
plot(pred2,main="ROC Curves",col="black",lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")
## Response = Humid
pred1_Humid = predict(RF_Humid,type = "prob")
perf = prediction(pred1_Humid[,2], data_Humid$Humid)
auc_Humid <- round(performance(perf, "auc")@y.values[[1]],4)
pred2 = performance(perf, "tpr","fpr")
plot(pred2,col="red",lwd=2, add = TRUE)
## Response = Montane
pred1_Montane = predict(RF_Montane,type = "prob")
perf = prediction(pred1_Montane[,2], data_Montane$Montane)
auc_Montane <- round(performance(perf, "auc")@y.values[[1]],4)
pred2 = performance(perf, "tpr","fpr")
plot(pred2,col="green",lwd=2, add = TRUE)
## Response = Subarid
pred1_Subarid = predict(RF_Subarid,type = "prob")
perf = prediction(pred1_Subarid[,2], data_Subarid$Subarid)
auc_Subarid <- round(performance(perf, "auc")@y.values[[1]],4)
pred2 = performance(perf, "tpr","fpr")
plot(pred2,col="blue",lwd=2, add = TRUE)
## Response = Subhumid
pred1_Subhumid = predict(RF_Subhumid,type = "prob")
perf = prediction(pred1_Subhumid[,2], data_Subhumid$Subhumid)
auc_Subhumid <- round(performance(perf, "auc")@y.values[[1]],4)
pred2 = performance(perf, "tpr","fpr")
plot(pred2,col="violet",lwd=2, add = TRUE)
## Add legend
legend(0.67, 0.49, legend=c("RF_Dry", "RF_Humid", "RF_Montane", "RF_Subarid", "RF_Subhumid"), col=c("black", "red", "green", "blue", "violet"), lty=1, lwd=2, title="Model")
```

```{r}
AUC <- round(c(auc_Dry, auc_Humid, auc_Montane, auc_Subarid, auc_Subhumid),4)
Model <- c("RF_Dry", "RF_Humid", "RF_Montane", "RF_Subarid", "RF_Subhumid")
temp <- data_frame(Model,AUC) %>%
  arrange(desc(AUC))
names(temp)[2] <- "Area Under the Curve"
temp <- t(temp)
kable(temp, 
      caption = '<b>Table 4. Areas Under the Five ROC Curves<b>',
      format = 'html') %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F) %>%
  column_spec(1, bold = TRUE, width="4.5cm") %>%
  column_spec(2, width="2.5cm") %>%
  column_spec(3, width="2.5cm") %>%
  column_spec(4, width="2.5cm") %>%
  column_spec(5, width="2.5cm") %>%
  column_spec(6, width="2.5cm")
```

The ROC curves of the five classification forests are shown in **Figure 9**. At almost all values of false positive rate, it is easily recognized that the true positive rates of `RF_Dry`, `RF_Humid`, `RF_Montane`, and `RF_Subarid` do not differ significantly while that of `RF_Subhumid` is considerably lower. Among the first four random forests, `RF_Dry` and `RF_Montane` are found to have the highest area under the ROC curve, based on **Table 4**. This demonstrates that our classification algorithm performs the best when predicting `Montane` and `Dry`, followed by `Humid`, `Subarid`, and `Subhumid`. Since all the random forests have the ROC curves lie mostly above the diagonal lines (or the area under the ROC curve above 0.5), we can claim that our classification algorithm is good for investigating the relationship between the mammals' attributes and their ability to be present in each of the five climates.

### Step 5: Predict `nClimates` in two different ways and evaluate their performance

```{r, fig.cap='<b>Figure 10. Density Plot for the Predictions of `n_Climates` Obtained by Method 1 and Method 2</b>'}
methods <- c()
actual <- c()
predictions <- c()
## Method 1
pred1_nClimates = predict(RF_nClimates, newdata=data_nClimates%>%select(-Order,-nClimates))
pred1_nClimates <- unname(pred1_nClimates)
sum_ResSqr <- 0
for (i in 1:length(data_nClimates)) {
  methods <- append(methods, "Method 1")
  actual <- append(actual, data_nClimates$nClimates[i])
  predictions <- append(predictions, pred1_nClimates[i])
  sum_ResSqr = sum_ResSqr + (pred1_nClimates[i]-data_nClimates$nClimates[i])^2
}
RMSE1 = sqrt(sum_ResSqr/length(data_nClimates))
## Method 2
pred_Dry <- round(unname(pred1_Dry[,2]),0)
pred_Humid <- round(unname(pred1_Humid[,2]),0)
pred_Montane <- round(unname(pred1_Montane[,2]),0)
pred_Subarid <- round(unname(pred1_Subarid[,2]),0)
pred_Subhumid <- round(unname(pred1_Subhumid[,2]),0)
pred2_nClimates <- pred_Dry+pred_Humid+pred_Montane+pred_Subarid+pred_Subhumid
sum_ResSqr <- 0
for (i in 1:length(data_nClimates)) {
  methods <- append(methods, "Method 2")
  actual <- append(actual, data_nClimates$nClimates[i])
  predictions <- append(predictions, pred2_nClimates[i])
  sum_ResSqr = sum_ResSqr + (pred2_nClimates[i]-data_nClimates$nClimates[i])^2
}
RMSE2 = sqrt(sum_ResSqr/length(data_nClimates))
## Plot
temp <- data_frame(methods,actual,predictions) %>%
  mutate(methods=ifelse(methods=="Method 1",paste0("Method 1: RMSE = ",round(RMSE1,4)),paste0("Method 2: RMSE = ",round(RMSE2,4))))
ideal <- data.frame(actual=1:5,predictions=1:5)
ggplot(data=temp, aes(y=as.factor(actual), x=predictions, fill=as.factor(actual))) +
  geom_density_ridges(alpha=0.7, color="darkgrey") +
  geom_point(data=ideal,aes(y=as.factor(actual),x=predictions),color="red") +
  labs(x = "Predicted Value", y = "Actual Value") +
  scale_x_continuous(breaks = seq(1,5,1)) +
  facet_grid(methods~.) +
  scale_fill_discrete(guide=FALSE) +
  theme_bw() +
  theme(strip.background=element_rect(colour="black", fill="white"))
```

**Figure 10** exhibits the density plot for the predictions of `n_Climates` with the red dots being placed where the predictions match its actual values. First of all, the figure reveals that the densities of the predictions pertaining to both methods shift from the right of the true value to its left as the true value increases. Besides, the predictions obtained by **Method 2** deviate further from the true values than those of **Method 1** do. Thus, the RMSE resulting from **Method 2** is more than twice as much as that of **Method 1**. This is not surprising because different random forests models are built with the parameters or the set of predictors being adjusted to obtain the best predictions for the corresponding response, but these adjustments may not be ideal for predicting the other responses. For instance, we cannot remove unnecessary predictors if they are included in the initial models, and it is hard to monitor the weights of the variables that are not very important in some participating models but turn out to be significant in predicting the new response. This will lead to errors in our predictions in addition to the unexplained variabilities created by noises and the factors that are not included in the source data set. The best way to avoid these types of errors is identifying the most appropriate response to create a single model that best answers the question we are working with.

\newpage

## 5. Remarks and Conclusions

Throughout the analysis, our proposed method appears to be helpful for studying the association between mammals’ attributes and their distribution in different climates. Cross-validation has revealed that our data allows using either `Order` or `Family` to grow the random forests since the best models of both types result in nearly the same performance. Thus, we use `Family` to create six random forests with `Dry`, `Humid`, `Montane`, `Subarid`, `Subhumid`, and `nClimates` as the responses since the corresponding taxonomic rank is better in partitioning the data, which is likely to give us better (or more detailed) generalizations about the mammals' distribution.

Analyzing the variable importance, we find that `HomeRange`, `Family`, `PopulationDensity`, `CranialCapacity`, and `SocialGrpSize` significantly affect the distribution of mammals in different climates while `HabitatBreadth`, `ActivityCycle`, and the mammals' diet are the least important ones. The factors involved in `RF_Subhumid` are nearly equally critical in studying the mammals' presence or absence in sub-humid areas while some of the factors in other random forests are found to be more dominant than the others in predicting the corresponding responses. From the ROC curves, it can be recognized that the random forest classification algorithm we employ is good at predicting the mammals' ability to be present in each of the five climates, especially montane and dry environments. Using two different ways to access the diagnostic ability of our random forests in predicting `nClimates`, we see that the method giving the predictions directly with `RF_nClimates` leads to much lower RMSE than the indirect approach that combines the predictions from five other random forests. This suggests that we should try to understand the problems we are working with and carefully do the data pre-processing so that we can create the most appropriate model for our analysis, which helps avoid some types of unwanted errors that we can totally control.

Although our work produces valuable results, there are some limitations that may affect the reliability of the analysis. Firstly, there are a large amount of missing data in our original data set, and so there is a possibility that `missForest` replaces missing values with incorrect data, which causes our results to be unreliable. Secondly, there can be some factors other than the 61 variables in the original data set that greatly impact the distribution of mammals in different environments that we cannot account for. Thirdly, since we work with only 214 mammals in Madagascar, we should not generalize our findings to mammals in other geographic areas or the species in the taxonomic families that are not included in the data set. Thus, our work can be further developed to study the distribution of animals or any types of living organisms anywhere if we are provided with enough data to do so. If our data set is larger in size, we may be able to use the taxonomic ranks lower than family (e.g. genus and species) to obtain more details about the relationship between animals' attributes and their distribution in different climates. We can also study the distribution of them in a lot more climates if we have a data set that consists of this information.

## 6. Acknowledgments

I would like to sincerely thank Dr. Andrew Sage for his insightful advice and tremendous support during my research.

\newpage

## 7. References

1. Razafindratsima, O. H., Yacoby, Y., & Park, D. S. (2019). *Data from: MADA: Malagasy Animal trait Data Archive* (Version 1) [Data set]. Dryad. https://doi.org/10.5061/DRYAD.44TT0

2. Trisurat, Y., Shrestha, R., & Alkemade, R. (2011). Consequences of Deforestation and Climate Change on Biodiversity. In *Land use, climate change and Biodiversity Modeling: Perspectives and Applications* (24–51). Information Science Reference. http://doi:10.4018/978-1-60960-619-0.ch002

3. Loh, R., David, J. R., Debat, V., & Bitner-Mathé, B. C. (2008). Adaptation to different climates results in divergent phenotypic plasticity of wing size and shape in an invasive drosophilid. *Journal of Genetics, 87*(3), 209–217. https://doi.org/10.1007/s12041-008-0034-2

4. Visser, M. E. (2008). Keeping up with a warming world; assessing the rate of adaptation to climate change. *Proceedings of the Royal Society B: Biological Sciences, 275*(1635), 649–659. https://doi.org/10.1098/rspb.2007.0997

5. Liaw, A. & Wiener, M. (2002). Classification and Regression by randomForest. *R News, 2*(3), 18-22.

6. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification And Regression Trees* (1st ed.). Routledge. https://doi.org/10.1201/9781315139470

7. Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/a:1010933404324

8. Hjerpe, A. (2016). *Computing Random Forests Variable Importance Measures (VIM) on Mixed Numerical and Categorical Data* [Masters thesis, School of Computer Science and Communication]. http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-185496

9. Stekhoven, D. J., & Buehlmann, P. (2011). MissForest - non-parametric missing value imputation for mixed-type data. *Bioinformatics, 28*(1), 112-118.

10. Stekhoven, D. J. (2013). missForest: Nonparametric Missing Value Imputation using Random Forest. R package version 1.4.

11. Stekhoven, D. J. (2011). Using the missForest Package (Version 1.2). https://stat.ethz.ch/education/semesters/ss2012/ams/paper/missForest_1.2.pdf

12. Oba, S., Sato, M. A., Takemasa, I., Monden, M., Matsubara, K. I., & Ishii, S. (2003). A Bayesian missing value estimation method for gene expression profile data. *Bioinformatics, 19*(16), 2088–2096. https://doi.org/10.1093/bioinformatics/btg287

13. Max, K. (2020). caret: Classification and Regression Training. R package version 6.0-86. https://CRAN.R-project.org/package=caret

14. Brownlee, J. (2016, February 5). *Tune Machine Learning Algorithms in R (random forest case study)*. Machine Learning Mastery. https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/

15. Sage, A. (2018). *Random forest robustness, variable importance, and tree aggregation* [Doctoral dissertation, Iowa State University]. Iowa State University Digital Repository. https://lib.dr.iastate.edu/etd/16453

16. Louppe, G., Wehenkel, L., Sutera, A., and Geurts, P. (2013). Understanding variable importances in forests of randomized trees. In *NIPS'13: Proceedings of the 26th International Conference on Neural Information Processing Systems - Volume 1* (431–439). Curran Associates Inc., Red Hook, NY, USA.

17. Song, B. (2015, January 5). *ROC Random Forest and Its Application* [Doctoral dissertation, Stony Brook University]. DSpace Repository. http://hdl.handle.net/11401/77477

\newpage

## 8. Appendix

```{r}
names(vars)[1] <- "No."
names(vars)[2] <- "Variable"
names(vars)[3] <- "Data Type"
names(vars)[4] <- "Number of Missing Values"
names(vars)[5] <- "Number of Unique Values"
kable(vars, 
      caption = '<b>Table 5. Summary of the Selected Variables</b>', 
      format = 'html') %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F)
```

```{r}
temp <- mammals_data %>%
  group_by(Order,Family) %>%
  summarise(n = n())
names(temp)[3] <- "Number of Species"
kable(temp, 
      caption = '<b>Table 6. Number of Species in Each Taxon</b>', 
      format = 'html') %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = F) %>%
  column_spec(1, width="3cm")
```

```{r, fig.cap='<b>Figure 11. Results of Cross-Validation on the Random Forests Involving `Dry`</b>', fig.height=8}
p1 <- plot(rf_useOrder_Dry, main = "Models Involving `Order`")
p2 <- plot(rf_useFamily_Dry, main = "Models Involving `Family`")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(1/2, 1/2))
```

```{r, fig.cap='<b>Figure 12. Results of Cross-Validation on the Random Forests Involving `Humid`</b>', fig.height=8}
p1 <- plot(rf_useOrder_Humid, main = "Models Involving `Order`")
p2 <- plot(rf_useFamily_Humid, main = "Models Involving `Family`")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(1/2, 1/2))
```

```{r, fig.cap='<b>Figure 13. Results of Cross-Validation on the Random Forests Involving `Montane`</b>', fig.height=8}
p1 <- plot(rf_useOrder_Montane, main = "Models Involving `Order`")
p2 <- plot(rf_useFamily_Montane, main = "Models Involving `Family`")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(1/2, 1/2))
```

```{r, fig.cap='<b>Figure 14. Results of Cross-Validation on the Random Forests Involving `Subarid`</b>', fig.height=8}
p1 <- plot(rf_useOrder_Subarid, main = "Models Involving `Order`")
p2 <- plot(rf_useFamily_Subarid, main = "Models Involving `Family`")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(1/2, 1/2))
```

```{r, fig.cap='<b>Figure 15. Results of Cross-Validation on the Random Forests Involving `Subhumid`</b>', fig.height=8}
p1 <- plot(rf_useOrder_Subhumid, main = "Models Involving `Order`")
p2 <- plot(rf_useFamily_Subhumid, main = "Models Involving `Family`")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(1/2, 1/2))
```

```{r, fig.cap='<b>Figure 16. Results of Cross-Validation on the Random Forests Involving `nClimates`</b>', fig.height=8}
p1 <- plot(rf_useOrder_nClimates, main = "Models Involving `Order`")
p2 <- plot(rf_useFamily_nClimates, main = "Models Involving `Family`")
plot_grid(p1, p2, align = "v", nrow = 2, rel_heights = c(1/2, 1/2))
```

# Flipkart Data Scientist Home Assignment

In this repository you will find all files and responses to the 5 tasks in the Flipkar DS Home Assignment.

The repository contains:
* [data/data.7z](./data/): all the relevant data files zipped so it could be loaded to this repository. Please extract them if you wish to run the notebooks.
* [data_cleaning.py](./data_cleaning.ipynb): a notebook designed to clean and enrich the data sources provided. The notebook accomplishes:
    * converting date values as needed
    * extracting features for color/s, memory, storage, premium level
    * ensuring that the date columns make sense and are not shifted in some way
    * please see notebook for further details on what was done.
* [eda.ipynb](./eda.ipynb): exploratory data analysis. Here I check how product features are linked to prices and # sold. I also explore the auto-regressive correlations between # impressions, # views, and # sold. This notebook laid the foundation to build some intuition to be used in building a prediciton model. Main conclusions are that:
    * age, brand, memory, storage, premium level are all features that clearly affect demand and price, but color is not as useful - at least not in this exercise.
    * the total # of phones sold during the Jan 2024 even can be estimated to within 4% absolute percentage error with some very simple techniques.
    * that # sold can be split probably be split into two time-invariant models (as in, time since added) - an Impulse model when the product is first released, and an auto-regressive model.
    * please see notebook for further details, and the next section for answers to Q1.
* [predict.ipynb](./predict.ipynb): here I implement the intuitions gathered from the EDA section and incorporate it into an XGBoost model. This notebook covers:
    * organizing the data so it can be fed into a tree model, incorporating both product features and historical performance as inputs to a model that should model both the impulse and auto-regressive behavior of phone sales.
    * see the notebook for further details and the next section for answers to Q2-5.

# Responses to Home Assignment

1. Which phone model had the highest interest from users during October’s event? Was it
in fact the event’s highest selling phone? Which one generated the most revenue?
    * **Highest Interest Measured by Views**: Apple iPhone 14 Blue 128 GB
    * **Highest Sold Measured by Units Sold**: realme C53 Champion Gold 128 GB 6 GB RAM
    * **Highest Revenue Measured by Total Revenue**: Apple iPhone 14 Blue 128 GB
2. Estimate the number of units sold during the Jan2024 event, for each product.
    * Using the technique employed in [eda.ipynb](./eda.ipynb), I estimated 1,014,315 total phones sold, with an absolute percent error of 3.96%. I started doing this because I thought it would be easier to get an accurate result - and I was right. And then I could split this number among the different products with another model. I did not take this approach to its end.
    * Instead, I used an XGBoost model that takes in:
        * Product features: brand, memory, storage, premium level.
            * For premium level, I clustered phones within the same brand, memory, storage and that came out at more-or-less the same time by their $\log(price)$.
        * Time-series features: I added a window for the # impressions, average price, and # units sold (except for the current date's # units sold).
            * I aggregate these values by week, and had 4 weeks of lag.
            * Because I used age of product for the windows and not absolute dates, I had to make two prediction for some products during the Jan 24 event, and taking the linear weighted proportion of each prediction based on how much each overlapped with the event. All the details are found in the notebook.
        * This combination was designed to provide all the inputs needed to a more intutive approach, where the # units sold for a product is given by:
            * $ S_t \sim J_t[\theta] + A_t[\theta]$
            * The formula tells us that the sales $S_t$ is composed of an impulse compnent $J_t$ that is the initial demand for a new phone, and $A_t$ the auto-regressive component.
            * see the [eda.ipynb](./eda.ipynb) and [predict.ipynb](./predict.ipynb) for further details on actual implementation and discussion, where I use an XGBoost model in place of having separate $J_t$ and $A_t$ components.
        * The model does not consider competition, and relies on the having the price set prior to the prediction of the number of units sold. Therefore, the problem of competition is taken into consideration by the price input, if we assume that the price-setting process tries to achieve competitiveness.
        * I optimize a distributional loss to get the  $\lambda$ of a Poisson distribution. Distributional regression is particularly suitable when we are trying to reduce absolute error. For this particular problem, the Poisson distribution is effective, because it models the count of events whose time-to-event follows an exponential distribution - like time until the next click on an ad, the next visit to the website, or the next purchase of a phone.
3. How would you evaluate your prediction?
    * Firstly, I used a validation set to decide on overfit criteria. Then I measure the goodness of fit to a Poisson distribution. I used 80th percentile boundaries, and both the validation and Jan event data had 70% of the samples fit inside their predicted 80% boundaries. Usually the validation/test sets are more dispersed, so 70% is quite good.
    * I also measured the MAPE and wMAPE.
4. The model acheived about a 58% wMAPE on predictions, and after doing proportional weightings to fit the Jan24 period, a 68% wMAPE. This is for 2769 different phone, but most of which had 0 sales in the period - however, 0 sales does not affect wMAPE very much. This estimate tells us that most products had very inaccurate predictions.
5. New phones:
    * It's possible to pick similar phones from those already launched and price the new phone more or less to match and adjust by some heuristic from there. , we could use two methods:
    * To predict the sales, simply use the model I implemented, which is designed to accept new products, as well. At the end of [predict.ipynb](./predict.ipynb) we show theses predictions for products launched during the Jan 24 event. 
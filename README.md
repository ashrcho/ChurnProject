# Telecoms Churn Analysis + Dashboard
### 📍 Overview 
In this project I build a predictive churn model for a Telecommunications company using Python and create an interactive Tableau dashboard for intuitive insights.
### 📖 Context 
#### <ins>Churn<ins>
Churn or churn rate according to Investopedia is the "**rate at which customers stop doing business with an entity.**" Understanding how churn affects your business is crucial. Good customer retention strategy starts with understanding: 
- Why customers are leaving
- Which types of customers are leaving
- And at what rate they're leaving

To drive the point further, without a thorough understanding of KPI metrics like Churn Rate you are effectively flying blind. Strategic insights provided via data are the eyes and ears of a business. It only becomes possible to solve and understand a problem by becoming aware of its existance and depth. Current machine learning models provide us useful tools that help us in this endeavor.

#### <ins>Data<ins>

The dataset I use comes from an Iranian telecommunications company and was donated to the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset). It contains information for 3,150 customers across a period of 12 months. The dataset features 14 columns:

1) **Call Failure** - # of call failures
2) **Complains** - complaint (1) or no complaint (0)
3) **Subscription Length** - total months of subscription
4) **Charge Amount** - lowest to highest amount (0-9) 
5) **Seconds of Use** - total seconds of calls
6) **Frequency of Use** - total # of calls
7) **Frequency of SMS** - total # of text messages
8) **Distinct Called Numbers** - total # of distinct phone calls
9) **Age Group** - younger to older age (1-5) 
10) **Tariff Plan** - pay-as-you-go (1) or contractual (2)
11) **Status** - active (1) or inactive (2)
12) **Age** - age of customer
13) **Customer Value** - calculated value of customer
14) **Churn** - whether or not the customer churned in the 9-12 month period (1: churned, 0: remained)

#### <ins>Problem Statement<ins>

Retaining existing customers is more cost-effective than acquiring new ones. With this project my aim is to provide insights that help with customer retention efforts by:
- Understanding churn drivers
- Building a predictive model with Decision-Tree-Classifier 
- Visualizing insights for business decisions as a dashboard in Tableau

### 🔎 Exploratory Data Analysis (EDA) 
#### <ins>Descriptive Statistics<ins>
![DescriptStat1](images/descriptfrthistime.png)
- There are customers with 0 seconds-of-use/frequency-of-use. This could potentially point to customers facing technical issues trying to connect. If we tracked customers with 0 seconds of use their churn rate would most likely be high.
- Would be interesting to see what level of call failure customers are more likely to churn at.
#### <ins>Univariate Analysis<ins>
![Histogram](images/histograms.png)
- Majority of customers experience relatively low call failure.
- Frequency-of-use and Seconds-of-use have a very similar distribution. 
- Age data is pretty odd. Only recorded ages are 15, 25, 30, 45, and 55.

![ChurnBar](images/ChurnBar.png)
- We have an imbalanced target (Churn). One might think to utilize resampling techniques. However, in our case this would be a mistake because we need to be prioritizing interpretability, understanding the key drivers of churn, over pure model optimization.
- Compared to the [U.S. telecommunications industry average churn rate of 21%](https://www.statista.com/statistics/816735/customer-churn-rate-by-industry-us/?__sso_cookie_checker=failed) this companies performing better at 15.7%.   

![ComplainsBar](images/complains.png)
- No Complaint - 2,909 (92.3%)
- Complaint - 241 (7.7%)

![TarrifBar](images/tarrifplan.png)
- Pay-As-You-Go - 2,905 (92.2%)
- Contractual - 245 (7.8%)

![statusbar](images/statusbar.png)
- Active - 2,368 (75.2%)
- Inactive - 782 (24.8%)
#### <ins>Multivariate Analysis<ins>
![heatmap](images/heatmap1.png)
- Churn is the most correlated with Status and Complains at 0.5 and 0.53 respectively.
- As expected Frequency-of-use and Seconds-of-use are highly correlated.
- Age and Age Group is highly correlated.
- Interestingly, Frequency-of-SMS and Customer Value are highly correlated.
### 🤠 Data Wrangling 

# 📺 Netflix Churn Prediction App

This is a machine learning web application built using **Streamlit** to predict whether a Netflix customer is likely to churn (i.e., cancel their subscription). The model is trained on a dataset containing customer demographics, behavior, and subscription details.
🚀 **[Live Demo](https://netflix-customer-churn-prediction.streamlit.app/)** 
---

## 🚀 Features

- Predict churn for any Netflix customer based on input features
- Visual display of churn probability (text + progress bar or pie chart)
- Preprocessing includes feature scaling and encoding
- Easy-to-use web interface built with Streamlit

---

## 📊 Dataset Features
[DATASET LINK](http://kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset)
The dataset includes the following columns:

- `customer_id`
- `age`
- `gender`
- `subscription_type`
- `watch_hours`
- `last_login_days`
- `region`
- `device`
- `monthly_fee`
- `churned` *(target label)*
- `payment_method`
- `number_of_profiles`
- `avg_watch_time_per_day`
- `favorite_genre`

---


## 🧠 Model Details

- Preprocessing: Label Encoding / One-hot encoding + Feature Scaling
- Model: RandomForestClassifier
- Performance: ~98% accuracy on test set

---


# Brent-oil-price-analysis

## Business Objective

The primary goal of this analysis is to study how significant events impact Brent oil prices. This project focuses on identifying correlations between price changes and major events such as political decisions, conflicts in oil-producing regions, global economic sanctions, and changes in OPEC policies. The insights generated aim to assist investors, analysts, and policymakers in understanding and reacting to oil price fluctuations effectively.

## Situational Overview

Birhan Energies is a leading consultancy firm specializing in providing data-driven insights and strategic advice to stakeholders in the energy sector. This project aims to analyze how major political and economic events affect Brent oil prices to guide investment strategies, policy development, and operational planning.

## Data

The dataset consists of historical Brent oil prices, covering daily prices from May 20, 1987, to September 30, 2022. The dataset includes the following fields:

- **Date**: The date of the recorded Brent oil price (formatted as `day-month-year`, e.g., `20-May-87`).
- **Price**: The price of Brent oil on the corresponding date (in USD per barrel).

## Objectives

### Global Business Objective

- Understand how political decisions, conflicts, international sanctions, and OPEC policy changes affect Brent oil prices.

### Sub-objectives

1. Define the data analysis workflow.
2. Understand the model and data.
3. Extract statistically valid insights related to the business objective.

## Tasks

### Task 1: Defining the Data Analysis Workflow and Understanding the Model and Data

- Outline the steps and processes for analyzing Brent oil prices data.
- Understand data generation, sampling, and compilation.
- Identify model inputs, parameters, and outputs.
- State assumptions and limitations of the analysis.
- Determine communication channels and formats for stakeholders.

### Task 2: Analyzing Brent Oil Prices

- Apply knowledge from Task 1 to analyze historical Brent oil prices data.
- Utilize additional statistical and econometric models (e.g., VAR, Markov-Switching ARIMA, LSTM).
- Explore factors influencing oil prices:
  - Economic Indicators (GDP, inflation, unemployment rates, exchange rates).
  - Technological Changes (extraction technologies, renewable energy developments).
  - Political and Regulatory Factors (environmental regulations, trade policies).

### Task 3: Developing an Interactive Dashboard for Data Analysis Results

- Build a dashboard application using Flask (backend) and React (frontend) to visualize analysis results.
- **Key Components**:
  - **Backend (Flask)**: Develop APIs to serve data, handle requests, and integrate data sources for real-time updates.
  - **Frontend (React)**: Create a user-friendly interface with interactive visualizations, filters, and comparisons.
  - **Key Features**:
    - Present historical trends, forecasts, and event correlations.
    - Allow users to explore how specific events influenced Brent oil prices.
    - Display key indicators like volatility and model accuracy metrics.

## Suggested Approach

1. **Data Collection**: Gather datasets on economic indicators, technological changes, and political factors.
2. **Data Preprocessing**: Clean and preprocess data to ensure consistency and accuracy.
3. **Exploratory Data Analysis (EDA)**: Identify patterns, trends, and relationships in the data.
4. **Model Building**: Develop multiple models (time series, econometric, and machine learning).
5. **Model Evaluation**: Use performance metrics (RMSE, MAE, R-squared) to assess models.
6. **Insight Generation**: Interpret model outputs to provide actionable insights.


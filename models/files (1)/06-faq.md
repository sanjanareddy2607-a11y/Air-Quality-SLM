# Air Quality API FAQ

Source: https://developers.google.com/maps/documentation/air-quality/faq (last updated 2026-09-17 UTC)

## Is there a rate limit on the API?

For the Air Quality API, there is a default rate limit of 6,000 queries per minute.

## What is an Air Quality Index (AQI)?

An air quality index (AQI) is the scale by which government agencies and other providers communicate to the public how polluted the air is or how polluted it is forecasted to be. Many AQIs are related to health effects and some have health recommendations for each AQI category.

There are many different AQIs globally, each may be calculated using different pollutants, different pollutant concentration thresholds, different averaging time periods and expressed on different scales, categories and colors.

The Air Quality API includes [over 70 air quality indexes](/maps/documentation/air-quality/laqis#supported_aqis) in [over 100 countries](/maps/documentation/air-quality/laqis#supported_countries_and_their_relevant_aqis).

## What are pollutants?

Air pollutants are particles and gasses in the air that can be harmful to people's health, especially at elevated concentrations. The Air Quality API provides information about the [most common pollutants](/maps/documentation/air-quality/pollutants#reported_pollutants), their concentration, emission sources and effects.

## Are the calculations accurate?

The underlying air quality model calculating the AQIs is a multi-layered model that uses various types of information and data as input. The overall accuracy of the model is validated and monitored by comparing the model outputs to government monitoring stations (these stations represent the ground truth regarding air quality at their precise location). This is performed by running a continuous cross-validation process using the Leave One Out Method: eliminating the input data from one sensor from the model calculations and then comparing the result to the actual station's reading to see how closely they match.

For more information about how Google calculates AQI, see [Air quality on Google](https://support.google.com/maps/answer/11270845).

## How often is air quality data updated?

Air Quality heatmap tiles are refreshed every hour in sync with current conditions data. Updates are typically published shortly after the hour.

---

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2026-09-17 UTC.

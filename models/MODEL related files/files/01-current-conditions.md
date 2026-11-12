# Current conditions | Air Quality API

Source: https://developers.google.com/maps/documentation/air-quality/current-conditions (last updated 2026-09-17 UTC)

The [currentConditions](/maps/documentation/air-quality/reference/rest/v1/currentConditions/lookup) endpoint provides current hourly air quality information. You can control which AQIs and pertinent information are returned (such as pollutant concentration levels and health recommendations).

You can request current hourly air quality using the `currentConditions` endpoint by sending an HTTP POST request to:

```
https://airquality.googleapis.com/v1/currentConditions:lookup?key=YOUR_API_KEY
```

Include your request options in the JSON request body. The request body contains the location for which you would like to get the air quality current conditions and various options to control what air quality information to include in the response.

The APIs Explorer lets you make live requests so that you can get familiar with the API and the API options: [Try it!](#try-it)

## Example of basic request

### Basic request body

The following code shows how to construct a basic request body for a `currentConditions` request. In this example, you set only the location.

```bash
curl -X POST -d '{
  "location": {
    "latitude": 37.419734,
    "longitude": -122.0827784
  }
}' \
-H 'Content-Type: application/json' \
'https://airquality.googleapis.com/v1/currentConditions:lookup?key=YOUR_API_KEY'
```

### Basic request response

The call above generates the following JSON response. For more details about the response data, see [Response data](/maps/documentation/air-quality/response-data).

> **Note:** By default, the response includes the Universal AQI with the basic AQI information. The AQI can be removed or replaced with a different AQI by setting the corresponding parameters.

> **Note:** If a color field value is 0, that field is omitted from the response.

```json
{
  "dateTime": "2023-10-26T18:00:00Z",
  "regionCode": "us",
  "indexes": [
    {
      "code": "uaqi",
      "displayName": "Universal AQI",
      "aqi": 52,
      "aqiDisplay": "52",
      "color": {
        "red": 0.9490196,
        "green": 0.98039216,
        "blue": 0.019607844
      },
      "category": "Moderate air quality",
      "dominantPollutant": "no2"
    }
  ]
}
```

## Example of request with multiple parameters

### Multiple parameters request

The following code shows how to construct a request body for a `currentConditions` request. In this example, in addition to the location, several extra computations and the response language are specified.

```bash
curl -X POST -d '{
  "universalAqi": true,
  "location": {
    "latitude": 37.419734,
    "longitude": -122.0827784
  },
  "extraComputations": [
    "HEALTH_RECOMMENDATIONS",
    "DOMINANT_POLLUTANT_CONCENTRATION",
    "POLLUTANT_CONCENTRATION",
    "LOCAL_AQI",
    "POLLUTANT_ADDITIONAL_INFO"
  ],
  "languageCode": "en"
}' \
-H 'Content-Type: application/json' \
'https://airquality.googleapis.com/v1/currentConditions:lookup?key=YOUR_API_KEY'
```

### Multiple parameters response

The call above generates the following JSON response. For more details about the response data, see [Response data](/maps/documentation/air-quality/response-data).

```json
{
  "dateTime": "2023-06-15T10:00:00Z",
  "regionCode": "us",
  "indexes": [
    {
      "code": "uaqi",
      "displayName": "Universal AQI",
      "aqi": 89,
      "aqiDisplay": "89",
      "color": {
        "red": 48,
        "green": 175,
        "blue": 55,
        "alpha": 255
      },
      "category": "Excellent air quality",
      "dominantPollutant": "o3"
    },
    {
      "code": "usa_epa",
      "displayName": "AQI (US)",
      "aqi": 41,
      "aqiDisplay": "41",
      "color": {
        "green": 228,
        "alpha": 255
      },
      "category": "Good air quality",
      "dominantPollutant": "pm25"
    }
  ],
  "pollutants": [
    {
      "code": "co",
      "displayName": "CO",
      "fullName": "Carbon monoxide",
      "concentration": {
        "value": 334.24,
        "units": "PARTS_PER_BILLION"
      },
      "additionalInfo": {
        "sources": "Typically originates from incomplete combustion..",
        "effects": "When inhaled, carbon monoxide can prevent the bloo"
      }
    },
    {
      "code": "no2",
      "displayName": "NO2",
      "fullName": "Nitrogen dioxide",
      "concentration": {
        "value": 13.12,
        "units": "PARTS_PER_BILLION"
      },
      "additionalInfo": {
        "sources": "Main sources are fuel burning processes, such as.",
        "effects": "Exposure may cause increased bronchial reactivity"
      }
    },
    {
      "code": "o3",
      "displayName": "O3",
      "fullName": "Ozone",
      "concentration": {
        "value": 13.88,
        "units": "PARTS_PER_BILLION"
      },
      "additionalInfo": {
        "sources": "Ozone is created in a chemical reaction between at",
        "effects": "Ozone can irritate the airways and cause coughing,"
      }
    },
    {
      "code": "pm10",
      "displayName": "PM10",
      "fullName": "Inhalable particulate matter (<10µm)",
      "concentration": {
        "value": 5.07,
        "units": "MICROGRAMS_PER_CUBIC_METER"
      },
      "additionalInfo": {
        "sources": "Main sources are combustion processes (e.g. indoor",
        "effects": "Inhalable particles can penetrate into the lungs."
      }
    },
    {
      "code": "pm25",
      "displayName": "PM2.5",
      "fullName": "Fine particulate matter (<2.5µm)",
      "concentration": {
        "value": 2.83,
        "units": "MICROGRAMS_PER_CUBIC_METER"
      },
      "additionalInfo": {
        "sources": "Main sources are combustion processes (e.g. power",
        "effects": "Fine particles can penetrate into the lungs and bl"
      }
    },
    {
      "code": "so2",
      "displayName": "SO2",
      "fullName": "Sulfur dioxide",
      "concentration": {
        "value": 0.02,
        "units": "PARTS_PER_BILLION"
      },
      "additionalInfo": {
        "sources": "Main sources are burning processes of sulfur-conta",
        "effects": "Exposure causes irritation of the respiratory trac"
      }
    }
  ],
  "healthRecommendations": {
    "generalPopulation": "With this level of air quality, you have no limi",
    "elderly": "If you start to feel respiratory discomfort such as coughi",
    "lungDiseasePopulation": "...",
    "heartDiseasePopulation": "...",
    "athletes": "...",
    "pregnantWomen": "...",
    "children": "..."
  }
}
```

*(Some string values are truncated in the source PDF.)*

## Try it!

The APIs Explorer lets you make sample requests so that you can get familiar with the API and the API options.

1. Select the API icon on the right side of the page.
2. Optionally edit the request parameters.
3. Select the **Execute** button. In the dialog, choose the account that you want to use to make the request.
4. In the APIs Explorer panel, select the fullscreen icon to expand the APIs Explorer window.

[Open API Explorer]

---

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2026-09-17 UTC.

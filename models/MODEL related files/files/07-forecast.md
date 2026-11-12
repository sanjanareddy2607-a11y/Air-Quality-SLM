# Forecast | Air Quality API

Source: https://developers.google.com/maps/documentation/air-quality/forecast (last updated 2026-09-17 UTC)

The [forecast](/maps/documentation/air-quality/reference/rest/v1/forecast/lookup) endpoint provides up to 96 hours (4 days) of hourly air quality forecast information for a given location. Each forecast includes hourly air quality indexes as well as additional information:

- Specific time and date or time period for forecast
- Language code for translated fields
- Local or custom air quality index (AQI)
- Health recommendations
- Dominant pollutants
- Pollutant concentration
- Additional information about pollutants
- Response page size
- Color palette

The APIs Explorer lets you make live requests so that you can get familiar with the API and the API options: [Try it!](#try-it)

To get air quality forecasts, set the `dateTime` parameter to a future date and time, as shown in the [basic request example](/maps/documentation/air-quality/forecast#example_of_basic_request) below.

## About forecast requests

You can request hourly air quality forecast information using the `forecast` endpoint by sending an HTTP POST request to:

```
https://airquality.googleapis.com/v1/forecast:lookup?key=YOUR_API_KEY
```

Include your [parameters](/maps/documentation/air-quality/reference/rest/v1/forecast/lookup#request-body) in the request body. The request body contains the location for which you would like to get air quality hourly forecast information and various options to control what additional information to include in the response.

## About forecast responses

The response body provides the UTC date and time for which to display air quality forecast information, air quality indexes, and additional customized information. The response body also includes a token to retrieve the next page if the information does not fit within the requested page size.

## Example of basic request

The following code shows how to construct a forecast request. In this example, you specify the location, date and time (beginning from the next hour up to 96 hours) for which to show hourly air quality forecast information.

```bash
curl -X POST -d '{
  "location": {
    "latitude": 37.4125333,
    "longitude": -122.0840937
  },
  "dateTime": "2024-02-09T08:00:00Z"
}' \
-H "Content-Type: application/json" \
-H "Accept-Language: *" \
"https://airquality.googleapis.com/v1/forecast:lookup?key=YOUR_API_KEY"
```

The call generates the following JSON response.

> **Note:** If a color field value is 0, that field is omitted from the response.

```json
{
  "hourlyForecasts": [
    {
      "dateTime": "2024-02-09T08:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 80,
          "aqiDisplay": "80",
          "color": {
            "red": 0.34509805,
            "green": 0.74509805,
            "blue": 0.20784314
          },
          "category": "Excellent air quality",
          "dominantPollutant": "pm25"
        }
      ]
    }
  ],
  "regionCode": "us"
}
```

## Example of request with multiple parameters

The following code shows how to construct a request body with multiple parameters. In this example, you specify the following:

- Location
- Time period for which to display air quality forecast information
- Language code
- Extra computations (such as health recommendations, the dominant pollutant concentration, and additional pollutant information)
- Color palette

```bash
curl -X POST -d '{
  "pageSize": "10",
  "universalAqi": "true",
  "location": {
    "latitude": "40.741895",
    "longitude": "-73.989308"
  },
  "period": {
    "startTime": "2024-02-27T13:58:14+02:00",
    "endTime": "2024-02-28T16:58:14+02:00"
  },
  "languageCode": "en",
  "extraComputations": [
    "HEALTH_RECOMMENDATIONS",
    "DOMINANT_POLLUTANT_CONCENTRATION",
    "POLLUTANT_ADDITIONAL_INFO"
  ],
  "uaqiColorPalette": "RED_GREEN"
}' \
-H "Content-Type: application/json" \
-H "Accept-Language: *" \
"https://airquality.googleapis.com/v1/forecast:lookup?key=YOUR_API_KEY"
```

The call generates the following JSON response.

```json
{
  "hourlyForecasts": [
    {
      "dateTime": "2024-02-27T11:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 75,
          "aqiDisplay": "75",
          "color": {
            "red": 0.43137255,
            "green": 0.7764706,
            "blue": 0.20392157
          },
          "category": "Good air quality",
          "dominantPollutant": "pm25"
        }
      ],
      "pollutants": [
        {
          "code": "pm25",
          "displayName": "PM2.5",
          "fullName": "Fine particulate matter (<2.5µm)",
          "concentration": {
            "value": 15.1,
            "units": "MICROGRAMS_PER_CUBIC_METER"
          },
          "additionalInfo": {
            "sources": "...",
            "effects": "..."
          }
        }
      ],
      "healthRecommendations": {
        "generalPopulation": "...",
        "lungDiseasePopulation": "...",
        "heartDiseasePopulation": "...",
        "athletes": "...",
        "pregnantWomen": "..."
      }
    },
    {
      "dateTime": "2024-02-27T12:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 67,
          "aqiDisplay": "67",
          "color": {
            "red": 0.5882353,
            "green": 0.8392157,
            "blue": 0.16862746
          },
          "category": "Good air quality",
          "dominantPollutant": "pm25"
        }
      ],
      "pollutants": [
        {
          "code": "pm25",
          "displayName": "PM2.5",
          "fullName": "Fine particulate matter (<2.5µm)",
          "concentration": {
            "value": 19.96,
            "units": "MICROGRAMS_PER_CUBIC_METER"
          },
          "additionalInfo": {
            "sources": "...",
            "effects": "..."
          }
        }
      ],
      "healthRecommendations": {
        "generalPopulation": "...",
        "lungDiseasePopulation": "...",
        "heartDiseasePopulation": "...",
        "athletes": "...",
        "pregnantWomen": "...",
        "children": "..."
      }
    }
    /.../
  ],
  "regionCode": "us",
  "nextPageToken": "CjMKEglwCFVq9l5EQBH0aoDSUH9SwBoQCgYI1pDirgYSBgiGiOiuBiABKg..."
}
```

*(The `nextPageToken` is truncated in the source PDF.)*

To view the next page of forecast information, pass in the `nextPageToken` returned at the end of the response body:

```bash
curl -X POST -d '{
  "universalAqi": "true",
  "location": {
    "latitude": "40.741895",
    "longitude": "-73.989308"
  },
  "period": {
    "startTime": "2024-02-27T13:58:14+02:00",
    "endTime": "2024-02-28T16:58:14+02:00"
  },
  "languageCode": "en",
  "extraComputations": [
    "HEALTH_RECOMMENDATIONS",
    "DOMINANT_POLLUTANT_CONCENTRATION",
    "POLLUTANT_ADDITIONAL_INFO"
  ],
  "uaqiColorPalette": "RED_GREEN",
  "pageToken": "CjMKEglwCFVq9l5EQBH0aoDSUH9SwBoQCgYI1pDirgYSBgiGiOiuBiABKgMC..."
}' \
-H "Content-Type: application/json" \
-H "Accept-Language: *" \
"https://airquality.googleapis.com/v1/forecast:lookup?key=YOUR_API_KEY"
```

This returns the next page of forecast information:

```json
{
  "hourlyForecasts": [
    {
      "dateTime": "2024-02-27T21:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 74,
          "aqiDisplay": "74",
          "color": {
            "red": 0.44705883,
            "green": 0.78431374,
            "blue": 0.2
          },
          "category": "Good air quality",
          "dominantPollutant": "o3"
        }
      ],
      "pollutants": [
        {
          "code": "o3",
          "displayName": "O3",
          "fullName": "Ozone",
          "concentration": {
            "value": 32.07,
            "units": "PARTS_PER_BILLION"
          },
          "additionalInfo": {
            "sources": "...",
            "effects": "..."
          }
        }
      ],
      "healthRecommendations": {
        "generalPopulation": "...",
        "elderly": "...",
        "lungDiseasePopulation": "...",
        "heartDiseasePopulation": "...",
        "athletes": "...",
        "pregnantWomen": "...",
        "children": "..."
      }
    },
    {
      "dateTime": "2024-02-27T22:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 75,
          "aqiDisplay": "75",
          "color": {
            "red": 0.43137255,
            "green": 0.7764706,
            "blue": 0.20392157
          },
          "category": "Good air quality",
          "dominantPollutant": "o3"
        }
      ],
      "pollutants": [
        {
          "code": "o3",
          "displayName": "O3",
          "fullName": "Ozone",
          "concentration": {
            "value": 30.94,
            "units": "PARTS_PER_BILLION"
          },
          "additionalInfo": {
            "sources": "...",
            "effects": "..."
          }
        }
      ],
      "healthRecommendations": {
        "generalPopulation": "...",
        "lungDiseasePopulation": "...",
        "heartDiseasePopulation": "...",
        "athletes": "...",
        "pregnantWomen": "...",
        "children": "..."
      }
    }
    /.../
  ],
  "regionCode": "us"
}
```

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

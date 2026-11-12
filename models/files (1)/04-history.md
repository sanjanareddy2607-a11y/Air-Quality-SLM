# History | Air Quality API

Source: https://developers.google.com/maps/documentation/air-quality/history (last updated 2026-09-17 UTC)

The [history](/maps/documentation/air-quality/reference/rest/v1/history/lookup) endpoint provides historical hourly air quality information for a specific location, for a given time range, up to a maximum of 30 days. You can control which AQIs and pertinent information are returned (such as pollutant concentration level and health recommendations).

You can request history of hourly air quality using the `history` endpoint by sending an HTTP POST request to:

```
https://airquality.googleapis.com/v1/history:lookup?key=YOUR_API_KEY
```

Include your request options in the JSON request body. The request body contains the location and the time range for which you would like to get the hourly air quality history. It may also include various options to control what air quality information to include in the response.

The APIs Explorer lets you make live requests so that you can get familiar with the API and the API options: [Try it!](#try-it)

## Example of a single hour request

### Single hour request body

The following code shows how to construct a request body for a single hour history request using the `history` method. In this example, you set the location and the past timestamp.

> **Note:** Ensure that the `dateTime` parameter specifies a time period within the past 30 days.

```bash
curl -X POST -d '{
  "dateTime": "2023-06-26T15:01:23Z",
  "location": {
    "latitude": 37.419734,
    "longitude": -122.0827784
  }
}' \
-H 'Content-Type: application/json' \
'https://airquality.googleapis.com/v1/history:lookup?key=YOUR_API_KEY'
```

### Single hour response

The call above generates the following JSON response. For more details about the response data, see [Response data](/maps/documentation/air-quality/response-data).

> **Note:** By default, the response includes the Universal AQI with the basic AQI information. The AQI can be removed or replaced with a different AQI by setting the corresponding parameter(s).

> **Note:** If a color field value is 0, that field is omitted from the response.

```json
{
  "hoursInfo": [
    {
      "dateTime": "2023-06-26T15:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 73,
          "aqiDisplay": "73",
          "color": {
            "red": 118,
            "green": 202,
            "blue": 51,
            "alpha": 255
          },
          "category": "Good air quality",
          "dominantPollutant": "pm10"
        }
      ]
    }
  ],
  "regionCode": "us"
}
```

## Example of time range request

### Time range request

The following code shows how to construct a request body for a `history` request which returns multiple records of historical hourly air quality.

You can request multiple records by specifying a time range, meaning a start and end timestamp, or by specifying the number of hours back from the current time.

> **Note:** Results are returned in pages, with the most recent results returned in the first page. Use the `pageSize` property to control the maximum number of results per page. If more than one page is required to retrieve all results, then each page includes the `nextPageToken` property that contains the token you use to retrieve the subsequent page.

In the example below, because you set the page size to 2, your request for four hours of air quality history is returned in two pages, where each page contains two hours of data.

```bash
curl -X POST -d '{
  "hours": 4,
  "pageSize": 2,
  "pageToken": "",
  "location": {
    "latitude": 37.419734,
    "longitude": -122.0827784
  }
}' \
-H 'Content-Type: application/json' \
'https://airquality.googleapis.com/v1/history:lookup?key=YOUR_API_KEY'
```

You can make a similar request by passing the start and end timestamps:

> **Note:** Ensure that the `startTime` and `endTime` parameters specify a time period within the past 30 days.

```bash
curl -X POST -d '{
  "period": {
    "startTime": "2023-06-15T08:00:00Z",
    "endTime": "2023-06-15T12:00:00Z"
  },
  "pageSize": 2,
  "pageToken": "",
  "location": {
    "latitude": 37.419734,
    "longitude": -122.0827784
  }
}' \
-H 'Content-Type: application/json' \
'https://airquality.googleapis.com/v1/history:lookup?key=YOUR_API_KEY'
```

### Time range response

The calls above generate a JSON response in the form below. For more details about the response data, see [Response data](/maps/documentation/air-quality/response-data).

Both calls above request four hours of air quality data. However, because you set the `pageSize` property to `2` in the request, the response only includes results for the most recent two hours.

```json
{
  "hoursInfo": [
    {
      "dateTime": "2023-06-15T11:00:00Z",
      "indexes": [
        {
          "code": "uaqi",
          "displayName": "Universal AQI",
          "aqi": 83,
          "aqiDisplay": "83",
          "color": {
            "red": 74,
            "green": 185,
            "blue": 54,
            "alpha": 255
          },
          "category": "Excellent air quality",
          "dominantPollutant": "o3"
        }
      ]
    },
    {
      "dateTime": "2023-06-15T10:00:00Z",
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
        }
      ]
    }
  ],
  "regionCode": "us",
  "nextPageToken": "ChYaEgl3gv3XubVCQBEsNMY9TTdUMTE6MDA6MDA"
}
```

Notice that the response also includes the `nextPageToken` property. Use this property to access the next page of the results, which contains the next two hours of data.

To access the next page of results, make a second request to the `history` endpoint, but this time set the `pageToken` property to the value of `nextPageToken` from the first response.

> **Note:** When passing the `pageToken` property to access subsequent data pages, all other request parameters must match the original call that provided the page token.

```bash
curl -X POST -d '{
  "hours": 4,
  "pageSize": 2,
  "pageToken": "ChYaEgl3gv3XubVCQBEsNMY9TTdUMTE6MDA6MDA",
  "location": {
    "latitude": 37.419734,
    "longitude": -122.0827784
  }
}' \
-H 'Content-Type: application/json' \
'https://airquality.googleapis.com/v1/history:lookup?key=YOUR_API_KEY'
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

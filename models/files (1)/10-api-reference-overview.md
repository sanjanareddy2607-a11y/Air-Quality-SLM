# Air Quality API (REST reference)

Source: https://developers.google.com/maps/documentation/air-quality/reference/rest (last updated 2025-08-27 UTC)

The Air Quality API.

**Service:** `airquality.googleapis.com`

To call this service, we recommend that you use the Google-provided [client libraries](https://cloud.google.com/apis/docs/client-libraries-explained). If your application needs to use your own libraries to call this service, use the following information when you make the API requests.

## Service endpoint

A [service endpoint](https://cloud.google.com/apis/design/glossary#api_service_endpoint) is a base URL that specifies the network address of an API service. One service might have multiple service endpoints. This service has the following service endpoint and all URIs below are relative to this service endpoint:

```
https://airquality.googleapis.com
```

## REST Resource: [v1.currentConditions](/maps/documentation/air-quality/reference/rest/v1/currentConditions)

| Method | HTTP request | Description |
|---|---|---|
| [lookup](/maps/documentation/air-quality/reference/rest/v1/currentConditions/lookup) | `POST /v1/currentConditions:lookup` | The Current Conditions endpoint provides hourly air quality information in more than 100 countries, up to a 500 x 500 meters resolution. |

## REST Resource: [v1.forecast](/maps/documentation/air-quality/reference/rest/v1/forecast)

| Method | HTTP request | Description |
|---|---|---|
| [lookup](/maps/documentation/air-quality/reference/rest/v1/forecast/lookup) | `POST /v1/forecast:lookup` | Returns air quality forecast for a specific location for a given time range. |

## REST Resource: [v1.history](/maps/documentation/air-quality/reference/rest/v1/history)

| Method | HTTP request | Description |
|---|---|---|
| [lookup](/maps/documentation/air-quality/reference/rest/v1/history/lookup) | `POST /v1/history:lookup` | Returns air quality history for a specific location for a given time range. |

## REST Resource: [v1.mapTypes.heatmapTiles](/maps/documentation/air-quality/reference/rest/v1/mapTypes.heatmapTiles)

| Method | HTTP request | Description |
|---|---|---|
| [lookupHeatmapTile](/maps/documentation/air-quality/reference/rest/v1/mapTypes.heatmapTiles/lookupHeatmapTile) | `GET /v1/mapTypes/{mapType}/heatmapTiles/{zoom}/{x}/{y}` | Returns a bytes array containing the data of the tile PNG image. |

---

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-08-27 UTC.

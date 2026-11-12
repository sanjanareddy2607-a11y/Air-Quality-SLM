# Air quality for places | Google Maps Platform

Source: https://developers.google.com/maps/architecture/air-quality-awareness-for-places (last updated 2026-09-17 UTC)

In this document, you will look into **how to use [Air Quality API](/maps/documentation/air-quality/overview) to build rich place search experiences.** We will show how air quality can be a signal of value for users from the very beginning until the end of the funnel and how Air Quality API data can be embedded into existing experiences.

*[Figure: Map of Paris with place pins, each labelled with an AQI value (e.g. 46, 52, 56, 80, 82, 120) and a color dot; a popup for "Tower Bridge Kitchen" (4.4 stars, 374 reviews, Mediterranean restaurant) shows "Good air quality 49".]*

## Use Cases and added value

Air Quality API signals can fit into various search experiences such as when users are looking for a location to visit. Air Quality API offers:

- **Real time\*** and historical data - fresh, accurate data that represents the current state as well as back in time data within a 30 days window in an hourly resolution. (\* an hourly snapshot of outdoor air quality)
- **Global and local data** - covering entire areas in 100+ countries yet highly localized (up to 500m) Air Quality API data provides key signals for users to compare and make decisions on the most appropriate place.

*[Figure: Heatmap of Paris with a 2 km scale bar. Caption: "Air quality can be drastically different even for near locations".]*

## Sample Walkthrough

### Show air quality data early in the experience

There are two main ways to show the Air Quality API data:

- [Current conditions](/maps/documentation/air-quality/current-conditions) / [hourly](/maps/documentation/air-quality/history): get individual values to display against each location. (go to section: [Display Air Quality API data on location pins](#display-air-quality-api-data-on-location-pins))
- [Heatmap tiles](/maps/documentation/air-quality/heatmaps): overlay a colored layer representing current outdoor air quality conditions on top of your map. Country specific layers indexes are available. (go to section: [Display Air Quality API heatmap layer on a map](#display-air-quality-api-heatmap-layer-on-a-map))

### Display Air Quality API data on location pins

- Choose your location index: Universal AQI or a Local AQI (up to 70 different [Air Quality Indexes (AQI)](/maps/documentation/air-quality/laqis) are available). An air quality index (AQI) is the scale by which a certain country, for example, the [U.S. EPA](https://www.epa.gov/), categorizes different air quality levels, based on different calculation methods, data sources, and designated purposes.
- Start visualizing the main Air Quality API value and color code straight from the place search experience.

  *[Figure: Paris map with colored dots only on each pin. Caption: "default display could include the color code only."]*

- Display more [Air Quality API information](https://developers.google.com/maps/documentation/air-quality/response-data) on pin selection:
  - Universal AQI / Local AQI meaning.
  - [Pollutant levels](/maps/documentation/air-quality/pollutants), which are a measure of particles and gasses in the air: Dominant pollutant, Concentration, Sources & Effects.
  - [Health recommendations](/maps/documentation/air-quality/health-recommendations), which are recommended actions for current air quality conditions.

> **Tip:** leverage Advanced Markers to build custom markers and manage markers collision.

- You can adjust the level of information displayed by default to serve your needs. Keep in mind that data points for a location are often useful in comparison to other locations.

*[Figure: Place card for "Tower Bridge Kitchen" — 4.4 stars (374), Mediterranean restaurant — with a "Good air quality 49" badge, dominant pollutant chips "pm10" and "o3", the health message "With this level of air quality, you have no limitations. Enjoy the outdoors!", and a review by "John traveler". Caption: "Air quality section display example".]*

#### Air Quality Response

```json
{
  "dateTime": "2023-08-09T10:00:00Z",
  "regionCode": "fr",
  "indexes": [
    {
      "code": "uaqi",
      "displayName": "Universal AQI",
      "aqi": 76,
      "aqiDisplay": "76",
      "color": {
        "red": 0.4117647,
        "green": 0.77254903,
        "blue": 0.20392157
      },
      "category": "Good air quality",
      "dominantPollutant": "pm10"
    },
    …
  ],
  "pollutants": [
    {
      "code": "co",
      "displayName": "CO",
      "fullName": "Carbon monoxide",
      "concentration": {
        "value": 95.18,
        "units": "PARTS_PER_BILLION"
      }...
    },
    …
  ],
  "healthRecommendations": {
    "generalPopulation": "With this level of air quality, you have no lim...",
    "elderly": "If you start to feel respiratory discomfort such as cough...",
    "lungDiseasePopulation": …
  }
}
```

- Alternatively, you can query Air Quality data only on interaction with a location.

  *[Figure: Paris map with a single "Tower Bridge Kitchen" popup showing "Good air quality 49" after the pin is selected.]*

You can use Air Quality API index to filter nearby places, e.g. if you are looking for parks in your area, you will show only parks with good air quality.

> **Tip:** to use the color code from the Air Quality API response in JavaScript, you can convert the response as described below. The individual colors (red, green, blue, alpha) might not always be part of the response.

Air Quality Response:

```json
...
"color":{
  "red": 0.9490196,
  "green": 0.98039216,
  "blue": 0.019607844
}
...
```

Convert to RGB:

```javascript
var red = parseInt(colorResponse.red*255)|| 1;
var green = parseInt(colorResponse.green*255)|| 1;
var blue = parseInt(colorResponse.blue*255)|| 1;
// --> output rgb(241,250,5)
```

### Display Air Quality API heatmap layer on a map

Heatmap colored tiles are a great way:

- to provide a large area overview.
- to quickly show nuances within that area.

*[Figure: Paris map with a green/yellow heatmap layer and place pins on top.]*

Use the [Maps JavaScript API](/maps/documentation/javascript/usage-and-billing#pricing-for-product) with the [ImageMapType](/maps/documentation/javascript/maptypes#ImageMapTypes) to display the individual tiles needed to cover the entire viewport of your Google map. Heatmap tiles will refresh as users engage with the map by panning, zooming or changing location.

*[Figure: Full-width heatmap of Paris and surroundings.]*

> **Tip:** if you want to display the basemap points of interest and markers on top of the map, leverage [GoogleMapsOverlay](https://deck.gl/docs/api-reference/google-maps/google-maps-overlay) with [deck.gl](https://deck.gl/docs/api-reference/google-maps/google-maps-overlay) and create a new [TileLayer](https://deck.gl/docs/api-reference/geo-layers/tile-layer).

Keep in mind the allowed [zoom level](/maps/documentation/air-quality/heatmaps#about_the_heatmap_endpoint) values are up to 16. The Air Quality API heatmap tile layer might disappear at hyperlocal search.

*[Figure: Side-by-side comparison — Zoom 13 shows the heatmap; Zoom 16.5 shows only the base map with no heatmap.]*

You can select [Heatmap map Types](/maps/documentation/air-quality/reference/rest/v1/mapTypes.heatmapTiles/lookupHeatmapTile#maptype) within a limited set of Air Quality Indexes:

- `UAQI_RED_GREEN` - Universal Air Quality Index red-green palette.
- `UAQI_INDIGO_PERSIAN` - Universal Air Quality Index indigo-persian palette.
- `PM25_INDIGO_PERSIAN` - PM2.5 index indigo-persian palette.
- `GBR_DEFRA` - Daily Air Quality Index (UK) color palette.
- `DEU_UBA` - German Local Air Quality Index color palette.
- `CAN_EC` - Canadian Air Quality Health Index color palette.
- `FRA_ATMO` - France Air Quality Index color palette.
- `US_AQI` - US Air Quality Index color palette.

Make sure you indicate the color scale for users to understand the value and its color ranking:

- gradient scale UAQI (0-100): Poor / Excellent

  *[Figure: horizontal gradient bar from purple (poor) through red, orange, yellow, green to white (excellent).]*

  Color palette CSS code snippet:

  ```css
  .gradient-scale {
  background: linear-gradient(to right,
      rgb(99, 20, 161) 0%, /* purple */
      rgb(149, 0, 25) 20%, /* red */
      rgb(248, 47, 21) 40%, /* orange */
      rgb(255, 248, 35) 60%, /* yellow */
      rgb(34,163,120) 80%, /* steelblue */
      rgb(255,255,255) 100% /* white */
     );
  }
  ```

- solid colors to represent key points in UAQI (0-100) scale: Poor / Excellent

  *[Figure: five solid color swatches — purple, orange, yellow-tan, green, teal.]*

## Queries count and cost associated

- [Maps JavaScript API](/maps/documentation/javascript/usage-and-billing#pricing-for-product): 1 map at load of the experience.
- [Air Quality API](/documentation/air-quality/usage-and-billing):
  - 1 query every place (all on map load or only on user selection).
  - Heatmap tile layer: every tile image is 256*256px and to cover a standard map presented on a mobile device or web you need 12-14 tiles. Each tile pulled is counted as 1 SKU unit.
- [Nearby Search (New) API](/maps/documentation/places/web-service/search-nearby-new): 1 query every 20 places displayed. Different billing according to [Place data](/maps/documentation/places/web-service/usage-and-billing#data-skus) that are part of the query response.

## Conclusion

We showed that Air Quality API can be an important data point for users to decide on a place to visit. Therefore, the Air Quality data should be visible the earliest possible in the search experience with care to the amount of information displayed at once and how it is made visible.

## Next Actions

Suggested Further Reading:

- [Improve local discovery with Nearby Search (New) API](/maps/architecture/improved-cost-effective-local-discovery)
- [See your data in real time with Data-driven styling](/maps/architecture/dds-real-time)

## Contributors

Google maintains this article. The following contributors originally wrote it.

Principal author: [Thomas Anglaret](https://www.linkedin.com/in/thomas-anglaret/) | Solutions Engineer

---

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2026-09-17 UTC.
